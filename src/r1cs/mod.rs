//! This module defines R1CS related types
use crate::{
  Blind, Commitment, CommitmentKey, PCS, PartialCommitment, VerifierKey,
  digest::SimpleDigestible,
  errors::SpartanError,
  start_span,
  traits::{
    Engine,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::cmp::max;
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{info, info_span};

mod sparse;
pub(crate) use sparse::SparseMatrix;

fn eq01<F: Field>(bit: u8, r: &F) -> F {
  if bit == 0 { F::ONE - *r } else { *r }
}

#[inline]
fn weights_from_r<F: Field>(r_bs: &[F], n: usize) -> Vec<F> {
  let ell = r_bs.len();
  (0..n)
    .map(|i| {
      let mut wi = F::ONE;
      let mut k = i;
      for r_bs_t in r_bs.iter().take(ell) {
        wi *= eq01((k & 1) as u8, r_bs_t);
        k >>= 1;
      }
      wi
    })
    .collect()
}

/// A type that holds the shape of the R1CS matrices
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSShape<E: Engine> {
  pub(crate) num_cons: usize,
  pub(crate) num_vars: usize,
  pub(crate) num_io: usize, // input/output
  pub(crate) A: SparseMatrix<E::Scalar>,
  pub(crate) B: SparseMatrix<E::Scalar>,
  pub(crate) C: SparseMatrix<E::Scalar>,
  #[serde(skip, default = "OnceCell::new")]
  pub(crate) digest: OnceCell<E::Scalar>,
}

impl<E: Engine> SimpleDigestible for R1CSShape<E> {}

/// A type that holds a witness for a given R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSWitness<E: Engine> {
  pub(crate) is_small: bool, // whether the witness elements fit in machine words
  pub(crate) W: Vec<E::Scalar>,
  pub(crate) r_W: Blind<E>,
}

/// A type that holds an R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSInstance<E: Engine> {
  pub(crate) comm_W: Commitment<E>,
  pub(crate) X: Vec<E::Scalar>,
}

/// Round `n` up to the next multiple of width.
/// (If `n` is already a multiple and higher than zero, it is returned unchanged.)
#[inline]
pub fn pad_to_width(width: usize, n: usize) -> usize {
  if n == 0 {
    return 0;
  }

  // width == 1024 == 1 << 10, so the mask is width-1 == 0b111_1111_1111 (10 bits set).
  n.saturating_add(width - 1) & !(width - 1)
}

fn is_sparse_matrix_valid<E: Engine>(
  num_rows: usize,
  num_cols: usize,
  M: &SparseMatrix<E::Scalar>,
) -> Result<(), SpartanError> {
  // Check if the indices and indptr are valid for the given number of rows and columns
  M.iter().try_for_each(|(row, col, _val)| {
    if row >= num_rows || col >= num_cols {
      Err(SpartanError::InvalidIndex)
    } else {
      Ok(())
    }
  })
}

impl<E: Engine> R1CSShape<E> {
  /// Create an object of type `R1CSShape` from the explicitly specified R1CS matrices
  pub fn new(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    A: SparseMatrix<E::Scalar>,
    B: SparseMatrix<E::Scalar>,
    C: SparseMatrix<E::Scalar>,
  ) -> Result<R1CSShape<E>, SpartanError> {
    let num_rows = num_cons;
    let num_cols = num_vars + 1 + num_io; // +1 for the constant term

    is_sparse_matrix_valid::<E>(num_rows, num_cols, &A)?;
    is_sparse_matrix_valid::<E>(num_rows, num_cols, &B)?;
    is_sparse_matrix_valid::<E>(num_rows, num_cols, &C)?;

    Ok(R1CSShape {
      num_cons,
      num_vars,
      num_io,
      A,
      B,
      C,
      digest: OnceCell::new(),
    })
  }

  /// Pads the `R1CSShape` so that the shape passes `is_regular_shape`
  /// Renumbers variables to accommodate padded variables
  pub fn pad(&self) -> Self {
    // check if the provided R1CSShape is already as required
    if self.is_regular_shape() {
      return self.clone();
    }

    // equalize the number of variables and public IO
    let m = self.num_vars.max(self.num_io).next_power_of_two();

    // check if the number of variables are as expected, then
    // we simply set the number of constraints to the next power of two
    if self.num_vars == m {
      return R1CSShape {
        num_cons: self.num_cons.next_power_of_two(),
        num_vars: m,
        num_io: self.num_io,
        A: self.A.clone(),
        B: self.B.clone(),
        C: self.C.clone(),
        digest: OnceCell::new(),
      };
    }

    // otherwise, we need to pad the number of variables and renumber variable accesses
    let num_vars_padded = m;
    let num_cons_padded = self.num_cons.next_power_of_two();

    let apply_pad = |mut M: SparseMatrix<E::Scalar>| -> SparseMatrix<E::Scalar> {
      M.indices.par_iter_mut().for_each(|c| {
        if *c >= self.num_vars {
          *c += num_vars_padded - self.num_vars
        }
      });

      M.cols += num_vars_padded - self.num_vars;

      let ex = {
        let nnz = M.indptr.last().unwrap();
        vec![*nnz; num_cons_padded - self.num_cons]
      };
      M.indptr.extend(ex);
      M
    };

    let A_padded = apply_pad(self.A.clone());
    let B_padded = apply_pad(self.B.clone());
    let C_padded = apply_pad(self.C.clone());

    R1CSShape {
      num_cons: num_cons_padded,
      num_vars: num_vars_padded,
      num_io: self.num_io,
      A: A_padded,
      B: B_padded,
      C: C_padded,
      digest: OnceCell::new(),
    }
  }

  // Checks regularity conditions on the R1CSShape, required in Spartan-class SNARKs
  // Returns false if num_cons or num_vars are not powers of two, or if num_io > num_vars
  #[inline]
  pub(crate) fn is_regular_shape(&self) -> bool {
    let cons_valid = self.num_cons.next_power_of_two() == self.num_cons;
    let vars_valid = self.num_vars.next_power_of_two() == self.num_vars;
    let io_lt_vars = self.num_io < self.num_vars;
    cons_valid && vars_valid && io_lt_vars
  }

  /// Checks if the R1CS instance is satisfiable given a witness and its shape
  #[allow(dead_code)]
  pub fn is_sat(
    &self,
    ck: &CommitmentKey<E>,
    U: &R1CSInstance<E>,
    W: &R1CSWitness<E>,
  ) -> Result<(), SpartanError> {
    assert_eq!(W.W.len(), self.num_vars);
    assert_eq!(U.X.len(), self.num_io);

    // verify if Az * Bz = u*Cz
    let res_eq = {
      let z = [W.W.clone(), vec![E::Scalar::ONE], U.X.clone()].concat();
      let (Az, Bz, Cz) = self.multiply_vec(&z)?;
      assert_eq!(Az.len(), self.num_cons);
      assert_eq!(Bz.len(), self.num_cons);
      assert_eq!(Cz.len(), self.num_cons);

      (0..self.num_cons).all(|i| Az[i] * Bz[i] == Cz[i])
    };

    // verify if comm_W is a commitment to W
    let res_comm = U.comm_W == PCS::<E>::commit(ck, &W.W, &W.r_W, W.is_small)?;

    if !res_eq {
      return Err(SpartanError::UnSat {
        reason: "R1CS is unsatisfiable".to_string(),
      });
    }

    if !res_comm {
      return Err(SpartanError::UnSat {
        reason: "Invalid commitment".to_string(),
      });
    }

    Ok(())
  }

  /// Generates public parameters for a Rank-1 Constraint System (R1CS).
  ///
  /// This function takes into consideration the shape of the R1CS matrices
  ///
  /// # Arguments
  ///
  /// * `S`: The shape of the R1CS matrices.
  ///
  pub fn commitment_key(&self) -> (CommitmentKey<E>, VerifierKey<E>) {
    E::PCS::setup(b"ck", self.num_vars)
  }

  pub fn multiply_vec(
    &self,
    z: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    if z.len() != self.num_io + 1 + self.num_vars {
      return Err(SpartanError::InvalidWitnessLength);
    }

    let (Az, (Bz, Cz)) = rayon::join(
      || self.A.multiply_vec(z),
      || rayon::join(|| self.B.multiply_vec(z), || self.C.multiply_vec(z)),
    );

    Ok((Az?, Bz?, Cz?))
  }
}

impl<E: Engine> R1CSWitness<E> {
  /// A method to create a witness object using a vector of scalars
  pub fn new(
    ck: &CommitmentKey<E>,
    S: &R1CSShape<E>,
    W: &mut Vec<E::Scalar>,
    is_small: bool,
  ) -> Result<(R1CSWitness<E>, Commitment<E>), SpartanError> {
    let r_W = PCS::<E>::blind(ck);

    // pad with zeros
    let (_pad_span, pad_t) = start_span!("pad_witness");
    if W.len() < S.num_vars {
      W.resize(S.num_vars, E::Scalar::ZERO);
    }
    info!(elapsed_ms = %pad_t.elapsed().as_millis(), "pad_witness");

    let (_commit_span, commit_t) = start_span!("commit_witness");
    let comm_W = PCS::<E>::commit(ck, W, &r_W, is_small)?;
    info!(elapsed_ms = %commit_t.elapsed().as_millis(), "commit_witness");

    let W = R1CSWitness {
      W: W.to_vec(),
      r_W,
      is_small,
    };

    Ok((W, comm_W))
  }

  /// A method to create a witness object using a vector of scalars
  pub fn new_unchecked(
    W: Vec<E::Scalar>,
    r_W: Blind<E>,
    is_small: bool,
  ) -> Result<R1CSWitness<E>, SpartanError> {
    Ok(Self { W, r_W, is_small })
  }

  /// Fold multiple witnesses with a sequence of r_b values
  pub fn fold_multiple(
    r_bs: &[E::Scalar],
    Ws: &[R1CSWitness<E>],
  ) -> Result<R1CSWitness<E>, SpartanError>
  where
    E::PCS: FoldingEngineTrait<E>,
  {
    let n = Ws.len();
    let w = weights_from_r::<E::Scalar>(r_bs, n);
    let dim = Ws[0].W.len();

    // Parallelize across witness vectors (not across individual coordinates).
    // Uses rayon's fold/reduce so only one accumulator per worker thread is allocated.
    use rayon::prelude::*;

    let acc_W = (0..n)
      .into_par_iter()
      .fold(
        || vec![E::Scalar::ZERO; dim],
        |mut acc, i| {
          let wi = w[i];
          let Wi = &Ws[i].W;
          for k in 0..dim {
            acc[k] += wi * Wi[k];
          }
          acc
        },
      )
      .reduce(
        || vec![E::Scalar::ZERO; dim],
        |mut a, b| {
          for (ai, bi) in a.iter_mut().zip(b.iter()) {
            *ai += *bi;
          }
          a
        },
      );

    let acc_r = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
      &Ws.iter().map(|wz| wz.r_W.clone()).collect::<Vec<_>>(),
      &w,
    )?;

    Ok(R1CSWitness::<E> {
      W: acc_W,
      r_W: acc_r,
      is_small: false,
    })
  }
}

impl<E: Engine> R1CSInstance<E> {
  /// A method to create an instance object using constituent elements
  pub fn new(
    S: &R1CSShape<E>,
    comm_W: &Commitment<E>,
    X: &[E::Scalar],
  ) -> Result<R1CSInstance<E>, SpartanError> {
    if S.num_io != X.len() {
      Err(SpartanError::InvalidInputLength {
        reason: format!(
          "R1CS instance: Expected {} elements in X, got {}",
          S.num_io,
          X.len()
        ),
      })
    } else {
      Ok(R1CSInstance {
        comm_W: comm_W.clone(),
        X: X.to_owned(),
      })
    }
  }

  /// A method to create an instance object using constituent elements
  pub fn new_unchecked(
    comm_W: Commitment<E>,
    X: Vec<E::Scalar>,
  ) -> Result<R1CSInstance<E>, SpartanError> {
    Ok(R1CSInstance { comm_W, X })
  }

  /// Fold multiple instances with a sequence of r_b values
  pub fn fold_multiple(r_bs: &[E::Scalar], Us: &[R1CSInstance<E>]) -> R1CSInstance<E>
  where
    E::PCS: FoldingEngineTrait<E>,
  {
    let n = Us.len();
    let w = weights_from_r::<E::Scalar>(r_bs, n);
    let d = Us[0].X.len();

    // X
    let mut X_acc = vec![E::Scalar::ZERO; d];
    for (i, Ui) in Us.iter().enumerate() {
      let wi = w[i];
      for (j, Uij) in Ui.X.iter().enumerate() {
        X_acc[j] += wi * Uij;
      }
    }

    // commitment (group lin. comb)
    let comm_acc = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &Us.iter().map(|U| U.comm_W.clone()).collect::<Vec<_>>(),
      &w,
    )
    .expect("fold_commitments");

    R1CSInstance::<E> {
      X: X_acc,
      comm_W: comm_acc,
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for R1CSInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_W.to_transcript_bytes(),
      self.X.as_slice().to_transcript_bytes(),
    ]
    .concat()
  }
}

///
////////////////// Split R1CS Types //////////////////
///
/// A type that holds a split R1CS shape
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitR1CSShape<E: Engine> {
  pub(crate) num_cons: usize,

  pub(crate) num_cons_unpadded: usize, // number of constraints before padding
  pub(crate) num_shared_unpadded: usize, // shared variables before padding
  pub(crate) num_precommitted_unpadded: usize, // precommitted variables before padding
  pub(crate) num_rest_unpadded: usize, // rest of the variables before padding

  pub(crate) num_shared: usize,       // shared variables
  pub(crate) num_precommitted: usize, // precommitted variables
  pub(crate) num_rest: usize,         // rest of the variables
  pub(crate) num_public: usize,       // number of public variables
  pub(crate) num_challenges: usize,   // number of public challenges
  pub(crate) A: SparseMatrix<E::Scalar>,
  pub(crate) B: SparseMatrix<E::Scalar>,
  pub(crate) C: SparseMatrix<E::Scalar>,
  #[serde(skip, default = "OnceCell::new")]
  pub(crate) digest: OnceCell<E::Scalar>,
}

impl<E: Engine> SimpleDigestible for SplitR1CSShape<E> {}

/// A type that holds a split R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SplitR1CSInstance<E: Engine> {
  pub(crate) comm_W_shared: Option<PartialCommitment<E>>,
  pub(crate) comm_W_precommitted: Option<PartialCommitment<E>>,
  pub(crate) comm_W_rest: PartialCommitment<E>,

  pub(crate) public_values: Vec<E::Scalar>,
  pub(crate) challenges: Vec<E::Scalar>,
}

impl<E: Engine> SplitR1CSShape<E> {
  /// Create an object of type `R1CSShape` from the explicitly specified R1CS matrices
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    width: usize,
    num_cons: usize,
    num_shared: usize,
    num_precommitted: usize,
    num_rest: usize,
    num_public: usize,
    num_challenges: usize,
    A: SparseMatrix<E::Scalar>,
    B: SparseMatrix<E::Scalar>,
    C: SparseMatrix<E::Scalar>,
  ) -> Result<SplitR1CSShape<E>, SpartanError> {
    let num_rows = num_cons;
    let num_cols = num_shared + num_precommitted + num_rest + 1 + num_public + num_challenges; // +1 for the constant term

    is_sparse_matrix_valid::<E>(num_rows, num_cols, &A)?;
    is_sparse_matrix_valid::<E>(num_rows, num_cols, &B)?;
    is_sparse_matrix_valid::<E>(num_rows, num_cols, &C)?;

    // We need to pad num_shared, num_precommitted, and num_rest. We need each of them to be a multiple of num_cols.
    let num_shared_padded = pad_to_width(width, num_shared);
    let num_precommitted_padded = pad_to_width(width, num_precommitted);
    let mut num_rest_padded = pad_to_width(width, num_rest);

    // We need to make sure num_vars_padded >= num_public + num_challenges + 1 (for the constant term).
    let num_vars_padded = num_shared_padded + num_precommitted_padded + num_rest_padded;
    if num_vars_padded < num_public + num_challenges + 1 {
      // If not, we need to pad the rest to make it at least num_public + num_challenges + 1.
      num_rest_padded = max(num_public + num_challenges + 1, num_vars_padded)
        - (num_shared_padded + num_precommitted_padded);
    }

    // We need to make sure num_shared_padded + num_precommitted_padded + num_rest_padded is a power of two.
    let num_vars_padded = num_shared_padded + num_precommitted_padded + num_rest_padded;
    if num_vars_padded.next_power_of_two() != num_vars_padded {
      // If not, we need to pad the rest to the next power of two.
      num_rest_padded =
        num_vars_padded.next_power_of_two() - (num_shared_padded + num_precommitted_padded);
    }

    let num_vars = num_shared + num_precommitted + num_rest;
    let num_vars_padded = num_shared_padded + num_precommitted_padded + num_rest_padded;
    let num_cons_padded = num_cons.next_power_of_two();

    let apply_pad = |mut M: SparseMatrix<E::Scalar>| -> SparseMatrix<E::Scalar> {
      M.indices.par_iter_mut().for_each(|c| {
        if *c >= num_shared && *c < num_shared + num_precommitted {
          // precommitted variables
          *c += num_shared_padded - num_shared;
        } else if *c >= num_shared + num_precommitted && *c < num_vars {
          // rest of the variables
          *c += num_shared_padded + num_precommitted_padded - num_shared - num_precommitted;
        } else if *c >= num_vars {
          // public and challenge variables
          *c += num_vars_padded - num_vars;
        }
      });

      M.cols += num_vars_padded - num_vars;

      let ex = {
        let nnz = if M.indptr.is_empty() {
          0
        } else {
          M.indptr[M.indptr.len() - 1]
        };
        vec![nnz; num_cons_padded - num_cons]
      };
      M.indptr.extend(ex);
      M
    };

    let A_padded = apply_pad(A);
    let B_padded = apply_pad(B);
    let C_padded = apply_pad(C);

    Ok(Self {
      num_cons: num_cons_padded,
      num_shared: num_shared_padded,
      num_precommitted: num_precommitted_padded,
      num_rest: num_rest_padded,

      num_cons_unpadded: num_cons,
      num_shared_unpadded: num_shared,
      num_precommitted_unpadded: num_precommitted,
      num_rest_unpadded: num_rest,

      num_public,
      num_challenges,
      A: A_padded,
      B: B_padded,
      C: C_padded,
      digest: OnceCell::new(),
    })
  }

  pub fn equalize(S_A: &mut Self, S_B: &mut Self) {
    let num_cons_padded = max(S_A.num_cons, S_B.num_cons);
    let num_vars_padded = max(
      S_A.num_shared + S_A.num_precommitted + S_A.num_rest,
      S_B.num_shared + S_B.num_precommitted + S_B.num_rest,
    );

    S_A.num_cons = num_cons_padded;
    S_B.num_cons = num_cons_padded;

    let move_public_vars = |M: &mut SparseMatrix<E::Scalar>, num_cons: usize, num_vars: usize| {
      M.indices.par_iter_mut().for_each(|c| {
        if *c >= num_vars {
          // public and challenge variables
          *c += num_vars_padded - num_vars;
        }
      });

      M.cols += num_vars_padded - num_vars;

      let ex = {
        let nnz = if M.indptr.is_empty() {
          0
        } else {
          M.indptr[M.indptr.len() - 1]
        };
        vec![nnz; num_cons_padded - num_cons]
      };
      M.indptr.extend(ex);
    };

    // get the total number of variables to `num_vars_padded` by increasing rest variables
    if S_A.num_shared + S_A.num_precommitted + S_A.num_rest != num_vars_padded {
      let num_cons = S_A.num_cons;
      let num_vars = S_A.num_shared + S_A.num_precommitted + S_A.num_rest;
      S_A.num_rest = num_vars_padded - (S_A.num_shared + S_A.num_precommitted);
      move_public_vars(&mut S_A.A, num_cons, num_vars);
      move_public_vars(&mut S_A.B, num_cons, num_vars);
      move_public_vars(&mut S_A.C, num_cons, num_vars);
    }

    if S_B.num_shared + S_B.num_precommitted + S_B.num_rest != num_vars_padded {
      let num_cons = S_B.num_cons;
      let num_vars = S_B.num_shared + S_B.num_precommitted + S_B.num_rest;
      S_B.num_rest = num_vars_padded - (S_B.num_shared + S_B.num_precommitted);
      move_public_vars(&mut S_B.A, num_cons, num_vars);
      move_public_vars(&mut S_B.B, num_cons, num_vars);
      move_public_vars(&mut S_B.C, num_cons, num_vars);
    }
  }

  pub fn to_regular_shape(&self) -> R1CSShape<E> {
    R1CSShape {
      num_cons: self.num_cons,
      num_vars: self.num_shared + self.num_precommitted + self.num_rest,
      num_io: self.num_public + self.num_challenges,
      A: self.A.clone(),
      B: self.B.clone(),
      C: self.C.clone(),
      digest: OnceCell::new(),
    }
  }

  /// Returns statistics about the shape of the R1CS matrices.
  ///
  /// This function returns an array of 10 elements, where each element represents a specific
  /// statistic about the R1CS matrices. The elements are as follows:
  /// - `num_cons_unpadded`: The number of constraints in the unpadded R1CS matrix.
  /// - `num_shared_unpadded`: The number of shared variables in the unpadded R1CS matrix.
  /// - `num_precommitted_unpadded`: The number of precommitted variables in the unpadded R1CS matrix.
  /// - `num_rest_unpadded`: The number of remaining variables in the unpadded R1CS matrix.
  /// - `num_cons`: The number of constraints in the padded R1CS matrix.
  /// - `num_shared`: The number of shared variables in the padded R1CS matrix.
  /// - `num_precommitted`: The number of precommitted variables in the padded R1CS matrix.
  /// - `num_rest`: The number of remaining variables in the padded R1CS matrix.
  /// - `num_public`: The number of public inputs/outputs in the R1CS matrix.
  /// - `num_challenges`: The number of challenges in the R1CS matrix.
  ///
  /// The terms "unpadded" and "padded" refer to the state of the R1CS matrices:
  /// - "Unpadded" values represent the original dimensions of the matrices before any padding
  ///   is applied to meet alignment or size requirements.
  /// - "Padded" values represent the dimensions of the matrices after padding has been applied.
  ///
  pub fn sizes(&self) -> [usize; 10] {
    [
      self.num_cons_unpadded,
      self.num_shared_unpadded,
      self.num_precommitted_unpadded,
      self.num_rest_unpadded,
      self.num_cons,
      self.num_shared,
      self.num_precommitted,
      self.num_rest,
      self.num_public,
      self.num_challenges,
    ]
  }

  /// Generates public parameters for a Rank-1 Constraint System (R1CS).
  ///
  /// This function takes into consideration the shape of the R1CS matrices
  ///
  /// # Arguments
  ///
  /// * `S`: The shape of the R1CS matrices.
  ///
  pub fn commitment_key(
    shapes: &[&SplitR1CSShape<E>],
  ) -> Result<(CommitmentKey<E>, VerifierKey<E>), SpartanError> {
    let max = shapes
      .iter()
      .map(|s| s.num_shared + s.num_precommitted + s.num_rest)
      .max()
      .ok_or(SpartanError::InvalidInputLength {
        reason: "commitment_key: unable to find max number of variables".to_string(),
      })?;

    Ok(E::PCS::setup(b"ck", max))
  }

  pub fn multiply_vec(
    &self,
    z: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    if z.len()
      != self.num_public
        + self.num_challenges
        + 1
        + self.num_shared
        + self.num_precommitted
        + self.num_rest
    {
      return Err(SpartanError::InvalidWitnessLength);
    }

    let (Az, (Bz, Cz)) = rayon::join(
      || self.A.multiply_vec(z),
      || rayon::join(|| self.B.multiply_vec(z), || self.C.multiply_vec(z)),
    );

    Ok((Az?, Bz?, Cz?))
  }
}

impl<E: Engine> SplitR1CSInstance<E> {
  /// A method to create a split R1CS instance object using constituent elements
  pub fn new(
    S: &SplitR1CSShape<E>,
    comm_W_shared: Option<PartialCommitment<E>>,
    comm_W_precommitted: Option<PartialCommitment<E>>,
    comm_W_rest: PartialCommitment<E>,
    public_values: Vec<E::Scalar>,
    challenges: Vec<E::Scalar>,
  ) -> Result<SplitR1CSInstance<E>, SpartanError> {
    if public_values.len() != S.num_public {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SplitR1CS instance: Expected {} public values, got {}",
          S.num_public,
          public_values.len()
        ),
      });
    }
    if challenges.len() != S.num_challenges {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SplitR1CS instance: Expected {} challenges, got {}",
          S.num_challenges,
          challenges.len()
        ),
      });
    }

    // check if the commitments commit to the right number of variables
    if S.num_shared > 0 && comm_W_shared.is_none() {
      return Err(SpartanError::InvalidCommitmentLength {
        reason: "comm_W_shared is missing".to_string(),
      });
    }
    if S.num_precommitted > 0 && comm_W_precommitted.is_none() {
      return Err(SpartanError::InvalidCommitmentLength {
        reason: "comm_W_precommitted is missing".to_string(),
      });
    }

    if let Some(ref comm) = comm_W_shared {
      E::PCS::check_partial(comm, S.num_shared)?;
    }
    if let Some(ref comm) = comm_W_precommitted {
      E::PCS::check_partial(comm, S.num_precommitted)?;
    }
    E::PCS::check_partial(&comm_W_rest, S.num_rest)?;

    Ok(SplitR1CSInstance {
      comm_W_shared,
      comm_W_precommitted,
      comm_W_rest,
      public_values,
      challenges,
    })
  }

  pub fn validate(
    &self,
    S: &SplitR1CSShape<E>,
    transcript: &mut E::TE,
  ) -> Result<(), SpartanError> {
    // absorb the public IO into the transcript
    transcript.absorb(b"public_values", &self.public_values.as_slice());

    if S.num_shared > 0 {
      if let Some(comm) = &self.comm_W_shared {
        E::PCS::check_partial(comm, S.num_shared)?;
        transcript.absorb(b"comm_W_shared", comm);
      } else {
        return Err(SpartanError::ProofVerifyError {
          reason: "comm_W_shared is missing".to_string(),
        });
      }
    }

    if S.num_precommitted > 0 {
      if let Some(comm) = &self.comm_W_precommitted {
        E::PCS::check_partial(comm, S.num_precommitted)?;
        transcript.absorb(b"comm_W_precommitted", comm);
      } else {
        return Err(SpartanError::ProofVerifyError {
          reason: "comm_W_precommitted is missing".to_string(),
        });
      }
    }

    // obtain challenges from the transcript
    let challenges = (0..S.num_challenges)
      .map(|_| transcript.squeeze(b"challenge"))
      .collect::<Result<Vec<E::Scalar>, SpartanError>>()?;

    // check that the challenges of the circuit matches the expected values
    if challenges != self.challenges {
      return Err(SpartanError::ProofVerifyError {
        reason: "Challenges do not match".to_string(),
      });
    }

    E::PCS::check_partial(&self.comm_W_rest, S.num_rest)?;
    transcript.absorb(b"comm_W_rest", &self.comm_W_rest);

    Ok(())
  }

  pub fn to_regular_instance(&self) -> Result<R1CSInstance<E>, SpartanError> {
    let partial_comms = [
      self.comm_W_shared.clone(),
      self.comm_W_precommitted.clone(),
      Some(self.comm_W_rest.clone()),
    ]
    .iter()
    .filter_map(|comm| comm.clone())
    .collect::<Vec<PartialCommitment<E>>>();
    let comm_W = PCS::<E>::combine_partial(&partial_comms)?;

    Ok(R1CSInstance {
      comm_W,
      X: [self.public_values.clone(), self.challenges.clone()].concat(),
    })
  }
}
