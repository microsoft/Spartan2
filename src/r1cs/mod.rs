//! This module defines R1CS related types
use crate::{
  Blind, Commitment, CommitmentKey, PCS, PartialCommitment, VerifierKey,
  digest::SimpleDigestible,
  errors::SpartanError,
  start_span,
  traits::{
    Engine,
    pcs::PCSEngineTrait,
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
pub struct R1CSWitness<E: Engine> {
  is_small: bool, // whether the witness elements fit in machine words
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
}

impl<E: Engine> R1CSInstance<E> {
  /// A method to create an instance object using constituent elements
  pub fn new(
    S: &R1CSShape<E>,
    comm_W: &Commitment<E>,
    X: &[E::Scalar],
  ) -> Result<R1CSInstance<E>, SpartanError> {
    if S.num_io != X.len() {
      Err(SpartanError::InvalidInputLength)
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
  pub fn commitment_key(&self) -> (CommitmentKey<E>, VerifierKey<E>) {
    let num_vars = self.num_shared + self.num_precommitted + self.num_rest;
    E::PCS::setup(b"ck", num_vars)
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
      return Err(SpartanError::InvalidInputLength);
    }
    if challenges.len() != S.num_challenges {
      return Err(SpartanError::InvalidInputLength);
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
