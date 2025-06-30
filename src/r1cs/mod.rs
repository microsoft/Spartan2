//! This module defines R1CS related types
use crate::{
  Blind, Commitment, CommitmentKey, PCS, VerifierKey,
  digest::SimpleDigestible,
  errors::SpartanError,
  traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptReprTrait},
};
use core::cmp::max;
use ff::{Field, PrimeField};
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod sparse;
pub(crate) use sparse::SparseMatrix;

/// A type that holds the shape of the R1CS matrices
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSShape<E: Engine> {
  pub(crate) num_cons: usize,
  pub(crate) num_vars: usize,
  pub(crate) num_io: usize,
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
    let is_valid = |num_cons: usize,
                    num_vars: usize,
                    num_io: usize,
                    M: &SparseMatrix<E::Scalar>|
     -> Result<Vec<()>, SpartanError> {
      M.iter()
        .map(|(row, col, _val)| {
          if row >= num_cons || col > num_io + num_vars {
            Err(SpartanError::InvalidIndex)
          } else {
            Ok(())
          }
        })
        .collect::<Result<Vec<()>, SpartanError>>()
    };

    is_valid(num_cons, num_vars, num_io, &A)?;
    is_valid(num_cons, num_vars, num_io, &B)?;
    is_valid(num_cons, num_vars, num_io, &C)?;

    let vars_valid = num_vars.next_power_of_two() == num_vars;
    let io_lt_vars = num_io < num_vars;

    let num_cons_padded = num_cons.next_power_of_two();

    if vars_valid && io_lt_vars {
      Ok(R1CSShape {
        num_cons: num_cons_padded,
        num_vars,
        num_io,
        A,
        B,
        C,
        digest: OnceCell::new(),
      })
    } else {
      let n = max(num_vars, num_io).next_power_of_two();

      // otherwise, we need to pad the number of variables and renumber variable accesses
      let num_vars_padded = n;

      let apply_pad = |mut M: SparseMatrix<E::Scalar>| -> SparseMatrix<E::Scalar> {
        M.indices.par_iter_mut().for_each(|c| {
          if *c >= num_vars {
            *c += num_vars_padded - num_vars
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

      Ok(R1CSShape {
        num_cons: num_cons_padded,
        num_vars: num_vars_padded,
        num_io,
        A: A_padded,
        B: B_padded,
        C: C_padded,
        digest: OnceCell::new(),
      })
    }
  }

  /// Generates public parameters for a Rank-1 Constraint System (R1CS).
  ///
  /// This function takes into consideration the shape of the R1CS matrices and a hint function
  /// for the number of generators. It returns a `CommitmentKey`.
  ///
  /// # Arguments
  ///
  /// * `S`: The shape of the R1CS matrices.
  /// * `ck_floor`: A function that provides a floor for the number of generators. A good function
  ///   to provide is the ck_floor field defined in the trait `R1CSSNARK`.
  ///
  pub fn commitment_key(&self) -> (CommitmentKey<E>, VerifierKey<E>) {
    let num_cons = self.num_cons;
    let num_vars = self.num_vars;
    E::PCS::setup(b"ck", max(num_cons, num_vars))
  }

  pub fn multiply_vec(
    &self,
    z: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    if z.len() != self.num_io + self.num_vars + 1 {
      return Err(SpartanError::InvalidWitnessLength);
    }

    let (Az, (Bz, Cz)) = rayon::join(
      || self.A.multiply_vec(z),
      || rayon::join(|| self.B.multiply_vec(z), || self.C.multiply_vec(z)),
    );

    Ok((Az?, Bz?, Cz?))
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
    let res_comm = U.comm_W == PCS::<E>::commit(ck, &W.W, &W.r_W)?;

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
    if W.len() < S.num_vars {
      W.resize(S.num_vars, E::Scalar::ZERO);
    }

    let comm_W = if is_small {
      // extract small values from the witness
      let W_small = W
        .par_iter()
        .map(|e| {
          // map field element to u64
          e.to_repr().as_ref()[0] as u64
        })
        .collect::<Vec<_>>();
      PCS::<E>::commit_small(ck, &W_small, &r_W)?
    } else {
      PCS::<E>::commit(ck, W, &r_W)?
    };

    let W = R1CSWitness { W: W.to_vec(), r_W };

    Ok((W, comm_W))
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
