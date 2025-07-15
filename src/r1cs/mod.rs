//! This module defines R1CS related types
use crate::{
  Blind, Commitment, CommitmentKey, VerifierKey,
  digest::SimpleDigestible,
  errors::SpartanError,
  traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptReprTrait},
};
use core::cmp::max;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod sparse;
pub(crate) use sparse::SparseMatrix;

/// A type that holds the shape of the R1CS matrices
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSShape<E: Engine> {
  pub(crate) num_cons: usize,
  // variables are in three vectors: shared, precommitted, and rest
  pub(crate) num_shared: usize,       // shared variables
  pub(crate) num_precommitted: usize, // precommitted variables
  pub(crate) num_rest: usize,         // rest of the variables
  pub(crate) num_io: usize,           // input/output
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

/// Round `n` up to the next multiple of width.
/// (If `n` is already a multiple and higher than zero, it is returned unchanged.)
#[inline]
pub fn pad_to_width(width: usize, n: usize) -> usize {
  // width == 1024 == 1 << 10, so the mask is width-1 == 0b111_1111_1111 (10 bits set).
  n.saturating_add(width - 1) & !(width - 1)
}

impl<E: Engine> R1CSShape<E> {
  /// Create an object of type `R1CSShape` from the explicitly specified R1CS matrices
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    width: usize,
    num_cons: usize,
    num_shared: usize,
    num_precommitted: usize,
    num_rest: usize,
    num_io: usize,
    A: SparseMatrix<E::Scalar>,
    B: SparseMatrix<E::Scalar>,
    C: SparseMatrix<E::Scalar>,
  ) -> Result<R1CSShape<E>, SpartanError> {
    let is_valid = |num_rows: usize,
                    num_cols: usize,
                    M: &SparseMatrix<E::Scalar>|
     -> Result<Vec<()>, SpartanError> {
      M.iter()
        .map(|(row, col, _val)| {
          if row >= num_rows || col >= num_cols {
            Err(SpartanError::InvalidIndex)
          } else {
            Ok(())
          }
        })
        .collect::<Result<Vec<()>, SpartanError>>()
    };

    let num_rows = num_cons;
    let num_cols = num_shared + num_precommitted + num_rest + 1 + num_io; // +1 for the constant term

    is_valid(num_rows, num_cols, &A)?;
    is_valid(num_rows, num_cols, &B)?;
    is_valid(num_rows, num_cols, &C)?;

    // We need to pad num_shared, num_precommitted, and num_rest. We need each of them to be a multiple of num_cols.
    let num_shared_padded = pad_to_width(width, num_shared);
    let num_precommitted_padded = pad_to_width(width, num_precommitted);
    let mut num_rest_padded = pad_to_width(width, num_rest);

    // We need to make sure num_vars_padded >= num_io + 1 (for the constant term).
    let num_vars_padded = num_shared_padded + num_precommitted_padded + num_rest_padded;
    if num_vars_padded < num_io + 1 {
      // If not, we need to pad the rest to make it at least num_io + 1.
      num_rest_padded =
        max(num_io + 1, num_vars_padded) - (num_shared_padded + num_precommitted_padded);
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
          // public IO variables
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

    Ok(R1CSShape {
      num_cons: num_cons_padded,
      num_shared: num_shared_padded,
      num_precommitted: num_precommitted_padded,
      num_rest: num_rest_padded,
      num_io,
      A: A_padded,
      B: B_padded,
      C: C_padded,
      digest: OnceCell::new(),
    })
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
    let num_vars = self.num_shared + self.num_precommitted + self.num_rest;
    E::PCS::setup(b"ck", num_vars)
  }

  pub fn multiply_vec(
    &self,
    z: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    if z.len() != self.num_io + 1 + self.num_shared + self.num_precommitted + self.num_rest {
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
  pub fn new_unchecked(W: Vec<E::Scalar>, r_W: Blind<E>) -> Result<R1CSWitness<E>, SpartanError> {
    Ok(Self { W, r_W })
  }
}

impl<E: Engine> R1CSInstance<E> {
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
