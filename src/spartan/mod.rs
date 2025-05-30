//! This module implements `R1CSSNARK` using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
//! We provide two implementations, one in snark.rs (which does not use any preprocessing)
//! and another in ppsnark.rs (which uses preprocessing to keep the verifier's state small if the PCS provides a succinct verifier)
//! We also provide direct.rs that allows proving a step circuit directly with either of the two SNARKs.
//!
//! In polynomial.rs we also provide foundational types and functions for manipulating multilinear polynomials.
pub mod snark;

#[macro_use]
mod macros;
pub(crate) mod math;
pub(crate) mod polys;
pub(crate) mod sumcheck;

use crate::{
  r1cs::{R1CSShape, SparseMatrix},
  traits::Engine,
};
use ff::Field;

/// Bounds "row" variables of (A, B, C) matrices viewed as 2d multilinear polynomials
fn compute_eval_table_sparse<E: Engine>(
  S: &R1CSShape<E>,
  rx: &[E::Scalar],
) -> (Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>) {
  assert_eq!(rx.len(), S.num_cons);

  let inner = |M: &SparseMatrix<E::Scalar>, M_evals: &mut Vec<E::Scalar>| {
    for (row_idx, ptrs) in M.indptr.windows(2).enumerate() {
      for (val, col_idx) in M.get_row_unchecked(ptrs.try_into().unwrap()) {
        M_evals[*col_idx] += rx[row_idx] * val;
      }
    }
  };

  let (A_evals, (B_evals, C_evals)) = rayon::join(
    || {
      let mut A_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars];
      inner(&S.A, &mut A_evals);
      A_evals
    },
    || {
      rayon::join(
        || {
          let mut B_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars];
          inner(&S.B, &mut B_evals);
          B_evals
        },
        || {
          let mut C_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars];
          inner(&S.C, &mut C_evals);
          C_evals
        },
      )
    },
  );

  (A_evals, B_evals, C_evals)
}
