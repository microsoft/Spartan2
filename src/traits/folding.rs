//! Folding utilities for linear combination of commitments/blinds.
//!
//! This is an extension trait over `PCSEngineTrait` used by the Hyrax PCS to
//! support the minimal folding operations needed by Spartan2's lightweight
//! NIFS tests.

use crate::{
  errors::SpartanError,
  traits::{Engine, pcs::PCSEngineTrait},
};

/// Extends a PCS engine with the ability to linearly combine ("fold")
/// commitments and blinds using scalar weights.
///
/// The semantics are identical for commitments and blinds: given vectors
/// `c_i`, weights `w_i`, it returns the weighted sum `\sum_i w_i * c_i`.
pub trait FoldingEngineTrait<E: Engine>: PCSEngineTrait<E> {
  /// Fold a slice of commitments with the provided weights.
  fn fold_commitments(
    comms: &[Self::Commitment],
    weights: &[E::Scalar],
  ) -> Result<Self::Commitment, SpartanError>;

  /// Fold a slice of blinds with the provided weights.
  fn fold_blinds(
    blinds: &[Self::Blind],
    weights: &[E::Scalar],
  ) -> Result<Self::Blind, SpartanError>;
}
