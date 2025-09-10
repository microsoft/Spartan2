//! This module defines a collection of traits that define the behavior of a commitment engine
//! We require the commitment engine to provide a commitment to vectors with a single group element
use crate::{
  errors::SpartanError,
  traits::{Engine, TranscriptReprTrait},
};
use core::fmt::Debug;
use serde::{Deserialize, Serialize};

/// This trait defines the behavior of the commitment
pub trait CommitmentTrait<E: Engine>:
  Clone
  + Debug
  + PartialEq
  + Eq
  + Send
  + Sync
  + TranscriptReprTrait<E::GE>
  + Serialize
  + for<'de> Deserialize<'de>
{
}

/// A trait that ties different pieces of the commitment generation together
pub trait PCSEngineTrait<E: Engine>: Clone + Send + Sync {
  /// Holds the type of the commitment key
  /// The key should quantify its length in terms of group generators.
  type CommitmentKey: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that holds the verifier key
  type VerifierKey: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Holds the type of the commitment
  type Commitment: CommitmentTrait<E>;

  /// Holds the type of the blind
  type Blind: Clone + Debug + Send + Sync + PartialEq + Eq + Serialize + for<'de> Deserialize<'de>;

  /// A type that holds the evaluation argument
  type EvaluationArgument: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Samples a new commitment key of a specified size and a verifier key
  fn setup(label: &'static [u8], n: usize) -> (Self::CommitmentKey, Self::VerifierKey) {
    Self::setup_with_width(label, n, Self::width())
  }

  /// Samples a new commitment key and verifier key using an explicit column width (`width`).
  fn setup_with_width(
    label: &'static [u8],
    n: usize,
    width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey);

  /// Size of the polynomial committed with one unit
  fn width() -> usize;

  /// Returns a blind to be used for commitment to a polynomial of size `n`
  fn blind(ck: &Self::CommitmentKey, n: usize) -> Self::Blind;

  /// Commits to the provided vector using the provided ck and returns the commitment
  fn commit(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    r: &Self::Blind,
    is_small: bool,
  ) -> Result<Self::Commitment, SpartanError>;

  /// Checks if the provided commitment commits to a vector of the specified length
  fn check_commitment(comm: &Self::Commitment, n: usize, width: usize) -> Result<(), SpartanError>;

  /// Rerandomizes the provided commitment using the provided blind
  fn rerandomize_commitment(
    ck: &Self::CommitmentKey,
    comm: &Self::Commitment,
    r_old: &Self::Blind,
    r_new: &Self::Blind,
  ) -> Result<Self::Commitment, SpartanError>;

  /// Combines the provided commitments (each committing to a multilinear polynomial) into a single commitment.
  ///
  /// # Parameters
  /// - `comms`: A slice of commitments to be combined. The order of the commitments in the slice must match the order in which they were generated using `commit`.
  ///
  /// # Constraints
  /// - All commitments in the slice must be valid and correspond to the same commitment key.
  /// - The number of commitments must match the expected number for the final commitment.
  ///
  /// # Returns
  /// - A single combined commitment if the operation is successful.
  /// - An error of type `SpartanError` if the combination fails due to invalid inputs or mismatched constraints.
  ///
  /// # Usage
  /// This method is used to finalize the commitment after multiple commitments have been made.
  /// Ensure that the commitments are provided in the correct order and meet all constraints.
  fn combine_commitments(comms: &[Self::Commitment]) -> Result<Self::Commitment, SpartanError>;

  /// Combines the provided blinds into a single blind.
  /// The order of the blinds must match the order of the commitments used to generate them
  /// Returns an error if the combination fails
  fn combine_blinds(blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError>;

  /// A method to prove the evaluation of a multilinear polynomial
  #[allow(clippy::too_many_arguments)]
  fn prove(
    ck: &Self::CommitmentKey,
    ck_eval: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[E::Scalar],
    blind: &Self::Blind,
    point: &[E::Scalar],
    comm_eval: &Self::Commitment,
    blind_eval: &Self::Blind,
  ) -> Result<Self::EvaluationArgument, SpartanError>;

  /// A method to verify the purported evaluation of a multilinear polynomials
  fn verify(
    vk: &Self::VerifierKey,
    ck_eval: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    comm_eval: &Self::Commitment,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError>;
}

/// A trait that extends the PCSEngineTrait to include folding capabilities
/// This trait allows for folding multiple commitments (and blinds) into a single commitment (a single blind) using specified weights
pub trait FoldingEngineTrait<E: Engine>: PCSEngineTrait<E> {
  /// A method to fold the provided commitments into a single commitment using the provided weights
  /// The weights should be the same length as the number of commitments
  fn fold_commitments(
    comms: &[Self::Commitment],
    weights: &[E::Scalar],
  ) -> Result<Self::Commitment, SpartanError>;

  /// A method to fold the provided blinds into a single blind using the provided weights
  /// The weights should be the same length as the number of blinds
  fn fold_blinds(
    blinds: &[Self::Blind],
    weights: &[E::Scalar],
  ) -> Result<Self::Blind, SpartanError>;
}
