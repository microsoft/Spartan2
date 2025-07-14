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

  /// Holds the type of the partial commitment
  type PartialCommitment: CommitmentTrait<E>;

  /// Holds the type of the blind
  type Blind: Clone + Debug + Send + Sync + PartialEq + Eq + Serialize + for<'de> Deserialize<'de>;

  /// A type that holds the evaluation argument
  type EvaluationArgument: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Samples a new commitment key of a specified size and a verifier key
  fn setup(label: &'static [u8], n: usize) -> (Self::CommitmentKey, Self::VerifierKey);

  /// Returns a blind to be used for commitment
  fn blind(ck: &Self::CommitmentKey) -> Self::Blind;

  /// Commits to the provided vector using the provided ck and returns the commitment
  fn commit(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    r: &Self::Blind,
    is_small: bool,
  ) -> Result<Self::Commitment, SpartanError>;

  /// Commits to v using the provided ck and returns a partial commitment
  /// Also, returns the unused blind if the commitment is partial
  fn commit_partial(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    r: &Self::Blind,
    is_small: bool,
  ) -> Result<(Self::PartialCommitment, Self::Blind), SpartanError>;

  /// Combines the provided partial commitments into a single commitment
  /// This method is used to finalize the commitment after multiple partial commitments have been made
  fn combine_partial(
    partial_comms: &[Self::PartialCommitment],
  ) -> Result<Self::Commitment, SpartanError>;

  /// A method to prove the evaluation of a multilinear polynomial
  fn prove(
    ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[E::Scalar],
    blind: &Self::Blind,
    point: &[E::Scalar],
  ) -> Result<(E::Scalar, Self::EvaluationArgument), SpartanError>;

  /// A method to verify the purported evaluation of a multilinear polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError>;
}
