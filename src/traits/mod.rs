//! This module defines various traits required by the users of the library to implement.
use crate::{errors::SpartanError, traits::evaluation::EvaluationEngineTrait};
use core::fmt::Debug;
use ff::{PrimeField, PrimeFieldBits};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

pub mod commitment;

use commitment::CommitmentEngineTrait;

/// Represents an element of a group
/// This is currently tailored for an elliptic curve group
pub trait Group: Clone + Copy + Debug + Send + Sync + Sized + Eq + PartialEq {
  /// A type representing an element of the base field of the group
  type Base: PrimeFieldBits + Serialize + for<'de> Deserialize<'de>;

  /// A type representing an element of the scalar field of the group
  type Scalar: PrimeFieldBits + PrimeFieldExt + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Returns A, B, the order of the group, the size of the base field as big integers
  fn group_params() -> (Self::Base, Self::Base, BigInt, BigInt);
}

/// A collection of engines that are required by the library
pub trait Engine: Clone + Copy + Debug + Send + Sync + Sized + Eq + PartialEq {
  /// A type representing an element of the base field of the group
  type Base: PrimeFieldBits + TranscriptReprTrait<Self::GE> + Serialize + for<'de> Deserialize<'de>;

  /// A type representing an element of the scalar field of the group
  type Scalar: PrimeFieldBits
    + PrimeFieldExt
    + Send
    + Sync
    + TranscriptReprTrait<Self::GE>
    + Serialize
    + for<'de> Deserialize<'de>;

  /// A type that represents an element of the group
  type GE: Group<Base = Self::Base, Scalar = Self::Scalar> + Serialize + for<'de> Deserialize<'de>;

  /// A type that provides a generic Fiat-Shamir transcript to be used when externalizing proofs
  type TE: TranscriptEngineTrait<Self>;

  /// A type that defines a commitment engine over scalars in the group
  type CE: CommitmentEngineTrait<Self>;

  /// A type that defines an evaluation engine over scalars in the group
  type EE: EvaluationEngineTrait<Self>;
}

/// This trait allows types to implement how they want to be added to `TranscriptEngine`
pub trait TranscriptReprTrait<G: Group>: Send + Sync {
  /// returns a byte representation of self to be added to the transcript
  fn to_transcript_bytes(&self) -> Vec<u8>;
}

/// This trait defines the behavior of a transcript engine compatible with Spartan
pub trait TranscriptEngineTrait<E: Engine>: Send + Sync {
  /// initializes the transcript
  fn new(label: &'static [u8]) -> Self;

  /// returns a scalar element of the group as a challenge
  fn squeeze(&mut self, label: &'static [u8]) -> Result<E::Scalar, SpartanError>;

  /// absorbs any type that implements `TranscriptReprTrait` under a label
  fn absorb<T: TranscriptReprTrait<E::GE>>(&mut self, label: &'static [u8], o: &T);

  /// adds a domain separator
  fn dom_sep(&mut self, bytes: &'static [u8]);
}

/// Defines additional methods on `PrimeField` objects
pub trait PrimeFieldExt: PrimeField {
  /// Returns a scalar representing the bytes
  fn from_uniform(bytes: &[u8]) -> Self;
}

impl<G: Group, T: TranscriptReprTrait<G>> TranscriptReprTrait<G> for &[T] {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self
      .iter()
      .flat_map(|t| t.to_transcript_bytes())
      .collect::<Vec<u8>>()
  }
}

pub mod evaluation;
pub mod snark;
