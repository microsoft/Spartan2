//! This module defines various traits required by the users of the library to implement.
use crate::errors::SpartanError;
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use num_bigint::BigInt;

use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub mod commitment;

use commitment::CommitmentEngineTrait;

/// Represents an element of a group
/// This is currently tailored for an elliptic curve group
pub trait Group:
  Clone
  + Copy
  + Debug
  + Eq
  + Sized
  + GroupOps
  + GroupOpsOwned
  + ScalarMul<<Self as Group>::Scalar>
  + ScalarMulOwned<<Self as Group>::Scalar>
  + Send
  + Sync
  + CanonicalSerialize
  + CanonicalDeserialize
{
  /// A type representing an element of the base field of the group
  type Base: PrimeField + TranscriptReprTrait<Self>;

  /// A type representing an element of the scalar field of the group
  type Scalar: PrimeField
    + PrimeFieldExt
    + Send
    + Sync
    + TranscriptReprTrait<Self>
    + CanonicalSerialize
    + CanonicalSerialize;

  /// A type representing the compressed version of the group element
  type CompressedGroupElement: CompressedGroup<GroupElement = Self>;

  /// A type representing preprocessed group element
  type PreprocessedGroupElement: Clone
    + Debug
    + Send
    + Sync
    + CanonicalSerialize
    + CanonicalDeserialize;

  /// A type that provides a generic Fiat-Shamir transcript to be used when externalizing proofs
  type TE: TranscriptEngineTrait<Self>;

  /// A type that defines a commitment engine over scalars in the group
  type CE: CommitmentEngineTrait<Self>;

  /// A method to compute a multiexponentation
  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self;

  /// Compresses the group element
  fn compress(&self) -> Self::CompressedGroupElement;

  /// Produces a preprocessed element
  fn preprocessed(&self) -> Self::PreprocessedGroupElement;

  /// Produce a vector of group elements using a static label
  fn from_label(label: &'static [u8], n: usize) -> Vec<Self::PreprocessedGroupElement>;

  /// Returns the affine coordinates (x, y, infinty) for the point
  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool);

  /// Returns an element that is the additive identity of the group
  fn zero() -> Self;

  /// Returns the generator of the group
  fn get_generator() -> Self;

  /// Returns A, B, and the order of the group as a big integer
  fn get_curve_params() -> (Self::Base, Self::Base, BigInt);
}

/// Represents a compressed version of a group element
pub trait CompressedGroup:
  Clone
  + Copy
  + Debug
  + Eq
  + Sized
  + Send
  + Sync
  + TranscriptReprTrait<Self::GroupElement>
  + CanonicalSerialize
  + CanonicalDeserialize
  + 'static
{
  /// A type that holds the decompressed version of the compressed group element
  type GroupElement: Group + CanonicalSerialize + CanonicalDeserialize;

  /// Decompresses the compressed group element
  fn decompress(&self) -> Option<Self::GroupElement>;
}

/// A helper trait for types with a group operation.
pub trait GroupOps<Rhs = Self, Output = Self>:
  Add<Rhs, Output = Output> + Sub<Rhs, Output = Output> + AddAssign<Rhs> + SubAssign<Rhs>
{
}

impl<T, Rhs, Output> GroupOps<Rhs, Output> for T where
  T: Add<Rhs, Output = Output> + Sub<Rhs, Output = Output> + AddAssign<Rhs> + SubAssign<Rhs>
{
}

/// A helper trait for references with a group operation.
pub trait GroupOpsOwned<Rhs = Self, Output = Self>: for<'r> GroupOps<&'r Rhs, Output> {}
impl<T, Rhs, Output> GroupOpsOwned<Rhs, Output> for T where T: for<'r> GroupOps<&'r Rhs, Output> {}

/// A helper trait for types implementing group scalar multiplication.
pub trait ScalarMul<Rhs, Output = Self>: Mul<Rhs, Output = Output> + MulAssign<Rhs> {}

impl<T, Rhs, Output> ScalarMul<Rhs, Output> for T where T: Mul<Rhs, Output = Output> + MulAssign<Rhs>
{}

/// A helper trait for references implementing group scalar multiplication.
pub trait ScalarMulOwned<Rhs, Output = Self>: for<'r> ScalarMul<&'r Rhs, Output> {}
impl<T, Rhs, Output> ScalarMulOwned<Rhs, Output> for T where T: for<'r> ScalarMul<&'r Rhs, Output> {}

/// This trait allows types to implement how they want to be added to TranscriptEngine
pub trait TranscriptReprTrait<G: Group>: Send + Sync {
  /// returns a byte representation of self to be added to the transcript
  fn to_transcript_bytes(&self) -> Vec<u8>;
}

/// This trait defines the behavior of a transcript engine compatible with Spartan
pub trait TranscriptEngineTrait<G: Group>: Send + Sync {
  /// initializes the transcript
  fn new(label: &'static [u8]) -> Self;

  /// returns a scalar element of the group as a challenge
  fn squeeze(&mut self, label: &'static [u8]) -> Result<G::Scalar, SpartanError>;

  /// absorbs any type that implements TranscriptReprTrait under a label
  fn absorb<T: TranscriptReprTrait<G>>(&mut self, label: &'static [u8], o: &T);

  /// adds a domain separator
  fn dom_sep(&mut self, bytes: &'static [u8]);
}

/// Defines additional methods on PrimeField objects
pub trait PrimeFieldExt: PrimeField {
  /// Returns a scalar representing the bytes
  fn from_uniform(bytes: &[u8]) -> Self;
}

impl<G: Group, T: TranscriptReprTrait<G>> TranscriptReprTrait<G> for &[T] {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    (0..self.len())
      .map(|i| self[i].to_transcript_bytes())
      .collect::<Vec<_>>()
      .into_iter()
      .flatten()
      .collect::<Vec<u8>>()
  }
}

pub mod evaluation;
pub mod snark;
