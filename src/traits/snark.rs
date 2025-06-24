//! This module defines a collection of traits that define the behavior of a zkSNARK for RelaxedR1CS
use crate::{
  errors::SpartanError,
  r1cs::{R1CSInstance, R1CSWitness},
  traits::{Engine, Group, TranscriptReprTrait},
};
use bellpepper_core::Circuit;
use serde::{Deserialize, Serialize};

/// A trait that defines the behavior of a zkSNARK
pub trait R1CSSNARKTrait<E: Engine>:
  Sized + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// A type that represents the prover's key
  type ProverKey: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that represents the verifier's key
  type VerifierKey: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Produces the keys for the prover and the verifier
  fn setup<C: Circuit<E::Scalar>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError>;

  /// Produces witness and instance for a given circuit
  fn gen_witness<C: Circuit<E::Scalar>>(
    pk: &Self::ProverKey,
    circuit: C,
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<(R1CSInstance<E>, R1CSWitness<E>), SpartanError>;

  /// Produces a new SNARK for a relaxed R1CS
  fn prove(
    pk: &Self::ProverKey,
    U: &R1CSInstance<E>,
    W: &R1CSWitness<E>,
  ) -> Result<Self, SpartanError>;

  /// Verifies a SNARK for a relaxed R1CS and returns the public IO
  fn verify(&self, vk: &Self::VerifierKey) -> Result<Vec<E::Scalar>, SpartanError>;
}

/// A type representing the digest of a verifier's key
pub type SpartanDigest = [u8; 32];

/// A helper trait that defines the behavior of a verifier key of `zkSNARK`
pub trait DigestHelperTrait<E: Engine> {
  /// Returns the digest of the verifier's key
  fn digest(&self) -> Result<SpartanDigest, SpartanError>;
}

// implement TranscriptReprTrait for the SpartanDigest
impl<G: Group> TranscriptReprTrait<G> for SpartanDigest {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_vec()
  }
}
