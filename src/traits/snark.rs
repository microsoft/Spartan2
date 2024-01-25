//! This module defines a collection of traits that define the behavior of a zkSNARK for RelaxedR1CS
use crate::{errors::SpartanError, traits::Group};
use bellpepper_core::Circuit;
use serde::{Deserialize, Serialize};

/// A trait that defines the behavior of a zkSNARK
pub trait RelaxedR1CSSNARKTrait<G: Group>:
  Sized + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// A type that represents the prover's key
  type ProverKey: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that represents the verifier's key
  type VerifierKey: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Produces the keys for the prover and the verifier
  fn setup<C: Circuit<G::Scalar>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError>;

  /// Produces a new SNARK for a relaxed R1CS
  fn prove<C: Circuit<G::Scalar>>(pk: &Self::ProverKey, circuit: C) -> Result<Self, SpartanError>;

  /// Verifies a SNARK for a relaxed R1CS
  fn verify(&self, vk: &Self::VerifierKey, io: &[G::Scalar]) -> Result<(), SpartanError>;
}

