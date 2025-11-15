// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module defines a collection of traits that define the behavior of a zkSNARK for RelaxedR1CS
use crate::{
  errors::SpartanError,
  traits::{Engine, Group, TranscriptReprTrait, circuit::SpartanCircuit},
};
use serde::{Deserialize, Serialize};

/// A trait that defines the behavior of a zkSNARK
pub trait R1CSSNARKTrait<E: Engine>:
  Sized + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// A type that represents the prover's key
  type ProverKey: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that represents the verifier's key
  type VerifierKey: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that holds the prep work for producing the SNARK
  type PrepSNARK: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Produces the keys for the prover and the verifier
  fn setup<C: SpartanCircuit<E>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError>;

  /// Prepares the SNARK for proving, given a prover key and a circuit
  fn prep_prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<Self::PrepSNARK, SpartanError>;

  /// Produces witness and instance for a given circuit, and proves it
  fn prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    prep_snark: &Self::PrepSNARK,
    is_small: bool, // do witness elements fit in machine words?
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
