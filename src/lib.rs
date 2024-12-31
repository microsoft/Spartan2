//! This library implements Spartan, a high-speed SNARK.
#![deny(
  // TODO: Uncomment
  // warnings,
  // unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  // TODO: Uncomment
  // missing_docs
)]
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![forbid(unsafe_code)]

// private modules
mod bellpepper; // TODO: Replace with arkworks module?
mod constants;
mod digest;
mod r1cs;

// public modules
pub mod errors;
pub mod provider;
pub mod spartan;
pub mod traits;

use ark_relations::r1cs::ConstraintSynthesizer;
use core::marker::PhantomData;
use errors::SpartanError;
use serde::{Deserialize, Serialize};
use traits::{
  commitment::{CommitmentEngineTrait, CommitmentTrait},
  snark::RelaxedR1CSSNARKTrait,
  Group,
};

/// A type that holds the prover key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<G, S>
where
  G: Group,
  S: RelaxedR1CSSNARKTrait<G>,
{
  pk: S::ProverKey,
}

/// A type that holds the verifier key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<G, S>
where
  G: Group,
  S: RelaxedR1CSSNARKTrait<G>,
{
  vk: S::VerifierKey,
}

/// A SNARK proving a circuit expressed with bellperson
/// This module provides interfaces to directly prove a step circuit by using Spartan SNARK.
/// In particular, it supports any SNARK that implements RelaxedR1CSSNARK trait
/// (e.g., with the SNARKs implemented in ppsnark.rs or snark.rs).
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SNARK<G, S, C>
where
  G: Group,
  S: RelaxedR1CSSNARKTrait<G>,
  C: ConstraintSynthesizer<G::Scalar>,
{
  snark: S, // snark proving the witness is satisfying
  _p: PhantomData<G>,
  _p2: PhantomData<C>,
}

impl<G: Group, S: RelaxedR1CSSNARKTrait<G>, C: ConstraintSynthesizer<G::Scalar>> SNARK<G, S, C> {
  /// Produces prover and verifier keys for the direct SNARK
  pub fn setup(circuit: C) -> Result<(ProverKey<G, S>, VerifierKey<G, S>), SpartanError> {
    let (pk, vk) = S::setup(circuit)?;

    Ok((ProverKey { pk }, VerifierKey { vk }))
  }

  /// Produces a proof of satisfiability of the provided circuit
  pub fn prove(pk: &ProverKey<G, S>, circuit: C) -> Result<Self, SpartanError> {
    // prove the instance using Spartan
    let snark = S::prove(&pk.pk, circuit)?;

    Ok(SNARK {
      snark,
      _p: Default::default(),
      _p2: Default::default(),
    })
  }

  /// Verifies a proof of satisfiability
  pub fn verify(&self, vk: &VerifierKey<G, S>, io: &[G::Scalar]) -> Result<(), SpartanError> {
    // verify the snark using the constructed instance
    self.snark.verify(&vk.vk, io)
  }
}

type CommitmentKey<G> = <<G as Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey;
type Commitment<G> = <<G as Group>::CE as CommitmentEngineTrait<G>>::Commitment;
type CompressedCommitment<G> = <<<G as Group>::CE as CommitmentEngineTrait<G>>::Commitment as CommitmentTrait<G>>::CompressedCommitment;
type CE<G> = <G as Group>::CE;

#[cfg(test)]
mod tests {
  use super::*;
  use ark_ff::PrimeField;
  use ark_relations::lc;
  use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError, Variable};
  #[derive(Clone, Debug, Default)]
  struct CubicCircuit {}

  impl<F: PrimeField> ConstraintSynthesizer<F> for CubicCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
      // Fixed values for testing
      let x = F::from(2u64); // Example: x = 2
      let y = F::from(15u64); // Example: y = 15 (2^3 + 2 + 5 = 15)

      // 1. Allocate x^2 as an intermediate variable (witness)
      let x_squared_var = cs.new_witness_variable(|| Ok(x * x))?;

      // Enforce x^2 = x * x
      cs.enforce_constraint(
        lc!() + (x, Variable::One),
        lc!() + (x, Variable::One),
        lc!() + x_squared_var,
      )?;

      // 2. Allocate x^3 as another intermediate variable (witness)
      let x_cubed_var = cs.new_witness_variable(|| Ok(x * x * x))?;

      // Enforce x^3 = x^2 * x
      cs.enforce_constraint(
        lc!() + x_squared_var,
        lc!() + (x, Variable::One),
        lc!() + x_cubed_var,
      )?;

      // 3. Add the constraint: x^3 + x + 5 = y
      cs.enforce_constraint(
        lc!() + x_cubed_var + (x, Variable::One) + (F::from(5u64), Variable::One),
        lc!() + Variable::One, // Identity multiplier for y
        lc!() + (y, Variable::One),
      )?;

      // 4. Expose y as a public input
      cs.new_input_variable(|| Ok(y))?;

      Ok(())
    }
  }

  #[test]
  fn test_snark() {
    type G = ark_bls12_381::G1Projective;
    type EE = provider::ipa_pc::EvaluationEngine<G>;
    type S = spartan::snark::RelaxedR1CSSNARK<G, EE>;
    type Spp = spartan::ppsnark::RelaxedR1CSSNARK<G, EE>;
    // test_snark_with::<G, S>(); // TODO
    // test_snark_with::<G, Spp>(); // TODO

    // type G2 = bn256::Point;
    // type EE2 = crate::provider::ipa_pc::EvaluationEngine<G2>;
    // type S2 = crate::spartan::snark::RelaxedR1CSSNARK<G2, EE2>;
    // type S2pp = crate::spartan::ppsnark::RelaxedR1CSSNARK<G2, EE2>;
    // test_snark_with::<G2, S2>();
    // test_snark_with::<G2, S2pp>();

    // type G3 = secp256k1::Point;
    // type EE3 = crate::provider::ipa_pc::EvaluationEngine<G3>;
    // type S3 = crate::spartan::snark::RelaxedR1CSSNARK<G3, EE3>;
    // type S3pp = crate::spartan::ppsnark::RelaxedR1CSSNARK<G3, EE3>;
    // test_snark_with::<G3, S3>();
    // test_snark_with::<G3, S3pp>();

    type G3 = ark_bls12_381::G1Projective;
    type EE3 = provider::ipa_pc::EvaluationEngine<G3>;
    type S3 = spartan::snark::RelaxedR1CSSNARK<G3, EE3>;
    type S3pp = spartan::ppsnark::RelaxedR1CSSNARK<G3, EE3>;
    test_snark_with::<G3, S3>();
    test_snark_with::<G3, S3pp>();
  }

  fn test_snark_with<G: Group, S: RelaxedR1CSSNARKTrait<G>>() {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) = SNARK::<G, S, CubicCircuit>::setup(circuit.clone()).unwrap();

    // produce a SNARK
    let res = SNARK::prove(&pk, circuit);
    assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk, &[<G as Group>::Scalar::from(15u64)]);
    // assert!(res.is_ok());
    res.unwrap();
  }
}
