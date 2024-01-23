//! This library implements Spartan, a high-speed SNARK.
#![deny(
  warnings,
//  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![forbid(unsafe_code)]

// private modules
mod bellpepper;
mod constants;
mod digest;
mod r1cs;

// public modules
pub mod errors;
pub mod provider;
pub mod spartan;
pub mod traits;

use bellpepper_core::Circuit;
use core::marker::PhantomData;
use errors::SpartanError;
use serde::{Deserialize, Serialize};
use traits::{
  commitment::{CommitmentEngineTrait, CommitmentTrait},
  snark::RelaxedR1CSSNARKTrait, 
  upsnark::{UniformSNARKTrait, PrecommittedSNARKTrait}, 
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
  C: Circuit<G::Scalar>,
{
  snark: S, // snark proving the witness is satisfying
  _p: PhantomData<G>,
  _p2: PhantomData<C>,
}

impl<G: Group, S: RelaxedR1CSSNARKTrait<G> + UniformSNARKTrait<G> + PrecommittedSNARKTrait<G>, C: Circuit<G::Scalar>> SNARK<G, S, C> {
  /// Produces prover and verifier keys for the direct SNARK
  pub fn setup(circuit: C) -> Result<(ProverKey<G, S>, VerifierKey<G, S>), SpartanError> {
    let (pk, vk) = S::setup(circuit)?;
    Ok((ProverKey { pk }, VerifierKey { vk }))
  }

  /// Produces prover and verifier keys for the direct SNARK
  pub fn setup_uniform(circuit: C, n: usize) -> Result<(ProverKey<G, S>, VerifierKey<G, S>), SpartanError> {
    let (pk, vk) = S::setup_uniform(circuit, n)?;
    Ok((ProverKey { pk }, VerifierKey { vk }))
  }

  /// Produces prover and verifier keys for the direct SNARK
  pub fn setup_precommitted(circuit: C, n: usize) -> Result<(ProverKey<G, S>, VerifierKey<G, S>), SpartanError> {
    let (pk, vk) = S::setup_precommitted(circuit, n)?;
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

type CommitmentKey<G> = <<G as traits::Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey;
type Commitment<G> = <<G as Group>::CE as CommitmentEngineTrait<G>>::Commitment;
type CompressedCommitment<G> = <<<G as Group>::CE as CommitmentEngineTrait<G>>::Commitment as CommitmentTrait<G>>::CompressedCommitment;
type CE<G> = <G as Group>::CE;

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::{bn256_grumpkin::bn256, secp_secq::secp256k1};
  use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
  use core::marker::PhantomData;
  use ff::PrimeField;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> Circuit<F> for CubicCircuit<F>
  where
    F: PrimeField,
  {
    fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(F::ONE + F::ONE))?;
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
      })?;
      let z = AllocatedNum::alloc(cs.namespace(|| "z"), || Ok(F::from(1u64)))?;

      cs.enforce(
        || "y = x^3 + x + 5",
        |lc| {
          lc + x_cu.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + y.get_variable(),
      );

      cs.enforce(
        || "z = 1",
        |lc| lc + z.get_variable(),
        |lc| lc + CS::one() - z.get_variable(),
        |lc| lc,
      );

      let _ = y.inputize(cs.namespace(|| "output"));

      Ok(())
    }
  }

  #[test]
  fn test_snark_hyrax_pc() {
    type G = pasta_curves::pallas::Point;
    type EE = crate::provider::hyrax_pc::HyraxEvaluationEngine<G>;
    type S = crate::spartan::relaxed_snark::RelaxedR1CSSNARK<G, EE>;
    //type Spp = crate::spartan::ppsnark::RelaxedR1CSSNARK<G, EE>;
    test_snark_with::<G, S>();
    //test_snark_with::<G, Spp>();

    type G2 = bn256::Point;
    type EE2 = crate::provider::hyrax_pc::HyraxEvaluationEngine<G2>;
    type S2 = crate::spartan::relaxed_snark::RelaxedR1CSSNARK<G2, EE2>;
    test_snark_with::<G2, S2>();
    //test_snark_with::<G2, S2pp>();

    type G3 = secp256k1::Point;
    type EE3 = crate::provider::hyrax_pc::HyraxEvaluationEngine<G3>;
    type S3 = crate::spartan::relaxed_snark::RelaxedR1CSSNARK<G3, EE3>;
    //type S3pp = crate::spartan::ppsnark::RelaxedR1CSSNARK<G3, EE3>;
    test_snark_with::<G3, S3>();
    //test_snark_with::<G3, S3pp>();
  }

  fn test_snark_with<G: Group, S: RelaxedR1CSSNARKTrait<G>>() {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) =
      SNARK::<G, S, CubicCircuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

    // produce a SNARK
    let res = SNARK::prove(&pk, circuit);
    assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk, &[<G as Group>::Scalar::from(15u64)]);
    assert!(res.is_ok());
  }
}
