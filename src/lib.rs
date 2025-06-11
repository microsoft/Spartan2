//! This library implements Spartan, a high-speed SNARK.
#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![forbid(unsafe_code)]

// private modules
mod digest;
mod math;
mod polys;
mod r1cs;
mod snark;
mod sumcheck;

#[macro_use]
mod macros;

// public modules
pub mod bellpepper;
pub mod errors;
pub mod provider;
pub mod traits;

use bellpepper_core::Circuit;
use core::marker::PhantomData;
use errors::SpartanError;
use serde::{Deserialize, Serialize};
use traits::{commitment::CommitmentEngineTrait, snark::R1CSSNARKTrait, Engine};

/// A type that holds the prover key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E, S>
where
  E: Engine,
  S: R1CSSNARKTrait<E>,
{
  pk: S::ProverKey,
}

/// A type that holds the verifier key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E, S>
where
  E: Engine,
  S: R1CSSNARKTrait<E>,
{
  vk: S::VerifierKey,
}

/// A SNARK proving a circuit expressed with bellperson
/// This module provides interfaces to directly prove a step circuit by using Spartan SNARK.
/// In particular, it supports any SNARK that implements R1CSSNARK trait
/// (e.g., with the SNARKs implemented in ppsnark.rs or snark.rs).
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SNARK<E, S, C>
where
  E: Engine,
  S: R1CSSNARKTrait<E>,
  C: Circuit<E::Scalar>,
{
  snark: S, // snark proving the witness is satisfying
  _p: PhantomData<(E, C)>,
}

impl<E: Engine, S: R1CSSNARKTrait<E>, C: Circuit<E::Scalar>> SNARK<E, S, C> {
  /// Produces prover and verifier keys for the direct SNARK
  pub fn setup(circuit: C) -> Result<(ProverKey<E, S>, VerifierKey<E, S>), SpartanError> {
    let (pk, vk) = S::setup(circuit)?;

    Ok((ProverKey { pk }, VerifierKey { vk }))
  }

  /// Produces a proof of satisfiability of the provided circuit
  pub fn prove(pk: &ProverKey<E, S>, circuit: C) -> Result<Self, SpartanError> {
    // prove the instance using Spartan
    let snark = S::prove(&pk.pk, circuit)?;

    Ok(SNARK {
      snark,
      _p: Default::default(),
    })
  }

  /// Verifies a proof of satisfiability
  pub fn verify(&self, vk: &VerifierKey<E, S>, io: &[E::Scalar]) -> Result<(), SpartanError> {
    // verify the snark using the constructed instance
    self.snark.verify(&vk.vk, io)
  }
}

type CommitmentKey<E> = <<E as traits::Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey;
type Commitment<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment;
type CE<E> = <E as Engine>::CE;
type DerandKey<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::DerandKey;

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
  use ff::PrimeField;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit {}

  impl<F> Circuit<F> for CubicCircuit
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

      let _ = y.inputize(cs.namespace(|| "output"));

      Ok(())
    }
  }

  #[test]
  fn test_snark() {
    type E = crate::provider::PallasEngine;
    type EE = crate::provider::ipa_pc::EvaluationEngine<E>;
    type S = crate::snark::R1CSSNARK<E, EE>;
    test_snark_with::<E, S>();
  }

  fn test_snark_with<E: Engine, S: R1CSSNARKTrait<E>>() {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) = SNARK::<E, S, CubicCircuit>::setup(circuit.clone()).unwrap();

    // produce a SNARK
    let res = SNARK::prove(&pk, circuit);
    assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk, &[<E as Engine>::Scalar::from(15u64)]);
    assert!(res.is_ok());
  }
}
