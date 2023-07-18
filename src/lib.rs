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
mod bellperson;
mod r1cs;

// public modules
pub mod errors;
pub mod provider;
pub mod spartan;
pub mod traits;

use crate::bellperson::{
  r1cs::{SpartanShape, SpartanWitness},
  shape_cs::ShapeCS,
  solver::SatisfyingAssignment,
};
use ::bellperson::{Circuit, ConstraintSystem};
use core::marker::PhantomData;
use errors::SpartanError;
use ff::Field;
use r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
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
  S: R1CSShape<G>,
  ck: CommitmentKey<G>,
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
  comm_W: Commitment<G>, // commitment to the witness
  io: Vec<G::Scalar>,    // public IO
  snark: S,              // snark proving the witness is satisfying
  _p: PhantomData<C>,
}

impl<G: Group, S: RelaxedR1CSSNARKTrait<G>, C: Circuit<G::Scalar>> SNARK<G, S, C> {
  /// Produces prover and verifier keys for the direct SNARK
  pub fn setup(circuit: C) -> Result<(ProverKey<G, S>, VerifierKey<G, S>), SpartanError> {
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (shape, ck) = cs.r1cs_shape();

    let (pk, vk) = S::setup(&ck, &shape)?;
    let pk = ProverKey { S: shape, ck, pk };
    let vk = VerifierKey { vk };

    Ok((pk, vk))
  }

  /// Produces a proof of satisfiability of the provided circuit
  pub fn prove(pk: &ProverKey<G, S>, circuit: C) -> Result<Self, SpartanError> {
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let _ = circuit.synthesize(&mut cs);
    let (u, w) = cs
      .r1cs_instance_and_witness(&pk.S, &pk.ck)
      .map_err(|_e| SpartanError::UnSat)?;

    // convert the instance and witness to relaxed form
    let (u_relaxed, w_relaxed) = (
      RelaxedR1CSInstance::from_r1cs_instance_unchecked(&u.comm_W, &u.X),
      RelaxedR1CSWitness::from_r1cs_witness(&pk.S, &w),
    );

    // prove the instance using Spartan
    let snark = S::prove(&pk.ck, &pk.pk, &u_relaxed, &w_relaxed)?;

    Ok(SNARK {
      comm_W: u.comm_W,
      io: u.X,
      snark,
      _p: Default::default(),
    })
  }

  /// Verifies a proof of satisfiability
  pub fn verify(&self, vk: &VerifierKey<G, S>) -> Result<Vec<G::Scalar>, SpartanError> {
    // construct an instance using the provided commitment to the witness and IO
    let u_relaxed = RelaxedR1CSInstance::from_r1cs_instance_unchecked(&self.comm_W, &self.io);

    // verify the snark using the constructed instance
    self.snark.verify(&vk.vk, &u_relaxed)?;

    Ok(self.io.clone())
  }
}

type CommitmentKey<G> = <<G as traits::Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey;
type Commitment<G> = <<G as Group>::CE as CommitmentEngineTrait<G>>::Commitment;
type CompressedCommitment<G> = <<<G as Group>::CE as CommitmentEngineTrait<G>>::Commitment as CommitmentTrait<G>>::CompressedCommitment;
type CE<G> = <G as Group>::CE;

fn compute_digest<G: Group, T: Serialize>(o: &T) -> G::Scalar {
  // obtain a vector of bytes representing public parameters
  let bytes = bincode::serialize(o).unwrap();
  // convert pp_bytes into a short digest
  let mut hasher = Sha3_256::new();
  hasher.update(&bytes);
  let digest = hasher.finalize();

  // truncate the digest to NUM_HASH_BITS bits
  let bv = (0..250).map(|i| {
    let (byte_pos, bit_pos) = (i / 8, i % 8);
    let bit = (digest[byte_pos] >> bit_pos) & 1;
    bit == 1
  });

  // turn the bit vector into a scalar
  let mut digest = G::Scalar::ZERO;
  let mut coeff = G::Scalar::ONE;
  for bit in bv {
    if bit {
      digest += coeff;
    }
    coeff += coeff;
  }
  digest
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::bn256_grumpkin::bn256;
  use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
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
    type G = pasta_curves::pallas::Point;
    type EE = crate::provider::ipa_pc::EvaluationEngine<G>;
    type S = crate::spartan::snark::RelaxedR1CSSNARK<G, EE>;
    type Spp = crate::spartan::ppsnark::RelaxedR1CSSNARK<G, EE>;
    test_snark_with::<G, S>();
    test_snark_with::<G, Spp>();

    type G2 = bn256::Point;
    type EE2 = crate::provider::ipa_pc::EvaluationEngine<G2>;
    type S2 = crate::spartan::snark::RelaxedR1CSSNARK<G2, EE2>;
    type S2pp = crate::spartan::ppsnark::RelaxedR1CSSNARK<G2, EE2>;
    test_snark_with::<G2, S2>();
    test_snark_with::<G2, S2pp>();
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
    let res = snark.verify(&vk);
    assert!(res.is_ok());

    let io = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(io, vec![<G as Group>::Scalar::from(15u64)]);
  }
}
