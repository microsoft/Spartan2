// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 circuit using bellpepper's implementation (baseline, not small-value compatible).

use super::{
  alloc_preimage_bits, assert_hash_matches, expose_hash_bits_as_public, hash_to_public_scalars,
};
use bellpepper::gadgets::sha256::sha256 as bellpepper_sha256;
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::{PrimeField, PrimeFieldBits};
use spartan2::traits::{Engine, circuit::SpartanCircuit};
use std::marker::PhantomData;

/// SHA-256 circuit using bellpepper's implementation.
///
/// This is the baseline circuit for comparison. It produces coefficients ~2^237
/// which breaks small-value optimization.
#[derive(Clone, Debug)]
pub struct Sha256Circuit<Scalar: PrimeField> {
  pub preimage: Vec<u8>,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> Sha256Circuit<Scalar> {
  pub fn new(preimage: Vec<u8>) -> Self {
    Self {
      preimage,
      _p: PhantomData,
    }
  }
}

impl<E: Engine> SpartanCircuit<E> for Sha256Circuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    Ok(hash_to_public_scalars(&self.preimage))
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // Allocate preimage bits (little-endian for bellpepper)
    let preimage_bits = alloc_preimage_bits::<E::Scalar, _>(cs, &self.preimage, false)?;

    // SHA-256 gadget
    let hash_bits = bellpepper_sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

    // Verify against native SHA-256
    assert_hash_matches(&hash_bits, &self.preimage);

    // Expose as public inputs
    expose_hash_bits_as_public::<E, _>(cs, &hash_bits)?;

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
    _: &[AllocatedNum<E::Scalar>],
    _: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

impl<Scalar: PrimeField + PrimeFieldBits> Circuit<Scalar> for Sha256Circuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let preimage_bits = alloc_preimage_bits(cs, &self.preimage, false)?;
    let _ = bellpepper_sha256(cs.namespace(|| "sha256"), &preimage_bits)?;
    Ok(())
  }
}
