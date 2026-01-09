// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 circuit using small_sha256 gadget (small-value compatible).

use super::{
  alloc_preimage_bits, assert_hash_matches, expose_hash_bits_as_public, hash_to_public_scalars,
};
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::{PrimeField, PrimeFieldBits};
use spartan2::{
  gadgets::{NoBatchEq, small_sha256, small_sha256_with_small_multi_eq},
  traits::{Engine, circuit::SpartanCircuit},
};
use std::marker::PhantomData;

/// SHA-256 circuit using small_sha256 gadget (small-value compatible).
///
/// Uses `SmallMultiEq` to keep coefficients bounded for small-value sumcheck.
#[derive(Debug, Clone)]
pub struct SmallSha256Circuit<Scalar: PrimeField> {
  pub preimage: Vec<u8>,
  /// If true, use BatchingEq<21> (i64 path); if false, use NoBatchEq (i32 path)
  pub use_batching: bool,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> SmallSha256Circuit<Scalar> {
  pub fn new(preimage: Vec<u8>, use_batching: bool) -> Self {
    Self {
      preimage,
      use_batching,
      _p: PhantomData,
    }
  }
}

impl<E: Engine> SpartanCircuit<E> for SmallSha256Circuit<E::Scalar>
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
    // Allocate preimage bits (big-endian for small_sha256)
    let preimage_bits = alloc_preimage_bits::<E::Scalar, _>(cs, &self.preimage, true)?;

    // SmallSHA-256 gadget
    let hash_bits = if self.use_batching {
      small_sha256(cs, &preimage_bits)?
    } else {
      let mut eq = NoBatchEq::<E::Scalar, _>::new(cs);
      let bits = small_sha256_with_small_multi_eq(&mut eq, &preimage_bits, "")?;
      drop(eq);
      bits
    };

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

impl<Scalar: PrimeField + PrimeFieldBits> Circuit<Scalar> for SmallSha256Circuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let preimage_bits = alloc_preimage_bits(cs, &self.preimage, true)?;
    let _ = if self.use_batching {
      small_sha256(cs, &preimage_bits)?
    } else {
      let mut eq = NoBatchEq::<Scalar, _>::new(cs);
      let bits = small_sha256_with_small_multi_eq(&mut eq, &preimage_bits, "")?;
      drop(eq);
      bits
    };
    Ok(())
  }
}
