// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 chain circuit using small_sha256 (small-value compatible).

use super::{
  alloc_preimage_bits, assert_bits_match_bytes, expose_hash_bits_as_public, hash_to_public_scalars,
};
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::{PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use spartan2::{
  gadgets::small_sha256_with_prefix,
  traits::{Engine, circuit::SpartanCircuit},
};
use std::marker::PhantomData;

/// SHA-256 chain circuit using small_sha256 (small-value compatible).
///
/// Chains `chain_length` SHA-256 hashes starting from a 256-bit input.
/// Hash[0] = SHA-256(input), Hash[i] = SHA-256(Hash[i-1])
#[derive(Debug, Clone)]
pub struct SmallSha256ChainCircuit<Scalar: PrimeField> {
  /// 32-byte (256-bit) input to start the chain
  pub input: [u8; 32],
  /// Number of SHA-256 hashes in the chain
  pub chain_length: usize,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> SmallSha256ChainCircuit<Scalar> {
  pub fn new(input: [u8; 32], chain_length: usize) -> Self {
    Self {
      input,
      chain_length,
      _p: PhantomData,
    }
  }

  /// Compute the expected final hash by applying SHA-256 chain_length times
  pub fn expected_output(&self) -> [u8; 32] {
    let mut current = self.input;
    for _ in 0..self.chain_length {
      let mut hasher = Sha256::new();
      hasher.update(current);
      current = hasher.finalize().into();
    }
    current
  }
}

impl<E: Engine> SpartanCircuit<E> for SmallSha256ChainCircuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    Ok(hash_to_public_scalars(&self.expected_output()))
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
    // Allocate input bits (big-endian for small_sha256)
    let mut current_bits = alloc_preimage_bits::<E::Scalar, _>(cs, &self.input, true)?;

    // Chain SHA-256 hashes
    for chain_idx in 0..self.chain_length {
      let prefix = format!("c{}_", chain_idx);
      let hash_bits = small_sha256_with_prefix(cs, &current_bits, &prefix)?;
      current_bits = hash_bits;
    }

    // Verify against expected output (already a hash, don't re-hash)
    assert_bits_match_bytes(&current_bits, &self.expected_output());

    // Expose as public inputs
    expose_hash_bits_as_public::<E, _>(cs, &current_bits)?;

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

impl<Scalar: PrimeField + PrimeFieldBits> Circuit<Scalar> for SmallSha256ChainCircuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // Allocate input bits (big-endian for small_sha256)
    let mut current_bits = alloc_preimage_bits(cs, &self.input, true)?;

    // Chain SHA-256 hashes
    for chain_idx in 0..self.chain_length {
      let prefix = format!("c{}_", chain_idx);
      let hash_bits = small_sha256_with_prefix(cs, &current_bits, &prefix)?;
      current_bits = hash_bits;
    }

    Ok(())
  }
}
