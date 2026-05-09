// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 chain circuit using small_sha256 (small-value compatible).

use super::{
  bytes_to_public_scalars,
  small::{alloc_preimage_small_bits, expose_small_hash_bits_as_public},
};
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::{PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use std::marker::PhantomData;

use crate::{
  gadgets::{NoBatchEq, SmallBoolean, small_sha256_int_with_prefix},
  small_constraint_system::SmallToBellpepperCS,
  traits::{Engine, circuit::SpartanCircuit},
};

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
  /// Create a new SHA-256 chain circuit.
  pub fn new(input: [u8; 32], chain_length: usize) -> Self {
    Self {
      input,
      chain_length,
      _p: PhantomData,
    }
  }

  /// Compute the expected final hash by applying SHA-256 chain_length times.
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
    Ok(bytes_to_public_scalars(&self.expected_output()))
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
    let mut small_cs = SmallToBellpepperCS::<E::Scalar, CS>::new(cs);
    let mut current_bits = alloc_preimage_small_bits::<i8, _>(&mut small_cs, &self.input)?;

    {
      let mut eq = NoBatchEq::<i8, i32, _>::new(&mut small_cs);
      for chain_idx in 0..self.chain_length {
        let prefix = format!("c{}_", chain_idx);
        current_bits = small_sha256_int_with_prefix::<i8, _>(&mut eq, &current_bits, &prefix)?;
      }
    }

    assert_small_bits_match_bytes(&current_bits, &self.expected_output());

    expose_small_hash_bits_as_public::<i8, _>(&mut small_cs, &current_bits)?;

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
    let mut small_cs = SmallToBellpepperCS::<Scalar, CS>::new(cs);
    let mut current_bits = alloc_preimage_small_bits::<i8, _>(&mut small_cs, &self.input)?;

    {
      let mut eq = NoBatchEq::<i8, i32, _>::new(&mut small_cs);
      for chain_idx in 0..self.chain_length {
        let prefix = format!("c{}_", chain_idx);
        current_bits = small_sha256_int_with_prefix::<i8, _>(&mut eq, &current_bits, &prefix)?;
      }
    }

    Ok(())
  }
}

fn assert_small_bits_match_bytes(bits: &[SmallBoolean], expected_bytes: &[u8]) {
  let expected_bits: Vec<bool> = expected_bytes
    .iter()
    .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1))
    .collect();

  assert_eq!(bits.len(), expected_bits.len());
  for (i, (computed, expected_bit)) in bits.iter().zip(expected_bits.iter()).enumerate() {
    assert_eq!(
      computed.get_value(),
      Some(*expected_bit),
      "Hash bit {i} mismatch"
    );
  }
}
