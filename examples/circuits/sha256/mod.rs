// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 circuits for examples and benchmarks.
//!
//! This module provides reusable SHA-256 circuit implementations:
//! - [`Sha256Circuit`]: Uses bellpepper's SHA-256 (baseline, not small-value compatible)
//! - [`SmallSha256Circuit`]: Uses small_sha256 gadget (small-value compatible)
//! - [`SmallSha256ChainCircuit`]: Chains multiple small_sha256 calls

mod bellpepper;
mod chain;
mod small;

pub use bellpepper::Sha256Circuit;
pub use chain::SmallSha256ChainCircuit;
pub use small::SmallSha256Circuit;

use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use ff::{Field, PrimeField};
use sha2::{Digest, Sha256};
use spartan2::traits::Engine;

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute SHA-256 hash and convert to public value scalars.
///
/// Each bit of the 256-bit hash becomes a field element (0 or 1).
pub fn hash_to_public_scalars<F: PrimeField>(data: &[u8]) -> Vec<F> {
  let mut hasher = Sha256::new();
  hasher.update(data);
  let hash = hasher.finalize();

  hash
    .iter()
    .flat_map(|&byte| {
      (0..8).rev().map(move |i| {
        if (byte >> i) & 1 == 1 {
          F::ONE
        } else {
          F::ZERO
        }
      })
    })
    .collect()
}

/// Allocate preimage bytes as witness bits.
///
/// If `big_endian` is true, bits are allocated in big-endian order per byte
/// (required for small_sha256). If false, little-endian (bellpepper's convention).
pub fn alloc_preimage_bits<Scalar, CS>(
  cs: &mut CS,
  preimage: &[u8],
  big_endian: bool,
) -> Result<Vec<Boolean>, SynthesisError>
where
  Scalar: PrimeField,
  CS: ConstraintSystem<Scalar>,
{
  preimage
    .iter()
    .enumerate()
    .flat_map(|(byte_idx, &byte)| {
      let bit_iter: Box<dyn Iterator<Item = (usize, bool)>> = if big_endian {
        Box::new(
          (0..8)
            .rev()
            .enumerate()
            .map(move |(bit_idx, i)| (bit_idx, (byte >> i) & 1 == 1)),
        )
      } else {
        Box::new((0..8).map(move |i| (i, (byte >> i) & 1 == 1)))
      };
      bit_iter.map(move |(bit_idx, bit_val)| (byte_idx, bit_idx, bit_val))
    })
    .map(|(byte_idx, bit_idx, bit_val)| {
      AllocatedBit::alloc(
        cs.namespace(|| format!("preimage_{}_{}", byte_idx, bit_idx)),
        Some(bit_val),
      )
      .map(Boolean::from)
    })
    .collect()
}

/// Assert that computed hash bits match native SHA-256 output.
///
/// Panics if there's a mismatch (debug assertion).
pub fn assert_hash_matches(hash_bits: &[Boolean], preimage: &[u8]) {
  let mut hasher = Sha256::new();
  hasher.update(preimage);
  let expected = hasher.finalize();
  assert_bits_match_bytes(hash_bits, &expected);
}

/// Assert that computed bits match expected bytes (big-endian per byte).
///
/// Panics if there's a mismatch (debug assertion).
pub fn assert_bits_match_bytes(bits: &[Boolean], expected_bytes: &[u8]) {
  let expected_bits: Vec<bool> = expected_bytes
    .iter()
    .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1))
    .collect();

  for (i, (computed, expected_bit)) in bits.iter().zip(expected_bits.iter()).enumerate() {
    let computed_val = match computed {
      Boolean::Is(bit) => bit.get_value().unwrap(),
      Boolean::Not(bit) => !bit.get_value().unwrap(),
      Boolean::Constant(b) => *b,
    };
    assert_eq!(
      computed_val, *expected_bit,
      "Hash bit {} mismatch: computed={}, expected={}",
      i, computed_val, expected_bit
    );
  }
}

/// Expose hash bits as public inputs with equality constraints.
pub fn expose_hash_bits_as_public<E, CS>(
  cs: &mut CS,
  hash_bits: &[Boolean],
) -> Result<(), SynthesisError>
where
  E: Engine,
  CS: ConstraintSystem<E::Scalar>,
{
  for (i, bit) in hash_bits.iter().enumerate() {
    let n = AllocatedNum::alloc_input(cs.namespace(|| format!("public num {i}")), || {
      Ok(
        if bit.get_value().ok_or(SynthesisError::AssignmentMissing)? {
          E::Scalar::ONE
        } else {
          E::Scalar::ZERO
        },
      )
    })?;

    cs.enforce(
      || format!("bit == num {i}"),
      |_| bit.lc(CS::one(), E::Scalar::ONE),
      |lc| lc + CS::one(),
      |lc| lc + n.get_variable(),
    );
  }
  Ok(())
}
