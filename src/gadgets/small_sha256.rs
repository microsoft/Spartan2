// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 circuit using the pure-integer small-value constraint system.
//!
//! This module provides a SHA-256 implementation using `SmallConstraintSystem<W, i32>`
//! and `SmallBoolean` gadgets. All witnesses are bits (0/1), all coefficients
//! are i32 — no field elements created during circuit synthesis.
//!
//! # Usage
//!
//! ```ignore
//! use spartan2::gadgets::{small_sha256_int, NoBatchEq, SmallBoolean};
//! use spartan2::small_constraint_system::SmallShapeCS;
//!
//! // Shape extraction (i32 coefficients)
//! let mut cs = SmallShapeCS::<i32>::new();
//! let mut eq = NoBatchEq::<i8, i32, _>::new(&mut cs);
//! let input_bits: Vec<SmallBoolean> = (0..512).map(|_| SmallBoolean::constant(false)).collect();
//! let hash_bits = small_sha256_int::<i8, _>(&mut eq, &input_bits)?;
//!
//! // Witness generation (i8 witnesses)
//! let mut cs = SmallSatisfyingAssignment::<i8>::new();
//! let mut eq = NoBatchEq::<i8, i32, _>::new(&mut cs);
//! let hash_bits = small_sha256_int::<i8, _>(&mut eq, &input_bits)?;
//! ```

use super::{SmallMultiEq, SmallUInt32};
use crate::{gadgets::small_boolean::SmallBoolean, small_constraint_system::SmallConstraintSystem};
use bellpepper_core::SynthesisError;

/// SHA-256 round constants K[0..63].
const ROUND_CONSTANTS: [u32; 64] = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 initial hash values H[0..7].
const IV: [u32; 8] = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Trait alias for the witness type bounds needed by SHA-256 gadgets.
pub trait Sha256Witness: Copy + From<bool> {}
impl<T: Copy + From<bool>> Sha256Witness for T {}

/// Σ0(x) = ROTR^2(x) ⊕ ROTR^13(x) ⊕ ROTR^22(x)
fn big_sigma_0<W, CS>(mut cs: CS, x: &SmallUInt32) -> Result<SmallUInt32, SynthesisError>
where
  W: Sha256Witness,
  CS: SmallConstraintSystem<W, i32>,
{
  let r2 = x.rotr(2);
  let r13 = x.rotr(13);
  let r22 = x.rotr(22);
  let tmp = r2.xor(cs.namespace(|| "s0_r2_xor_r13"), &r13)?;
  tmp.xor(cs.namespace(|| "s0_xor_r22"), &r22)
}

/// Σ1(x) = ROTR^6(x) ⊕ ROTR^11(x) ⊕ ROTR^25(x)
fn big_sigma_1<W, CS>(mut cs: CS, x: &SmallUInt32) -> Result<SmallUInt32, SynthesisError>
where
  W: Sha256Witness,
  CS: SmallConstraintSystem<W, i32>,
{
  let r6 = x.rotr(6);
  let r11 = x.rotr(11);
  let r25 = x.rotr(25);
  let tmp = r6.xor(cs.namespace(|| "s1_r6_xor_r11"), &r11)?;
  tmp.xor(cs.namespace(|| "s1_xor_r25"), &r25)
}

/// σ0(x) = ROTR^7(x) ⊕ ROTR^18(x) ⊕ SHR^3(x)
fn small_sigma_0<W, CS>(mut cs: CS, x: &SmallUInt32) -> Result<SmallUInt32, SynthesisError>
where
  W: Sha256Witness,
  CS: SmallConstraintSystem<W, i32>,
{
  let r7 = x.rotr(7);
  let r18 = x.rotr(18);
  let s3 = x.shr(3);
  let tmp = r7.xor(cs.namespace(|| "s0_r7_xor_r18"), &r18)?;
  tmp.xor(cs.namespace(|| "s0_xor_s3"), &s3)
}

/// σ1(x) = ROTR^17(x) ⊕ ROTR^19(x) ⊕ SHR^10(x)
fn small_sigma_1<W, CS>(mut cs: CS, x: &SmallUInt32) -> Result<SmallUInt32, SynthesisError>
where
  W: Sha256Witness,
  CS: SmallConstraintSystem<W, i32>,
{
  let r17 = x.rotr(17);
  let r19 = x.rotr(19);
  let s10 = x.shr(10);
  let tmp = r17.xor(cs.namespace(|| "s1_r17_xor_r19"), &r19)?;
  tmp.xor(cs.namespace(|| "s1_xor_s10"), &s10)
}

/// SHA-256 compression function.
fn sha256_compression<W, M>(
  cs: &mut M,
  h: &mut [SmallUInt32; 8],
  w: &[SmallUInt32; 16],
  block_idx: usize,
  prefix: &str,
) -> Result<(), SynthesisError>
where
  W: Sha256Witness,
  M: SmallMultiEq<W, i32>,
{
  // Message schedule: expand 16 words to 64 words
  let mut w_expanded: Vec<SmallUInt32> = Vec::with_capacity(64);
  w_expanded.extend_from_slice(w);

  for i in 16..64 {
    let s1 = small_sigma_1::<W, _>(
      cs.namespace(|| format!("{prefix}b{block_idx}_w{i}_s1")),
      &w_expanded[i - 2],
    )?;
    let s0 = small_sigma_0::<W, _>(
      cs.namespace(|| format!("{prefix}b{block_idx}_w{i}_s0")),
      &w_expanded[i - 15],
    )?;
    let wi = cs.addmany(&[
      s1,
      w_expanded[i - 7].clone(),
      s0,
      w_expanded[i - 16].clone(),
    ])?;
    w_expanded.push(wi);
  }

  let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h_var) = (
    h[0].clone(),
    h[1].clone(),
    h[2].clone(),
    h[3].clone(),
    h[4].clone(),
    h[5].clone(),
    h[6].clone(),
    h[7].clone(),
  );

  for i in 0..64 {
    let sigma1 = big_sigma_1::<W, _>(
      cs.namespace(|| format!("{prefix}b{block_idx}_r{i}_sigma1")),
      &e,
    )?;
    let ch = SmallUInt32::sha256_ch::<W, i32, _>(
      cs.namespace(|| format!("{prefix}b{block_idx}_r{i}_ch")),
      &e,
      &f,
      &g,
    )?;
    let k = SmallUInt32::constant(ROUND_CONSTANTS[i]);
    let t1 = cs.addmany(&[h_var.clone(), sigma1, ch, k, w_expanded[i].clone()])?;

    let sigma0 = big_sigma_0::<W, _>(
      cs.namespace(|| format!("{prefix}b{block_idx}_r{i}_sigma0")),
      &a,
    )?;
    let maj = SmallUInt32::sha256_maj::<W, i32, _>(
      cs.namespace(|| format!("{prefix}b{block_idx}_r{i}_maj")),
      &a,
      &b,
      &c,
    )?;

    h_var = g;
    g = f;
    f = e;
    e = cs.addmany(&[d, t1.clone()])?;
    d = c;
    c = b;
    b = a;
    a = cs.addmany(&[t1, sigma0, maj])?;
  }

  h[0] = cs.addmany(&[h[0].clone(), a])?;
  h[1] = cs.addmany(&[h[1].clone(), b])?;
  h[2] = cs.addmany(&[h[2].clone(), c])?;
  h[3] = cs.addmany(&[h[3].clone(), d])?;
  h[4] = cs.addmany(&[h[4].clone(), e])?;
  h[5] = cs.addmany(&[h[5].clone(), f])?;
  h[6] = cs.addmany(&[h[6].clone(), g])?;
  h[7] = cs.addmany(&[h[7].clone(), h_var])?;

  Ok(())
}

/// Compute SHA-256 hash of input bits using the pure-integer constraint system.
///
/// Input bits are `SmallBoolean` values. Returns 256 output bits.
///
/// # Shape extraction
/// Pass a `SmallShapeCS<i32>` or `NoBatchEq<W, i32, SmallShapeCS<i32>>`.
/// All constraints use i32 coefficients.
pub fn small_sha256_int<W, M>(
  cs: &mut M,
  input: &[SmallBoolean],
) -> Result<Vec<SmallBoolean>, SynthesisError>
where
  W: Sha256Witness,
  M: SmallMultiEq<W, i32>,
{
  small_sha256_int_with_prefix(cs, input, "")
}

/// Compute SHA-256 with a prefix for variable names (pure-integer path).
pub fn small_sha256_int_with_prefix<W, M>(
  cs: &mut M,
  input: &[SmallBoolean],
  prefix: &str,
) -> Result<Vec<SmallBoolean>, SynthesisError>
where
  W: Sha256Witness,
  M: SmallMultiEq<W, i32>,
{
  let padded = sha256_padding(input);

  assert!(padded.len().is_multiple_of(512));
  let num_blocks = padded.len() / 512;

  let mut h: [SmallUInt32; 8] = IV.map(SmallUInt32::constant);

  for block_idx in 0..num_blocks {
    let block_start = block_idx * 512;
    let block_bits = &padded[block_start..block_start + 512];

    let mut w: [SmallUInt32; 16] = std::array::from_fn(|_| SmallUInt32::constant(0));
    for (i, w_item) in w.iter_mut().enumerate() {
      let word_bits: [SmallBoolean; 32] = std::array::from_fn(|j| block_bits[i * 32 + j].clone());
      *w_item = SmallUInt32::from_bits_be(&word_bits);
    }

    sha256_compression::<W, _>(cs, &mut h, &w, block_idx, prefix)?;
  }

  let mut output = Vec::with_capacity(256);
  for h_i in h {
    output.extend(h_i.into_bits_be());
  }

  Ok(output)
}

/// Run one SHA-256 compression block with the small-value gadget.
///
/// `input_bits` are interpreted as 16 big-endian `u32` words, and
/// `current_hash` is the 8-word SHA-256 chaining state.
pub fn small_sha256_compression_function_int<W, M>(
  cs: &mut M,
  input_bits: &[SmallBoolean],
  current_hash: &[SmallUInt32],
) -> Result<Vec<SmallUInt32>, SynthesisError>
where
  W: Sha256Witness,
  M: SmallMultiEq<W, i32>,
{
  small_sha256_compression_function_int_with_prefix(cs, input_bits, current_hash, "")
}

/// Run one SHA-256 compression block with the small-value gadget and name prefix.
pub fn small_sha256_compression_function_int_with_prefix<W, M>(
  cs: &mut M,
  input_bits: &[SmallBoolean],
  current_hash: &[SmallUInt32],
  prefix: &str,
) -> Result<Vec<SmallUInt32>, SynthesisError>
where
  W: Sha256Witness,
  M: SmallMultiEq<W, i32>,
{
  assert_eq!(input_bits.len(), 512);
  assert_eq!(current_hash.len(), 8);

  let mut h: [SmallUInt32; 8] = std::array::from_fn(|i| current_hash[i].clone());
  let mut w: [SmallUInt32; 16] = std::array::from_fn(|_| SmallUInt32::constant(0));
  for (i, w_item) in w.iter_mut().enumerate() {
    let word_bits: [SmallBoolean; 32] = std::array::from_fn(|j| input_bits[i * 32 + j].clone());
    *w_item = SmallUInt32::from_bits_be(&word_bits);
  }

  sha256_compression::<W, _>(cs, &mut h, &w, 0, prefix)?;

  Ok(h.to_vec())
}

/// SHA-256 padding using SmallBoolean bits.
fn sha256_padding(input: &[SmallBoolean]) -> Vec<SmallBoolean> {
  let msg_len = input.len();

  let mut padded_len = msg_len + 1 + 64;
  if !padded_len.is_multiple_of(512) {
    padded_len += 512 - (padded_len % 512);
  }

  let mut padded = Vec::with_capacity(padded_len);

  padded.extend_from_slice(input);
  padded.push(SmallBoolean::constant(true));

  let zero_count = padded_len - msg_len - 1 - 64;
  for _ in 0..zero_count {
    padded.push(SmallBoolean::constant(false));
  }

  let len_bits: u64 = msg_len as u64;
  for i in (0..64).rev() {
    padded.push(SmallBoolean::constant((len_bits >> i) & 1 == 1));
  }

  padded
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{gadgets::NoBatchEq, small_constraint_system::SmallShapeCS};

  /// Convert bytes to SmallBoolean bits (big-endian per byte).
  fn bytes_to_small_bits(bytes: &[u8]) -> Vec<SmallBoolean> {
    bytes
      .iter()
      .flat_map(|byte| {
        (0..8)
          .rev()
          .map(move |i| SmallBoolean::constant((byte >> i) & 1 == 1))
      })
      .collect()
  }

  /// Convert SmallBoolean bits to bytes.
  fn small_bits_to_bytes(bits: &[SmallBoolean]) -> Vec<u8> {
    assert!(bits.len().is_multiple_of(8));
    bits
      .chunks(8)
      .map(|chunk| {
        chunk.iter().fold(0u8, |acc, bit| {
          let b = bit.get_value().unwrap_or(false);
          (acc << 1) | (b as u8)
        })
      })
      .collect()
  }

  #[test]
  fn test_small_sha256_shape_empty() {
    let mut cs = SmallShapeCS::<i32>::new();
    let mut eq = NoBatchEq::<i8, i32, _>::new(&mut cs);

    // Empty input: all-constant computation, shape records 0 constraints
    let input: Vec<SmallBoolean> = vec![];
    let hash_bits = small_sha256_int::<i8, _>(&mut eq, &input).unwrap();
    assert_eq!(hash_bits.len(), 256);
  }

  #[test]
  fn test_small_sha256_shape_nonempty() {
    use crate::gadgets::small_boolean::SmallBit;
    let mut cs = SmallShapeCS::<i32>::new();
    let mut eq = NoBatchEq::<i8, i32, _>::new(&mut cs);

    // 8 allocated bit inputs → should produce constraints
    let input: Vec<SmallBoolean> = (0..8)
      .map(|i| {
        SmallBoolean::Is(
          SmallBit::alloc(&mut eq.namespace(|| format!("in{i}")), Some(false)).unwrap(),
        )
      })
      .collect();
    let hash_bits = small_sha256_int::<i8, _>(&mut eq, &input).unwrap();
    assert_eq!(hash_bits.len(), 256);
    drop(eq);
    assert!(cs.num_constraints() > 0);
  }

  #[test]
  fn test_small_sha256_correctness_empty() {
    use sha2::{Digest, Sha256};

    let mut cs = SmallShapeCS::<i32>::new();
    let mut eq = NoBatchEq::<i8, i32, _>::new(&mut cs);

    let input: Vec<SmallBoolean> = vec![];
    let hash_bits = small_sha256_int::<i8, _>(&mut eq, &input).unwrap();
    let hash_bytes = small_bits_to_bytes(&hash_bits);
    let expected = Sha256::digest(b"");

    assert_eq!(&hash_bytes[..], &expected[..]);
  }

  #[test]
  fn test_small_sha256_correctness_abc() {
    use sha2::{Digest, Sha256};

    let mut cs = SmallShapeCS::<i32>::new();
    let mut eq = NoBatchEq::<i8, i32, _>::new(&mut cs);

    let input = bytes_to_small_bits(b"abc");
    let hash_bits = small_sha256_int::<i8, _>(&mut eq, &input).unwrap();
    let hash_bytes = small_bits_to_bytes(&hash_bits);
    let expected = Sha256::digest(b"abc");

    assert_eq!(&hash_bytes[..], &expected[..]);
  }
}
