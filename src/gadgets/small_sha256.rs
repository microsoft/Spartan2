// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 circuit using small-value compatible gadgets.
//!
//! This module provides a SHA-256 implementation that is compatible with the
//! small-value sumcheck optimization. Unlike bellpepper's SHA-256 which uses
//! `MultiEq` and can create coefficients up to 2^237, this implementation uses
//! `SmallMultiEq` which either enforces directly or batches with bounded coefficients.
//!
//! # Usage
//!
//! ```ignore
//! use spartan2::gadgets::small_sha256;
//!
//! // Simple API (recommended) - uses BatchingEq<21> for optimal performance
//! let hash_bits = small_sha256(&mut cs, &input_bits)?;
//!
//! // With prefix for hash chains
//! let hash_bits = small_sha256_with_prefix(&mut cs, &input_bits, "block0_")?;
//!
//! // Advanced API - bring your own SmallMultiEq
//! let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
//! let hash_bits = small_sha256_with_small_multi_eq(&mut eq, &input_bits, "")?;
//! ```

use super::{BatchingEq, SmallMultiEq, SmallUInt32};
use bellpepper_core::{ConstraintSystem, SynthesisError, boolean::Boolean};
use ff::PrimeField;

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

/// Σ0(x) = ROTR^2(x) ⊕ ROTR^13(x) ⊕ ROTR^22(x)
fn big_sigma_0<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  x: &SmallUInt32,
) -> Result<SmallUInt32, SynthesisError> {
  let r2 = x.rotr(2);
  let r13 = x.rotr(13);
  let r22 = x.rotr(22);
  let tmp = r2.xor(cs.namespace(|| "sigma0_r2_xor_r13"), &r13)?;
  tmp.xor(cs.namespace(|| "sigma0_xor_r22"), &r22)
}

/// Σ1(x) = ROTR^6(x) ⊕ ROTR^11(x) ⊕ ROTR^25(x)
fn big_sigma_1<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  x: &SmallUInt32,
) -> Result<SmallUInt32, SynthesisError> {
  let r6 = x.rotr(6);
  let r11 = x.rotr(11);
  let r25 = x.rotr(25);
  let tmp = r6.xor(cs.namespace(|| "sigma1_r6_xor_r11"), &r11)?;
  tmp.xor(cs.namespace(|| "sigma1_xor_r25"), &r25)
}

/// σ0(x) = ROTR^7(x) ⊕ ROTR^18(x) ⊕ SHR^3(x)
fn small_sigma_0<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  x: &SmallUInt32,
) -> Result<SmallUInt32, SynthesisError> {
  let r7 = x.rotr(7);
  let r18 = x.rotr(18);
  let s3 = x.shr(3);
  let tmp = r7.xor(cs.namespace(|| "s0_r7_xor_r18"), &r18)?;
  tmp.xor(cs.namespace(|| "s0_xor_s3"), &s3)
}

/// σ1(x) = ROTR^17(x) ⊕ ROTR^19(x) ⊕ SHR^10(x)
fn small_sigma_1<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  x: &SmallUInt32,
) -> Result<SmallUInt32, SynthesisError> {
  let r17 = x.rotr(17);
  let r19 = x.rotr(19);
  let s10 = x.shr(10);
  let tmp = r17.xor(cs.namespace(|| "s1_r17_xor_r19"), &r19)?;
  tmp.xor(cs.namespace(|| "s1_xor_s10"), &s10)
}

/// SHA-256 compression function.
///
/// Takes the current hash state H and a 512-bit message block W,
/// returns the updated hash state.
///
/// The `prefix` is prepended to all variable names to allow multiple SHA-256
/// calls in the same constraint system (e.g., for hash chains).
fn sha256_compression<Scalar, M>(
  cs: &mut M,
  h: &mut [SmallUInt32; 8],
  w: &[SmallUInt32; 16],
  block_idx: usize,
  prefix: &str,
) -> Result<(), SynthesisError>
where
  Scalar: PrimeField,
  M: SmallMultiEq<Scalar>,
{
  // Message schedule: expand 16 words to 64 words
  let mut w_expanded: Vec<SmallUInt32> = w.to_vec();
  w_expanded.reserve(48);

  for i in 16..64 {
    // W[i] = σ1(W[i-2]) + W[i-7] + σ0(W[i-15]) + W[i-16]
    let s1 = small_sigma_1(
      cs.namespace(|| format!("{}b{}_w{}_s1", prefix, block_idx, i)),
      &w_expanded[i - 2],
    )?;
    let s0 = small_sigma_0(
      cs.namespace(|| format!("{}b{}_w{}_s0", prefix, block_idx, i)),
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

  // Initialize working variables
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

  // 64 rounds
  for i in 0..64 {
    // T1 = h + Σ1(e) + Ch(e,f,g) + K[i] + W[i]
    let sigma1 = big_sigma_1(
      cs.namespace(|| format!("{}b{}_r{}_sigma1", prefix, block_idx, i)),
      &e,
    )?;
    let ch = SmallUInt32::sha256_ch(
      cs.namespace(|| format!("{}b{}_r{}_ch", prefix, block_idx, i)),
      &e,
      &f,
      &g,
    )?;
    let k = SmallUInt32::constant(ROUND_CONSTANTS[i]);

    let t1 = cs.addmany(&[h_var.clone(), sigma1, ch, k, w_expanded[i].clone()])?;

    // T2 components: Σ0(a) and Maj(a,b,c)
    // Instead of computing T2 = sigma0 + maj separately, we fuse it into 'a' below.
    let sigma0 = big_sigma_0(
      cs.namespace(|| format!("{}b{}_r{}_sigma0", prefix, block_idx, i)),
      &a,
    )?;
    let maj = SmallUInt32::sha256_maj(
      cs.namespace(|| format!("{}b{}_r{}_maj", prefix, block_idx, i)),
      &a,
      &b,
      &c,
    )?;

    // Update working variables
    h_var = g;
    g = f;
    f = e;
    e = cs.addmany(&[d, t1.clone()])?;
    d = c;
    c = b;
    b = a;
    // Fused: a = T1 + T2 = T1 + sigma0 + maj (saves one addmany call per round)
    a = cs.addmany(&[t1, sigma0, maj])?;
  }

  // Compute final hash values
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

/// Compute SHA-256 hash of input bits.
///
/// Uses `BatchingEq<21>` internally for optimal performance with i64 small values.
/// This batches up to 21 equality constraints and uses full 35-bit addition.
///
/// Returns 256 bits of the hash in big-endian order.
///
/// # Example
///
/// ```ignore
/// let mut cs = TestConstraintSystem::<Fq>::new();
/// let hash_bits = small_sha256::<Fq, _>(&mut cs, &input_bits)?;
/// ```
pub fn small_sha256<Scalar, CS>(
  cs: &mut CS,
  input: &[Boolean],
) -> Result<Vec<Boolean>, SynthesisError>
where
  Scalar: PrimeField,
  CS: ConstraintSystem<Scalar>,
{
  small_sha256_with_prefix::<Scalar, CS>(cs, input, "")
}

/// Compute SHA-256 hash with a prefix for variable names.
///
/// Uses `BatchingEq<21>` internally for optimal performance.
/// This variant allows multiple SHA-256 computations in the same constraint
/// system (e.g., for hash chains) by prefixing all internal variable names.
///
/// # Example
///
/// ```ignore
/// // Hash chain: H(H(H(x)))
/// let h1 = small_sha256_with_prefix(&mut cs, &input, "hash1_")?;
/// let h2 = small_sha256_with_prefix(&mut cs, &h1, "hash2_")?;
/// let h3 = small_sha256_with_prefix(&mut cs, &h2, "hash3_")?;
/// ```
pub fn small_sha256_with_prefix<Scalar, CS>(
  cs: &mut CS,
  input: &[Boolean],
  prefix: &str,
) -> Result<Vec<Boolean>, SynthesisError>
where
  Scalar: PrimeField,
  CS: ConstraintSystem<Scalar>,
{
  // Push namespace to scope constraints under the prefix
  cs.push_namespace(|| format!("{}sha256", prefix));

  // Create BatchingEq<21> for optimal performance
  let mut eq = BatchingEq::<Scalar, CS, 21>::new(cs);
  let result = small_sha256_with_small_multi_eq(&mut eq, input, prefix);
  drop(eq); // Flush any pending constraints

  // Pop the namespace before returning
  cs.pop_namespace();

  result
}

/// Compute SHA-256 hash using a custom `SmallMultiEq` implementation.
///
/// This is the advanced API that gives full control over the batching strategy.
/// Use this when you need:
/// - `NoBatchEq` for i32 small value compatibility
/// - Custom `BatchingEq<K>` with a different batch size
///
/// # Example
///
/// ```ignore
/// // Using NoBatchEq for i32 compatibility
/// let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
/// let hash = small_sha256_with_small_multi_eq(&mut eq, &input, "")?;
///
/// // Using custom BatchingEq<10>
/// let mut eq = BatchingEq::<Fq, _, 10>::new(&mut cs);
/// let hash = small_sha256_with_small_multi_eq(&mut eq, &input, "")?;
/// ```
pub fn small_sha256_with_small_multi_eq<Scalar, M>(
  eq: &mut M,
  input: &[Boolean],
  prefix: &str,
) -> Result<Vec<Boolean>, SynthesisError>
where
  Scalar: PrimeField,
  M: SmallMultiEq<Scalar>,
{
  // Pad the input according to SHA-256 spec
  let padded = sha256_padding(input);

  // Process in 512-bit blocks
  assert!(padded.len().is_multiple_of(512));
  let num_blocks = padded.len() / 512;

  // Initialize hash state
  let mut h: [SmallUInt32; 8] = IV.map(SmallUInt32::constant);

  for block_idx in 0..num_blocks {
    let block_start = block_idx * 512;
    let block_bits = &padded[block_start..block_start + 512];

    // Convert 512 bits to 16 32-bit words (big-endian)
    let mut w: [SmallUInt32; 16] = std::array::from_fn(|_| SmallUInt32::constant(0));
    for (i, w_item) in w.iter_mut().enumerate() {
      let word_bits: [Boolean; 32] = block_bits[i * 32..(i + 1) * 32]
        .to_vec()
        .try_into()
        .unwrap();
      *w_item = SmallUInt32::from_bits_be(&word_bits);
    }

    // Run compression
    sha256_compression(eq, &mut h, &w, block_idx, prefix)?;
  }

  // Collect output bits in big-endian order
  let mut output = Vec::with_capacity(256);
  for h_i in h {
    output.extend(h_i.into_bits_be());
  }

  Ok(output)
}

/// SHA-256 padding: append 1 bit, zeros, and 64-bit length.
fn sha256_padding(input: &[Boolean]) -> Vec<Boolean> {
  let msg_len = input.len();

  // Calculate padded length: message + 1 + zeros + 64-bit length
  // Must be multiple of 512
  let mut padded_len = msg_len + 1 + 64; // message + '1' bit + length
  if !padded_len.is_multiple_of(512) {
    padded_len += 512 - (padded_len % 512);
  }

  let mut padded = Vec::with_capacity(padded_len);

  // Copy message bits
  padded.extend_from_slice(input);

  // Append '1' bit
  padded.push(Boolean::constant(true));

  // Append zeros
  let zero_count = padded_len - msg_len - 1 - 64;
  for _ in 0..zero_count {
    padded.push(Boolean::constant(false));
  }

  // Append 64-bit length (big-endian)
  let len_bits: u64 = msg_len as u64;
  for i in (0..64).rev() {
    padded.push(Boolean::constant((len_bits >> i) & 1 == 1));
  }

  assert_eq!(padded.len(), padded_len);
  assert!(padded.len() % 512 == 0);

  padded
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::gadgets::NoBatchEq;
  use bellpepper_core::test_cs::TestConstraintSystem;
  use halo2curves::pasta::Fq;
  use rand::{Rng, SeedableRng, rngs::StdRng};
  use sha2::{Digest, Sha256};

  /// Convert bytes to Boolean bits (big-endian per byte).
  fn bytes_to_bits(bytes: &[u8]) -> Vec<Boolean> {
    bytes
      .iter()
      .flat_map(|byte| {
        (0..8)
          .rev()
          .map(move |i| Boolean::constant((byte >> i) & 1 == 1))
      })
      .collect()
  }

  /// Convert Boolean bits to bytes (big-endian per byte).
  fn bits_to_bytes(bits: &[Boolean]) -> Vec<u8> {
    assert!(bits.len().is_multiple_of(8));
    bits
      .chunks(8)
      .map(|chunk| {
        chunk.iter().fold(0u8, |acc, bit| {
          let b = match bit {
            Boolean::Constant(b) => *b,
            Boolean::Is(ab) => ab.get_value().unwrap(),
            Boolean::Not(ab) => !ab.get_value().unwrap(),
          };
          (acc << 1) | (b as u8)
        })
      })
      .collect()
  }

  #[test]
  fn test_small_sha256_empty() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    let input: Vec<Boolean> = vec![];
    let hash_bits = small_sha256::<Fq, _>(&mut cs, &input).unwrap();

    let hash_bytes = bits_to_bytes(&hash_bits);
    let expected = Sha256::digest(b"");

    assert_eq!(&hash_bytes[..], &expected[..]);
    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_small_sha256_abc() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // SHA256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    let input = bytes_to_bits(b"abc");
    let hash_bits = small_sha256::<Fq, _>(&mut cs, &input).unwrap();

    let hash_bytes = bits_to_bytes(&hash_bits);
    let expected = Sha256::digest(b"abc");

    assert_eq!(&hash_bytes[..], &expected[..]);
    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_small_sha256_matches_native_32_times() {
    // Use seeded RNG for reproducibility
    let mut rng = StdRng::seed_from_u64(12345);

    for i in 0..32 {
      let mut cs = TestConstraintSystem::<Fq>::new();

      // Random preimage length: 1 to 128 bytes
      let len = rng.gen_range(1..=128);
      let preimage: Vec<u8> = (0..len).map(|_| rng.r#gen()).collect();

      // Native SHA-256
      let expected = Sha256::digest(&preimage);

      // Circuit SHA-256 (now uses BatchingEq<21> by default)
      let input_bits = bytes_to_bits(&preimage);
      let hash_bits = small_sha256::<Fq, _>(&mut cs, &input_bits).unwrap();
      let hash_bytes = bits_to_bytes(&hash_bits);

      assert_eq!(
        &hash_bytes[..],
        &expected[..],
        "Mismatch at iteration {}, preimage len {}",
        i,
        len
      );
      assert!(cs.is_satisfied(), "CS not satisfied at iteration {}", i);
    }
  }

  #[test]
  fn test_small_sha256_with_small_multi_eq_nobatch() {
    // Test the advanced API with NoBatchEq
    let mut rng = StdRng::seed_from_u64(54321);

    for i in 0..8 {
      let mut cs = TestConstraintSystem::<Fq>::new();

      let len = rng.gen_range(1..=64);
      let preimage: Vec<u8> = (0..len).map(|_| rng.r#gen()).collect();

      let expected = Sha256::digest(&preimage);

      let input_bits = bytes_to_bits(&preimage);

      // Use NoBatchEq via the advanced API
      let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
      let hash_bits = small_sha256_with_small_multi_eq(&mut eq, &input_bits, "").unwrap();
      #[allow(clippy::drop_non_drop)]
      // Intentional: signals "done with eq" for consistency with BatchingEq
      drop(eq);

      let hash_bytes = bits_to_bytes(&hash_bits);

      assert_eq!(
        &hash_bytes[..],
        &expected[..],
        "Mismatch at iteration {}, preimage len {}",
        i,
        len
      );
      assert!(cs.is_satisfied(), "CS not satisfied at iteration {}", i);
    }
  }
}
