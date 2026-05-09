// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 compression built from bounded-coefficient gadgets.

use super::{BatchingEq, SmallMultiEq, SmallUInt32};
use bellpepper_core::{ConstraintSystem, SynthesisError, boolean::Boolean};
use ff::PrimeField;

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
  let mut w_expanded = w.to_vec();
  w_expanded.reserve(48);

  for i in 16..64 {
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

/// Run one SHA-256 compression on a 512-bit block and an 8-word state.
///
/// The input bits are interpreted as 16 big-endian `u32` message words. The
/// returned state is not exposed as public input by this helper; callers decide
/// whether to use or discard it.
pub fn small_sha256_compression_function<Scalar, CS>(
  mut cs: CS,
  input_bits: &[Boolean],
  current_hash: &[SmallUInt32],
) -> Result<Vec<SmallUInt32>, SynthesisError>
where
  Scalar: PrimeField,
  CS: ConstraintSystem<Scalar>,
{
  assert_eq!(input_bits.len(), 512);
  assert_eq!(current_hash.len(), 8);

  let mut h: [SmallUInt32; 8] = current_hash.to_vec().try_into().unwrap();
  let mut w = std::array::from_fn(|_| SmallUInt32::constant(0));
  for (i, w_item) in w.iter_mut().enumerate() {
    let word_bits: [Boolean; 32] = input_bits[i * 32..(i + 1) * 32]
      .to_vec()
      .try_into()
      .unwrap();
    *w_item = SmallUInt32::from_bits_be(&word_bits);
  }

  let mut eq = BatchingEq::<Scalar, CS, 21>::new(&mut cs);
  let result = sha256_compression(&mut eq, &mut h, &w, 0, "");
  drop(eq);
  result?;

  Ok(h.to_vec())
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::test_cs::TestConstraintSystem;
  use halo2curves::pasta::Fq;

  const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ];

  fn native_compress(mut h: [u32; 8], block: [u8; 64]) -> [u32; 8] {
    let mut w = [0u32; 64];
    for i in 0..16 {
      w[i] = u32::from_be_bytes([
        block[4 * i],
        block[4 * i + 1],
        block[4 * i + 2],
        block[4 * i + 3],
      ]);
    }
    for i in 16..64 {
      let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
      let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16]
        .wrapping_add(s0)
        .wrapping_add(w[i - 7])
        .wrapping_add(s1);
    }

    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
      (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    for i in 0..64 {
      let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
      let ch = (e & f) ^ ((!e) & g);
      let temp1 = hh
        .wrapping_add(s1)
        .wrapping_add(ch)
        .wrapping_add(ROUND_CONSTANTS[i])
        .wrapping_add(w[i]);
      let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
      let maj = (a & b) ^ (a & c) ^ (b & c);
      let temp2 = s0.wrapping_add(maj);
      hh = g;
      g = f;
      f = e;
      e = d.wrapping_add(temp1);
      d = c;
      c = b;
      b = a;
      a = temp1.wrapping_add(temp2);
    }

    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
    h
  }

  #[test]
  fn test_small_sha256_compression_matches_native() {
    let mut cs = TestConstraintSystem::<Fq>::new();
    let mut block = [0u8; 64];
    for (i, byte) in block.iter_mut().enumerate() {
      *byte = (i as u8).wrapping_mul(17).wrapping_add(3);
    }

    let input_bits = block
      .iter()
      .flat_map(|byte| {
        (0..8)
          .rev()
          .map(move |i| Boolean::constant((byte >> i) & 1 == 1))
      })
      .collect::<Vec<_>>();
    let current_hash = IV.map(SmallUInt32::constant);
    let got =
      small_sha256_compression_function::<Fq, _>(&mut cs, &input_bits, &current_hash).unwrap();
    let got = got
      .iter()
      .map(|word| word.get_value().unwrap())
      .collect::<Vec<_>>();

    assert_eq!(got, native_compress(IV, block));
    assert!(cs.is_satisfied());
  }
}
