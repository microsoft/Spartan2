// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Barrett reduction for small × field multiplication.

use super::limbs::{gte_4_4, gte_5_4, mul_4_by_1, mul_5_by_1, sub_4_4, sub_5_4, sub_5_5};
use halo2curves::{
  bn256::Fr as Bn254Fr,
  pasta::{Fp, Fq},
};
use std::ops::Neg;

// ==========================================================================
// FieldReductionConstants - Trait for field-specific reduction constants
// ==========================================================================

/// Trait providing precomputed constants for efficient modular reduction.
///
/// # Overview
///
/// When reducing a wide integer (more than 4 limbs = 256 bits) modulo a prime p,
/// we need to handle "overflow limbs" that represent values ≥ 2^256. Each overflow
/// limb at position i represents the value `limb[i] × 2^(64×i)`.
///
/// # The R Constants
///
/// For each bit position beyond 256 bits, we precompute `2^k mod p`:
///
/// | Constant | Value | Used When |
/// |----------|-------|-----------|
/// | `R256_MOD` | 2^256 mod p | Reducing 5th limb (bits 256-319) |
/// | `R320_MOD` | 2^320 mod p | Reducing 6th limb (bits 320-383) |
/// | `R384_MOD` | 2^384 mod p | Reducing 7th limb (bits 384-447) |
/// | `R448_MOD` | 2^448 mod p | Reducing 8th limb (bits 448-511) |
/// | `R512_MOD` | 2^512 mod p | Reducing 9th limb (bits 512-575) |
///
/// # Example: 6-limb Reduction
///
/// For a 6-limb value `c = [c0, c1, c2, c3, c4, c5]` representing:
/// ```text
/// c = c0 + c1·2^64 + c2·2^128 + c3·2^192 + c4·2^256 + c5·2^320
/// ```
///
/// We reduce by computing:
/// ```text
/// c mod p = (c0 + c1·2^64 + c2·2^128 + c3·2^192)
///         + c4·(2^256 mod p)
///         + c5·(2^320 mod p)
/// ```
///
/// Since `R256_MOD` and `R320_MOD` are 4-limb values (< 2^256), multiplying
/// by a single limb produces at most a 5-limb result, which can then be
/// reduced further if needed.
///
/// # Why This Works
///
/// By the properties of modular arithmetic:
/// `a ≡ b (mod p) ⟹ c·a ≡ c·b (mod p)`
///
/// So `c5·2^320 ≡ c5·R320_MOD (mod p)`, and the right side is much smaller.
///
/// # Performance
///
/// Much faster than naive division: avoids division entirely, uses only
/// 4-5 64-bit multiplications with precomputed constants.
pub trait FieldReductionConstants {
  /// The 4-limb prime modulus p (little-endian, 256 bits)
  const MODULUS: [u64; 4];

  /// 2×p as a 5-limb value (for Barrett reduction comparisons)
  const MODULUS_2P: [u64; 5];

  /// Barrett approximation constant μ = floor(2^128 / (p >> 191))
  /// Used to estimate the quotient in Barrett reduction
  const MU: u64;

  /// 2^256 mod p - reduces the 5th limb (index 4) of a wide integer
  const R256_MOD: [u64; 4];

  /// 2^320 mod p - reduces the 6th limb (index 5) of a wide integer
  const R320_MOD: [u64; 4];

  /// 2^384 mod p - reduces the 7th limb (index 6) of a wide integer
  const R384_MOD: [u64; 4];

  /// 2^448 mod p - reduces the 8th limb (index 7) of a wide integer
  const R448_MOD: [u64; 4];

  /// 2^512 mod p - reduces the 9th limb (index 8) of a wide integer
  const R512_MOD: [u64; 4];

  /// Montgomery inverse: -p^(-1) mod 2^64
  /// Used in Montgomery REDC to eliminate low limbs
  const MONT_INV: u64;
}

// ==========================================================================
// FieldReductionConstants implementation for Fp (Pallas base field)
// ==========================================================================

impl FieldReductionConstants for Fp {
  // p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
  const MODULUS: [u64; 4] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
  ];

  const MODULUS_2P: [u64; 5] = double_limbs(Self::MODULUS);

  const MU: u64 = 0xffffffffffffffff;

  // 2^256 mod p = 0x3fffffffffffffff992c350be41914ad34786d38fffffffd
  const R256_MOD: [u64; 4] = [
    0x34786d38fffffffd,
    0x992c350be41914ad,
    0xffffffffffffffff,
    0x3fffffffffffffff,
  ];

  // 2^320 mod p = 0x3fffffffffffffff76e59c0fdacc1b91bd91d548094cf917992d30ed00000001
  const R320_MOD: [u64; 4] = [
    0x992d30ed00000001,
    0xbd91d548094cf917,
    0x76e59c0fdacc1b91,
    0x3fffffffffffffff,
  ];

  // 2^384 mod p
  const R384_MOD: [u64; 4] = [
    0xcb8792c700000003,
    0x66d3caf41be6eb52,
    0x9b4b3c4bfffffffc,
    0x36e59c0fdacc1b91,
  ];

  // 2^448 mod p
  const R448_MOD: [u64; 4] = [
    0x9b9858f294cf91ba,
    0x8635bd2c4252b065,
    0x496d41af7b9cb714,
    0x1b4b3c4bfffffffc,
  ];

  // 2^512 mod p
  const R512_MOD: [u64; 4] = [
    0x8c78ecb30000000f,
    0xd7d30dbd8b0de0e7,
    0x7797a99bc3c95d18,
    0x096d41af7b9cb714,
  ];

  // -p^(-1) mod 2^64
  const MONT_INV: u64 = 0x992d30ecffffffff;
}

// ==========================================================================
// FieldReductionConstants implementation for Fq (Pallas scalar field)
// ==========================================================================

impl FieldReductionConstants for Fq {
  // q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
  const MODULUS: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
  ];

  const MODULUS_2P: [u64; 5] = double_limbs(Self::MODULUS);

  const MU: u64 = 0xffffffffffffffff;

  // 2^256 mod q
  const R256_MOD: [u64; 4] = [
    0x5b2b3e9cfffffffd,
    0x992c350be3420567,
    0xffffffffffffffff,
    0x3fffffffffffffff,
  ];

  // 2^320 mod q
  const R320_MOD: [u64; 4] = [
    0x8c46eb2100000001,
    0xf12aec780994a8d9,
    0x76e59c0fd9ad5c89,
    0x3fffffffffffffff,
  ];

  // 2^384 mod q
  const R384_MOD: [u64; 4] = [
    0xa4d4c16300000003,
    0x66d3caf41cbdfa98,
    0xcee4537bfffffffc,
    0x36e59c0fd9ad5c89,
  ];

  // 2^448 mod q
  const R448_MOD: [u64; 4] = [
    0xcc920bb9994a8dd9,
    0x87a7dcbe1ff6e0d7,
    0x496d41af7ccfdaa9,
    0x0ee4537bfffffffc,
  ];

  // 2^512 mod q
  const R512_MOD: [u64; 4] = [
    0xfc9678ff0000000f,
    0x67bb433d891a16e3,
    0x7fae231004ccf590,
    0x096d41af7ccfdaa9,
  ];

  // -q^(-1) mod 2^64
  const MONT_INV: u64 = 0x8c46eb20ffffffff;
}

// ==========================================================================
// FieldReductionConstants implementation for Bn254Fr (BN254 scalar field)
// ==========================================================================

impl FieldReductionConstants for Bn254Fr {
  // r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
  const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
  ];

  const MODULUS_2P: [u64; 5] = double_limbs(Self::MODULUS);

  const MU: u64 = 0xffffffffffffffff;

  // 2^256 mod r
  const R256_MOD: [u64; 4] = [
    0xac96341c4ffffffb,
    0x36fc76959f60cd29,
    0x666ea36f7879462e,
    0x0e0a77c19a07df2f,
  ];

  // 2^320 mod r
  const R320_MOD: [u64; 4] = [
    0xb4c6edf97c5fb586,
    0x708c8d50bfeb93be,
    0x9ffd1de404f7e0ef,
    0x215b02ac9a392866,
  ];

  // 2^384 mod r
  const R384_MOD: [u64; 4] = [
    0xb075da81ef8cfeb9,
    0xa7f12acca5b6cd8c,
    0x32c475047957bf7b,
    0x03d581d748ffa25e,
  ];

  // 2^448 mod r
  const R448_MOD: [u64; 4] = [
    0x5665c3b5c177f51a,
    0x00e7f02ade75c713,
    0xb09192e52f747168,
    0x0621c0bbcccdc65d,
  ];

  // 2^512 mod r
  const R512_MOD: [u64; 4] = [
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
  ];

  // -r^(-1) mod 2^64
  const MONT_INV: u64 = 0xc2e1f593efffffff;
}

// ==========================================================================
// BarrettField - Trait for generic small × field multiplication
// ==========================================================================

/// Trait for field types supporting optimized small × field multiplication.
///
/// Extends `FieldReductionConstants` with methods for accessing the internal
/// limb representation and constructing field elements from limbs.
pub(crate) trait BarrettField: FieldReductionConstants + Neg<Output = Self> + Copy {
  /// 2^64 as a field element, used for u128 multiplication decomposition.
  const TWO_POW_64: Self;

  /// Construct a field element from 4 Montgomery-form limbs.
  fn from_limbs(limbs: [u64; 4]) -> Self;

  /// Access the internal Montgomery-form limbs.
  fn to_limbs(&self) -> &[u64; 4];
}

impl BarrettField for Fp {
  const TWO_POW_64: Self = Fp::from_raw([0, 1, 0, 0]);

  #[inline]
  fn from_limbs(limbs: [u64; 4]) -> Self {
    Fp(limbs)
  }

  #[inline]
  fn to_limbs(&self) -> &[u64; 4] {
    &self.0
  }
}

impl BarrettField for Fq {
  const TWO_POW_64: Self = Fq::from_raw([0, 1, 0, 0]);

  #[inline]
  fn from_limbs(limbs: [u64; 4]) -> Self {
    Fq(limbs)
  }

  #[inline]
  fn to_limbs(&self) -> &[u64; 4] {
    &self.0
  }
}

impl BarrettField for Bn254Fr {
  const TWO_POW_64: Self = Bn254Fr::from_raw([0, 1, 0, 0]);

  #[inline]
  fn from_limbs(limbs: [u64; 4]) -> Self {
    Bn254Fr(limbs)
  }

  #[inline]
  fn to_limbs(&self) -> &[u64; 4] {
    &self.0
  }
}

/// Computes 2*p, returning a 5-limb result.
const fn double_limbs(p: [u64; 4]) -> [u64; 5] {
  let (r0, c0) = p[0].overflowing_add(p[0]);
  let (r1, c1) = {
    let (sum, c1a) = p[1].overflowing_add(p[1]);
    let (sum, c1b) = sum.overflowing_add(c0 as u64);
    (sum, c1a || c1b)
  };
  let (r2, c2) = {
    let (sum, c2a) = p[2].overflowing_add(p[2]);
    let (sum, c2b) = sum.overflowing_add(c1 as u64);
    (sum, c2a || c2b)
  };
  let (r3, c3) = {
    let (sum, c3a) = p[3].overflowing_add(p[3]);
    let (sum, c3b) = sum.overflowing_add(c2 as u64);
    (sum, c3a || c3b)
  };
  let r4 = c3 as u64;
  [r0, r1, r2, r3, r4]
}

// ==========================================================================
// Generic small × field multiplication
// ==========================================================================

/// Multiply field element by i64 (signed).
#[inline]
pub(crate) fn mul_by_i64<F: BarrettField>(large: &F, small: i64) -> F {
  if small >= 0 {
    mul_by_u64(large, small as u64)
  } else {
    mul_by_u64(large, small.wrapping_neg() as u64).neg()
  }
}

/// Multiply field element by i128 (signed).
#[inline]
#[allow(dead_code)]
pub(crate) fn mul_by_i128<
  F: BarrettField + std::ops::Add<Output = F> + std::ops::Mul<Output = F>,
>(
  large: &F,
  small: i128,
) -> F {
  if small >= 0 {
    mul_by_u128(large, small as u128)
  } else {
    mul_by_u128(large, small.wrapping_neg() as u128).neg()
  }
}

/// Multiply field element by u128 (unsigned).
#[inline]
#[allow(dead_code)]
fn mul_by_u128<F: BarrettField + std::ops::Add<Output = F> + std::ops::Mul<Output = F>>(
  large: &F,
  small: u128,
) -> F {
  let low = small as u64;
  let high = (small >> 64) as u64;

  if high == 0 {
    mul_by_u64(large, low)
  } else {
    // result = large * low + large * high * 2^64
    let low_part = mul_by_u64(large, low);
    let high_part = mul_by_u64(large, high);
    low_part + high_part * F::TWO_POW_64
  }
}

/// Multiply field element by u64 (unsigned).
#[inline]
fn mul_by_u64<F: BarrettField>(large: &F, small: u64) -> F {
  if small == 0 {
    return F::from_limbs([0, 0, 0, 0]);
  }
  if small == 1 {
    return *large;
  }
  let c = mul_4_by_1(large.to_limbs(), small);
  F::from_limbs(barrett_reduce_5::<F>(&c))
}

/// Generic 5-limb Barrett reduction using trait constants.
/// Reduces a 5-limb value (up to 320 bits) modulo p.
#[inline(always)]
fn barrett_reduce_5<F: FieldReductionConstants>(c: &[u64; 5]) -> [u64; 4] {
  let c_tilde = (c[3] >> 63) | (c[4] << 1);
  let m = {
    let product = (c_tilde as u128) * (F::MU as u128);
    (product >> 64) as u64
  };
  let m_times_2p = mul_5_by_1(&F::MODULUS_2P, m);
  let mut r = sub_5_5(c, &m_times_2p);
  // At most 2 iterations needed: after Barrett approximation, 0 ≤ r < 2p
  while gte_5_4(&r, &F::MODULUS) {
    r = sub_5_4(&r, &F::MODULUS);
  }
  [r[0], r[1], r[2], r[3]]
}

// ==========================================================================
// 6-limb Barrett reduction (for UnreducedFieldInt accumulator)
// ==========================================================================

/// Generic 6-limb Barrett reduction using trait constants.
///
/// Reduces a 6-limb value (up to 384 bits) modulo p using precomputed
/// R256 = 2^256 mod p and R320 = 2^320 mod p constants.
///
/// Input is already in Montgomery form (R-scaled). This function reduces
/// the 6-limb value mod p while preserving the Montgomery scaling.
#[inline]
fn barrett_reduce_6<F: FieldReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
  // Reduce c[5] * 2^320 ≡ c[5] * R320 (mod p), then reduce c[4] * 2^256, etc.
  // R320 = 2^320 mod p, R256 = 2^256 mod p
  //
  // We do: result = c[0..4] + c[4] * R256 + c[5] * R320 (mod p)

  // c[4] * R256_MOD (4x1 -> 5 limbs)
  let c4_contrib = mul_4_by_1(&F::R256_MOD, c[4]);
  // c[5] * R320_MOD (4x1 -> 5 limbs)
  let c5_contrib = mul_4_by_1(&F::R320_MOD, c[5]);

  // Sum: c[0..4] + c4_contrib + c5_contrib (could be up to 6 limbs)
  let mut sum = [0u64; 6];
  let mut carry = 0u128;
  for i in 0..4 {
    let s = (c[i] as u128) + (c4_contrib[i] as u128) + (c5_contrib[i] as u128) + carry;
    sum[i] = s as u64;
    carry = s >> 64;
  }
  // Limb 4: c4_contrib[4] + c5_contrib[4] + carry
  let s = (c4_contrib[4] as u128) + (c5_contrib[4] as u128) + carry;
  sum[4] = s as u64;
  sum[5] = (s >> 64) as u64;

  // Now reduce the 6-limb sum. If sum[5] or sum[4] is non-zero, recurse (limited depth)
  if sum[5] == 0 && sum[4] == 0 {
    // Result fits in 4 limbs, just do final reduction
    let mut r = [sum[0], sum[1], sum[2], sum[3]];
    while gte_4_4(&r, &F::MODULUS) {
      r = sub_4_4(&r, &F::MODULUS);
    }
    return r;
  }

  // Recurse (this will terminate because sum < c in most cases)
  barrett_reduce_6::<F>(&sum)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn barrett_reduce_6_fp(c: &[u64; 6]) -> [u64; 4] {
  barrett_reduce_6::<Fp>(c)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn barrett_reduce_6_fq(c: &[u64; 6]) -> [u64; 4] {
  barrett_reduce_6::<Fq>(c)
}

// ==========================================================================
// 8-limb Barrett reduction (for i64/i128 UnreducedFieldInt accumulator)
// ==========================================================================

/// Generic 8-limb Barrett reduction using trait constants.
///
/// Reduces an 8-limb value (up to 512 bits) modulo p using precomputed
/// R384 = 2^384 mod p and R448 = 2^448 mod p constants, then delegates
/// to 6-limb reduction.
#[inline]
fn barrett_reduce_8<F: FieldReductionConstants>(c: &[u64; 8]) -> [u64; 4] {
  // Reduce high limbs: c[6] * 2^384 + c[7] * 2^448
  let c6_contrib = mul_4_by_1(&F::R384_MOD, c[6]);
  let c7_contrib = mul_4_by_1(&F::R448_MOD, c[7]);

  // Sum: c[0..6] + c6_contrib + c7_contrib
  let mut sum = [0u64; 6];
  let mut carry = 0u128;
  for i in 0..4 {
    let s = (c[i] as u128) + (c6_contrib[i] as u128) + (c7_contrib[i] as u128) + carry;
    sum[i] = s as u64;
    carry = s >> 64;
  }
  // Limbs 4-5: add c[4], c[5], and carry from high contributions
  let s = (c[4] as u128) + (c6_contrib[4] as u128) + (c7_contrib[4] as u128) + carry;
  sum[4] = s as u64;
  carry = s >> 64;
  let s = (c[5] as u128) + carry;
  sum[5] = s as u64;

  // Now reduce the 6-limb result
  barrett_reduce_6::<F>(&sum)
}

#[inline]
pub(crate) fn barrett_reduce_8_fp(c: &[u64; 8]) -> [u64; 4] {
  barrett_reduce_8::<Fp>(c)
}

#[inline]
pub(crate) fn barrett_reduce_8_fq(c: &[u64; 8]) -> [u64; 4] {
  barrett_reduce_8::<Fq>(c)
}

// ==========================================================================
// 9-limb Montgomery REDC (for UnreducedFieldField accumulator)
// ==========================================================================

/// Generic Montgomery REDC for 9-limb input using trait constants.
///
/// Reduces a 2R-scaled value (sum of field×field products) to 1R-scaled.
/// Input: T representing x·R² (up to 9 limbs)
/// Output: x·R mod p (4 limbs, standard Montgomery form)
#[inline]
fn montgomery_reduce_9<F: FieldReductionConstants>(c: &[u64; 9]) -> [u64; 4] {
  // Step 1: Reduce 9 limbs to 8 limbs using precomputed 2^512 mod p
  let mut t = [0u64; 9];
  if c[8] == 0 {
    t[..8].copy_from_slice(&c[..8]);
  } else {
    // t = c[0..8] + c[8] * R512_MOD
    let high_contribution = mul_4_by_1(&F::R512_MOD, c[8]);
    let mut carry = 0u128;
    for i in 0..5 {
      let sum = (c[i] as u128) + (high_contribution[i] as u128) + carry;
      t[i] = sum as u64;
      carry = sum >> 64;
    }
    for i in 5..8 {
      let sum = (c[i] as u128) + carry;
      t[i] = sum as u64;
      carry = sum >> 64;
    }
    t[8] = carry as u64;

    // Recurse if still > 8 limbs
    if t[8] > 0 {
      return montgomery_reduce_9::<F>(&t);
    }
  }

  // Step 2: Montgomery REDC on 8-limb value
  montgomery_reduce_8::<F>(&[t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]])
}

/// Generic Montgomery REDC for 8-limb input.
/// Standard Montgomery reduction: T × R⁻¹ mod p
#[inline]
fn montgomery_reduce_8<F: FieldReductionConstants>(t: &[u64; 8]) -> [u64; 4] {
  // Use 9 limbs to track overflow
  let mut r = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], 0u64];

  // Montgomery reduction: for each of the low 4 limbs, eliminate it
  // by adding appropriate multiples of p
  for i in 0..4 {
    // q = r[i] * p' mod 2^64
    let q = r[i].wrapping_mul(F::MONT_INV);

    // r += q * p * 2^(64*i)
    // qp = q * p, which is 5 limbs (since p is 4 limbs and q is 1 limb)
    let qp = mul_4_by_1(&F::MODULUS, q);

    let mut carry = 0u128;
    for j in 0..5 {
      let sum = (r[i + j] as u128) + (qp[j] as u128) + carry;
      r[i + j] = sum as u64;
      carry = sum >> 64;
    }
    // Propagate remaining carry through the rest of the array
    for item in r[(i + 5)..9].iter_mut() {
      let sum = (*item as u128) + carry;
      *item = sum as u64;
      carry = sum >> 64;
      if carry == 0 {
        break;
      }
    }
  }

  // Now r[0..4] should be zero (by construction), result is in r[4..9]
  // We need to reduce this to [0, p)
  let mut result = [r[4], r[5], r[6], r[7], r[8]];

  // Reduce until result < p
  while result[4] > 0 || gte_4_4(&[result[0], result[1], result[2], result[3]], &F::MODULUS) {
    let sub = sub_5_4(&result, &F::MODULUS);
    result = sub;
  }

  [result[0], result[1], result[2], result[3]]
}

#[inline]
#[allow(dead_code)]
pub(crate) fn montgomery_reduce_9_fp(c: &[u64; 9]) -> [u64; 4] {
  montgomery_reduce_9::<Fp>(c)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn montgomery_reduce_9_fq(c: &[u64; 9]) -> [u64; 4] {
  montgomery_reduce_9::<Fq>(c)
}

// ==========================================================================
// BN254 Fr reduction functions
// ==========================================================================

#[inline]
#[allow(dead_code)]
pub(crate) fn barrett_reduce_6_bn254_fr(c: &[u64; 6]) -> [u64; 4] {
  barrett_reduce_6::<Bn254Fr>(c)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn barrett_reduce_8_bn254_fr(c: &[u64; 8]) -> [u64; 4] {
  barrett_reduce_8::<Bn254Fr>(c)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn montgomery_reduce_9_bn254_fr(c: &[u64; 9]) -> [u64; 4] {
  montgomery_reduce_9::<Bn254Fr>(c)
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use rand_core::{OsRng, RngCore};

  #[test]
  fn test_barrett_fp_matches_naive() {
    for small in [0u64, 1, 2, 42, 1000, u32::MAX as u64, u64::MAX] {
      let large = Fp::random(&mut OsRng);
      let naive = Fp::from(small) * large;
      let barrett = mul_by_u64(&large, small);
      assert_eq!(naive, barrett, "Fp mismatch for small = {}", small);
    }
  }

  #[test]
  fn test_barrett_fp_random() {
    let mut rng = OsRng;
    for _ in 0..1000 {
      let large = Fp::random(&mut rng);
      let small: u64 = rng.next_u64();
      assert_eq!(Fp::from(small) * large, mul_by_u64(&large, small));
    }
  }

  #[test]
  fn test_barrett_fp_i64() {
    let large = Fp::from(42u64);
    assert_eq!(mul_by_i64(&large, 100i64), Fp::from(100u64) * large);
    assert_eq!(mul_by_i64(&large, -100i64), -Fp::from(100u64) * large);
  }

  #[test]
  fn test_barrett_fq_matches_naive() {
    for small in [0u64, 1, 2, 42, 1000, u32::MAX as u64, u64::MAX] {
      let large = Fq::random(&mut OsRng);
      let naive = Fq::from(small) * large;
      let barrett = mul_by_u64(&large, small);
      assert_eq!(naive, barrett, "Fq mismatch for small = {}", small);
    }
  }

  #[test]
  fn test_barrett_fq_random() {
    let mut rng = OsRng;
    for _ in 0..1000 {
      let large = Fq::random(&mut rng);
      let small: u64 = rng.next_u64();
      assert_eq!(Fq::from(small) * large, mul_by_u64(&large, small));
    }
  }

  #[test]
  fn test_barrett_fq_i64() {
    let large = Fq::from(42u64);
    assert_eq!(mul_by_i64(&large, 100i64), Fq::from(100u64) * large);
    assert_eq!(mul_by_i64(&large, -100i64), -Fq::from(100u64) * large);
  }

  #[test]
  fn test_barrett_fp_i128() {
    let large = Fp::random(&mut OsRng);

    // Test small i128 values (fits in i64)
    assert_eq!(mul_by_i128(&large, 100), Fp::from(100u64) * large);
    assert_eq!(mul_by_i128(&large, -100), -Fp::from(100u64) * large);

    // Test large i128 values (requires 2^64 decomposition)
    let big: i128 = (1i128 << 70) + 12345;
    let expected = crate::small_field::i128_to_field::<Fp>(big) * large;
    assert_eq!(mul_by_i128(&large, big), expected);

    let neg_big: i128 = -((1i128 << 70) + 12345);
    let expected_neg = crate::small_field::i128_to_field::<Fp>(neg_big) * large;
    assert_eq!(mul_by_i128(&large, neg_big), expected_neg);
  }

  #[test]
  fn test_barrett_fq_i128() {
    let large = Fq::random(&mut OsRng);

    // Test small i128 values (fits in i64)
    assert_eq!(mul_by_i128(&large, 100), Fq::from(100u64) * large);
    assert_eq!(mul_by_i128(&large, -100), -Fq::from(100u64) * large);

    // Test large i128 values (requires 2^64 decomposition)
    let big: i128 = (1i128 << 70) + 12345;
    let expected = crate::small_field::i128_to_field::<Fq>(big) * large;
    assert_eq!(mul_by_i128(&large, big), expected);

    let neg_big: i128 = -((1i128 << 70) + 12345);
    let expected_neg = crate::small_field::i128_to_field::<Fq>(neg_big) * large;
    assert_eq!(mul_by_i128(&large, neg_big), expected_neg);
  }

  #[test]
  fn test_barrett_i128_random() {
    let mut rng = OsRng;
    for _ in 0..100 {
      let large = Fq::random(&mut rng);
      // Generate random i128 in reasonable range
      let small: i128 = ((rng.next_u64() as i128) << 32) | (rng.next_u64() as i128);
      let small = if rng.next_u32().is_multiple_of(2) {
        small
      } else {
        -small
      };

      let result = mul_by_i128(&large, small);
      let expected = crate::small_field::i128_to_field::<Fq>(small) * large;
      assert_eq!(result, expected);
    }
  }

  #[test]
  fn test_two_pow_64_constants() {
    // Verify Fp::TWO_POW_64 is correct
    let computed = Fp::from(1u64 << 32) * Fp::from(1u64 << 32);
    assert_eq!(Fp::TWO_POW_64, computed);

    // Verify Fq::TWO_POW_64 is correct
    let computed = Fq::from(1u64 << 32) * Fq::from(1u64 << 32);
    assert_eq!(Fq::TWO_POW_64, computed);
  }

  #[test]
  fn test_constants_match_halo2curves() {
    let p_minus_one = -Fp::ONE;
    let expected = Fp::from_raw([
      Fp::MODULUS[0].wrapping_sub(1),
      Fp::MODULUS[1],
      Fp::MODULUS[2],
      Fp::MODULUS[3],
    ]);
    assert_eq!(p_minus_one, expected);

    let q_minus_one = -Fq::ONE;
    let expected = Fq::from_raw([
      Fq::MODULUS[0].wrapping_sub(1),
      Fq::MODULUS[1],
      Fq::MODULUS[2],
      Fq::MODULUS[3],
    ]);
    assert_eq!(q_minus_one, expected);
  }

  // ========================================================================
  // 6-limb Barrett reduction tests
  // ========================================================================

  #[test]
  fn test_barrett_6_fp_zero() {
    let c = [0u64; 6];
    let result = Fp(barrett_reduce_6_fp(&c));
    assert_eq!(result, Fp::ZERO);
  }

  #[test]
  fn test_barrett_6_fp_from_product() {
    // Test with an actual field × integer product (the real use case)
    let field_elem = Fp::from(12345u64); // Creates Montgomery form
    let small = 9999u64;

    // Compute field × small as 5 limbs (Montgomery form)
    let product = mul_4_by_1(&field_elem.0, small);

    // Extend to 6 limbs
    let c = [
      product[0], product[1], product[2], product[3], product[4], 0,
    ];

    // Reduce
    let result = Fp(barrett_reduce_6_fp(&c));

    // Expected: field_elem * small
    let expected = field_elem * Fp::from(small);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_barrett_6_fp_sum_of_products() {
    // Test summing multiple products (the accumulator use case)
    let mut rng = OsRng;
    let mut acc = [0u64; 6];
    let mut expected_sum = Fp::ZERO;

    // Sum 100 products
    for _ in 0..100 {
      let field_elem = Fp::random(&mut rng);
      let small = rng.next_u64() >> 32; // Keep small to avoid overflow in test

      // Accumulate expected result
      expected_sum += field_elem * Fp::from(small);

      // Compute field × small as 5 limbs
      let product = mul_4_by_1(&field_elem.0, small);

      // Add to accumulator with carry propagation
      let mut carry = 0u128;
      for i in 0..5 {
        let sum = (acc[i] as u128) + (product[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
      }
      acc[5] = acc[5].wrapping_add(carry as u64);
    }

    // Reduce and compare
    let result = Fp(barrett_reduce_6_fp(&acc));
    assert_eq!(result, expected_sum);
  }

  #[test]
  fn test_barrett_6_fp_many_products() {
    // Stress test: sum many products to exercise 6-limb reduction
    // Note: In the real use case, we sum at most 2^(ℓ/2) products where ℓ ≤ 130.
    // For ℓ = 20 (a typical size), that's 2^10 = 1024 products.
    let mut rng = OsRng;
    let mut acc = [0u64; 6];
    let mut expected_sum = Fp::ZERO;

    // Sum 2000 products (realistic bound for medium-sized polynomials)
    for _ in 0..2000 {
      let field_elem = Fp::random(&mut rng);
      let small = rng.next_u64();

      expected_sum += field_elem * Fp::from(small);

      let product = mul_4_by_1(&field_elem.0, small);
      let mut carry = 0u128;
      for i in 0..5 {
        let sum = (acc[i] as u128) + (product[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
      }
      acc[5] = acc[5].wrapping_add(carry as u64);
    }

    let result = Fp(barrett_reduce_6_fp(&acc));
    assert_eq!(result, expected_sum);
  }

  #[test]
  fn test_barrett_6_fq_sum_of_products() {
    let mut rng = OsRng;
    let mut acc = [0u64; 6];
    let mut expected_sum = Fq::ZERO;

    for _ in 0..100 {
      let field_elem = Fq::random(&mut rng);
      let small = rng.next_u64() >> 32;

      expected_sum += field_elem * Fq::from(small);

      let product = mul_4_by_1(&field_elem.0, small);
      let mut carry = 0u128;
      for i in 0..5 {
        let sum = (acc[i] as u128) + (product[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
      }
      acc[5] = acc[5].wrapping_add(carry as u64);
    }

    let result = Fq(barrett_reduce_6_fq(&acc));
    assert_eq!(result, expected_sum);
  }

  // ========================================================================
  // 9-limb Montgomery REDC tests
  // ========================================================================

  /// Helper to multiply two 4-limb values, producing an 8-limb result
  fn mul_4_by_4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
    let mut result = [0u64; 8];
    for i in 0..4 {
      let mut carry = 0u128;
      for j in 0..4 {
        let prod = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + carry;
        result[i + j] = prod as u64;
        carry = prod >> 64;
      }
      result[i + 4] = carry as u64;
    }
    result
  }

  #[test]
  fn test_montgomery_9_fp_single_product() {
    // Test with a single field × field product
    let a = Fp::from(12345u64);
    let b = Fp::from(67890u64);

    // Compute a_mont × b_mont (8 limbs, representing (a*b)*R² in unreduced form)
    let product = mul_4_by_4(&a.0, &b.0);

    // Extend to 9 limbs
    let c = [
      product[0], product[1], product[2], product[3], product[4], product[5], product[6],
      product[7], 0,
    ];

    // Montgomery reduce: should give (a*b)*R mod p = (a*b) in Montgomery form
    let result = Fp(montgomery_reduce_9_fp(&c));

    // Expected: a * b in field
    let expected = a * b;
    assert_eq!(result, expected);
  }

  #[test]
  fn test_montgomery_9_fp_sum_of_products() {
    // Test summing multiple field × field products
    let mut rng = OsRng;
    let mut acc = [0u64; 9];
    let mut expected_sum = Fp::ZERO;

    // Sum 100 products
    for _ in 0..100 {
      let a = Fp::random(&mut rng);
      let b = Fp::random(&mut rng);

      // Accumulate expected result
      expected_sum += a * b;

      // Compute a_mont × b_mont as 8 limbs
      let product = mul_4_by_4(&a.0, &b.0);

      // Add to accumulator with carry propagation
      let mut carry = 0u128;
      for i in 0..8 {
        let sum = (acc[i] as u128) + (product[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
      }
      acc[8] = acc[8].wrapping_add(carry as u64);
    }

    // Montgomery reduce and compare
    let result = Fp(montgomery_reduce_9_fp(&acc));
    assert_eq!(result, expected_sum);
  }

  #[test]
  fn test_montgomery_9_fp_many_products() {
    // Stress test with many products
    let mut rng = OsRng;
    let mut acc = [0u64; 9];
    let mut expected_sum = Fp::ZERO;

    // Sum 1000 products
    for _ in 0..1000 {
      let a = Fp::random(&mut rng);
      let b = Fp::random(&mut rng);

      expected_sum += a * b;

      let product = mul_4_by_4(&a.0, &b.0);
      let mut carry = 0u128;
      for i in 0..8 {
        let sum = (acc[i] as u128) + (product[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
      }
      acc[8] = acc[8].wrapping_add(carry as u64);
    }

    let result = Fp(montgomery_reduce_9_fp(&acc));
    assert_eq!(result, expected_sum);
  }

  #[test]
  fn test_montgomery_9_fq_sum_of_products() {
    let mut rng = OsRng;
    let mut acc = [0u64; 9];
    let mut expected_sum = Fq::ZERO;

    for _ in 0..100 {
      let a = Fq::random(&mut rng);
      let b = Fq::random(&mut rng);

      expected_sum += a * b;

      let product = mul_4_by_4(&a.0, &b.0);
      let mut carry = 0u128;
      for i in 0..8 {
        let sum = (acc[i] as u128) + (product[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
      }
      acc[8] = acc[8].wrapping_add(carry as u64);
    }

    let result = Fq(montgomery_reduce_9_fq(&acc));
    assert_eq!(result, expected_sum);
  }
}
