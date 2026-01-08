// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value field operations for optimized sumcheck.
//!
//! This module provides types and operations for the small-value optimization
//! described in "Speeding Up Sum-Check Proving". The key insight is distinguishing
//! between three multiplication types:
//!
//! - **ss** (small × small): Native i32/i64 multiplication
//! - **sl** (small × large): Barrett-optimized multiplication (~3× faster)
//! - **ll** (large × large): Standard field multiplication
//!
//! For polynomial evaluations on the boolean hypercube (typically i32 values),
//! we can perform many operations in native integers before converting to field.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    SmallValueField Trait                     │
//! │  (ss_mul, sl_mul, isl_mul, small_to_field, etc.)            │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Barrett Reduction (internal)                    │
//! │  mul_fp_by_i64, mul_fq_by_i64 - ~9 base muls vs ~32 naive   │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::wide_limbs::{SignedWideLimbs, WideLimbs};
use ff::PrimeField;
use std::{
  fmt::Debug,
  ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

// ============================================================================
// SmallValueField Trait
// ============================================================================

/// Trait for fields that support small-value optimization.
///
/// This trait defines operations for efficient arithmetic when polynomial
/// evaluations fit in native integers. The key optimization is avoiding
/// expensive field operations until absolutely necessary.
///
/// # Type Parameters
/// - `SmallValue`: Native type for witness values (i32 or i64)
///
/// # Implementations
/// - `Fp: SmallValueField<i32>` with `IntermediateSmallValue = i64`
/// - `Fp: SmallValueField<i64>` with `IntermediateSmallValue = i128`
///
/// # Overflow Bounds (for D=2 Spartan with typical witness values)
///
/// | Step | Bound | Bits | Container |
/// |------|-------|------|-----------|
/// | Original witness values | 2²⁰ | 20 | i32 |
/// | After 3 extensions (D=2) | 2²³ | 23 | i32 ✓ |
/// | Product of two extended | 2⁴⁶ | 46 | i64 ✓ |
pub trait SmallValueField<SmallValue>: PrimeField
where
  SmallValue: Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + Eq
    + Add<Output = SmallValue>
    + Sub<Output = SmallValue>
    + Neg<Output = SmallValue>
    + AddAssign
    + SubAssign
    + Send
    + Sync,
{
  /// Intermediate type for products (i64 for i32 inputs, i128 for i64 inputs).
  /// Used when multiplying two SmallValues together.
  type IntermediateSmallValue: Copy + Clone + Default + Debug + PartialEq + Eq + Send + Sync;

  /// Unreduced accumulator for field × integer products.
  /// - For i32/i64: SignedWideLimbs<6> (384 bits)
  /// - For i64/i128: SignedWideLimbs<8> (512 bits)
  type UnreducedMontInt: Copy + Clone + Default + Debug + AddAssign + Send + Sync;

  /// Unreduced accumulator for field × field products (9 limbs, 576 bits).
  /// Used to delay modular reduction when summing many F × F products.
  /// The value is in 2R-scaled Montgomery form, reduced via Montgomery REDC.
  type UnreducedMontMont: Copy + Clone + Default + Debug + AddAssign + Send + Sync;

  // ===== Constructors =====

  /// Create a SmallValue from u32.
  fn small_from_u32(val: u32) -> SmallValue;

  /// Create a SmallValue from i32.
  fn small_from_i32(val: i32) -> SmallValue;

  /// Extract the inner i32 from a SmallValue (truncates for i64).
  fn small_inner(val: SmallValue) -> i32;

  // ===== Arithmetic =====

  /// Multiply SmallValue by a small constant (for Lagrange extension).
  /// p(k) = p0 + k * diff
  fn ss_mul_const(a: SmallValue, k: i32) -> SmallValue;

  // ===== Core Multiplications =====

  /// ss: small × small → intermediate (i32 × i32 → i64, or i64 × i64 → i128)
  fn ss_mul(a: SmallValue, b: SmallValue) -> Self::IntermediateSmallValue;

  /// sl: small × large → large (small × field → field)
  fn sl_mul(small: SmallValue, large: &Self) -> Self;

  /// isl: intermediate × large → large (intermediate × field → field)
  /// This is the key operation for accumulator building.
  fn isl_mul(small: Self::IntermediateSmallValue, large: &Self) -> Self;

  // ===== Conversions =====

  /// Convert SmallValue to field element.
  fn small_to_field(val: SmallValue) -> Self;

  /// Convert IntermediateSmallValue to field element.
  fn intermediate_to_field(val: Self::IntermediateSmallValue) -> Self;

  /// Try to convert a field element to SmallValue.
  /// Returns None if the value doesn't fit.
  fn try_field_to_small(val: &Self) -> Option<SmallValue>;

  // ===== Delayed Reduction Operations =====

  /// Multiply field element by signed integer and add to unreduced accumulator.
  /// acc += field × intermediate (keeps result in unreduced form, handles sign internally)
  fn unreduced_mont_int_mul_add(
    acc: &mut Self::UnreducedMontInt,
    field: &Self,
    small: Self::IntermediateSmallValue,
  );

  /// Multiply two field elements and add to unreduced accumulator.
  /// acc += field_a × field_b (keeps result in 2R-scaled unreduced form)
  fn unreduced_mont_mont_mul_add(acc: &mut Self::UnreducedMontMont, field_a: &Self, field_b: &Self);

  /// Reduce an unreduced field×integer accumulator to a field element.
  fn reduce_mont_int(acc: &Self::UnreducedMontInt) -> Self;

  /// Reduce an unreduced field×field accumulator to a field element.
  fn reduce_mont_mont(acc: &Self::UnreducedMontMont) -> Self;
}

/// Convert i64 to field element (handles negative values correctly).
#[inline]
pub fn i64_to_field<F: PrimeField>(val: i64) -> F {
  if val >= 0 {
    F::from(val as u64)
  } else {
    // Use wrapping_neg to handle i64::MIN correctly
    -F::from(val.wrapping_neg() as u64)
  }
}

// ============================================================================
// Barrett Reduction - Internal Implementation
// ============================================================================

/// Barrett reduction for optimized sl (small × large) multiplication.
///
/// Provides ~3× speedup for multiplying a field element by a small integer
/// compared to the naive approach of converting to field then multiplying.
///
/// # Cost Analysis
///
/// | Operation | Base Multiplications |
/// |-----------|---------------------|
/// | Naive (convert + multiply) | ~32 |
/// | Barrett reduction | ~9-13 |
///
/// # How It Works
///
/// Field elements are stored in Montgomery form: `a_mont = a × R mod p`
///
/// When multiplying by a small integer:
/// ```text
/// small × a_mont = small × (a × R) = (small × a) × R mod p
/// ```
///
/// The result is already in Montgomery form! We just need to:
/// 1. Multiply the Montgomery limbs by the small integer (4 base muls)
/// 2. Barrett-reduce the result back to 4 limbs (~5-9 base muls)
mod barrett {
  use halo2curves::pasta::{Fp, Fq};
  use std::ops::Neg;

  // ==========================================================================
  // Const helper to double a 4-limb value to 5-limb value
  // ==========================================================================

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
  // Precomputed constants for Pallas Fp (Base field)
  // ==========================================================================

  const PALLAS_FP: [u64; 4] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
  ];

  const PALLAS_FP_2P: [u64; 5] = double_limbs(PALLAS_FP);
  const PALLAS_FP_MU: u64 = 0xffffffffffffffff;

  // ==========================================================================
  // Precomputed constants for Pallas Fq (Scalar field)
  // ==========================================================================

  const PALLAS_FQ: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
  ];

  const PALLAS_FQ_2Q: [u64; 5] = double_limbs(PALLAS_FQ);
  const PALLAS_FQ_MU: u64 = 0xffffffffffffffff;

  // ==========================================================================
  // Public API - Only i64 needed for SmallValueField
  // ==========================================================================

  /// Multiply Pallas base field element by i64 (signed).
  #[inline]
  pub(super) fn mul_fp_by_i64(large: &Fp, small: i64) -> Fp {
    if small >= 0 {
      mul_fp_by_u64(large, small as u64)
    } else {
      mul_fp_by_u64(large, small.wrapping_neg() as u64).neg()
    }
  }

  /// Multiply Pallas scalar field element by i64 (signed).
  #[inline]
  pub(super) fn mul_fq_by_i64(large: &Fq, small: i64) -> Fq {
    if small >= 0 {
      mul_fq_by_u64(large, small as u64)
    } else {
      mul_fq_by_u64(large, small.wrapping_neg() as u64).neg()
    }
  }

  // ==========================================================================
  // Internal u64 multiplication
  // ==========================================================================

  #[inline]
  fn mul_fp_by_u64(large: &Fp, small: u64) -> Fp {
    if small == 0 {
      return Fp::zero();
    }
    if small == 1 {
      return *large;
    }
    let c = mul_4_by_1(&large.0, small);
    Fp(barrett_reduce_5_fp(&c))
  }

  #[inline]
  fn mul_fq_by_u64(large: &Fq, small: u64) -> Fq {
    if small == 0 {
      return Fq::zero();
    }
    if small == 1 {
      return *large;
    }
    let c = mul_4_by_1(&large.0, small);
    Fq(barrett_reduce_5_fq(&c))
  }

  // ==========================================================================
  // Core bignum operations
  // ==========================================================================

  #[inline(always)]
  fn mul_4_by_1(a: &[u64; 4], b: u64) -> [u64; 5] {
    let mut result = [0u64; 5];
    let mut carry = 0u128;
    for i in 0..4 {
      let prod = (a[i] as u128) * (b as u128) + carry;
      result[i] = prod as u64;
      carry = prod >> 64;
    }
    result[4] = carry as u64;
    result
  }

  /// Public version of mul_4_by_1 for use in trait impls.
  #[inline(always)]
  pub(super) fn mul_4_by_1_ext(a: &[u64; 4], b: u64) -> [u64; 5] {
    mul_4_by_1(a, b)
  }

  /// Multiply two 4-limb values, producing an 8-limb result.
  #[inline(always)]
  pub(super) fn mul_4_by_4_ext(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
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

  /// Multiply 4-limb field by 2-limb integer (u128), producing a 6-limb result.
  /// Used for i64/i128 small-value optimization where IntermediateSmallValue is i128.
  #[inline(always)]
  pub(super) fn mul_4_by_2_ext(a: &[u64; 4], b: u128) -> [u64; 6] {
    let b_lo = b as u64;
    let b_hi = (b >> 64) as u64;

    // Multiply a by b_lo (4x1 -> 5 limbs)
    let mut result = [0u64; 6];
    let mut carry = 0u128;
    for i in 0..4 {
      let prod = (a[i] as u128) * (b_lo as u128) + carry;
      result[i] = prod as u64;
      carry = prod >> 64;
    }
    result[4] = carry as u64;

    // Multiply a by b_hi and add at offset 1 (4x1 -> 5 limbs, shifted)
    carry = 0u128;
    for i in 0..4 {
      let prod = (a[i] as u128) * (b_hi as u128) + (result[i + 1] as u128) + carry;
      result[i + 1] = prod as u64;
      carry = prod >> 64;
    }
    result[5] = carry as u64;

    result
  }

  #[inline(always)]
  fn barrett_reduce_5_fp(c: &[u64; 5]) -> [u64; 4] {
    let c_tilde = (c[3] >> 63) | (c[4] << 1);
    let m = {
      let product = (c_tilde as u128) * (PALLAS_FP_MU as u128);
      (product >> 64) as u64
    };
    let m_times_2p = mul_5_by_1(&PALLAS_FP_2P, m);
    let mut r = sub_5_5(c, &m_times_2p);
    // At most 2 iterations needed: after Barrett approximation, 0 ≤ r < 2p
    while gte_5_4(&r, &PALLAS_FP) {
      r = sub_5_4(&r, &PALLAS_FP);
    }
    [r[0], r[1], r[2], r[3]]
  }

  #[inline(always)]
  fn barrett_reduce_5_fq(c: &[u64; 5]) -> [u64; 4] {
    let c_tilde = (c[3] >> 63) | (c[4] << 1);
    let m = {
      let product = (c_tilde as u128) * (PALLAS_FQ_MU as u128);
      (product >> 64) as u64
    };
    let m_times_2q = mul_5_by_1(&PALLAS_FQ_2Q, m);
    let mut r = sub_5_5(c, &m_times_2q);
    // At most 2 iterations needed: after Barrett approximation, 0 ≤ r < 2q
    while gte_5_4(&r, &PALLAS_FQ) {
      r = sub_5_4(&r, &PALLAS_FQ);
    }
    [r[0], r[1], r[2], r[3]]
  }

  #[inline(always)]
  fn mul_5_by_1(a: &[u64; 5], b: u64) -> [u64; 5] {
    let mut result = [0u64; 5];
    let mut carry = 0u128;
    for i in 0..5 {
      let prod = (a[i] as u128) * (b as u128) + carry;
      result[i] = prod as u64;
      carry = prod >> 64;
    }
    result
  }

  #[inline(always)]
  fn sub_5_5(a: &[u64; 5], b: &[u64; 5]) -> [u64; 5] {
    let mut result = [0u64; 5];
    let mut borrow = 0u64;
    for i in 0..5 {
      let (diff, b1) = a[i].overflowing_sub(b[i]);
      let (diff2, b2) = diff.overflowing_sub(borrow);
      result[i] = diff2;
      borrow = (b1 as u64) + (b2 as u64);
    }
    result
  }

  #[inline(always)]
  fn sub_5_4(a: &[u64; 5], b: &[u64; 4]) -> [u64; 5] {
    let mut result = [0u64; 5];
    let mut borrow = 0u64;
    for i in 0..4 {
      let (diff, b1) = a[i].overflowing_sub(b[i]);
      let (diff2, b2) = diff.overflowing_sub(borrow);
      result[i] = diff2;
      borrow = (b1 as u64) + (b2 as u64);
    }
    let (diff, _) = a[4].overflowing_sub(borrow);
    result[4] = diff;
    result
  }

  #[inline(always)]
  fn gte_5_4(a: &[u64; 5], b: &[u64; 4]) -> bool {
    if a[4] > 0 {
      return true;
    }
    for i in (0..4).rev() {
      if a[i] > b[i] {
        return true;
      }
      if a[i] < b[i] {
        return false;
      }
    }
    true
  }

  // ==========================================================================
  // 6-limb Barrett reduction (for UnreducedMontInt accumulator)
  // ==========================================================================

  // Precomputed: 2^320 mod p (for reducing the 6th limb)
  // This allows us to convert c[5] * 2^320 into an equivalent value mod p
  // that fits in 4 limbs, which we then add to the lower limbs.
  //
  // Computed as: pow(2, 320, p) for Pallas Fp
  // p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
  // 2^320 mod p = 0x3fffffffffffffff76e59c0fdacc1b91bd91d548094cf917992d30ed00000001
  #[allow(dead_code)]
  const R320_MOD_FP: [u64; 4] = [
    0x992d30ed00000001,
    0xbd91d548094cf917,
    0x76e59c0fdacc1b91,
    0x3fffffffffffffff,
  ];

  // 2^320 mod q for Pallas Fq
  // q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
  // 2^320 mod q = 0x3fffffffffffffff76e59c0fd9ad5c89f12aec780994a8d98c46eb2100000001
  #[allow(dead_code)]
  const R320_MOD_FQ: [u64; 4] = [
    0x8c46eb2100000001,
    0xf12aec780994a8d9,
    0x76e59c0fd9ad5c89,
    0x3fffffffffffffff,
  ];

  /// Barrett reduction for 6-limb input to Fp.
  ///
  /// Input is already in Montgomery form (R-scaled). This function reduces
  /// the 6-limb value mod p while preserving the Montgomery scaling.
  #[inline]
  #[allow(dead_code)] // Will be used in accumulator optimization
  pub(crate) fn barrett_reduce_6_fp(c: &[u64; 6]) -> [u64; 4] {
    // Reduce c[5] * 2^320 ≡ c[5] * R320 (mod p), then reduce c[4] * 2^256, etc.
    // R320 = 2^320 mod p, R256 = 2^256 mod p
    //
    // We do: result = c[0..4] + c[4] * R256 + c[5] * R320 (mod p)

    // c[4] * R256_MOD_FP (4x1 -> 5 limbs)
    let c4_contrib = mul_4_by_1(&R256_MOD_FP, c[4]);
    // c[5] * R320_MOD_FP (4x1 -> 5 limbs)
    let c5_contrib = mul_4_by_1(&R320_MOD_FP, c[5]);

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
      while gte_4_4(&r, &PALLAS_FP) {
        r = sub_4_4(&r, &PALLAS_FP);
      }
      return r;
    }

    // Recurse (this will terminate because sum < c in most cases)
    barrett_reduce_6_fp(&sum)
  }

  // 2^256 mod p for Pallas Fp
  // Computed as: pow(2, 256, p)
  const R256_MOD_FP: [u64; 4] = [
    0x34786d38fffffffd,
    0x992c350be41914ad,
    0xffffffffffffffff,
    0x3fffffffffffffff,
  ];

  #[inline(always)]
  fn sub_4_4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
      let (diff, b1) = a[i].overflowing_sub(b[i]);
      let (diff2, b2) = diff.overflowing_sub(borrow);
      result[i] = diff2;
      borrow = (b1 as u64) + (b2 as u64);
    }
    result
  }

  /// Barrett reduction for 6-limb input to Fq.
  #[inline]
  #[allow(dead_code)] // Will be used in accumulator optimization
  pub(crate) fn barrett_reduce_6_fq(c: &[u64; 6]) -> [u64; 4] {
    let c4_contrib = mul_4_by_1(&R256_MOD_FQ, c[4]);
    let c5_contrib = mul_4_by_1(&R320_MOD_FQ, c[5]);

    let mut sum = [0u64; 6];
    let mut carry = 0u128;
    for i in 0..4 {
      let s = (c[i] as u128) + (c4_contrib[i] as u128) + (c5_contrib[i] as u128) + carry;
      sum[i] = s as u64;
      carry = s >> 64;
    }
    let s = (c4_contrib[4] as u128) + (c5_contrib[4] as u128) + carry;
    sum[4] = s as u64;
    sum[5] = (s >> 64) as u64;

    if sum[5] == 0 && sum[4] == 0 {
      let mut r = [sum[0], sum[1], sum[2], sum[3]];
      while gte_4_4(&r, &PALLAS_FQ) {
        r = sub_4_4(&r, &PALLAS_FQ);
      }
      return r;
    }

    barrett_reduce_6_fq(&sum)
  }

  // 2^256 mod q for Pallas Fq
  // Computed as: pow(2, 256, q)
  const R256_MOD_FQ: [u64; 4] = [
    0x5b2b3e9cfffffffd,
    0x992c350be3420567,
    0xffffffffffffffff,
    0x3fffffffffffffff,
  ];

  // ==========================================================================
  // 8-limb Barrett reduction (for i64/i128 UnreducedMontInt accumulator)
  // ==========================================================================

  // 2^384 mod p for Pallas Fp
  // Computed as: pow(2, 384, p)
  const R384_MOD_FP: [u64; 4] = [
    0xcb8792c700000003,
    0x66d3caf41be6eb52,
    0x9b4b3c4bfffffffc,
    0x36e59c0fdacc1b91,
  ];

  // 2^448 mod p for Pallas Fp
  // Computed as: pow(2, 448, p)
  const R448_MOD_FP: [u64; 4] = [
    0x9b9858f294cf91ba,
    0x8635bd2c4252b065,
    0x496d41af7b9cb714,
    0x1b4b3c4bfffffffc,
  ];

  // 2^384 mod q for Pallas Fq
  // Computed as: pow(2, 384, q)
  const R384_MOD_FQ: [u64; 4] = [
    0xa4d4c16300000003,
    0x66d3caf41cbdfa98,
    0xcee4537bfffffffc,
    0x36e59c0fd9ad5c89,
  ];

  // 2^448 mod q for Pallas Fq
  // Computed as: pow(2, 448, q)
  const R448_MOD_FQ: [u64; 4] = [
    0xcc920bb9994a8dd9,
    0x87a7dcbe1ff6e0d7,
    0x496d41af7ccfdaa9,
    0x0ee4537bfffffffc,
  ];

  /// Barrett reduction for 8-limb input to Fp.
  /// Used for i64/i128 small-value optimization where products are 6 limbs
  /// and we accumulate into 8 limbs.
  #[inline]
  pub(crate) fn barrett_reduce_8_fp(c: &[u64; 8]) -> [u64; 4] {
    // Reduce high limbs: c[6] * 2^384 + c[7] * 2^448
    let c6_contrib = mul_4_by_1(&R384_MOD_FP, c[6]);
    let c7_contrib = mul_4_by_1(&R448_MOD_FP, c[7]);

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
    barrett_reduce_6_fp(&sum)
  }

  /// Barrett reduction for 8-limb input to Fq.
  #[inline]
  pub(crate) fn barrett_reduce_8_fq(c: &[u64; 8]) -> [u64; 4] {
    let c6_contrib = mul_4_by_1(&R384_MOD_FQ, c[6]);
    let c7_contrib = mul_4_by_1(&R448_MOD_FQ, c[7]);

    let mut sum = [0u64; 6];
    let mut carry = 0u128;
    for i in 0..4 {
      let s = (c[i] as u128) + (c6_contrib[i] as u128) + (c7_contrib[i] as u128) + carry;
      sum[i] = s as u64;
      carry = s >> 64;
    }
    let s = (c[4] as u128) + (c6_contrib[4] as u128) + (c7_contrib[4] as u128) + carry;
    sum[4] = s as u64;
    carry = s >> 64;
    let s = (c[5] as u128) + carry;
    sum[5] = s as u64;

    barrett_reduce_6_fq(&sum)
  }

  // ==========================================================================
  // 9-limb Montgomery REDC (for UnreducedMontMont accumulator)
  // ==========================================================================

  // Precomputed: 2^512 mod p (for reducing high limbs)
  // Used to reduce 9-limb values to 8 limbs before REDC
  const R512_MOD_FP: [u64; 4] = [
    0x8c78ecb30000000f,
    0xd7d30dbd8b0de0e7,
    0x7797a99bc3c95d18,
    0x096d41af7b9cb714,
  ];

  const R512_MOD_FQ: [u64; 4] = [
    0xfc9678ff0000000f,
    0x67bb433d891a16e3,
    0x7fae231004ccf590,
    0x096d41af7ccfdaa9,
  ];

  // Montgomery constant: p' = -p^(-1) mod 2^64 (low word only needed for REDC)
  const PALLAS_FP_INV: u64 = 0x992d30ecffffffff;
  const PALLAS_FQ_INV: u64 = 0x8c46eb20ffffffff;

  /// Montgomery REDC for 9-limb input to Fp.
  ///
  /// Reduces a 2R-scaled value (sum of field×field products) to 1R-scaled.
  /// Input: T representing x·R² (up to 9 limbs)
  /// Output: x·R mod p (4 limbs, standard Montgomery form)
  #[inline]
  #[allow(dead_code)] // Will be used in accumulator optimization
  pub(crate) fn montgomery_reduce_9_fp(c: &[u64; 9]) -> [u64; 4] {
    // Step 1: Reduce 9 limbs to 8 limbs using precomputed 2^512 mod p
    let mut t = [0u64; 9];
    if c[8] == 0 {
      t[..8].copy_from_slice(&c[..8]);
    } else {
      // t = c[0..8] + c[8] * R512_MOD_FP
      let high_contribution = mul_4_by_1(&R512_MOD_FP, c[8]);
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
        return montgomery_reduce_9_fp(&t);
      }
    }

    // Step 2: Montgomery REDC on 8-limb value
    // t contains our value, need to reduce to 4 limbs
    montgomery_reduce_8_fp(&[t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]])
  }

  /// Montgomery REDC for 8-limb input.
  /// Standard Montgomery reduction: T × R⁻¹ mod p
  #[inline]
  #[allow(dead_code)]
  fn montgomery_reduce_8_fp(t: &[u64; 8]) -> [u64; 4] {
    // Use 9 limbs to track overflow
    let mut r = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], 0u64];

    // Montgomery reduction: for each of the low 4 limbs, eliminate it
    // by adding appropriate multiples of p
    for i in 0..4 {
      // q = r[i] * p' mod 2^64
      let q = r[i].wrapping_mul(PALLAS_FP_INV);

      // r += q * p * 2^(64*i)
      // qp = q * p, which is 5 limbs (since p is 4 limbs and q is 1 limb)
      let qp = mul_4_by_1(&PALLAS_FP, q);

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
    while result[4] > 0 || gte_4_4(&[result[0], result[1], result[2], result[3]], &PALLAS_FP) {
      let sub = sub_5_4(&result, &PALLAS_FP);
      result = sub;
    }

    [result[0], result[1], result[2], result[3]]
  }

  /// Montgomery REDC for 9-limb input to Fq.
  #[inline]
  #[allow(dead_code)] // Will be used in accumulator optimization
  pub(crate) fn montgomery_reduce_9_fq(c: &[u64; 9]) -> [u64; 4] {
    let mut t = [0u64; 9];
    if c[8] == 0 {
      t[..8].copy_from_slice(&c[..8]);
    } else {
      let high_contribution = mul_4_by_1(&R512_MOD_FQ, c[8]);
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

      if t[8] > 0 {
        return montgomery_reduce_9_fq(&t);
      }
    }

    montgomery_reduce_8_fq(&[t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]])
  }

  #[inline]
  #[allow(dead_code)]
  fn montgomery_reduce_8_fq(t: &[u64; 8]) -> [u64; 4] {
    // Use 9 limbs to track overflow
    let mut r = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], 0u64];

    for i in 0..4 {
      let q = r[i].wrapping_mul(PALLAS_FQ_INV);
      let qp = mul_4_by_1(&PALLAS_FQ, q);

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
    // We need to reduce this to [0, q)
    let mut result = [r[4], r[5], r[6], r[7], r[8]];

    // Reduce until result < q
    while result[4] > 0 || gte_4_4(&[result[0], result[1], result[2], result[3]], &PALLAS_FQ) {
      let sub = sub_5_4(&result, &PALLAS_FQ);
      result = sub;
    }

    [result[0], result[1], result[2], result[3]]
  }

  // Helper functions for 4-limb operations
  #[inline(always)]
  #[allow(dead_code)]
  fn gte_4_4(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
      if a[i] > b[i] {
        return true;
      }
      if a[i] < b[i] {
        return false;
      }
    }
    true // equal
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
        let barrett = mul_fp_by_u64(&large, small);
        assert_eq!(naive, barrett, "Fp mismatch for small = {}", small);
      }
    }

    #[test]
    fn test_barrett_fp_random() {
      let mut rng = OsRng;
      for _ in 0..1000 {
        let large = Fp::random(&mut rng);
        let small: u64 = rng.next_u64();
        assert_eq!(Fp::from(small) * large, mul_fp_by_u64(&large, small));
      }
    }

    #[test]
    fn test_barrett_fp_i64() {
      let large = Fp::from(42u64);
      assert_eq!(mul_fp_by_i64(&large, 100), Fp::from(100u64) * large);
      assert_eq!(mul_fp_by_i64(&large, -100), -Fp::from(100u64) * large);
    }

    #[test]
    fn test_barrett_fq_matches_naive() {
      for small in [0u64, 1, 2, 42, 1000, u32::MAX as u64, u64::MAX] {
        let large = Fq::random(&mut OsRng);
        let naive = Fq::from(small) * large;
        let barrett = mul_fq_by_u64(&large, small);
        assert_eq!(naive, barrett, "Fq mismatch for small = {}", small);
      }
    }

    #[test]
    fn test_barrett_fq_random() {
      let mut rng = OsRng;
      for _ in 0..1000 {
        let large = Fq::random(&mut rng);
        let small: u64 = rng.next_u64();
        assert_eq!(Fq::from(small) * large, mul_fq_by_u64(&large, small));
      }
    }

    #[test]
    fn test_barrett_fq_i64() {
      let large = Fq::from(42u64);
      assert_eq!(mul_fq_by_i64(&large, 100), Fq::from(100u64) * large);
      assert_eq!(mul_fq_by_i64(&large, -100), -Fq::from(100u64) * large);
    }

    #[test]
    fn test_constants_match_halo2curves() {
      let p_minus_one = -Fp::ONE;
      let expected = Fp::from_raw([
        PALLAS_FP[0].wrapping_sub(1),
        PALLAS_FP[1],
        PALLAS_FP[2],
        PALLAS_FP[3],
      ]);
      assert_eq!(p_minus_one, expected);

      let q_minus_one = -Fq::ONE;
      let expected = Fq::from_raw([
        PALLAS_FQ[0].wrapping_sub(1),
        PALLAS_FQ[1],
        PALLAS_FQ[2],
        PALLAS_FQ[3],
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
}

// ============================================================================
// SmallValueField Implementations
// ============================================================================

/// Helper for try_field_to_small: attempts to convert a field element to i32.
///
/// Returns Some(v) if the field element represents a small integer in [-2^31, 2^31-1].
fn try_field_to_small_impl<F: PrimeField>(val: &F) -> Option<i32> {
  let repr = val.to_repr();
  let bytes = repr.as_ref();

  // Check if value fits in positive i32
  let high_zero = bytes[4..].iter().all(|&b| b == 0);
  if high_zero {
    let val_u32 = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    if val_u32 <= i32::MAX as u32 {
      return Some(val_u32 as i32);
    }
  }

  // Check if negation fits in i32 (value is negative)
  let neg_val = val.neg();
  let neg_repr = neg_val.to_repr();
  let neg_bytes = neg_repr.as_ref();
  let neg_high_zero = neg_bytes[4..].iter().all(|&b| b == 0);
  if neg_high_zero {
    let neg_u32 = u32::from_le_bytes(neg_bytes[..4].try_into().unwrap());
    if neg_u32 > 0 && neg_u32 <= (i32::MAX as u32) + 1 {
      return Some(-(neg_u32 as i64) as i32);
    }
  }

  None
}

impl SmallValueField<i32> for halo2curves::pasta::Fp {
  type IntermediateSmallValue = i64;
  type UnreducedMontInt = SignedWideLimbs<6>;
  type UnreducedMontMont = WideLimbs<9>;

  #[inline]
  fn small_from_u32(val: u32) -> i32 {
    val as i32
  }

  #[inline]
  fn small_from_i32(val: i32) -> i32 {
    val
  }

  #[inline]
  fn small_inner(val: i32) -> i32 {
    val
  }

  #[inline]
  fn ss_mul_const(a: i32, k: i32) -> i32 {
    a * k
  }

  #[inline]
  fn ss_mul(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
  }

  #[inline]
  fn sl_mul(small: i32, large: &Self) -> Self {
    barrett::mul_fp_by_i64(large, small as i64)
  }

  #[inline]
  fn isl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_fp_by_i64(large, small)
  }

  #[inline]
  fn small_to_field(val: i32) -> Self {
    if val >= 0 {
      Self::from(val as u64)
    } else {
      -Self::from((-val) as u64)
    }
  }

  #[inline]
  fn intermediate_to_field(val: i64) -> Self {
    i64_to_field(val)
  }

  fn try_field_to_small(val: &Self) -> Option<i32> {
    try_field_to_small_impl(val)
  }

  #[inline]
  fn unreduced_mont_int_mul_add(acc: &mut Self::UnreducedMontInt, field: &Self, small: i64) {
    // Handle sign: accumulate into pos or neg based on sign of small
    let (target, magnitude) = if small >= 0 {
      (&mut acc.pos, small as u64)
    } else {
      (&mut acc.neg, (-small) as u64)
    };
    // Compute field × |small| as 5 limbs and add to target accumulator
    let product = barrett::mul_4_by_1_ext(&field.0, magnitude);
    let mut carry = 0u128;
    for (target_limb, &prod_limb) in target.0.iter_mut().take(5).zip(product.iter()) {
      let sum = (*target_limb as u128) + (prod_limb as u128) + carry;
      *target_limb = sum as u64;
      carry = sum >> 64;
    }
    target.0[5] = target.0[5].wrapping_add(carry as u64);
  }

  #[inline]
  fn unreduced_mont_mont_mul_add(
    acc: &mut Self::UnreducedMontMont,
    field_a: &Self,
    field_b: &Self,
  ) {
    // Compute field_a × field_b as 8 limbs and add to accumulator
    let product = barrett::mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline]
  fn reduce_mont_int(acc: &Self::UnreducedMontInt) -> Self {
    // Reduce both accumulators and subtract: pos - neg
    let pos_reduced = Self(barrett::barrett_reduce_6_fp(&acc.pos.0));
    let neg_reduced = Self(barrett::barrett_reduce_6_fp(&acc.neg.0));
    pos_reduced - neg_reduced
  }

  #[inline]
  fn reduce_mont_mont(acc: &Self::UnreducedMontMont) -> Self {
    Self(barrett::montgomery_reduce_9_fp(&acc.0))
  }
}

impl SmallValueField<i64> for halo2curves::pasta::Fp {
  type IntermediateSmallValue = i128;
  type UnreducedMontInt = SignedWideLimbs<8>;
  type UnreducedMontMont = WideLimbs<9>;

  #[inline]
  fn small_from_u32(val: u32) -> i64 {
    val as i64
  }

  #[inline]
  fn small_from_i32(val: i32) -> i64 {
    val as i64
  }

  #[inline]
  fn small_inner(val: i64) -> i32 {
    val as i32
  }

  #[inline]
  fn ss_mul_const(a: i64, k: i32) -> i64 {
    a * (k as i64)
  }

  #[inline]
  fn ss_mul(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }

  #[inline]
  fn sl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_fp_by_i64(large, small)
  }

  #[inline]
  fn isl_mul(small: i128, large: &Self) -> Self {
    if small == 0 {
      return Self::zero();
    }
    let (is_neg, mag) = if small >= 0 {
      (false, small as u128)
    } else {
      (true, (-small) as u128)
    };
    let product = barrett::mul_4_by_2_ext(&large.0, mag);
    let c8 = [
      product[0], product[1], product[2], product[3], product[4], product[5], 0, 0,
    ];
    let result = Self(barrett::barrett_reduce_8_fp(&c8));
    if is_neg { -result } else { result }
  }

  #[inline]
  fn small_to_field(val: i64) -> Self {
    i64_to_field(val)
  }

  #[inline]
  fn intermediate_to_field(val: i128) -> Self {
    i128_to_field(val)
  }

  fn try_field_to_small(val: &Self) -> Option<i64> {
    let repr = val.to_repr();
    let bytes = repr.as_ref();

    // Check if value fits in positive i64
    let high_zero = bytes[8..].iter().all(|&b| b == 0);
    if high_zero {
      let val_u64 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
      if val_u64 <= i64::MAX as u64 {
        return Some(val_u64 as i64);
      }
    }

    // Check if negation fits in i64
    let neg_val = val.neg();
    let neg_repr = neg_val.to_repr();
    let neg_bytes = neg_repr.as_ref();
    let neg_high_zero = neg_bytes[8..].iter().all(|&b| b == 0);
    if neg_high_zero {
      let neg_u64 = u64::from_le_bytes(neg_bytes[..8].try_into().unwrap());
      if neg_u64 > 0 && neg_u64 <= (i64::MAX as u64) + 1 {
        return Some(-(neg_u64 as i128) as i64);
      }
    }

    None
  }

  #[inline]
  fn unreduced_mont_int_mul_add(acc: &mut Self::UnreducedMontInt, field: &Self, small: i128) {
    let (target, magnitude) = if small >= 0 {
      (&mut acc.pos, small as u128)
    } else {
      (&mut acc.neg, (-small) as u128)
    };
    let product = barrett::mul_4_by_2_ext(&field.0, magnitude);
    let mut carry = 0u128;
    for (target_limb, &prod_limb) in target.0.iter_mut().take(6).zip(product.iter()) {
      let sum = (*target_limb as u128) + (prod_limb as u128) + carry;
      *target_limb = sum as u64;
      carry = sum >> 64;
    }
    for target_limb in target.0[6..8].iter_mut() {
      let sum = (*target_limb as u128) + carry;
      *target_limb = sum as u64;
      carry = sum >> 64;
    }
  }

  #[inline]
  fn unreduced_mont_mont_mul_add(
    acc: &mut Self::UnreducedMontMont,
    field_a: &Self,
    field_b: &Self,
  ) {
    let product = barrett::mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline]
  fn reduce_mont_int(acc: &Self::UnreducedMontInt) -> Self {
    let pos_reduced = Self(barrett::barrett_reduce_8_fp(&acc.pos.0));
    let neg_reduced = Self(barrett::barrett_reduce_8_fp(&acc.neg.0));
    pos_reduced - neg_reduced
  }

  #[inline]
  fn reduce_mont_mont(acc: &Self::UnreducedMontMont) -> Self {
    Self(barrett::montgomery_reduce_9_fp(&acc.0))
  }
}

impl SmallValueField<i32> for halo2curves::pasta::Fq {
  type IntermediateSmallValue = i64;
  type UnreducedMontInt = SignedWideLimbs<6>;
  type UnreducedMontMont = WideLimbs<9>;

  #[inline]
  fn small_from_u32(val: u32) -> i32 {
    val as i32
  }

  #[inline]
  fn small_from_i32(val: i32) -> i32 {
    val
  }

  #[inline]
  fn small_inner(val: i32) -> i32 {
    val
  }

  #[inline]
  fn ss_mul_const(a: i32, k: i32) -> i32 {
    a * k
  }

  #[inline]
  fn ss_mul(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
  }

  #[inline]
  fn sl_mul(small: i32, large: &Self) -> Self {
    barrett::mul_fq_by_i64(large, small as i64)
  }

  #[inline]
  fn isl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_fq_by_i64(large, small)
  }

  #[inline]
  fn small_to_field(val: i32) -> Self {
    if val >= 0 {
      Self::from(val as u64)
    } else {
      -Self::from((-val) as u64)
    }
  }

  #[inline]
  fn intermediate_to_field(val: i64) -> Self {
    i64_to_field(val)
  }

  fn try_field_to_small(val: &Self) -> Option<i32> {
    try_field_to_small_impl(val)
  }

  #[inline]
  fn unreduced_mont_int_mul_add(acc: &mut Self::UnreducedMontInt, field: &Self, small: i64) {
    // Handle sign: accumulate into pos or neg based on sign of small
    let (target, magnitude) = if small >= 0 {
      (&mut acc.pos, small as u64)
    } else {
      (&mut acc.neg, (-small) as u64)
    };
    // Compute field × |small| as 5 limbs and add to target accumulator
    let product = barrett::mul_4_by_1_ext(&field.0, magnitude);
    let mut carry = 0u128;
    for (target_limb, &prod_limb) in target.0.iter_mut().take(5).zip(product.iter()) {
      let sum = (*target_limb as u128) + (prod_limb as u128) + carry;
      *target_limb = sum as u64;
      carry = sum >> 64;
    }
    target.0[5] = target.0[5].wrapping_add(carry as u64);
  }

  #[inline]
  fn unreduced_mont_mont_mul_add(
    acc: &mut Self::UnreducedMontMont,
    field_a: &Self,
    field_b: &Self,
  ) {
    let product = barrett::mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline]
  fn reduce_mont_int(acc: &Self::UnreducedMontInt) -> Self {
    // Reduce both accumulators and subtract: pos - neg
    let pos_reduced = Self(barrett::barrett_reduce_6_fq(&acc.pos.0));
    let neg_reduced = Self(barrett::barrett_reduce_6_fq(&acc.neg.0));
    pos_reduced - neg_reduced
  }

  #[inline]
  fn reduce_mont_mont(acc: &Self::UnreducedMontMont) -> Self {
    Self(barrett::montgomery_reduce_9_fq(&acc.0))
  }
}

impl SmallValueField<i64> for halo2curves::pasta::Fq {
  type IntermediateSmallValue = i128;
  type UnreducedMontInt = SignedWideLimbs<8>;
  type UnreducedMontMont = WideLimbs<9>;

  #[inline]
  fn small_from_u32(val: u32) -> i64 {
    val as i64
  }

  #[inline]
  fn small_from_i32(val: i32) -> i64 {
    val as i64
  }

  #[inline]
  fn small_inner(val: i64) -> i32 {
    val as i32
  }

  #[inline]
  fn ss_mul_const(a: i64, k: i32) -> i64 {
    a * (k as i64)
  }

  #[inline]
  fn ss_mul(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }

  #[inline]
  fn sl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_fq_by_i64(large, small)
  }

  #[inline]
  fn isl_mul(small: i128, large: &Self) -> Self {
    if small == 0 {
      return Self::zero();
    }
    let (is_neg, mag) = if small >= 0 {
      (false, small as u128)
    } else {
      (true, (-small) as u128)
    };
    let product = barrett::mul_4_by_2_ext(&large.0, mag);
    let c8 = [
      product[0], product[1], product[2], product[3], product[4], product[5], 0, 0,
    ];
    let result = Self(barrett::barrett_reduce_8_fq(&c8));
    if is_neg { -result } else { result }
  }

  #[inline]
  fn small_to_field(val: i64) -> Self {
    i64_to_field(val)
  }

  #[inline]
  fn intermediate_to_field(val: i128) -> Self {
    i128_to_field(val)
  }

  fn try_field_to_small(val: &Self) -> Option<i64> {
    let repr = val.to_repr();
    let bytes = repr.as_ref();

    let high_zero = bytes[8..].iter().all(|&b| b == 0);
    if high_zero {
      let val_u64 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
      if val_u64 <= i64::MAX as u64 {
        return Some(val_u64 as i64);
      }
    }

    let neg_val = val.neg();
    let neg_repr = neg_val.to_repr();
    let neg_bytes = neg_repr.as_ref();
    let neg_high_zero = neg_bytes[8..].iter().all(|&b| b == 0);
    if neg_high_zero {
      let neg_u64 = u64::from_le_bytes(neg_bytes[..8].try_into().unwrap());
      if neg_u64 > 0 && neg_u64 <= (i64::MAX as u64) + 1 {
        return Some(-(neg_u64 as i128) as i64);
      }
    }

    None
  }

  #[inline]
  fn unreduced_mont_int_mul_add(acc: &mut Self::UnreducedMontInt, field: &Self, small: i128) {
    let (target, magnitude) = if small >= 0 {
      (&mut acc.pos, small as u128)
    } else {
      (&mut acc.neg, (-small) as u128)
    };
    let product = barrett::mul_4_by_2_ext(&field.0, magnitude);
    let mut carry = 0u128;
    for (target_limb, &prod_limb) in target.0.iter_mut().take(6).zip(product.iter()) {
      let sum = (*target_limb as u128) + (prod_limb as u128) + carry;
      *target_limb = sum as u64;
      carry = sum >> 64;
    }
    for target_limb in target.0[6..8].iter_mut() {
      let sum = (*target_limb as u128) + carry;
      *target_limb = sum as u64;
      carry = sum >> 64;
    }
  }

  #[inline]
  fn unreduced_mont_mont_mul_add(
    acc: &mut Self::UnreducedMontMont,
    field_a: &Self,
    field_b: &Self,
  ) {
    let product = barrett::mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline]
  fn reduce_mont_int(acc: &Self::UnreducedMontInt) -> Self {
    let pos_reduced = Self(barrett::barrett_reduce_8_fq(&acc.pos.0));
    let neg_reduced = Self(barrett::barrett_reduce_8_fq(&acc.neg.0));
    pos_reduced - neg_reduced
  }

  #[inline]
  fn reduce_mont_mont(acc: &Self::UnreducedMontMont) -> Self {
    Self(barrett::montgomery_reduce_9_fq(&acc.0))
  }
}

/// Convert i128 to field element (handles negative values correctly).
#[inline]
pub fn i128_to_field<F: PrimeField>(val: i128) -> F {
  if val >= 0 {
    // Split into high and low u64 parts
    let lo = val as u64;
    let hi = (val >> 64) as u64;
    F::from(lo) + F::from(hi) * F::from(1u64 << 32) * F::from(1u64 << 32)
  } else {
    -i128_to_field::<F>(-val)
  }
}

// ============================================================================
// Tests
// ============================================================================

// FpI64/FqI64 wrapper types have been removed - use SmallValueField<i64> directly

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{polys::multilinear::MultilinearPolynomial, provider::pasta::pallas};

  type Scalar = pallas::Scalar;

  #[test]
  fn test_small_value_field_arithmetic() {
    let a = <Scalar as SmallValueField<i32>>::small_from_i32(10);
    let b = <Scalar as SmallValueField<i32>>::small_from_i32(3);

    assert_eq!(a + b, 13);
    assert_eq!(a - b, 7);
    assert_eq!(-a, -10);
    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(a, b), 30i64);
    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul_const(a, 5), 50);
  }

  #[test]
  fn test_small_value_field_negative() {
    let a = <Scalar as SmallValueField<i32>>::small_from_i32(-5);
    let b = <Scalar as SmallValueField<i32>>::small_from_i32(3);

    assert_eq!(a + b, -2);
    assert_eq!(a - b, -8);
    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(a, b), -15i64);

    let field_a = <Scalar as SmallValueField<i32>>::small_to_field(a);
    assert_eq!(field_a, -Scalar::from(5u64));
  }

  #[test]
  fn test_i64_to_field() {
    let pos: Scalar = i64_to_field(100);
    assert_eq!(pos, Scalar::from(100u64));

    let neg: Scalar = i64_to_field(-50);
    assert_eq!(neg, -Scalar::from(50u64));
  }

  #[test]
  fn test_small_multilinear_polynomial() {
    let poly = MultilinearPolynomial::new(vec![1i32, 2, 3, 4]);

    assert_eq!(poly.num_vars(), 2);
    assert_eq!(poly.Z.len(), 4);
    assert_eq!(poly[0], 1);
    assert_eq!(poly[3], 4);
  }

  #[test]
  fn test_to_field_conversion() {
    let evals: Vec<i32> = vec![1, -2, 3, -4];
    let small_poly = MultilinearPolynomial::new(evals);
    let field_poly: MultilinearPolynomial<Scalar> = small_poly.to_field();

    assert_eq!(field_poly.Z[0], Scalar::from(1u64));
    assert_eq!(field_poly.Z[1], -Scalar::from(2u64));
    assert_eq!(field_poly.Z[2], Scalar::from(3u64));
    assert_eq!(field_poly.Z[3], -Scalar::from(4u64));
  }

  #[test]
  fn test_try_field_to_small_roundtrip() {
    assert_eq!(
      <Scalar as SmallValueField<i32>>::try_field_to_small(&Scalar::from(42u64)),
      Some(42)
    );
    assert_eq!(
      <Scalar as SmallValueField<i32>>::try_field_to_small(&-Scalar::from(100u64)),
      Some(-100)
    );
    assert_eq!(
      <Scalar as SmallValueField<i32>>::try_field_to_small(&Scalar::from(u64::MAX)),
      None
    );
  }

  #[test]
  fn test_isl_mul() {
    use ff::Field;
    use rand_core::OsRng;

    let large = Scalar::random(&mut OsRng);
    let small: i64 = 12345;

    let result = <Scalar as SmallValueField<i32>>::isl_mul(small, &large);
    let expected = i64_to_field::<Scalar>(small) * large;

    assert_eq!(result, expected);
  }

  #[test]
  fn test_sl_mul() {
    use ff::Field;
    use rand_core::OsRng;

    let large = Scalar::random(&mut OsRng);
    let small: i32 = -999;

    let result = <Scalar as SmallValueField<i32>>::sl_mul(small, &large);
    let expected = <Scalar as SmallValueField<i32>>::small_to_field(small) * large;

    assert_eq!(result, expected);
  }

  #[test]
  fn test_overflow_bounds() {
    let typical_witness = 1i32 << 20;
    let extension_factor = 27i32;
    let after_extension = typical_witness * extension_factor;

    let prod = <Scalar as SmallValueField<i32>>::ss_mul(after_extension, after_extension);
    assert!(prod > 0);
    assert!(prod < (1i64 << 55));
  }

  #[test]
  fn test_ss_sign_combinations() {
    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(100, 200), 20000i64);
    assert_eq!(
      <Scalar as SmallValueField<i32>>::ss_mul(-100, -200),
      20000i64
    );
    assert_eq!(
      <Scalar as SmallValueField<i32>>::ss_mul(100, -200),
      -20000i64
    );
    assert_eq!(
      <Scalar as SmallValueField<i32>>::ss_mul(-100, 200),
      -20000i64
    );
  }

  #[test]
  fn test_ss_zero_edge_cases() {
    let zero = 0i32;
    let val = <Scalar as SmallValueField<i32>>::small_from_i32(12345);

    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(zero, val), 0i64);
    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(val, zero), 0i64);
  }

  #[test]
  fn test_isl_with_random() {
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    let mut rng = OsRng;
    for _ in 0..100 {
      let large = Scalar::random(&mut rng);
      let small = (rng.next_u64() % (i64::MAX as u64)) as i64;
      let small = if rng.next_u32() % 2 == 0 {
        small
      } else {
        -small
      };

      let result = <Scalar as SmallValueField<i32>>::isl_mul(small, &large);
      let expected = i64_to_field::<Scalar>(small) * large;

      assert_eq!(result, expected);
    }
  }

  #[test]
  fn test_fp_small_value_field() {
    use halo2curves::pasta::Fp;

    let a = <Fp as SmallValueField<i32>>::small_from_i32(42);
    let b = <Fp as SmallValueField<i32>>::small_from_i32(-10);

    assert_eq!(a + b, 32);
    assert_eq!(<Fp as SmallValueField<i32>>::ss_mul(a, b), -420i64);
    assert_eq!(
      <Fp as SmallValueField<i32>>::small_to_field(a),
      Fp::from(42u64)
    );
  }

  #[test]
  fn test_unreduced_mont_int_mul_add() {
    use crate::wide_limbs::SignedWideLimbs;
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    let mut rng = OsRng;
    let mut acc: SignedWideLimbs<6> = Default::default();
    let mut expected = Scalar::ZERO;

    // Sum 100 field × i64 products (mix of positive and negative)
    for i in 0..100 {
      let field = Scalar::random(&mut rng);
      let small_u = rng.next_u64() >> 32; // Keep smaller to avoid extreme overflow
      // Alternate signs for variety
      let small: i64 = if i % 2 == 0 {
        small_u as i64
      } else {
        -(small_u as i64)
      };

      <Scalar as SmallValueField<i32>>::unreduced_mont_int_mul_add(&mut acc, &field, small);
      expected += field * i64_to_field::<Scalar>(small);
    }

    let result = <Scalar as SmallValueField<i32>>::reduce_mont_int(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_unreduced_mont_mont_mul_add() {
    use crate::wide_limbs::WideLimbs;
    use ff::Field;
    use rand_core::OsRng;

    let mut rng = OsRng;
    let mut acc: WideLimbs<9> = Default::default();
    let mut expected = Scalar::ZERO;

    // Sum 100 field × field products
    for _ in 0..100 {
      let a = Scalar::random(&mut rng);
      let b = Scalar::random(&mut rng);

      <Scalar as SmallValueField<i32>>::unreduced_mont_mont_mul_add(&mut acc, &a, &b);
      expected += a * b;
    }

    let result = <Scalar as SmallValueField<i32>>::reduce_mont_mont(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_unreduced_mont_int_many_products() {
    use crate::wide_limbs::SignedWideLimbs;
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    let mut rng = OsRng;
    let mut acc: SignedWideLimbs<6> = Default::default();
    let mut expected = Scalar::ZERO;

    // Stress test: sum 2000 products (mix of positive and negative)
    for i in 0..2000 {
      let field = Scalar::random(&mut rng);
      let small_u = rng.next_u64();
      // Alternate signs for variety
      let small: i64 = if i % 3 == 0 {
        (small_u >> 1) as i64 // positive, shifted to fit in i64
      } else if i % 3 == 1 {
        -((small_u >> 1) as i64) // negative
      } else {
        0 // occasionally zero
      };

      <Scalar as SmallValueField<i32>>::unreduced_mont_int_mul_add(&mut acc, &field, small);
      expected += field * i64_to_field::<Scalar>(small);
    }

    let result = <Scalar as SmallValueField<i32>>::reduce_mont_int(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_unreduced_mont_mont_many_products() {
    use crate::wide_limbs::WideLimbs;
    use ff::Field;
    use rand_core::OsRng;

    let mut rng = OsRng;
    let mut acc: WideLimbs<9> = Default::default();
    let mut expected = Scalar::ZERO;

    // Stress test: sum 1000 products
    for _ in 0..1000 {
      let a = Scalar::random(&mut rng);
      let b = Scalar::random(&mut rng);

      <Scalar as SmallValueField<i32>>::unreduced_mont_mont_mul_add(&mut acc, &a, &b);
      expected += a * b;
    }

    let result = <Scalar as SmallValueField<i32>>::reduce_mont_mont(&acc);
    assert_eq!(result, expected);
  }
}
