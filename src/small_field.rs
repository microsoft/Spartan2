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
/// This trait defines associated types and operations for efficient arithmetic
/// when polynomial evaluations fit in native integers. The key optimization is
/// avoiding expensive field operations until absolutely necessary.
///
/// # Type Parameters
/// - `SmallValue`: Native type for original witness values (i32)
/// - `IntermediateSmallValue`: Native type for products (i64)
///
/// # Overflow Bounds (for D=2 Spartan with typical witness values)
///
/// | Step | Bound | Bits | Container |
/// |------|-------|------|-----------|
/// | Original witness values | 2²⁰ | 20 | i32 |
/// | After 3 extensions (D=2) | 2²³ | 23 | i32 ✓ |
/// | Product of two extended | 2⁴⁶ | 46 | i64 ✓ |
pub trait SmallValueField: PrimeField {
  /// Small value type (i32) for original witness values.
  /// Must support arithmetic via operator traits.
  type SmallValue: Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + Eq
    + Add<Output = Self::SmallValue>
    + Sub<Output = Self::SmallValue>
    + Neg<Output = Self::SmallValue>
    + AddAssign
    + SubAssign
    + Send
    + Sync;

  /// Intermediate type for products (i64).
  /// Used when multiplying two SmallValues together.
  type IntermediateSmallValue: Copy + Clone + Default + Debug + PartialEq + Eq + Send + Sync;

  // ===== Constructors =====

  /// Create a SmallValue from u32.
  fn small_from_u32(val: u32) -> Self::SmallValue;

  /// Create a SmallValue from i32.
  fn small_from_i32(val: i32) -> Self::SmallValue;

  /// Get the zero SmallValue.
  fn small_zero() -> Self::SmallValue;

  /// Extract the inner i32 from a SmallValue.
  fn small_inner(val: Self::SmallValue) -> i32;

  // ===== Arithmetic =====

  /// Multiply SmallValue by a small constant (for Lagrange extension).
  /// p(k) = p0 + k * diff
  fn ss_mul_const(a: Self::SmallValue, k: i32) -> Self::SmallValue;

  // ===== Core Multiplications =====

  /// ss: small × small → intermediate (i32 × i32 → i64)
  fn ss_mul(a: Self::SmallValue, b: Self::SmallValue) -> Self::IntermediateSmallValue;

  /// sl: small × large → large (i32 × field → field)
  fn sl_mul(small: Self::SmallValue, large: &Self) -> Self;

  /// isl: intermediate × large → large (i64 × field → field)
  /// This is the key operation for accumulator building.
  fn isl_mul(small: Self::IntermediateSmallValue, large: &Self) -> Self;

  // ===== Conversions =====

  /// Convert SmallValue to field element.
  fn small_to_field(val: Self::SmallValue) -> Self;

  /// Convert IntermediateSmallValue to field element.
  fn intermediate_to_field(val: Self::IntermediateSmallValue) -> Self;

  /// Try to convert a field element to SmallValue.
  /// Returns None if the value doesn't fit in i32.
  fn try_field_to_small(val: &Self) -> Option<Self::SmallValue>;
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

impl SmallValueField for halo2curves::pasta::Fp {
  type SmallValue = i32;
  type IntermediateSmallValue = i64;

  #[inline]
  fn small_from_u32(val: u32) -> i32 {
    val as i32
  }

  #[inline]
  fn small_from_i32(val: i32) -> i32 {
    val
  }

  #[inline]
  fn small_zero() -> i32 {
    0
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
}

impl SmallValueField for halo2curves::pasta::Fq {
  type SmallValue = i32;
  type IntermediateSmallValue = i64;

  #[inline]
  fn small_from_u32(val: u32) -> i32 {
    val as i32
  }

  #[inline]
  fn small_from_i32(val: i32) -> i32 {
    val
  }

  #[inline]
  fn small_zero() -> i32 {
    0
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
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{polys::multilinear::MultilinearPolynomial, provider::pasta::pallas};

  type Scalar = pallas::Scalar;

  #[test]
  fn test_small_value_field_arithmetic() {
    let a = Scalar::small_from_i32(10);
    let b = Scalar::small_from_i32(3);

    assert_eq!(a + b, 13);
    assert_eq!(a - b, 7);
    assert_eq!(-a, -10);
    assert_eq!(Scalar::ss_mul(a, b), 30i64);
    assert_eq!(Scalar::ss_mul_const(a, 5), 50);
  }

  #[test]
  fn test_small_value_field_negative() {
    let a = Scalar::small_from_i32(-5);
    let b = Scalar::small_from_i32(3);

    assert_eq!(a + b, -2);
    assert_eq!(a - b, -8);
    assert_eq!(Scalar::ss_mul(a, b), -15i64);

    let field_a = Scalar::small_to_field(a);
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
    assert_eq!(Scalar::try_field_to_small(&Scalar::from(42u64)), Some(42));
    assert_eq!(
      Scalar::try_field_to_small(&-Scalar::from(100u64)),
      Some(-100)
    );
    assert_eq!(Scalar::try_field_to_small(&Scalar::from(u64::MAX)), None);
  }

  #[test]
  fn test_isl_mul() {
    use ff::Field;
    use rand_core::OsRng;

    let large = Scalar::random(&mut OsRng);
    let small: i64 = 12345;

    let result = Scalar::isl_mul(small, &large);
    let expected = i64_to_field::<Scalar>(small) * large;

    assert_eq!(result, expected);
  }

  #[test]
  fn test_sl_mul() {
    use ff::Field;
    use rand_core::OsRng;

    let large = Scalar::random(&mut OsRng);
    let small: i32 = -999;

    let result = Scalar::sl_mul(small, &large);
    let expected = Scalar::small_to_field(small) * large;

    assert_eq!(result, expected);
  }

  #[test]
  fn test_overflow_bounds() {
    let typical_witness = 1i32 << 20;
    let extension_factor = 27i32;
    let after_extension = typical_witness * extension_factor;

    let prod = Scalar::ss_mul(after_extension, after_extension);
    assert!(prod > 0);
    assert!(prod < (1i64 << 55));
  }

  #[test]
  fn test_ss_sign_combinations() {
    assert_eq!(Scalar::ss_mul(100, 200), 20000i64);
    assert_eq!(Scalar::ss_mul(-100, -200), 20000i64);
    assert_eq!(Scalar::ss_mul(100, -200), -20000i64);
    assert_eq!(Scalar::ss_mul(-100, 200), -20000i64);
  }

  #[test]
  fn test_ss_zero_edge_cases() {
    let zero = Scalar::small_zero();
    let val = Scalar::small_from_i32(12345);

    assert_eq!(Scalar::ss_mul(zero, val), 0i64);
    assert_eq!(Scalar::ss_mul(val, zero), 0i64);
  }

  #[test]
  fn test_isl_with_random() {
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    let mut rng = OsRng;
    for _ in 0..100 {
      let large = Scalar::random(&mut rng);
      let small = (rng.next_u64() % (i64::MAX as u64)) as i64;
      let small = if rng.next_u32().is_multiple_of(2) {
        small
      } else {
        -small
      };

      let result = Scalar::isl_mul(small, &large);
      let expected = i64_to_field::<Scalar>(small) * large;

      assert_eq!(result, expected);
    }
  }

  #[test]
  fn test_fp_small_value_field() {
    use halo2curves::pasta::Fp;

    let a = Fp::small_from_i32(42);
    let b = Fp::small_from_i32(-10);

    assert_eq!(a + b, 32);
    assert_eq!(Fp::ss_mul(a, b), -420i64);
    assert_eq!(Fp::small_to_field(a), Fp::from(42u64));
  }
}
