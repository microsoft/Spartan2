// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallValueField and DelayedReduction implementations for Fp, Fq, and BN254 Fr.

use super::{
  DelayedReduction, SmallValueField, SupportsSmallI32, SupportsSmallI64, barrett, montgomery,
  i64_to_field, i128_to_field, mul_by_i64,
  limbs::{SignedWideLimbs, SubMagResult, WideLimbs, mac, mul_4_by_2_ext, mul_4_by_4_ext, mul_and_accumulate_6_by_2, sub_mag},
  try_field_to_i64,
};
use ff::PrimeField;
use halo2curves::{bn256::Fr as Bn254Fr, t256::Fq as T256Fq};

// ============================================================================
// Helper function for try_field_to_small
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

// ============================================================================
// Marker trait implementations
// ============================================================================

impl SupportsSmallI32 for halo2curves::pasta::Fp {}
impl SupportsSmallI32 for halo2curves::pasta::Fq {}
impl SupportsSmallI64 for halo2curves::pasta::Fp {}
impl SupportsSmallI64 for halo2curves::pasta::Fq {}
impl SupportsSmallI64 for Bn254Fr {}
impl SupportsSmallI32 for T256Fq {}
impl SupportsSmallI64 for T256Fq {}

// ============================================================================
// Blanket SmallValueField<i32> for all SupportsSmallI32 fields
// ============================================================================

impl<F: SupportsSmallI32 + PrimeField> SmallValueField<i32> for F {
  type IntermediateSmallValue = i64;

  #[inline]
  fn ss_mul(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
  }

  #[inline]
  fn sl_mul(small: i32, large: &Self) -> Self {
    mul_by_i64(large, small as i64)
  }

  #[inline]
  fn isl_mul(small: i64, large: &Self) -> Self {
    mul_by_i64(large, small)
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

  #[inline]
  fn small_to_intermediate(val: i32) -> i64 {
    val as i64
  }

  fn try_field_to_small(val: &Self) -> Option<i32> {
    try_field_to_small_impl(val)
  }
}

// ============================================================================
// Blanket DelayedReduction<i32> for all SupportsSmallI32 fields
// ============================================================================

impl<F: SupportsSmallI32 + PrimeField> DelayedReduction<i32> for F {
  type UnreducedFieldInt = SignedWideLimbs<6>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_small_mul_add(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    small_a: i32,
    small_b: i32,
  ) {
    // Compute product of two small values (i32 × i32 → i64)
    let product = (small_a as i64) * (small_b as i64);
    // Handle sign: accumulate into pos or neg based on sign of product
    let (target, mag) = if product >= 0 {
      (&mut acc.pos, product as u64)
    } else {
      (&mut acc.neg, (-product) as u64)
    };
    // Fused multiply-accumulate: no intermediate array
    let a = field.to_limbs();
    let (r0, c) = mac(target.0[0], a[0], mag, 0);
    let (r1, c) = mac(target.0[1], a[1], mag, c);
    let (r2, c) = mac(target.0[2], a[2], mag, c);
    let (r3, c) = mac(target.0[3], a[3], mag, c);
    // Propagate carry without multiply (just add)
    let (r4, of) = target.0[4].overflowing_add(c);
    target.0[0] = r0;
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = target.0[5].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn accumulate_field_intermediate_val(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    intermediate: i64,
  ) {
    // Handle sign: accumulate into pos or neg based on sign of intermediate
    let (target, mag) = if intermediate >= 0 {
      (&mut acc.pos, intermediate as u64)
    } else {
      (&mut acc.neg, (-intermediate) as u64)
    };
    // Fused multiply-accumulate: no intermediate array
    let a = field.to_limbs();
    let (r0, c) = mac(target.0[0], a[0], mag, 0);
    let (r1, c) = mac(target.0[1], a[1], mag, c);
    let (r2, c) = mac(target.0[2], a[2], mag, c);
    let (r3, c) = mac(target.0[3], a[3], mag, c);
    // Propagate carry without multiply (just add)
    let (r4, of) = target.0[4].overflowing_add(c);
    target.0[0] = r0;
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = target.0[5].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  ) {
    // Compute field_a × field_b as 8 limbs and add to accumulator
    let product = mul_4_by_4_ext(field_a.to_limbs(), field_b.to_limbs());
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn unreduced_field_ext_mul_add(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    ext_a: i64,
    ext_b: i64,
  ) {
    let is_neg = (ext_a < 0) ^ (ext_b < 0);
    let mag_a = ext_a.unsigned_abs() as u128;
    let mag_b = ext_b.unsigned_abs() as u128;
    let tmp6 = mul_4_by_2_ext(field.to_limbs(), mag_a);
    let target = if is_neg { &mut acc.neg } else { &mut acc.pos };
    mul_and_accumulate_6_by_2::<6>(&mut target.0, &tmp6, mag_b);
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    match sub_mag::<6>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => F::from_limbs(barrett::barrett_reduce_6::<F>(&mag)),
      SubMagResult::Negative(mag) => -F::from_limbs(barrett::barrett_reduce_6::<F>(&mag)),
    }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    F::from_limbs(montgomery::montgomery_reduce_9::<F>(&acc.0))
  }

  type UnreducedField = Self;

  #[inline(always)]
  fn reduce_field_int_to_unreduced(acc: &Self::UnreducedFieldInt) -> Self {
    Self::reduce_field_int(acc)
  }

  #[inline(always)]
  fn to_unreduced(&self) -> Self {
    *self
  }

  #[inline(always)]
  fn accumulate_raw_field_field_products(acc: &mut Self::UnreducedFieldField, a: &Self, b: &Self) {
    Self::unreduced_field_field_mul_add(acc, a, b);
  }

  #[inline(always)]
  fn reduce_unreduced_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self::reduce_field_field(acc)
  }

  #[inline(always)]
  fn accumulate_field_small_small_products(
    acc: &mut Self::UnreducedFieldInt,
    e: &Self,
    small_a: i32,
    small_b: i32,
  ) {
    Self::unreduced_field_small_mul_add(acc, e, small_a, small_b);
  }

  #[inline(always)]
  fn reduce_raw_field_int_to_unreduced(acc: &Self::UnreducedFieldInt) -> Self {
    Self::reduce_field_int(acc)
  }
}

// ============================================================================
// Blanket SmallValueField<i64> for all SupportsSmallI64 fields
// ============================================================================

impl<F: SupportsSmallI64 + PrimeField> SmallValueField<i64> for F {
  type IntermediateSmallValue = i128;

  #[inline]
  fn ss_mul(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }

  #[inline]
  fn sl_mul(small: i64, large: &Self) -> Self {
    mul_by_i64(large, small)
  }

  #[inline]
  fn isl_mul(small: i128, large: &Self) -> Self {
    if small == 0 {
      return F::ZERO;
    }
    let (is_neg, mag) = if small >= 0 {
      (false, small as u128)
    } else {
      (true, (-small) as u128)
    };
    // mul_4_by_2_ext produces 6 limbs, use barrett_reduce_6 directly (no padding)
    let product = mul_4_by_2_ext(large.to_limbs(), mag);
    let result = Self::from_limbs(barrett::barrett_reduce_6::<F>(&product));
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

  #[inline]
  fn small_to_intermediate(val: i64) -> i128 {
    val as i128
  }

  fn try_field_to_small(val: &Self) -> Option<i64> {
    try_field_to_i64(val)
  } 
}

// ============================================================================
// Blanket DelayedReduction<i64> for all SupportsSmallI64 fields
// ============================================================================

impl<F: SupportsSmallI64 + PrimeField> DelayedReduction<i64> for F {
  type UnreducedFieldInt = SignedWideLimbs<7>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_small_mul_add(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    small_a: i64,
    small_b: i64,
  ) {
    // Compute product of two small values (i64 × i64 → i128)
    let product = (small_a as i128) * (small_b as i128);
    let (target, mag) = if product >= 0 {
      (&mut acc.pos, product as u128)
    } else {
      (&mut acc.neg, (-product) as u128)
    };
    // Fused 4×2 multiply-accumulate: two passes at different offsets
    let a = field.to_limbs();
    let b_lo = mag as u64;
    let b_hi = (mag >> 64) as u64;

    // Pass 1: multiply by b_lo at offset 0
    let (r0, c) = mac(target.0[0], a[0], b_lo, 0);
    let (r1, c) = mac(target.0[1], a[1], b_lo, c);
    let (r2, c) = mac(target.0[2], a[2], b_lo, c);
    let (r3, c) = mac(target.0[3], a[3], b_lo, c);
    // Propagate carry without multiply (just add)
    let (r4, of1) = target.0[4].overflowing_add(c);
    let c1 = of1 as u64;
    target.0[0] = r0;

    // Pass 2: multiply by b_hi at offset 1 (add to r1..r5)
    let (r1, c) = mac(r1, a[0], b_hi, 0);
    let (r2, c) = mac(r2, a[1], b_hi, c);
    let (r3, c) = mac(r3, a[2], b_hi, c);
    let (r4, c) = mac(r4, a[3], b_hi, c);
    // Add both carries (c from pass 2, c1 from pass 1) into position 5
    let (r5, c) = mac(target.0[5], c1, 1, c);
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = r5;
    // Propagate final carry through remaining limbs (just add)
    target.0[6] = target.0[6].wrapping_add(c);
  }

  #[inline(always)]
  fn accumulate_field_intermediate_val(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    intermediate: i128,
  ) {
    let (target, mag) = if intermediate >= 0 {
      (&mut acc.pos, intermediate as u128)
    } else {
      (&mut acc.neg, (-intermediate) as u128)
    };
    // Fused 4×2 multiply-accumulate: two passes at different offsets
    let a = field.to_limbs();
    let b_lo = mag as u64;
    let b_hi = (mag >> 64) as u64;

    // Pass 1: multiply by b_lo at offset 0
    let (r0, c) = mac(target.0[0], a[0], b_lo, 0);
    let (r1, c) = mac(target.0[1], a[1], b_lo, c);
    let (r2, c) = mac(target.0[2], a[2], b_lo, c);
    let (r3, c) = mac(target.0[3], a[3], b_lo, c);
    // Propagate carry without multiply (just add)
    let (r4, of1) = target.0[4].overflowing_add(c);
    let c1 = of1 as u64;
    target.0[0] = r0;

    // Pass 2: multiply by b_hi at offset 1 (add to r1..r5)
    let (r1, c) = mac(r1, a[0], b_hi, 0);
    let (r2, c) = mac(r2, a[1], b_hi, c);
    let (r3, c) = mac(r3, a[2], b_hi, c);
    let (r4, c) = mac(r4, a[3], b_hi, c);
    // Add both carries (c from pass 2, c1 from pass 1) into position 5
    let (r5, c) = mac(target.0[5], c1, 1, c);
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = r5;
    // Propagate final carry through remaining limbs (just add)
    target.0[6] = target.0[6].wrapping_add(c);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  ) {
    let product = mul_4_by_4_ext(field_a.to_limbs(), field_b.to_limbs());
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn unreduced_field_ext_mul_add(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    ext_a: i128,
    ext_b: i128,
  ) {
    let is_neg = (ext_a < 0) ^ (ext_b < 0);
    let mag_a = ext_a.unsigned_abs();
    let mag_b = ext_b.unsigned_abs();
    let tmp6 = mul_4_by_2_ext(field.to_limbs(), mag_a);
    let target = if is_neg { &mut acc.neg } else { &mut acc.pos };
    mul_and_accumulate_6_by_2::<7>(&mut target.0, &tmp6, mag_b);
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    match sub_mag::<7>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => F::from_limbs(barrett::barrett_reduce_7::<F>(&mag)),
      SubMagResult::Negative(mag) => -F::from_limbs(barrett::barrett_reduce_7::<F>(&mag)),
    }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    F::from_limbs(montgomery::montgomery_reduce_9::<F>(&acc.0))
  }

  type UnreducedField = Self;

  #[inline(always)]
  fn reduce_field_int_to_unreduced(acc: &Self::UnreducedFieldInt) -> Self {
    Self::reduce_field_int(acc)
  }

  #[inline(always)]
  fn to_unreduced(&self) -> Self {
    *self
  }

  #[inline(always)]
  fn accumulate_raw_field_field_products(acc: &mut Self::UnreducedFieldField, a: &Self, b: &Self) {
    Self::unreduced_field_field_mul_add(acc, a, b);
  }

  #[inline(always)]
  fn reduce_unreduced_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self::reduce_field_field(acc)
  }

  #[inline(always)]
  fn accumulate_field_small_small_products(
    acc: &mut Self::UnreducedFieldInt,
    e: &Self,
    small_a: i64,
    small_b: i64,
  ) {
    Self::unreduced_field_small_mul_add(acc, e, small_a, small_b);
  }

  #[inline(always)]
  fn reduce_raw_field_int_to_unreduced(acc: &Self::UnreducedFieldInt) -> Self {
    Self::reduce_field_int(acc)
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
    let a: i32 = 10;
    let b: i32 = 3;

    assert_eq!(a + b, 13);
    assert_eq!(a - b, 7);
    assert_eq!(-a, -10);
    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(a, b), 30i64);
    assert_eq!(a * 5, 50); // ss_mul_const is just native multiplication
  }

  #[test]
  fn test_small_value_field_negative() {
    let a: i32 = -5;
    let b: i32 = 3;

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
  fn test_try_field_to_i64_roundtrip() {
    use super::super::try_field_to_i64;

    // Test positive values
    for val in [0i64, 1, 100, 1_000_000, i64::MAX / 2, i64::MAX] {
      let field: Scalar = i64_to_field(val);
      let back = try_field_to_i64(&field).expect("should fit");
      assert_eq!(back, val, "roundtrip failed for {}", val);
    }

    // Test negative values
    for val in [-1i64, -100, -1_000_000, i64::MIN / 2, i64::MIN + 1] {
      let field: Scalar = i64_to_field(val);
      let back = try_field_to_i64(&field).expect("should fit");
      assert_eq!(back, val, "roundtrip failed for {}", val);
    }

    // Test i64::MIN separately (edge case)
    let field: Scalar = i64_to_field(i64::MIN);
    let back = try_field_to_i64(&field).expect("should fit");
    assert_eq!(back, i64::MIN, "roundtrip failed for i64::MIN");

    // Test values that don't fit in i64
    let too_large = Scalar::from(u64::MAX) + Scalar::from(1u64);
    assert!(try_field_to_i64(&too_large).is_none());
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
    let val = 12345i32;

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
      let small = if rng.next_u32().is_multiple_of(2) {
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

    let a: i32 = 42;
    let b: i32 = -10;

    assert_eq!(a + b, 32);
    assert_eq!(<Fp as SmallValueField<i32>>::ss_mul(a, b), -420i64);
    assert_eq!(
      <Fp as SmallValueField<i32>>::small_to_field(a),
      Fp::from(42u64)
    );
  }

  #[test]
  fn test_unreduced_field_small_mul_add() {
    use crate::small_field::limbs::SignedWideLimbs;
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    let mut rng = OsRng;
    let mut acc: SignedWideLimbs<6> = Default::default();
    let mut expected = Scalar::ZERO;

    // Sum 100 field × (small_a × small_b) products (mix of positive and negative)
    for i in 0..100 {
      let field = Scalar::random(&mut rng);
      let small_a_u = (rng.next_u64() >> 48) as i32; // Keep small to fit in i32
      let small_b_u = (rng.next_u64() >> 48) as i32;
      // Alternate signs for variety
      let small_a: i32 = if i % 2 == 0 { small_a_u } else { -small_a_u };
      let small_b: i32 = if i % 3 == 0 { small_b_u } else { -small_b_u };

      <Scalar as DelayedReduction<i32>>::unreduced_field_small_mul_add(&mut acc, &field, small_a, small_b);
      let product = (small_a as i64) * (small_b as i64);
      expected += field * i64_to_field::<Scalar>(product);
    }

    let result = <Scalar as DelayedReduction<i32>>::reduce_field_int(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_unreduced_field_field_mul_add() {
    use crate::small_field::limbs::WideLimbs;
    use ff::Field;
    use rand_core::OsRng;

    let mut rng = OsRng;
    let mut acc: WideLimbs<9> = Default::default();
    let mut expected = Scalar::ZERO;

    // Sum 100 field × field products
    for _ in 0..100 {
      let a = Scalar::random(&mut rng);
      let b = Scalar::random(&mut rng);

      <Scalar as DelayedReduction<i32>>::unreduced_field_field_mul_add(&mut acc, &a, &b);
      expected += a * b;
    }

    let result = <Scalar as DelayedReduction<i32>>::reduce_field_field(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_unreduced_field_small_many_products() {
    use crate::small_field::limbs::SignedWideLimbs;
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    let mut rng = OsRng;
    let mut acc: SignedWideLimbs<6> = Default::default();
    let mut expected = Scalar::ZERO;

    // Stress test: sum 2000 products (mix of positive and negative)
    for i in 0..2000 {
      let field = Scalar::random(&mut rng);
      let small_a_u = (rng.next_u64() >> 48) as i32;
      let small_b_u = (rng.next_u64() >> 48) as i32;
      // Alternate signs for variety
      let (small_a, small_b): (i32, i32) = if i % 3 == 0 {
        (small_a_u, small_b_u) // positive × positive
      } else if i % 3 == 1 {
        (-small_a_u, small_b_u) // negative × positive
      } else {
        (0, small_b_u) // zero × positive
      };

      <Scalar as DelayedReduction<i32>>::unreduced_field_small_mul_add(&mut acc, &field, small_a, small_b);
      let product = (small_a as i64) * (small_b as i64);
      expected += field * i64_to_field::<Scalar>(product);
    }

    let result = <Scalar as DelayedReduction<i32>>::reduce_field_int(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_unreduced_field_field_many_products() {
    use crate::small_field::limbs::WideLimbs;
    use ff::Field;
    use rand_core::OsRng;

    let mut rng = OsRng;
    let mut acc: WideLimbs<9> = Default::default();
    let mut expected = Scalar::ZERO;

    // Stress test: sum 1000 products
    for _ in 0..1000 {
      let a = Scalar::random(&mut rng);
      let b = Scalar::random(&mut rng);

      <Scalar as DelayedReduction<i32>>::unreduced_field_field_mul_add(&mut acc, &a, &b);
      expected += a * b;
    }

    let result = <Scalar as DelayedReduction<i32>>::reduce_field_field(&acc);
    assert_eq!(result, expected);
  }
}
