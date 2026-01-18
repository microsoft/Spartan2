// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallValueField and DelayedReduction implementations for Fp, Fq, and BN254 Fr.

use super::{
  DelayedReduction, SmallValueField, barrett, i64_to_field, i128_to_field,
  limbs::{SignedWideLimbs, SubMagResult, WideLimbs, mac, mul_4_by_2_ext, mul_4_by_4_ext, sub_mag},
};
use ff::PrimeField;
use halo2curves::bn256::Fr as Bn254Fr;

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
// SmallValueField<i32> for Fp
// ============================================================================

impl SmallValueField<i32> for halo2curves::pasta::Fp {
  type IntermediateSmallValue = i64;

  #[inline]
  fn ss_mul(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
  }

  #[inline]
  fn sl_mul(small: i32, large: &Self) -> Self {
    barrett::mul_by_i64(large, small as i64)
  }

  #[inline]
  fn isl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_by_i64(large, small)
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
// DelayedReduction<i32> for Fp
// ============================================================================

impl DelayedReduction<i32> for halo2curves::pasta::Fp {
  type UnreducedFieldInt = SignedWideLimbs<6>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_int_mul_add(acc: &mut Self::UnreducedFieldInt, field: &Self, small: i64) {
    // Handle sign: accumulate into pos or neg based on sign of small
    let (target, mag) = if small >= 0 {
      (&mut acc.pos, small as u64)
    } else {
      (&mut acc.neg, (-small) as u64)
    };
    // Fused multiply-accumulate: no intermediate array
    let a = &field.0;
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
    let product = mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn unreduced_field_int_mul_add_batch4(
    accs: [&mut Self::UnreducedFieldInt; 4],
    field: &Self,
    smalls: [i64; 4],
  ) {
    // Batched ILP version: interleave 4 independent carry chains
    batch_unreduced_field_int_mul_add_x4_fp(accs, field, smalls);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add_batch4(
    accs: [&mut Self::UnreducedFieldField; 4],
    a: [&Self; 4],
    b: [&Self; 4],
  ) {
    batch_unreduced_field_field_mul_add_x4(
      accs,
      [&a[0].0, &a[1].0, &a[2].0, &a[3].0],
      [&b[0].0, &b[1].0, &b[2].0, &b[3].0],
    );
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    match sub_mag::<6>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => Self(barrett::barrett_reduce_6_fp(&mag)),
      SubMagResult::Negative(mag) => -Self(barrett::barrett_reduce_6_fp(&mag)),
    }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self(barrett::montgomery_reduce_9_fp(&acc.0))
  }
}

// ============================================================================
// SmallValueField<i64> for Fp
// ============================================================================

impl SmallValueField<i64> for halo2curves::pasta::Fp {
  type IntermediateSmallValue = i128;

  #[inline]
  fn ss_mul(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }

  #[inline]
  fn sl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_by_i64(large, small)
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
    // mul_4_by_2_ext produces 6 limbs, use barrett_reduce_6 directly (no padding)
    let product = mul_4_by_2_ext(&large.0, mag);
    let result = Self(barrett::barrett_reduce_6_fp(&product));
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
}

// ============================================================================
// DelayedReduction<i64> for Fp
// ============================================================================

impl DelayedReduction<i64> for halo2curves::pasta::Fp {
  type UnreducedFieldInt = SignedWideLimbs<8>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_int_mul_add(acc: &mut Self::UnreducedFieldInt, field: &Self, small: i128) {
    let (target, mag) = if small >= 0 {
      (&mut acc.pos, small as u128)
    } else {
      (&mut acc.neg, (-small) as u128)
    };
    // Fused 4×2 multiply-accumulate: two passes at different offsets
    let a = &field.0;
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
    let (r6, of) = target.0[6].overflowing_add(c);
    target.0[6] = r6;
    target.0[7] = target.0[7].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  ) {
    let product = mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add_batch4(
    accs: [&mut Self::UnreducedFieldField; 4],
    a: [&Self; 4],
    b: [&Self; 4],
  ) {
    batch_unreduced_field_field_mul_add_x4(
      accs,
      [&a[0].0, &a[1].0, &a[2].0, &a[3].0],
      [&b[0].0, &b[1].0, &b[2].0, &b[3].0],
    );
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    match sub_mag::<8>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => Self(barrett::barrett_reduce_8_fp(&mag)),
      SubMagResult::Negative(mag) => -Self(barrett::barrett_reduce_8_fp(&mag)),
    }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self(barrett::montgomery_reduce_9_fp(&acc.0))
  }
}

// ============================================================================
// SmallValueField<i32> for Fq
// ============================================================================

impl SmallValueField<i32> for halo2curves::pasta::Fq {
  type IntermediateSmallValue = i64;

  #[inline]
  fn ss_mul(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
  }

  #[inline]
  fn sl_mul(small: i32, large: &Self) -> Self {
    barrett::mul_by_i64(large, small as i64)
  }

  #[inline]
  fn isl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_by_i64(large, small)
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
// DelayedReduction<i32> for Fq
// ============================================================================

impl DelayedReduction<i32> for halo2curves::pasta::Fq {
  type UnreducedFieldInt = SignedWideLimbs<6>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_int_mul_add(acc: &mut Self::UnreducedFieldInt, field: &Self, small: i64) {
    // Handle sign: accumulate into pos or neg based on sign of small
    let (target, mag) = if small >= 0 {
      (&mut acc.pos, small as u64)
    } else {
      (&mut acc.neg, (-small) as u64)
    };
    // Fused multiply-accumulate: no intermediate array
    let a = &field.0;
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
    let product = mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn unreduced_field_int_mul_add_batch4(
    accs: [&mut Self::UnreducedFieldInt; 4],
    field: &Self,
    smalls: [i64; 4],
  ) {
    // Batched ILP version: interleave 4 independent carry chains
    batch_unreduced_field_int_mul_add_x4_fq(accs, field, smalls);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add_batch4(
    accs: [&mut Self::UnreducedFieldField; 4],
    a: [&Self; 4],
    b: [&Self; 4],
  ) {
    batch_unreduced_field_field_mul_add_x4(
      accs,
      [&a[0].0, &a[1].0, &a[2].0, &a[3].0],
      [&b[0].0, &b[1].0, &b[2].0, &b[3].0],
    );
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    match sub_mag::<6>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => Self(barrett::barrett_reduce_6_fq(&mag)),
      SubMagResult::Negative(mag) => -Self(barrett::barrett_reduce_6_fq(&mag)),
    }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self(barrett::montgomery_reduce_9_fq(&acc.0))
  }
}

// ============================================================================
// SmallValueField<i64> for Fq
// ============================================================================

impl SmallValueField<i64> for halo2curves::pasta::Fq {
  type IntermediateSmallValue = i128;

  #[inline]
  fn ss_mul(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }

  #[inline]
  fn sl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_by_i64(large, small)
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
    // mul_4_by_2_ext produces 6 limbs, use barrett_reduce_6 directly (no padding)
    let product = mul_4_by_2_ext(&large.0, mag);
    let result = Self(barrett::barrett_reduce_6_fq(&product));
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
}

// ============================================================================
// DelayedReduction<i64> for Fq
// ============================================================================

impl DelayedReduction<i64> for halo2curves::pasta::Fq {
  type UnreducedFieldInt = SignedWideLimbs<8>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_int_mul_add(acc: &mut Self::UnreducedFieldInt, field: &Self, small: i128) {
    let (target, mag) = if small >= 0 {
      (&mut acc.pos, small as u128)
    } else {
      (&mut acc.neg, (-small) as u128)
    };
    // Fused 4×2 multiply-accumulate: two passes at different offsets
    let a = &field.0;
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
    // Add both carries (c1 from pass 1, c from pass 2) into position 5
    let (r5, c) = mac(target.0[5], c1, 1, c);
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = r5;
    // Propagate final carry through remaining limbs (just add)
    let (r6, of) = target.0[6].overflowing_add(c);
    target.0[6] = r6;
    target.0[7] = target.0[7].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  ) {
    let product = mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add_batch4(
    accs: [&mut Self::UnreducedFieldField; 4],
    a: [&Self; 4],
    b: [&Self; 4],
  ) {
    batch_unreduced_field_field_mul_add_x4(
      accs,
      [&a[0].0, &a[1].0, &a[2].0, &a[3].0],
      [&b[0].0, &b[1].0, &b[2].0, &b[3].0],
    );
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    match sub_mag::<8>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => Self(barrett::barrett_reduce_8_fq(&mag)),
      SubMagResult::Negative(mag) => -Self(barrett::barrett_reduce_8_fq(&mag)),
    }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self(barrett::montgomery_reduce_9_fq(&acc.0))
  }
}

// ============================================================================
// SmallValueField<i64> for BN254 Fr
// ============================================================================

impl SmallValueField<i64> for Bn254Fr {
  type IntermediateSmallValue = i128;

  #[inline]
  fn ss_mul(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }

  #[inline]
  fn sl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_by_i64(large, small)
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
    let product = mul_4_by_2_ext(&large.0, mag);
    let result = Self(barrett::barrett_reduce_6_bn254_fr(&product));
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
}

// ============================================================================
// DelayedReduction<i64> for BN254 Fr
// ============================================================================

impl DelayedReduction<i64> for Bn254Fr {
  type UnreducedFieldInt = SignedWideLimbs<8>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_int_mul_add(acc: &mut Self::UnreducedFieldInt, field: &Self, small: i128) {
    let (target, mag) = if small >= 0 {
      (&mut acc.pos, small as u128)
    } else {
      (&mut acc.neg, (-small) as u128)
    };
    let a = &field.0;
    let b_lo = mag as u64;
    let b_hi = (mag >> 64) as u64;

    // Pass 1: multiply by b_lo at offset 0
    let (r0, c) = mac(target.0[0], a[0], b_lo, 0);
    let (r1, c) = mac(target.0[1], a[1], b_lo, c);
    let (r2, c) = mac(target.0[2], a[2], b_lo, c);
    let (r3, c) = mac(target.0[3], a[3], b_lo, c);
    let (r4, of1) = target.0[4].overflowing_add(c);
    let c1 = of1 as u64;
    target.0[0] = r0;

    // Pass 2: multiply by b_hi at offset 1
    let (r1, c) = mac(r1, a[0], b_hi, 0);
    let (r2, c) = mac(r2, a[1], b_hi, c);
    let (r3, c) = mac(r3, a[2], b_hi, c);
    let (r4, c) = mac(r4, a[3], b_hi, c);
    let (r5, c) = mac(target.0[5], c1, 1, c);
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = r5;
    let (r6, of) = target.0[6].overflowing_add(c);
    target.0[6] = r6;
    target.0[7] = target.0[7].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  ) {
    let product = mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add_batch4(
    accs: [&mut Self::UnreducedFieldField; 4],
    a: [&Self; 4],
    b: [&Self; 4],
  ) {
    batch_unreduced_field_field_mul_add_x4(
      accs,
      [&a[0].0, &a[1].0, &a[2].0, &a[3].0],
      [&b[0].0, &b[1].0, &b[2].0, &b[3].0],
    );
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    match sub_mag::<8>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => Self(barrett::barrett_reduce_8_bn254_fr(&mag)),
      SubMagResult::Negative(mag) => -Self(barrett::barrett_reduce_8_bn254_fr(&mag)),
    }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self(barrett::montgomery_reduce_9_bn254_fr(&acc.0))
  }
}

// ============================================================================
// Batched ILP Operations
// ============================================================================

/// Internal helper for batched MAC operations (works for any 4-limb field).
///
/// Interleaves 4 independent carry chains for better ILP on AArch64.
#[inline(always)]
fn batch_mac_4limb_x4(targets: [&mut WideLimbs<6>; 4], a: &[u64; 4], mags: [u64; 4]) {
  // Limb 0: 4 independent macs (ILP - CPU can overlap these)
  let (r0_0, c0) = mac(targets[0].0[0], a[0], mags[0], 0);
  let (r1_0, c1) = mac(targets[1].0[0], a[0], mags[1], 0);
  let (r2_0, c2) = mac(targets[2].0[0], a[0], mags[2], 0);
  let (r3_0, c3) = mac(targets[3].0[0], a[0], mags[3], 0);

  // Limb 1: 4 independent macs (each uses its own carry)
  let (r0_1, c0) = mac(targets[0].0[1], a[1], mags[0], c0);
  let (r1_1, c1) = mac(targets[1].0[1], a[1], mags[1], c1);
  let (r2_1, c2) = mac(targets[2].0[1], a[1], mags[2], c2);
  let (r3_1, c3) = mac(targets[3].0[1], a[1], mags[3], c3);

  // Limb 2: 4 independent macs
  let (r0_2, c0) = mac(targets[0].0[2], a[2], mags[0], c0);
  let (r1_2, c1) = mac(targets[1].0[2], a[2], mags[1], c1);
  let (r2_2, c2) = mac(targets[2].0[2], a[2], mags[2], c2);
  let (r3_2, c3) = mac(targets[3].0[2], a[2], mags[3], c3);

  // Limb 3: 4 independent macs
  let (r0_3, c0) = mac(targets[0].0[3], a[3], mags[0], c0);
  let (r1_3, c1) = mac(targets[1].0[3], a[3], mags[1], c1);
  let (r2_3, c2) = mac(targets[2].0[3], a[3], mags[2], c2);
  let (r3_3, c3) = mac(targets[3].0[3], a[3], mags[3], c3);

  // Final carry propagation for all 4 (still independent)
  let (r0_4, of0) = targets[0].0[4].overflowing_add(c0);
  let (r1_4, of1) = targets[1].0[4].overflowing_add(c1);
  let (r2_4, of2) = targets[2].0[4].overflowing_add(c2);
  let (r3_4, of3) = targets[3].0[4].overflowing_add(c3);

  // Store results for accumulator 0
  targets[0].0[0] = r0_0;
  targets[0].0[1] = r0_1;
  targets[0].0[2] = r0_2;
  targets[0].0[3] = r0_3;
  targets[0].0[4] = r0_4;
  targets[0].0[5] = targets[0].0[5].wrapping_add(of0 as u64);

  // Store results for accumulator 1
  targets[1].0[0] = r1_0;
  targets[1].0[1] = r1_1;
  targets[1].0[2] = r1_2;
  targets[1].0[3] = r1_3;
  targets[1].0[4] = r1_4;
  targets[1].0[5] = targets[1].0[5].wrapping_add(of1 as u64);

  // Store results for accumulator 2
  targets[2].0[0] = r2_0;
  targets[2].0[1] = r2_1;
  targets[2].0[2] = r2_2;
  targets[2].0[3] = r2_3;
  targets[2].0[4] = r2_4;
  targets[2].0[5] = targets[2].0[5].wrapping_add(of2 as u64);

  // Store results for accumulator 3
  targets[3].0[0] = r3_0;
  targets[3].0[1] = r3_1;
  targets[3].0[2] = r3_2;
  targets[3].0[3] = r3_3;
  targets[3].0[4] = r3_4;
  targets[3].0[5] = targets[3].0[5].wrapping_add(of3 as u64);
}

/// Batch 4 independent field×int multiply-accumulates for instruction-level parallelism.
///
/// On AArch64 (M1/M2), this allows the CPU to overlap mul/umulh latencies across
/// 4 independent carry chains, significantly improving throughput compared to
/// processing one accumulation at a time.
///
/// # Safety
/// - All 4 accumulators must be valid and non-overlapping
/// - This is for Fq (4 limbs, 256-bit field)
#[inline(always)]
pub fn batch_unreduced_field_int_mul_add_x4_fq(
  accs: [&mut SignedWideLimbs<6>; 4],
  field: &halo2curves::pasta::Fq,
  smalls: [i64; 4],
) {
  // Destructure to get 4 independent mutable references (satisfies borrow checker)
  let [acc0, acc1, acc2, acc3] = accs;

  // Prepare targets and magnitudes for each of the 4 operations
  let (target0, mag0) = if smalls[0] >= 0 {
    (&mut acc0.pos, smalls[0] as u64)
  } else {
    (&mut acc0.neg, (-smalls[0]) as u64)
  };
  let (target1, mag1) = if smalls[1] >= 0 {
    (&mut acc1.pos, smalls[1] as u64)
  } else {
    (&mut acc1.neg, (-smalls[1]) as u64)
  };
  let (target2, mag2) = if smalls[2] >= 0 {
    (&mut acc2.pos, smalls[2] as u64)
  } else {
    (&mut acc2.neg, (-smalls[2]) as u64)
  };
  let (target3, mag3) = if smalls[3] >= 0 {
    (&mut acc3.pos, smalls[3] as u64)
  } else {
    (&mut acc3.neg, (-smalls[3]) as u64)
  };

  let a = &field.0;

  // Limb 0: 4 independent macs (ILP - CPU can overlap these)
  let (r0_0, c0) = mac(target0.0[0], a[0], mag0, 0);
  let (r1_0, c1) = mac(target1.0[0], a[0], mag1, 0);
  let (r2_0, c2) = mac(target2.0[0], a[0], mag2, 0);
  let (r3_0, c3) = mac(target3.0[0], a[0], mag3, 0);

  // Limb 1: 4 independent macs (each uses its own carry)
  let (r0_1, c0) = mac(target0.0[1], a[1], mag0, c0);
  let (r1_1, c1) = mac(target1.0[1], a[1], mag1, c1);
  let (r2_1, c2) = mac(target2.0[1], a[1], mag2, c2);
  let (r3_1, c3) = mac(target3.0[1], a[1], mag3, c3);

  // Limb 2: 4 independent macs
  let (r0_2, c0) = mac(target0.0[2], a[2], mag0, c0);
  let (r1_2, c1) = mac(target1.0[2], a[2], mag1, c1);
  let (r2_2, c2) = mac(target2.0[2], a[2], mag2, c2);
  let (r3_2, c3) = mac(target3.0[2], a[2], mag3, c3);

  // Limb 3: 4 independent macs
  let (r0_3, c0) = mac(target0.0[3], a[3], mag0, c0);
  let (r1_3, c1) = mac(target1.0[3], a[3], mag1, c1);
  let (r2_3, c2) = mac(target2.0[3], a[3], mag2, c2);
  let (r3_3, c3) = mac(target3.0[3], a[3], mag3, c3);

  // Final carry propagation for all 4 (still independent)
  let (r0_4, of0) = target0.0[4].overflowing_add(c0);
  let (r1_4, of1) = target1.0[4].overflowing_add(c1);
  let (r2_4, of2) = target2.0[4].overflowing_add(c2);
  let (r3_4, of3) = target3.0[4].overflowing_add(c3);

  // Store results for accumulator 0
  target0.0[0] = r0_0;
  target0.0[1] = r0_1;
  target0.0[2] = r0_2;
  target0.0[3] = r0_3;
  target0.0[4] = r0_4;
  target0.0[5] = target0.0[5].wrapping_add(of0 as u64);

  // Store results for accumulator 1
  target1.0[0] = r1_0;
  target1.0[1] = r1_1;
  target1.0[2] = r1_2;
  target1.0[3] = r1_3;
  target1.0[4] = r1_4;
  target1.0[5] = target1.0[5].wrapping_add(of1 as u64);

  // Store results for accumulator 2
  target2.0[0] = r2_0;
  target2.0[1] = r2_1;
  target2.0[2] = r2_2;
  target2.0[3] = r2_3;
  target2.0[4] = r2_4;
  target2.0[5] = target2.0[5].wrapping_add(of2 as u64);

  // Store results for accumulator 3
  target3.0[0] = r3_0;
  target3.0[1] = r3_1;
  target3.0[2] = r3_2;
  target3.0[3] = r3_3;
  target3.0[4] = r3_4;
  target3.0[5] = target3.0[5].wrapping_add(of3 as u64);
}

/// Batch 4 independent field×int multiply-accumulates for Fp.
#[inline(always)]
pub fn batch_unreduced_field_int_mul_add_x4_fp(
  accs: [&mut SignedWideLimbs<6>; 4],
  field: &halo2curves::pasta::Fp,
  smalls: [i64; 4],
) {
  // Destructure to get 4 independent mutable references (satisfies borrow checker)
  let [acc0, acc1, acc2, acc3] = accs;

  // Prepare targets and magnitudes for each of the 4 operations
  let (target0, mag0) = if smalls[0] >= 0 {
    (&mut acc0.pos, smalls[0] as u64)
  } else {
    (&mut acc0.neg, (-smalls[0]) as u64)
  };
  let (target1, mag1) = if smalls[1] >= 0 {
    (&mut acc1.pos, smalls[1] as u64)
  } else {
    (&mut acc1.neg, (-smalls[1]) as u64)
  };
  let (target2, mag2) = if smalls[2] >= 0 {
    (&mut acc2.pos, smalls[2] as u64)
  } else {
    (&mut acc2.neg, (-smalls[2]) as u64)
  };
  let (target3, mag3) = if smalls[3] >= 0 {
    (&mut acc3.pos, smalls[3] as u64)
  } else {
    (&mut acc3.neg, (-smalls[3]) as u64)
  };

  batch_mac_4limb_x4(
    [target0, target1, target2, target3],
    &field.0,
    [mag0, mag1, mag2, mag3],
  );
}

/// Batch 4 independent field×field multiply-accumulates for ILP optimization.
///
/// Computes 4 products in parallel (mul_4_by_4_ext) and adds them to 4 separate
/// 9-limb accumulators with interleaved carry propagation for better ILP on AArch64.
#[inline(always)]
pub fn batch_unreduced_field_field_mul_add_x4(
  accs: [&mut WideLimbs<9>; 4],
  a: [&[u64; 4]; 4],
  b: [&[u64; 4]; 4],
) {
  let [acc0, acc1, acc2, acc3] = accs;

  // Compute 4 products (ILP: these can be computed independently)
  let prod0 = mul_4_by_4_ext(a[0], b[0]);
  let prod1 = mul_4_by_4_ext(a[1], b[1]);
  let prod2 = mul_4_by_4_ext(a[2], b[2]);
  let prod3 = mul_4_by_4_ext(a[3], b[3]);

  // Add products to accumulators with interleaved carry propagation
  // Limb 0
  let sum0 = (acc0.0[0] as u128) + (prod0[0] as u128);
  let sum1 = (acc1.0[0] as u128) + (prod1[0] as u128);
  let sum2 = (acc2.0[0] as u128) + (prod2[0] as u128);
  let sum3 = (acc3.0[0] as u128) + (prod3[0] as u128);
  acc0.0[0] = sum0 as u64;
  acc1.0[0] = sum1 as u64;
  acc2.0[0] = sum2 as u64;
  acc3.0[0] = sum3 as u64;
  let (mut c0, mut c1, mut c2, mut c3) = (sum0 >> 64, sum1 >> 64, sum2 >> 64, sum3 >> 64);

  // Limb 1
  let sum0 = (acc0.0[1] as u128) + (prod0[1] as u128) + c0;
  let sum1 = (acc1.0[1] as u128) + (prod1[1] as u128) + c1;
  let sum2 = (acc2.0[1] as u128) + (prod2[1] as u128) + c2;
  let sum3 = (acc3.0[1] as u128) + (prod3[1] as u128) + c3;
  acc0.0[1] = sum0 as u64;
  acc1.0[1] = sum1 as u64;
  acc2.0[1] = sum2 as u64;
  acc3.0[1] = sum3 as u64;
  (c0, c1, c2, c3) = (sum0 >> 64, sum1 >> 64, sum2 >> 64, sum3 >> 64);

  // Limb 2
  let sum0 = (acc0.0[2] as u128) + (prod0[2] as u128) + c0;
  let sum1 = (acc1.0[2] as u128) + (prod1[2] as u128) + c1;
  let sum2 = (acc2.0[2] as u128) + (prod2[2] as u128) + c2;
  let sum3 = (acc3.0[2] as u128) + (prod3[2] as u128) + c3;
  acc0.0[2] = sum0 as u64;
  acc1.0[2] = sum1 as u64;
  acc2.0[2] = sum2 as u64;
  acc3.0[2] = sum3 as u64;
  (c0, c1, c2, c3) = (sum0 >> 64, sum1 >> 64, sum2 >> 64, sum3 >> 64);

  // Limb 3
  let sum0 = (acc0.0[3] as u128) + (prod0[3] as u128) + c0;
  let sum1 = (acc1.0[3] as u128) + (prod1[3] as u128) + c1;
  let sum2 = (acc2.0[3] as u128) + (prod2[3] as u128) + c2;
  let sum3 = (acc3.0[3] as u128) + (prod3[3] as u128) + c3;
  acc0.0[3] = sum0 as u64;
  acc1.0[3] = sum1 as u64;
  acc2.0[3] = sum2 as u64;
  acc3.0[3] = sum3 as u64;
  (c0, c1, c2, c3) = (sum0 >> 64, sum1 >> 64, sum2 >> 64, sum3 >> 64);

  // Limb 4
  let sum0 = (acc0.0[4] as u128) + (prod0[4] as u128) + c0;
  let sum1 = (acc1.0[4] as u128) + (prod1[4] as u128) + c1;
  let sum2 = (acc2.0[4] as u128) + (prod2[4] as u128) + c2;
  let sum3 = (acc3.0[4] as u128) + (prod3[4] as u128) + c3;
  acc0.0[4] = sum0 as u64;
  acc1.0[4] = sum1 as u64;
  acc2.0[4] = sum2 as u64;
  acc3.0[4] = sum3 as u64;
  (c0, c1, c2, c3) = (sum0 >> 64, sum1 >> 64, sum2 >> 64, sum3 >> 64);

  // Limb 5
  let sum0 = (acc0.0[5] as u128) + (prod0[5] as u128) + c0;
  let sum1 = (acc1.0[5] as u128) + (prod1[5] as u128) + c1;
  let sum2 = (acc2.0[5] as u128) + (prod2[5] as u128) + c2;
  let sum3 = (acc3.0[5] as u128) + (prod3[5] as u128) + c3;
  acc0.0[5] = sum0 as u64;
  acc1.0[5] = sum1 as u64;
  acc2.0[5] = sum2 as u64;
  acc3.0[5] = sum3 as u64;
  (c0, c1, c2, c3) = (sum0 >> 64, sum1 >> 64, sum2 >> 64, sum3 >> 64);

  // Limb 6
  let sum0 = (acc0.0[6] as u128) + (prod0[6] as u128) + c0;
  let sum1 = (acc1.0[6] as u128) + (prod1[6] as u128) + c1;
  let sum2 = (acc2.0[6] as u128) + (prod2[6] as u128) + c2;
  let sum3 = (acc3.0[6] as u128) + (prod3[6] as u128) + c3;
  acc0.0[6] = sum0 as u64;
  acc1.0[6] = sum1 as u64;
  acc2.0[6] = sum2 as u64;
  acc3.0[6] = sum3 as u64;
  (c0, c1, c2, c3) = (sum0 >> 64, sum1 >> 64, sum2 >> 64, sum3 >> 64);

  // Limb 7
  let sum0 = (acc0.0[7] as u128) + (prod0[7] as u128) + c0;
  let sum1 = (acc1.0[7] as u128) + (prod1[7] as u128) + c1;
  let sum2 = (acc2.0[7] as u128) + (prod2[7] as u128) + c2;
  let sum3 = (acc3.0[7] as u128) + (prod3[7] as u128) + c3;
  acc0.0[7] = sum0 as u64;
  acc1.0[7] = sum1 as u64;
  acc2.0[7] = sum2 as u64;
  acc3.0[7] = sum3 as u64;
  (c0, c1, c2, c3) = (sum0 >> 64, sum1 >> 64, sum2 >> 64, sum3 >> 64);

  // Limb 8 (final carry)
  acc0.0[8] = acc0.0[8].wrapping_add(c0 as u64);
  acc1.0[8] = acc1.0[8].wrapping_add(c1 as u64);
  acc2.0[8] = acc2.0[8].wrapping_add(c2 as u64);
  acc3.0[8] = acc3.0[8].wrapping_add(c3 as u64);
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
  fn test_unreduced_field_int_mul_add() {
    use crate::small_field::limbs::SignedWideLimbs;
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

      <Scalar as DelayedReduction<i32>>::unreduced_field_int_mul_add(&mut acc, &field, small);
      expected += field * i64_to_field::<Scalar>(small);
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
  fn test_unreduced_field_int_many_products() {
    use crate::small_field::limbs::SignedWideLimbs;
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

      <Scalar as DelayedReduction<i32>>::unreduced_field_int_mul_add(&mut acc, &field, small);
      expected += field * i64_to_field::<Scalar>(small);
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
