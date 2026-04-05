// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! DelayedReduction trait and standalone accumulation functions.
//!
//! Modular reduction (Montgomery REDC) is expensive. When summing many
//! products `Σ (field_i × field_j)`, the standard approach does N reductions.
//! Delayed reduction accumulates unreduced products in wide integers, reducing
//! only once at the end.
//!
//! # Architecture
//!
//! This module provides:
//! - [`DelayedReduction`] trait defining the accumulation interface
//! - Standalone accumulation functions for different product types
//! - Macros to generate trait implementations for specific fields
//!
//! The trait implementations are generated via macros in provider files:
//! - `impl_delayed_reduction!` for all fields (generic Barrett)

use super::{
  limbs::{SignedWideLimbs, WideLimbs, mac, mul_4_by_4},
  montgomery::{MontgomeryLimbs, montgomery_reduce_9},
};
use ff::PrimeField;
use num_traits::Zero;
use std::ops::AddAssign;

/// Trait for delayed modular reduction operations.
///
/// Accumulates unreduced products in wide integers, reducing only at the end.
pub trait DelayedReduction<Value>: Sized {
  /// Wide accumulator type for unreduced products.
  type Accumulator: Copy + Clone + Default + AddAssign + Send + Sync + Zero;

  /// Accumulate: `acc += field × value` without modular reduction.
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &Value);

  /// Reduce the accumulator to a field element.
  fn reduce(acc: &Self::Accumulator) -> Self;
}

/// Accumulate field × small_value where small_value is a single u64 magnitude.
///
/// This function is shared by i32 and i64 accumulation (same 4×1 multiply pattern).
/// The caller handles sign extraction and chooses the target (pos or neg).
///
/// # Arguments
/// - `target`: The unsigned accumulator to add to (either pos or neg side)
/// - `field`: The field element (4 limbs in Montgomery form)
/// - `magnitude`: The absolute value of the small integer (as u64)
///
/// # Overflow Bounds
/// - Supported field element: 254-256 bits
/// - u64 magnitude: 64 bits
/// - Product size: 318-320 bits (5 limbs)
/// - WideLimbs<6>: 384 bits capacity
/// - Headroom: 64-66 bits → supports at least 2^64 accumulations
#[inline(always)]
pub(crate) fn accumulate_field_times_small<F: MontgomeryLimbs>(
  target: &mut WideLimbs<6>,
  field: &F,
  magnitude: u64,
) {
  let a = field.to_limbs();
  let (r0, c) = mac(target.0[0], a[0], magnitude, 0);
  let (r1, c) = mac(target.0[1], a[1], magnitude, c);
  let (r2, c) = mac(target.0[2], a[2], magnitude, c);
  let (r3, c) = mac(target.0[3], a[3], magnitude, c);
  let (r4, of) = target.0[4].overflowing_add(c);
  target.0[0] = r0;
  target.0[1] = r1;
  target.0[2] = r2;
  target.0[3] = r3;
  target.0[4] = r4;
  let old_limb5 = target.0[5];
  target.0[5] = target.0[5].wrapping_add(of as u64);
  debug_assert!(
    target.0[5] >= old_limb5,
    "DelayedReduction small-value accumulator overflow: limb 5 wrapped from {} to {}",
    old_limb5,
    target.0[5]
  );
}

/// Accumulate field × i128 product into a SignedWideLimbs<7> accumulator.
///
/// Uses 4×2 multiply pattern since i128 spans 2 limbs.
///
/// # Overflow Bounds
/// - Supported field element: 254-256 bits
/// - i128 magnitude: 128 bits
/// - Product size: 382-384 bits (6 limbs)
/// - SignedWideLimbs<7>: 448 bits capacity
/// - Headroom: 64-66 bits → supports at least 2^64 accumulations
#[inline(always)]
pub(crate) fn accumulate_field_times_i128<F: MontgomeryLimbs>(
  acc: &mut SignedWideLimbs<7>,
  field: &F,
  value: &i128,
) {
  let (target, mag) = if *value >= 0 {
    (&mut acc.pos, *value as u128)
  } else {
    (&mut acc.neg, (*value).wrapping_neg() as u128)
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
  let old_limb6 = target.0[6];
  target.0[6] = target.0[6].wrapping_add(c);
  debug_assert!(
    target.0[6] >= old_limb6,
    "DelayedReduction i128 accumulator overflow: limb 6 wrapped from {} to {} (carry={})",
    old_limb6,
    target.0[6],
    c
  );
}

/// Accumulate field × field product into a WideLimbs<9> accumulator.
///
/// Uses full 4×4 multiply producing 8-limb result.
///
/// # Capacity Invariant
///
/// The 9th limb (index 8) accumulates carries from the lower 8 limbs. Each
/// field×field product contributes at most 1 to the carry chain into limb 8.
/// With a u64 limb, we can accumulate up to 2^64 products before overflow.
/// In practice, sumcheck rounds are bounded by polynomial size (≤ 2^40),
/// so this limit is never approached. The debug_assert below catches misuse.
#[inline(always)]
pub(crate) fn accumulate_field_times_field<F: MontgomeryLimbs + Copy>(
  acc: &mut WideLimbs<9>,
  field_a: &F,
  field_b: &F,
) {
  // Compute field_a × field_b as 8 limbs and add to accumulator
  let product = mul_4_by_4(field_a.to_limbs(), field_b.to_limbs());
  let mut carry = 0u128;
  for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
    let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
    *acc_limb = sum as u64;
    carry = sum >> 64;
  }

  // Accumulate carry into the 9th limb. Overflow here means we've exceeded
  // the accumulator's capacity (~2^64 products) - this should never happen
  // in valid usage since sumcheck polynomials are bounded by practical sizes.
  let old_limb8 = acc.0[8];
  acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  debug_assert!(
    acc.0[8] >= old_limb8,
    "DelayedReduction accumulator overflow: limb 8 wrapped from {} to {} (carry={}). \
     Too many products accumulated without reduction.",
    old_limb8,
    acc.0[8],
    carry
  );
}

// ============================================================================
// DelayedReduction<F> implementation for field × field
// ============================================================================

/// DelayedReduction<F> for field × field products.
///
/// Uses WideLimbs<9> (576 bits) as accumulator, supporting up to 2^68 products.
impl<F: MontgomeryLimbs + PrimeField + Copy> DelayedReduction<F> for F {
  type Accumulator = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field_a: &Self, field_b: &F) {
    accumulate_field_times_field(acc, field_a, field_b);
  }

  #[inline(always)]
  fn reduce(acc: &Self::Accumulator) -> Self {
    F::from_limbs(montgomery_reduce_9::<F>(&acc.0))
  }
}

// =============================================================================
// Test helpers (exported for use by provider test modules)
// =============================================================================

#[cfg(test)]
pub(crate) fn test_delayed_reduction_sum_impl<F: MontgomeryLimbs + PrimeField + Copy>() {
  use rand::{SeedableRng, rngs::StdRng};

  let mut rng = StdRng::seed_from_u64(54321);

  let n = 1000;
  let a_vec: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
  let b_vec: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();

  // Compute sum using standard field arithmetic
  let expected: F = a_vec.iter().zip(b_vec.iter()).map(|(a, b)| *a * *b).sum();

  // Compute using delayed reduction
  let mut acc = WideLimbs::<9>::default();
  for (a, b) in a_vec.iter().zip(b_vec.iter()) {
    <F as DelayedReduction<F>>::unreduced_multiply_accumulate(&mut acc, a, b);
  }
  let result = <F as DelayedReduction<F>>::reduce(&acc);

  assert_eq!(
    result, expected,
    "Delayed reduction sum failed: accumulated result != direct sum"
  );
}

/// Generic test for DelayedReduction<V> where V is any small value type (i32, i64, i128).
///
/// Tests that accumulating field × small_value products with delayed reduction
/// produces the same result as immediate field multiplication.
#[cfg(test)]
pub(crate) fn test_delayed_reduction_small_impl<F, V>()
where
  F: MontgomeryLimbs + PrimeField + Copy + DelayedReduction<V> + super::SmallValueField<V>,
  V: Copy,
  rand::distributions::Standard: rand::distributions::Distribution<V>,
{
  use rand::{Rng, SeedableRng, rngs::StdRng};

  let mut rng = StdRng::seed_from_u64(54321);

  let mut acc = <F as DelayedReduction<V>>::Accumulator::default();
  let mut expected = F::ZERO;

  // Sum 100 field × V products with full range sampling
  for _ in 0..100 {
    let field = F::random(&mut rng);
    let value: V = rng.r#gen();

    <F as DelayedReduction<V>>::unreduced_multiply_accumulate(&mut acc, &field, &value);
    expected += field * F::small_to_field(value);
  }

  let result = <F as DelayedReduction<V>>::reduce(&acc);
  assert_eq!(
    result, expected,
    "Delayed reduction failed for small value type"
  );
}

/// Regression coverage for `DelayedReduction<i32>` signed boundary values.
#[cfg(test)]
pub(crate) fn test_delayed_reduction_i32_boundaries_impl<F>()
where
  F: MontgomeryLimbs + PrimeField + Copy + DelayedReduction<i32>,
{
  let fields = [F::from(1u64), F::from(7u64), F::from(u32::MAX as u64)];
  let values = [i32::MIN, i32::MAX];

  for field in fields {
    for value in values {
      let mut acc = <F as DelayedReduction<i32>>::Accumulator::default();
      <F as DelayedReduction<i32>>::unreduced_multiply_accumulate(&mut acc, &field, &value);

      let result = <F as DelayedReduction<i32>>::reduce(&acc);
      let expected = field * super::small_value_field::i64_to_field::<F>(value as i64);

      assert_eq!(
        result, expected,
        "Delayed reduction failed for i32 boundary value {value}"
      );
    }
  }

  let mut acc = <F as DelayedReduction<i32>>::Accumulator::default();
  let mut expected = F::ZERO;
  for (field, value) in fields.into_iter().zip([i32::MIN, i32::MAX, i32::MIN]) {
    <F as DelayedReduction<i32>>::unreduced_multiply_accumulate(&mut acc, &field, &value);
    expected += field * super::small_value_field::i64_to_field::<F>(value as i64);
  }

  let result = <F as DelayedReduction<i32>>::reduce(&acc);
  assert_eq!(result, expected, "Delayed reduction failed for accumulated i32 boundaries");
}

/// Regression coverage for `DelayedReduction<i64>` signed boundary values.
#[cfg(test)]
pub(crate) fn test_delayed_reduction_i64_boundaries_impl<F>()
where
  F: MontgomeryLimbs + PrimeField + Copy + DelayedReduction<i64> + super::SmallValueField<i64>,
{
  let fields = [F::from(1u64), F::from(7u64), F::from(u32::MAX as u64)];
  let values = [i64::MIN, i64::MAX];

  for field in fields {
    for value in values {
      let mut acc = <F as DelayedReduction<i64>>::Accumulator::default();
      <F as DelayedReduction<i64>>::unreduced_multiply_accumulate(&mut acc, &field, &value);

      let result = <F as DelayedReduction<i64>>::reduce(&acc);
      let expected = field * F::small_to_field(value);

      assert_eq!(
        result, expected,
        "Delayed reduction failed for i64 boundary value {value}"
      );
    }
  }

  let mut acc = <F as DelayedReduction<i64>>::Accumulator::default();
  let mut expected = F::ZERO;
  for (field, value) in fields.into_iter().zip([i64::MIN, i64::MAX, i64::MIN]) {
    <F as DelayedReduction<i64>>::unreduced_multiply_accumulate(&mut acc, &field, &value);
    expected += field * F::small_to_field(value);
  }

  let result = <F as DelayedReduction<i64>>::reduce(&acc);
  assert_eq!(result, expected, "Delayed reduction failed for accumulated i64 boundaries");
}

/// Regression coverage for `DelayedReduction<i128>` signed boundary values.
#[cfg(test)]
pub(crate) fn test_delayed_reduction_i128_boundaries_impl<F>()
where
  F: MontgomeryLimbs + PrimeField + Copy + DelayedReduction<i128>,
{
  let fields = [F::from(1u64), F::from(7u64), F::from(u32::MAX as u64)];
  let values = [i128::MIN, i128::MAX];

  for field in fields {
    for value in values {
      let mut acc = <F as DelayedReduction<i128>>::Accumulator::default();
      <F as DelayedReduction<i128>>::unreduced_multiply_accumulate(&mut acc, &field, &value);

      let result = <F as DelayedReduction<i128>>::reduce(&acc);
      let expected = field * super::small_value_field::i128_to_field::<F>(value);

      assert_eq!(
        result, expected,
        "Delayed reduction failed for i128 boundary value {value}"
      );
    }
  }

  let mut acc = <F as DelayedReduction<i128>>::Accumulator::default();
  let mut expected = F::ZERO;
  for (field, value) in fields.into_iter().zip([i128::MIN, i128::MAX, i128::MIN]) {
    <F as DelayedReduction<i128>>::unreduced_multiply_accumulate(&mut acc, &field, &value);
    expected += field * super::small_value_field::i128_to_field::<F>(value);
  }

  let result = <F as DelayedReduction<i128>>::reduce(&acc);
  assert_eq!(result, expected, "Delayed reduction failed for accumulated i128 boundaries");
}

/// Generate tests for `DelayedReduction` implementation (field × field only).
#[cfg(test)]
#[macro_export]
macro_rules! test_delayed_reduction {
  ($mod_name:ident, $field:ty) => {
    mod $mod_name {
      #[test]
      fn delayed_reduction_sum() {
        $crate::big_num::delayed_reduction::test_delayed_reduction_sum_impl::<$field>();
      }
    }
  };
}

/// Generate tests for `DelayedReduction` with small value types (i32, i64, i128).
#[cfg(test)]
#[macro_export]
macro_rules! test_delayed_reduction_small {
  ($mod_name:ident, $field:ty) => {
    mod $mod_name {
      #[test]
      fn delayed_reduction_i32() {
        $crate::big_num::delayed_reduction::test_delayed_reduction_small_impl::<$field, i32>();
        $crate::big_num::delayed_reduction::test_delayed_reduction_i32_boundaries_impl::<$field>();
      }

      #[test]
      fn delayed_reduction_i64() {
        $crate::big_num::delayed_reduction::test_delayed_reduction_small_impl::<$field, i64>();
        $crate::big_num::delayed_reduction::test_delayed_reduction_i64_boundaries_impl::<$field>();
      }

      #[test]
      fn delayed_reduction_i128() {
        $crate::big_num::delayed_reduction::test_delayed_reduction_small_impl::<$field, i128>();
        $crate::big_num::delayed_reduction::test_delayed_reduction_i128_boundaries_impl::<$field>();
      }
    }
  };
}
