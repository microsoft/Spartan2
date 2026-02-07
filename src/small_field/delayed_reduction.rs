// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! DelayedReduction trait for accumulating unreduced products.
//!
//! This module provides:
//! - [`DelayedReduction<Value>`]: Unified interface for delayed modular reduction
//!
//! # Background
//!
//! Modular reduction (Barrett or Montgomery REDC) is expensive. When summing many
//! products `Σ (field_i × value_i)`, the standard approach does N reductions.
//! Delayed reduction accumulates unreduced products in wide integers, reducing
//! only once at the end.
//!
//! # Design: Fused Multiply-Accumulate
//!
//! We use a single [`DelayedReduction::unreduced_multiply_accumulate`] rather than separate
//! `multiply(field, value) → product` and `accumulate(acc, product)` operations.
//! While separate functions would be more compositional and easier to test in
//! isolation, the fused approach provides significant performance benefits:
//!
//! **Benefits of fused MAC:**
//! - **No intermediate storage**: Avoids writing/reading a 5-8 limb intermediate
//!   array (~40-64 bytes). In tight loops with millions of iterations, this
//!   memory traffic is significant.
//! - **Better register utilization**: Intermediate values stay in CPU registers
//!   throughout the operation, avoiding stack spills.
//! - **Single carry chain**: Carry propagates through one MAC chain. Separate
//!   multiply and add would require two independent carry propagations.
//! - **Instruction-level parallelism**: x86-64 `mulx` + `adcx/adox` instructions
//!   can be interleaved. A separate add loop cannot start until multiply completes.
//!
//! **Quantitative impact**: For the i64 path, fused MAC uses ~6 instructions per
//! limb while separate multiply+add would use ~10-12. In sumcheck with n = 2^20
//! variables, this represents millions of saved operations.
//!
//! # Usage Sites
//!
//! This trait is used in several performance-critical paths:
//!
//! - **Small-value sumcheck** ([`crate::small_sumcheck`]): Lagrange accumulator
//!   building uses `field × small` and `field × (small × small)` products.
//! - **Eq-split sumcheck** ([`crate::sumcheck::eq_sumcheck`]): The delayed reduction
//!   variant uses `field × field` products to reduce Montgomery reductions from
//!   O(2^k) to O(2^{k/2}) per round.
//! - **Sparse matrix-vector multiplication** ([`crate::r1cs::sparse`]): Computing
//!   Az, Bz, Cz with small witness values uses `field × small` products.
//!
//! These are among the most time-consuming prover operations, making the fused
//! approach a worthwhile tradeoff. Each product is used exactly once in our
//! algorithms, so the inability to reuse intermediate products has no impact.
//!
//! # Accumulator Types
//!
//! - `DelayedReduction<i64>`: For `field × i64` products. Uses `SignedWideLimbs<6>` (6×64=384 bits).
//!   Sufficient for accumulating ~2^52 terms without overflow.
//! - `DelayedReduction<i128>`: For `field × i128` products. Uses `SignedWideLimbs<7>` (7×64=448 bits).
//!   Sufficient for accumulating ~2^52 terms without overflow.
//! - `DelayedReduction<F>`: For `field × field` products. Uses `WideLimbs<9>` (9×64=576 bits).

use super::{
  barrett::{barrett_reduce_6, barrett_reduce_7},
  limbs::{SignedWideLimbs, SubMagResult, WideLimbs, mac, mul_4_by_4, sub_mag},
  montgomery::{MontgomeryLimbs, montgomery_reduce_9},
  small_value_field::SupportsSmallI64,
};
use ff::PrimeField;
use std::ops::AddAssign;

/// Trait for delayed modular reduction operations.
///
/// This trait provides operations that accumulate unreduced products in wide
/// integers, reducing only at the end. Used in hot paths like sumcheck
/// accumulator building where many products are summed together.
///
/// # Type Parameter
///
/// - `Value`: The type being multiplied with the field element (i32, i64, i128, or Self)
pub trait DelayedReduction<Value>: Sized {
  /// Wide accumulator type for unreduced products.
  ///
  /// Must support addition without overflow for the expected number of terms.
  type Accumulator: Copy + Clone + Default + AddAssign + Send + Sync + num_traits::Zero;

  /// Accumulate: `acc += field × value`
  ///
  /// Performs widening multiplication and accumulates without modular reduction.
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &Value);

  /// Reduce the accumulator to a field element.
  ///
  /// Performs the deferred modular reduction.
  fn reduce(acc: &Self::Accumulator) -> Self;
}

// ============================================================================
// DelayedReduction<i32> - for field × i32 products (direct small values)
// ============================================================================

impl<F: MontgomeryLimbs + PrimeField> DelayedReduction<i32> for F {
  /// Accumulator for field × i32 products.
  ///
  /// # Overflow Bounds
  /// - Field element: 254 bits (BN254 Fr)
  /// - i32 magnitude: 32 bits
  /// - Product size: 286 bits (5 limbs)
  /// - SignedWideLimbs<6>: 384 bits capacity
  /// - Headroom: 98 bits → supports up to 2^98 accumulations
  type Accumulator = SignedWideLimbs<6>;

  #[inline(always)]
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &i32) {
    // Extend i32 to i64 and use the same accumulation logic
    let value64 = *value as i64;
    let (target, mag) = if value64 >= 0 {
      (&mut acc.pos, value64 as u64)
    } else {
      (&mut acc.neg, value64.wrapping_neg() as u64)
    };
    let a = field.to_limbs();
    let (r0, c) = mac(target.0[0], a[0], mag, 0);
    let (r1, c) = mac(target.0[1], a[1], mag, c);
    let (r2, c) = mac(target.0[2], a[2], mag, c);
    let (r3, c) = mac(target.0[3], a[3], mag, c);
    let (r4, of) = target.0[4].overflowing_add(c);
    target.0[0] = r0;
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = target.0[5].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn reduce(acc: &Self::Accumulator) -> Self {
    match sub_mag::<6>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => F::from_limbs(barrett_reduce_6::<F>(&mag)),
      SubMagResult::Negative(mag) => -F::from_limbs(barrett_reduce_6::<F>(&mag)),
    }
  }
}

// ============================================================================
// DelayedReduction<i64> - for field × i64 products (from i32 × i32 or direct)
// ============================================================================

impl<F: MontgomeryLimbs + PrimeField> DelayedReduction<i64> for F {
  /// Accumulator for field × i64 products.
  ///
  /// # Overflow Bounds
  /// - Field element: 254 bits (BN254 Fr)
  /// - i64 magnitude: 64 bits
  /// - Product size: 318 bits (5 limbs)
  /// - SignedWideLimbs<6>: 384 bits capacity
  /// - Headroom: 66 bits → supports up to 2^66 accumulations
  type Accumulator = SignedWideLimbs<6>;

  #[inline(always)]
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &i64) {
    // Handle sign: accumulate into pos or neg based on sign of value
    let (target, mag) = if *value >= 0 {
      (&mut acc.pos, *value as u64)
    } else {
      (&mut acc.neg, (*value).wrapping_neg() as u64)
    };
    // Fused multiply-accumulate: field × value
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
  fn reduce(acc: &Self::Accumulator) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    match sub_mag::<6>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => F::from_limbs(barrett_reduce_6::<F>(&mag)),
      SubMagResult::Negative(mag) => -F::from_limbs(barrett_reduce_6::<F>(&mag)),
    }
  }
}

// ============================================================================
// DelayedReduction<i128> - for field × i128 products (from i64 × i64)
// ============================================================================

impl<F: SupportsSmallI64 + PrimeField> DelayedReduction<i128> for F {
  /// Accumulator for field × i128 products (from i64 × i64).
  ///
  /// # Overflow Bounds
  /// - Field element: 254 bits (BN254 Fr)
  /// - i128 magnitude: 128 bits
  /// - Product size: 382 bits (6 limbs)
  /// - SignedWideLimbs<7>: 448 bits capacity
  /// - Headroom: 66 bits → supports up to 2^66 accumulations
  type Accumulator = SignedWideLimbs<7>;

  #[inline(always)]
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &i128) {
    let (target, mag) = if *value >= 0 {
      (&mut acc.pos, *value as u128)
    } else {
      (&mut acc.neg, (-*value) as u128)
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
  fn reduce(acc: &Self::Accumulator) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    match sub_mag::<7>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => F::from_limbs(barrett_reduce_7::<F>(&mag)),
      SubMagResult::Negative(mag) => -F::from_limbs(barrett_reduce_7::<F>(&mag)),
    }
  }
}

// ============================================================================
// DelayedReduction<F> - for field × field products
// ============================================================================

impl<F: MontgomeryLimbs + PrimeField + Copy> DelayedReduction<F> for F {
  /// Accumulator for field × field products.
  ///
  /// # Overflow Bounds
  /// - Field element: 254 bits (BN254 Fr)
  /// - Product size: 508 bits (8 limbs)
  /// - WideLimbs<9>: 576 bits capacity
  /// - Headroom: 68 bits → supports up to 2^68 accumulations
  type Accumulator = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field_a: &Self, field_b: &F) {
    // Compute field_a × field_b as 8 limbs and add to accumulator
    let product = mul_4_by_4(field_a.to_limbs(), field_b.to_limbs());
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn reduce(acc: &Self::Accumulator) -> Self {
    F::from_limbs(montgomery_reduce_9::<F>(&acc.0))
  }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::pasta::pallas,
    small_field::{SmallValueField, WideMul},
  };
  use ff::Field;
  use num_traits::Zero;
  use rand_core::{OsRng, RngCore};
  use std::ops::{Add, Sub};

  type Scalar = pallas::Scalar;

  #[test]
  fn test_delayed_reduction_i64() {
    let mut rng = OsRng;
    let mut acc = SignedWideLimbs::<6>::zero();
    let mut expected = Scalar::ZERO;

    // Sum 100 field × i64 products (mix of positive and negative)
    for i in 0..100 {
      let field = Scalar::random(&mut rng);
      let value_u = (rng.next_u64() >> 48) as i64; // Keep small
      let value: i64 = if i % 2 == 0 { value_u } else { -value_u };

      <Scalar as DelayedReduction<i64>>::unreduced_multiply_accumulate(&mut acc, &field, &value);
      expected += field * super::super::small_value_field::i64_to_field::<Scalar>(value);
    }

    let result = <Scalar as DelayedReduction<i64>>::reduce(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_delayed_reduction_i128() {
    let mut rng = OsRng;
    let mut acc = SignedWideLimbs::<7>::zero();
    let mut expected = Scalar::ZERO;

    // Sum 100 field × i128 products (mix of positive and negative)
    for i in 0..100 {
      let field = Scalar::random(&mut rng);
      let a = (rng.next_u64() >> 48) as i64;
      let b = (rng.next_u64() >> 48) as i64;
      let value: i128 = (a as i128) * (b as i128);
      let value: i128 = if i % 2 == 0 { value } else { -value };

      <Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(&mut acc, &field, &value);

      // For expected, convert i128 to field manually
      let field_value = if value >= 0 {
        Scalar::from(value as u64)
          + Scalar::from((value >> 64) as u64) * Scalar::from(1u64 << 32) * Scalar::from(1u64 << 32)
      } else {
        let abs = (-value) as u128;
        -(Scalar::from(abs as u64)
          + Scalar::from((abs >> 64) as u64) * Scalar::from(1u64 << 32) * Scalar::from(1u64 << 32))
      };
      expected += field * field_value;
    }

    let result = <Scalar as DelayedReduction<i128>>::reduce(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_delayed_reduction_field() {
    let mut rng = OsRng;
    let mut acc = WideLimbs::<9>::zero();
    let mut expected = Scalar::ZERO;

    // Sum 100 field × field products
    for _ in 0..100 {
      let a = Scalar::random(&mut rng);
      let b = Scalar::random(&mut rng);

      <Scalar as DelayedReduction<Scalar>>::unreduced_multiply_accumulate(&mut acc, &a, &b);
      expected += a * b;
    }

    let result = <Scalar as DelayedReduction<Scalar>>::reduce(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_small_value_field_bounds() {
    // This test just verifies that the trait bounds work correctly
    fn check_bounds<F, SmallValue>()
    where
      F: SmallValueField<SmallValue>
        + DelayedReduction<SmallValue>
        + DelayedReduction<SmallValue::Product>
        + DelayedReduction<F>,
      SmallValue: WideMul
        + Copy
        + Default
        + num_traits::Zero
        + Add<Output = SmallValue>
        + Sub<Output = SmallValue>
        + Send
        + Sync,
    {
    }

    // These should compile if bounds are correct
    check_bounds::<Scalar, i32>();
    check_bounds::<Scalar, i64>();
  }
}
