// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Barrett reduction for wide limb values.

use super::field_reduction_constants::FieldReductionConstants;
use super::limbs::{gte_4_4, mul_4_by_1, sub_4_4};

// ==========================================================================
// 6-limb Barrett reduction (for SignedWideLimbs<6>, i.e. DelayedReduction<i64>::Accumulator)
// ==========================================================================

/// Generic 6-limb Barrett reduction using trait constants.
///
/// Reduces a 6-limb value (up to 384 bits) modulo p using precomputed
/// R256 = 2^256 mod p and R320 = 2^320 mod p constants.
///
/// Input is already in Montgomery form (R-scaled). This function reduces
/// the 6-limb value mod p while preserving the Montgomery scaling.
///
/// # Algorithm
///
/// 1. Bounded folding loop: fold limbs 4-5 using R256_MOD and R320_MOD
/// 2. Each fold reduces the magnitude by a factor of ~2^64/r3 (field-dependent)
/// 3. After MAX_BARRETT_FOLDS iterations, limbs 4-5 are guaranteed to be zero
/// 4. Bounded canonicalization: subtract p up to MAX_CANONICALIZE_SUBS times
#[inline]
pub(crate) fn barrett_reduce_6<F: FieldReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
  let mut limbs = *c;

  // Bounded folding loop - eliminates recursion
  for _ in 0..F::MAX_BARRETT_FOLDS {
    if limbs[4] == 0 && limbs[5] == 0 {
      break;
    }

    // Fold limbs[4] and limbs[5] using R256_MOD and R320_MOD
    let c4_contrib = mul_4_by_1(&F::R256_MOD, limbs[4]);
    let c5_contrib = mul_4_by_1(&F::R320_MOD, limbs[5]);

    // Sum: limbs[0..4] + c4_contrib + c5_contrib
    let mut carry = 0u128;
    for i in 0..4 {
      let s = (limbs[i] as u128) + (c4_contrib[i] as u128) + (c5_contrib[i] as u128) + carry;
      limbs[i] = s as u64;
      carry = s >> 64;
    }
    let s = (c4_contrib[4] as u128) + (c5_contrib[4] as u128) + carry;
    limbs[4] = s as u64;
    limbs[5] = (s >> 64) as u64;
  }

  debug_assert!(
    limbs[4] == 0 && limbs[5] == 0,
    "Barrett folding did not converge after {} iterations",
    F::MAX_BARRETT_FOLDS
  );

  // Bounded canonicalization
  let mut r = [limbs[0], limbs[1], limbs[2], limbs[3]];
  for _ in 0..F::MAX_CANONICALIZE_SUBS {
    if gte_4_4(&r, &F::MODULUS) {
      r = sub_4_4(&r, &F::MODULUS);
    }
  }

  debug_assert!(
    !gte_4_4(&r, &F::MODULUS),
    "Barrett canonicalization failed after {} subtractions",
    F::MAX_CANONICALIZE_SUBS
  );

  r
}

// ==========================================================================
// 7-limb Barrett reduction (for SignedWideLimbs<7>, i.e. DelayedReduction<i128>::Accumulator)
// ==========================================================================

/// Generic 7-limb Barrett reduction using trait constants.
///
/// Reduces a 7-limb value (up to 448 bits) modulo p by folding limb 6
/// using R384 = 2^384 mod p, then delegating to the bounded 6-limb reduction.
///
/// This function is non-recursive: limb 6 is folded once, then `barrett_reduce_6`
/// handles the remaining 6 limbs with its bounded loop.
#[inline]
pub(crate) fn barrett_reduce_7<F: FieldReductionConstants>(c: &[u64; 7]) -> [u64; 4] {
  let c6_contrib = mul_4_by_1(&F::R384_MOD, c[6]);

  let mut sum = [0u64; 6];
  let mut carry = 0u128;
  for i in 0..4 {
    let s = (c[i] as u128) + (c6_contrib[i] as u128) + carry;
    sum[i] = s as u64;
    carry = s >> 64;
  }
  let s = (c[4] as u128) + (c6_contrib[4] as u128) + carry;
  sum[4] = s as u64;
  carry = s >> 64;
  let s = (c[5] as u128) + carry;
  sum[5] = s as u64;

  barrett_reduce_6::<F>(&sum)
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use halo2curves::pasta::{Fp, Fq};
  use rand_core::{OsRng, RngCore};

  #[test]
  fn test_constants_match_halo2curves() {
    use crate::small_field::field_reduction_constants::FieldReductionConstants;

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
    let result = Fp(barrett_reduce_6::<Fp>(&c));
    assert_eq!(result, Fp::ZERO);
  }

  #[test]
  fn test_barrett_6_fp_from_product() {
    use super::super::limbs::mul_4_by_1;

    // Test with an actual field x integer product (the real use case)
    let field_elem = Fp::from(12345u64); // Creates Montgomery form
    let small = 9999u64;

    // Compute field x small as 5 limbs (Montgomery form)
    let product = mul_4_by_1(&field_elem.0, small);

    // Extend to 6 limbs
    let c = [
      product[0], product[1], product[2], product[3], product[4], 0,
    ];

    // Reduce
    let result = Fp(barrett_reduce_6::<Fp>(&c));

    // Expected: field_elem * small
    let expected = field_elem * Fp::from(small);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_barrett_6_fp_sum_of_products() {
    use super::super::limbs::mul_4_by_1;

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

      // Compute field x small as 5 limbs
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
    let result = Fp(barrett_reduce_6::<Fp>(&acc));
    assert_eq!(result, expected_sum);
  }

  #[test]
  fn test_barrett_6_fp_many_products() {
    use super::super::limbs::mul_4_by_1;

    // Stress test: sum many products to exercise 6-limb reduction
    // Note: In the real use case, we sum at most 2^(l/2) products where l <= 130.
    // For l = 20 (a typical size), that's 2^10 = 1024 products.
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

    let result = Fp(barrett_reduce_6::<Fp>(&acc));
    assert_eq!(result, expected_sum);
  }

  #[test]
  fn test_barrett_6_fq_sum_of_products() {
    use super::super::limbs::mul_4_by_1;

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

    let result = Fq(barrett_reduce_6::<Fq>(&acc));
    assert_eq!(result, expected_sum);
  }
}
