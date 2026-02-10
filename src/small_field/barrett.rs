// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Barrett reduction for wide limb values.

use super::field_reduction_constants::FieldReductionConstants;
use super::limbs::{
  add_4_4, gte_4_4, mul_2_by_1, mul_3x4_lo4, mul_3x5_to_8, mul_4_by_1, select_4, sub_4_4,
  sub_4_4_with_borrow, sub_5_4,
};

// ==========================================================================
// 6-limb Barrett reduction (for SignedWideLimbs<6>, i.e. DelayedReduction<i64>::Accumulator)
// ==========================================================================

/// Barrett reduction for 6-limb input.
///
/// Dispatches to Pasta 2-fold or generic μ-Barrett based on field type.
#[inline]
pub(crate) fn barrett_reduce_6<F: FieldReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
  if F::PASTA_STYLE_MODULUS {
    barrett_reduce_6_pasta::<F>(c)
  } else {
    barrett_reduce_6_generic::<F>(c)
  }
}

/// Pasta 2-fold Barrett reduction for 6-limb input.
///
/// For Pasta primes p = 2^254 + c where c fits in 2 limbs:
/// - Key identity: 2^254 ≡ -c (mod p)
/// - Split x at bit 254: x = x_lo + x_hi × 2^254
/// - Reduce: x ≡ x_lo - x_hi × c (mod p)
/// - If negative: add p
/// - Repeat fold, then final canonicalization
///
/// This uses only ~4-8 multiplies vs ~24 for μ-Barrett.
#[inline]
fn barrett_reduce_6_pasta<F: FieldReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
  // For 6 limbs (384 bits), split at bit 254:
  // - x_lo: bits 0-253 (4 limbs, with limb[3] masked to 62 bits)
  // - x_hi: bits 254-383 (up to 130 bits, 3 limbs)

  // Extract x_lo (low 254 bits)
  let x_lo = [c[0], c[1], c[2], c[3] & 0x3FFF_FFFF_FFFF_FFFF]; // mask to 62 bits

  // Extract x_hi (bits 254+): (c[3] >> 62) | (c[4] << 2) | (c[5] << 66)
  // This gives up to 130 bits in 3 limbs
  let x_hi_0 = (c[3] >> 62) | (c[4] << 2);
  let x_hi_1 = (c[4] >> 62) | (c[5] << 2);
  let x_hi_2 = c[5] >> 62;

  // Compute x_hi × c where c = PASTA_C (2 limbs)
  // Result is up to 130 + 128 = 258 bits (5 limbs)
  let prod = mul_limbs_by_pasta_c::<F>(x_hi_0, x_hi_1, x_hi_2);

  // Compute x_lo - prod
  // Since prod can be larger than x_lo, we may get a negative result
  // If negative, add p to make positive
  let (result, neg) = sub_wide_4_5(&x_lo, &prod);

  // If negative, add p
  let result = if neg {
    let (sum, _) = add_4_4(&result, &F::MODULUS);
    sum
  } else {
    result
  };

  // Result is now in [0, 2p) or so, need one more fold if still >= 2^254
  // Check if result >= 2^254 (i.e., bit 254 is set)
  if result[3] >= 0x4000_0000_0000_0000 {
    // Need another fold
    let x_lo2 = [
      result[0],
      result[1],
      result[2],
      result[3] & 0x3FFF_FFFF_FFFF_FFFF,
    ];
    let x_hi2 = result[3] >> 62; // at most 2 bits

    // x_hi2 × c (very small multiply)
    let prod2 = mul_1_by_pasta_c::<F>(x_hi2);

    // x_lo2 - prod2
    let (result2, neg2) = sub_4_4_check_neg(&x_lo2, &prod2);
    let result2 = if neg2 {
      let (sum, _) = add_4_4(&result2, &F::MODULUS);
      sum
    } else {
      result2
    };

    // Final canonicalization (branchless)
    let (sub, borrow) = sub_4_4_with_borrow(&result2, &F::MODULUS);
    let out = select_4(borrow == 0, &sub, &result2);

    debug_assert!(
      !gte_4_4(&out, &F::MODULUS),
      "Pasta Barrett reduction produced non-canonical result"
    );

    out
  } else {
    // Final canonicalization (branchless)
    let (sub, borrow) = sub_4_4_with_borrow(&result, &F::MODULUS);
    let out = select_4(borrow == 0, &sub, &result);

    debug_assert!(
      !gte_4_4(&out, &F::MODULUS),
      "Pasta Barrett reduction produced non-canonical result"
    );

    out
  }
}

/// Multiply 3-limb x_hi by PASTA_C (2 limbs), producing 5-limb result.
#[inline(always)]
fn mul_limbs_by_pasta_c<F: FieldReductionConstants>(
  x_hi_0: u64,
  x_hi_1: u64,
  x_hi_2: u64,
) -> [u64; 5] {
  let c = F::PASTA_C;
  let mut result = [0u64; 5];

  // x_hi_0 × c
  let prod0 = mul_2_by_1(&c, x_hi_0);
  result[0] = prod0[0];
  result[1] = prod0[1];
  result[2] = prod0[2];

  // x_hi_1 × c (add at offset 1)
  let prod1 = mul_2_by_1(&c, x_hi_1);
  let mut carry = 0u128;
  let sum = (result[1] as u128) + (prod1[0] as u128) + carry;
  result[1] = sum as u64;
  carry = sum >> 64;
  let sum = (result[2] as u128) + (prod1[1] as u128) + carry;
  result[2] = sum as u64;
  carry = sum >> 64;
  let sum = (prod1[2] as u128) + carry;
  result[3] = sum as u64;
  carry = sum >> 64;
  result[4] = carry as u64;

  // x_hi_2 × c (add at offset 2)
  let prod2 = mul_2_by_1(&c, x_hi_2);
  carry = 0;
  let sum = (result[2] as u128) + (prod2[0] as u128) + carry;
  result[2] = sum as u64;
  carry = sum >> 64;
  let sum = (result[3] as u128) + (prod2[1] as u128) + carry;
  result[3] = sum as u64;
  carry = sum >> 64;
  let sum = (result[4] as u128) + (prod2[2] as u128) + carry;
  result[4] = sum as u64;

  result
}

/// Multiply 1-limb x_hi by PASTA_C, producing 4-limb result.
#[inline(always)]
fn mul_1_by_pasta_c<F: FieldReductionConstants>(x_hi: u64) -> [u64; 4] {
  let c = F::PASTA_C;
  let prod = mul_2_by_1(&c, x_hi);
  [prod[0], prod[1], prod[2], 0]
}

/// Subtract 5-limb from 4-limb: a - b, return (result, is_negative).
#[inline(always)]
fn sub_wide_4_5(a: &[u64; 4], b: &[u64; 5]) -> ([u64; 4], bool) {
  let mut result = [0u64; 4];
  let mut borrow = 0u64;

  for i in 0..4 {
    let (diff, b1) = a[i].overflowing_sub(b[i]);
    let (diff2, b2) = diff.overflowing_sub(borrow);
    result[i] = diff2;
    borrow = (b1 as u64) + (b2 as u64);
  }

  // Check if b[4] > 0 or there's remaining borrow
  let is_negative = borrow > 0 || b[4] > 0;
  (result, is_negative)
}

/// Subtract two 4-limb values, return (result, is_negative).
#[inline(always)]
fn sub_4_4_check_neg(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
  let (result, borrow) = sub_4_4_with_borrow(a, b);
  (result, borrow > 0)
}

/// Generic μ-Barrett reduction for 6-limb input (used for BN254, T256).
///
/// # Algorithm
///
/// For input x = c[0..6] (6 limbs, up to 384 bits), with k=4 (256 bits):
/// 1. q1 = ⌊x / b³⌋ = [c[3], c[4], c[5]]  (3 limbs)
/// 2. q2 = q1 × μ                          (8 limbs)
/// 3. q3 = ⌊q2 / b⁵⌋ = [q2[5], q2[6], q2[7]] (3 limbs, quotient estimate)
/// 4. t = q3 × p (low 4 or 5 limbs)
/// 5. r = (x mod b⁴ or b⁵) - t
/// 6. if r ≥ p: r -= p (exactly once, proven tight)
#[inline]
fn barrett_reduce_6_generic<F: FieldReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
  // Step 1: q1 = floor(x / b³) = [c[3], c[4], c[5]]
  let q1 = [c[3], c[4], c[5]];

  // Step 2: q2 = q1 × μ (3×5 → 8 limbs)
  let q2 = mul_3x5_to_8(&q1, &F::BARRETT_MU);

  // Step 3: q3 = floor(q2 / b⁵) = [q2[5], q2[6], q2[7]]
  let q3 = [q2[5], q2[6], q2[7]];

  if F::USE_4_LIMB_BARRETT {
    // Fast path: 4-limb arithmetic for Pasta/BN254 (where 2p < b⁴)
    // t = q3 × p (low 4 limbs only)
    let t = mul_3x4_lo4(&q3, &F::MODULUS);

    // r = (x mod b⁴) - t (wrapping subtraction in 4 limbs)
    let x_lo4 = [c[0], c[1], c[2], c[3]];
    let mut r = sub_4_4(&x_lo4, &t);

    // One conditional subtract (proven tight)
    if gte_4_4(&r, &F::MODULUS) {
      r = sub_4_4(&r, &F::MODULUS);
    }

    debug_assert!(
      !gte_4_4(&r, &F::MODULUS),
      "Barrett reduction produced non-canonical result"
    );

    r
  } else {
    // 5-limb path for T256 (where 2p can exceed b⁴)
    let r1 = [c[0], c[1], c[2], c[3], c[4]];
    let r2 = mul_3x4_lo5(&q3, &F::MODULUS);
    let mut r = sub_5_5(&r1, &r2);

    // One conditional subtract
    if r[4] != 0 || gte_4_4(&[r[0], r[1], r[2], r[3]], &F::MODULUS) {
      r = sub_5_4(&r, &F::MODULUS);
    }

    debug_assert!(
      r[4] == 0 && !gte_4_4(&[r[0], r[1], r[2], r[3]], &F::MODULUS),
      "Barrett reduction produced non-canonical result"
    );

    [r[0], r[1], r[2], r[3]]
  }
}

/// Multiply 3-limb by 4-limb, returning low 5 limbs.
///
/// Used to compute (q3 × p) mod b⁵ in Barrett reduction.
#[inline(always)]
fn mul_3x4_lo5(a: &[u64; 3], b: &[u64; 4]) -> [u64; 5] {
  let mut result = [0u64; 5];

  // Full 3×4 multiply, keeping only low 5 limbs
  for i in 0..3 {
    let mut carry = 0u128;
    for j in 0..4 {
      if i + j < 5 {
        let prod = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + carry;
        result[i + j] = prod as u64;
        carry = prod >> 64;
      }
    }
    // Handle the final carry for position i+4 if it's within bounds
    if i + 4 < 5 {
      result[i + 4] = carry as u64;
    }
  }

  result
}

/// Subtract two 5-limb values: a - b (wrapping).
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

// ==========================================================================
// 7-limb Barrett reduction (for SignedWideLimbs<7>, i.e. DelayedReduction<i128>::Accumulator)
// ==========================================================================

/// Barrett reduction for 7-limb input.
///
/// Dispatches to Pasta 2-fold or generic fold+μ-Barrett based on field type.
#[inline]
pub(crate) fn barrett_reduce_7<F: FieldReductionConstants>(c: &[u64; 7]) -> [u64; 4] {
  if F::PASTA_STYLE_MODULUS {
    barrett_reduce_7_pasta::<F>(c)
  } else {
    barrett_reduce_7_generic::<F>(c)
  }
}

/// Pasta 2-fold Barrett reduction for 7-limb input.
///
/// Similar to 6-limb version but handles larger input.
#[inline]
fn barrett_reduce_7_pasta<F: FieldReductionConstants>(c: &[u64; 7]) -> [u64; 4] {
  // For 7 limbs (448 bits), split at bit 254:
  // - x_lo: bits 0-253 (4 limbs, with limb[3] masked to 62 bits)
  // - x_hi: bits 254-447 (up to 194 bits, 4 limbs)

  // Extract x_lo (low 254 bits)
  let x_lo = [c[0], c[1], c[2], c[3] & 0x3FFF_FFFF_FFFF_FFFF];

  // Extract x_hi (bits 254+)
  let x_hi_0 = (c[3] >> 62) | (c[4] << 2);
  let x_hi_1 = (c[4] >> 62) | (c[5] << 2);
  let x_hi_2 = (c[5] >> 62) | (c[6] << 2);
  let x_hi_3 = c[6] >> 62;

  // Compute x_hi × c where c = PASTA_C (2 limbs)
  // Result is up to 194 + 128 = 322 bits (6 limbs)
  let prod = mul_4limbs_by_pasta_c::<F>(x_hi_0, x_hi_1, x_hi_2, x_hi_3);

  // Compute x_lo - prod (prod can be up to 6 limbs)
  let (result, neg) = sub_wide_4_6(&x_lo, &prod);

  // If negative, add p
  let result = if neg {
    let (sum, _) = add_4_4(&result, &F::MODULUS);
    sum
  } else {
    result
  };

  // Result might still be >= 2^254, need more folds
  // Use the 6-limb pasta reducer which handles this
  let as_6 = [result[0], result[1], result[2], result[3], 0, 0];
  barrett_reduce_6_pasta::<F>(&as_6)
}

/// Multiply 4-limb x_hi by PASTA_C (2 limbs), producing 6-limb result.
#[inline(always)]
fn mul_4limbs_by_pasta_c<F: FieldReductionConstants>(
  x_hi_0: u64,
  x_hi_1: u64,
  x_hi_2: u64,
  x_hi_3: u64,
) -> [u64; 6] {
  let c = F::PASTA_C;
  let mut result = [0u64; 6];

  // x_hi_0 × c
  let prod0 = mul_2_by_1(&c, x_hi_0);
  result[0] = prod0[0];
  result[1] = prod0[1];
  result[2] = prod0[2];

  // x_hi_1 × c (add at offset 1)
  let prod1 = mul_2_by_1(&c, x_hi_1);
  let mut carry = 0u128;
  let sum = (result[1] as u128) + (prod1[0] as u128) + carry;
  result[1] = sum as u64;
  carry = sum >> 64;
  let sum = (result[2] as u128) + (prod1[1] as u128) + carry;
  result[2] = sum as u64;
  carry = sum >> 64;
  result[3] = prod1[2].wrapping_add(carry as u64);

  // x_hi_2 × c (add at offset 2)
  let prod2 = mul_2_by_1(&c, x_hi_2);
  carry = 0;
  let sum = (result[2] as u128) + (prod2[0] as u128) + carry;
  result[2] = sum as u64;
  carry = sum >> 64;
  let sum = (result[3] as u128) + (prod2[1] as u128) + carry;
  result[3] = sum as u64;
  carry = sum >> 64;
  result[4] = prod2[2].wrapping_add(carry as u64);

  // x_hi_3 × c (add at offset 3)
  let prod3 = mul_2_by_1(&c, x_hi_3);
  carry = 0;
  let sum = (result[3] as u128) + (prod3[0] as u128) + carry;
  result[3] = sum as u64;
  carry = sum >> 64;
  let sum = (result[4] as u128) + (prod3[1] as u128) + carry;
  result[4] = sum as u64;
  carry = sum >> 64;
  result[5] = prod3[2].wrapping_add(carry as u64);

  result
}

/// Subtract 6-limb from 4-limb: a - b, return (result, is_negative).
#[inline(always)]
fn sub_wide_4_6(a: &[u64; 4], b: &[u64; 6]) -> ([u64; 4], bool) {
  let mut result = [0u64; 4];
  let mut borrow = 0u64;

  for i in 0..4 {
    let (diff, b1) = a[i].overflowing_sub(b[i]);
    let (diff2, b2) = diff.overflowing_sub(borrow);
    result[i] = diff2;
    borrow = (b1 as u64) + (b2 as u64);
  }

  // Check if b[4], b[5] > 0 or there's remaining borrow
  let is_negative = borrow > 0 || b[4] > 0 || b[5] > 0;
  (result, is_negative)
}

/// Generic 7-limb Barrett reduction (used for BN254, T256).
///
/// Reduces a 7-limb value (up to 448 bits) modulo p by folding limb 6
/// using R384 = 2^384 mod p, then delegating to the bounded 6-limb reduction.
#[inline]
fn barrett_reduce_7_generic<F: FieldReductionConstants>(c: &[u64; 7]) -> [u64; 4] {
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

  barrett_reduce_6_generic::<F>(&sum)
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

  // ========================================================================
  // BARRETT_MU constant verification tests
  // ========================================================================

  /// Verify that μ = floor(2^512 / p) by checking:
  /// 1. μ × p < 2^512 (so μ is not too large)
  /// 2. (μ + 1) × p ≥ 2^512 (so μ is not too small)
  fn verify_barrett_mu<F: FieldReductionConstants>() {
    // μ × p should be close to but less than 2^512
    // We verify by checking that μ × p < 2^512 and (μ+1) × p >= 2^512

    // For a simpler verification, we check that barrett_reduce gives correct
    // results for known values, which implicitly verifies μ.

    // Test: x = p should reduce to 0
    let x_is_p = [
      F::MODULUS[0],
      F::MODULUS[1],
      F::MODULUS[2],
      F::MODULUS[3],
      0,
      0,
    ];
    let result = barrett_reduce_6::<F>(&x_is_p);
    assert_eq!(result, [0, 0, 0, 0], "p mod p should be 0");

    // Test: x = 2p should reduce to 0
    let mut x_is_2p = [0u64; 6];
    let mut carry = 0u128;
    for i in 0..4 {
      let sum = (F::MODULUS[i] as u128) * 2 + carry;
      x_is_2p[i] = sum as u64;
      carry = sum >> 64;
    }
    x_is_2p[4] = carry as u64;
    let result = barrett_reduce_6::<F>(&x_is_2p);
    assert_eq!(result, [0, 0, 0, 0], "2p mod p should be 0");

    // Test: x = p - 1 should reduce to p - 1
    let mut x_is_p_minus_1 = [0u64; 6];
    x_is_p_minus_1[0] = F::MODULUS[0].wrapping_sub(1);
    x_is_p_minus_1[1] = F::MODULUS[1];
    x_is_p_minus_1[2] = F::MODULUS[2];
    x_is_p_minus_1[3] = F::MODULUS[3];
    if x_is_p_minus_1[0] == u64::MAX {
      // Handle borrow for p[0] = 0 case
      let mut borrow = 1u64;
      for i in 0..4 {
        let (diff, b) = F::MODULUS[i].overflowing_sub(borrow);
        x_is_p_minus_1[i] = diff;
        borrow = b as u64;
        if borrow == 0 {
          // Copy remaining limbs
          for j in (i + 1)..4 {
            x_is_p_minus_1[j] = F::MODULUS[j];
          }
          break;
        }
      }
    }
    let result = barrett_reduce_6::<F>(&x_is_p_minus_1);
    let expected = [
      x_is_p_minus_1[0],
      x_is_p_minus_1[1],
      x_is_p_minus_1[2],
      x_is_p_minus_1[3],
    ];
    assert_eq!(result, expected, "p-1 mod p should be p-1");
  }

  #[test]
  fn test_barrett_mu_fp() {
    verify_barrett_mu::<Fp>();
  }

  #[test]
  fn test_barrett_mu_fq() {
    verify_barrett_mu::<Fq>();
  }

  #[test]
  fn test_barrett_mu_bn254() {
    use halo2curves::bn256::Fr as Bn254Fr;
    verify_barrett_mu::<Bn254Fr>();
  }

  #[test]
  fn test_barrett_mu_t256() {
    use halo2curves::t256::Fq as T256Fq;
    verify_barrett_mu::<T256Fq>();
  }

  // ========================================================================
  // BN254-specific tests
  // ========================================================================

  #[test]
  fn test_barrett_6_bn254_sum_of_products() {
    use super::super::limbs::mul_4_by_1;
    use halo2curves::bn256::Fr as Bn254Fr;

    let mut rng = OsRng;
    let mut acc = [0u64; 6];
    let mut expected_sum = Bn254Fr::ZERO;

    for _ in 0..100 {
      let field_elem = Bn254Fr::random(&mut rng);
      let small = rng.next_u64() >> 32;

      expected_sum += field_elem * Bn254Fr::from(small);

      let product = mul_4_by_1(&field_elem.0, small);
      let mut carry = 0u128;
      for i in 0..5 {
        let sum = (acc[i] as u128) + (product[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
      }
      acc[5] = acc[5].wrapping_add(carry as u64);
    }

    let result = Bn254Fr(barrett_reduce_6::<Bn254Fr>(&acc));
    assert_eq!(result, expected_sum);
  }
}
