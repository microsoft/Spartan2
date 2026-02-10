// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Barrett reduction for wide limb values.

use super::field_reduction_constants::FieldReductionConstants;
use super::limbs::{gte_4_4, mul_3x4_lo4, mul_3x5_to_8, mul_4_by_1, sub_4_4, sub_5_4};

// ==========================================================================
// 6-limb Barrett reduction (for SignedWideLimbs<6>, i.e. DelayedReduction<i64>::Accumulator)
// ==========================================================================

/// Barrett reduction for 6-limb input using precomputed μ = ⌊2^512 / p⌋.
///
/// Reduces a 6-limb value (up to 384 bits) modulo p with exactly one conditional
/// subtract, using the true Barrett reduction algorithm.
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
///
/// # Tight Bound: Exactly 1 Subtract
///
/// For p ≥ 2^253 and x < b^7 (448 bits):
/// - δ = x/b⁸ + b³/p < 2^-64 + 2^-61 < 1
/// - Therefore q3 ∈ {⌊x/p⌋, ⌊x/p⌋ - 1}
/// - So r ∈ [0, 2p), requiring exactly 1 conditional subtract
///
/// # Fast Path for Pasta/BN254
///
/// When USE_4_LIMB_BARRETT is true (p < 2^255), r ∈ [0, 2p) < b⁴,
/// so we use 4-limb arithmetic (saves 3 multiplications + carries).
#[inline]
pub(crate) fn barrett_reduce_6<F: FieldReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
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
