// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Barrett reduction for wide limb values.

use super::field_reduction_constants::FieldReductionConstants;
use super::limbs::{gte_4_4, gte_5_4, mul_4_by_1, mul_5_by_1, sub_4_4, sub_5_4, sub_5_5};

// ==========================================================================
// 5-limb Barrett reduction
// ==========================================================================

/// Generic 5-limb Barrett reduction using trait constants.
/// Reduces a 5-limb value (up to 320 bits) modulo p.
#[inline(always)]
pub(crate) fn barrett_reduce_5<F: FieldReductionConstants>(c: &[u64; 5]) -> [u64; 4] {
  let c_tilde = (c[3] >> 63) | (c[4] << 1);
  let m = {
    let product = (c_tilde as u128) * (F::MU as u128);
    (product >> 64) as u64
  };
  let m_times_2p = mul_5_by_1(&F::MODULUS_2P, m);
  let mut r = sub_5_5(c, &m_times_2p);
  // At most 2 iterations needed: after Barrett approximation, 0 <= r < 2p
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
pub(crate) fn barrett_reduce_6<F: FieldReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
  // Reduce c[5] * 2^320 = c[5] * R320 (mod p), then reduce c[4] * 2^256, etc.
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

// ==========================================================================
// 7-limb Barrett reduction (for i64/i128 UnreducedFieldInt accumulator)
// ==========================================================================

/// Generic 7-limb Barrett reduction using trait constants.
///
/// Reduces a 7-limb value (up to 448 bits) modulo p by folding limb 6
/// using R384 = 2^384 mod p, then delegating to 6-limb reduction.
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
  use crate::small_field::{mul_by_i64, mul_by_u64};
  use ff::Field;
  use halo2curves::pasta::{Fp, Fq};
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
