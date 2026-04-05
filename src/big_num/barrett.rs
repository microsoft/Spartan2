// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Barrett reduction for wide limb values.
//!
//! # The Problem
//!
//! Computing `x mod p` normally requires division, which is expensive (~30 cycles
//! for 64-bit div vs ~3-4 cycles for mul). When accumulating many field × small_int
//! products, we want to reduce once at the end rather than after each multiply.
//!
//! # Barrett's Algorithm
//!
//! Replace division by `p` with multiplication by a precomputed reciprocal:
//!
//! ```text
//! μ = ⌊2^k / p⌋           // precomputed "magic" constant
//!
//! To compute x mod p:
//! 1. q = ⌊(x × μ) / 2^k⌋  // estimate quotient via multiply + shift
//! 2. r = x - q × p         // remainder = x - quotient × modulus
//! 3. if r ≥ p: r -= p      // at most 1-2 conditional subtracts
//! ```
//!
//! # Why Barrett (not Montgomery) for small values?
//!
//! - **Montgomery** requires values in Montgomery form (multiplied by R = 2^256).
//!   Great for field × field since both operands are already in Montgomery form.
//!
//! - **Barrett** works on raw limbs directly. When multiplying field × small_int,
//!   the small integer (e.g., i32) is NOT in Montgomery form. Barrett lets us:
//!   1. Accumulate raw products: `Σ (field_limbs × small_int)` as wide integers
//!   2. Reduce once at the end with Barrett
//!
//! This avoids N Montgomery conversions for N small values.
//!
//! # Constants
//!
//! | Constant | Value | Purpose |
//! |----------|-------|---------|
//! | `BARRETT_MU` | `⌊2^512 / p⌋` | Reciprocal for quotient estimate |
//! | `R384_MOD` | `2^384 mod p` | Fold 7-limb input to 6 limbs first |
//! | `USE_4_LIMB_BARRETT` | `2p < 2^256?` | Fast path when remainder fits 4 limbs |

use super::{
  field_reduction_constants::BarrettReductionConstants,
  limbs::{add, gte, mul_3x4_lo4, mul_3x4_lo5, mul_3x5_to_8, mul_4_by_1, sub, sub_5_4},
};

// ==========================================================================
// Generic μ-Barrett reduction (for BN254, P256, T256, Pasta)
// ==========================================================================

/// Barrett reduction for 6-limb input.
///
/// Uses μ-Barrett algorithm with BARRETT_MU reciprocal.
#[inline]
pub(crate) fn barrett_reduce_6<F: BarrettReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
  // Step 1: q1 = floor(x / b³) = [c[3], c[4], c[5]]
  let q1 = [c[3], c[4], c[5]];

  // Step 2: q2 = q1 × μ (3×5 → 8 limbs)
  let q2 = mul_3x5_to_8(&q1, &F::BARRETT_MU);

  // Step 3: q3 = floor(q2 / b⁵) = [q2[5], q2[6], q2[7]]
  let q3 = [q2[5], q2[6], q2[7]];

  if F::USE_4_LIMB_BARRETT {
    // Fast path: 4-limb arithmetic for BN254 (where 2p < b⁴)
    // t = q3 × p (low 4 limbs only)
    let t = mul_3x4_lo4(&q3, &F::MODULUS);

    // r = (x mod b⁴) - t (wrapping subtraction in 4 limbs)
    let x_lo4 = [c[0], c[1], c[2], c[3]];
    let mut r = sub::<4>(&x_lo4, &t);

    // One conditional subtract (proven tight)
    if gte::<4>(&r, &F::MODULUS) {
      r = sub::<4>(&r, &F::MODULUS);
    }

    debug_assert!(
      !gte::<4>(&r, &F::MODULUS),
      "Barrett reduction produced non-canonical result"
    );

    r
  } else {
    // 5-limb path for T256 (where 2p can exceed b⁴)
    let r1 = [c[0], c[1], c[2], c[3], c[4]];
    let r2 = mul_3x4_lo5(&q3, &F::MODULUS);
    let mut r = sub::<5>(&r1, &r2);

    // One conditional subtract
    if r[4] != 0 || gte::<4>(&[r[0], r[1], r[2], r[3]], &F::MODULUS) {
      r = sub_5_4(&r, &F::MODULUS);
    }

    debug_assert!(
      r[4] == 0 && !gte::<4>(&[r[0], r[1], r[2], r[3]], &F::MODULUS),
      "Barrett reduction produced non-canonical result"
    );

    [r[0], r[1], r[2], r[3]]
  }
}

/// Barrett reduction for 7-limb input.
///
/// We intentionally do not run a "direct" 7-limb Barrett here. That direct
/// form would:
/// 1. take `q1 = [c[3], c[4], c[5], c[6]]`,
/// 2. compute `q2 = q1 × μ` with a 4×5→9 multiply,
/// 3. take `q3 = floor(q2 / b^5)`,
/// 4. compute `q3 × p` with a low 4×4 multiply on the fast path.
///
/// For our supported 254-256 bit fields, folding is cheaper on the fast path:
/// - fold-based path here = `mul_4_by_1` (4 multiplies) + `barrett_reduce_6`
///   fast path (`mul_3x5_to_8`: 15 multiplies, `mul_3x4_lo4`: 9 multiplies)
///   = 28 limb multiplies total,
/// - direct 7-limb Barrett = `mul_4x5_to_9` (20 multiplies) +
///   `mul_4x4_lo4` (10 multiplies) = 30 limb multiplies total.
///
/// So we fold `c[6]·2^384` into the low 6 limbs with `2^384 ≡ R384_MOD (mod p)`
/// and reuse the already-tuned 6-limb reducer. If that fold overflows limb 5,
/// the lost bit is one extra `2^384` term, so we repair it by adding
/// `R384_MOD` after the 6-limb reduction.
#[inline]
pub(crate) fn barrett_reduce_7<F: BarrettReductionConstants>(c: &[u64; 7]) -> [u64; 4] {
  // Fold limb 6: c[6] × 2^384 ≡ c[6] × R384_MOD (mod p)
  let c6_contrib = mul_4_by_1(&F::R384_MOD, c[6]);

  // Add the folded contribution into the low 6 limbs. We intentionally route
  // the result through barrett_reduce_6 because that path is already tuned for
  // our supported fields; the only extra case we must handle is a carry out of
  // limb 5, which represents one more 2^384 term.
  let (s0, cy) = c[0].carrying_add(c6_contrib[0], false);
  let (s1, cy) = c[1].carrying_add(c6_contrib[1], cy);
  let (s2, cy) = c[2].carrying_add(c6_contrib[2], cy);
  let (s3, cy) = c[3].carrying_add(c6_contrib[3], cy);
  let (s4, cy) = c[4].carrying_add(c6_contrib[4], cy);
  let (s5, extra) = c[5].carrying_add(0, cy);

  let reduced = barrett_reduce_6::<F>(&[s0, s1, s2, s3, s4, s5]);
  if extra {
    // The dropped carry is one extra 2^384 term, so add back 2^384 mod p.
    add_r384_mod::<F>(reduced)
  } else {
    reduced
  }
}

#[inline]
fn add_r384_mod<F: BarrettReductionConstants>(r: [u64; 4]) -> [u64; 4] {
  if F::USE_4_LIMB_BARRETT {
    let (mut sum, carry) = add::<4>(&r, &F::R384_MOD);
    debug_assert_eq!(carry, 0, "R384_MOD repair overflowed the 4-limb fast path");

    if gte::<4>(&sum, &F::MODULUS) {
      sum = sub::<4>(&sum, &F::MODULUS);
    }

    sum
  } else {
    let (sum_lo4, carry) = add::<4>(&r, &F::R384_MOD);
    let mut sum = [sum_lo4[0], sum_lo4[1], sum_lo4[2], sum_lo4[3], carry];

    if sum[4] != 0 || gte::<4>(&sum_lo4, &F::MODULUS) {
      sum = sub_5_4(&sum, &F::MODULUS);
    }

    debug_assert!(
      sum[4] == 0 && !gte::<4>(&[sum[0], sum[1], sum[2], sum[3]], &F::MODULUS),
      "R384_MOD repair produced non-canonical result"
    );

    [sum[0], sum[1], sum[2], sum[3]]
  }
}

// ==========================================================================
// Test helpers and macro (exported for use by provider test modules)
// ==========================================================================

#[cfg(test)]
use super::montgomery::MontgomeryLimbs;
#[cfg(test)]
use num_bigint::BigUint;

#[cfg(test)]
fn limbs_to_biguint<const N: usize>(limbs: &[u64; N]) -> BigUint {
  let mut bytes = Vec::with_capacity(N * 8);
  for limb in limbs {
    bytes.extend_from_slice(&limb.to_le_bytes());
  }
  BigUint::from_bytes_le(&bytes)
}

#[cfg(test)]
fn biguint_to_limbs4(value: &BigUint) -> [u64; 4] {
  let mut bytes = value.to_bytes_le();
  bytes.resize(32, 0);

  let mut limbs = [0u64; 4];
  for (i, limb) in limbs.iter_mut().enumerate() {
    let mut chunk = [0u8; 8];
    let start = i * 8;
    chunk.copy_from_slice(&bytes[start..start + 8]);
    *limb = u64::from_le_bytes(chunk);
  }

  limbs
}

/// Test that reducing zero gives zero.
#[cfg(test)]
pub(crate) fn test_barrett_zero_impl<R>(reduce_6: R)
where
  R: Fn(&[u64; 6]) -> [u64; 4],
{
  let c = [0u64; 6];
  let result = reduce_6(&c);
  assert_eq!(result, [0, 0, 0, 0]);
}

/// Test single product reduction.
#[cfg(test)]
pub(crate) fn test_barrett_single_product_impl<F, R>(reduce_6: R)
where
  F: ff::Field + ff::PrimeField + MontgomeryLimbs,
  R: Fn(&[u64; 6]) -> [u64; 4],
{
  let field_elem = F::from(12345u64);
  let small = 9999u64;

  let product = mul_4_by_1(field_elem.to_limbs(), small);
  let c = [
    product[0], product[1], product[2], product[3], product[4], 0,
  ];

  let result = F::from_limbs(reduce_6(&c));
  let expected = field_elem * F::from(small);
  assert_eq!(result, expected);
}

/// Test sum of 100 products.
#[cfg(test)]
pub(crate) fn test_barrett_sum_of_products_impl<F, R>(reduce_6: R)
where
  F: ff::Field + ff::PrimeField + MontgomeryLimbs,
  R: Fn(&[u64; 6]) -> [u64; 4],
{
  use rand_core::{OsRng, RngCore};

  let mut rng = OsRng;
  let mut acc = [0u64; 6];
  let mut expected_sum = F::ZERO;

  for _ in 0..100 {
    let field_elem = F::random(&mut rng);
    let small = rng.next_u64() >> 32;

    expected_sum += field_elem * F::from(small);

    let product = mul_4_by_1(field_elem.to_limbs(), small);
    let mut carry = 0u128;
    for i in 0..5 {
      let sum = (acc[i] as u128) + (product[i] as u128) + carry;
      acc[i] = sum as u64;
      carry = sum >> 64;
    }
    acc[5] = acc[5].wrapping_add(carry as u64);
  }

  let result = F::from_limbs(reduce_6(&acc));
  assert_eq!(result, expected_sum);
}

/// Stress test with 2000 products (release builds only).
#[cfg(all(test, not(debug_assertions)))]
pub(crate) fn test_barrett_many_products_impl<F, R>(reduce_6: R)
where
  F: ff::Field + ff::PrimeField + MontgomeryLimbs,
  R: Fn(&[u64; 6]) -> [u64; 4],
{
  use rand_core::{OsRng, RngCore};

  let mut rng = OsRng;
  let mut acc = [0u64; 6];
  let mut expected_sum = F::ZERO;

  for _ in 0..2000 {
    let field_elem = F::random(&mut rng);
    let small = rng.next_u64();

    expected_sum += field_elem * F::from(small);

    let product = mul_4_by_1(field_elem.to_limbs(), small);
    let mut carry = 0u128;
    for i in 0..5 {
      let sum = (acc[i] as u128) + (product[i] as u128) + carry;
      acc[i] = sum as u64;
      carry = sum >> 64;
    }
    acc[5] = acc[5].wrapping_add(carry as u64);
  }

  let result = F::from_limbs(reduce_6(&acc));
  assert_eq!(result, expected_sum);
}

/// Regression for the original 7-limb fold bug.
///
/// The old implementation dropped the carry after folding the 7th limb with
/// `2^384 mod p`. This vector is `x = 2^385 - 1` and reliably triggers that path.
#[cfg(test)]
pub(crate) fn test_barrett_reduce_7_regression_impl<F>()
where
  F: BarrettReductionConstants,
{
  let c = [u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX, 1];

  let modulus = limbs_to_biguint(&F::MODULUS);
  let x = limbs_to_biguint(&c);
  let expected = biguint_to_limbs4(&(x % &modulus));
  let actual = barrett_reduce_7::<F>(&c);
  assert_eq!(actual, expected, "barrett_reduce_7 mismatch for input {:?}", c);
}

/// Generate tests for Barrett reduction functions.
///
/// # Example
/// ```ignore
/// crate::test_barrett_reduction!(scalar_br, Scalar, crate::big_num::barrett::barrett_reduce_6::<Scalar>);
/// ```
#[cfg(test)]
#[macro_export]
macro_rules! test_barrett_reduction {
  ($mod_name:ident, $field:ty, $reduce_fn:expr) => {
    mod $mod_name {
      #[test]
      fn zero() {
        $crate::big_num::barrett::test_barrett_zero_impl($reduce_fn);
      }
      #[test]
      fn single_product() {
        $crate::big_num::barrett::test_barrett_single_product_impl::<$field, _>($reduce_fn);
      }
      #[test]
      fn sum_of_products() {
        $crate::big_num::barrett::test_barrett_sum_of_products_impl::<$field, _>($reduce_fn);
      }
      #[test]
      #[cfg(not(debug_assertions))]
      fn many_products() {
        $crate::big_num::barrett::test_barrett_many_products_impl::<$field, _>($reduce_fn);
      }
    }
  };
}

/// Generate the targeted regression test for the 7-limb Barrett fold carry bug.
#[cfg(test)]
#[macro_export]
macro_rules! test_barrett_reduction_7 {
  ($mod_name:ident, $field:ty) => {
    mod $mod_name {
      #[test]
      fn fold_carry_regression() {
        $crate::big_num::barrett::test_barrett_reduce_7_regression_impl::<$field>();
      }
    }
  };
}
