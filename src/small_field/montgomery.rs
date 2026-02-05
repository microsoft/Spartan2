// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Montgomery form operations: limb access and REDC reduction.

use super::{
  field_reduction_constants::FieldReductionConstants,
  limbs::{gte_4_4, mul_4_by_1, sub_5_4},
};
use halo2curves::{
  bn256::Fr as Bn254Fr,
  pasta::{Fp, Fq},
  t256::Fq as T256Fq,
};

// ==========================================================================
// MontgomeryLimbs - Trait for accessing Montgomery-form limbs
// ==========================================================================

/// Trait for field types that expose their internal Montgomery-form limbs.
///
/// Field elements are stored as `value * R mod p` where R = 2^256.
/// This trait provides direct access to those R-scaled limbs.
pub(crate) trait MontgomeryLimbs: FieldReductionConstants {
  /// Construct a field element from 4 Montgomery-form limbs.
  fn from_limbs(limbs: [u64; 4]) -> Self;

  /// Access the internal Montgomery-form limbs.
  fn to_limbs(&self) -> &[u64; 4];
}

impl MontgomeryLimbs for Fp {
  #[inline]
  fn from_limbs(limbs: [u64; 4]) -> Self {
    Fp(limbs)
  }

  #[inline]
  fn to_limbs(&self) -> &[u64; 4] {
    &self.0
  }
}

impl MontgomeryLimbs for Fq {
  #[inline]
  fn from_limbs(limbs: [u64; 4]) -> Self {
    Fq(limbs)
  }

  #[inline]
  fn to_limbs(&self) -> &[u64; 4] {
    &self.0
  }
}

impl MontgomeryLimbs for Bn254Fr {
  #[inline]
  fn from_limbs(limbs: [u64; 4]) -> Self {
    Bn254Fr(limbs)
  }

  #[inline]
  fn to_limbs(&self) -> &[u64; 4] {
    &self.0
  }
}

impl MontgomeryLimbs for T256Fq {
  #[inline]
  fn from_limbs(limbs: [u64; 4]) -> Self {
    T256Fq(limbs)
  }

  #[inline]
  fn to_limbs(&self) -> &[u64; 4] {
    &self.0
  }
}

// ==========================================================================
// 9-limb Montgomery REDC (for WideLimbs<9>, i.e. DelayedReduction<F>::Accumulator)
// ==========================================================================

/// Generic Montgomery REDC for 9-limb input using trait constants.
///
/// Reduces a 2R-scaled value (sum of field x field products) to 1R-scaled.
/// Input: T representing x*R^2 (up to 9 limbs)
/// Output: x*R mod p (4 limbs, standard Montgomery form)
#[inline]
pub(crate) fn montgomery_reduce_9<F: FieldReductionConstants>(c: &[u64; 9]) -> [u64; 4] {
  // Step 1: Reduce 9 limbs to 8 limbs using precomputed 2^512 mod p
  let mut t = [0u64; 9];
  if c[8] == 0 {
    t[..8].copy_from_slice(&c[..8]);
  } else {
    // t = c[0..8] + c[8] * R512_MOD
    let high_contribution = mul_4_by_1(&F::R512_MOD, c[8]);
    let mut carry = 0u128;
    for i in 0..5 {
      let sum = (c[i] as u128) + (high_contribution[i] as u128) + carry;
      t[i] = sum as u64;
      carry = sum >> 64;
    }
    for i in 5..8 {
      let sum = (c[i] as u128) + carry;
      t[i] = sum as u64;
      carry = sum >> 64;
    }
    t[8] = carry as u64;

    // Recurse if still > 8 limbs
    if t[8] > 0 {
      return montgomery_reduce_9::<F>(&t);
    }
  }

  // Step 2: Montgomery REDC on 8-limb value
  montgomery_reduce_8::<F>(&[t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]])
}

/// Generic Montgomery REDC for 8-limb input.
/// Standard Montgomery reduction: T x R^(-1) mod p
#[inline]
fn montgomery_reduce_8<F: FieldReductionConstants>(t: &[u64; 8]) -> [u64; 4] {
  // Use 9 limbs to track overflow
  let mut r = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], 0u64];

  // Montgomery reduction: for each of the low 4 limbs, eliminate it
  // by adding appropriate multiples of p
  for i in 0..4 {
    // q = r[i] * p' mod 2^64
    let q = r[i].wrapping_mul(F::MONT_INV);

    // r += q * p * 2^(64*i)
    // qp = q * p, which is 5 limbs (since p is 4 limbs and q is 1 limb)
    let qp = mul_4_by_1(&F::MODULUS, q);

    let mut carry = 0u128;
    for j in 0..5 {
      let sum = (r[i + j] as u128) + (qp[j] as u128) + carry;
      r[i + j] = sum as u64;
      carry = sum >> 64;
    }
    // Propagate remaining carry through the rest of the array
    for item in r[(i + 5)..9].iter_mut() {
      let sum = (*item as u128) + carry;
      *item = sum as u64;
      carry = sum >> 64;
      if carry == 0 {
        break;
      }
    }
  }

  // Now r[0..4] should be zero (by construction), result is in r[4..9]
  // We need to reduce this to [0, p)
  let mut result = [r[4], r[5], r[6], r[7], r[8]];

  // Reduce until result < p
  while result[4] > 0 || gte_4_4(&[result[0], result[1], result[2], result[3]], &F::MODULUS) {
    let sub = sub_5_4(&result, &F::MODULUS);
    result = sub;
  }

  [result[0], result[1], result[2], result[3]]
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::small_field::limbs::mul_4_by_4;
  use ff::Field;
  use halo2curves::pasta::{Fp, Fq};
  use rand_core::OsRng;

  #[test]
  fn test_montgomery_9_fp_single_product() {
    // Test with a single field x field product
    let a = Fp::from(12345u64);
    let b = Fp::from(67890u64);

    // Compute a_mont x b_mont (8 limbs, representing (a*b)*R^2 in unreduced form)
    let product = mul_4_by_4(&a.0, &b.0);

    // Extend to 9 limbs
    let c = [
      product[0], product[1], product[2], product[3], product[4], product[5], product[6],
      product[7], 0,
    ];

    // Montgomery reduce: should give (a*b)*R mod p = (a*b) in Montgomery form
    let result = Fp(montgomery_reduce_9::<Fp>(&c));

    // Expected: a * b in field
    let expected = a * b;
    assert_eq!(result, expected);
  }

  #[test]
  fn test_montgomery_9_fp_sum_of_products() {
    // Test summing multiple field x field products
    let mut rng = OsRng;
    let mut acc = [0u64; 9];
    let mut expected_sum = Fp::ZERO;

    // Sum 100 products
    for _ in 0..100 {
      let a = Fp::random(&mut rng);
      let b = Fp::random(&mut rng);

      // Accumulate expected result
      expected_sum += a * b;

      // Compute a_mont x b_mont as 8 limbs
      let product = mul_4_by_4(&a.0, &b.0);

      // Add to accumulator with carry propagation
      let mut carry = 0u128;
      for i in 0..8 {
        let sum = (acc[i] as u128) + (product[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
      }
      acc[8] = acc[8].wrapping_add(carry as u64);
    }

    // Montgomery reduce and compare
    let result = Fp(montgomery_reduce_9::<Fp>(&acc));
    assert_eq!(result, expected_sum);
  }

  #[test]
  fn test_montgomery_9_fp_many_products() {
    // Stress test with many products
    let mut rng = OsRng;
    let mut acc = [0u64; 9];
    let mut expected_sum = Fp::ZERO;

    // Sum 1000 products
    for _ in 0..1000 {
      let a = Fp::random(&mut rng);
      let b = Fp::random(&mut rng);

      expected_sum += a * b;

      let product = mul_4_by_4(&a.0, &b.0);
      let mut carry = 0u128;
      for i in 0..8 {
        let sum = (acc[i] as u128) + (product[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
      }
      acc[8] = acc[8].wrapping_add(carry as u64);
    }

    let result = Fp(montgomery_reduce_9::<Fp>(&acc));
    assert_eq!(result, expected_sum);
  }

  #[test]
  fn test_montgomery_9_fq_sum_of_products() {
    let mut rng = OsRng;
    let mut acc = [0u64; 9];
    let mut expected_sum = Fq::ZERO;

    for _ in 0..100 {
      let a = Fq::random(&mut rng);
      let b = Fq::random(&mut rng);

      expected_sum += a * b;

      let product = mul_4_by_4(&a.0, &b.0);
      let mut carry = 0u128;
      for i in 0..8 {
        let sum = (acc[i] as u128) + (product[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
      }
      acc[8] = acc[8].wrapping_add(carry as u64);
    }

    let result = Fq(montgomery_reduce_9::<Fq>(&acc));
    assert_eq!(result, expected_sum);
  }
}
