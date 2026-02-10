// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Montgomery form operations: limb access and REDC reduction.

use super::{
  field_reduction_constants::FieldReductionConstants,
  limbs::{add_4_4, gte_4_4, sub_4_4, sub_5_4},
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

/// Montgomery REDC for 9-limb input (optimized single-fold algorithm).
///
/// Reduces a 2R-scaled value (sum of field×field products) to 1R-scaled.
/// Input: T representing x*R² (up to 9 limbs)
/// Output: x*R mod p (4 limbs, canonical Montgomery form in [0, p))
///
/// # Algorithm
///
/// 1. **Fold**: Compute `low8 += h * R512_MOD` where h = c[8].
///    Track the carry c ∈ {0,1} from this fold.
///
/// 2. **REDC**: Call `montgomery_reduce_8` which returns canonical [0, p).
///
/// 3. **Carry correction**: If c=1, add R_MOD and do one conditional subtract.
#[inline]
pub(crate) fn montgomery_reduce_9<F: FieldReductionConstants>(c: &[u64; 9]) -> [u64; 4] {
  // STEP 1: Fold - reduce 9 limbs to 8 limbs + carry bit
  //
  // We have: C = L + h*R² where L = c[0..8], h = c[8]
  // Compute: low8 = L + h*R512_MOD (where R512_MOD = R² mod p)
  // The fold_carry c ∈ {0,1} is provably bounded (see proof in comments below).

  let mut low8 = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
  let h = c[8];
  let mut fold_carry = 0u64;

  if h != 0 {
    // Fused: low8[0..4] += h * R512_MOD[0..4] with carry chain
    let mut carry = 0u128;
    for (limb, &r512_limb) in low8.iter_mut().zip(F::R512_MOD.iter()) {
      let prod = (h as u128) * (r512_limb as u128) + (*limb as u128) + carry;
      *limb = prod as u64;
      carry = prod >> 64;
    }

    // The 5th limb of h*R512_MOD is `carry`; add it into low8[4..7]
    for limb in &mut low8[4..] {
      let sum = (*limb as u128) + carry;
      *limb = sum as u64;
      carry = sum >> 64;
    }

    fold_carry = carry as u64;
  }

  // Proof of bound: h < 2^64, R512_MOD < p < R = 2^256
  // So h * R512_MOD < 2^64 * 2^256 = 2^320
  // And low8 + h * R512_MOD < 2^512 + 2^320 < 2 * 2^512 = 2R²
  // Therefore fold_carry = ⌊result / R²⌋ ∈ {0, 1}
  debug_assert!(
    fold_carry <= 1,
    "fold carry must be 0 or 1, got {}",
    fold_carry
  );

  // STEP 2: Montgomery REDC on 8 limbs → canonical result in [0, p)
  let mut out = montgomery_reduce_8::<F>(&low8);

  // STEP 3: Carry correction
  // If fold_carry == 1, we have an extra R² term that REDC turns into R.
  // In Montgomery form, R mod p = R_MOD. So: out += R_MOD, then canonicalize.
  if fold_carry == 1 {
    let (sum, carry) = add_4_4(&out, &F::R_MOD);
    out = sum;

    // out is now in [0, 2p) (since out was in [0,p) and R_MOD < p).
    // At most one subtract needed.
    if carry == 1 || gte_4_4(&out, &F::MODULUS) {
      out = sub_4_4(&out, &F::MODULUS);
    }
  }

  out
}

/// Montgomery REDC for 8-limb input.
///
/// Input: T[8] limbs representing an integer in [0, R²)
/// Output: canonical 4-limb result in [0, p)
///
/// # Algorithm
///
/// 1. 4 Montgomery elimination rounds with fused multiply+add
/// 2. Extract 5-limb result x5 = [r[4]..r[8]] in [0, R+p)
/// 3. If x5[4] == 1: subtract p once → value now in [0, R)
/// 4. Canonicalize from [0, R) → [0, p) via Q = ⌊R/p⌋ conditional subtracts
///
/// **Key insight**: Standard REDC produces a value in [0, R), NOT [0, p).
/// Since R > p for 256-bit primes, we need Q subtractions to canonicalize.
#[inline]
fn montgomery_reduce_8<F: FieldReductionConstants>(t: &[u64; 8]) -> [u64; 4] {
  let mut r = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], 0u64];

  // Montgomery reduction: eliminate low 4 limbs (exactly 4 iterations)
  for i in 0..4 {
    // Compute multiplier q that zeros out r[i]: q ≡ r[i] * (-p⁻¹) mod 2^64
    let q = r[i].wrapping_mul(F::MONT_INV);

    // Fused: r[i..i+5] += q * MODULUS[0..4] with carry chain
    let mut carry = 0u128;
    for j in 0..4 {
      let prod = (q as u128) * (F::MODULUS[j] as u128) + (r[i + j] as u128) + carry;
      r[i + j] = prod as u64;
      carry = prod >> 64;
    }

    // Add carry into r[i+4], then propagate through remaining limbs
    let sum = (r[i + 4] as u128) + carry;
    r[i + 4] = sum as u64;
    carry = sum >> 64;

    // Propagate remaining carry (bounded, rarely needed)
    let mut k = i + 5;
    while carry != 0 && k < 9 {
      let sum = (r[k] as u128) + carry;
      r[k] = sum as u64;
      carry = sum >> 64;
      k += 1;
    }
  }

  // Result is in r[4..9] (5 limbs), in range [0, R+p)
  let mut x5 = [r[4], r[5], r[6], r[7], r[8]];

  // If 5th limb is 1, subtract p once to bring below R
  if x5[4] == 1 {
    x5 = sub_5_4(&x5, &F::MODULUS);
    debug_assert!(x5[4] == 0, "after 5th-limb subtract, x5[4] should be 0");
  }

  // Now x5[0..4] < R, canonicalize to [0, p) with Q conditional subtracts
  let mut out = [x5[0], x5[1], x5[2], x5[3]];
  for _ in 0..F::MAX_CANONICALIZE_SUBS {
    if gte_4_4(&out, &F::MODULUS) {
      out = sub_4_4(&out, &F::MODULUS);
    }
  }

  debug_assert!(
    !gte_4_4(&out, &F::MODULUS),
    "canonicalization failed after {} subtractions",
    F::MAX_CANONICALIZE_SUBS
  );

  out
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
