// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! DelayedReduction trait for accumulating unreduced field products.
//!
//! Modular reduction (Montgomery REDC) is expensive. When summing many
//! products `Σ (field_i × field_j)`, the standard approach does N reductions.
//! Delayed reduction accumulates unreduced products in wide integers, reducing
//! only once at the end.

use super::{
  limbs::{WideLimbs, mul_4_by_4},
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

/// DelayedReduction<F> for field × field products.
///
/// Uses WideLimbs<9> (576 bits) as accumulator, supporting up to 2^68 products.
///
/// # Capacity Invariant
///
/// The 9th limb (index 8) accumulates carries from the lower 8 limbs. Each
/// field×field product contributes at most 1 to the carry chain into limb 8.
/// With a u64 limb, we can accumulate up to 2^64 products before overflow.
/// In practice, sumcheck rounds are bounded by polynomial size (≤ 2^40),
/// so this limit is never approached. The debug_assert below catches misuse.
impl<F: MontgomeryLimbs + PrimeField + Copy> DelayedReduction<F> for F {
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

/// Generate tests for `DelayedReduction` implementation.
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
