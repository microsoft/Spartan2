// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Wide integers and limb operations for delayed modular reduction.
//!
//! This module provides:
//! - [`WideLimbs<N>`]: Stack-allocated wide integers for accumulating unreduced products
//! - [`SignedWideLimbs<N>`]: Signed variant for accumulating signed products
//! - Limb arithmetic operations (multiply, subtract, compare)
//!
//! # Why not `num-bigint`?
//!
//! We define our own types rather than using `num-bigint::BigInt` because:
//! - `BigInt` is heap-allocated (uses `Vec<u64>`)
//! - We need stack-allocated fixed-size arrays for hot-path performance
//! - We want the `Copy` trait for cheap pass-by-value in tight loops

use num_traits::Zero;
use std::ops::{Add, AddAssign};

// ============================================================================
// WideLimbs - Stack-allocated wide integer
// ============================================================================

/// Stack-allocated wide integer with N 64-bit limbs.
///
/// Limbs are stored in little-endian order: `limbs[0]` is the least significant.
///
/// # Type Parameters
///
/// - `N`: Number of 64-bit limbs. Common values:
///   - `N=6` (384 bits): For `UnreducedFieldInt` (field × integer products)
///   - `N=9` (576 bits): For `UnreducedFieldField` (field × field products)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideLimbs<const N: usize>(pub [u64; N]);

impl<const N: usize> Default for WideLimbs<N> {
  fn default() -> Self {
    Self([0u64; N])
  }
}

impl<const N: usize> AddAssign for WideLimbs<N> {
  /// Wide addition with carry propagation.
  #[inline]
  fn add_assign(&mut self, other: Self) {
    let mut carry = 0u64;
    for i in 0..N {
      let (sum, c1) = self.0[i].overflowing_add(other.0[i]);
      let (sum, c2) = sum.overflowing_add(carry);
      self.0[i] = sum;
      carry = (c1 as u64) + (c2 as u64);
    }
    // Note: We intentionally don't check for overflow here.
    // The caller is responsible for ensuring the sum doesn't exceed N limbs.
  }
}

impl<const N: usize> AddAssign<&Self> for WideLimbs<N> {
  /// Wide addition with carry propagation (reference variant).
  #[inline]
  fn add_assign(&mut self, other: &Self) {
    let mut carry = 0u64;
    for i in 0..N {
      let (sum, c1) = self.0[i].overflowing_add(other.0[i]);
      let (sum, c2) = sum.overflowing_add(carry);
      self.0[i] = sum;
      carry = (c1 as u64) + (c2 as u64);
    }
  }
}

impl<const N: usize> Add for WideLimbs<N> {
  type Output = Self;

  #[inline]
  fn add(mut self, other: Self) -> Self {
    self += other;
    self
  }
}

impl<const N: usize> Add<&Self> for WideLimbs<N> {
  type Output = Self;

  #[inline]
  fn add(mut self, other: &Self) -> Self {
    self += other;
    self
  }
}

impl<const N: usize> Zero for WideLimbs<N> {
  #[inline]
  fn zero() -> Self {
    Self([0u64; N])
  }

  #[inline]
  fn is_zero(&self) -> bool {
    self.0.iter().all(|&x| x == 0)
  }
}

// ============================================================================
// SignedWideLimbs - for accumulating signed products
// ============================================================================

/// Pair of wide integers for accumulating signed products.
///
/// Since `WideLimbs` only supports unsigned addition, we track positive and
/// negative contributions separately, then subtract at the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedWideLimbs<const N: usize> {
  /// Accumulator for positive contributions
  pub pos: WideLimbs<N>,
  /// Accumulator for negative contributions (stored as positive magnitude)
  pub neg: WideLimbs<N>,
}

impl<const N: usize> Default for SignedWideLimbs<N> {
  fn default() -> Self {
    Self {
      pos: WideLimbs::zero(),
      neg: WideLimbs::zero(),
    }
  }
}

impl<const N: usize> AddAssign for SignedWideLimbs<N> {
  /// Merge two signed accumulators by adding their respective parts.
  #[inline]
  fn add_assign(&mut self, other: Self) {
    self.pos += other.pos;
    self.neg += other.neg;
  }
}

impl<const N: usize> AddAssign<&Self> for SignedWideLimbs<N> {
  #[inline]
  fn add_assign(&mut self, other: &Self) {
    self.pos += &other.pos;
    self.neg += &other.neg;
  }
}

impl<const N: usize> Add for SignedWideLimbs<N> {
  type Output = Self;

  #[inline]
  fn add(mut self, other: Self) -> Self {
    self.pos += other.pos;
    self.neg += other.neg;
    self
  }
}

impl<const N: usize> Zero for SignedWideLimbs<N> {
  #[inline]
  fn zero() -> Self {
    Self {
      pos: WideLimbs::zero(),
      neg: WideLimbs::zero(),
    }
  }

  #[inline]
  fn is_zero(&self) -> bool {
    self.pos.is_zero() && self.neg.is_zero()
  }
}

// ============================================================================
// SubMagResult - magnitude subtraction result
// ============================================================================

/// Result of magnitude subtraction |a - b|.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubMagResult<const N: usize> {
  /// a >= b, contains a - b
  Positive([u64; N]),
  /// a < b, contains b - a
  Negative([u64; N]),
}

/// Compute |a - b| and return the magnitude with sign information.
///
/// Used to reduce two wide integers to one before Barrett reduction,
/// saving one expensive reduction operation.
#[inline(always)]
pub fn sub_mag<const N: usize>(a: &[u64; N], b: &[u64; N]) -> SubMagResult<N> {
  let mut out = [0u64; N];
  let mut borrow = 0u64;
  for i in 0..N {
    let (d1, b1) = a[i].overflowing_sub(b[i]);
    let (d2, b2) = d1.overflowing_sub(borrow);
    out[i] = d2;
    borrow = (b1 as u64) + (b2 as u64);
  }
  if borrow == 0 {
    SubMagResult::Positive(out)
  } else {
    // a < b, compute b - a instead
    let mut out2 = [0u64; N];
    borrow = 0;
    for i in 0..N {
      let (d1, b1) = b[i].overflowing_sub(a[i]);
      let (d2, b2) = d1.overflowing_sub(borrow);
      out2[i] = d2;
      borrow = (b1 as u64) + (b2 as u64);
    }
    SubMagResult::Negative(out2)
  }
}

// ============================================================================
// Limb multiplication operations
// ============================================================================

/// Multiply-accumulate: acc + a * b + carry → (low, high)
///
/// Fused operation that computes one limb of a multiply-accumulate in a single step,
/// avoiding materialization of intermediate arrays.
#[inline(always)]
pub fn mac(acc: u64, a: u64, b: u64, carry: u64) -> (u64, u64) {
  let prod = (a as u128) * (b as u128) + (acc as u128) + (carry as u128);
  (prod as u64, (prod >> 64) as u64)
}

/// Multiply two 4-limb values, producing an 8-limb result.
#[inline(always)]
pub fn mul_4_by_4_ext(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
  let mut result = [0u64; 8];
  for i in 0..4 {
    let mut carry = 0u128;
    for j in 0..4 {
      let prod = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + carry;
      result[i + j] = prod as u64;
      carry = prod >> 64;
    }
    result[i + 4] = carry as u64;
  }
  result
}

/// Multiply 4-limb field by 2-limb integer (u128), producing a 6-limb result.
/// Used for i64/i128 small-value optimization where IntermediateSmallValue is i128.
#[inline(always)]
pub fn mul_4_by_2_ext(a: &[u64; 4], b: u128) -> [u64; 6] {
  let b_lo = b as u64;
  let b_hi = (b >> 64) as u64;

  // Multiply a by b_lo (4x1 -> 5 limbs)
  let mut result = [0u64; 6];
  let mut carry = 0u128;
  for i in 0..4 {
    let prod = (a[i] as u128) * (b_lo as u128) + carry;
    result[i] = prod as u64;
    carry = prod >> 64;
  }
  result[4] = carry as u64;

  // Multiply a by b_hi and add at offset 1 (4x1 -> 5 limbs, shifted)
  carry = 0u128;
  for i in 0..4 {
    let prod = (a[i] as u128) * (b_hi as u128) + (result[i + 1] as u128) + carry;
    result[i + 1] = prod as u64;
    carry = prod >> 64;
  }
  result[5] = carry as u64;

  result
}

/// Multiply 4-limb by 1-limb, producing a 5-limb result.
#[inline(always)]
pub(super) fn mul_4_by_1(a: &[u64; 4], b: u64) -> [u64; 5] {
  let mut result = [0u64; 5];
  let mut carry = 0u128;
  for i in 0..4 {
    let prod = (a[i] as u128) * (b as u128) + carry;
    result[i] = prod as u64;
    carry = prod >> 64;
  }
  result[4] = carry as u64;
  result
}

/// Multiply 5-limb by 1-limb, producing a 5-limb result (overflow ignored).
#[inline(always)]
pub(super) fn mul_5_by_1(a: &[u64; 5], b: u64) -> [u64; 5] {
  let mut result = [0u64; 5];
  let mut carry = 0u128;
  for i in 0..5 {
    let prod = (a[i] as u128) * (b as u128) + carry;
    result[i] = prod as u64;
    carry = prod >> 64;
  }
  result
}

// ============================================================================
// Limb subtraction operations
// ============================================================================

/// Subtract two 5-limb values: a - b.
#[inline(always)]
pub(super) fn sub_5_5(a: &[u64; 5], b: &[u64; 5]) -> [u64; 5] {
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

/// Subtract 4-limb from 5-limb: a - b.
#[inline(always)]
pub(super) fn sub_5_4(a: &[u64; 5], b: &[u64; 4]) -> [u64; 5] {
  let mut result = [0u64; 5];
  let mut borrow = 0u64;
  for i in 0..4 {
    let (diff, b1) = a[i].overflowing_sub(b[i]);
    let (diff2, b2) = diff.overflowing_sub(borrow);
    result[i] = diff2;
    borrow = (b1 as u64) + (b2 as u64);
  }
  let (diff, _) = a[4].overflowing_sub(borrow);
  result[4] = diff;
  result
}

/// Subtract two 4-limb values: a - b.
#[inline(always)]
pub(super) fn sub_4_4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
  let mut result = [0u64; 4];
  let mut borrow = 0u64;
  for i in 0..4 {
    let (diff, b1) = a[i].overflowing_sub(b[i]);
    let (diff2, b2) = diff.overflowing_sub(borrow);
    result[i] = diff2;
    borrow = (b1 as u64) + (b2 as u64);
  }
  result
}

// ============================================================================
// Limb comparison operations
// ============================================================================

/// Check if 5-limb value >= 4-limb value.
#[inline(always)]
pub(super) fn gte_5_4(a: &[u64; 5], b: &[u64; 4]) -> bool {
  if a[4] > 0 {
    return true;
  }
  for i in (0..4).rev() {
    if a[i] > b[i] {
      return true;
    }
    if a[i] < b[i] {
      return false;
    }
  }
  true
}

/// Check if 4-limb value a >= 4-limb value b.
#[inline(always)]
pub(super) fn gte_4_4(a: &[u64; 4], b: &[u64; 4]) -> bool {
  for i in (0..4).rev() {
    if a[i] > b[i] {
      return true;
    }
    if a[i] < b[i] {
      return false;
    }
  }
  true // equal
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_zero() {
    let z: WideLimbs<6> = WideLimbs::zero();
    assert!(z.is_zero());
    assert_eq!(z.0, [0u64; 6]);
  }

  #[test]
  fn test_default_is_zero() {
    let z: WideLimbs<6> = WideLimbs::default();
    assert!(z.is_zero());
  }

  #[test]
  fn test_add_no_carry() {
    let a: WideLimbs<4> = WideLimbs([1, 2, 3, 4]);
    let b: WideLimbs<4> = WideLimbs([10, 20, 30, 40]);
    let c = a + b;
    assert_eq!(c.0, [11, 22, 33, 44]);
  }

  #[test]
  fn test_add_with_carry() {
    // Test carry propagation
    let a: WideLimbs<4> = WideLimbs([u64::MAX, 0, 0, 0]);
    let b: WideLimbs<4> = WideLimbs([1, 0, 0, 0]);
    let c = a + b;
    assert_eq!(c.0, [0, 1, 0, 0]);
  }

  #[test]
  fn test_add_with_multi_carry() {
    // Test carry propagation across multiple limbs
    let a: WideLimbs<4> = WideLimbs([u64::MAX, u64::MAX, u64::MAX, 0]);
    let b: WideLimbs<4> = WideLimbs([1, 0, 0, 0]);
    let c = a + b;
    assert_eq!(c.0, [0, 0, 0, 1]);
  }

  #[test]
  fn test_add_assign() {
    let mut a: WideLimbs<4> = WideLimbs([1, 2, 3, 4]);
    let b: WideLimbs<4> = WideLimbs([10, 20, 30, 40]);
    a += b;
    assert_eq!(a.0, [11, 22, 33, 44]);
  }

  #[test]
  fn test_add_assign_ref() {
    let mut a: WideLimbs<4> = WideLimbs([1, 2, 3, 4]);
    let b: WideLimbs<4> = WideLimbs([10, 20, 30, 40]);
    a += &b;
    assert_eq!(a.0, [11, 22, 33, 44]);
  }

  #[test]
  fn test_is_zero() {
    let z: WideLimbs<4> = WideLimbs::zero();
    assert!(z.is_zero());

    let nz: WideLimbs<4> = WideLimbs([0, 0, 1, 0]);
    assert!(!nz.is_zero());
  }

  #[test]
  fn test_different_sizes() {
    // Test that WideLimbs works with different N values
    let _a: WideLimbs<6> = WideLimbs::zero();
    let _b: WideLimbs<9> = WideLimbs::zero();

    let x: WideLimbs<6> = WideLimbs([1, 2, 3, 4, 5, 6]);
    let y: WideLimbs<6> = WideLimbs([6, 5, 4, 3, 2, 1]);
    let z = x + y;
    assert_eq!(z.0, [7, 7, 7, 7, 7, 7]);
  }

  #[test]
  fn test_mul_4_by_1() {
    let a = [1u64, 2, 3, 4];
    let b = 10u64;
    let result = mul_4_by_1(&a, b);
    assert_eq!(result, [10, 20, 30, 40, 0]);
  }

  #[test]
  fn test_mul_4_by_1_with_carry() {
    let a = [u64::MAX, 0, 0, 0];
    let b = 2u64;
    let result = mul_4_by_1(&a, b);
    assert_eq!(result, [u64::MAX - 1, 1, 0, 0, 0]);
  }

  #[test]
  fn test_sub_4_4() {
    let a = [10u64, 20, 30, 40];
    let b = [1u64, 2, 3, 4];
    let result = sub_4_4(&a, &b);
    assert_eq!(result, [9, 18, 27, 36]);
  }

  #[test]
  fn test_gte_4_4() {
    let a = [10u64, 20, 30, 40];
    let b = [1u64, 2, 3, 4];
    assert!(gte_4_4(&a, &b));
    assert!(!gte_4_4(&b, &a));

    // Equal case
    assert!(gte_4_4(&a, &a));
  }
}
