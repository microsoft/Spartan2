// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Stack-allocated wide integers for delayed modular reduction.
//!
//! This module provides [`WideLimbs<N>`], a fixed-size wide integer type used
//! for accumulating unreduced field element products in hot paths.
//!
//! # Why not `num-bigint`?
//!
//! We define our own type rather than using `num-bigint::BigInt` because:
//! - `BigInt` is heap-allocated (uses `Vec<u64>`)
//! - We need stack-allocated fixed-size arrays for hot-path performance
//! - We want the `Copy` trait for cheap pass-by-value in tight loops
//!
//! # Usage
//!
//! ```ignore
//! use spartan2::wide_limbs::WideLimbs;
//!
//! let mut acc: WideLimbs<6> = WideLimbs::zero();
//! // ... accumulate products ...
//! // Then reduce once at the end
//! ```

use num_traits::Zero;
use std::ops::{Add, AddAssign};

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

#[allow(dead_code)] // Will be used in accumulator optimization
impl<const N: usize> WideLimbs<N> {
  /// Create a zero value.
  #[inline]
  pub const fn zero() -> Self {
    Self([0u64; N])
  }

  /// Check if all limbs are zero.
  #[inline]
  pub fn is_zero(&self) -> bool {
    self.0.iter().all(|&x| x == 0)
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
    // This is guaranteed by the bit-sizing analysis in the plan.
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

// ============================================================================
// SignedWideLimbs - for accumulating signed products
// ============================================================================

/// Pair of wide integers for accumulating signed products.
///
/// Since `WideLimbs` only supports unsigned addition, we track positive and
/// negative contributions separately, then subtract at the end.
///
/// Used for delayed reduction when accumulating `field × i64` products where
/// the i64 can be negative.
#[allow(dead_code)] // May be used in future optimizations
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

#[allow(dead_code)]
impl<const N: usize> SignedWideLimbs<N> {
  /// Create a zero value.
  #[inline]
  pub const fn zero() -> Self {
    Self {
      pos: WideLimbs::zero(),
      neg: WideLimbs::zero(),
    }
  }

  /// Check if both positive and negative accumulators are zero.
  #[inline]
  pub fn is_zero(&self) -> bool {
    self.pos.is_zero() && self.neg.is_zero()
  }
}

#[allow(dead_code)]
impl<const N: usize> AddAssign for SignedWideLimbs<N> {
  /// Merge two signed accumulators by adding their respective parts.
  #[inline]
  fn add_assign(&mut self, other: Self) {
    self.pos += other.pos;
    self.neg += other.neg;
  }
}

#[allow(dead_code)]
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

/// Compute |a - b| and return (is_negative, magnitude).
///
/// Used to reduce two wide integers to one before Barrett reduction,
/// saving one expensive reduction operation.
#[inline(always)]
pub fn sub_mag<const N: usize>(a: &[u64; N], b: &[u64; N]) -> (bool, [u64; N]) {
  let mut out = [0u64; N];
  let mut borrow = 0u64;
  for i in 0..N {
    let (d1, b1) = a[i].overflowing_sub(b[i]);
    let (d2, b2) = d1.overflowing_sub(borrow);
    out[i] = d2;
    borrow = (b1 as u64) + (b2 as u64);
  }
  if borrow == 0 {
    (false, out)
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
    (true, out2)
  }
}

// ============================================================================
// SignedAccumulator - generic wrapper for any unreduced type
// ============================================================================

/// Generic pair of accumulators for signed products.
///
/// This is the type-erased version of `SignedWideLimbs` that works with
/// any unreduced accumulator type (via `SmallValueField::UnreducedFieldInt`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedAccumulator<T> {
  /// Accumulator for positive contributions
  pub pos: T,
  /// Accumulator for negative contributions (stored as positive magnitude)
  pub neg: T,
}

impl<T: Default> Default for SignedAccumulator<T> {
  fn default() -> Self {
    Self {
      pos: T::default(),
      neg: T::default(),
    }
  }
}

#[allow(dead_code)]
impl<T: Default> SignedAccumulator<T> {
  /// Create a zero value.
  #[inline]
  pub fn zero() -> Self {
    Self::default()
  }
}

impl<T: AddAssign> AddAssign for SignedAccumulator<T> {
  /// Merge two signed accumulators by adding their respective parts.
  #[inline]
  fn add_assign(&mut self, other: Self) {
    self.pos += other.pos;
    self.neg += other.neg;
  }
}

impl<T: AddAssign + Copy> AddAssign<&Self> for SignedAccumulator<T> {
  #[inline]
  fn add_assign(&mut self, other: &Self) {
    self.pos += other.pos;
    self.neg += other.neg;
  }
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
}
