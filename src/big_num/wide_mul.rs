// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! WideMul trait for typed widening/cross multiplication.
//!
//! This trait enables products whose output type is intentionally different
//! from Rust's built-in `Mul` output, such as `i64 × i64 → i128` or
//! `i32 × bool → i64`.

#![allow(dead_code)]

/// Trait for typed multiplication where the output is chosen by the `(lhs, rhs)`
/// pair.
///
/// Used to compute products that should widen or change representation instead
/// of using Rust's built-in `Mul::Output`.
pub trait WideMul<Rhs = Self>: Copy {
  /// Product type for this `(Self, Rhs)` pair.
  type Output: Copy + Send + Sync;

  /// Compute `self × rhs` using the product type selected by this impl.
  fn wide_mul(self, rhs: Rhs) -> Self::Output;
}

impl WideMul for i32 {
  type Output = i64;

  #[inline(always)]
  fn wide_mul(self, rhs: i32) -> i64 {
    i64::from(self) * i64::from(rhs)
  }
}

impl WideMul<bool> for i32 {
  type Output = i64;

  #[inline(always)]
  fn wide_mul(self, rhs: bool) -> i64 {
    let mask = -i32::from(u8::from(rhs));
    i64::from(self & mask)
  }
}

impl WideMul<i8> for i32 {
  type Output = i64;

  #[inline(always)]
  fn wide_mul(self, rhs: i8) -> i64 {
    i64::from(self) * i64::from(rhs)
  }
}

impl WideMul<bool> for i8 {
  type Output = i32;

  #[inline(always)]
  fn wide_mul(self, rhs: bool) -> i32 {
    let value = i32::from(self);
    let mask = -i32::from(u8::from(rhs));
    value & mask
  }
}

impl WideMul for i8 {
  type Output = i32;

  #[inline(always)]
  fn wide_mul(self, rhs: i8) -> i32 {
    i32::from(self) * i32::from(rhs)
  }
}

impl WideMul for i64 {
  type Output = i128;

  #[inline(always)]
  fn wide_mul(self, rhs: i64) -> i128 {
    i128::from(self) * i128::from(rhs)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_wide_mul_i32() {
    // Normal case
    assert_eq!(10i32.wide_mul(20), 200i64);

    // Edge case: large values that would overflow i32
    let a = i32::MAX;
    let b = 2i32;
    let result = a.wide_mul(b);
    assert_eq!(result, i64::from(i32::MAX) * 2);

    // Negative values
    assert_eq!((-5i32).wide_mul(3), -15i64);
    assert_eq!((-5i32).wide_mul(-3), 15i64);
  }

  #[test]
  fn test_wide_mul_i32_bool_branchless_mask_semantics() {
    assert_eq!(42i32.wide_mul(true), 42i64);
    assert_eq!(42i32.wide_mul(false), 0i64);
    assert_eq!((-42i32).wide_mul(true), -42i64);
    assert_eq!((-42i32).wide_mul(false), 0i64);
  }

  #[test]
  fn test_wide_mul_i32_i8() {
    assert_eq!(12i32.wide_mul(-3i8), -36i64);
    assert_eq!((-12i32).wide_mul(-3i8), 36i64);
  }

  #[test]
  fn test_wide_mul_i8_bool_and_i8() {
    assert_eq!(7i8.wide_mul(true), 7i32);
    assert_eq!(7i8.wide_mul(false), 0i32);
    assert_eq!((-7i8).wide_mul(true), -7i32);
    assert_eq!(7i8.wide_mul(-3i8), -21i32);
  }

  #[test]
  fn test_wide_mul_i64() {
    // Normal case
    assert_eq!(10i64.wide_mul(20), 200i128);

    // Edge case: large values that would overflow i64
    let a = i64::MAX;
    let b = 2i64;
    let result = a.wide_mul(b);
    assert_eq!(result, i128::from(i64::MAX) * 2);

    // Negative values
    assert_eq!((-5i64).wide_mul(3), -15i128);
    assert_eq!((-5i64).wide_mul(-3), 15i128);
  }
}
