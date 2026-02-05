// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! WideMul trait for widening multiplication of small values.
//!
//! This trait enables `small × small → intermediate` multiplication
//! without overflow, used in the Lagrange accumulator algorithm.

/// Trait for widening multiplication: `Self × Self → Product`.
///
/// Used to compute products of small values (i32 or i64) that may
/// overflow the input type, producing a wider intermediate type.
pub trait WideMul: Sized {
  /// The wider product type (e.g., i64 for i32, i128 for i64).
  type Product: Copy + Send + Sync;

  /// Compute `a × b` with widening to avoid overflow.
  fn wide_mul(a: Self, b: Self) -> Self::Product;
}

impl WideMul for i32 {
  type Product = i64;

  #[inline(always)]
  fn wide_mul(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
  }
}

impl WideMul for i64 {
  type Product = i128;

  #[inline(always)]
  fn wide_mul(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_wide_mul_i32() {
    // Normal case
    assert_eq!(i32::wide_mul(10, 20), 200i64);

    // Edge case: large values that would overflow i32
    let a = i32::MAX;
    let b = 2i32;
    let result = i32::wide_mul(a, b);
    assert_eq!(result, (i32::MAX as i64) * 2);

    // Negative values
    assert_eq!(i32::wide_mul(-5, 3), -15i64);
    assert_eq!(i32::wide_mul(-5, -3), 15i64);
  }

  #[test]
  fn test_wide_mul_i64() {
    // Normal case
    assert_eq!(i64::wide_mul(10, 20), 200i128);

    // Edge case: large values that would overflow i64
    let a = i64::MAX;
    let b = 2i64;
    let result = i64::wide_mul(a, b);
    assert_eq!(result, (i64::MAX as i128) * 2);

    // Negative values
    assert_eq!(i64::wide_mul(-5, 3), -15i128);
    assert_eq!(i64::wide_mul(-5, -3), 15i128);
  }
}
