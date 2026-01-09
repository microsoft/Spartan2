// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallValueField trait for small-value optimization.

use ff::PrimeField;
use std::{
  fmt::Debug,
  ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

/// Trait for fields that support small-value optimization.
///
/// This trait defines operations for efficient arithmetic when polynomial
/// evaluations fit in native integers. The key optimization is avoiding
/// expensive field operations until absolutely necessary.
///
/// # Type Parameters
/// - `SmallValue`: Native type for witness values (i32 or i64)
///
/// # Implementations
/// - `Fp: SmallValueField<i32>` with `IntermediateSmallValue = i64`
/// - `Fp: SmallValueField<i64>` with `IntermediateSmallValue = i128`
///
/// # Overflow Bounds (for D=2 Spartan with typical witness values)
///
/// | Step | Bound | Bits | Container |
/// |------|-------|------|-----------|
/// | Original witness values | 2²⁰ | 20 | i32 |
/// | After 3 extensions (D=2) | 2²³ | 23 | i32 ✓ |
/// | Product of two extended | 2⁴⁶ | 46 | i64 ✓ |
pub trait SmallValueField<SmallValue>: PrimeField
where
  SmallValue: Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + Eq
    + Add<Output = SmallValue>
    + Sub<Output = SmallValue>
    + Neg<Output = SmallValue>
    + AddAssign
    + SubAssign
    + Send
    + Sync,
{
  /// Intermediate type for products (i64 for i32 inputs, i128 for i64 inputs).
  /// Used when multiplying two SmallValues together.
  type IntermediateSmallValue: Copy + Clone + Default + Debug + PartialEq + Eq + Send + Sync;

  // ===== Core Multiplications =====

  /// ss: small × small → intermediate (i32 × i32 → i64, or i64 × i64 → i128)
  fn ss_mul(a: SmallValue, b: SmallValue) -> Self::IntermediateSmallValue;

  /// sl: small × large → large (small × field → field)
  fn sl_mul(small: SmallValue, large: &Self) -> Self;

  /// isl: intermediate × large → large (intermediate × field → field)
  /// This is the key operation for accumulator building.
  fn isl_mul(small: Self::IntermediateSmallValue, large: &Self) -> Self;

  // ===== Conversions =====

  /// Convert SmallValue to field element.
  fn small_to_field(val: SmallValue) -> Self;

  /// Convert IntermediateSmallValue to field element.
  fn intermediate_to_field(val: Self::IntermediateSmallValue) -> Self;

  /// Try to convert a field element to SmallValue.
  /// Returns None if the value doesn't fit.
  fn try_field_to_small(val: &Self) -> Option<SmallValue>;
}
