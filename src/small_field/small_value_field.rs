// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallValueField trait for small-value optimization.

use ff::PrimeField;

/// Trait for fields that support small-value optimization.
///
/// This trait defines operations for efficient arithmetic when polynomial
/// evaluations fit in native integers. The key optimization is avoiding
/// expensive field operations until absolutely necessary.
///
/// See [`DelayedReduction`](super::DelayedReduction) for the accumulator optimization
/// that builds on this trait, enabling sum-of-products with a single reduction.
///
/// # Type Parameters
/// - `SmallValue`: Native type for witness values (i32 or i64)
///
/// # Implementations
/// - `Fp: SmallValueField<i32>`
/// - `Fp: SmallValueField<i64>`
pub trait SmallValueField<SmallValue>: PrimeField {
  /// Convert SmallValue to field element.
  fn small_to_field(val: SmallValue) -> Self;

  /// Try to convert a field element to SmallValue.
  /// Returns None if the value doesn't fit.
  fn try_field_to_small(val: &Self) -> Option<SmallValue>;
}
