// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! DelayedReduction trait for accumulating unreduced products.

use super::SmallValueField;
use std::{fmt::Debug, ops::AddAssign};

/// Extension trait for delayed modular reduction operations.
///
/// This trait extends `SmallValueField` with operations that accumulate
/// unreduced products in wide integers, reducing only at the end.
/// Used in hot paths like sumcheck accumulator building where many products
/// are summed together.
///
/// # Computations
///
/// Two main sum-of-products patterns are supported:
///
/// ## Field × Small × Small (inner loop accumulation)
///
/// Computes: `Σᵢ (field_i × small_a_i × small_b_i)` for N terms
///
/// **Standard approach** (N Montgomery reductions):
/// - N native multiplies: `small_a × small_b → product`
/// - N Montgomery multiplies: `field × product → result` (each includes REDC)
/// - N Montgomery additions: `acc += result`
/// - **Total: N native muls + N Montgomery muls + N REDCs**
///
/// **Delayed reduction** (1 Barrett reduction):
/// - N native multiplies: `small_a × small_b → product` (i32×i32→i64, i64×i64→i128)
/// - N wide multiply-accumulates: `acc_wide += field_limbs × product` (no reduction)
/// - 1 Barrett reduction at the end
/// - **Total: N native muls + N wide mul-adds + 1 Barrett reduction**
///
/// ## Field × Field (scatter phase accumulation)
///
/// Computes: `Σᵢ (field_a_i × field_b_i)` for N terms
///
/// **Standard approach** (N Montgomery reductions):
/// - N Montgomery multiplies: `field_a × field_b → result` (each REDC reduces 8→4 limbs)
/// - N Montgomery additions: `acc += result`
/// - **Total: N Montgomery muls + N REDCs**
///
/// **Delayed reduction** (1 Montgomery REDC):
/// - N wide multiplies: 4×4 limb multiply → 8 limbs (no REDC)
/// - N wide additions: `acc_wide += product_8limbs` (no reduction)
/// - 1 Montgomery REDC at the end (reduces 9→4 limbs)
/// - **Total: N wide muls + N wide adds + 1 Montgomery REDC**
///
/// # Performance
///
/// Savings come from eliminating N-1 modular reductions. Each REDC/Barrett
/// reduction involves multiple 64-bit multiplies and conditional subtractions.
///
/// # Montgomery Scaling
///
/// - `UnreducedFieldInt`: Stores 1R-scaled values. Multiplying `(field × R) × small_value`
///   yields `(field × small_value) × R`. Barrett reduction preserves R-scaling.
/// - `UnreducedFieldField`: Stores 2R-scaled values. Multiplying two 1R-scaled field elements
///   yields a 2R-scaled product. Montgomery REDC converts 2R → 1R.
pub trait DelayedReduction<SmallValue>: SmallValueField<SmallValue> {
  /// Unreduced accumulator for field × small value products.
  ///
  /// Current implementations: `SignedWideLimbs<6>` (384 bits) for i32,
  /// `SignedWideLimbs<7>` (448 bits) for i64.
  ///
  /// Sized to safely sum many terms without overflow:
  /// `field_bits + product_bits + log2(num_terms) < 64×N`
  /// where field_bits ≈ 254, product_bits = 64 (i32×i32) or 128 (i64×i64),
  /// num_terms = max accumulation count, N = limb count.
  type UnreducedFieldInt: Copy
    + Clone
    + Default
    + Debug
    + AddAssign
    + Send
    + Sync
    + num_traits::Zero;

  /// Unreduced accumulator for field × field products.
  ///
  /// Current implementation: `WideLimbs<9>` (576 bits).
  type UnreducedFieldField: Copy
    + Clone
    + Default
    + Debug
    + AddAssign
    + Send
    + Sync
    + num_traits::Zero;

  /// acc += field × (small_a × small_b)
  fn accumulate_field_small_small_prod(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    small_a: SmallValue,
    small_b: SmallValue,
  );

  /// acc += field × small_value
  fn accumulate_field_small_prod(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    small: SmallValue,
  );

  /// acc += field_a × field_b
  fn accumulate_field_field_prod(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  );

  /// Reduce field × small value accumulator to a field element.
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self;

  /// Reduce an unreduced field×field accumulator to a field element.
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self;
}
