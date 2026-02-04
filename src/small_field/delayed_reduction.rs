// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! DelayedReduction trait for accumulating unreduced products.

use super::SmallValueField;
use std::{
  fmt::Debug,
  ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

/// Extension trait for delayed modular reduction operations.
///
/// This trait extends `SmallValueField` with operations that accumulate
/// unreduced products in wide integers, reducing only at the end.
/// Used in hot paths like matrix-vector multiplication where many products
/// are summed together.
///
/// # Performance
///
/// Delaying reduction saves ~1 field multiplication per accumulation:
/// - Without delayed reduction: N additions + N reductions
/// - With delayed reduction: N additions + 1 reduction
pub trait DelayedReduction<SmallValue>: SmallValueField<SmallValue>
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
  /// Unreduced accumulator for field × integer products.
  /// - For i32/i64: SignedWideLimbs<6> (384 bits)
  /// - For i64/i128: SignedWideLimbs<7> (448 bits)
  ///
  /// The accumulated value is 1R Montgomery-scaled: when we multiply
  /// `(field_val × R) × small_int`, the result is `(field_val × small_int) × R`.
  /// Barrett reduction produces 1R-scaled limbs directly usable with `from_limbs()`.
  ///
  /// Sized to safely sum 2^(l/2) terms without overflow, assuming:
  /// `field_bits + product_bits + (l/2) < 64*N`
  /// (N = limb count for this accumulator, 64 bits per limb).
  type UnreducedFieldInt: Copy
    + Clone
    + Default
    + Debug
    + AddAssign
    + Send
    + Sync
    + num_traits::Zero;

  /// Unreduced accumulator for field × field products (9 limbs, 576 bits).
  /// Used to delay modular reduction when summing many F × F products.
  /// The value is in 2R-scaled Montgomery form, reduced via Montgomery REDC.
  type UnreducedFieldField: Copy
    + Clone
    + Default
    + Debug
    + AddAssign
    + Send
    + Sync
    + num_traits::Zero;

  /// Multiply field element by product of two small values and add to unreduced accumulator.
  /// acc += field × (small_a × small_b) (keeps result in unreduced form, handles sign internally)
  fn accumulate_field_small_small_prod(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    small_a: SmallValue,
    small_b: SmallValue,
  );

  /// Accumulate field element multiplied by an intermediate value into unreduced accumulator.
  /// acc += field × intermediate (keeps result in unreduced form, handles sign internally)
  ///
  /// Used for single-value accumulation (e.g., eq × poly[idx]) where the small value
  /// has already been converted to IntermediateSmallValue via `small_to_intermediate`.
  fn accumulate_field_intermediate_val(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    intermediate: Self::IntermediateSmallValue,
  );

  /// Accumulate field element multiplied by a small value into unreduced accumulator.
  /// acc += field × small (keeps result in unreduced form, handles sign internally)
  ///
  /// Used for matrix-vector multiplication where matrix values are field elements
  /// and vector values are small integers.
  fn accumulate_field_small_prod(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    small: SmallValue,
  );

  /// Multiply two field elements and add to unreduced accumulator.
  /// acc += field_a × field_b (keeps result in 2R-scaled unreduced form)
  fn accumulate_field_field_prod(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  );

  /// Fused three-way: `acc += field × (ext_a × ext_b)`, zero field reductions.
  ///
  /// For extended domain evaluations (after Lagrange interpolation).
  ///
  /// # Bit-width contract
  ///
  /// `ext_a` and `ext_b` are typed as `IntermediateSmallValue` (i64 for i32, i128 for i64)
  /// but their **actual** bit-width is `SmallValue_bits + l_b`, where `l_b` is the number
  /// of Lagrange extension rounds (= log2(num_instances) for NeutronNova).
  ///
  /// Specifically:
  /// - i32 witnesses (SmallValue = i32): ext values use at most 32 + l_b bits
  ///   (e.g., l_b = 4 → 36 bits, fits in i64)
  /// - i64 witnesses (SmallValue = i64): ext values use at most 64 + l_b bits
  ///   (e.g., l_b = 4 → 68 bits, fits in i128)
  ///
  /// The product `field(256b) × ext_a × ext_b` per term is at most:
  /// - i32: 256 + 2*(32 + l_b) = 320 + 2*l_b bits
  /// - i64: 256 + 2*(64 + l_b) = 384 + 2*l_b bits
  ///
  /// Accumulating N = 2^(l/2) terms (where l = log2(num_constraints)) adds l/2 bits:
  /// - i32: 320 + 2*l_b + l/2 bits, must fit in SignedWideLimbs<6> (384b) → l_b ≤ 20, l ≤ 128
  /// - i64: 384 + 2*l_b + l/2 bits, must fit in SignedWideLimbs<7> (448b) → l_b ≤ 20, l ≤ 128
  ///
  /// For practical NeutronNova (l_b ≤ 8, l ≤ 26): well within bounds.
  fn accumulate_field_ext_ext_prod(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    ext_a: Self::IntermediateSmallValue,
    ext_b: Self::IntermediateSmallValue,
  );

  /// Reduce an unreduced field×integer accumulator to a field element.
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self;

  /// Reduce an unreduced field×field accumulator to a field element.
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self;

  // ========================================================================
  // Scatter support
  // ========================================================================

  /// Cached field element type for scatter operations.
  /// Can be `Self` (Montgomery form) or `[u64; 4]` (raw limbs).
  /// Using `Self` avoids conversions; using raw limbs may have other benefits.
  type UnreducedField: Copy + Clone + Default + Debug + PartialEq + Send + Sync;

  /// Reduce a field×int accumulator to the scatter cache type.
  fn reduce_field_int_to_unreduced(acc: &Self::UnreducedFieldInt) -> Self::UnreducedField;

  /// Convert a field element to the scatter cache type.
  fn to_unreduced(&self) -> Self::UnreducedField;

  /// Multiply-accumulate two cached field values into a 9-limb accumulator.
  /// `acc += a × b`
  fn accumulate_raw_field_field_products(
    acc: &mut Self::UnreducedFieldField,
    a: &Self::UnreducedField,
    b: &Self::UnreducedField,
  );

  /// Reduce a 9-limb unreduced field×field accumulator to a field element.
  fn reduce_unreduced_field_field(acc: &Self::UnreducedFieldField) -> Self;

  // ========================================================================
  // Field × small accumulation with cached field values
  // ========================================================================

  /// Accumulate field × (small × small) product into unreduced accumulator.
  /// `acc += e × (small_a × small_b)` where e is the cached field type.
  fn accumulate_field_small_small_products(
    acc: &mut Self::UnreducedFieldInt,
    e: &Self::UnreducedField,
    small_a: SmallValue,
    small_b: SmallValue,
  );

  /// Reduce a field×int accumulator to the scatter cache type.
  fn reduce_raw_field_int_to_unreduced(acc: &Self::UnreducedFieldInt) -> Self::UnreducedField;
}
