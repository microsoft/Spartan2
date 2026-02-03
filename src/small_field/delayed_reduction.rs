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

  /// Multiply field element by signed integer and add to unreduced accumulator.
  /// acc += field × intermediate (keeps result in unreduced form, handles sign internally)
  fn unreduced_field_int_mul_add(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    small: Self::IntermediateSmallValue,
  );

  /// Multiply two field elements and add to unreduced accumulator.
  /// acc += field_a × field_b (keeps result in 2R-scaled unreduced form)
  fn unreduced_field_field_mul_add(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  );

  /// Batch 4 independent field×int multiply-accumulates for ILP optimization.
  /// Default implementation calls single version 4 times.
  #[inline(always)]
  fn unreduced_field_int_mul_add_batch4(
    accs: [&mut Self::UnreducedFieldInt; 4],
    field: &Self,
    smalls: [Self::IntermediateSmallValue; 4],
  ) {
    let [acc0, acc1, acc2, acc3] = accs;
    Self::unreduced_field_int_mul_add(acc0, field, smalls[0]);
    Self::unreduced_field_int_mul_add(acc1, field, smalls[1]);
    Self::unreduced_field_int_mul_add(acc2, field, smalls[2]);
    Self::unreduced_field_int_mul_add(acc3, field, smalls[3]);
  }

  /// Batch 4 independent field×field multiply-accumulates for ILP optimization.
  /// Default implementation calls single version 4 times.
  #[inline(always)]
  fn unreduced_field_field_mul_add_batch4(
    accs: [&mut Self::UnreducedFieldField; 4],
    a: [&Self; 4],
    b: [&Self; 4],
  ) {
    let [acc0, acc1, acc2, acc3] = accs;
    Self::unreduced_field_field_mul_add(acc0, a[0], b[0]);
    Self::unreduced_field_field_mul_add(acc1, a[1], b[1]);
    Self::unreduced_field_field_mul_add(acc2, a[2], b[2]);
    Self::unreduced_field_field_mul_add(acc3, a[3], b[3]);
  }

  /// Fused three-way: `acc += field × (ext_a × ext_b)`, zero field reductions.
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
  fn unreduced_field_int_product_mul_add(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    ext_a: Self::IntermediateSmallValue,
    ext_b: Self::IntermediateSmallValue,
  );

  /// Batch 4 independent fused three-way multiply-accumulates for ILP.
  /// Same bit-width contract as [`Self::unreduced_field_int_product_mul_add`].
  #[inline(always)]
  fn unreduced_field_int_product_mul_add_batch4(
    accs: [&mut Self::UnreducedFieldInt; 4],
    field: &Self,
    ext_a: [Self::IntermediateSmallValue; 4],
    ext_b: [Self::IntermediateSmallValue; 4],
  ) {
    let [acc0, acc1, acc2, acc3] = accs;
    Self::unreduced_field_int_product_mul_add(acc0, field, ext_a[0], ext_b[0]);
    Self::unreduced_field_int_product_mul_add(acc1, field, ext_a[1], ext_b[1]);
    Self::unreduced_field_int_product_mul_add(acc2, field, ext_a[2], ext_b[2]);
    Self::unreduced_field_int_product_mul_add(acc3, field, ext_a[3], ext_b[3]);
  }

  /// Reduce an unreduced field×integer accumulator to a field element.
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self;

  /// Reduce an unreduced field×field accumulator to a field element.
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self;

  // ========================================================================
  // Non-Montgomery scatter support
  // ========================================================================

  /// Non-R-scaled raw limbs (not in Montgomery form). `[u64; 4]` for 256-bit fields.
  /// Used to avoid Montgomery multiplications in the scatter hot path.
  type UnreducedField: Copy + Clone + Default + Debug + Send + Sync;

  /// Barrett-reduce a field×int accumulator to non-R-scaled raw limbs.
  /// Equivalent to `from_mont(reduce_field_int(acc))` but avoids constructing
  /// an intermediate field element.
  fn reduce_field_int_to_unreduced(acc: &Self::UnreducedFieldInt) -> Self::UnreducedField;

  /// Convert a Montgomery-form field element to non-R-scaled raw limbs.
  fn to_unreduced(&self) -> Self::UnreducedField;

  /// Raw (non-Montgomery) 4×4 limb multiply-accumulate into a 9-limb accumulator.
  /// `acc += a_raw × b_raw` where both inputs are non-R-scaled.
  fn unreduced_raw_mul_add(
    acc: &mut Self::UnreducedFieldField,
    a: &Self::UnreducedField,
    b: &Self::UnreducedField,
  );

  /// Barrett-reduce a 9-limb raw product accumulator to a field element.
  /// The accumulated value is non-R-scaled; this function Barrett-reduces
  /// and then converts to Montgomery form.
  fn barrett_reduce_field_field(acc: &Self::UnreducedFieldField) -> Self;
}
