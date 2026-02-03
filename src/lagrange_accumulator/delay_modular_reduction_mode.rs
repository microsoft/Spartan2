// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Delayed Modular Reduction (DMR) mode selection for accumulator building.
//!
//! This module provides compile-time selection between DMR-enabled and DMR-disabled
//! accumulation strategies in [`super::build_accumulators_spartan`].
//!
//! # Design
//!
//! Uses marker types + trait with associated types to select behavior at compile time:
//! - [`DelayedModularReductionEnabled`]: Accumulates into unreduced wide-limb form, reduces once at end
//! - [`DelayedModularReductionDisabled`]: Immediate Montgomery reduction on each operation (baseline)
//!
//! This gives zero wasted allocations and zero runtime branching overhead.
//!
//! # Traits
//!
//! - [`AccumulateProduct`]: Bridges product types (i64, i128, F) to their unreduced accumulators
//! - [`DelayedModularReductionMode`]: Defines element-level accumulation operations for both F×int and F×F

use super::mat_vec_mle::MatVecMLE;
use crate::small_field::DelayedReduction;
use ff::PrimeField;
use num_traits::Zero;
use std::{
  fmt::Debug,
  marker::PhantomData,
  ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

// =============================================================================
// AccumulateProduct trait
// =============================================================================

/// Bridges product types to their unreduced accumulators.
///
/// This trait connects `MLE::Product` (i64, i128, or F) to the appropriate
/// unreduced accumulator type. It enables `DelayedModularReductionEnabled`
/// to work generically with any product type.
///
/// # Type Parameters
///
/// - `F`: The field type to reduce to
///
/// # Implementations
///
/// - `i64: AccumulateProduct<F>` where `Unreduced = F::UnreducedFieldInt` (for i32 witnesses)
/// - `i128: AccumulateProduct<F>` where `Unreduced = F::UnreducedFieldInt` (for i64 witnesses)
pub trait AccumulateProduct<F: PrimeField>: Copy {
  /// The unreduced accumulator type for this product.
  /// Has Zero bound so callers can use `.is_zero()` directly.
  type Unreduced: Copy + Clone + Default + AddAssign + Send + Sync + Zero;

  /// Accumulate `e × prod` into `acc` (no modular reduction).
  fn accumulate(acc: &mut Self::Unreduced, e: &F, prod: Self);

  /// Reduce the accumulator to a field element.
  fn reduce(acc: &Self::Unreduced) -> F;
}

// i64 products (from i32 witnesses)
impl<F> AccumulateProduct<F> for i64
where
  F: DelayedReduction<i32, IntermediateSmallValue = i64>,
{
  type Unreduced = F::UnreducedFieldInt;

  #[inline]
  fn accumulate(acc: &mut Self::Unreduced, e: &F, prod: i64) {
    // Single-value accumulation
    F::unreduced_field_intermediate_mul_add(acc, e, prod);
  }

  #[inline]
  fn reduce(acc: &Self::Unreduced) -> F {
    F::reduce_field_int(acc)
  }
}

// i128 products (from i64 witnesses)
impl<F> AccumulateProduct<F> for i128
where
  F: DelayedReduction<i64, IntermediateSmallValue = i128>,
{
  type Unreduced = F::UnreducedFieldInt;

  #[inline]
  fn accumulate(acc: &mut Self::Unreduced, e: &F, prod: i128) {
    // Single-value accumulation
    F::unreduced_field_intermediate_mul_add(acc, e, prod);
  }

  #[inline]
  fn reduce(acc: &Self::Unreduced) -> F {
    F::reduce_field_int(acc)
  }
}

// Note: We don't implement AccumulateProduct<F> for F (field products)
// because field witnesses should use DelayedModularReductionDisabled, not DelayedModularReductionEnabled.
// DelayedModularReductionEnabled requires DelayedReduction<SmallValue> which isn't available for field witnesses.

// =============================================================================
// DelayedModularReductionMode trait
// =============================================================================

/// Compile-time selection of delayed modular reduction strategy.
///
/// This trait defines element-level accumulation operations for both
/// F×int (inner loop) and F×F (accumulator building) phases.
///
/// # Type Parameters
///
/// - `F`: Field type (e.g., `pallas::Scalar`)
/// - `MLE`: Polynomial type implementing [`MatVecMLE`]
/// - `D`: Polynomial degree bound (2 for Spartan)
///
/// # Associated Types
///
/// - `AccumulatedFieldInt`: Accumulator for F×int products (per-beta)
/// - `AccumulatedFieldField`: Accumulator for F×F products (per-bucket)
/// - `UnreducedField`: Unreduced field representation for cache and intermediate values
pub trait DelayedModularReductionMode<F, MLE, const D: usize>: Sized
where
  F: PrimeField + Send + Sync,
  MLE: MatVecMLE<F>,
{
  /// Accumulator for Field × Int products (inner loop).
  ///
  /// - `Enabled`: `<MLE::Product as AccumulateProduct<F>>::Unreduced` (wide limbs)
  /// - `Disabled`: `F` (immediate reduction)
  type AccumulatedFieldInt: Copy + Default + Send;

  /// Accumulator for Field × Field products (accumulator building).
  ///
  /// - `Enabled`: `F::UnreducedFieldField` (9-limb accumulator)
  /// - `Disabled`: `F` (immediate reduction)
  type AccumulatedFieldField: Copy + Default + Send + AddAssign;

  /// Unreduced field representation for cache and intermediate values.
  ///
  /// - `Enabled`: `F::UnreducedField` (raw [u64;4])
  /// - `Disabled`: `F`
  type UnreducedField: Copy + Default + PartialEq + Send + Sync;

  // ===========================================================================
  // Inner loop operations (F×int)
  // ===========================================================================

  /// Precompute e_in cache in appropriate form for inner loop.
  ///
  /// - `Enabled`: converts to raw limbs via `to_unreduced()` (eliminates `from_mont` in reduction)
  /// - `Disabled`: returns as-is (UnreducedField = F)
  fn precompute_e_in(e_in: &[F]) -> Vec<Self::UnreducedField>;

  /// Accumulate eq-weighted product into partial sum.
  ///
  /// Called in hot inner loop over x_in for each beta with infinity.
  /// Takes two small values directly (not pre-multiplied) for better optimization.
  fn accumulate_eq_product(sum: &mut Self::AccumulatedFieldInt, e: &Self::UnreducedField, a: MLE::Value, b: MLE::Value);

  /// Check if partial sum is zero.
  fn partial_sum_is_zero(sum: &Self::AccumulatedFieldInt) -> bool;

  // ===========================================================================
  // Accumulator building operations
  // ===========================================================================

  /// Build eq cache by precomputing `e_xout[x] × e_y[round][y]` for all combinations.
  ///
  /// Returns a nested vector indexed as `cache[round][y_idx * num_x_out + x_out]`.
  fn build_eq_cache(e_xout: &[F], e_y: &[Vec<F>]) -> Vec<Vec<Self::UnreducedField>>;

  /// Reduce accumulated F×int to unreduced field representation.
  fn reduce_field_int(sum: &Self::AccumulatedFieldInt) -> Self::UnreducedField;

  /// Check if unreduced field value is zero.
  fn unreduced_is_zero(val: &Self::UnreducedField) -> bool;

  /// Multiply-accumulate two unreduced field values into F×F accumulator.
  ///
  /// `elem += val × eq_eval`
  fn unreduced_mul_add(
    elem: &mut Self::AccumulatedFieldField,
    val: &Self::UnreducedField,
    eq_eval: &Self::UnreducedField,
  );

  // ===========================================================================
  // Final reduction operations
  // ===========================================================================

  /// Check if accumulated F×F value is zero.
  fn is_accumulated_field_field_zero(elem: &Self::AccumulatedFieldField) -> bool;

  /// Reduce accumulated F×F to field element.
  fn reduce_field_field(elem: &Self::AccumulatedFieldField) -> F;
}

// =============================================================================
// DelayedModularReductionEnabled<SmallValue>
// =============================================================================

/// Marker type for delayed modular reduction (optimized path).
///
/// Accumulates F×int products into unreduced form during the inner loop,
/// and F×F products into unreduced form during scatter. Reduces once at the end.
///
/// # Type Parameters
///
/// - `SmallValue`: The small integer witness type (i32 or i64)
pub struct DelayedModularReductionEnabled<SmallValue>(PhantomData<SmallValue>);

impl<SmallValue> Default for DelayedModularReductionEnabled<SmallValue> {
  fn default() -> Self {
    Self(PhantomData)
  }
}

impl<F, MLE, SmallValue, const D: usize> DelayedModularReductionMode<F, MLE, D>
  for DelayedModularReductionEnabled<SmallValue>
where
  F: PrimeField + DelayedReduction<SmallValue> + Send + Sync,
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
  MLE: MatVecMLE<F, Value = SmallValue>,
{
  type AccumulatedFieldInt = F::UnreducedFieldInt;
  type AccumulatedFieldField = F::UnreducedFieldField;
  type UnreducedField = F::UnreducedField;

  fn precompute_e_in(e_in: &[F]) -> Vec<Self::UnreducedField> {
    e_in.iter().map(|e| e.to_unreduced()).collect()
  }

  #[inline]
  fn accumulate_eq_product(sum: &mut Self::AccumulatedFieldInt, e: &Self::UnreducedField, a: MLE::Value, b: MLE::Value) {
    // Use raw limb accumulation - e is already non-R-scaled
    F::unreduced_raw_small_mul_add(sum, e, a, b);
  }

  #[inline]
  fn partial_sum_is_zero(sum: &Self::AccumulatedFieldInt) -> bool {
    sum.is_zero()
  }

  fn build_eq_cache(e_xout: &[F], e_y: &[Vec<F>]) -> Vec<Vec<Self::UnreducedField>> {
    e_y
      .iter()
      .map(|round_ey| {
        round_ey
          .iter()
          .flat_map(|ey| e_xout.iter().map(|ex| (*ey * *ex).to_unreduced()))
          .collect()
      })
      .collect()
  }

  #[inline]
  fn reduce_field_int(sum: &Self::AccumulatedFieldInt) -> Self::UnreducedField {
    // Accumulator is already non-R-scaled (from using unreduced_raw_small_mul_add),
    // so just Barrett reduce without from_mont conversion
    F::reduce_raw_field_int_to_unreduced(sum)
  }

  #[inline]
  fn unreduced_is_zero(val: &Self::UnreducedField) -> bool {
    *val == Self::UnreducedField::default()
  }

  #[inline]
  fn unreduced_mul_add(
    elem: &mut Self::AccumulatedFieldField,
    val: &Self::UnreducedField,
    eq_eval: &Self::UnreducedField,
  ) {
    F::unreduced_raw_mul_add(elem, val, eq_eval);
  }

  #[inline]
  fn is_accumulated_field_field_zero(elem: &Self::AccumulatedFieldField) -> bool {
    elem.is_zero()
  }

  #[inline]
  fn reduce_field_field(elem: &Self::AccumulatedFieldField) -> F {
    F::barrett_reduce_field_field(elem)
  }
}

// =============================================================================
// DelayedModularReductionDisabled
// =============================================================================

/// Marker type for immediate reduction (baseline path).
///
/// Performs Montgomery reduction on each accumulation operation.
/// Used for benchmarking to measure DMR speedup.
pub struct DelayedModularReductionDisabled;

impl<F, MLE, const D: usize> DelayedModularReductionMode<F, MLE, D>
  for DelayedModularReductionDisabled
where
  F: PrimeField + Send + Sync,
  MLE: MatVecMLE<F>,
{
  type AccumulatedFieldInt = F;
  type AccumulatedFieldField = F;
  type UnreducedField = F;

  fn precompute_e_in(e_in: &[F]) -> Vec<Self::UnreducedField> {
    e_in.to_vec()
  }

  #[inline]
  fn accumulate_eq_product(sum: &mut F, e: &F, a: MLE::Value, b: MLE::Value) {
    // Immediate reduction: compute product and multiply by e
    let prod = MLE::multiply_witnesses(a, b);
    let field_prod = MLE::product_to_field(prod);
    *sum += *e * field_prod;
  }

  #[inline]
  fn partial_sum_is_zero(sum: &F) -> bool {
    ff::Field::is_zero(sum).into()
  }

  fn build_eq_cache(e_xout: &[F], e_y: &[Vec<F>]) -> Vec<Vec<Self::UnreducedField>> {
    e_y
      .iter()
      .map(|round_ey| {
        round_ey
          .iter()
          .flat_map(|ey| e_xout.iter().map(|ex| *ey * *ex))
          .collect()
      })
      .collect()
  }

  #[inline]
  fn reduce_field_int(sum: &F) -> F {
    *sum // Already reduced
  }

  #[inline]
  fn unreduced_is_zero(val: &F) -> bool {
    ff::Field::is_zero(val).into()
  }

  #[inline]
  fn unreduced_mul_add(elem: &mut F, val: &F, eq_eval: &F) {
    *elem += *eq_eval * *val;
  }

  #[inline]
  fn is_accumulated_field_field_zero(elem: &F) -> bool {
    ff::Field::is_zero(elem).into()
  }

  #[inline]
  fn reduce_field_field(elem: &F) -> F {
    *elem // Already reduced
  }
}
