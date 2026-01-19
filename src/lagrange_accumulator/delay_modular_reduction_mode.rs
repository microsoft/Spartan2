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
    F::unreduced_field_int_mul_add(acc, e, prod);
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
    F::unreduced_field_int_mul_add(acc, e, prod);
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
/// F×int (inner loop) and F×F (scatter phase) accumulation.
///
/// # Type Parameters
///
/// - `F`: Field type (e.g., `pallas::Scalar`)
/// - `MLE`: Polynomial type implementing [`MatVecMLE`]
/// - `D`: Polynomial degree bound (2 for Spartan)
///
/// # Associated Types
///
/// - `PartialSum`: Element type for F×int accumulation (per-beta)
/// - `ScatterElement`: Element type for F×F accumulation (per-bucket)
pub trait DelayedModularReductionMode<F, MLE, const D: usize>: Sized
where
  F: PrimeField + Send + Sync,
  MLE: MatVecMLE<F>,
{
  /// Element type for F×int partial sums (per-beta accumulation).
  ///
  /// - `Enabled`: `<MLE::Product as AccumulateProduct<F>>::Unreduced` (delayed reduction)
  /// - `Disabled`: `F` (immediate reduction)
  type PartialSum: Copy + Default + Send;

  /// Element type for F×F scatter buckets.
  ///
  /// - `Enabled`: `F::UnreducedFieldField` (delayed F×F)
  /// - `Disabled`: `F` (immediate F×F)
  type ScatterElement: Copy + Default + Send + AddAssign;

  // ===========================================================================
  // F×int operations (inner loop)
  // ===========================================================================

  /// Accumulate eq-weighted product into partial sum.
  ///
  /// Called in hot inner loop over x_in for each beta with infinity.
  fn accumulate_eq_product(sum: &mut Self::PartialSum, prod: MLE::Product, e: &F);

  /// Reduce partial sum to field element.
  fn modular_reduction_partial_sum(sum: &Self::PartialSum) -> F;

  /// Check if partial sum is zero.
  fn partial_sum_is_zero(sum: &Self::PartialSum) -> bool;

  // ===========================================================================
  // F×F operations (scatter phase)
  // ===========================================================================

  /// Accumulate `ey × z_beta` into scatter element.
  fn accumulate_scatter(elem: &mut Self::ScatterElement, ey: &F, z_beta: &F);

  /// Reduce scatter element to field element.
  fn modular_reduction_scatter(elem: &Self::ScatterElement) -> F;

  /// Check if scatter element is zero.
  fn scatter_element_is_zero(elem: &Self::ScatterElement) -> bool;
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
  MLE: MatVecMLE<F>,
  MLE::Product: AccumulateProduct<F>,
{
  type PartialSum = <MLE::Product as AccumulateProduct<F>>::Unreduced;
  type ScatterElement = F::UnreducedFieldField;

  #[inline]
  fn accumulate_eq_product(sum: &mut Self::PartialSum, prod: MLE::Product, e: &F) {
    MLE::Product::accumulate(sum, e, prod);
  }

  #[inline]
  fn modular_reduction_partial_sum(sum: &Self::PartialSum) -> F {
    MLE::Product::reduce(sum)
  }

  #[inline]
  fn partial_sum_is_zero(sum: &Self::PartialSum) -> bool {
    sum.is_zero()
  }

  #[inline]
  fn accumulate_scatter(elem: &mut Self::ScatterElement, ey: &F, z_beta: &F) {
    F::unreduced_field_field_mul_add(elem, ey, z_beta);
  }

  #[inline]
  fn modular_reduction_scatter(elem: &Self::ScatterElement) -> F {
    F::reduce_field_field(elem)
  }

  #[inline]
  fn scatter_element_is_zero(elem: &Self::ScatterElement) -> bool {
    elem.is_zero()
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
  type PartialSum = F;
  type ScatterElement = F;

  #[inline]
  fn accumulate_eq_product(sum: &mut F, prod: MLE::Product, e: &F) {
    // Immediate reduction: convert product to field element directly
    let field_prod = MLE::product_to_field(prod);
    *sum += *e * field_prod;
  }

  #[inline]
  fn modular_reduction_partial_sum(sum: &F) -> F {
    *sum // Already reduced
  }

  #[inline]
  fn partial_sum_is_zero(sum: &F) -> bool {
    ff::Field::is_zero(sum).into()
  }

  #[inline]
  fn accumulate_scatter(elem: &mut F, ey: &F, z_beta: &F) {
    *elem += *ey * *z_beta; // Immediate F×F
  }

  #[inline]
  fn modular_reduction_scatter(elem: &F) -> F {
    *elem // Already reduced
  }

  #[inline]
  fn scatter_element_is_zero(elem: &F) -> bool {
    ff::Field::is_zero(elem).into()
  }
}
