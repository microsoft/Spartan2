// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Trait for computing matrix-vector MLE evaluations.
//!
//! This module provides [`MatVecMLE`], which computes multilinear extensions of
//! matrix-vector products Az, Bz, Cz. It abstracts over:
//! - Field-element witnesses (`MultilinearPolynomial<S>`)
//! - Small-value witnesses (`MultilinearPolynomial<i32>`) for the small-value optimization

use crate::{polys::multilinear::MultilinearPolynomial, small_field::DelayedReduction};
use ff::PrimeField;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Sub};

/// Trait for multilinear extensions of matrix-vector products (Az, Bz, Cz).
///
/// Abstracts over field-element witnesses vs small-value (i32) witnesses,
/// enabling the same accumulator-building code to work with both representations.
///
/// # Type Parameters
///
/// - `S`: The field type used for accumulation (always a `PrimeField`)
///
/// # Associated Types
///
/// - `Value`: The witness coefficient type (`S` for field polynomials, `i32` for small-value)
/// - `Product`: The product type (`S` for field, `i64` for small-value to avoid overflow)
/// - `UnreducedSum`: Accumulator for delayed modular reduction
pub trait MatVecMLE<S: PrimeField>: Sync {
  /// The witness value type (S for field, i32 for small)
  type Value: Copy + Default + Add<Output = Self::Value> + Sub<Output = Self::Value> + Send + Sync;

  /// The product type (S for field, i64 for small)
  type Product;

  /// Unreduced accumulator for delayed modular reduction.
  ///
  /// For small-value polynomials: pair of `WideLimbs<6>` (positive, negative).
  /// For field polynomials: just `S` (no delayed reduction in baseline).
  type UnreducedSum: Copy + Clone + Default + AddAssign + Send + Sync;

  /// Get witness value at index
  fn get(&self, idx: usize) -> Self::Value;

  /// Get polynomial length
  fn len(&self) -> usize;

  /// Multiply two witness values: a × b
  fn multiply_witnesses(a: Self::Value, b: Self::Value) -> Self::Product;

  /// Accumulate eq-weighted product: sum += e_in_eval * prod (with immediate reduction)
  #[allow(dead_code)] // Kept for debugging/testing; main path uses unreduced version
  fn accumulate_eq_product(prod: Self::Product, e_in_eval: &S, sum: &mut S);

  /// Accumulate eq-weighted product into unreduced form (no modular reduction).
  ///
  /// This is the key method for delayed reduction optimization. Instead of
  /// reducing after every multiplication, we accumulate into a wide integer
  /// and reduce once at the end via `modular_reduction`.
  fn accumulate_eq_product_unreduced(
    acc: &mut Self::UnreducedSum,
    prod: Self::Product,
    e_in_eval: &S,
  );

  /// Reduce accumulated unreduced sum to a field element.
  fn modular_reduction(acc: &Self::UnreducedSum) -> S;

  /// Fast check if unreduced accumulator is zero (avoids expensive modular_reduction).
  fn unreduced_is_zero(acc: &Self::UnreducedSum) -> bool;
}

/// Implementation for field-element polynomials.
///
/// This is the standard case where witness coefficients are field elements.
/// No delayed reduction optimization - each accumulation reduces immediately.
impl<S: PrimeField + Sync> MatVecMLE<S> for MultilinearPolynomial<S> {
  type Value = S;
  type Product = S;
  /// No delayed reduction for field polynomials - UnreducedSum is just S.
  type UnreducedSum = S;

  fn get(&self, idx: usize) -> S {
    self.Z[idx]
  }

  fn len(&self) -> usize {
    self.Z.len()
  }

  fn multiply_witnesses(a: S, b: S) -> S {
    a * b
  }

  fn accumulate_eq_product(prod: S, e_in_eval: &S, sum: &mut S) {
    *sum += *e_in_eval * prod;
  }

  fn accumulate_eq_product_unreduced(acc: &mut S, prod: S, e_in_eval: &S) {
    // For field polynomials, just do immediate reduction (no optimization)
    *acc += *e_in_eval * prod;
  }

  fn modular_reduction(acc: &S) -> S {
    // Already reduced
    *acc
  }

  fn unreduced_is_zero(acc: &S) -> bool {
    acc.is_zero().into()
  }
}

/// Implementation for small-value polynomials (i32 coefficients).
///
/// This enables the small-value sumcheck optimization where witness coefficients
/// are known to be small integers. Products are computed as i64 to avoid overflow,
/// then multiplied with field elements using optimized Barrett reduction.
///
/// Delayed reduction: products are accumulated into unreduced form (SignedWideLimbs),
/// then reduced once at the end.
impl<S> MatVecMLE<S> for MultilinearPolynomial<i32>
where
  S: DelayedReduction<i32, IntermediateSmallValue = i64> + Sync,
{
  type Value = i32;
  type Product = i64;
  /// Unreduced accumulator for delayed reduction (handles sign internally).
  type UnreducedSum = S::UnreducedFieldInt;

  fn get(&self, idx: usize) -> i32 {
    self.Z[idx]
  }

  fn len(&self) -> usize {
    self.Z.len()
  }

  fn multiply_witnesses(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
  }

  fn accumulate_eq_product(prod: i64, e_in_eval: &S, sum: &mut S) {
    *sum += S::isl_mul(prod, e_in_eval);
  }

  fn accumulate_eq_product_unreduced(acc: &mut S::UnreducedFieldInt, prod: i64, e_in_eval: &S) {
    // Sign handling is done internally by unreduced_field_int_mul_add
    S::unreduced_field_int_mul_add(acc, e_in_eval, prod);
  }

  fn modular_reduction(acc: &S::UnreducedFieldInt) -> S {
    S::reduce_field_int(acc)
  }

  fn unreduced_is_zero(acc: &S::UnreducedFieldInt) -> bool {
    acc.is_zero()
  }
}

/// Implementation for i64-valued polynomials (i64 coefficients, i128 products).
///
/// This enables larger small-value witnesses that don't fit in i32.
/// Products are computed as i128 to avoid overflow.
///
/// Delayed reduction: products are accumulated into unreduced form (SignedWideLimbs<8>),
/// then reduced once at the end.
impl<S> MatVecMLE<S> for MultilinearPolynomial<i64>
where
  S: DelayedReduction<i64, IntermediateSmallValue = i128> + Sync,
{
  type Value = i64;
  type Product = i128;
  /// Unreduced accumulator for delayed reduction (8 limbs for wider products).
  type UnreducedSum = S::UnreducedFieldInt;

  fn get(&self, idx: usize) -> i64 {
    self.Z[idx]
  }

  fn len(&self) -> usize {
    self.Z.len()
  }

  fn multiply_witnesses(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }

  fn accumulate_eq_product(prod: i128, e_in_eval: &S, sum: &mut S) {
    *sum += S::isl_mul(prod, e_in_eval);
  }

  fn accumulate_eq_product_unreduced(acc: &mut S::UnreducedFieldInt, prod: i128, e_in_eval: &S) {
    S::unreduced_field_int_mul_add(acc, e_in_eval, prod);
  }

  fn modular_reduction(acc: &S::UnreducedFieldInt) -> S {
    S::reduce_field_int(acc)
  }

  fn unreduced_is_zero(acc: &S::UnreducedFieldInt) -> bool {
    acc.is_zero()
  }
}
