// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Trait abstraction for witness polynomials in Spartan accumulator building.
//!
//! This module provides [`SpartanAccumulatorInputPolynomial`], which abstracts over:
//! - Field-element witnesses (`MultilinearPolynomial<S>`)
//! - Small-value witnesses (`MultilinearPolynomial<i32>`) for the small-value optimization

use crate::{polys::multilinear::MultilinearPolynomial, small_field::SmallValueField};
use ff::PrimeField;
use std::ops::{Add, Sub};

/// Trait for witness polynomials used in Spartan accumulator building.
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
pub trait SpartanAccumulatorInputPolynomial<S: PrimeField>: Sync {
  /// The witness value type (S for field, i32 for small)
  type Value: Copy + Default + Add<Output = Self::Value> + Sub<Output = Self::Value> + Send + Sync;

  /// The product type (S for field, i64 for small)
  type Product;

  /// Get witness value at index
  fn get(&self, idx: usize) -> Self::Value;

  /// Get polynomial length
  fn len(&self) -> usize;

  /// Multiply two witness values: a × b
  fn multiply_witnesses(a: Self::Value, b: Self::Value) -> Self::Product;

  /// Accumulate eq-weighted product: sum += e_in_eval * prod
  fn accumulate_eq_product(prod: Self::Product, e_in_eval: &S, sum: &mut S);
}

/// Implementation for field-element polynomials.
///
/// This is the standard case where witness coefficients are field elements.
impl<S: PrimeField + Sync> SpartanAccumulatorInputPolynomial<S> for MultilinearPolynomial<S> {
  type Value = S;
  type Product = S;

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
}

/// Implementation for small-value polynomials (i32 coefficients).
///
/// This enables the small-value sumcheck optimization where witness coefficients
/// are known to be small integers. Products are computed as i64 to avoid overflow,
/// then multiplied with field elements using optimized Barrett reduction.
impl<S> SpartanAccumulatorInputPolynomial<S> for MultilinearPolynomial<i32>
where
  S: SmallValueField<SmallValue = i32, IntermediateSmallValue = i64> + Sync,
{
  type Value = i32;
  type Product = i64;

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
}
