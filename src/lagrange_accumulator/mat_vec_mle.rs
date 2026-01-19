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
//! - Small-value witnesses (`MultilinearPolynomial<i32>`, `MultilinearPolynomial<i64>`)

use crate::{polys::multilinear::MultilinearPolynomial, small_field::SmallValueField};
use ff::PrimeField;
use std::ops::{Add, Sub};

/// Trait for multilinear extensions of matrix-vector products (Az, Bz, Cz).
///
/// Abstracts over field-element witnesses vs small-value (i32/i64) witnesses,
/// enabling the same accumulator-building code to work with both representations.
///
/// # Type Parameters
///
/// - `S`: The field type used for accumulation (always a `PrimeField`)
///
/// # Associated Types
///
/// - `Value`: The witness coefficient type (`S` for field polynomials, `i32`/`i64` for small-value)
/// - `Product`: The product type (`S` for field, `i64`/`i128` for small-value to avoid overflow)
pub trait MatVecMLE<S: PrimeField>: Sync {
  /// The witness value type (S for field, i32/i64 for small)
  type Value: Copy + Default + Add<Output = Self::Value> + Sub<Output = Self::Value> + Send + Sync;

  /// The product type (S for field, i64/i128 for small)
  type Product: Copy;

  /// Get witness value at index
  fn get(&self, idx: usize) -> Self::Value;

  /// Get polynomial length
  fn len(&self) -> usize;

  /// Returns true if the polynomial is empty.
  fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Multiply two witness values: a × b
  fn multiply_witnesses(a: Self::Value, b: Self::Value) -> Self::Product;

  /// Convert a product to a field element (for immediate reduction path).
  /// This is used by `DelayedModularReductionDisabled` to avoid going through unreduced form.
  fn product_to_field(prod: Self::Product) -> S;
}

/// Macro to implement MatVecMLE for field-element polynomials.
/// This avoids conflicting with the i32/i64 impls due to Rust's coherence rules.
macro_rules! impl_mat_vec_mle_for_field {
  ($($field:ty),* $(,)?) => {
    $(
      impl MatVecMLE<$field> for MultilinearPolynomial<$field> {
        type Value = $field;
        type Product = $field;

        #[inline]
        fn get(&self, idx: usize) -> $field {
          self.Z[idx]
        }

        #[inline]
        fn len(&self) -> usize {
          self.Z.len()
        }

        #[inline]
        fn multiply_witnesses(a: $field, b: $field) -> $field {
          a * b
        }

        #[inline]
        fn product_to_field(prod: $field) -> $field {
          prod // Already a field element
        }
      }
    )*
  };
}

// Implement for supported field types
use crate::provider::{
  bn254::bn254,
  pasta::{pallas, vesta},
};

impl_mat_vec_mle_for_field!(pallas::Scalar, vesta::Scalar, bn254::Scalar,);

/// Implementation for i32-valued polynomials (i32 coefficients, i64 products).
impl<S: PrimeField + SmallValueField<i32, IntermediateSmallValue = i64> + Sync> MatVecMLE<S>
  for MultilinearPolynomial<i32>
{
  type Value = i32;
  type Product = i64;

  #[inline]
  fn get(&self, idx: usize) -> i32 {
    self.Z[idx]
  }

  #[inline]
  fn len(&self) -> usize {
    self.Z.len()
  }

  #[inline]
  fn multiply_witnesses(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
  }

  #[inline]
  fn product_to_field(prod: i64) -> S {
    S::intermediate_to_field(prod)
  }
}

/// Implementation for i64-valued polynomials (i64 coefficients, i128 products).
impl<S: PrimeField + SmallValueField<i64, IntermediateSmallValue = i128> + Sync> MatVecMLE<S>
  for MultilinearPolynomial<i64>
{
  type Value = i64;
  type Product = i128;

  #[inline]
  fn get(&self, idx: usize) -> i64 {
    self.Z[idx]
  }

  #[inline]
  fn len(&self) -> usize {
    self.Z.len()
  }

  #[inline]
  fn multiply_witnesses(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }

  #[inline]
  fn product_to_field(prod: i128) -> S {
    S::intermediate_to_field(prod)
  }
}
