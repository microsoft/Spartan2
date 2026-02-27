// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value field operations for optimized sumcheck.
//!
//! This module provides types and operations for the small-value optimization
//! described in "Speeding Up Sum-Check Proving". The key insight is that
//! polynomial evaluations on the boolean hypercube are typically small integers
//! (i32/i64 values), so we can perform many operations in native integers before
//! converting to field elements.

pub(crate) mod barrett;
mod delayed_reduction;
pub(crate) mod field_reduction_constants;
pub(crate) mod limbs;
pub(crate) mod montgomery;
mod small_value_field;
mod wide_mul;

// Public exports
pub use delayed_reduction::DelayedReduction;
pub use limbs::{SignedWideLimbs, SubMagResult, WideLimbs, sub_mag};
pub use small_value_field::{SmallValueField, i64_to_field, try_field_to_i64};
pub use wide_mul::WideMul;

use crate::errors::SpartanError;
use num_traits::{Bounded, One, Signed};
use rayon::prelude::*;

// ============================================================================
// ExtensionBound
// ============================================================================

/// Precomputed bound for Lagrange extension safety checks.
///
/// When extending a multilinear polynomial from the boolean hypercube {0,1}^n to
/// the Lagrange domain {0,1,...,D}^n, values grow by a factor of (D+1)^n due to
/// Lagrange interpolation coefficients. For D=2, this is 3^n.
///
/// This struct caches `max_safe = SV::max_value() / (D+1)^lb` to avoid recomputing
/// in hot loops. Values with `|v| <= max_safe` can safely survive extension.
///
/// # Type Parameters
/// - `SV`: Small value type (i32, i64)
/// - `D`: Polynomial degree (typically 2)
///
/// # Example
/// ```ignore
/// let bound = ExtensionBound::<i64, 2>::new(4);
/// // Check values in a loop - bound computed once
/// for val in field_values {
///     if let Some(small) = bound.try_to_small(&val) { ... }
/// }
/// ```
pub struct ExtensionBound<SV: WideMul, const D: usize> {
  max_safe: SV::Product,
}

impl<SV, const D: usize> ExtensionBound<SV, D>
where
  SV: WideMul + Bounded + Copy + Into<SV::Product>,
  SV::Product: Copy
    + Ord
    + Signed
    + std::ops::Div<Output = SV::Product>
    + std::ops::Mul<Output = SV::Product>
    + One
    + From<i32>,
{
  /// Create a new bound for `lb` extension rounds.
  ///
  /// Precomputes `max_safe = SV::max_value() / (D+1)^lb`, the largest absolute
  /// value that can safely survive Lagrange extension without overflowing `SV`.
  ///
  /// # The growth factor
  ///
  /// At each variable, Lagrange extension from {0,1} to {0,1,2} uses coefficients
  /// with absolute sum = 3. After `lb` rounds, values can grow by up to 3^lb.
  ///
  /// Examples for D=2 (i64, max ≈ 2^63):
  /// - lb=4: max_safe ≈ 2^56
  /// - lb=26: max_safe ≈ 2^21
  pub fn new(lb: usize) -> Self {
    let base: SV::Product = (D as i32 + 1).into();
    let mut power = SV::Product::one();
    for _ in 0..lb {
      power = power * base;
    }
    let max_value: SV::Product = SV::max_value().into();
    let max_safe = max_value / power;
    Self { max_safe }
  }

  /// Check if a small value is safe for Lagrange extension.
  ///
  /// Returns `true` if `|small| <= max_safe`.
  #[inline]
  pub fn is_safe(&self, small: SV) -> bool {
    let abs_small: SV::Product = small.into();
    abs_small.abs() <= self.max_safe
  }

  /// Get the precomputed max_safe value (for error messages).
  #[inline]
  pub fn max_safe(&self) -> SV::Product {
    self.max_safe
  }

  /// Try to convert a field element to a small value with extension bound check.
  ///
  /// Returns `Some(small)` if the field element can be represented as `SV`
  /// and `|small| <= max_safe`. Returns `None` otherwise.
  #[inline]
  pub fn try_to_small<F>(&self, val: &F) -> Option<SV>
  where
    F: SmallValueField<SV>,
  {
    let small = F::try_field_to_small(val)?;
    if self.is_safe(small) {
      Some(small)
    } else {
      None
    }
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert a vector of field elements to small values (parallel).
///
/// Returns Error if any value doesn't fit in the small value type.
pub fn vec_to_small<F, SV>(v: &[F]) -> Result<Vec<SV>, SpartanError>
where
  F: SmallValueField<SV> + Sync,
  SV: Copy + Send + Sync,
{
  v.par_iter()
    .enumerate()
    .map(|(i, f)| {
      F::try_field_to_small(f).ok_or_else(|| SpartanError::SmallValueOverflow {
        value: format!("{:?}", f),
        context: format!("vec_to_small at index {}", i),
      })
    })
    .collect()
}

/// Convert a vector of field elements to SmallValue with bound check for Lagrange extension.
///
/// Uses [`ExtensionBound`] to check that values are small enough to survive
/// Lagrange extension with growth factor `(D+1)^lb`.
///
/// # Type Parameters
///
/// - `SV`: The small value type (e.g., i32, i64)
/// - `D`: The polynomial degree for Lagrange extension (typically 2)
///
/// # Arguments
///
/// - `v`: Vector of field elements to convert
/// - `lb`: Number of Lagrange extension rounds (determines growth factor of (D+1)^lb)
///
/// # Returns
///
/// `Ok(Vec<SV>)` if all values fit within the safe bound, or `Err` with details.
pub fn vec_to_small_for_extension<F, SV, const D: usize>(
  v: &[F],
  lb: usize,
) -> Result<Vec<SV>, SpartanError>
where
  F: SmallValueField<SV> + Sync,
  SV: WideMul + Bounded + Copy + Send + Sync + Into<SV::Product>,
  SV::Product: Copy
    + Ord
    + Signed
    + std::ops::Div<Output = SV::Product>
    + std::ops::Mul<Output = SV::Product>
    + One
    + From<i32>,
{
  // Compute bound once, check all values
  let bound = ExtensionBound::<SV, D>::new(lb);

  v.par_iter()
    .enumerate()
    .map(|(i, f)| {
      bound
        .try_to_small(f)
        .ok_or_else(|| SpartanError::SmallValueOverflow {
          value: format!("{:?}", f),
          context: format!(
            "vec_to_small_for_extension: value at index {} exceeds bound for D={}, lb={}",
            i, D, lb
          ),
        })
    })
    .collect()
}
