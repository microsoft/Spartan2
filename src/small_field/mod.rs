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
mod impls;
pub(crate) mod limbs;
pub(crate) mod montgomery;
mod small_value_field;

pub use delayed_reduction::DelayedReduction;
pub use limbs::{SignedWideLimbs, SubMagResult, WideLimbs, sub_mag};
pub use small_value_field::SmallValueField;

use montgomery::MontgomeryLimbs;

/// Marker trait: field supports `SmallValueField<i32>` via blanket impl.
pub(crate) trait SupportsSmallI32: MontgomeryLimbs {}

/// Marker trait: field supports `SmallValueField<i64>` via blanket impl.
pub(crate) trait SupportsSmallI64: MontgomeryLimbs {}

use crate::errors::SpartanError;
use ff::PrimeField;
use rayon::prelude::*;

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert i64 to field element (handles negative values correctly).
#[inline]
pub fn i64_to_field<F: PrimeField>(val: i64) -> F {
  if val >= 0 {
    F::from(val as u64)
  } else {
    // Use wrapping_neg to handle i64::MIN correctly
    -F::from(val.wrapping_neg() as u64)
  }
}

/// Try to convert a field element to i64.
/// Returns None if the value doesn't fit in the i64 range.
#[inline]
pub fn try_field_to_i64<F: PrimeField>(val: &F) -> Option<i64> {
  let repr = val.to_repr();
  let bytes = repr.as_ref();

  // Check if value fits in positive i64 (high bytes all zero)
  let high_zero = bytes[8..].iter().all(|&b| b == 0);
  if high_zero {
    let val_u64 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    if val_u64 <= i64::MAX as u64 {
      return Some(val_u64 as i64);
    }
  }

  // Check if negation fits in i64 (value is negative)
  let neg_val = val.neg();
  let neg_repr = neg_val.to_repr();
  let neg_bytes = neg_repr.as_ref();
  let neg_high_zero = neg_bytes[8..].iter().all(|&b| b == 0);
  if neg_high_zero {
    let neg_u64 = u64::from_le_bytes(neg_bytes[..8].try_into().unwrap());
    if neg_u64 > 0 && neg_u64 <= (i64::MAX as u64) + 1 {
      return Some(-(neg_u64 as i128) as i64);
    }
  }

  None
}

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

/// Try to convert a field element to i64 with bound check for Lagrange extension.
///
/// # Why this bound exists
///
/// When extending a multilinear polynomial from the boolean hypercube {0,1}^n to
/// the Lagrange domain {0,1,...,D}^n, values can grow significantly due to
/// Lagrange interpolation coefficients.
///
/// ## The growth factor
///
/// At each variable, we extend from 2 points {0,1} to D+1 points {0,1,...,D}.
/// The extension formula for point k > 1 uses Lagrange extrapolation:
///
///   p(k) = p(0) + k·(p(1) - p(0))
///
/// For k = 2 (when D ≥ 2):
///   p(2) = 2·p(1) - p(0)
///
/// The coefficients are [-1, +2], with sum of absolute values = 3.
///
/// ## Compounding over n variables
///
/// After extending all n variables, the value at corner point (2,2,...,2) is a
/// linear combination of all 2^n original boolean evaluations. The coefficient
/// of each original value v_b is:
///
///   ∏ᵢ (if bᵢ = 0 then -1 else 2)
///
/// The sum of absolute values of all coefficients is:
///
///   Σ_b |coeff_b| = ∏ᵢ (|-1| + |2|) = 3^n
///
/// More generally, for degree D: the growth factor is (D+1)^n.
///
/// ## The bound
///
/// To ensure the extended values fit in i64 (max ≈ 2^63), we require:
///
///   |input| × (D+1)^lb ≤ 2^63
///   |input| ≤ 2^63 / (D+1)^lb
///
/// Examples for D=2:
///   - lb=3: max |input| ≈ 2^58 (341 quadrillion)
///   - lb=4: max |input| ≈ 2^56 (72 quadrillion)
///   - lb=26: max |input| ≈ 2^21 (2 million)
///
/// This check ensures we can use pure i64 arithmetic throughout the extension
/// without overflow, avoiding the need for i128 intermediate values.
#[inline]
pub fn try_field_to_small_for_extension<F, const D: usize>(val: &F, lb: usize) -> Option<i64>
where
  F: SmallValueField<i64>,
{
  let small = F::try_field_to_small(val)?;
  let base = (D + 1) as i64;
  let max_safe = (i64::MAX / base.pow(lb as u32)) as u64;
  if small.unsigned_abs() <= max_safe {
    Some(small)
  } else {
    None
  }
}

/// Convert a vector of field elements to i64 with bound check for Lagrange extension.
///
/// See [`try_field_to_small_for_extension`] for details on why this bound is needed.
///
/// # Type Parameters
///
/// - `D`: The polynomial degree for Lagrange extension (typically 2)
///
/// # Arguments
///
/// - `v`: Vector of field elements to convert
/// - `lb`: Number of Lagrange extension rounds (determines growth factor of (D+1)^lb)
///
/// # Returns
///
/// `Ok(Vec<i64>)` if all values fit within the safe bound, or `Err` with details.
pub fn vec_to_small_for_extension<F, const D: usize>(
  v: &[F],
  lb: usize,
) -> Result<Vec<i64>, SpartanError>
where
  F: SmallValueField<i64> + Sync,
{
  let base = (D + 1) as i64;
  let max_safe = (i64::MAX / base.pow(lb as u32)) as u64;

  v.par_iter()
    .enumerate()
    .map(|(i, f)| {
      let small = F::try_field_to_small(f).ok_or_else(|| SpartanError::SmallValueOverflow {
        value: format!("{:?}", f),
        context: format!("vec_to_small_for_extension at index {}", i),
      })?;
      if small.unsigned_abs() <= max_safe {
        Ok(small)
      } else {
        Err(SpartanError::SmallValueOverflow {
          value: format!("{}", small),
          context: format!(
            "vec_to_small_for_extension at index {}: |{}| > {} (max safe for D={}, lb={})",
            i, small, max_safe, D, lb
          ),
        })
      }
    })
    .collect()
}

