// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value field operations for optimized sumcheck.
//!
//! This module provides types and operations for the small-value optimization
//! described in "Speeding Up Sum-Check Proving". The key insight is distinguishing
//! between three multiplication types:
//!
//! - **ss** (small × small): Native i32/i64 multiplication
//! - **sl** (small × large): Barrett-optimized multiplication (~3× faster)
//! - **ll** (large × large): Standard field multiplication
//!
//! For polynomial evaluations on the boolean hypercube (typically i32 values),
//! we can perform many operations in native integers before converting to field.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    SmallValueField Trait                     │
//! │  (ss_mul, sl_mul, isl_mul, small_to_field, etc.)            │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Barrett Reduction (internal)                    │
//! │  mul_fp_by_i64, mul_fq_by_i64 - ~9 base muls vs ~32 naive   │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod small_value_field;
mod delayed_reduction;
pub(crate) mod barrett;
mod impls;

pub use delayed_reduction::DelayedReduction;
pub use small_value_field::SmallValueField;

use ff::PrimeField;

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

/// Convert i128 to field element (handles negative values correctly).
#[inline]
pub fn i128_to_field<F: PrimeField>(val: i128) -> F {
  if val >= 0 {
    // Split into high and low u64 parts
    let low = val as u64;
    let high = (val >> 64) as u64;
    if high == 0 {
      F::from(low)
    } else {
      // result = low + high * 2^64
      F::from(low) + F::from(high) * two_pow_64::<F>()
    }
  } else {
    // Use wrapping_neg to handle i128::MIN correctly
    let pos = val.wrapping_neg() as u128;
    let low = pos as u64;
    let high = (pos >> 64) as u64;
    if high == 0 {
      -F::from(low)
    } else {
      -(F::from(low) + F::from(high) * two_pow_64::<F>())
    }
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

/// Returns 2^64 as a field element (cached via lazy computation).
#[inline]
fn two_pow_64<F: PrimeField>() -> F {
  // 2^64 = (2^32)^2
  let two_32 = F::from(1u64 << 32);
  two_32 * two_32
}

