// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallValueField trait for small-value optimization.
//!
//! This module provides generic helpers plus blanket `SmallValueField`
//! implementations for the supported scalar field types.

#![allow(dead_code)]

use super::montgomery::MontgomeryLimbs;
use ff::PrimeField;

/// Trait for fields that support small-value optimization.
///
/// This trait defines operations for efficient arithmetic when polynomial
/// evaluations fit in native integers. The key optimization is avoiding
/// expensive field operations until absolutely necessary.
///
/// See [`DelayedReduction`](super::DelayedReduction) for the accumulator optimization
/// that builds on this trait, enabling sum-of-products with a single reduction.
///
/// # Type Parameters
/// - `SmallValue`: Native type for witness values (i32, i64, or i128)
///
pub trait SmallValueField<SmallValue>: PrimeField {
  /// Convert SmallValue to field element.
  fn small_to_field(val: SmallValue) -> Self;

  /// Try to convert a field element to SmallValue.
  /// Returns None if the value doesn't fit.
  fn try_field_to_small(val: &Self) -> Option<SmallValue>;
}

impl<F> SmallValueField<i32> for F
where
  F: PrimeField + MontgomeryLimbs,
{
  #[inline]
  fn small_to_field(val: i32) -> Self {
    i64_to_field(val as i64)
  }

  #[inline]
  fn try_field_to_small(val: &Self) -> Option<i32> {
    try_field_to_small_i32(val)
  }
}

impl<F> SmallValueField<i64> for F
where
  F: PrimeField + MontgomeryLimbs,
{
  #[inline]
  fn small_to_field(val: i64) -> Self {
    i64_to_field(val)
  }

  #[inline]
  fn try_field_to_small(val: &Self) -> Option<i64> {
    try_field_to_i64(val)
  }
}

impl<F> SmallValueField<i128> for F
where
  F: PrimeField + MontgomeryLimbs,
{
  #[inline]
  fn small_to_field(val: i128) -> Self {
    i128_to_field(val)
  }

  #[inline]
  fn try_field_to_small(val: &Self) -> Option<i128> {
    try_field_to_i128(val)
  }
}

// ============================================================================
// Helper functions
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
  // 2^64 = u64::MAX + 1 in field arithmetic
  let two_64 = F::from(u64::MAX) + F::ONE;

  if val >= 0 {
    let val_u128 = val as u128;
    let lo = val_u128 as u64;
    let hi = (val_u128 >> 64) as u64;
    F::from(lo) + F::from(hi) * two_64
  } else {
    // Use wrapping_neg to handle i128::MIN correctly
    let mag = val.wrapping_neg() as u128;
    let lo = mag as u64;
    let hi = (mag >> 64) as u64;
    -(F::from(lo) + F::from(hi) * two_64)
  }
}

/// Helper for try_field_to_small: attempts to convert a field element to i32.
///
/// Returns Some(v) if the field element represents a small integer in [-2^31, 2^31-1].
pub fn try_field_to_small_i32<F: PrimeField>(val: &F) -> Option<i32> {
  let repr = val.to_repr();
  let bytes = repr.as_ref();

  // Check if value fits in positive i32
  let high_zero = bytes[4..].iter().all(|&b| b == 0);
  if high_zero {
    let val_u32 = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    if val_u32 <= i32::MAX as u32 {
      return Some(val_u32 as i32);
    }
  }

  // Check if negation fits in i32 (value is negative)
  let neg_val = val.neg();
  let neg_repr = neg_val.to_repr();
  let neg_bytes = neg_repr.as_ref();
  let neg_high_zero = neg_bytes[4..].iter().all(|&b| b == 0);
  if neg_high_zero {
    let neg_u32 = u32::from_le_bytes(neg_bytes[..4].try_into().unwrap());
    if neg_u32 > 0 && neg_u32 <= (i32::MAX as u32) + 1 {
      return Some(-(neg_u32 as i64) as i32);
    }
  }

  None
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

/// Try to convert a field element to i128.
/// Returns None if the value doesn't fit in the i128 range.
#[inline]
pub fn try_field_to_i128<F: PrimeField>(val: &F) -> Option<i128> {
  let repr = val.to_repr();
  let bytes = repr.as_ref();

  // Check if value fits in positive i128 (high bytes all zero)
  let high_zero = bytes[16..].iter().all(|&b| b == 0);
  if high_zero {
    let val_u128 = u128::from_le_bytes(bytes[..16].try_into().unwrap());
    if val_u128 <= i128::MAX as u128 {
      return Some(val_u128 as i128);
    }
  }

  // Check if negation fits in i128 (value is negative)
  let neg_val = val.neg();
  let neg_repr = neg_val.to_repr();
  let neg_bytes = neg_repr.as_ref();
  let neg_high_zero = neg_bytes[16..].iter().all(|&b| b == 0);
  if neg_high_zero {
    let neg_u128 = u128::from_le_bytes(neg_bytes[..16].try_into().unwrap());
    if neg_u128 > 0 && neg_u128 <= (i128::MAX as u128) + 1 {
      // Handle i128::MIN case: -(i128::MAX + 1) = i128::MIN
      return Some(neg_u128.wrapping_neg() as i128);
    }
  }

  None
}

// ============================================================================
// Vector conversion
// ============================================================================

use crate::errors::SpartanError;
use rayon::prelude::*;

/// Convert a vector of field elements to small values.
///
/// Returns Err if any element doesn't fit in the small value range.
pub fn vec_to_small<F, SV>(v: &[F]) -> Result<Vec<SV>, SpartanError>
where
  F: SmallValueField<SV> + Sync,
  SV: Copy + Send + Sync,
{
  v.par_iter()
    .enumerate()
    .map(|(i, f)| {
      F::try_field_to_small(f).ok_or_else(|| SpartanError::SmallValueOverflow {
        value: format!("0x{}", hex::encode(f.to_repr().as_ref())),
        context: format!("vec_to_small at index {}", i),
      })
    })
    .collect()
}

// =============================================================================
// SmallValueField Test Macro
// =============================================================================

/// Generate SmallValueField tests for a field type.
///
/// This macro generates tests for `SmallValueField<i32>`, `SmallValueField<i64>`,
/// and `SmallValueField<i128>` conversions and roundtrips.
///
/// # Example
/// ```ignore
/// crate::test_small_value_field!(pallas_svf, crate::provider::pasta::pallas::Scalar);
/// ```
#[macro_export]
macro_rules! test_small_value_field {
  ($name:ident, $field:ty) => {
    mod $name {
      use $crate::big_num::{
        SmallValueField,
        small_value_field::{i64_to_field, i128_to_field, try_field_to_i64, try_field_to_i128},
      };

      #[test]
      fn small_value_i32_negative() {
        let a: i32 = -5;
        let field_a = <$field as SmallValueField<i32>>::small_to_field(a);
        assert_eq!(field_a, -<$field>::from(5u64));
      }

      #[test]
      fn small_value_i32_boundaries() {
        let min_field = <$field as SmallValueField<i32>>::small_to_field(i32::MIN);
        assert_eq!(min_field, i64_to_field::<$field>(i32::MIN as i64));

        for val in [i32::MIN, i32::MAX, -1, 0, 1] {
          let field = <$field as SmallValueField<i32>>::small_to_field(val);
          let back = <$field as SmallValueField<i32>>::try_field_to_small(&field);
          assert_eq!(back, Some(val));
        }
      }

      #[test]
      fn small_value_i32_roundtrip() {
        assert_eq!(
          <$field as SmallValueField<i32>>::try_field_to_small(&<$field>::from(42u64)),
          Some(42)
        );
        assert_eq!(
          <$field as SmallValueField<i32>>::try_field_to_small(&-<$field>::from(100u64)),
          Some(-100)
        );
        assert_eq!(
          <$field as SmallValueField<i32>>::try_field_to_small(&<$field>::from(u64::MAX)),
          None
        );
      }

      #[test]
      fn i64_to_field_basic() {
        let pos: $field = i64_to_field(100);
        assert_eq!(pos, <$field>::from(100u64));
        let neg: $field = i64_to_field(-50);
        assert_eq!(neg, -<$field>::from(50u64));
      }

      #[test]
      fn i64_roundtrip() {
        for val in [0i64, 1, 100, 1_000_000, i64::MAX / 2, i64::MAX] {
          let field: $field = i64_to_field(val);
          let back = try_field_to_i64(&field).expect("should fit");
          assert_eq!(back, val);
        }
        for val in [
          -1i64,
          -100,
          -1_000_000,
          i64::MIN / 2,
          i64::MIN + 1,
          i64::MIN,
        ] {
          let field: $field = i64_to_field(val);
          let back = try_field_to_i64(&field).expect("should fit");
          assert_eq!(back, val);
        }
      }

      #[test]
      fn i128_roundtrip() {
        for val in [0i128, 1, 100, i64::MAX as i128, i128::MAX / 2] {
          let field: $field = i128_to_field(val);
          let back = try_field_to_i128(&field).expect("should fit");
          assert_eq!(back, val);
        }
        for val in [-1i128, -100, i64::MIN as i128, i128::MIN / 2 + 1] {
          let field: $field = i128_to_field(val);
          let back = try_field_to_i128(&field).expect("should fit");
          assert_eq!(back, val);
        }
      }
    }
  };
}
