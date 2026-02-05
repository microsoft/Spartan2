// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallValueField trait for small-value optimization.

use super::montgomery::MontgomeryLimbs;
use ff::PrimeField;
use halo2curves::{bn256::Fr as Bn254Fr, t256::Fq as T256Fq};

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
/// - `SmallValue`: Native type for witness values (i32 or i64)
///
/// # Implementations
/// - `Fp: SmallValueField<i32>`
/// - `Fp: SmallValueField<i64>`
pub trait SmallValueField<SmallValue>: PrimeField {
  /// Convert SmallValue to field element.
  fn small_to_field(val: SmallValue) -> Self;

  /// Try to convert a field element to SmallValue.
  /// Returns None if the value doesn't fit.
  fn try_field_to_small(val: &Self) -> Option<SmallValue>;
}

// ============================================================================
// Marker traits for blanket implementations
// ============================================================================

/// Marker trait: field supports `SmallValueField<i32>` via blanket impl.
pub(crate) trait SupportsSmallI32: MontgomeryLimbs {}

/// Marker trait: field supports `SmallValueField<i64>` via blanket impl.
pub(crate) trait SupportsSmallI64: MontgomeryLimbs {}

// Marker trait implementations
impl SupportsSmallI32 for halo2curves::pasta::Fp {}
impl SupportsSmallI32 for halo2curves::pasta::Fq {}
impl SupportsSmallI64 for halo2curves::pasta::Fp {}
impl SupportsSmallI64 for halo2curves::pasta::Fq {}
impl SupportsSmallI64 for Bn254Fr {}
impl SupportsSmallI32 for T256Fq {}
impl SupportsSmallI64 for T256Fq {}

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

/// Helper for try_field_to_small: attempts to convert a field element to i32.
///
/// Returns Some(v) if the field element represents a small integer in [-2^31, 2^31-1].
fn try_field_to_small_i32<F: PrimeField>(val: &F) -> Option<i32> {
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

// ============================================================================
// Blanket SmallValueField<i32> for all SupportsSmallI32 fields
// ============================================================================

impl<F: SupportsSmallI32 + PrimeField> SmallValueField<i32> for F {
  #[inline]
  fn small_to_field(val: i32) -> Self {
    if val >= 0 {
      Self::from(val as u64)
    } else {
      -Self::from((-val) as u64)
    }
  }

  fn try_field_to_small(val: &Self) -> Option<i32> {
    try_field_to_small_i32(val)
  }
}

// ============================================================================
// Blanket SmallValueField<i64> for all SupportsSmallI64 fields
// ============================================================================

impl<F: SupportsSmallI64 + PrimeField> SmallValueField<i64> for F {
  #[inline]
  fn small_to_field(val: i64) -> Self {
    i64_to_field(val)
  }

  fn try_field_to_small(val: &Self) -> Option<i64> {
    try_field_to_i64(val)
  }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;

  type Scalar = pallas::Scalar;

  #[test]
  fn test_small_value_field_arithmetic() {
    let a: i32 = 10;
    let b: i32 = 3;

    assert_eq!(a + b, 13);
    assert_eq!(a - b, 7);
    assert_eq!(-a, -10);
    assert_eq!(a * 5, 50);
  }

  #[test]
  fn test_small_value_field_negative() {
    let a: i32 = -5;
    let b: i32 = 3;

    assert_eq!(a + b, -2);
    assert_eq!(a - b, -8);

    let field_a = <Scalar as SmallValueField<i32>>::small_to_field(a);
    assert_eq!(field_a, -Scalar::from(5u64));
  }

  #[test]
  fn test_i64_to_field() {
    let pos: Scalar = i64_to_field(100);
    assert_eq!(pos, Scalar::from(100u64));

    let neg: Scalar = i64_to_field(-50);
    assert_eq!(neg, -Scalar::from(50u64));
  }

  #[test]
  fn test_try_field_to_i64_roundtrip() {
    // Test positive values
    for val in [0i64, 1, 100, 1_000_000, i64::MAX / 2, i64::MAX] {
      let field: Scalar = i64_to_field(val);
      let back = try_field_to_i64(&field).expect("should fit");
      assert_eq!(back, val, "roundtrip failed for {}", val);
    }

    // Test negative values
    for val in [-1i64, -100, -1_000_000, i64::MIN / 2, i64::MIN + 1] {
      let field: Scalar = i64_to_field(val);
      let back = try_field_to_i64(&field).expect("should fit");
      assert_eq!(back, val, "roundtrip failed for {}", val);
    }

    // Test i64::MIN separately (edge case)
    let field: Scalar = i64_to_field(i64::MIN);
    let back = try_field_to_i64(&field).expect("should fit");
    assert_eq!(back, i64::MIN, "roundtrip failed for i64::MIN");

    // Test values that don't fit in i64
    let too_large = Scalar::from(u64::MAX) + Scalar::from(1u64);
    assert!(try_field_to_i64(&too_large).is_none());
  }

  #[test]
  fn test_try_field_to_small_roundtrip() {
    assert_eq!(
      <Scalar as SmallValueField<i32>>::try_field_to_small(&Scalar::from(42u64)),
      Some(42)
    );
    assert_eq!(
      <Scalar as SmallValueField<i32>>::try_field_to_small(&-Scalar::from(100u64)),
      Some(-100)
    );
    assert_eq!(
      <Scalar as SmallValueField<i32>>::try_field_to_small(&Scalar::from(u64::MAX)),
      None
    );
  }

  #[test]
  fn test_fp_small_value_field() {
    use halo2curves::pasta::Fp;

    let a: i32 = 42;
    let b: i32 = -10;

    assert_eq!(a + b, 32);
    assert_eq!(
      <Fp as SmallValueField<i32>>::small_to_field(a),
      Fp::from(42u64)
    );
  }
}
