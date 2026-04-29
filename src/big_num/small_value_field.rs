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

use super::{
  DelayedReduction, WideMul, field_reduction_constants::FieldReductionConstants,
  montgomery::MontgomeryLimbs,
};
use ff::PrimeField;
use num_traits::{Bounded, One, Signed, Zero};
use std::ops::{Add, Div, Mul, Sub};

// =============================================================================
// SmallValue trait
// =============================================================================

/// Small integer type usable in small-value sumcheck.
///
/// Bundles all required bounds for Lagrange extension and accumulation:
/// - `WideMul`: widening multiplication for product computation
/// - `Copy + Default + Zero`: basic value semantics
/// - `Add + Sub`: arithmetic for Lagrange extension
/// - `Send + Sync`: thread safety for parallel processing
pub trait SmallValue:
  WideMul + Copy + Default + Zero + Add<Output = Self> + Sub<Output = Self> + Send + Sync
{
}

impl SmallValue for i32 {}
impl SmallValue for i64 {}

// =============================================================================
// SmallValueEngine trait
// =============================================================================

/// Field that supports small-value sumcheck with value type `SV`.
///
/// Bundles all field requirements for small-value optimization:
/// - `SmallValueField<SV>`: conversion between field and small values
/// - `DelayedReduction<SV>`: accumulate field × small products
/// - `DelayedReduction<SV::Product>`: accumulate field × wide products
/// - `DelayedReduction<Self>`: accumulate field × field products
pub trait SmallValueEngine<SV: SmallValue>:
  PrimeField
  + SmallValueField<SV>
  + DelayedReduction<SV>
  + DelayedReduction<SV::Product>
  + DelayedReduction<Self>
  + Send
  + Sync
{
}

// Blanket implementation
impl<F, SV> SmallValueEngine<SV> for F
where
  SV: SmallValue,
  F: PrimeField
    + SmallValueField<SV>
    + DelayedReduction<SV>
    + DelayedReduction<SV::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
{
}

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

/// Maximum absolute value for "small" field elements stored as i64.
///
/// Chosen so that all i128 arithmetic in small-value sumcheck consumers remains
/// overflow-free.
const SMALL_VALUE_MAX: u64 = (1u64 << 62) - 1;

const SMALL_VALUE_MIN_I64: i64 = -(SMALL_VALUE_MAX as i64);

#[derive(Clone, Copy)]
enum SignedMagnitude {
  Positive(u128),
  Negative(u128),
}

#[inline]
fn high_bytes_are_zero(bytes: &[u8], width_bytes: usize) -> bool {
  bytes[width_bytes..].iter().all(|&b| b == 0)
}

#[inline]
fn lower_bytes_to_u128(bytes: &[u8], width_bytes: usize) -> u128 {
  let mut buf = [0u8; 16];
  buf[..width_bytes].copy_from_slice(&bytes[..width_bytes]);
  u128::from_le_bytes(buf)
}

#[inline]
fn try_field_to_signed_magnitude<F: PrimeField>(
  val: &F,
  width_bytes: usize,
) -> Option<SignedMagnitude> {
  let repr = val.to_repr();
  let bytes = repr.as_ref();

  if high_bytes_are_zero(bytes, width_bytes) {
    return Some(SignedMagnitude::Positive(lower_bytes_to_u128(
      bytes,
      width_bytes,
    )));
  }

  let neg_repr = val.neg().to_repr();
  let neg_bytes = neg_repr.as_ref();
  if high_bytes_are_zero(neg_bytes, width_bytes) {
    let mag = lower_bytes_to_u128(neg_bytes, width_bytes);
    if mag > 0 {
      return Some(SignedMagnitude::Negative(mag));
    }
  }

  None
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
  match try_field_to_signed_magnitude(val, 4)? {
    SignedMagnitude::Positive(mag) if mag <= i32::MAX as u128 => Some(mag as i32),
    SignedMagnitude::Negative(mag) if mag <= (i32::MAX as u128) + 1 => Some(-(mag as i64) as i32),
    _ => None,
  }
}

/// Try to convert a field element to i64.
/// Returns None if the value doesn't fit in the i64 range.
#[inline]
pub fn try_field_to_i64<F: PrimeField>(val: &F) -> Option<i64> {
  match try_field_to_signed_magnitude(val, 8)? {
    SignedMagnitude::Positive(mag) if mag <= i64::MAX as u128 => Some(mag as i64),
    SignedMagnitude::Negative(mag) if mag <= (i64::MAX as u128) + 1 => Some(-(mag as i128) as i64),
    _ => None,
  }
}

/// Try to convert a field element to i128.
/// Returns None if the value doesn't fit in the i128 range.
#[inline]
pub fn try_field_to_i128<F: PrimeField>(val: &F) -> Option<i128> {
  match try_field_to_signed_magnitude(val, 16)? {
    SignedMagnitude::Positive(mag) if mag <= i128::MAX as u128 => Some(mag as i128),
    SignedMagnitude::Negative(mag) if mag <= (i128::MAX as u128) + 1 => {
      Some(mag.wrapping_neg() as i128)
    }
    _ => None,
  }
}

// ============================================================================
// Vector conversion
// ============================================================================

use crate::errors::SpartanError;
use rayon::prelude::*;

/// Convert field elements to i64 values, storing 0 for values outside the
/// small-value range and recording those positions for field correction.
#[inline(never)]
pub(crate) fn to_small_vec_or_zero<F: PrimeField + FieldReductionConstants>(
  poly: &[F],
) -> (Vec<i64>, Vec<usize>) {
  let mut result = Vec::with_capacity(poly.len());
  let mut large_positions = Vec::new();

  for (idx, f) in poly.iter().enumerate() {
    match try_field_to_i64(f) {
      Some(val) if (SMALL_VALUE_MIN_I64..=SMALL_VALUE_MAX as i64).contains(&val) => {
        result.push(val);
      }
      _ => {
        result.push(0);
        large_positions.push(idx);
      }
    }
  }

  (result, large_positions)
}

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

// ============================================================================
// Extension bounds
// ============================================================================

/// Precomputed bound for values that will be extended from `{0,1}` to a
/// degree-`D` Lagrange domain for `lb` rounds.
///
/// The extension can grow native magnitudes by at most `(D + 1)^lb`, so full-small
/// accumulator mode only accepts values whose absolute value survives that growth.
pub struct ExtensionBound<SV: WideMul, const D: usize> {
  max_safe: SV::Product,
}

impl<SV, const D: usize> ExtensionBound<SV, D>
where
  SV: WideMul + Bounded + Copy + Into<SV::Product>,
  SV::Product:
    Copy + Ord + Signed + Div<Output = SV::Product> + Mul<Output = SV::Product> + One + From<i32>,
{
  pub fn new(lb: usize) -> Self {
    let base: SV::Product = (D as i32 + 1).into();
    let mut power = SV::Product::one();
    for _ in 0..lb {
      power = power * base;
    }
    let max_safe = SV::max_value().into() / power;
    Self { max_safe }
  }

  #[inline]
  pub fn is_safe(&self, small: SV) -> bool {
    let abs_small: SV::Product = small.into();
    abs_small.abs() <= self.max_safe
  }

  #[inline]
  pub fn try_to_small<F>(&self, val: &F) -> Option<SV>
  where
    F: SmallValueField<SV>,
  {
    let small = F::try_field_to_small(val)?;
    self.is_safe(small).then_some(small)
  }
}

/// Convert field elements to small values and enforce the extension-safe bound.
pub fn vec_to_small_for_extension<F, SV, const D: usize>(
  v: &[F],
  lb: usize,
) -> Result<Vec<SV>, SpartanError>
where
  F: SmallValueField<SV> + Sync,
  SV: WideMul + Bounded + Copy + Send + Sync + Into<SV::Product>,
  SV::Product:
    Copy + Ord + Signed + Div<Output = SV::Product> + Mul<Output = SV::Product> + One + From<i32>,
{
  let bound = ExtensionBound::<SV, D>::new(lb);

  v.par_iter()
    .enumerate()
    .map(|(i, f)| {
      bound
        .try_to_small(f)
        .ok_or_else(|| SpartanError::SmallValueOverflow {
          value: format!("0x{}", hex::encode(f.to_repr().as_ref())),
          context: format!(
            "vec_to_small_for_extension: value at index {} exceeds bound for D={}, lb={}",
            i, D, lb
          ),
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
      fn small_vec_or_zero() {
        $crate::big_num::small_value_field::tests::test_small_vec_or_zero_impl::<$field>();
      }

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

#[cfg(test)]
pub(crate) mod tests {
  use super::*;

  /// Test to_small_vec_or_zero with mixed small and large values.
  pub(crate) fn test_small_vec_or_zero_impl<F: PrimeField + FieldReductionConstants + Copy>() {
    use rand::{SeedableRng, rngs::StdRng};

    let mut rng = StdRng::seed_from_u64(11111);

    let vals: Vec<F> = (0..10).map(|i| F::from(i as u64)).collect();
    let (small, large) = to_small_vec_or_zero(&vals);
    assert!(large.is_empty());
    for (i, &v) in small.iter().enumerate() {
      assert_eq!(v, i as i64);
    }

    let mixed = vec![
      F::from(5u64),
      F::random(&mut rng),
      -F::from(3u64),
      F::random(&mut rng),
      F::from(100u64),
    ];
    let (small, large) = to_small_vec_or_zero(&mixed);
    assert_eq!(small.len(), 5);
    assert_eq!(small[0], 5);
    assert_eq!(small[1], 0);
    assert_eq!(small[2], -3);
    assert_eq!(small[3], 0);
    assert_eq!(small[4], 100);
    assert_eq!(large, vec![1, 3]);

    let boundary = vec![F::from(SMALL_VALUE_MAX), -F::from(SMALL_VALUE_MAX)];
    let (small, large) = to_small_vec_or_zero(&boundary);
    assert!(large.is_empty(), "values at threshold should be small");
    assert_eq!(small[0], SMALL_VALUE_MAX as i64);
    assert_eq!(small[1], -(SMALL_VALUE_MAX as i64));

    let above = vec![F::from(SMALL_VALUE_MAX + 1)];
    let (small, large) = to_small_vec_or_zero(&above);
    assert_eq!(large, vec![0], "values above threshold should be large");
    assert_eq!(small[0], 0);
  }
}
