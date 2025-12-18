// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Domain types for the Algorithm 6 small-value sum-check optimization.
//!
//! This module defines:
//! - [`UdPoint`]: Points in U_d = {∞, 0, 1, ..., d-1}
//! - [`UdHatPoint`]: Points in Û_d = U_d \ {1} (reduced domain)
//! - [`UdTuple`]: Tuples β ∈ U_d^k
//! - [`ValueOneExcluded`]: Error for invalid conversions

// Allow dead code until later chunks use these types
#![allow(dead_code)]

use ff::PrimeField;

/// A point in the domain U_d = {∞, 0, 1, ..., d-1}
///
/// The domain has d+1 points. The ∞ point represents evaluation of the
/// leading coefficient (see Lemma 2.2 in the paper).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UdPoint {
  /// The point at infinity — represents leading coefficient
  Infinity,
  /// A finite field value 0, 1, ..., d-1
  Finite(usize),
}

impl UdPoint {
  // === Construction ===

  /// Create the infinity point
  pub const fn infinity() -> Self {
    UdPoint::Infinity
  }

  /// Create the zero point
  pub const fn zero() -> Self {
    UdPoint::Finite(0)
  }

  /// Create the one point
  pub const fn one() -> Self {
    UdPoint::Finite(1)
  }

  /// Create a finite point from value v ∈ {0, 1, ..., d-1}
  pub const fn finite(v: usize) -> Self {
    UdPoint::Finite(v)
  }

  // === Index Conversion ===

  /// Convert to flat index for array access.
  /// Infinity → 0, Finite(v) → v + 1
  #[inline]
  pub fn to_index(self) -> usize {
    match self {
      UdPoint::Infinity => 0,
      UdPoint::Finite(v) => v + 1,
    }
  }

  /// Convert from flat index.
  /// 0 → Infinity, k → Finite(k - 1)
  #[inline]
  pub fn from_index(idx: usize) -> Self {
    if idx == 0 {
      UdPoint::Infinity
    } else {
      UdPoint::Finite(idx - 1)
    }
  }

  // === Properties ===

  /// Check if this is the infinity point
  #[inline]
  pub fn is_infinity(self) -> bool {
    matches!(self, UdPoint::Infinity)
  }

  /// Check if this is the zero point
  #[inline]
  pub fn is_zero(self) -> bool {
    matches!(self, UdPoint::Finite(0))
  }

  /// Is this a binary point (0 or 1)?
  #[inline]
  pub fn is_binary(self) -> bool {
    matches!(self, UdPoint::Finite(0) | UdPoint::Finite(1))
  }

  // === Domain Conversion ===

  /// Convert to field element. Returns `None` for Infinity.
  #[inline]
  pub fn to_field<F: PrimeField>(self) -> Option<F> {
    match self {
      UdPoint::Infinity => None,
      UdPoint::Finite(v) => Some(F::from(v as u64)),
    }
  }

  /// Convert to Û_d point (the reduced domain excluding value 1).
  ///
  /// Returns `None` for Finite(1) since 1 ∉ Û_d.
  #[inline]
  pub fn to_ud_hat(self) -> Option<UdHatPoint> {
    UdHatPoint::try_from(self).ok()
  }
}

/// Error returned when trying to convert `Finite(1)` to `UdHatPoint`.
///
/// The value 1 is excluded from Û_d because s(1) can be recovered
/// from the sum-check constraint s(0) + s(1) = claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueOneExcluded;

impl std::fmt::Display for ValueOneExcluded {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "value 1 is not in Û_d (excluded from reduced domain)")
  }
}

impl std::error::Error for ValueOneExcluded {}

/// A point in the reduced domain Û_d = U_d \ {1} = {∞, 0, 2, 3, ..., d-1}
///
/// This domain has d elements (one less than U_d).
/// Value 1 is excluded because s(1) can be recovered from s(0) + s(1) = claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UdHatPoint {
  /// The point at infinity — represents leading coefficient
  Infinity,
  /// A finite field value: 0, 2, 3, ... (never 1)
  Finite(usize),
}

impl UdHatPoint {
  // === Construction ===

  /// Create the infinity point
  pub const fn infinity() -> Self {
    UdHatPoint::Infinity
  }

  /// Create the zero point
  pub const fn zero() -> Self {
    UdHatPoint::Finite(0)
  }

  /// Create a finite point. Returns None for v=1 (not in Û_d).
  pub fn finite(v: usize) -> Option<Self> {
    if v == 1 {
      None
    } else {
      Some(UdHatPoint::Finite(v))
    }
  }

  // === Index Conversion ===

  /// Convert to array index.
  /// Mapping: ∞ → 0, 0 → 1, 2 → 2, 3 → 3, ...
  #[inline]
  pub fn to_index(self) -> usize {
    match self {
      UdHatPoint::Infinity => 0,
      UdHatPoint::Finite(0) => 1,
      UdHatPoint::Finite(k) => k, // 2→2, 3→3, etc.
    }
  }

  /// Create from array index.
  /// Mapping: 0 → ∞, 1 → 0, 2 → 2, 3 → 3, ...
  #[inline]
  pub fn from_index(idx: usize) -> Self {
    match idx {
      0 => UdHatPoint::Infinity,
      1 => UdHatPoint::Finite(0),
      k => UdHatPoint::Finite(k),
    }
  }

  // === Domain Conversion ===

  /// Convert to UdPoint (U_d point)
  #[inline]
  pub fn to_ud_point(self) -> UdPoint {
    match self {
      UdHatPoint::Infinity => UdPoint::Infinity,
      UdHatPoint::Finite(v) => UdPoint::Finite(v),
    }
  }

  /// Convert to field element. Returns None for Infinity.
  #[inline]
  pub fn to_field<F: PrimeField>(self) -> Option<F> {
    match self {
      UdHatPoint::Infinity => None,
      UdHatPoint::Finite(v) => Some(F::from(v as u64)),
    }
  }

  // === Properties ===

  /// Is this the infinity point?
  #[inline]
  pub fn is_infinity(self) -> bool {
    matches!(self, UdHatPoint::Infinity)
  }

  /// Is this the zero point?
  #[inline]
  pub fn is_zero(self) -> bool {
    matches!(self, UdHatPoint::Finite(0))
  }

  // === Iteration ===

  /// Iterate over all points in Û_d for degree d.
  /// Yields: ∞, 0, 2, 3, ..., d-1 (total of d elements)
  pub fn iter(d: usize) -> impl Iterator<Item = UdHatPoint> {
    (0..d).map(UdHatPoint::from_index)
  }
}

// === Trait Implementations ===

impl From<UdHatPoint> for UdPoint {
  fn from(p: UdHatPoint) -> Self {
    p.to_ud_point()
  }
}

impl TryFrom<UdPoint> for UdHatPoint {
  type Error = ValueOneExcluded;

  fn try_from(p: UdPoint) -> Result<Self, Self::Error> {
    match p {
      UdPoint::Infinity => Ok(UdHatPoint::Infinity),
      UdPoint::Finite(1) => Err(ValueOneExcluded),
      UdPoint::Finite(v) => Ok(UdHatPoint::Finite(v)),
    }
  }
}

/// A tuple β ∈ U_d^k — an index into the extended domain.
///
/// Used to index into LagrangeEvals which stores evaluations over U_d^ℓ₀.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UdTuple(pub Vec<UdPoint>);

impl UdTuple {
  /// Number of coordinates
  pub fn len(&self) -> usize {
    self.0.len()
  }

  /// Check if empty
  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
  }

  /// Check if all coordinates are binary (0 or 1, no ∞)
  ///
  /// Useful for Spartan optimization: binary points yield zero by R1CS relation.
  pub fn is_all_binary(&self) -> bool {
    self.0.iter().all(|p| p.is_binary())
  }

  /// Check if any coordinate is ∞
  ///
  /// Useful for Spartan optimization: Cz term vanishes when any coord is ∞.
  pub fn has_infinity(&self) -> bool {
    self.0.iter().any(|p| matches!(p, UdPoint::Infinity))
  }

  /// Convert to flat index for array access (mixed-radix encoding)
  pub fn to_flat_index(&self, base: usize) -> usize {
    self.0.iter().fold(0, |acc, p| acc * base + p.to_index())
  }

  /// Convert from flat index (mixed-radix decoding)
  pub fn from_flat_index(mut idx: usize, base: usize, len: usize) -> Self {
    let mut points = vec![UdPoint::Infinity; len];
    for i in (0..len).rev() {
      points[i] = UdPoint::from_index(idx % base);
      idx /= base;
    }
    UdTuple(points)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;
  use ff::Field;

  type Scalar = pallas::Scalar;

  // === UdPoint tests ===

  #[test]
  fn test_ud_point_index_roundtrip() {
    // Test all points for d=4 (indices 0..5)
    for idx in 0..5 {
      let p = UdPoint::from_index(idx);
      assert_eq!(p.to_index(), idx);
    }

    // Test specific values
    assert_eq!(UdPoint::Infinity.to_index(), 0);
    assert_eq!(UdPoint::Finite(0).to_index(), 1);
    assert_eq!(UdPoint::Finite(1).to_index(), 2);
    assert_eq!(UdPoint::Finite(2).to_index(), 3);
  }

  #[test]
  fn test_ud_point_is_binary() {
    assert!(!UdPoint::Infinity.is_binary());
    assert!(UdPoint::Finite(0).is_binary());
    assert!(UdPoint::Finite(1).is_binary());
    assert!(!UdPoint::Finite(2).is_binary());
    assert!(!UdPoint::Finite(3).is_binary());
  }

  #[test]
  fn test_ud_point_to_field() {
    assert_eq!(UdPoint::Infinity.to_field::<Scalar>(), None);
    assert_eq!(UdPoint::Finite(0).to_field::<Scalar>(), Some(Scalar::ZERO));
    assert_eq!(UdPoint::Finite(1).to_field::<Scalar>(), Some(Scalar::ONE));
    assert_eq!(
      UdPoint::Finite(2).to_field::<Scalar>(),
      Some(Scalar::from(2u64))
    );
  }

  // === UdHatPoint tests ===

  #[test]
  fn test_ud_hat_index_roundtrip() {
    // Test all points for d=4 (indices 0..4)
    for idx in 0..5 {
      let p = UdHatPoint::from_index(idx);
      assert_eq!(p.to_index(), idx);
    }
  }

  #[test]
  fn test_ud_hat_index_mapping() {
    // Verify exact mapping: ∞→0, 0→1, 2→2, 3→3
    assert_eq!(UdHatPoint::Infinity.to_index(), 0);
    assert_eq!(UdHatPoint::Finite(0).to_index(), 1);
    assert_eq!(UdHatPoint::Finite(2).to_index(), 2);
    assert_eq!(UdHatPoint::Finite(3).to_index(), 3);

    // Reverse mapping
    assert_eq!(UdHatPoint::from_index(0), UdHatPoint::Infinity);
    assert_eq!(UdHatPoint::from_index(1), UdHatPoint::Finite(0));
    assert_eq!(UdHatPoint::from_index(2), UdHatPoint::Finite(2));
    assert_eq!(UdHatPoint::from_index(3), UdHatPoint::Finite(3));
  }

  #[test]
  fn test_ud_hat_iter() {
    // For d=3, Û_d = {∞, 0, 2}
    let points: Vec<_> = UdHatPoint::iter(3).collect();
    assert_eq!(points.len(), 3);
    assert_eq!(points[0], UdHatPoint::Infinity);
    assert_eq!(points[1], UdHatPoint::Finite(0));
    assert_eq!(points[2], UdHatPoint::Finite(2));
  }

  #[test]
  fn test_ud_hat_finite_one_rejected() {
    assert!(UdHatPoint::finite(0).is_some());
    assert!(UdHatPoint::finite(1).is_none()); // 1 not in Û_d
    assert!(UdHatPoint::finite(2).is_some());
    assert!(UdHatPoint::finite(3).is_some());
  }

  // === Conversion tests ===

  #[test]
  fn test_ud_to_ud_hat() {
    assert_eq!(
      UdHatPoint::try_from(UdPoint::Infinity),
      Ok(UdHatPoint::Infinity)
    );
    assert_eq!(
      UdHatPoint::try_from(UdPoint::Finite(0)),
      Ok(UdHatPoint::Finite(0))
    );
    assert_eq!(
      UdHatPoint::try_from(UdPoint::Finite(1)),
      Err(ValueOneExcluded)
    );
    assert_eq!(
      UdHatPoint::try_from(UdPoint::Finite(2)),
      Ok(UdHatPoint::Finite(2))
    );
  }

  #[test]
  fn test_ud_hat_to_ud() {
    // Via From trait
    assert_eq!(UdPoint::from(UdHatPoint::Infinity), UdPoint::Infinity);
    assert_eq!(UdPoint::from(UdHatPoint::Finite(0)), UdPoint::Finite(0));
    assert_eq!(UdPoint::from(UdHatPoint::Finite(2)), UdPoint::Finite(2));

    // Roundtrip for valid points
    let valid_points = [UdPoint::Infinity, UdPoint::Finite(0), UdPoint::Finite(2)];
    for p in valid_points {
      let hat = UdHatPoint::try_from(p).unwrap();
      assert_eq!(UdPoint::from(hat), p);
    }
  }

  // === UdTuple tests ===

  #[test]
  fn test_tuple_flat_index_roundtrip() {
    let base: usize = 4; // d+1 for d=3
    let len: usize = 3;

    // Test all tuples in U_4^3
    for idx in 0..base.pow(len as u32) {
      let tuple = UdTuple::from_flat_index(idx, base, len);
      assert_eq!(tuple.to_flat_index(base), idx);
      assert_eq!(tuple.len(), len);
    }
  }

  #[test]
  fn test_tuple_is_all_binary() {
    // [0, 1, 0] - all binary
    let binary = UdTuple(vec![
      UdPoint::Finite(0),
      UdPoint::Finite(1),
      UdPoint::Finite(0),
    ]);
    assert!(binary.is_all_binary());

    // [0, ∞, 1] - has infinity
    let has_inf = UdTuple(vec![
      UdPoint::Finite(0),
      UdPoint::Infinity,
      UdPoint::Finite(1),
    ]);
    assert!(!has_inf.is_all_binary());

    // [0, 2, 1] - has non-binary finite
    let has_two = UdTuple(vec![
      UdPoint::Finite(0),
      UdPoint::Finite(2),
      UdPoint::Finite(1),
    ]);
    assert!(!has_two.is_all_binary());
  }

  #[test]
  fn test_tuple_has_infinity() {
    // [0, ∞, 1] - has infinity
    let has_inf = UdTuple(vec![
      UdPoint::Finite(0),
      UdPoint::Infinity,
      UdPoint::Finite(1),
    ]);
    assert!(has_inf.has_infinity());

    // [0, 1, 2] - no infinity
    let no_inf = UdTuple(vec![
      UdPoint::Finite(0),
      UdPoint::Finite(1),
      UdPoint::Finite(2),
    ]);
    assert!(!no_inf.has_infinity());
  }

  #[test]
  fn test_tuple_specific_encoding() {
    // For base=4 (d=3), test specific encodings
    // Tuple (∞, 0, 1) = (idx 0, idx 1, idx 2) -> 0*16 + 1*4 + 2 = 6
    let tuple = UdTuple(vec![
      UdPoint::Infinity,
      UdPoint::Finite(0),
      UdPoint::Finite(1),
    ]);
    assert_eq!(tuple.to_flat_index(4), 6);

    // Reverse: 6 -> (0, 1, 2) -> (∞, 0, 1)
    let decoded = UdTuple::from_flat_index(6, 4, 3);
    assert_eq!(decoded, tuple);
  }
}
