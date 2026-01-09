// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Domain types for the Algorithm 6 small-value sum-check optimization.
//!
//! This module defines:
//! - [`LagrangePoint`]: Points in U_d = {∞, 0, 1, ..., d-1}
//! - [`LagrangeHatPoint`]: Points in Û_d = U_d \ {1} (reduced domain)
//! - [`LagrangeIndex`]: Tuples β ∈ U_d^k
//! - [`ValueOneExcluded`]: Error for invalid conversions
//!
//! All types are parameterized by `const D: usize` representing the degree bound.
//! This enables compile-time type safety and debug assertions for bounds checking.

/// A point in the domain U_d = {∞, 0, 1, ..., d-1}
///
/// The domain has d+1 points. The ∞ point represents evaluation of the
/// leading coefficient (see Lemma 2.2 in the paper).
///
/// Type parameter `D` is the degree bound, so valid finite values are 0..D-1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LagrangePoint<const D: usize> {
  /// The point at infinity — represents leading coefficient
  Infinity,
  /// A finite field value 0, 1, ..., D-1
  Finite(usize),
}

impl<const D: usize> LagrangePoint<D> {
  /// Base of the domain U_D (= D + 1 points)
  pub const BASE: usize = D + 1;

  /// Convert to flat index for array access.
  /// Infinity → 0, Finite(v) → v + 1
  #[inline]
  pub fn to_index(self) -> usize {
    match self {
      LagrangePoint::Infinity => 0,
      LagrangePoint::Finite(v) => v + 1,
    }
  }

  /// Convert from flat index.
  /// 0 → Infinity, k → Finite(k - 1)
  ///
  /// # Panics (debug builds only)
  /// Panics if idx > D
  #[inline]
  pub fn from_index(idx: usize) -> Self {
    debug_assert!(
      idx <= D,
      "LagrangePoint::from_index({idx}) out of bounds for D={D}"
    );
    if idx == 0 {
      LagrangePoint::Infinity
    } else {
      LagrangePoint::Finite(idx - 1)
    }
  }

  /// Is this a binary point (0 or 1)?
  #[inline]
  pub fn is_binary(self) -> bool {
    matches!(self, LagrangePoint::Finite(0) | LagrangePoint::Finite(1))
  }

  /// Convert to Û_d point (the reduced domain excluding value 1).
  ///
  /// Returns `None` for Finite(1) since 1 ∉ Û_d.
  #[inline]
  pub fn to_ud_hat(self) -> Option<LagrangeHatPoint<D>> {
    LagrangeHatPoint::try_from(self).ok()
  }
}

/// Test-only helper methods for LagrangePoint.
#[cfg(test)]
impl<const D: usize> LagrangePoint<D> {
  /// Convert to field element. Returns `None` for Infinity.
  #[inline]
  pub fn to_field<F: ff::PrimeField>(self) -> Option<F> {
    match self {
      LagrangePoint::Infinity => None,
      LagrangePoint::Finite(v) => Some(F::from(v as u64)),
    }
  }
}

/// Error returned when trying to convert `Finite(1)` to `LagrangeHatPoint`.
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
///
/// Type parameter `D` is the degree bound (size of Û_d).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LagrangeHatPoint<const D: usize> {
  /// The point at infinity — represents leading coefficient
  Infinity,
  /// A finite field value: 0, 2, 3, ... (never 1)
  Finite(usize),
}

impl<const D: usize> LagrangeHatPoint<D> {
  /// Convert to array index.
  /// Mapping: ∞ → 0, 0 → 1, 2 → 2, 3 → 3, ...
  #[inline]
  pub fn to_index(self) -> usize {
    match self {
      LagrangeHatPoint::Infinity => 0,
      LagrangeHatPoint::Finite(0) => 1,
      LagrangeHatPoint::Finite(k) => k, // 2→2, 3→3, etc.
    }
  }

  /// Convert to LagrangePoint (U_d point)
  #[inline]
  pub fn to_ud_point(self) -> LagrangePoint<D> {
    match self {
      LagrangeHatPoint::Infinity => LagrangePoint::Infinity,
      LagrangeHatPoint::Finite(v) => LagrangePoint::Finite(v),
    }
  }
}

/// Test-only helper methods for LagrangeHatPoint.
#[cfg(test)]
impl<const D: usize> LagrangeHatPoint<D> {
  /// Create a finite point. Returns None for v=1 (not in Û_d).
  pub fn finite(v: usize) -> Option<Self> {
    if v == 1 {
      None
    } else {
      debug_assert!(
        v < D,
        "LagrangeHatPoint::finite({v}) out of bounds for D={D}"
      );
      Some(LagrangeHatPoint::Finite(v))
    }
  }

  /// Create from array index.
  /// Mapping: 0 → ∞, 1 → 0, 2 → 2, 3 → 3, ...
  #[inline]
  pub fn from_index(idx: usize) -> Self {
    debug_assert!(
      idx < D,
      "LagrangeHatPoint::from_index({idx}) out of bounds for D={D}"
    );
    match idx {
      0 => LagrangeHatPoint::Infinity,
      1 => LagrangeHatPoint::Finite(0),
      k => LagrangeHatPoint::Finite(k),
    }
  }

  /// Iterate over all points in Û_d.
  /// Yields: ∞, 0, 2, 3, ..., D-1 (total of D elements)
  pub fn iter() -> impl Iterator<Item = LagrangeHatPoint<D>> {
    (0..D).map(LagrangeHatPoint::from_index)
  }
}

// === Trait Implementations ===

impl<const D: usize> From<LagrangeHatPoint<D>> for LagrangePoint<D> {
  fn from(p: LagrangeHatPoint<D>) -> Self {
    p.to_ud_point()
  }
}

impl<const D: usize> TryFrom<LagrangePoint<D>> for LagrangeHatPoint<D> {
  type Error = ValueOneExcluded;

  fn try_from(p: LagrangePoint<D>) -> Result<Self, Self::Error> {
    match p {
      LagrangePoint::Infinity => Ok(LagrangeHatPoint::Infinity),
      LagrangePoint::Finite(1) => Err(ValueOneExcluded),
      LagrangePoint::Finite(v) => Ok(LagrangeHatPoint::Finite(v)),
    }
  }
}

/// A tuple β ∈ U_d^k — an index into the extended domain.
///
/// Used to index into LagrangeEvaluatedMultilinearPolynomial which stores evaluations over U_d^ℓ₀.
///
/// Type parameter `D` is the degree bound (U_D has D+1 points).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LagrangeIndex<const D: usize>(pub Vec<LagrangePoint<D>>);

impl<const D: usize> LagrangeIndex<D> {
  /// Base of the domain U_D (= D + 1)
  pub const BASE: usize = D + 1;

  /// Number of coordinates
  pub fn len(&self) -> usize {
    self.0.len()
  }

  /// Convert from flat index (mixed-radix decoding)
  ///
  /// Uses compile-time BASE = D + 1
  pub fn from_flat_index(mut idx: usize, len: usize) -> Self {
    let mut points = vec![LagrangePoint::Infinity; len];
    for i in (0..len).rev() {
      points[i] = LagrangePoint::from_index(idx % Self::BASE);
      idx /= Self::BASE;
    }
    LagrangeIndex(points)
  }
}

/// Test-only helper methods for LagrangeIndex.
#[cfg(test)]
impl<const D: usize> LagrangeIndex<D> {
  /// Convert to flat index for array access (mixed-radix encoding)
  ///
  /// Uses compile-time BASE = D + 1
  pub fn to_flat_index(&self) -> usize {
    self
      .0
      .iter()
      .fold(0, |acc, p| acc * Self::BASE + p.to_index())
  }

  /// Check if all coordinates are binary (0 or 1, no ∞)
  pub fn is_all_binary(&self) -> bool {
    self.0.iter().all(|p| p.is_binary())
  }

  /// Check if any coordinate is ∞
  pub fn has_infinity(&self) -> bool {
    self.0.iter().any(|p| matches!(p, LagrangePoint::Infinity))
  }

  /// Create a LagrangeIndex from a binary index in {0,1}^num_bits.
  #[inline]
  pub fn from_binary(bits: usize, num_bits: usize) -> Self {
    let mut points = Vec::with_capacity(num_bits);
    for j in 0..num_bits {
      let bit = (bits >> (num_bits - 1 - j)) & 1;
      points.push(LagrangePoint::Finite(bit));
    }
    LagrangeIndex(points)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;
  use ff::Field;

  type Scalar = pallas::Scalar;

  // === LagrangePoint tests ===

  #[test]
  fn test_ud_point_index_roundtrip() {
    // Test all points for D=4 (indices 0..5)
    for idx in 0..5 {
      let p = LagrangePoint::<4>::from_index(idx);
      assert_eq!(p.to_index(), idx);
    }

    // Test specific values
    assert_eq!(LagrangePoint::<3>::Infinity.to_index(), 0);
    assert_eq!(LagrangePoint::<3>::Finite(0).to_index(), 1);
    assert_eq!(LagrangePoint::<3>::Finite(1).to_index(), 2);
    assert_eq!(LagrangePoint::<3>::Finite(2).to_index(), 3);
  }

  #[test]
  fn test_ud_point_is_binary() {
    assert!(!LagrangePoint::<3>::Infinity.is_binary());
    assert!(LagrangePoint::<3>::Finite(0).is_binary());
    assert!(LagrangePoint::<3>::Finite(1).is_binary());
    assert!(!LagrangePoint::<3>::Finite(2).is_binary());
  }

  #[test]
  fn test_ud_point_to_field() {
    assert_eq!(LagrangePoint::<3>::Infinity.to_field::<Scalar>(), None);
    assert_eq!(
      LagrangePoint::<3>::Finite(0).to_field::<Scalar>(),
      Some(Scalar::ZERO)
    );
    assert_eq!(
      LagrangePoint::<3>::Finite(1).to_field::<Scalar>(),
      Some(Scalar::ONE)
    );
    assert_eq!(
      LagrangePoint::<3>::Finite(2).to_field::<Scalar>(),
      Some(Scalar::from(2u64))
    );
  }

  #[test]
  fn test_ud_point_base_const() {
    assert_eq!(LagrangePoint::<3>::BASE, 4);
    assert_eq!(LagrangePoint::<4>::BASE, 5);
  }

  // === LagrangeHatPoint tests ===

  #[test]
  fn test_ud_hat_index_roundtrip() {
    // Test all points for D=4 (indices 0..4)
    for idx in 0..4 {
      let p = LagrangeHatPoint::<4>::from_index(idx);
      assert_eq!(p.to_index(), idx);
    }
  }

  #[test]
  fn test_ud_hat_index_mapping() {
    // Verify exact mapping: ∞→0, 0→1, 2→2, 3→3
    assert_eq!(LagrangeHatPoint::<4>::Infinity.to_index(), 0);
    assert_eq!(LagrangeHatPoint::<4>::Finite(0).to_index(), 1);
    assert_eq!(LagrangeHatPoint::<4>::Finite(2).to_index(), 2);
    assert_eq!(LagrangeHatPoint::<4>::Finite(3).to_index(), 3);

    // Reverse mapping
    assert_eq!(
      LagrangeHatPoint::<4>::from_index(0),
      LagrangeHatPoint::Infinity
    );
    assert_eq!(
      LagrangeHatPoint::<4>::from_index(1),
      LagrangeHatPoint::Finite(0)
    );
    assert_eq!(
      LagrangeHatPoint::<4>::from_index(2),
      LagrangeHatPoint::Finite(2)
    );
    assert_eq!(
      LagrangeHatPoint::<4>::from_index(3),
      LagrangeHatPoint::Finite(3)
    );
  }

  #[test]
  fn test_ud_hat_iter() {
    // For D=3, Û_d = {∞, 0, 2}
    let points: Vec<_> = LagrangeHatPoint::<3>::iter().collect();
    assert_eq!(points.len(), 3);
    assert_eq!(points[0], LagrangeHatPoint::Infinity);
    assert_eq!(points[1], LagrangeHatPoint::Finite(0));
    assert_eq!(points[2], LagrangeHatPoint::Finite(2));
  }

  #[test]
  fn test_ud_hat_finite_one_rejected() {
    assert!(LagrangeHatPoint::<3>::finite(0).is_some());
    assert!(LagrangeHatPoint::<3>::finite(1).is_none()); // 1 not in Û_d
    assert!(LagrangeHatPoint::<3>::finite(2).is_some());
  }

  // === Conversion tests ===

  #[test]
  fn test_ud_to_ud_hat() {
    assert_eq!(
      LagrangeHatPoint::<3>::try_from(LagrangePoint::<3>::Infinity),
      Ok(LagrangeHatPoint::Infinity)
    );
    assert_eq!(
      LagrangeHatPoint::<3>::try_from(LagrangePoint::<3>::Finite(0)),
      Ok(LagrangeHatPoint::Finite(0))
    );
    assert_eq!(
      LagrangeHatPoint::<3>::try_from(LagrangePoint::<3>::Finite(1)),
      Err(ValueOneExcluded)
    );
    assert_eq!(
      LagrangeHatPoint::<3>::try_from(LagrangePoint::<3>::Finite(2)),
      Ok(LagrangeHatPoint::Finite(2))
    );
  }

  #[test]
  fn test_ud_hat_to_ud() {
    // Via From trait
    assert_eq!(
      LagrangePoint::<3>::from(LagrangeHatPoint::<3>::Infinity),
      LagrangePoint::Infinity
    );
    assert_eq!(
      LagrangePoint::<3>::from(LagrangeHatPoint::<3>::Finite(0)),
      LagrangePoint::Finite(0)
    );
    assert_eq!(
      LagrangePoint::<3>::from(LagrangeHatPoint::<3>::Finite(2)),
      LagrangePoint::Finite(2)
    );

    // Roundtrip for valid points
    let valid_points = [
      LagrangePoint::<3>::Infinity,
      LagrangePoint::<3>::Finite(0),
      LagrangePoint::<3>::Finite(2),
    ];
    for p in valid_points {
      let hat = LagrangeHatPoint::try_from(p).unwrap();
      assert_eq!(LagrangePoint::from(hat), p);
    }
  }

  // === LagrangeIndex tests ===

  #[test]
  fn test_tuple_flat_index_roundtrip() {
    let len: usize = 3;

    // Test all tuples in U_4^3 (D=3, BASE=4)
    for idx in 0..LagrangeIndex::<3>::BASE.pow(len as u32) {
      let tuple = LagrangeIndex::<3>::from_flat_index(idx, len);
      assert_eq!(tuple.to_flat_index(), idx);
      assert_eq!(tuple.len(), len);
    }
  }

  #[test]
  fn test_tuple_base_const() {
    assert_eq!(LagrangeIndex::<3>::BASE, 4);
    assert_eq!(LagrangeIndex::<4>::BASE, 5);
  }

  #[test]
  fn test_tuple_is_all_binary() {
    // [0, 1, 0] - all binary
    let binary = LagrangeIndex::<3>(vec![
      LagrangePoint::Finite(0),
      LagrangePoint::Finite(1),
      LagrangePoint::Finite(0),
    ]);
    assert!(binary.is_all_binary());

    // [0, ∞, 1] - has infinity
    let has_inf = LagrangeIndex::<3>(vec![
      LagrangePoint::Finite(0),
      LagrangePoint::Infinity,
      LagrangePoint::Finite(1),
    ]);
    assert!(!has_inf.is_all_binary());

    // [0, 2, 1] - has non-binary finite
    let has_two = LagrangeIndex::<3>(vec![
      LagrangePoint::Finite(0),
      LagrangePoint::Finite(2),
      LagrangePoint::Finite(1),
    ]);
    assert!(!has_two.is_all_binary());
  }

  #[test]
  fn test_tuple_has_infinity() {
    // [0, ∞, 1] - has infinity
    let has_inf = LagrangeIndex::<3>(vec![
      LagrangePoint::Finite(0),
      LagrangePoint::Infinity,
      LagrangePoint::Finite(1),
    ]);
    assert!(has_inf.has_infinity());

    // [0, 1, 2] - no infinity
    let no_inf = LagrangeIndex::<3>(vec![
      LagrangePoint::Finite(0),
      LagrangePoint::Finite(1),
      LagrangePoint::Finite(2),
    ]);
    assert!(!no_inf.has_infinity());
  }

  #[test]
  fn test_tuple_specific_encoding() {
    // For D=3 (BASE=4), test specific encodings
    // Tuple (∞, 0, 1) = (idx 0, idx 1, idx 2) -> 0*16 + 1*4 + 2 = 6
    let tuple = LagrangeIndex::<3>(vec![
      LagrangePoint::Infinity,
      LagrangePoint::Finite(0),
      LagrangePoint::Finite(1),
    ]);
    assert_eq!(tuple.to_flat_index(), 6);

    // Reverse: 6 -> (0, 1, 2) -> (∞, 0, 1)
    let decoded = LagrangeIndex::<3>::from_flat_index(6, 3);
    assert_eq!(decoded, tuple);
  }
}
