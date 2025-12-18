// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2
#![allow(dead_code)]

//! Domain types for the Algorithm 6 small-value sum-check optimization.
//!
//! This module defines:
//! - [`UdPoint`]: Points in U_d = {∞, 0, 1, ..., d-1}
//! - [`UdHatPoint`]: Points in Û_d = U_d \ {1} (reduced domain)
//! - [`UdTuple`]: Tuples β ∈ U_d^k
//! - [`ValueOneExcluded`]: Error for invalid conversions
//!
//! All types are parameterized by `const D: usize` representing the degree bound.
//! This enables compile-time type safety and debug assertions for bounds checking.

use crate::polys::multilinear::MultilinearPolynomial;
use ff::PrimeField;

/// A point in the domain U_d = {∞, 0, 1, ..., d-1}
///
/// The domain has d+1 points. The ∞ point represents evaluation of the
/// leading coefficient (see Lemma 2.2 in the paper).
///
/// Type parameter `D` is the degree bound, so valid finite values are 0..D-1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UdPoint<const D: usize> {
  /// The point at infinity — represents leading coefficient
  Infinity,
  /// A finite field value 0, 1, ..., D-1
  Finite(usize),
}

impl<const D: usize> UdPoint<D> {
  /// Base of the domain U_D (= D + 1 points)
  pub const BASE: usize = D + 1;

  // === Construction ===

  /// Create the infinity point
  #[allow(dead_code)]
  pub const fn infinity() -> Self {
    UdPoint::Infinity
  }

  /// Create the zero point
  #[allow(dead_code)]
  pub const fn zero() -> Self {
    UdPoint::Finite(0)
  }

  /// Create the one point
  #[allow(dead_code)]
  pub const fn one() -> Self {
    UdPoint::Finite(1)
  }

  /// Create a finite point from value v ∈ {0, 1, ..., D-1}
  ///
  /// # Panics (debug builds only)
  /// Panics if v >= D
  #[allow(dead_code)]
  pub fn finite(v: usize) -> Self {
    debug_assert!(v < D, "UdPoint::finite({v}) out of bounds for D={D}");
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
  ///
  /// # Panics (debug builds only)
  /// Panics if idx > D
  #[inline]
  pub fn from_index(idx: usize) -> Self {
    debug_assert!(idx <= D, "UdPoint::from_index({idx}) out of bounds for D={D}");
    if idx == 0 {
      UdPoint::Infinity
    } else {
      UdPoint::Finite(idx - 1)
    }
  }

  // === Properties ===

  /// Check if this is the infinity point
  #[allow(dead_code)]
  #[inline]
  pub fn is_infinity(self) -> bool {
    matches!(self, UdPoint::Infinity)
  }

  /// Check if this is the zero point
  #[allow(dead_code)]
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
  pub fn to_ud_hat(self) -> Option<UdHatPoint<D>> {
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
///
/// Type parameter `D` is the degree bound (size of Û_d).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UdHatPoint<const D: usize> {
  /// The point at infinity — represents leading coefficient
  Infinity,
  /// A finite field value: 0, 2, 3, ... (never 1)
  Finite(usize),
}

impl<const D: usize> UdHatPoint<D> {
  // === Construction ===

  /// Create the infinity point
  #[allow(dead_code)]
  pub const fn infinity() -> Self {
    UdHatPoint::Infinity
  }

  /// Create the zero point
  #[allow(dead_code)]
  pub const fn zero() -> Self {
    UdHatPoint::Finite(0)
  }

  /// Create a finite point. Returns None for v=1 (not in Û_d).
  ///
  /// # Panics (debug builds only)
  /// Panics if v >= D (and v != 1)
  pub fn finite(v: usize) -> Option<Self> {
    if v == 1 {
      None
    } else {
      debug_assert!(v < D, "UdHatPoint::finite({v}) out of bounds for D={D}");
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
  ///
  /// # Panics (debug builds only)
  /// Panics if idx >= D
  #[inline]
  pub fn from_index(idx: usize) -> Self {
    debug_assert!(idx < D, "UdHatPoint::from_index({idx}) out of bounds for D={D}");
    match idx {
      0 => UdHatPoint::Infinity,
      1 => UdHatPoint::Finite(0),
      k => UdHatPoint::Finite(k),
    }
  }

  // === Domain Conversion ===

  /// Convert to UdPoint (U_d point)
  #[inline]
  pub fn to_ud_point(self) -> UdPoint<D> {
    match self {
      UdHatPoint::Infinity => UdPoint::Infinity,
      UdHatPoint::Finite(v) => UdPoint::Finite(v),
    }
  }

  /// Convert to field element. Returns None for Infinity.
  #[allow(dead_code)]
  #[inline]
  pub fn to_field<F: PrimeField>(self) -> Option<F> {
    match self {
      UdHatPoint::Infinity => None,
      UdHatPoint::Finite(v) => Some(F::from(v as u64)),
    }
  }

  // === Properties ===

  /// Is this the infinity point?
  #[allow(dead_code)]
  #[inline]
  pub fn is_infinity(self) -> bool {
    matches!(self, UdHatPoint::Infinity)
  }

  /// Is this the zero point?
  #[allow(dead_code)]
  #[inline]
  pub fn is_zero(self) -> bool {
    matches!(self, UdHatPoint::Finite(0))
  }

  // === Iteration ===

  /// Iterate over all points in Û_d.
  /// Yields: ∞, 0, 2, 3, ..., D-1 (total of D elements)
  pub fn iter() -> impl Iterator<Item = UdHatPoint<D>> {
    (0..D).map(UdHatPoint::from_index)
  }
}

// === Trait Implementations ===

impl<const D: usize> From<UdHatPoint<D>> for UdPoint<D> {
  fn from(p: UdHatPoint<D>) -> Self {
    p.to_ud_point()
  }
}

impl<const D: usize> TryFrom<UdPoint<D>> for UdHatPoint<D> {
  type Error = ValueOneExcluded;

  fn try_from(p: UdPoint<D>) -> Result<Self, Self::Error> {
    match p {
      UdPoint::Infinity => Ok(UdHatPoint::Infinity),
      UdPoint::Finite(1) => Err(ValueOneExcluded),
      UdPoint::Finite(v) => Ok(UdHatPoint::Finite(v)),
    }
  }
}

/// A tuple β ∈ U_d^k — an index into the extended domain.
///
/// Used to index into LagrangeEvaluatedMultilinearPolynomial which stores evaluations over U_d^ℓ₀.
///
/// Type parameter `D` is the degree bound (U_D has D+1 points).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UdTuple<const D: usize>(pub Vec<UdPoint<D>>);

impl<const D: usize> UdTuple<D> {
  /// Base of the domain U_D (= D + 1)
  pub const BASE: usize = D + 1;

  /// Number of coordinates
  pub fn len(&self) -> usize {
    self.0.len()
  }

  /// Check if empty
  #[allow(dead_code)]
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
  ///
  /// Uses compile-time BASE = D + 1
  pub fn to_flat_index(&self) -> usize {
    self.0.iter().fold(0, |acc, p| acc * Self::BASE + p.to_index())
  }

  /// Convert from flat index (mixed-radix decoding)
  ///
  /// Uses compile-time BASE = D + 1
  pub fn from_flat_index(mut idx: usize, len: usize) -> Self {
    let mut points = vec![UdPoint::Infinity; len];
    for i in (0..len).rev() {
      points[i] = UdPoint::from_index(idx % Self::BASE);
      idx /= Self::BASE;
    }
    UdTuple(points)
  }

  /// Create a UdTuple from a binary index in {0,1}^num_bits.
  ///
  /// Each bit maps to a binary U_D point: 0 → Finite(0), 1 → Finite(1).
  /// Bits are read MSB-first (high bit is first coordinate).
  ///
  /// # Example
  /// `from_binary::<3>(0b101, 3)` → `(Finite(1), Finite(0), Finite(1))`
  #[inline]
  pub fn from_binary(bits: usize, num_bits: usize) -> Self {
    let mut points = Vec::with_capacity(num_bits);
    for j in 0..num_bits {
      let bit = (bits >> (num_bits - 1 - j)) & 1;
      points.push(UdPoint::Finite(bit));
    }
    UdTuple(points)
  }
}

// ============================================================================
// Lagrange Extension (Procedure 6)
// ============================================================================

/// Evaluations of a multilinear polynomial over the Lagrange domain U_d^ℓ₀.
///
/// Extended from the boolean hypercube {0,1}^ℓ₀ to U_d^ℓ₀ = {∞, 0, 1, ..., d-1}^ℓ₀,
/// enabling efficient round polynomial computation via Lagrange interpolation.
///
/// Type parameter `D` is the degree bound (U_D has D+1 points).
pub struct LagrangeEvaluatedMultilinearPolynomial<Scalar: PrimeField, const D: usize> {
  evals: Vec<Scalar>, // size (D+1)^num_vars
  num_vars: usize,
}

impl<Scalar: PrimeField, const D: usize> LagrangeEvaluatedMultilinearPolynomial<Scalar, D> {
  /// Base of the extended domain U_D (= D + 1)
  const BASE: usize = D + 1;

  /// Procedure 6: Extend polynomial evaluations from {0,1}^ℓ₀ to U_D^ℓ₀.
  ///
  /// At each step j, we have evaluations over U_D^{j-1} × {0,1}^{ℓ₀-j+1}.
  /// We extend the j-th coordinate from {0,1} to U_D.
  ///
  /// **Key insight:** After extending the first variable, the data layout changes.
  /// We cannot simply split in half for subsequent extensions. Instead, we must
  /// iterate over each (prefix, suffix) pair and extend the middle coordinate.
  ///
  /// # Arguments
  /// * `poly` - Multilinear polynomial with evaluations over boolean hypercube
  pub fn from_multilinear(poly: &MultilinearPolynomial<Scalar>) -> Self {
    let num_vars = poly.Z.len().trailing_zeros() as usize;
    debug_assert_eq!(poly.Z.len(), 1 << num_vars, "Input size must be power of 2");

    let mut current = poly.Z.clone();

    for j in 1..=num_vars {
      // At step j:
      // - prefix_count = (D+1)^{j-1} (number of extended prefix combinations)
      // - suffix_count = 2^{num_vars-j} (number of remaining boolean suffix combinations)
      // - current has size = prefix_count × 2 × suffix_count
      // - next will have size = prefix_count × (D+1) × suffix_count

      let prefix_count = Self::BASE.pow((j - 1) as u32);
      let suffix_count = 1usize << (num_vars - j);
      let current_stride = 2 * suffix_count; // stride between prefixes in current
      let next_stride = Self::BASE * suffix_count; // stride between prefixes in next

      let next_size = prefix_count * next_stride;
      let mut next = vec![Scalar::ZERO; next_size];

      for prefix_idx in 0..prefix_count {
        for suffix_idx in 0..suffix_count {
          // Read p(prefix, 0, suffix) and p(prefix, 1, suffix)
          let base_current = prefix_idx * current_stride;
          let p0 = current[base_current + suffix_idx];
          let p1 = current[base_current + suffix_count + suffix_idx];

          // Extend using Procedure 5: compute p(prefix, γ, suffix) for γ ∈ U_D
          let diff = p1 - p0;
          let base_next = prefix_idx * next_stride;

          // γ = ∞ (index 0): leading coefficient
          next[base_next + suffix_idx] = diff;

          // γ = 0 (index 1): p(prefix, 0, suffix)
          next[base_next + suffix_count + suffix_idx] = p0;

          // γ = 1 (index 2): p(prefix, 1, suffix)
          next[base_next + 2 * suffix_count + suffix_idx] = p1;

          // γ = 2, 3, ..., D-1 (indices 3, 4, ..., D): extrapolate
          for k in 2..D {
            let k_scalar = Scalar::from(k as u64);
            let val = p0 + k_scalar * diff;
            next[base_next + (k + 1) * suffix_count + suffix_idx] = val;
          }
        }
      }

      current = next;
    }

    Self {
      evals: current,
      num_vars,
    }
  }

  /// Number of evaluations
  #[inline]
  pub fn len(&self) -> usize {
    self.evals.len()
  }

  /// Check if empty
  #[allow(dead_code)]
  #[inline]
  pub fn is_empty(&self) -> bool {
    self.evals.is_empty()
  }

  /// Get evaluation by flat index (performance path)
  #[inline]
  pub fn get(&self, idx: usize) -> Scalar {
    self.evals[idx]
  }

  /// Get evaluation by domain tuple (type-safe path)
  #[inline]
  pub fn get_by_domain(&self, tuple: &UdTuple<D>) -> Scalar {
    self.evals[tuple.to_flat_index()]
  }

  /// Convert flat index to domain tuple (for debugging/clarity)
  pub fn to_domain_tuple(&self, flat_idx: usize) -> UdTuple<D> {
    UdTuple::from_flat_index(flat_idx, self.num_vars)
  }

  /// Number of variables
  pub fn num_vars(&self) -> usize {
    self.num_vars
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
    // Test all points for D=4 (indices 0..5)
    for idx in 0..5 {
      let p = UdPoint::<4>::from_index(idx);
      assert_eq!(p.to_index(), idx);
    }

    // Test specific values
    assert_eq!(UdPoint::<3>::Infinity.to_index(), 0);
    assert_eq!(UdPoint::<3>::Finite(0).to_index(), 1);
    assert_eq!(UdPoint::<3>::Finite(1).to_index(), 2);
    assert_eq!(UdPoint::<3>::Finite(2).to_index(), 3);
  }

  #[test]
  fn test_ud_point_is_binary() {
    assert!(!UdPoint::<3>::Infinity.is_binary());
    assert!(UdPoint::<3>::Finite(0).is_binary());
    assert!(UdPoint::<3>::Finite(1).is_binary());
    assert!(!UdPoint::<3>::Finite(2).is_binary());
  }

  #[test]
  fn test_ud_point_to_field() {
    assert_eq!(UdPoint::<3>::Infinity.to_field::<Scalar>(), None);
    assert_eq!(
      UdPoint::<3>::Finite(0).to_field::<Scalar>(),
      Some(Scalar::ZERO)
    );
    assert_eq!(
      UdPoint::<3>::Finite(1).to_field::<Scalar>(),
      Some(Scalar::ONE)
    );
    assert_eq!(
      UdPoint::<3>::Finite(2).to_field::<Scalar>(),
      Some(Scalar::from(2u64))
    );
  }

  #[test]
  fn test_ud_point_base_const() {
    assert_eq!(UdPoint::<3>::BASE, 4);
    assert_eq!(UdPoint::<4>::BASE, 5);
  }

  // === UdHatPoint tests ===

  #[test]
  fn test_ud_hat_index_roundtrip() {
    // Test all points for D=4 (indices 0..4)
    for idx in 0..4 {
      let p = UdHatPoint::<4>::from_index(idx);
      assert_eq!(p.to_index(), idx);
    }
  }

  #[test]
  fn test_ud_hat_index_mapping() {
    // Verify exact mapping: ∞→0, 0→1, 2→2, 3→3
    assert_eq!(UdHatPoint::<4>::Infinity.to_index(), 0);
    assert_eq!(UdHatPoint::<4>::Finite(0).to_index(), 1);
    assert_eq!(UdHatPoint::<4>::Finite(2).to_index(), 2);
    assert_eq!(UdHatPoint::<4>::Finite(3).to_index(), 3);

    // Reverse mapping
    assert_eq!(UdHatPoint::<4>::from_index(0), UdHatPoint::Infinity);
    assert_eq!(UdHatPoint::<4>::from_index(1), UdHatPoint::Finite(0));
    assert_eq!(UdHatPoint::<4>::from_index(2), UdHatPoint::Finite(2));
    assert_eq!(UdHatPoint::<4>::from_index(3), UdHatPoint::Finite(3));
  }

  #[test]
  fn test_ud_hat_iter() {
    // For D=3, Û_d = {∞, 0, 2}
    let points: Vec<_> = UdHatPoint::<3>::iter().collect();
    assert_eq!(points.len(), 3);
    assert_eq!(points[0], UdHatPoint::Infinity);
    assert_eq!(points[1], UdHatPoint::Finite(0));
    assert_eq!(points[2], UdHatPoint::Finite(2));
  }

  #[test]
  fn test_ud_hat_finite_one_rejected() {
    assert!(UdHatPoint::<3>::finite(0).is_some());
    assert!(UdHatPoint::<3>::finite(1).is_none()); // 1 not in Û_d
    assert!(UdHatPoint::<3>::finite(2).is_some());
  }

  // === Conversion tests ===

  #[test]
  fn test_ud_to_ud_hat() {
    assert_eq!(
      UdHatPoint::<3>::try_from(UdPoint::<3>::Infinity),
      Ok(UdHatPoint::Infinity)
    );
    assert_eq!(
      UdHatPoint::<3>::try_from(UdPoint::<3>::Finite(0)),
      Ok(UdHatPoint::Finite(0))
    );
    assert_eq!(
      UdHatPoint::<3>::try_from(UdPoint::<3>::Finite(1)),
      Err(ValueOneExcluded)
    );
    assert_eq!(
      UdHatPoint::<3>::try_from(UdPoint::<3>::Finite(2)),
      Ok(UdHatPoint::Finite(2))
    );
  }

  #[test]
  fn test_ud_hat_to_ud() {
    // Via From trait
    assert_eq!(
      UdPoint::<3>::from(UdHatPoint::<3>::Infinity),
      UdPoint::Infinity
    );
    assert_eq!(
      UdPoint::<3>::from(UdHatPoint::<3>::Finite(0)),
      UdPoint::Finite(0)
    );
    assert_eq!(
      UdPoint::<3>::from(UdHatPoint::<3>::Finite(2)),
      UdPoint::Finite(2)
    );

    // Roundtrip for valid points
    let valid_points = [
      UdPoint::<3>::Infinity,
      UdPoint::<3>::Finite(0),
      UdPoint::<3>::Finite(2),
    ];
    for p in valid_points {
      let hat = UdHatPoint::try_from(p).unwrap();
      assert_eq!(UdPoint::from(hat), p);
    }
  }

  // === UdTuple tests ===

  #[test]
  fn test_tuple_flat_index_roundtrip() {
    let len: usize = 3;

    // Test all tuples in U_4^3 (D=3, BASE=4)
    for idx in 0..UdTuple::<3>::BASE.pow(len as u32) {
      let tuple = UdTuple::<3>::from_flat_index(idx, len);
      assert_eq!(tuple.to_flat_index(), idx);
      assert_eq!(tuple.len(), len);
    }
  }

  #[test]
  fn test_tuple_base_const() {
    assert_eq!(UdTuple::<3>::BASE, 4);
    assert_eq!(UdTuple::<4>::BASE, 5);
  }

  #[test]
  fn test_tuple_is_all_binary() {
    // [0, 1, 0] - all binary
    let binary = UdTuple::<3>(vec![
      UdPoint::Finite(0),
      UdPoint::Finite(1),
      UdPoint::Finite(0),
    ]);
    assert!(binary.is_all_binary());

    // [0, ∞, 1] - has infinity
    let has_inf = UdTuple::<3>(vec![
      UdPoint::Finite(0),
      UdPoint::Infinity,
      UdPoint::Finite(1),
    ]);
    assert!(!has_inf.is_all_binary());

    // [0, 2, 1] - has non-binary finite
    let has_two = UdTuple::<3>(vec![
      UdPoint::Finite(0),
      UdPoint::Finite(2),
      UdPoint::Finite(1),
    ]);
    assert!(!has_two.is_all_binary());
  }

  #[test]
  fn test_tuple_has_infinity() {
    // [0, ∞, 1] - has infinity
    let has_inf = UdTuple::<3>(vec![
      UdPoint::Finite(0),
      UdPoint::Infinity,
      UdPoint::Finite(1),
    ]);
    assert!(has_inf.has_infinity());

    // [0, 1, 2] - no infinity
    let no_inf = UdTuple::<3>(vec![
      UdPoint::Finite(0),
      UdPoint::Finite(1),
      UdPoint::Finite(2),
    ]);
    assert!(!no_inf.has_infinity());
  }

  #[test]
  fn test_tuple_specific_encoding() {
    // For D=3 (BASE=4), test specific encodings
    // Tuple (∞, 0, 1) = (idx 0, idx 1, idx 2) -> 0*16 + 1*4 + 2 = 6
    let tuple = UdTuple::<3>(vec![
      UdPoint::Infinity,
      UdPoint::Finite(0),
      UdPoint::Finite(1),
    ]);
    assert_eq!(tuple.to_flat_index(), 6);

    // Reverse: 6 -> (0, 1, 2) -> (∞, 0, 1)
    let decoded = UdTuple::<3>::from_flat_index(6, 3);
    assert_eq!(decoded, tuple);
  }

  // === extend_to_lagrange_domain tests ===

  #[test]
  fn test_extend_output_size() {
    for num_vars in 1..=4 {
      let input_size = 1 << num_vars;
      let input: Vec<Scalar> = (0..input_size).map(|i| Scalar::from(i as u64)).collect();
      let poly = MultilinearPolynomial::new(input);

      let extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, 3>::from_multilinear(&poly);

      let expected_size = 4usize.pow(num_vars as u32); // (D+1)^num_vars = 4^num_vars
      assert_eq!(extended.len(), expected_size);
      assert_eq!(extended.num_vars(), num_vars);
    }
  }

  #[test]
  fn test_extend_preserves_boolean() {
    use ff::Field;

    let num_vars = 3;
    const D: usize = 3;
    let base = D + 1;

    let input: Vec<Scalar> =
      (0..(1 << num_vars)).map(|_| Scalar::random(&mut rand_core::OsRng)).collect();
    let poly = MultilinearPolynomial::new(input.clone());

    let extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&poly);

    // In U_d indexing: 0 → index 1, 1 → index 2
    for b in 0..(1 << num_vars) {
      let mut ud_idx = 0;
      for j in 0..num_vars {
        let bit = (b >> (num_vars - 1 - j)) & 1;
        let ud_val = bit + 1; // 0→1, 1→2
        ud_idx = ud_idx * base + ud_val;
      }

      assert_eq!(extended.get(ud_idx), input[b]);
    }
  }

  #[test]
  fn test_extend_single_var() {
    let p0 = Scalar::from(7u64);
    let p1 = Scalar::from(19u64);

    let poly = MultilinearPolynomial::new(vec![p0, p1]);
    let extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, 3>::from_multilinear(&poly);

    // U_d = {∞, 0, 1, 2} with indices 0, 1, 2, 3
    assert_eq!(extended.get(0), p1 - p0, "p(∞) = leading coeff");
    assert_eq!(extended.get(1), p0, "p(0)");
    assert_eq!(extended.get(2), p1, "p(1)");
    assert_eq!(extended.get(3), p1.double() - p0, "p(2) = 2*p1 - p0");
  }

  #[test]
  fn test_extend_matches_direct() {
    use ff::Field;

    let num_vars = 3;
    const D: usize = 3;
    let base = D + 1;

    let input: Vec<Scalar> =
      (0..(1 << num_vars)).map(|_| Scalar::random(&mut rand_core::OsRng)).collect();
    let poly = MultilinearPolynomial::new(input.clone());
    let extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&poly);

    // Check all finite points via direct multilinear evaluation
    for idx in 0..extended.len() {
      let tuple = index_to_tuple(idx, base, num_vars);

      // Skip infinity points (index 0 in any coordinate)
      if tuple.iter().any(|&t| t == 0) {
        continue;
      }

      // Convert U_d indices to field values: index k → value k-1
      let point: Vec<Scalar> = tuple.iter().map(|&t| Scalar::from((t - 1) as u64)).collect();

      let direct = evaluate_multilinear(&input, &point);
      assert_eq!(extended.get(idx), direct);
    }
  }

  #[test]
  fn test_extend_infinity_leading_coeff() {
    use ff::Field;

    let num_vars = 3;
    const D: usize = 3;
    let base = D + 1;

    let input: Vec<Scalar> =
      (0..(1 << num_vars)).map(|_| Scalar::random(&mut rand_core::OsRng)).collect();
    let poly = MultilinearPolynomial::new(input.clone());
    let extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&poly);

    // p(∞, y₂, y₃) = p(1, y₂, y₃) - p(0, y₂, y₃)
    for y2 in 0..2usize {
      for y3 in 0..2usize {
        let idx_0 = (0 << 2) | (y2 << 1) | y3; // p(0, y2, y3)
        let idx_1 = (1 << 2) | (y2 << 1) | y3; // p(1, y2, y3)

        let expected = input[idx_1] - input[idx_0];
        let ext_idx = 0 * base * base + (y2 + 1) * base + (y3 + 1);

        assert_eq!(extended.get(ext_idx), expected);
      }
    }
  }

  #[test]
  fn test_extend_known_polynomial() {
    // p(X, Y, Z) = X + 2Y + 4Z
    const D: usize = 3;
    let base = D + 1;

    let mut input = Vec::with_capacity(8);
    for x in 0..2u64 {
      for y in 0..2u64 {
        for z in 0..2u64 {
          input.push(Scalar::from(x + 2 * y + 4 * z));
        }
      }
    }
    let poly = MultilinearPolynomial::new(input);

    let extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&poly);

    // Finite points: p(a,b,c) = a + 2b + 4c
    for a in 0..D {
      for b in 0..D {
        for c in 0..D {
          let idx = (a + 1) * base * base + (b + 1) * base + (c + 1);
          let expected = Scalar::from(a as u64 + 2 * b as u64 + 4 * c as u64);
          assert_eq!(extended.get(idx), expected);
        }
      }
    }

    // Infinity points = variable coefficients
    assert_eq!(
      extended.get(0 * base * base + 1 * base + 1),
      Scalar::ONE,
      "p(∞,0,0) = coeff of X"
    );
    assert_eq!(
      extended.get(1 * base * base + 0 * base + 1),
      Scalar::from(2u64),
      "p(0,∞,0) = coeff of Y"
    );
    assert_eq!(
      extended.get(1 * base * base + 1 * base + 0),
      Scalar::from(4u64),
      "p(0,0,∞) = coeff of Z"
    );
    assert_eq!(
      extended.get(0),
      Scalar::ZERO,
      "p(∞,∞,∞) = 0 (no XYZ term)"
    );
  }

  #[test]
  fn test_get_by_domain() {
    let p0 = Scalar::from(7u64);
    let p1 = Scalar::from(19u64);

    let poly = MultilinearPolynomial::new(vec![p0, p1]);
    let extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, 3>::from_multilinear(&poly);

    // Test type-safe access
    let tuple_inf = UdTuple::<3>(vec![UdPoint::Infinity]);
    let tuple_zero = UdTuple::<3>(vec![UdPoint::Finite(0)]);
    let tuple_one = UdTuple::<3>(vec![UdPoint::Finite(1)]);

    assert_eq!(extended.get_by_domain(&tuple_inf), p1 - p0);
    assert_eq!(extended.get_by_domain(&tuple_zero), p0);
    assert_eq!(extended.get_by_domain(&tuple_one), p1);
  }

  // === Test helpers ===

  fn index_to_tuple(mut idx: usize, base: usize, len: usize) -> Vec<usize> {
    let mut tuple = vec![0; len];
    for i in (0..len).rev() {
      tuple[i] = idx % base;
      idx /= base;
    }
    tuple
  }

  /// Direct multilinear evaluation: p(r) = Σ_x p(x) · eq(x, r)
  fn evaluate_multilinear(evals: &[Scalar], point: &[Scalar]) -> Scalar {
    let l = point.len();
    let mut result = Scalar::ZERO;
    for (i, &val) in evals.iter().enumerate() {
      let mut eq_term = Scalar::ONE;
      for j in 0..l {
        let bit = (i >> (l - 1 - j)) & 1;
        if bit == 1 {
          eq_term *= point[j];
        } else {
          eq_term *= Scalar::ONE - point[j];
        }
      }
      result += val * eq_term;
    }
    result
  }
}
