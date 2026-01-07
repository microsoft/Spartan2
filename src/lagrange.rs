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
//!
//! All types are parameterized by `const D: usize` representing the degree bound.
//! This enables compile-time type safety and debug assertions for bounds checking.

use ff::PrimeField;
use std::ops::{Add, Sub};

#[cfg(test)]
use crate::{polys::multilinear::MultilinearPolynomial, small_field::SmallValueField};

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
    debug_assert!(
      idx <= D,
      "UdPoint::from_index({idx}) out of bounds for D={D}"
    );
    if idx == 0 {
      UdPoint::Infinity
    } else {
      UdPoint::Finite(idx - 1)
    }
  }

  /// Is this a binary point (0 or 1)?
  #[inline]
  pub fn is_binary(self) -> bool {
    matches!(self, UdPoint::Finite(0) | UdPoint::Finite(1))
  }

  /// Convert to Û_d point (the reduced domain excluding value 1).
  ///
  /// Returns `None` for Finite(1) since 1 ∉ Û_d.
  #[inline]
  pub fn to_ud_hat(self) -> Option<UdHatPoint<D>> {
    UdHatPoint::try_from(self).ok()
  }
}

/// Test-only helper methods for UdPoint.
#[cfg(test)]
impl<const D: usize> UdPoint<D> {
  /// Convert to field element. Returns `None` for Infinity.
  #[inline]
  pub fn to_field<F: PrimeField>(self) -> Option<F> {
    match self {
      UdPoint::Infinity => None,
      UdPoint::Finite(v) => Some(F::from(v as u64)),
    }
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

  /// Convert to UdPoint (U_d point)
  #[inline]
  pub fn to_ud_point(self) -> UdPoint<D> {
    match self {
      UdHatPoint::Infinity => UdPoint::Infinity,
      UdHatPoint::Finite(v) => UdPoint::Finite(v),
    }
  }
}

/// Test-only helper methods for UdHatPoint.
#[cfg(test)]
impl<const D: usize> UdHatPoint<D> {
  /// Create a finite point. Returns None for v=1 (not in Û_d).
  pub fn finite(v: usize) -> Option<Self> {
    if v == 1 {
      None
    } else {
      debug_assert!(v < D, "UdHatPoint::finite({v}) out of bounds for D={D}");
      Some(UdHatPoint::Finite(v))
    }
  }

  /// Create from array index.
  /// Mapping: 0 → ∞, 1 → 0, 2 → 2, 3 → 3, ...
  #[inline]
  pub fn from_index(idx: usize) -> Self {
    debug_assert!(
      idx < D,
      "UdHatPoint::from_index({idx}) out of bounds for D={D}"
    );
    match idx {
      0 => UdHatPoint::Infinity,
      1 => UdHatPoint::Finite(0),
      k => UdHatPoint::Finite(k),
    }
  }

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
}

/// Test-only helper methods for UdTuple.
#[cfg(test)]
impl<const D: usize> UdTuple<D> {
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
    self.0.iter().any(|p| matches!(p, UdPoint::Infinity))
  }

  /// Create a UdTuple from a binary index in {0,1}^num_bits.
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

// ========================================================================
// Lagrange basis for U_d = {∞, 0, 1, ..., d-1}
// ========================================================================

/// Precomputes data for the 1-D Lagrange basis on U_d = {∞, 0, 1, ..., d-1}.
///
/// `finite_points[k]` is the field element for UdPoint::Finite(k).
/// `weights[k]` is the barycentric weight:
///   w_k = 1 / ∏_{m≠k} (x_k - x_m)
///
/// With these weights, basis evaluation at any r costs O(D) multiplies
/// and uses no per-round inversions.
pub struct LagrangeBasisFactory<F, const D: usize> {
  finite_points: [F; D],
  weights: [F; D],
}

/// Evaluated Lagrange basis at a single r, stored in UdPoint order.
type LagrangeBasisEval<F, const D: usize> = UdEvaluations<F, D>;

// ========================================================================
// Evaluation containers for U_d and Û_d domains
// ========================================================================

/// Evaluations at all D+1 points of U_d = {∞, 0, 1, ..., D-1}.
///
/// This type stores values indexed by [`UdPoint<D>`], with the infinity
/// point stored separately from the D finite points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UdEvaluations<T, const D: usize> {
  /// Value at the infinity point
  pub infinity: T,
  /// Values at finite points 0, 1, ..., D-1
  pub finite: [T; D],
}

impl<T: Copy, const D: usize> UdEvaluations<T, D> {
  /// Create new evaluations from infinity and finite values.
  #[inline]
  pub fn new(infinity: T, finite: [T; D]) -> Self {
    Self { infinity, finite }
  }

  /// Get value at infinity.
  #[inline]
  pub fn at_infinity(&self) -> T {
    self.infinity
  }

  /// Get value at zero (finite point 0).
  #[inline]
  pub fn at_zero(&self) -> T {
    self.finite[0]
  }

  /// Get value at one (finite point 1).
  ///
  /// # Panics (debug builds only)
  /// Panics if D < 2.
  #[inline]
  pub fn at_one(&self) -> T {
    debug_assert!(D >= 2, "at_one() requires D >= 2");
    self.finite[1]
  }

  /// Iterate values in U_d order: [∞, 0, 1, ..., D-1].
  pub fn iter_ud_order(&self) -> impl Iterator<Item = T> + '_ {
    std::iter::once(self.infinity).chain(self.finite.iter().copied())
  }
}

/// Test-only helper methods for UdEvaluations.
#[cfg(test)]
impl<T: Copy, const D: usize> UdEvaluations<T, D> {
  /// Get value at a domain point.
  #[inline]
  pub fn get(&self, p: UdPoint<D>) -> T {
    match p {
      UdPoint::Infinity => self.infinity,
      UdPoint::Finite(k) => self.finite[k],
    }
  }
}

impl<F: PrimeField> UdEvaluations<F, 2> {
  /// Evaluate linear polynomial at u: L(u) = infinity * u + finite[0].
  ///
  /// For evaluations of a degree-1 polynomial over U_2 = {∞, 0, 1},
  /// this computes L(u) = l_∞ · u + l_0.
  #[inline]
  pub fn eval_linear_at(&self, u: F) -> F {
    self.infinity * u + self.finite[0]
  }
}

/// Evaluations at all D points of Û_d = U_d \ {1} = {∞, 0, 2, ..., D-1}.
///
/// This reduced domain excludes point 1 because s(1) can be recovered
/// from the sum-check constraint s(0) + s(1) = claim.
///
/// Indexing follows [`UdHatPoint::to_index()`]: ∞→0, 0→1, 2→2, 3→3, ...
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UdHatEvaluations<T, const D: usize> {
  data: [T; D],
}

impl<T: Copy, const D: usize> UdHatEvaluations<T, D> {
  /// Create from array indexed by `UdHatPoint::to_index()`.
  #[inline]
  pub fn from_array(data: [T; D]) -> Self {
    Self { data }
  }

  /// Get value at infinity (index 0).
  #[inline]
  pub fn at_infinity(&self) -> T {
    self.data[0]
  }

  /// Get value at zero (index 1).
  #[inline]
  pub fn at_zero(&self) -> T {
    self.data[1]
  }
}

/// R_i tensor coefficients used in Algorithm 6.
///
/// Indexing matches UdTuple::to_flat_index() over U_d^{i-1}.
pub struct LagrangeCoeff<F, const D: usize> {
  coeffs: Vec<F>,
}

impl<F: PrimeField, const D: usize> LagrangeCoeff<F, D> {
  /// Initialize R_1 = [1].
  pub fn new() -> Self {
    Self {
      coeffs: vec![F::ONE],
    }
  }

  pub fn len(&self) -> usize {
    self.coeffs.len()
  }

  pub fn as_slice(&self) -> &[F] {
    &self.coeffs
  }

  /// Extend: R_{i+1} = R_i ⊗ L(r_i).
  pub fn extend(&mut self, basis: &LagrangeBasisEval<F, D>) {
    let base = D + 1;
    let mut next = vec![F::ZERO; self.coeffs.len() * base];
    for (i, &c) in self.coeffs.iter().enumerate() {
      for (k, b) in basis.iter_ud_order().enumerate() {
        next[i * base + k] = c * b;
      }
    }
    self.coeffs = next;
  }
}

/// Test-only helper methods for LagrangeCoeff.
#[cfg(test)]
impl<F: PrimeField, const D: usize> LagrangeCoeff<F, D> {
  pub fn get(&self, idx: usize) -> F {
    self.coeffs[idx]
  }
}

impl<F: PrimeField, const D: usize> LagrangeBasisFactory<F, D> {
  /// Construct the domain using an embedding from indices to field elements.
  pub fn new(embed: impl Fn(usize) -> F) -> Self {
    let finite_points = std::array::from_fn(embed);
    let weights = Self::weights_general(&finite_points);

    Self {
      finite_points,
      weights,
    }
  }

  /// Evaluate the Lagrange basis at r.
  ///
  /// Returns values in UdPoint order: [L∞(r), L0(r), L1(r), ..., L_{d-1}(r)].
  pub fn basis_at(&self, r: F) -> LagrangeBasisEval<F, D> {
    // One-hot if r equals a finite domain point.
    for (k, &xk) in self.finite_points.iter().enumerate() {
      if r == xk {
        let mut finite = [F::ZERO; D];
        finite[k] = F::ONE;
        return UdEvaluations::new(F::ZERO, finite);
      }
    }

    let diffs: [F; D] = std::array::from_fn(|i| r - self.finite_points[i]);

    // prefix[k] = ∏_{j < k} diffs[j]
    let base = UdPoint::<D>::BASE;
    let mut prefix = vec![F::ONE; base];
    for i in 0..D {
      prefix[i + 1] = prefix[i] * diffs[i];
    }

    // suffix[k] = ∏_{j > k} diffs[j]
    let mut suffix = vec![F::ONE; base];
    for i in (0..D).rev() {
      suffix[i] = suffix[i + 1] * diffs[i];
    }

    let prod = prefix[D]; // P(r) = ∏(r - x_k)

    let mut finite = [F::ZERO; D];
    for k in 0..D {
      let numer = prefix[k] * suffix[k + 1]; // P(r)/(r - x_k)
      finite[k] = numer * self.weights[k];
    }

    UdEvaluations::new(prod, finite)
  }

  fn weights_general(points: &[F; D]) -> [F; D] {
    let denoms = std::array::from_fn(|k| {
      let xk = points[k];
      let mut denom = F::ONE;
      for (m, &xm) in points.iter().enumerate() {
        if m == k {
          continue;
        }
        denom *= xk - xm;
      }
      denom
    });
    Self::batch_invert_array(denoms)
  }

  fn batch_invert_array(values: [F; D]) -> [F; D] {
    let mut prefix: Vec<F> = vec![F::ONE; D + 1];
    for i in 0..D {
      prefix[i + 1] = prefix[i] * values[i];
    }

    let inv_prod = prefix[D].invert().unwrap();

    let mut out = [F::ZERO; D];
    let mut suffix = F::ONE;
    for i in (0..D).rev() {
      out[i] = prefix[i] * suffix * inv_prod;
      suffix *= values[i];
    }
    out
  }
}

/// Test-only helper methods for LagrangeBasisFactory.
#[cfg(test)]
impl<F: PrimeField, const D: usize> LagrangeBasisFactory<F, D> {
  /// Evaluate an extended polynomial at r using the tensor-product Lagrange basis.
  pub fn eval_extended(
    &self,
    extended: &LagrangeEvaluatedMultilinearPolynomial<F, D>,
    r: &[F],
  ) -> F {
    assert_eq!(extended.num_vars(), r.len());

    let mut coeff = LagrangeCoeff::<F, D>::new();
    for &ri in r {
      let basis = self.basis_at(ri);
      coeff.extend(&basis);
    }

    let mut acc = F::ZERO;
    for idx in 0..extended.len() {
      acc += coeff.get(idx) * extended.get(idx);
    }
    acc
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
/// Type parameter `T` is the element type (field elements or i32 for small-value optimization).
pub struct LagrangeEvaluatedMultilinearPolynomial<T, const D: usize>
where
  T: Copy + Default + Add<Output = T> + Sub<Output = T>,
{
  #[allow(dead_code)] // Used by test-only methods (get, get_by_domain, len)
  evals: Vec<T>, // size (D+1)^num_vars
  #[allow(dead_code)] // Used by test-only num_vars() method
  num_vars: usize,
}

impl<T, const D: usize> LagrangeEvaluatedMultilinearPolynomial<T, D>
where
  T: Copy + Default + Add<Output = T> + Sub<Output = T>,
{
  /// Base of the extended domain U_D (= D + 1)
  const BASE: usize = D + 1;

  /// Extend boolean hypercube evaluations to Lagrange domain in-place.
  /// Returns (result_buffer_index, final_size) where:
  /// - result_buffer_index: 0 means result is in buf_a, 1 means result is in buf_b
  /// - final_size: number of valid elements in the result buffer
  ///
  /// This is the zero-allocation version - caller reads results directly from the buffer.
  ///
  /// # Arguments
  /// * `input` - Boolean hypercube evaluations (read-only slice, length must be power of 2)
  /// * `buf_a`, `buf_b` - Scratch buffers, will be resized if needed to (D+1)^num_vars
  pub fn extend_in_place(input: &[T], buf_a: &mut Vec<T>, buf_b: &mut Vec<T>) -> (usize, usize) {
    let num_vars = input.len().trailing_zeros() as usize;
    debug_assert_eq!(input.len(), 1 << num_vars, "Input size must be power of 2");

    if num_vars == 0 {
      // Single element: copy to buf_a and return
      if buf_a.is_empty() {
        buf_a.push(T::default());
      }
      buf_a[0] = input[0];
      return (0, 1);
    }

    let final_size = Self::BASE.pow(num_vars as u32);

    // Ensure buffers are large enough
    if buf_a.len() < final_size {
      buf_a.resize(final_size, T::default());
    }
    if buf_b.len() < final_size {
      buf_b.resize(final_size, T::default());
    }

    // Copy input into buf_a to start
    buf_a[..input.len()].copy_from_slice(input);

    for j in 1..=num_vars {
      // At step j:
      // - prefix_count = (D+1)^{j-1} extended prefix combinations
      // - suffix_count = 2^{num_vars-j} remaining boolean suffix combinations
      let prefix_count = Self::BASE.pow((j - 1) as u32);
      let suffix_count = 1usize << (num_vars - j);
      // Current layout: prefix_count rows × 2 boolean values × suffix_count elements
      let current_stride = 2 * suffix_count;
      // Next layout: prefix_count rows × (D+1) domain values × suffix_count elements
      let next_stride = Self::BASE * suffix_count;

      // Ping-pong between buffers
      let (src, dst) = if j % 2 == 1 {
        (&buf_a[..], &mut buf_b[..])
      } else {
        (&buf_b[..], &mut buf_a[..])
      };

      for prefix_idx in 0..prefix_count {
        for suffix_idx in 0..suffix_count {
          let base_current = prefix_idx * current_stride;
          let p0 = src[base_current + suffix_idx];
          let p1 = src[base_current + suffix_count + suffix_idx];

          let diff = p1 - p0;
          let base_next = prefix_idx * next_stride;

          // γ = ∞ (index 0): leading coefficient
          dst[base_next + suffix_idx] = diff;

          // γ = 0 (index 1): p(prefix, 0, suffix)
          dst[base_next + suffix_count + suffix_idx] = p0;

          if D >= 2 {
            // γ = 1 (index 2): p(prefix, 1, suffix)
            dst[base_next + 2 * suffix_count + suffix_idx] = p1;

            // γ = 2, 3, ..., D-1: extrapolate
            let mut val = p1;
            for k in 2..D {
              val = val + diff;
              dst[base_next + (k + 1) * suffix_count + suffix_idx] = val;
            }
          }
        }
      }
    }

    // Result is in whichever buffer was the last destination
    let result_buf = if num_vars % 2 == 1 { 1 } else { 0 };
    (result_buf, final_size)
  }
}

/// Test-only helper methods for LagrangeEvaluatedMultilinearPolynomial.
#[cfg(test)]
impl<T, const D: usize> LagrangeEvaluatedMultilinearPolynomial<T, D>
where
  T: Copy + Default + Add<Output = T> + Sub<Output = T>,
{
  /// Procedure 6: Extend polynomial evaluations from {0,1}^ℓ₀ to U_D^ℓ₀.
  pub fn from_boolean_evals(input: &[T]) -> Self {
    let num_vars = input.len().trailing_zeros() as usize;
    debug_assert_eq!(input.len(), 1 << num_vars, "Input size must be power of 2");

    let mut current = input.to_vec();

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
      let mut next = vec![T::default(); next_size];

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

          if D >= 2 {
            // γ = 1 (index 2): p(prefix, 1, suffix)
            next[base_next + 2 * suffix_count + suffix_idx] = p1;

            // γ = 2, 3, ..., D-1: extrapolate using accumulation (faster than multiplication)
            // val starts at p1 = p0 + 1*diff, then we add diff each iteration
            let mut val = p1;
            for k in 2..D {
              val = val + diff; // val = p0 + k*diff
              next[base_next + (k + 1) * suffix_count + suffix_idx] = val;
            }
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

  /// Get evaluation by flat index (performance path)
  #[inline]
  pub fn get(&self, idx: usize) -> T {
    self.evals[idx]
  }

  /// Number of evaluations
  #[inline]
  pub fn len(&self) -> usize {
    self.evals.len()
  }

  /// Get evaluation by domain tuple (type-safe path)
  #[inline]
  pub fn get_by_domain(&self, tuple: &UdTuple<D>) -> T {
    self.evals[tuple.to_flat_index()]
  }

  /// Number of variables
  pub fn num_vars(&self) -> usize {
    self.num_vars
  }

  /// Convert flat index to domain tuple
  pub fn to_domain_tuple(&self, flat_idx: usize) -> UdTuple<D> {
    UdTuple::from_flat_index(flat_idx, self.num_vars)
  }
}

/// Test-only: Create from a MultilinearPolynomial.
#[cfg(test)]
impl<F: PrimeField, const D: usize> LagrangeEvaluatedMultilinearPolynomial<F, D> {
  pub fn from_multilinear(poly: &MultilinearPolynomial<F>) -> Self {
    Self::from_boolean_evals(&poly.Z)
  }
}

/// Test-only: Convert i32 evaluations to field elements.
#[cfg(test)]
impl<const D: usize> LagrangeEvaluatedMultilinearPolynomial<i32, D> {
  pub fn to_field<F: SmallValueField<i32>>(&self) -> LagrangeEvaluatedMultilinearPolynomial<F, D> {
    LagrangeEvaluatedMultilinearPolynomial {
      evals: self.evals.iter().map(|&v| F::small_to_field(v)).collect(),
      num_vars: self.num_vars,
    }
  }
}

/// Test-only: Convert i64 evaluations to field elements.
#[cfg(test)]
#[allow(dead_code)]
impl<const D: usize> LagrangeEvaluatedMultilinearPolynomial<i64, D> {
  pub fn to_field<F: SmallValueField>(&self) -> LagrangeEvaluatedMultilinearPolynomial<F, D> {
    LagrangeEvaluatedMultilinearPolynomial {
      evals: self
        .evals
        .iter()
        .map(|&v| crate::small_field::i64_to_field(v))
        .collect(),
      num_vars: self.num_vars,
    }
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

  // === Lagrange basis tests ===

  // Property: basis is one-hot at a finite domain point and L∞(x_k)=0.
  // Why: ensures interpolation behaves like standard Lagrange on U_d.
  #[test]
  fn test_basis_at_domain_points_one_hot() {
    const D: usize = 3;

    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));

    for k in 0..D {
      let r = Scalar::from(k as u64);
      let basis = factory.basis_at(r);

      assert_eq!(basis.infinity, Scalar::ZERO);
      for j in 0..D {
        let expected = if j == k { Scalar::ONE } else { Scalar::ZERO };
        assert_eq!(basis.finite[j], expected);
      }
    }
  }

  // Property: L∞(r) equals ∏(r - x_k).
  // Why: this is the “infinity” basis polynomial from Lemma 2.2.
  #[test]
  fn test_basis_at_l_inf_product() {
    const D: usize = 3;

    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));
    let r = Scalar::from(5u64);
    let basis = factory.basis_at(r);

    let expected = (0..D).fold(Scalar::ONE, |acc, k| acc * (r - Scalar::from(k as u64)));
    assert_eq!(basis.infinity, expected);
  }

  // Property: Σ_k L_k(r) = 1 for any r (constant polynomial).
  // Why: validates finite basis is a partition of unity.
  #[test]
  fn test_basis_at_finite_sum_is_one() {
    const D: usize = 3;

    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));
    let r = Scalar::from(7u64);
    let basis = factory.basis_at(r);

    let sum = basis.finite.iter().fold(Scalar::ZERO, |acc, v| acc + v);
    assert_eq!(sum, Scalar::ONE);
  }

  // Property: UdPoint ordering and getters are consistent.
  // Why: R_i tensor update depends on the exact order.
  #[test]
  fn test_basis_eval_order_and_get() {
    const D: usize = 3;

    let eval = UdEvaluations::<Scalar, D>::new(
      Scalar::from(2u64),
      [Scalar::from(3u64), Scalar::from(5u64), Scalar::from(7u64)],
    );

    let vals: Vec<_> = eval.iter_ud_order().collect();
    assert_eq!(
      vals,
      vec![
        Scalar::from(2u64),
        Scalar::from(3u64),
        Scalar::from(5u64),
        Scalar::from(7u64),
      ]
    );

    assert_eq!(eval.get(UdPoint::Infinity), Scalar::from(2u64));
    assert_eq!(eval.get(UdPoint::Finite(2)), Scalar::from(7u64));
  }

  // Property: degree-2 polynomial is reconstructed from {∞,0,1}.
  // Why: checks ∞ handling for leading coefficient (d=2) across domain + random points.
  #[test]
  fn test_basis_reconstructs_deg2_poly() {
    const D: usize = 2;

    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));

    let eval = |x: Scalar| x * x + Scalar::from(2u64) * x + Scalar::ONE;
    let s_inf = Scalar::ONE; // leading coeff of x^2 + 2x + 1
    let s0 = eval(Scalar::ZERO);
    let s1 = eval(Scalar::ONE);

    let mut rng = rand_core::OsRng;
    let mut rs = vec![Scalar::ZERO, Scalar::ONE];
    for _ in 0..3 {
      rs.push(Scalar::random(&mut rng));
    }
    for r in rs {
      let basis = factory.basis_at(r);
      let reconstructed = s_inf * basis.infinity + s0 * basis.finite[0] + s1 * basis.finite[1];
      assert_eq!(reconstructed, eval(r));
    }
  }

  // Property: degree-3 polynomial is reconstructed from {∞,0,1,2}.
  // Why: checks full d=3 interpolation with ∞ term across domain + random points.
  #[test]
  fn test_basis_reconstructs_deg3_poly() {
    const D: usize = 3;

    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));

    let eval = |x: Scalar| {
      let x2 = x * x;
      let x3 = x2 * x;
      Scalar::from(2u64) * x3 - Scalar::from(4u64) * x2 + Scalar::from(5u64) * x - Scalar::ONE
    };

    let s_inf = Scalar::from(2u64); // leading coeff of 2x^3 - 4x^2 + 5x - 1
    let s0 = eval(Scalar::ZERO);
    let s1 = eval(Scalar::ONE);
    let s2 = eval(Scalar::from(2u64));

    let mut rng = rand_core::OsRng;
    let mut rs = vec![Scalar::ZERO, Scalar::ONE, Scalar::from(2u64)];
    for _ in 0..4 {
      rs.push(Scalar::random(&mut rng));
    }
    for r in rs {
      let basis = factory.basis_at(r);
      let reconstructed =
        s_inf * basis.infinity + s0 * basis.finite[0] + s1 * basis.finite[1] + s2 * basis.finite[2];
      assert_eq!(reconstructed, eval(r));
    }
  }

  // === LagrangeCoeff tests ===

  // Property: R_1 starts at 1 and extend copies basis in Ud order.
  // Why: verifies tensor growth for Algorithm 6.
  #[test]
  fn test_lagrange_coeff_new_and_extend() {
    const D: usize = 3;

    let mut coeff = LagrangeCoeff::<Scalar, D>::new();
    assert_eq!(coeff.len(), 1);
    assert_eq!(coeff.get(0), Scalar::ONE);

    let basis = UdEvaluations::<Scalar, D>::new(
      Scalar::from(2u64),
      [Scalar::from(3u64), Scalar::from(5u64), Scalar::from(7u64)],
    );

    coeff.extend(&basis);
    assert_eq!(coeff.len(), D + 1);
    assert_eq!(
      coeff.as_slice(),
      &[
        Scalar::from(2u64),
        Scalar::from(3u64),
        Scalar::from(5u64),
        Scalar::from(7u64),
      ]
    );
  }

  // Property: R_2 equals outer product of two basis vectors.
  // Why: ensures extend() matches Kronecker definition.
  #[test]
  #[allow(clippy::needless_range_loop)]
  fn test_lagrange_coeff_tensor_product() {
    const D: usize = 3;
    let base = D + 1;

    let basis1 = UdEvaluations::<Scalar, D>::new(
      Scalar::from(2u64),
      [Scalar::from(3u64), Scalar::from(5u64), Scalar::from(7u64)],
    );
    let basis2 = UdEvaluations::<Scalar, D>::new(
      Scalar::from(11u64),
      [
        Scalar::from(13u64),
        Scalar::from(17u64),
        Scalar::from(19u64),
      ],
    );

    let mut coeff = LagrangeCoeff::<Scalar, D>::new();
    coeff.extend(&basis1);
    coeff.extend(&basis2);

    let b1: Vec<_> = basis1.iter_ud_order().collect();
    let b2: Vec<_> = basis2.iter_ud_order().collect();
    for i in 0..base {
      for j in 0..base {
        assert_eq!(coeff.get(i * base + j), b1[i] * b2[j]);
      }
    }
  }

  // Property: LagrangeCoeff + extended evals matches direct multilinear evaluation.
  // Why: validates tensor-product basis evaluation for a single multilinear factor (D=1).
  #[test]
  fn test_lagrange_coeff_matches_direct_eval_multilinear() {
    const D: usize = 1;
    let num_vars = 4;
    let mut rng = rand_core::OsRng;

    let evals: Vec<Scalar> = (0..(1 << num_vars))
      .map(|_| Scalar::random(&mut rng))
      .collect();
    let poly = MultilinearPolynomial::new(evals.clone());
    let extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&poly);
    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));

    // Check the finite domain point in U_d^4 (only 0 when D=1).
    let r = [Scalar::ZERO, Scalar::ZERO, Scalar::ZERO, Scalar::ZERO];
    let direct = evaluate_multilinear(&evals, &r);
    let lagrange = factory.eval_extended(&extended, &r);
    assert_eq!(direct, lagrange);

    // Check a few random points in F^4.
    for _ in 0..3 {
      let r = [
        Scalar::random(&mut rng),
        Scalar::random(&mut rng),
        Scalar::random(&mut rng),
        Scalar::random(&mut rng),
      ];
      let direct = evaluate_multilinear(&evals, &r);
      let lagrange = factory.eval_extended(&extended, &r);
      assert_eq!(direct, lagrange);
    }
  }

  // Property: LagrangeCoeff matches direct eval for product of three multilinear polynomials.
  // Why: validates degree-3 behavior over U_d with d=3.
  #[test]
  fn test_lagrange_coeff_matches_direct_eval_product_of_three() {
    const D: usize = 3;
    let num_vars = 4;
    let mut rng = rand_core::OsRng;

    let evals1: Vec<Scalar> = (0..(1 << num_vars))
      .map(|_| Scalar::random(&mut rng))
      .collect();
    let evals2: Vec<Scalar> = (0..(1 << num_vars))
      .map(|_| Scalar::random(&mut rng))
      .collect();
    let evals3: Vec<Scalar> = (0..(1 << num_vars))
      .map(|_| Scalar::random(&mut rng))
      .collect();

    let p1 = MultilinearPolynomial::new(evals1.clone());
    let p2 = MultilinearPolynomial::new(evals2.clone());
    let p3 = MultilinearPolynomial::new(evals3.clone());

    let ext1 = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&p1);
    let ext2 = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&p2);
    let ext3 = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&p3);

    let mut prod_evals = vec![Scalar::ZERO; ext1.len()];
    #[allow(clippy::needless_range_loop)]
    for i in 0..ext1.len() {
      prod_evals[i] = ext1.get(i) * ext2.get(i) * ext3.get(i);
    }

    let prod_extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D> {
      evals: prod_evals,
      num_vars,
    };
    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));

    // Check all finite domain points in U_d^4.
    for a in 0..D {
      for b in 0..D {
        for c in 0..D {
          for d in 0..D {
            let r = [
              Scalar::from(a as u64),
              Scalar::from(b as u64),
              Scalar::from(c as u64),
              Scalar::from(d as u64),
            ];
            let direct = evaluate_multilinear(&evals1, &r)
              * evaluate_multilinear(&evals2, &r)
              * evaluate_multilinear(&evals3, &r);
            let lagrange = factory.eval_extended(&prod_extended, &r);
            assert_eq!(direct, lagrange);
          }
        }
      }
    }

    // Check a few random points in F^4.
    for _ in 0..3 {
      let r = [
        Scalar::random(&mut rng),
        Scalar::random(&mut rng),
        Scalar::random(&mut rng),
        Scalar::random(&mut rng),
      ];
      let direct = evaluate_multilinear(&evals1, &r)
        * evaluate_multilinear(&evals2, &r)
        * evaluate_multilinear(&evals3, &r);
      let lagrange = factory.eval_extended(&prod_extended, &r);
      assert_eq!(direct, lagrange);
    }
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

    let input: Vec<Scalar> = (0..(1 << num_vars))
      .map(|_| Scalar::random(&mut rand_core::OsRng))
      .collect();
    let poly = MultilinearPolynomial::new(input.clone());

    let extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&poly);

    // In U_d indexing: 0 → index 1, 1 → index 2
    #[allow(clippy::needless_range_loop)]
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

    let input: Vec<Scalar> = (0..(1 << num_vars))
      .map(|_| Scalar::random(&mut rand_core::OsRng))
      .collect();
    let poly = MultilinearPolynomial::new(input.clone());
    let extended = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&poly);

    // Check all finite points via direct multilinear evaluation
    for idx in 0..extended.len() {
      let tuple = index_to_tuple(idx, base, num_vars);

      // Skip infinity points (index 0 in any coordinate)
      if tuple.contains(&0) {
        continue;
      }

      // Convert U_d indices to field values: index k → value k-1
      let point: Vec<Scalar> = tuple
        .iter()
        .map(|&t| Scalar::from((t - 1) as u64))
        .collect();

      let direct = evaluate_multilinear(&input, &point);
      assert_eq!(extended.get(idx), direct);
    }
  }

  #[test]
  #[allow(clippy::identity_op, clippy::erasing_op)]
  fn test_extend_infinity_leading_coeff() {
    use ff::Field;

    let num_vars = 3;
    const D: usize = 3;
    let base = D + 1;

    let input: Vec<Scalar> = (0..(1 << num_vars))
      .map(|_| Scalar::random(&mut rand_core::OsRng))
      .collect();
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
  #[allow(clippy::identity_op, clippy::erasing_op)]
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
    assert_eq!(extended.get(0), Scalar::ZERO, "p(∞,∞,∞) = 0 (no XYZ term)");
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

  /// Direct multilinear evaluation: p(r) = Σ_x p(x) · eq(x, r).
  ///
  /// This mirrors EqPolynomial::evals_from_points() so the bit ordering
  /// matches the codebase's {0,1}^ℓ indexing.
  fn evaluate_multilinear(evals: &[Scalar], point: &[Scalar]) -> Scalar {
    let chis = crate::polys::eq::EqPolynomial::evals_from_points(point);
    evals
      .iter()
      .zip(chis.iter())
      .fold(Scalar::ZERO, |acc, (z, chi)| acc + *z * *chi)
  }

  // === SmallLagrangePolynomial tests ===

  #[test]
  fn test_small_lagrange_matches_field_version() {
    use crate::small_field::SmallValueField;

    const D: usize = 3;
    let num_vars = 3;

    // Create input as small values (using SmallValueField trait)
    let input_small: Vec<i32> = (0..(1 << num_vars))
      .map(|i| Scalar::small_from_i32(i + 1))
      .collect();

    // Create same input as field elements
    let input_field: Vec<Scalar> = (0..(1 << num_vars))
      .map(|i| Scalar::from((i + 1) as u64))
      .collect();
    let poly = MultilinearPolynomial::new(input_field);

    // Extend using both methods
    let small_ext =
      LagrangeEvaluatedMultilinearPolynomial::<i32, D>::from_boolean_evals(&input_small);
    let field_ext = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&poly);

    // Verify they match
    assert_eq!(small_ext.len(), field_ext.len());
    for i in 0..small_ext.len() {
      let small_as_field: Scalar = Scalar::small_to_field(small_ext.get(i));
      assert_eq!(small_as_field, field_ext.get(i), "mismatch at index {i}");
    }
  }

  #[test]
  fn test_small_lagrange_single_var() {
    use crate::small_field::SmallValueField;

    let p0: i32 = Scalar::small_from_i32(7);
    let p1: i32 = Scalar::small_from_i32(19);

    let input = vec![p0, p1];
    let extended = LagrangeEvaluatedMultilinearPolynomial::<i32, 3>::from_boolean_evals(&input);

    // U_d = {∞, 0, 1, 2} with indices 0, 1, 2, 3
    assert_eq!(extended.get(0), p1 - p0, "p(∞) = leading coeff");
    assert_eq!(extended.get(1), p0, "p(0)");
    assert_eq!(extended.get(2), p1, "p(1)");
    // p(2) = p0 + 2 * (p1 - p0) = 2*p1 - p0 = 2*19 - 7 = 31
    assert_eq!(extended.get(3), 31i32, "p(2) = 2*p1 - p0");
  }

  #[test]
  fn test_small_lagrange_extend_in_place() {
    use crate::small_field::SmallValueField;

    const D: usize = 2;
    let num_vars = 3;

    let input: Vec<i32> = (0..(1 << num_vars))
      .map(|i| Scalar::small_from_i32(i * 2 + 1))
      .collect();

    // Extend using allocating version
    let ext1 = LagrangeEvaluatedMultilinearPolynomial::<i32, D>::from_boolean_evals(&input);

    // Extend in-place (zero allocation after initial buffer setup)
    let mut buf_a = Vec::new();
    let mut buf_b = Vec::new();
    let (result_buf, final_size) =
      LagrangeEvaluatedMultilinearPolynomial::<i32, D>::extend_in_place(
        &input, &mut buf_a, &mut buf_b,
      );
    let ext2 = if result_buf == 0 {
      &buf_a[..final_size]
    } else {
      &buf_b[..final_size]
    };

    // Verify they match
    assert_eq!(ext1.len(), final_size);
    for (i, &ext2_val) in ext2.iter().enumerate() {
      assert_eq!(ext1.get(i), ext2_val, "mismatch at index {i}");
    }
  }

  #[test]
  fn test_small_lagrange_to_field() {
    use crate::small_field::SmallValueField;

    const D: usize = 2;
    let num_vars = 2;

    let input: Vec<i32> = (0..(1 << num_vars))
      .map(|i| Scalar::small_from_i32(i + 1))
      .collect();

    let small_ext = LagrangeEvaluatedMultilinearPolynomial::<i32, D>::from_boolean_evals(&input);
    let field_ext: LagrangeEvaluatedMultilinearPolynomial<Scalar, D> =
      small_ext.to_field::<Scalar>();

    // Verify conversion
    for i in 0..small_ext.len() {
      let expected: Scalar = Scalar::small_to_field(small_ext.get(i));
      assert_eq!(field_ext.get(i), expected);
    }
  }

  #[test]
  fn test_small_lagrange_negative_values() {
    use crate::small_field::SmallValueField;

    // Test with negative differences (p0 > p1)
    let p0: i32 = Scalar::small_from_i32(100);
    let p1: i32 = Scalar::small_from_i32(50);

    let input = vec![p0, p1];
    let extended = LagrangeEvaluatedMultilinearPolynomial::<i32, 2>::from_boolean_evals(&input);

    // p(∞) = p1 - p0 = -50
    assert_eq!(extended.get(0), -50i32);
    assert_eq!(extended.get(1), p0);
    assert_eq!(extended.get(2), p1);

    // Verify field conversion handles negatives correctly
    let field_ext: LagrangeEvaluatedMultilinearPolynomial<Scalar, 2> =
      extended.to_field::<Scalar>();
    assert_eq!(field_ext.get(0), -Scalar::from(50u64));
  }
}
