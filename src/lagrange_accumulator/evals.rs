// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Evaluation containers for Lagrange domains U_d and Û_d.

use ff::PrimeField;

#[cfg(test)]
use super::domain::LagrangePoint;

/// Evaluations at all D+1 points of U_d = {∞, 0, 1, ..., D-1}.
///
/// This type stores values indexed by [`LagrangePoint<D>`], with the infinity
/// point stored separately from the D finite points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LagrangeEvals<T, const D: usize> {
  /// Value at the infinity point
  pub infinity: T,
  /// Values at finite points 0, 1, ..., D-1
  pub finite: [T; D],
}

impl<T: Copy, const D: usize> LagrangeEvals<T, D> {
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

/// Test-only helper methods for LagrangeEvals.
#[cfg(test)]
impl<T: Copy, const D: usize> LagrangeEvals<T, D> {
  /// Get value at a domain point.
  #[inline]
  pub fn get(&self, p: LagrangePoint<D>) -> T {
    match p {
      LagrangePoint::Infinity => self.infinity,
      LagrangePoint::Finite(k) => self.finite[k],
    }
  }
}

impl<F: PrimeField> LagrangeEvals<F, 2> {
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
/// Indexing follows [`LagrangeHatPoint::to_index()`]: ∞→0, 0→1, 2→2, 3→3, ...
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LagrangeHatEvals<T, const D: usize> {
  data: [T; D],
}

impl<T: Copy, const D: usize> LagrangeHatEvals<T, D> {
  /// Create from array indexed by `LagrangeHatPoint::to_index()`.
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
