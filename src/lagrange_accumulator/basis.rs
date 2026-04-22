// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Lagrange-domain evaluation types and basis computation for
//! `U_d = {∞, 0, 1, ..., d-1}`.

use super::domain::LagrangePoint;
use ff::PrimeField;

/// Evaluations at all `D + 1` points of `U_d = {∞, 0, 1, ..., D-1}`.
///
/// Values are stored in [`LagrangePoint<D>`] order, with the point at infinity
/// separated from the `D` finite points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LagrangeDomainEvals<T, const D: usize> {
  infinity: T,
  finite: [T; D],
}

impl<T: Copy, const D: usize> LagrangeDomainEvals<T, D> {
  /// Create new evaluations from infinity and finite values.
  #[inline]
  pub(crate) fn new(infinity: T, finite: [T; D]) -> Self {
    Self { infinity, finite }
  }

  /// Get value at infinity.
  #[inline]
  pub(crate) fn at_infinity(&self) -> T {
    self.infinity
  }

  /// Get value at zero (finite point 0).
  #[inline]
  pub(crate) fn at_zero(&self) -> T {
    self.finite[0]
  }

  /// Get value at one (finite point 1).
  ///
  /// # Panics (debug builds only)
  /// Panics if `D < 2`.
  #[inline]
  pub(crate) fn at_one(&self) -> T {
    debug_assert!(D >= 2, "at_one() requires D >= 2");
    self.finite[1]
  }

  /// Iterate values in `U_d` order: `[∞, 0, 1, ..., D-1]`.
  pub(crate) fn iter_ud_order(&self) -> impl Iterator<Item = T> + '_ {
    std::iter::once(self.infinity).chain(self.finite.iter().copied())
  }
}

/// Test-only helper methods for `LagrangeDomainEvals`.
#[cfg(test)]
impl<T: Copy, const D: usize> LagrangeDomainEvals<T, D> {
  /// Get the value at a specific domain point.
  #[inline]
  pub(in crate::lagrange_accumulator) fn get(&self, p: LagrangePoint<D>) -> T {
    match p {
      LagrangePoint::Infinity => self.infinity,
      LagrangePoint::Finite(k) => self.finite[k],
    }
  }
}

impl<F: PrimeField> LagrangeDomainEvals<F, 2> {
  /// Evaluate the represented linear polynomial at `u`.
  ///
  /// For evaluations of a degree-1 polynomial over `U_2 = {∞, 0, 1}`, this
  /// computes `L(u) = l_∞ · u + l_0`.
  #[inline]
  pub(crate) fn eval_linear_at(&self, u: F) -> F {
    self.infinity * u + self.finite[0]
  }
}

/// Evaluations at all `D` points of `Û_d = U_d \ {1}`.
///
/// This reduced domain excludes `1` because `s(1)` is recovered later from the
/// sum-check relation `s(0) + s(1) = claim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReducedLagrangeDomainEvals<T, const D: usize> {
  data: [T; D],
}

impl<T: Copy, const D: usize> ReducedLagrangeDomainEvals<T, D> {
  /// Create from array indexed by `LagrangeHatPoint::to_index()`.
  #[inline]
  pub(crate) fn from_array(data: [T; D]) -> Self {
    Self { data }
  }

  /// Get value at infinity (index 0).
  #[inline]
  pub(crate) fn at_infinity(&self) -> T {
    self.data[0]
  }

  /// Get value at zero (index 1).
  #[inline]
  pub(crate) fn at_zero(&self) -> T {
    self.data[1]
  }
}

/// Precomputes data for the 1-D Lagrange basis on U_d = {∞, 0, 1, ..., d-1}.
///
/// `finite_points[k]` is the field element for LagrangePoint::Finite(k).
/// `weights[k]` is the barycentric weight:
///   w_k = 1 / ∏_{m≠k} (x_k - x_m)
///
/// With these weights, basis evaluation at any r costs O(D) multiplies
/// and uses no per-round inversions.
pub(crate) struct LagrangeBasisFactory<F, const D: usize> {
  finite_points: [F; D],
  weights: [F; D],
}

/// Tensor-product coefficients `R_i(v)` used to recover `t_i(u)` from the
/// accumulator table `A_i(v, u)`.
///
/// After the first `i - 1` verifier challenges have been bound, the round-`i`
/// values are computed as
///
/// `t_i(u) = Σ_{v ∈ U_d^{i-1}} R_i(v) · A_i(v, u)`.
///
/// For `v = (v_1, ..., v_{i-1})`, the coefficient is the tensor-product
/// Lagrange basis evaluation
///
/// `R_i(v) = ∏_{j=1}^{i-1} L_{v_j}(r_j)`,
///
/// where `L_{v_j}` is the 1-D Lagrange basis polynomial on `U_d` evaluated at
/// the verifier challenge `r_j`.
///
/// `coeffs` stores the flattened table of `R_i(v)` values in
/// `LagrangeIndex::to_flat_index()` order over `U_d^{i-1}`.
pub(crate) struct LagrangeCoeff<F, const D: usize> {
  coeffs: Vec<F>,
}

impl<F: PrimeField, const D: usize> Default for LagrangeCoeff<F, D> {
  fn default() -> Self {
    Self::new()
  }
}

impl<F: PrimeField, const D: usize> LagrangeCoeff<F, D> {
  /// Initialize the base case `R_1 = [1]`.
  ///
  /// In the first round there is no prefix `v`, so the coefficient table has a
  /// single entry equal to 1.
  pub(crate) fn new() -> Self {
    Self {
      coeffs: vec![F::ONE],
    }
  }

  /// Returns the number of coefficients.
  pub(crate) fn len(&self) -> usize {
    self.coeffs.len()
  }

  /// Returns a slice of the coefficients.
  pub(crate) fn as_slice(&self) -> &[F] {
    &self.coeffs
  }

  /// Extend the coefficient table by one verifier challenge:
  /// `R_{i+1} = R_i ⊗ L(r_i)`.
  ///
  /// If `R_i` is indexed by `v ∈ U_d^{i-1}` and `L(r_i)` is the 1-D Lagrange
  /// basis on `U_d`, then the new table is indexed by `(v, k) ∈ U_d^i` and
  /// stores `R_{i+1}(v, k) = R_i(v) · L_k(r_i)`.
  pub(crate) fn extend(&mut self, basis: &LagrangeDomainEvals<F, D>) {
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
#[allow(missing_docs)]
impl<F: PrimeField, const D: usize> LagrangeCoeff<F, D> {
  pub fn get(&self, idx: usize) -> F {
    self.coeffs[idx]
  }
}

impl<F: PrimeField, const D: usize> LagrangeBasisFactory<F, D> {
  /// Construct the domain using an embedding from indices to field elements.
  pub(crate) fn new(embed: impl Fn(usize) -> F) -> Self {
    let finite_points = std::array::from_fn(embed);
    let weights = Self::weights_general(&finite_points);

    Self {
      finite_points,
      weights,
    }
  }

  /// Evaluate the Lagrange basis at r.
  ///
  /// Returns values in LagrangePoint order: [L∞(r), L0(r), L1(r), ..., L_{d-1}(r)].
  pub(crate) fn basis_at(&self, r: F) -> LagrangeDomainEvals<F, D> {
    // One-hot if r equals a finite domain point.
    for (k, &xk) in self.finite_points.iter().enumerate() {
      if r == xk {
        let mut finite = [F::ZERO; D];
        finite[k] = F::ONE;
        return LagrangeDomainEvals::new(F::ZERO, finite);
      }
    }

    let diffs: [F; D] = std::array::from_fn(|i| r - self.finite_points[i]);

    // prefix[k] = ∏_{j < k} diffs[j]
    let base = LagrangePoint::<D>::BASE;
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

    LagrangeDomainEvals::new(prod, finite)
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

    let inv_prod = prefix[D]
      .invert()
      .expect("batch inversion failed: input contains zero or duplicate points");

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
  ///
  /// `extended` is a slice of length (D+1)^num_vars in LagrangeIndex order.
  pub fn eval_extended(&self, extended: &[F], r: &[F]) -> F {
    let base = D + 1;
    let expected_len = base.pow(r.len() as u32);
    assert_eq!(extended.len(), expected_len);

    let mut coeff = LagrangeCoeff::<F, D>::new();
    for &ri in r {
      let basis = self.basis_at(ri);
      coeff.extend(&basis);
    }

    let mut acc = F::ZERO;
    for idx in 0..extended.len() {
      acc += coeff.get(idx) * extended[idx];
    }
    acc
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    lagrange_accumulator::extension::extend_to_lagrange_domain, provider::pasta::pallas,
  };
  use ff::Field;

  type Scalar = pallas::Scalar;

  /// Test helper: extend boolean evaluations to Lagrange domain.
  fn extend_for_test<const D: usize>(input: &[Scalar]) -> Vec<Scalar> {
    let mut buf = Vec::new();
    let mut scratch = Vec::new();
    let size = extend_to_lagrange_domain::<Scalar, D>(input, &mut buf, &mut scratch);
    buf.truncate(size);
    buf
  }

  fn evaluate_multilinear(evals: &[Scalar], point: &[Scalar]) -> Scalar {
    let chis = crate::polys::eq::EqPolynomial::evals_from_points(point);
    evals
      .iter()
      .zip(chis.iter())
      .fold(Scalar::ZERO, |acc, (z, chi)| acc + *z * *chi)
  }

  // === Lagrange basis tests ===

  // Property: basis is one-hot at a finite domain point and L∞(x_k)=0.
  #[test]
  fn test_basis_at_domain_points_one_hot() {
    const D: usize = 3;

    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));

    for k in 0..D {
      let r = Scalar::from(k as u64);
      let basis = factory.basis_at(r);

      assert_eq!(basis.at_infinity(), Scalar::ZERO);
      for j in 0..D {
        let expected = if j == k { Scalar::ONE } else { Scalar::ZERO };
        assert_eq!(basis.get(LagrangePoint::Finite(j)), expected);
      }
    }
  }

  // Property: L∞(r) equals ∏(r - x_k).
  #[test]
  fn test_basis_at_l_inf_product() {
    const D: usize = 3;

    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));
    let r = Scalar::from(5u64);
    let basis = factory.basis_at(r);

    let expected = (0..D).fold(Scalar::ONE, |acc, k| acc * (r - Scalar::from(k as u64)));
    assert_eq!(basis.at_infinity(), expected);
  }

  // Property: Σ_k L_k(r) = 1 for any r (constant polynomial).
  #[test]
  fn test_basis_at_finite_sum_is_one() {
    const D: usize = 3;

    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));
    let r = Scalar::from(7u64);
    let basis = factory.basis_at(r);

    let sum = (0..D).fold(Scalar::ZERO, |acc, j| {
      acc + basis.get(LagrangePoint::Finite(j))
    });
    assert_eq!(sum, Scalar::ONE);
  }

  // Property: LagrangePoint ordering and getters are consistent.
  #[test]
  fn test_basis_eval_order_and_get() {
    const D: usize = 3;

    let eval = LagrangeDomainEvals::<Scalar, D>::new(
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

    assert_eq!(eval.get(LagrangePoint::Infinity), Scalar::from(2u64));
    assert_eq!(eval.get(LagrangePoint::Finite(2)), Scalar::from(7u64));
  }

  // Property: degree-2 polynomial is reconstructed from {∞,0,1}.
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
      let reconstructed = s_inf * basis.at_infinity() + s0 * basis.at_zero() + s1 * basis.at_one();
      assert_eq!(reconstructed, eval(r));
    }
  }

  // Property: degree-3 polynomial is reconstructed from {∞,0,1,2}.
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
      let reconstructed = s_inf * basis.at_infinity()
        + s0 * basis.at_zero()
        + s1 * basis.at_one()
        + s2 * basis.get(LagrangePoint::Finite(2));
      assert_eq!(reconstructed, eval(r));
    }
  }

  // === LagrangeCoeff tests ===

  // Property: R_1 starts at 1 and extend copies basis in Ud order.
  #[test]
  fn test_lagrange_coeff_new_and_extend() {
    const D: usize = 3;

    let mut coeff = LagrangeCoeff::<Scalar, D>::new();
    assert_eq!(coeff.len(), 1);
    assert_eq!(coeff.get(0), Scalar::ONE);

    let basis = LagrangeDomainEvals::<Scalar, D>::new(
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
  #[test]
  #[allow(clippy::needless_range_loop)]
  fn test_lagrange_coeff_tensor_product() {
    const D: usize = 3;
    let base = D + 1;

    let basis1 = LagrangeDomainEvals::<Scalar, D>::new(
      Scalar::from(2u64),
      [Scalar::from(3u64), Scalar::from(5u64), Scalar::from(7u64)],
    );
    let basis2 = LagrangeDomainEvals::<Scalar, D>::new(
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
  #[test]
  fn test_lagrange_coeff_matches_direct_eval_multilinear() {
    const D: usize = 1;
    let num_vars = 4;
    let mut rng = rand_core::OsRng;

    let evals: Vec<Scalar> = (0..(1 << num_vars))
      .map(|_| Scalar::random(&mut rng))
      .collect();
    let extended = extend_for_test::<D>(&evals);
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

    let ext1 = extend_for_test::<D>(&evals1);
    let ext2 = extend_for_test::<D>(&evals2);
    let ext3 = extend_for_test::<D>(&evals3);

    let prod_evals: Vec<Scalar> = ext1
      .iter()
      .zip(ext2.iter())
      .zip(ext3.iter())
      .map(|((&a, &b), &c)| a * b * c)
      .collect();

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
            let lagrange = factory.eval_extended(&prod_evals, &r);
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
      let lagrange = factory.eval_extended(&prod_evals, &r);
      assert_eq!(direct, lagrange);
    }
  }
}
