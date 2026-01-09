// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Lagrange basis computation for U_d = {∞, 0, 1, ..., d-1}.

use super::{domain::LagrangePoint, evals::LagrangeEvals};
use ff::PrimeField;

#[cfg(test)]
use super::extension::LagrangeEvaluatedMultilinearPolynomial;

/// Evaluated Lagrange basis at a single r, stored in LagrangePoint order.
pub type LagrangeBasisEval<F, const D: usize> = LagrangeEvals<F, D>;

/// Precomputes data for the 1-D Lagrange basis on U_d = {∞, 0, 1, ..., d-1}.
///
/// `finite_points[k]` is the field element for LagrangePoint::Finite(k).
/// `weights[k]` is the barycentric weight:
///   w_k = 1 / ∏_{m≠k} (x_k - x_m)
///
/// With these weights, basis evaluation at any r costs O(D) multiplies
/// and uses no per-round inversions.
pub struct LagrangeBasisFactory<F, const D: usize> {
  finite_points: [F; D],
  weights: [F; D],
}

/// R_i tensor coefficients used in Algorithm 6.
///
/// Indexing matches LagrangeIndex::to_flat_index() over U_d^{i-1}.
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

  /// Returns the number of coefficients.
  pub fn len(&self) -> usize {
    self.coeffs.len()
  }

  /// Returns a slice of the coefficients.
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
#[allow(missing_docs)]
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
  /// Returns values in LagrangePoint order: [L∞(r), L0(r), L1(r), ..., L_{d-1}(r)].
  pub fn basis_at(&self, r: F) -> LagrangeBasisEval<F, D> {
    // One-hot if r equals a finite domain point.
    for (k, &xk) in self.finite_points.iter().enumerate() {
      if r == xk {
        let mut finite = [F::ZERO; D];
        finite[k] = F::ONE;
        return LagrangeEvals::new(F::ZERO, finite);
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

    LagrangeEvals::new(prod, finite)
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{polys::multilinear::MultilinearPolynomial, provider::pasta::pallas};
  use ff::Field;

  type Scalar = pallas::Scalar;

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

      assert_eq!(basis.infinity, Scalar::ZERO);
      for j in 0..D {
        let expected = if j == k { Scalar::ONE } else { Scalar::ZERO };
        assert_eq!(basis.finite[j], expected);
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
    assert_eq!(basis.infinity, expected);
  }

  // Property: Σ_k L_k(r) = 1 for any r (constant polynomial).
  #[test]
  fn test_basis_at_finite_sum_is_one() {
    const D: usize = 3;

    let factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));
    let r = Scalar::from(7u64);
    let basis = factory.basis_at(r);

    let sum = basis.finite.iter().fold(Scalar::ZERO, |acc, v| acc + v);
    assert_eq!(sum, Scalar::ONE);
  }

  // Property: LagrangePoint ordering and getters are consistent.
  #[test]
  fn test_basis_eval_order_and_get() {
    const D: usize = 3;

    let eval = LagrangeEvals::<Scalar, D>::new(
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
      let reconstructed = s_inf * basis.infinity + s0 * basis.finite[0] + s1 * basis.finite[1];
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
      let reconstructed =
        s_inf * basis.infinity + s0 * basis.finite[0] + s1 * basis.finite[1] + s2 * basis.finite[2];
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

    let basis = LagrangeEvals::<Scalar, D>::new(
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

    let basis1 = LagrangeEvals::<Scalar, D>::new(
      Scalar::from(2u64),
      [Scalar::from(3u64), Scalar::from(5u64), Scalar::from(7u64)],
    );
    let basis2 = LagrangeEvals::<Scalar, D>::new(
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

    let prod_extended =
      LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_evals(prod_evals, num_vars);
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
}
