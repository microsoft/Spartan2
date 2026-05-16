// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Main components:
//! - `MultilinearPolynomial`: Dense representation of multilinear polynomials, represented by evaluations over all possible binary inputs.
//! - `SparsePolynomial`: Efficient representation of sparse multilinear polynomials, storing only non-zero evaluations.

use crate::{math::Math, polys::eq::EqPolynomial};
use core::ops::Index;
use ff::{Field, PrimeField};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A multilinear extension of a polynomial $Z(\cdot)$, denote it as $\tilde{Z}(x_1, ..., x_m)$
/// where the degree of each variable is at most one.
///
/// This is the dense representation of a multilinear polynomial.
/// Let it be $\mathbb{G}(\cdot): \mathbb{F}^m \rightarrow \mathbb{F}$, it can be represented uniquely by the list of
/// evaluations of $\mathbb{G}(\cdot)$ over the Boolean hypercube $\{0, 1\}^m$.
///
/// For example, a 3 variables multilinear polynomial can be represented by evaluation
/// at points $[0, 2^3-1]$.
///
/// The implementation follows
/// $$
/// \tilde{Z}(x_1, ..., x_m) = \sum_{e\in {0,1}^m}Z(e) \cdot \prod_{i=1}^m(x_i \cdot e_i + (1-x_i) \cdot (1-e_i))
/// $$
///
/// Vector $Z$ indicates $Z(e)$ where $e$ ranges from $0$ to $2^m-1$.
///
/// The type parameter `T` is the coefficient type. Typically this is a field element,
/// but can also be any type with ring operations (add, sub, mul, zero).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultilinearPolynomial<T> {
  pub(crate) Z: Vec<T>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
  // Non-zero prefix lengths for each half of Z, enabling zero-skip in bind/eval.
  // When Z has 2n elements: lo_eff covers Z[0..n), hi_eff covers Z[n..2n).
  // Z[i] = 0 for lo_eff <= i < n, and Z[n+i] = 0 for hi_eff <= i < n.
  // Default usize::MAX means "unknown, assume fully non-zero".
  #[serde(skip, default = "default_eff")]
  pub(crate) lo_eff: usize,
  #[serde(skip, default = "default_eff")]
  pub(crate) hi_eff: usize,
}

fn default_eff() -> usize {
  usize::MAX
}

impl<T> MultilinearPolynomial<T> {
  /// Creates a new `MultilinearPolynomial` from the given evaluations.
  ///
  /// # Panics
  /// Panics if the number of evaluations is empty or not a power of two.
  pub fn new(Z: Vec<T>) -> Self {
    assert!(
      !Z.is_empty() && Z.len().is_power_of_two(),
      "Vector Z must be non-empty and its length must be a power of two."
    );
    MultilinearPolynomial {
      Z,
      lo_eff: usize::MAX,
      hi_eff: usize::MAX,
    }
  }

  /// Creates a new polynomial with known non-zero prefix lengths for each half.
  /// For a polynomial of size 2n: Z[i]=0 for lo_eff <= i < n, Z[n+i]=0 for hi_eff <= i < n.
  pub fn new_with_halves(Z: Vec<T>, lo_eff: usize, hi_eff: usize) -> Self {
    MultilinearPolynomial { Z, lo_eff, hi_eff }
  }

  /// Returns the effective number of pairs to iterate in bind/eval: max(lo, hi) clamped to n.
  #[inline(always)]
  pub fn eff_pairs(&self) -> usize {
    let n = self.Z.len() / 2;
    let lo = self.lo_eff.min(n);
    let hi = self.hi_eff.min(n);
    lo.max(hi)
  }

  /// Consume and return the inner Vec (preserving capacity for reuse).
  pub fn into_vec(self) -> Vec<T> {
    self.Z
  }
}

impl<T: Field> MultilinearPolynomial<T> {
  /// Binds the polynomial's top variable using the given scalar.
  ///
  /// Exploits zero-structure: when hi half is all zero or sparse, uses
  /// cheaper operations (scale by (1-r) instead of sub+mul+add).
  #[inline(always)]

  /// This operation modifies the polynomial in-place.
  ///
  /// Variable order is MSB-first: the "top" variable is the most-significant
  /// Boolean coordinate in the evaluation table. Equivalently, the table is
  /// laid out as `[p(0, ...), p(1, ...)]`, so binding the top variable combines
  /// the left half and right half of `Z`.
  ///
  /// Formula: `new[i] = old[i] + r * (old[i + n] - old[i])`.
  pub fn bind_poly_var_top(&mut self, r: &T) {
    assert!(
      self.Z.len() >= 2,
      "Vector Z must have at least two elements to bind the top variable."
    );

    let n = self.Z.len() / 2;
    let lo = self.lo_eff.min(n);
    let hi = self.hi_eff.min(n);
    let eff = lo.max(hi);

    if hi == 0 {
      // Hi half is all zero: result = left[i] * (1-r)
      let one_minus_r = T::ONE - *r;
      if rayon::current_num_threads() <= 1 {
        for a in self.Z[..lo].iter_mut() {
          *a *= one_minus_r;
        }
      } else {
        self.Z[..lo].par_iter_mut().for_each(|a| *a *= one_minus_r);
      }
    } else if hi <= lo {
      // Both halves have non-zeros but hi is shorter
      let (left, right) = self.Z.split_at_mut(n);
      if rayon::current_num_threads() <= 1 {
        for i in 0..hi {
          left[i] += *r * (right[i] - left[i]);
        }
        let one_minus_r = T::ONE - *r;
        for a in left[hi..lo].iter_mut() {
          *a *= one_minus_r;
        }
      } else {
        let r_val = *r;
        let one_minus_r = T::ONE - r_val;
        left[..lo].par_iter_mut().enumerate().for_each(|(i, a)| {
          if i < hi {
            *a += r_val * (right[i] - *a);
          } else {
            *a *= one_minus_r;
          }
        });
      }
    } else {
      // hi > lo: lo half runs out first
      let (left, right) = self.Z.split_at_mut(n);
      if rayon::current_num_threads() <= 1 {
        for i in 0..lo {
          left[i] += *r * (right[i] - left[i]);
        }
        for i in lo..hi {
          left[i] = *r * right[i];
        }
      } else {
        let r_val = *r;
        left[..hi].par_iter_mut().enumerate().for_each(|(i, a)| {
          if i < lo {
            *a += r_val * (right[i] - *a);
          } else {
            *a = r_val * right[i];
          }
        });
      }
    }

    self.Z.truncate(n);

    self.lo_eff = eff.min(n / 2);
    self.hi_eff = eff.saturating_sub(n / 2);
  }
}

impl<T> Index<usize> for MultilinearPolynomial<T> {
  type Output = T;

  #[inline(always)]
  fn index(&self, _index: usize) -> &T {
    &(self.Z[_index])
  }
}

/// Sparse multilinear polynomial, which means the $Z(\cdot)$ is zero at most points.
/// In our context, sparse polynomials are non-zeros over the hypercube at locations that map to "small" integers
/// We exploit this property to implement a time-optimal algorithm
pub(crate) struct SparsePolynomial<Scalar: PrimeField> {
  num_vars: usize,
  Z: Vec<Scalar>,
}

impl<Scalar: PrimeField> SparsePolynomial<Scalar> {
  pub fn new(num_vars: usize, Z: Vec<Scalar>) -> Self {
    SparsePolynomial { num_vars, Z }
  }

  // a time-optimal algorithm to evaluate sparse polynomials
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(self.num_vars, r.len());

    let num_vars_z = self.Z.len().next_power_of_two().log_2();
    let chis = EqPolynomial::evals_from_points(&r[self.num_vars - 1 - num_vars_z..]);
    let eval_partial: Scalar = self
      .Z
      .iter()
      .zip(chis.iter())
      .map(|(z, chi)| *z * *chi)
      .sum();

    let common = (0..self.num_vars - 1 - num_vars_z)
      .map(|i| Scalar::ONE - r[i])
      .product::<Scalar>();

    common * eval_partial
  }
}

#[cfg(test)]
mod tests {
  use rand_core::{CryptoRng, OsRng, RngCore};

  use super::*;
  use crate::provider::pasta::pallas;
  use ff::Field;
  use pallas::Scalar;

  /// Evaluates the polynomial at the given point.
  /// Returns Z(r) in O(n) time.
  ///
  /// The point must have a value for each variable.
  pub fn evaluate<S: PrimeField>(poly: &MultilinearPolynomial<S>, r: &[S]) -> S {
    // r must have a value for each variable
    let chis = EqPolynomial::evals_from_points(r);

    zip_with!(
      (chis.into_par_iter(), poly.Z.par_iter()),
      |chi_i, Z_i| chi_i * Z_i
    )
    .sum()
  }

  /// Evaluates the polynomial with the given evaluations and point.
  pub fn evaluate_with<S: PrimeField>(Z: &[S], r: &[S]) -> S {
    zip_with!(
      (
        EqPolynomial::evals_from_points(r).into_par_iter(),
        Z.par_iter()
      ),
      |a, b| a * b
    )
    .sum()
  }

  fn test_multilinear_polynomial_with<F: PrimeField>() {
    // Let the polynomial has 3 variables, p(x_1, x_2, x_3) = (x_1 + x_2) * x_3
    // Evaluations of the polynomial at boolean cube are [0, 0, 0, 1, 0, 1, 0, 2].

    let TWO = F::from(2);

    let Z = vec![
      F::ZERO,
      F::ZERO,
      F::ZERO,
      F::ONE,
      F::ZERO,
      F::ONE,
      F::ZERO,
      TWO,
    ];
    let m_poly = MultilinearPolynomial::<F>::new(Z.clone());

    let x = vec![F::ONE, F::ONE, F::ONE];
    assert_eq!(evaluate(&m_poly, x.as_slice()), TWO);

    let y = evaluate_with(Z.as_slice(), x.as_slice());
    assert_eq!(y, TWO);
  }

  fn test_sparse_polynomial_with<F: PrimeField>() {
    // Let the polynomial have 4 variables, but is non-zero at only 3 locations (out of 2^4 = 16) over the hypercube
    let mut Z = vec![F::ONE, F::ONE, F::from(2)];
    let m_poly = SparsePolynomial::<F>::new(4, Z.clone());

    Z.resize(16, F::ZERO); // append with zeros to make it a dense polynomial
    let m_poly_dense = MultilinearPolynomial::new(Z);

    // evaluation point
    let x = vec![F::from(5), F::from(8), F::from(5), F::from(3)];

    // check evaluations
    assert_eq!(
      m_poly.evaluate(x.as_slice()),
      evaluate(&m_poly_dense, x.as_slice())
    );
  }

  #[test]
  fn test_multilinear_polynomial() {
    test_multilinear_polynomial_with::<pallas::Scalar>();
  }

  #[test]
  fn test_sparse_polynomial() {
    test_sparse_polynomial_with::<pallas::Scalar>();
  }

  fn test_evaluation_with<F: PrimeField>() {
    let num_evals = 4;
    let mut evals: Vec<F> = Vec::with_capacity(num_evals);
    for _ in 0..num_evals {
      evals.push(F::from(8));
    }
    let dense_poly: MultilinearPolynomial<F> = MultilinearPolynomial::new(evals.clone());

    // Evaluate at 3:
    // (0, 0) = 1
    // (0, 1) = 1
    // (1, 0) = 1
    // (1, 1) = 1
    // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
    // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
    // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
    assert_eq!(
      evaluate(&dense_poly, vec![F::from(3), F::from(4)].as_slice()),
      F::from(8)
    );
  }

  #[test]
  fn test_evaluation() {
    test_evaluation_with::<pallas::Scalar>();
  }

  /// Returns a random ML polynomial
  fn random<R: RngCore + CryptoRng, S: PrimeField>(
    num_vars: usize,
    mut rng: &mut R,
  ) -> MultilinearPolynomial<S> {
    MultilinearPolynomial::new(
      std::iter::from_fn(|| Some(S::random(&mut rng)))
        .take(1 << num_vars)
        .collect(),
    )
  }

  /// This binds the variables of a multilinear polynomial to a provided sequence
  /// of values.
  ///
  /// Assuming `bind_poly_var_top` defines the "top" variable of the polynomial,
  /// this aims to test whether variables should be provided to the
  /// `evaluate` function in topmost-first (big endian) of topmost-last (lower endian)
  /// order.
  fn bind_sequence<F: PrimeField>(
    poly: &MultilinearPolynomial<F>,
    values: &[F],
  ) -> MultilinearPolynomial<F> {
    // Assert that the size of the polynomial being evaluated is a power of 2 greater than (1 << values.len())
    assert!(poly.Z.len().is_power_of_two());
    assert!(poly.Z.len() >= 1 << values.len());

    let mut tmp = poly.clone();
    for v in values.iter() {
      tmp.bind_poly_var_top(v);
    }
    tmp
  }

  fn bind_and_evaluate_with<F: PrimeField>() {
    for _ in 0..50 {
      // Initialize a random polynomial
      let n = 7;
      let poly = random(n, &mut OsRng);

      // draw a random point
      let pt: Vec<_> = std::iter::from_fn(|| Some(F::random(&mut OsRng)))
        .take(n)
        .collect();
      // this shows the order in which coordinates are evaluated
      assert_eq!(evaluate(&poly, &pt), bind_sequence(&poly, &pt).Z[0])
    }
  }

  #[test]
  fn test_bind_and_evaluate() {
    bind_and_evaluate_with::<pallas::Scalar>();
  }

  /// Explicit check that bind_poly_var_top matches manual linear interpolation on the MSB.
  #[test]
  fn test_bind_matches_direct_evaluation_explicit() {
    // ℓ=3, poly[i] = i^2 + 1
    let l = 3;
    let size = 1 << l;
    let vals: Vec<Scalar> = (0..size)
      .map(|i| Scalar::from((i * i + 1) as u64))
      .collect();
    let mut poly = MultilinearPolynomial::new(vals.clone());

    let r = Scalar::from(7u64);
    poly.bind_poly_var_top(&r);
    assert_eq!(poly.Z.len(), size / 2);

    // Bound variable is the MSB: new[j] = (1-r)*vals[j] + r*vals[j+4]
    for j in 0..(size / 2) {
      let expected = (Scalar::ONE - r) * vals[j] + r * vals[j + size / 2];
      assert_eq!(poly.Z[j], expected, "Mismatch at j={}", j);
    }
  }

  /// Ensure "top" refers to the MSB (high-order variable), not the LSB.
  #[test]
  fn test_bind_top_is_msb_not_lsb() {
    // ℓ=2, values encode (x0,x1) with x0 as MSB: [p(0,0), p(0,1), p(1,0), p(1,1)]
    let vals = vec![
      Scalar::from(1u64), // (0,0)
      Scalar::from(2u64), // (0,1)
      Scalar::from(3u64), // (1,0)
      Scalar::from(4u64), // (1,1)
    ];
    let mut poly = MultilinearPolynomial::new(vals.clone());
    let r = Scalar::from(5u64);

    poly.bind_poly_var_top(&r);
    assert_eq!(poly.Z.len(), 2);

    // Expected with MSB binding:
    // new[0] = (1-r)*p(0,0) + r*p(1,0) = (1-5)*1 + 5*3 = 11
    // new[1] = (1-r)*p(0,1) + r*p(1,1) = (1-5)*2 + 5*4 = 12
    assert_eq!(poly.Z[0], Scalar::from(11u64));
    assert_eq!(poly.Z[1], Scalar::from(12u64));

    // If LSB were bound, results would differ (6 and 8 respectively).
    assert_ne!(poly.Z[0], Scalar::from(6u64));
    assert_ne!(poly.Z[1], Scalar::from(8u64));
  }

  #[test]
  #[should_panic(expected = "Vector Z must be non-empty and its length must be a power of two.")]
  fn test_new_rejects_empty() {
    let _: MultilinearPolynomial<Scalar> = MultilinearPolynomial::new(vec![]);
  }

  #[test]
  #[should_panic(expected = "Vector Z must be non-empty and its length must be a power of two.")]
  fn test_new_rejects_non_power_of_two_length() {
    let _ = MultilinearPolynomial::new(vec![Scalar::ONE, Scalar::ZERO, Scalar::ONE]);
  }
}
