//! Main components:
//! - `MultilinearPolynomial`: Dense representation of multilinear polynomials, represented by evaluations over all possible binary inputs.
//! - `SparsePolynomial`: Efficient representation of sparse multilinear polynomials, storing only non-zero evaluations.

use crate::{math::Math, polys::eq::EqPolynomial, start_span, zip_with, zip_with_for_each};
use core::ops::Index;
use ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{info, info_span};

/// A multilinear extension of a polynomial $Z(\cdot)$, denote it as $\tilde{Z}(x_1, ..., x_m)$
/// where the degree of each variable is at most one.
///
/// This is the dense representation of a multilinear poynomial.
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultilinearPolynomial<Scalar: PrimeField> {
  pub(crate) Z: Vec<Scalar>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<Scalar: PrimeField> MultilinearPolynomial<Scalar> {
  /// Creates a new `MultilinearPolynomial` from the given evaluations.
  ///
  /// # Panics
  /// The number of evaluations must be a power of two.
  pub fn new(Z: Vec<Scalar>) -> Self {
    MultilinearPolynomial { Z }
  }

  /// Binds the polynomial's top variable using the given scalar.
  ///
  /// This operation modifies the polynomial in-place.
  pub fn bind_poly_var_top(&mut self, r: &Scalar) {
    assert!(
      self.Z.len() >= 2,
      "Vector Z must have at least two elements to bind the top variable."
    );

    let n = self.Z.len() / 2;

    let (left, right) = self.Z.split_at_mut(n);

    zip_with_for_each!((left.par_iter_mut(), right.par_iter()), |a, b| {
      *a += *r * (*b - *a);
    });

    self.Z.truncate(n);
  }

  /// binds the polynomial's top variables using the given scalars.
  pub fn bind_with(poly: &[Scalar], L: &[Scalar], R_len: usize) -> Vec<Scalar> {
    // we need to view poly as a matrix of dimensions (L.len(), R_len)
    // we need to take weighted sum of each row with the corresponding L[i]
    // however, some rows may be shorter than R_len, so we need to handle that
    (0..R_len)
      .into_par_iter()
      .map(|i| {
        (0..L.len())
          .map(|j| {
            if (j * R_len + i) >= poly.len() {
              Scalar::ZERO
            } else {
              L[j] * poly[j * R_len + i]
            }
          })
          .sum()
      })
      .collect()
  }

  /// Evaluates the polynomial at the given point.
  /// Returns Z(r) in O(n) time.
  ///
  /// The point must have a value for each variable.
  #[allow(dead_code)]
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    // r must have a value for each variable
    let chis = EqPolynomial::evals_from_points(r);

    zip_with!(
      (chis.into_par_iter(), self.Z.par_iter()),
      |chi_i, Z_i| chi_i * Z_i
    )
    .sum()
  }

  /// Evaluates the polynomial with the given evaluations and point.
  pub fn evaluate_with(Z: &[Scalar], r: &[Scalar]) -> Scalar {
    let (_eval_span, eval_t) =
      start_span!("multilinear_evaluate_with", vars = r.len(), evals = Z.len());

    let result = zip_with!(
      (
        EqPolynomial::evals_from_points(r).into_par_iter(),
        Z.par_iter()
      ),
      |a, b| a * b
    )
    .sum();

    info!(elapsed_ms = %eval_t.elapsed().as_millis(), vars = r.len(), evals = Z.len(), "multilinear_evaluate_with");
    result
  }
}

impl<Scalar: PrimeField> Index<usize> for MultilinearPolynomial<Scalar> {
  type Output = Scalar;

  #[inline(always)]
  fn index(&self, _index: usize) -> &Scalar {
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
      .map(|i| (Scalar::ONE - r[i]))
      .product::<Scalar>();

    common * eval_partial
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;
  use rand_core::{CryptoRng, OsRng, RngCore};

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
    assert_eq!(m_poly.evaluate(x.as_slice()), TWO);

    let y = MultilinearPolynomial::<F>::evaluate_with(Z.as_slice(), x.as_slice());
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
      m_poly_dense.evaluate(x.as_slice())
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
      dense_poly.evaluate(vec![F::from(3), F::from(4)].as_slice()),
      F::from(8)
    );
  }

  #[test]
  fn test_evaluation() {
    test_evaluation_with::<pallas::Scalar>();
  }

  /// Returns a random ML polynomial
  #[allow(clippy::needless_borrows_for_generic_args)]
  fn random<R: RngCore + CryptoRng, Scalar: PrimeField>(
    num_vars: usize,
    mut rng: &mut R,
  ) -> MultilinearPolynomial<Scalar> {
    MultilinearPolynomial::new(
      std::iter::from_fn(|| Some(Scalar::random(&mut rng)))
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
      assert_eq!(poly.evaluate(&pt), bind_sequence(&poly, &pt).Z[0])
    }
  }

  #[test]
  fn test_bind_and_evaluate() {
    bind_and_evaluate_with::<pallas::Scalar>();
  }
}
