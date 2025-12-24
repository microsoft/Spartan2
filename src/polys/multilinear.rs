// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Main components:
//! - `MultilinearPolynomial`: Dense representation of multilinear polynomials, represented by evaluations over all possible binary inputs.
//! - `SparsePolynomial`: Efficient representation of sparse multilinear polynomials, storing only non-zero evaluations.

use crate::{math::Math, polys::eq::EqPolynomial, small_field::SmallValueField, zip_with_for_each};
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
}

impl<T> MultilinearPolynomial<T> {
  /// Creates a new `MultilinearPolynomial` from the given evaluations.
  ///
  /// # Panics
  /// The number of evaluations must be a power of two.
  pub fn new(Z: Vec<T>) -> Self {
    MultilinearPolynomial { Z }
  }
}

impl<T: Field> MultilinearPolynomial<T> {
  /// Binds the polynomial's top variable using the given scalar.
  ///
  /// This operation modifies the polynomial in-place.
  /// Formula: new[i] = old[i] + r * (old[i + n] - old[i])
  pub fn bind_poly_var_top(&mut self, r: &T) {
    assert!(
      self.Z.len() >= 2,
      "Vector Z must have at least two elements to bind the top variable."
    );

    let n = self.Z.len() / 2;

    let (left, right) = self.Z.split_at_mut(n);

    zip_with_for_each!((left.par_iter_mut(), right.par_iter()), |a, b| {
      // Field types implement Copy, so no cloning needed
      *a += *r * (*b - *a);
    });

    self.Z.truncate(n);
  }

  /// Binds the polynomial's top variables using the given scalars.
  pub fn bind_with(poly: &[T], L: &[T], r_len: usize) -> Vec<T> {
    assert_eq!(
      poly.len(),
      L.len() * r_len,
      "poly length ({}) must equal L.len() * r_len ({} * {}) = {}",
      poly.len(),
      L.len(),
      r_len,
      L.len() * r_len
    );

    (0..r_len)
      .into_par_iter()
      .map(|i| {
        let mut acc = T::ZERO;
        for j in 0..L.len() {
          // row-major: index = j * r_len + i
          acc += L[j] * poly[j * r_len + i];
        }
        acc
      })
      .collect()
  }
}

impl<T: Copy> MultilinearPolynomial<T> {
  /// Gathers prefix evaluations p(b, suffix) for all binary prefixes b ∈ {0,1}^ℓ₀.
  ///
  /// For a polynomial with ℓ variables, this extracts a strided slice where:
  /// - First ℓ₀ variables form the "prefix" (high bits)
  /// - Remaining ℓ-ℓ₀ variables form the "suffix" (low bits)
  ///
  /// Index layout: `index = (prefix << suffix_vars) | suffix`
  ///
  /// # Arguments
  /// * `l0` - Number of prefix variables
  /// * `suffix` - Fixed suffix value in range [0, 2^{ℓ-ℓ₀})
  ///
  /// # Returns
  /// MultilinearPolynomial of size 2^ℓ₀ where result[prefix] = self[(prefix << suffix_vars) | suffix]
  ///
  /// # Example
  /// For ℓ=4, ℓ₀=2, suffix=1:
  /// - Returns polynomial with evals [self[1], self[5], self[9], self[13]]
  /// - Indices: 0b0001, 0b0101, 0b1001, 0b1101 (prefix varies, suffix=01 fixed)
  // Allow dead code until Chunk 7 (build_accumulators) uses this method
  #[allow(dead_code)]
  pub fn gather_prefix_evals(&self, l0: usize, suffix: usize) -> Self {
    let l = self.Z.len().trailing_zeros() as usize;
    debug_assert_eq!(self.Z.len(), 1 << l, "poly size must be power of 2");

    let suffix_vars = l - l0;
    let prefix_size = 1 << l0;

    debug_assert!(suffix < (1 << suffix_vars), "suffix out of range");

    let mut Z = Vec::with_capacity(prefix_size);
    for prefix in 0..prefix_size {
      let idx = (prefix << suffix_vars) | suffix;
      Z.push(self.Z[idx]); // Copy, no clone needed
    }

    MultilinearPolynomial::new(Z)
  }
}

// ============================================================================
// Small-value polynomial operations (MultilinearPolynomial<i32>)
// ============================================================================

impl MultilinearPolynomial<i32> {
  /// Try to create from a field-element polynomial.
  /// Returns None if any value doesn't fit in i32.
  pub fn try_from_field<F: SmallValueField<SmallValue = i32>>(
    poly: &MultilinearPolynomial<F>,
  ) -> Option<Self> {
    let evals: Option<Vec<i32>> = poly.Z.iter().map(|f| F::try_field_to_small(f)).collect();
    evals.map(Self::new)
  }

  /// Get the number of variables.
  pub fn num_vars(&self) -> usize {
    self.Z.len().trailing_zeros() as usize
  }

  /// Convert to field-element polynomial.
  pub fn to_field<F: SmallValueField<SmallValue = i32>>(&self) -> MultilinearPolynomial<F> {
    MultilinearPolynomial::new(self.Z.iter().map(|&s| F::small_to_field(s)).collect())
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
  use super::*;
  use crate::{provider::pasta::pallas, zip_with};
  use ff::Field;
  use rand_core::{CryptoRng, OsRng, RngCore};
  use pallas::Scalar;

  /// Evaluates the polynomial at the given point.
  /// Returns Z(r) in O(n) time.
  ///
  /// The point must have a value for each variable.
  pub fn evaluate<Scalar: PrimeField>(
    poly: &MultilinearPolynomial<Scalar>,
    r: &[Scalar],
  ) -> Scalar {
    // r must have a value for each variable
    let chis = EqPolynomial::evals_from_points(r);

    zip_with!(
      (chis.into_par_iter(), poly.Z.par_iter()),
      |chi_i, Z_i| chi_i * Z_i
    )
    .sum()
  }

  /// Evaluates the polynomial with the given evaluations and point.
  pub fn evaluate_with<Scalar: PrimeField>(Z: &[Scalar], r: &[Scalar]) -> Scalar {
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
    let vals: Vec<Scalar> = (0..size).map(|i| Scalar::from((i * i + 1) as u64)).collect();
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

  // === gather_prefix_evals tests ===

  #[test]
  fn test_gather_prefix_evals_all_suffixes() {
    // ℓ=4, ℓ₀=2: 4 variables, first 2 are prefix
    // poly[i] = i, so value equals index for easy verification
    let l = 4;
    let l0 = 2;
    let size = 1 << l; // 16

    let evals: Vec<pallas::Scalar> = (0..size).map(|i| pallas::Scalar::from(i as u64)).collect();
    let poly = MultilinearPolynomial::new(evals);

    let num_prefix = 1 << l0; // 4 prefix combinations
    let suffix_vars = l - l0; // 2 suffix variables
    let num_suffix = 1 << suffix_vars; // 4 suffix combinations

    for suffix in 0..num_suffix {
      let gathered = poly.gather_prefix_evals(l0, suffix);
      assert_eq!(gathered.Z.len(), num_prefix);

      for prefix in 0..num_prefix {
        let expected_idx = (prefix << suffix_vars) | suffix;
        let expected = pallas::Scalar::from(expected_idx as u64);
        assert_eq!(
          gathered[prefix], expected,
          "Mismatch at suffix={}, prefix={}",
          suffix, prefix
        );
      }
    }
  }

  #[test]
  fn test_gather_prefix_l0_equals_l() {
    // ℓ₀ = ℓ: no suffix variables, suffix must be 0
    let l0 = 3;
    let evals: Vec<pallas::Scalar> = (0..8).map(|i| pallas::Scalar::from(i as u64)).collect();
    let poly = MultilinearPolynomial::new(evals);

    let gathered = poly.gather_prefix_evals(l0, 0);

    // Should return entire polynomial
    assert_eq!(gathered.Z.len(), 8);
    for i in 0..8 {
      assert_eq!(gathered[i], pallas::Scalar::from(i as u64));
    }
  }

  #[test]
  fn test_gather_prefix_l0_equals_1() {
    // ℓ₀ = 1: single prefix bit
    let l0 = 1;
    let evals: Vec<pallas::Scalar> = (0..16).map(|i| pallas::Scalar::from(i as u64)).collect();
    let poly = MultilinearPolynomial::new(evals);

    // suffix = 5 (binary 101): should get indices 5, 13
    let gathered = poly.gather_prefix_evals(l0, 5);
    assert_eq!(gathered.Z.len(), 2);
    assert_eq!(gathered[0], pallas::Scalar::from(5u64)); // prefix=0: 0*8 + 5 = 5
    assert_eq!(gathered[1], pallas::Scalar::from(13u64)); // prefix=1: 1*8 + 5 = 13
  }

  #[test]
  fn test_gather_then_extend_preserves_binary_points() {
    use crate::lagrange::{LagrangeEvaluatedMultilinearPolynomial, UdTuple};
    use ff::Field;

    let l = 4;
    let l0 = 2;

    // Random polynomial
    let evals: Vec<pallas::Scalar> = (0..(1 << l))
      .map(|_| pallas::Scalar::random(&mut OsRng))
      .collect();
    let poly = MultilinearPolynomial::new(evals);

    let suffix_vars = l - l0;
    let num_suffix = 1 << suffix_vars;

    for suffix in 0..num_suffix {
      let gathered = poly.gather_prefix_evals(l0, suffix);

      // Extend to Lagrange domain
      let extended = LagrangeEvaluatedMultilinearPolynomial::<_, 3>::from_multilinear(&gathered);

      // Verify: at binary points, extended values match original poly values
      for prefix_bits in 0..(1 << l0) {
        let original_idx = (prefix_bits << suffix_vars) | suffix;
        let original_val = poly[original_idx];

        // Convert binary prefix to U_D^ℓ₀ tuple
        let tuple = UdTuple::<3>::from_binary(prefix_bits, l0);

        assert_eq!(
          extended.get_by_domain(&tuple),
          original_val,
          "Mismatch at suffix={}, prefix_bits={}",
          suffix,
          prefix_bits
        );
      }
    }
  }
}
