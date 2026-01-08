// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! `EqPolynomial`: Represents multilinear extension of equality polynomials, evaluated based on binary input values.
use crate::zip_with_for_each;
use ff::PrimeField;
use rayon::prelude::*;

/// Represents the multilinear extension polynomial (MLE) of the equality polynomial $eq(x,e)$, denoted as $\tilde{eq}(x, e)$.
///
/// The polynomial is defined by the formula:
/// $$
/// \tilde{eq}(x, e) = \prod_{i=1}^m(e_i * x_i + (1 - e_i) * (1 - x_i))
/// $$
///
/// Each element in the vector `r` corresponds to a component $e_i$, representing a bit from the binary representation of an input value $e$.
/// This polynomial evaluates to 1 if every component $x_i$ equals its corresponding $e_i$, and 0 otherwise.
///
/// For instance, for e = 6 (with a binary representation of 0b110), the vector r would be [1, 1, 0].
#[derive(Debug)]
pub struct EqPolynomial<Scalar: PrimeField> {
  r: Vec<Scalar>,
}

impl<Scalar: PrimeField> EqPolynomial<Scalar> {
  /// Creates a new `EqPolynomial` from a vector of Scalars `r`.
  ///
  /// Each Scalar in `r` corresponds to a bit from the binary representation of an input value `e`.
  pub const fn new(r: Vec<Scalar>) -> Self {
    EqPolynomial { r }
  }

  /// Evaluates the `EqPolynomial` at a given point `rx`.
  ///
  /// This function computes the value of the polynomial at the point specified by `rx`.
  /// It expects `rx` to have the same length as the internal vector `r`.
  ///
  /// Panics if `rx` and `r` have different lengths.
  pub fn evaluate(&self, rx: &[Scalar]) -> Scalar {
    assert_eq!(self.r.len(), rx.len());
    (0..rx.len())
      .map(|i| rx[i] * self.r[i] + (Scalar::ONE - rx[i]) * (Scalar::ONE - self.r[i]))
      .fold(Scalar::ONE, |acc, item| acc * item)
  }

  /// Evaluates the `EqPolynomial` at all the `2^|r|` points in its domain.
  ///
  /// Returns a vector of Scalars, each corresponding to the polynomial evaluation at a specific point.
  pub fn evals(&self) -> Vec<Scalar> {
    Self::evals_from_points(&self.r)
  }

  /// Evaluates the `EqPolynomial` from the `2^|r|` points in its domain, without creating an intermediate polynomial
  /// representation.
  ///
  /// Returns a vector of Scalars, each corresponding to the polynomial evaluation at a specific point.
  pub fn evals_from_points(r: &[Scalar]) -> Vec<Scalar> {
    let ell = r.len();
    let mut evals: Vec<Scalar> = vec![Scalar::ZERO; (2_usize).pow(ell as u32)];
    let mut size = 1;
    evals[0] = Scalar::ONE;

    for r in r.iter().rev() {
      let (evals_left, evals_right) = evals.split_at_mut(size);
      let (evals_right, _) = evals_right.split_at_mut(size);

      zip_with_for_each!(par_iter_mut, (evals_left, evals_right), |x, y| {
        *y = *x * r;
        *x -= &*y;
      });

      size *= 2;
    }

    evals
  }
}

impl<Scalar: PrimeField> FromIterator<Scalar> for EqPolynomial<Scalar> {
  fn from_iter<I: IntoIterator<Item = Scalar>>(iter: I) -> Self {
    let r: Vec<_> = iter.into_iter().collect();
    EqPolynomial { r }
  }
}

/// Computes suffix eq-polynomials: E_y[i] = eq(τ[i+1:ℓ₀], y) for all i ∈ [0, ℓ₀).
///
/// This is used for Algorithm 6 (small-value sumcheck) in
/// <https://eprint.iacr.org/2025/1117>.
///
/// Uses a pyramid approach: build from the end (τ[ℓ₀-1]) backwards to τ[1].
/// Each step extends the previous suffix by prepending one more τ value.
///
/// # Arguments
/// * `taus` - τ values for the first ℓ₀ variables (τ[0:ℓ₀])
/// * `l0` - number of small-value rounds
///
/// # Returns
/// Vec of length ℓ₀, where `result[i]` has 2^{ℓ₀-i-1} elements.
/// `result[i][y]` = eq(τ[i+1:ℓ₀], y) for y ∈ {0,1}^{ℓ₀-i-1}
///
/// # Complexity
/// O(2^ℓ₀) total field multiplications (vs O(ℓ₀ · 2^ℓ₀) naive)
// Allow dead code until later chunks use this function
pub fn compute_suffix_eq_pyramid<S: PrimeField>(taus: &[S], l0: usize) -> Vec<Vec<S>> {
  // Handle l0 == 0: no suffix tables needed (small-value optimization disabled)
  if l0 == 0 {
    return Vec::new();
  }

  assert!(taus.len() >= l0, "taus must have at least l0 elements");

  let mut result: Vec<Vec<S>> = vec![vec![]; l0];

  // Base case: E_y[l0-1] = eq([], ·) = [1] (empty suffix)
  result[l0 - 1] = vec![S::ONE];

  // Build backwards: each step prepends one τ value
  // E_y[i] = eq(τ[i+1:l0], ·) is built from E_y[i+1] = eq(τ[i+2:l0], ·)
  // by prepending τ[i+1]
  for i in (0..l0 - 1).rev() {
    let tau = taus[i + 1];
    let prev = &result[i + 1];
    let prev_len = prev.len();

    // New table has 2× the entries (prepending a new variable)
    // For multilinear indexing: first variable is high bit
    // new_idx = new_bit * prev_len + old_idx
    //
    // new_bit = 0: eq factor is (1 - τ)
    // new_bit = 1: eq factor is τ
    //
    // Optimized: use 1 multiplication per element instead of 2
    // hi = v * τ, lo = v - hi = v * (1 - τ)
    let mut next = vec![S::ZERO; prev_len * 2];
    let (lo_half, hi_half) = next.split_at_mut(prev_len);

    for ((lo, hi), v) in lo_half.iter_mut().zip(hi_half.iter_mut()).zip(prev.iter()) {
      *hi = *v * tau;
      *lo = *v - *hi;
    }

    result[i] = next;
  }

  result
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;
  use ff::Field;

  fn test_eq_polynomial_with<F: PrimeField>() {
    let eq_poly = EqPolynomial::<F>::new(vec![F::ONE, F::ZERO, F::ONE]);
    let y = eq_poly.evaluate(vec![F::ONE, F::ONE, F::ONE].as_slice());
    assert_eq!(y, F::ZERO);

    let y = eq_poly.evaluate(vec![F::ONE, F::ZERO, F::ONE].as_slice());
    assert_eq!(y, F::ONE);

    let eval_list = eq_poly.evals();
    for (i, &coeff) in eval_list.iter().enumerate().take((2_usize).pow(3)) {
      if i == 5 {
        assert_eq!(coeff, F::ONE);
      } else {
        assert_eq!(coeff, F::ZERO);
      }
    }
  }

  #[test]
  fn test_eq_polynomial() {
    test_eq_polynomial_with::<pallas::Scalar>();
  }

  // === Suffix Eq Pyramid tests ===

  #[test]
  fn test_suffix_pyramid_l0_zero() {
    let taus: Vec<pallas::Scalar> = vec![pallas::Scalar::from(1), pallas::Scalar::from(2)];
    let pyramid = compute_suffix_eq_pyramid(&taus, 0);
    assert!(pyramid.is_empty());
  }

  #[test]
  fn test_suffix_pyramid_sizes() {
    let l0 = 4;
    let taus: Vec<pallas::Scalar> = (0..l0)
      .map(|i| pallas::Scalar::from(i as u64 + 2))
      .collect();
    let pyramid = compute_suffix_eq_pyramid(&taus, l0);

    assert_eq!(pyramid.len(), l0);
    #[allow(clippy::needless_range_loop)]
    for i in 0..l0 {
      let expected_size = 1 << (l0 - i - 1);
      assert_eq!(
        pyramid[i].len(),
        expected_size,
        "E_y[{}] should have size {}",
        i,
        expected_size
      );
    }
  }

  #[test]
  fn test_suffix_pyramid_base_case() {
    use ff::Field;
    let l0 = 3;
    let taus: Vec<pallas::Scalar> = (0..l0)
      .map(|_| pallas::Scalar::random(&mut rand_core::OsRng))
      .collect();
    let pyramid = compute_suffix_eq_pyramid(&taus, l0);

    // E_y[l0-1] = [1] (empty suffix)
    assert_eq!(pyramid[l0 - 1].len(), 1);
    assert_eq!(pyramid[l0 - 1][0], pallas::Scalar::ONE);
  }

  #[test]
  fn test_suffix_pyramid_single_tau() {
    use ff::Field;
    let l0 = 3;
    let taus: Vec<pallas::Scalar> = (0..l0)
      .map(|_| pallas::Scalar::random(&mut rand_core::OsRng))
      .collect();
    let pyramid = compute_suffix_eq_pyramid(&taus, l0);

    // E_y[l0-2] = eq([τ_{l0-1}], ·) = [1-τ, τ]
    let tau_last = taus[l0 - 1];
    assert_eq!(pyramid[l0 - 2].len(), 2);
    assert_eq!(
      pyramid[l0 - 2][0],
      pallas::Scalar::ONE - tau_last,
      "eq(τ, 0) = 1-τ"
    );
    assert_eq!(pyramid[l0 - 2][1], tau_last, "eq(τ, 1) = τ");
  }

  #[test]
  fn test_suffix_pyramid_matches_naive() {
    use ff::Field;
    // Verify pyramid matches independent computation
    let l0 = 4;
    let taus: Vec<pallas::Scalar> = (0..l0)
      .map(|_| pallas::Scalar::random(&mut rand_core::OsRng))
      .collect();

    let pyramid = compute_suffix_eq_pyramid(&taus, l0);

    for i in 0..l0 {
      let naive = if i + 1 >= l0 {
        vec![pallas::Scalar::ONE]
      } else {
        EqPolynomial::evals_from_points(&taus[i + 1..l0])
      };

      assert_eq!(pyramid[i].len(), naive.len(), "Size mismatch at i={}", i);

      for (j, (&p, &n)) in pyramid[i].iter().zip(naive.iter()).enumerate() {
        assert_eq!(p, n, "Value mismatch at E_y[{}][{}]", i, j);
      }
    }
  }

  #[test]
  fn test_suffix_pyramid_indexing() {
    use ff::Field;
    // Verify index semantics: pyramid[i][y] = eq(τ[i+1:l0], y)
    let l0 = 3;
    let tau1 = pallas::Scalar::from(5);
    let tau2 = pallas::Scalar::from(7);
    let tau0 = pallas::Scalar::from(3); // Not used in any E_y suffix
    let taus = vec![tau0, tau1, tau2];

    let pyramid = compute_suffix_eq_pyramid(&taus, l0);

    // E_y[0] = eq([τ₁, τ₂], y) for y ∈ {0,1}²
    // Index 0 = (0,0): eq = (1-τ₁)(1-τ₂)
    // Index 1 = (0,1): eq = (1-τ₁)(τ₂)
    // Index 2 = (1,0): eq = (τ₁)(1-τ₂)
    // Index 3 = (1,1): eq = (τ₁)(τ₂)
    assert_eq!(
      pyramid[0][0],
      (pallas::Scalar::ONE - tau1) * (pallas::Scalar::ONE - tau2)
    );
    assert_eq!(pyramid[0][1], (pallas::Scalar::ONE - tau1) * tau2);
    assert_eq!(pyramid[0][2], tau1 * (pallas::Scalar::ONE - tau2));
    assert_eq!(pyramid[0][3], tau1 * tau2);

    // E_y[1] = eq([τ₂], y) for y ∈ {0,1}
    assert_eq!(pyramid[1][0], pallas::Scalar::ONE - tau2);
    assert_eq!(pyramid[1][1], tau2);

    // E_y[2] = eq([], ·) = [1]
    assert_eq!(pyramid[2][0], pallas::Scalar::ONE);
  }

  /// Ensure evals_from_points uses MSB-first indexing and matches direct product formula.
  #[test]
  #[allow(clippy::needless_range_loop)]
  fn test_eq_table_index_convention() {
    let r = vec![
      pallas::Scalar::from(2u64),
      pallas::Scalar::from(3u64),
      pallas::Scalar::from(5u64),
    ];
    let m = r.len();
    let evals = EqPolynomial::evals_from_points(&r);
    assert_eq!(evals.len(), 1 << m);

    for idx in 0..(1usize << m) {
      let mut expected = pallas::Scalar::ONE;
      for j in 0..m {
        // MSB-first: bit j of idx (from left) corresponds to variable j
        let bit = (idx >> (m - 1 - j)) & 1;
        expected *= if bit == 1 {
          r[j]
        } else {
          pallas::Scalar::ONE - r[j]
        };
      }
      assert_eq!(
        evals[idx], expected,
        "Mismatch at idx {}: got {:?}, expected {:?}",
        idx, evals[idx], expected
      );
    }
  }

  /// Spot-check specific values to catch bit-order flips.
  #[test]
  fn test_eq_table_specific_values() {
    // m=2, r = [2,3]; MSB-first convention
    let r = vec![pallas::Scalar::from(2u64), pallas::Scalar::from(3u64)];
    let evals = EqPolynomial::evals_from_points(&r);

    assert_eq!(evals[0], pallas::Scalar::from(2u64)); // (1-2)(1-3) = 2
    assert_eq!(evals[1], -pallas::Scalar::from(3u64)); // (1-2)*3 = -3
    assert_eq!(evals[2], -pallas::Scalar::from(4u64)); // 2*(1-3) = -4
    assert_eq!(evals[3], pallas::Scalar::from(6u64)); // 2*3 = 6
  }
}
