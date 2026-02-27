// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! `EqPolynomial`: Represents multilinear extension of equality polynomials, evaluated based on binary input values.
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

    if rayon::current_num_threads() <= 1 {
      for r in r.iter().rev() {
        let (evals_left, evals_right) = evals.split_at_mut(size);
        let (evals_right, _) = evals_right.split_at_mut(size);

        for (x, y) in evals_left.iter_mut().zip(evals_right.iter_mut()) {
          *y = *x * r;
          *x -= &*y;
        }

        size *= 2;
      }
    } else {
      for r in r.iter().rev() {
        let (evals_left, evals_right) = evals.split_at_mut(size);
        let (evals_right, _) = evals_right.split_at_mut(size);

        zip_with_for_each!(par_iter_mut, (evals_left, evals_right), |x, y| {
          *y = *x * r;
          *x -= &*y;
        });

        size *= 2;
      }
    }

    evals
  }
  /// Like evals_from_points but reuses a pre-allocated buffer.
  /// Builds the evaluation table incrementally via push (no zeroing needed).
  #[inline(always)]
  pub fn evals_from_points_into(r: &[Scalar], out: &mut Vec<Scalar>) {
    let ell = r.len();
    let n = (2_usize).pow(ell as u32);
    out.clear();
    out.reserve(n);
    out.push(Scalar::ONE);

    for r_val in r.iter().rev() {
      let sz = out.len();
      // Push new right-half values: out[sz+i] = out[i] * r
      for i in 0..sz {
        let val = out[i] * r_val;
        out.push(val);
      }
      // Update left-half: out[i] -= out[sz+i] using split_at_mut
      let (left, right) = out.split_at_mut(sz);
      for i in 0..sz {
        left[i] -= right[i];
      }
    }
    debug_assert_eq!(out.len(), n);
  }
}

impl<Scalar: PrimeField> FromIterator<Scalar> for EqPolynomial<Scalar> {
  fn from_iter<I: IntoIterator<Item = Scalar>>(iter: I) -> Self {
    let r: Vec<_> = iter.into_iter().collect();
    EqPolynomial { r }
  }
}

/// Build a full eq polynomial pyramid from tau values.
///
/// Given taus = [τ₀, τ₁, ..., τ_{n-1}], builds a pyramid where:
/// - Layer 0: [1]
/// - Layer 1: [1-τ_{n-1}, τ_{n-1}]
/// - Layer k: eq([τ_{n-k}, ..., τ_{n-1}], ·), size 2^k
/// - Layer n: eq([τ₀, ..., τ_{n-1}], ·), size 2^n (the full eq table)
///
/// The pyramid has n+1 layers total. This structure matches EqSumCheckInstance's
/// internal representation, enabling pyramid reuse between accumulator building
/// and sumcheck evaluation.
///
/// The taus are processed in reverse order (last tau first) to match the
/// standard multilinear indexing convention where the first tau corresponds
/// to the most significant bit of the index.
///
/// Each layer is stored as [lo_0, lo_1, ..., lo_n, hi_0, hi_1, ..., hi_n]
/// where lo values are multiplied by (1-τ) and hi values by τ.
pub fn build_eq_pyramid<S: PrimeField>(taus: &[S]) -> Vec<Vec<S>> {
  use rayon::prelude::*;

  let n = taus.len();
  let mut pyramid = Vec::with_capacity(n + 1);

  // Layer 0: base case [1]
  pyramid.push(vec![S::ONE]);

  // Build layers 1..n by adding one tau at a time (in reverse order)
  // Uses the same parallel pattern as EqSumCheckInstance::new
  for i in 0..n {
    let tau = taus[n - 1 - i]; // Process taus in reverse
    let prev = &pyramid[i];

    // Build next layer: [lo_0, ..., lo_n, hi_0, ..., hi_n]
    // First, copy prev and extend with hi values (prev * tau)
    let mut next = prev.to_vec();
    next.par_extend(prev.par_iter().map(|v| *v * tau));

    // Then subtract hi from lo: lo = prev - hi = prev * (1 - tau)
    let (first, last) = next.split_at_mut(prev.len());
    first
      .par_iter_mut()
      .zip(last)
      .for_each(|(a, b)| *a -= *b);

    pyramid.push(next);
  }

  pyramid
}

/// Compute suffix eq tables for small-value sumcheck optimization.
///
/// Given τ = (τ₀, τ₁, ..., τ_{l-1}) and `l0` (the number of small-value rounds),
/// computes tables `E_y[i]` = eq(τ[i+1:l0], ·) for i = 0..l0-1.
///
/// These tables allow O(1) lookup of eq suffix products during accumulation,
/// avoiding redundant computation in the inner loop.
///
/// Returns empty vec if l0 == 0 (optimization disabled).
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
}
