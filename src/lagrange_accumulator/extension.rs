// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Procedure 6: Extension of multilinear polynomial evaluations from
//! boolean hypercube {0,1}^ℓ to Lagrange domain U_D^ℓ.

#[cfg(test)]
use super::domain::LagrangeIndex;
use std::ops::{Add, Sub};

#[cfg(test)]
use crate::polys::multilinear::MultilinearPolynomial;
#[cfg(test)]
use crate::small_field::SmallValueField;
#[cfg(test)]
use ff::PrimeField;

/// Multilinear polynomial evaluations extended to the Lagrange domain U_D^ℓ.
///
/// Stores evaluations at all (D+1)^num_vars points of the extended domain,
/// indexed by `LagrangeIndex<D>`.
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
  ///
  /// Returns the number of valid elements in `buf_curr` (= (D+1)^num_vars).
  /// After this call, `buf_curr[..result]` contains the extended evaluations.
  ///
  /// This is the zero-allocation version - caller reads results directly from `buf_curr`.
  ///
  /// # Arguments
  /// * `input` - Boolean hypercube evaluations (read-only slice, length must be power of 2)
  /// * `buf_curr` - Result buffer, will contain extended evaluations after call
  /// * `buf_scratch` - Scratch buffer used during iterative extension
  ///
  /// Both buffers will be resized if needed to (D+1)^num_vars.
  pub fn extend_in_place(input: &[T], buf_curr: &mut Vec<T>, buf_scratch: &mut Vec<T>) -> usize {
    let num_vars = input.len().trailing_zeros() as usize;
    debug_assert_eq!(input.len(), 1 << num_vars, "Input size must be power of 2");

    if num_vars == 0 {
      // Single element: copy to buf_curr and return
      if buf_curr.is_empty() {
        buf_curr.push(T::default());
      }
      buf_curr[0] = input[0];
      return 1;
    }

    let final_size = Self::BASE.pow(num_vars as u32);

    // Ensure buffers are large enough
    if buf_curr.len() < final_size {
      buf_curr.resize(final_size, T::default());
    }
    if buf_scratch.len() < final_size {
      buf_scratch.resize(final_size, T::default());
    }

    // Copy input into buf_curr to start
    buf_curr[..input.len()].copy_from_slice(input);

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

      // Alternate between buffers each iteration
      let (src, dst) = if j % 2 == 1 {
        (&buf_curr[..], &mut buf_scratch[..])
      } else {
        (&buf_scratch[..], &mut buf_curr[..])
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

    // Ensure result ends up in buf_curr (swap if result is currently in buf_scratch)
    if num_vars % 2 == 1 {
      std::mem::swap(buf_curr, buf_scratch);
    }
    final_size
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
  pub fn get_by_domain(&self, tuple: &LagrangeIndex<D>) -> T {
    self.evals[tuple.to_flat_index()]
  }

  /// Number of variables
  pub fn num_vars(&self) -> usize {
    self.num_vars
  }

  /// Convert flat index to domain tuple
  pub fn to_domain_tuple(&self, flat_idx: usize) -> LagrangeIndex<D> {
    LagrangeIndex::from_flat_index(flat_idx, self.num_vars)
  }
}

/// Test-only: Create from a MultilinearPolynomial.
#[cfg(test)]
#[allow(missing_docs)]
impl<F: PrimeField, const D: usize> LagrangeEvaluatedMultilinearPolynomial<F, D> {
  pub fn from_multilinear(poly: &MultilinearPolynomial<F>) -> Self {
    Self::from_boolean_evals(&poly.Z)
  }

  pub fn from_evals(evals: Vec<F>, num_vars: usize) -> Self {
    debug_assert_eq!(evals.len(), (D + 1).pow(num_vars as u32));
    Self { evals, num_vars }
  }
}

/// Test-only: Convert i32 evaluations to field elements.
#[cfg(test)]
#[allow(missing_docs)]
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
#[allow(dead_code, missing_docs)]
impl<const D: usize> LagrangeEvaluatedMultilinearPolynomial<i64, D> {
  pub fn to_field<F: SmallValueField<i64>>(&self) -> LagrangeEvaluatedMultilinearPolynomial<F, D> {
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
  use crate::{
    polys::multilinear::MultilinearPolynomial, provider::pasta::pallas,
    small_field::SmallValueField,
  };
  use ff::Field;

  use super::super::domain::LagrangePoint;

  type Scalar = pallas::Scalar;

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
    let tuple_inf = LagrangeIndex::<3>(vec![LagrangePoint::Infinity]);
    let tuple_zero = LagrangeIndex::<3>(vec![LagrangePoint::Finite(0)]);
    let tuple_one = LagrangeIndex::<3>(vec![LagrangePoint::Finite(1)]);

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
    const D: usize = 3;
    let num_vars = 3;

    // Create input as small values (i32 is identity)
    let input_small: Vec<i32> = (0..(1 << num_vars)).map(|i| (i + 1) as i32).collect();

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
    let p0: i32 = 7;
    let p1: i32 = 19;

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
    const D: usize = 2;
    let num_vars = 3;

    let input: Vec<i32> = (0..(1 << num_vars)).map(|i| (i * 2 + 1) as i32).collect();

    // Extend using allocating version
    let ext1 = LagrangeEvaluatedMultilinearPolynomial::<i32, D>::from_boolean_evals(&input);

    // Extend in-place (zero allocation after initial buffer setup)
    let mut buf_curr = Vec::new();
    let mut buf_scratch = Vec::new();
    let final_size = LagrangeEvaluatedMultilinearPolynomial::<i32, D>::extend_in_place(
      &input,
      &mut buf_curr,
      &mut buf_scratch,
    );

    // Result is always in buf_curr after extend_in_place
    let ext2 = &buf_curr[..final_size];

    // Verify they match
    assert_eq!(ext1.len(), final_size);
    for (i, &ext2_val) in ext2.iter().enumerate() {
      assert_eq!(ext1.get(i), ext2_val, "mismatch at index {i}");
    }
  }

  #[test]
  fn test_small_lagrange_to_field() {
    const D: usize = 2;
    let num_vars = 2;

    let input: Vec<i32> = (0..(1 << num_vars)).map(|i| (i + 1) as i32).collect();

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
    // Test with negative differences (p0 > p1)
    let p0: i32 = 100;
    let p1: i32 = 50;

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
