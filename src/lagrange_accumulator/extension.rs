// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Procedure 6: Extension of multilinear polynomial evaluations from
//! boolean hypercube {0,1}^ℓ to Lagrange domain U_D^ℓ.

use std::ops::{Add, Sub};

// ============================================================================
// Helper functions for Lagrange extension
// ============================================================================

/// Extend a single suffix element from boolean to Lagrange domain.
#[inline(always)]
fn extend<T, const D: usize>(
  src: &[T],
  dst: &mut [T],
  base_src: usize,
  base_dst: usize,
  suffix_count: usize,
  suffix_idx: usize,
) where
  T: Copy + Default + Add<Output = T> + Sub<Output = T>,
{
  let p0 = src[base_src + suffix_idx];
  let p1 = src[base_src + suffix_count + suffix_idx];
  let diff = p1 - p0;

  // γ = ∞ (index 0)
  dst[base_dst + suffix_idx] = diff;
  // γ = 0 (index 1)
  dst[base_dst + suffix_count + suffix_idx] = p0;

  if D >= 2 {
    // γ = 1 (index 2)
    dst[base_dst + 2 * suffix_count + suffix_idx] = p1;
    // γ = 2..D-1: extrapolate
    let mut val = p1;
    for k in 2..D {
      val = val + diff;
      dst[base_dst + (k + 1) * suffix_count + suffix_idx] = val;
    }
  }
}

/// Extend boolean hypercube evaluations to Lagrange domain in-place.
///
/// This is Procedure 6: extends polynomial evaluations from {0,1}^ℓ to U_D^ℓ.
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
pub(crate) fn extend_to_lagrange_domain<T, const D: usize>(
  input: &[T],
  buf_curr: &mut Vec<T>,
  buf_scratch: &mut Vec<T>,
) -> usize
where
  T: Copy + Default + Add<Output = T> + Sub<Output = T>,
{
  let base: usize = D + 1;
  let num_vars = input.len().trailing_zeros() as usize;
  assert_eq!(input.len(), 1 << num_vars, "Input size must be power of 2");

  if num_vars == 0 {
    // Single element: copy to buf_curr and return
    if buf_curr.is_empty() {
      buf_curr.push(T::default());
    }
    buf_curr[0] = input[0];
    return 1;
  }

  let final_size = base.pow(num_vars as u32);

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
    let prefix_count = base.pow((j - 1) as u32);
    let suffix_count = 1usize << (num_vars - j);
    // Current layout: prefix_count rows × 2 boolean values × suffix_count elements
    let current_stride = 2 * suffix_count;
    // Next layout: prefix_count rows × (D+1) domain values × suffix_count elements
    let next_stride = base * suffix_count;

    // Alternate between buffers each iteration
    let (src, dst) = if j % 2 == 1 {
      (&buf_curr[..], &mut buf_scratch[..])
    } else {
      (&buf_scratch[..], &mut buf_curr[..])
    };

    for prefix_idx in 0..prefix_count {
      let base_src = prefix_idx * current_stride;
      let base_dst = prefix_idx * next_stride;

      for s in 0..suffix_count {
        extend::<T, D>(src, dst, base_src, base_dst, suffix_count, s);
      }
    }
  }

  // Ensure result ends up in buf_curr (swap if result is currently in buf_scratch)
  if num_vars % 2 == 1 {
    std::mem::swap(buf_curr, buf_scratch);
  }
  final_size
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{big_num::SmallValueField, provider::pasta::pallas};
  use ff::Field;

  use super::super::domain::{LagrangeIndex, LagrangePoint};

  type Scalar = pallas::Scalar;

  /// Test helper: extend boolean evals and return owned Vec.
  fn extend_for_test<T: Copy + Default + Add<Output = T> + Sub<Output = T>, const D: usize>(
    input: &[T],
  ) -> Vec<T> {
    let mut buf = Vec::new();
    let mut scratch = Vec::new();
    let size = extend_to_lagrange_domain::<T, D>(input, &mut buf, &mut scratch);
    buf.truncate(size);
    buf
  }

  #[test]
  fn test_extend_output_size() {
    for num_vars in 1..=4 {
      let input_size = 1 << num_vars;
      let input: Vec<Scalar> = (0..input_size).map(|i| Scalar::from(i as u64)).collect();

      let extended = extend_for_test::<Scalar, 3>(&input);

      let expected_size = 4usize.pow(num_vars as u32); // (D+1)^num_vars = 4^num_vars
      assert_eq!(extended.len(), expected_size);
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

    let extended = extend_for_test::<Scalar, D>(&input);

    // In U_d indexing: 0 → index 1, 1 → index 2
    #[allow(clippy::needless_range_loop)]
    for b in 0..(1 << num_vars) {
      let mut ud_idx = 0;
      for j in 0..num_vars {
        let bit = (b >> (num_vars - 1 - j)) & 1;
        let ud_val = bit + 1; // 0→1, 1→2
        ud_idx = ud_idx * base + ud_val;
      }

      assert_eq!(extended[ud_idx], input[b]);
    }
  }

  #[test]
  fn test_extend_single_var() {
    let p0 = Scalar::from(7u64);
    let p1 = Scalar::from(19u64);

    let extended = extend_for_test::<Scalar, 3>(&[p0, p1]);

    // U_d = {∞, 0, 1, 2} with indices 0, 1, 2, 3
    assert_eq!(extended[0], p1 - p0, "p(∞) = leading coeff");
    assert_eq!(extended[1], p0, "p(0)");
    assert_eq!(extended[2], p1, "p(1)");
    assert_eq!(extended[3], p1.double() - p0, "p(2) = 2*p1 - p0");
  }

  #[test]
  fn test_extend_matches_direct() {
    let num_vars = 3;
    const D: usize = 3;
    let base = D + 1;

    let input: Vec<Scalar> = (0..(1 << num_vars))
      .map(|_| Scalar::random(&mut rand_core::OsRng))
      .collect();

    let extended = extend_for_test::<Scalar, D>(&input);

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
      assert_eq!(extended[idx], direct);
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

    let extended = extend_for_test::<Scalar, D>(&input);

    // p(∞, y₂, y₃) = p(1, y₂, y₃) - p(0, y₂, y₃)
    for y2 in 0..2usize {
      for y3 in 0..2usize {
        let idx_0 = (0 << 2) | (y2 << 1) | y3; // p(0, y2, y3)
        let idx_1 = (1 << 2) | (y2 << 1) | y3; // p(1, y2, y3)

        let expected = input[idx_1] - input[idx_0];
        let ext_idx = 0 * base * base + (y2 + 1) * base + (y3 + 1);

        assert_eq!(extended[ext_idx], expected);
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

    let extended = extend_for_test::<Scalar, D>(&input);

    // Finite points: p(a,b,c) = a + 2b + 4c
    for a in 0..D {
      for b in 0..D {
        for c in 0..D {
          let idx = (a + 1) * base * base + (b + 1) * base + (c + 1);
          let expected = Scalar::from(a as u64 + 2 * b as u64 + 4 * c as u64);
          assert_eq!(extended[idx], expected);
        }
      }
    }

    // Infinity points = variable coefficients
    assert_eq!(
      extended[0 * base * base + 1 * base + 1],
      Scalar::ONE,
      "p(∞,0,0) = coeff of X"
    );
    assert_eq!(
      extended[1 * base * base + 0 * base + 1],
      Scalar::from(2u64),
      "p(0,∞,0) = coeff of Y"
    );
    assert_eq!(
      extended[1 * base * base + 1 * base + 0],
      Scalar::from(4u64),
      "p(0,0,∞) = coeff of Z"
    );
    assert_eq!(extended[0], Scalar::ZERO, "p(∞,∞,∞) = 0 (no XYZ term)");
  }

  #[test]
  fn test_get_by_domain() {
    let p0 = Scalar::from(7u64);
    let p1 = Scalar::from(19u64);

    let extended = extend_for_test::<Scalar, 3>(&[p0, p1]);

    // Test type-safe access via LagrangeIndex
    let tuple_inf = LagrangeIndex::<3>(vec![LagrangePoint::Infinity]);
    let tuple_zero = LagrangeIndex::<3>(vec![LagrangePoint::Finite(0)]);
    let tuple_one = LagrangeIndex::<3>(vec![LagrangePoint::Finite(1)]);

    assert_eq!(extended[tuple_inf.to_flat_index()], p1 - p0);
    assert_eq!(extended[tuple_zero.to_flat_index()], p0);
    assert_eq!(extended[tuple_one.to_flat_index()], p1);
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
  fn evaluate_multilinear(evals: &[Scalar], point: &[Scalar]) -> Scalar {
    let chis = crate::polys::eq::EqPolynomial::evals_from_points(point);
    evals
      .iter()
      .zip(chis.iter())
      .fold(Scalar::ZERO, |acc, (z, chi)| acc + *z * *chi)
  }

  // === Small-value extension tests ===

  #[test]
  fn test_small_lagrange_matches_field_version() {
    const D: usize = 3;
    let num_vars = 3;

    let input_small: Vec<i32> = (0..(1 << num_vars)).map(|i| i + 1).collect();
    let input_field: Vec<Scalar> = (0..(1 << num_vars))
      .map(|i| Scalar::from((i + 1) as u64))
      .collect();

    let small_ext = extend_for_test::<i32, D>(&input_small);
    let field_ext = extend_for_test::<Scalar, D>(&input_field);

    assert_eq!(small_ext.len(), field_ext.len());
    for i in 0..small_ext.len() {
      let small_as_field: Scalar = Scalar::small_to_field(small_ext[i]);
      assert_eq!(small_as_field, field_ext[i], "mismatch at index {i}");
    }
  }

  #[test]
  fn test_small_lagrange_single_var() {
    let p0: i32 = 7;
    let p1: i32 = 19;

    let extended = extend_for_test::<i32, 3>(&[p0, p1]);

    assert_eq!(extended[0], p1 - p0, "p(∞) = leading coeff");
    assert_eq!(extended[1], p0, "p(0)");
    assert_eq!(extended[2], p1, "p(1)");
    assert_eq!(extended[3], 31i32, "p(2) = 2*p1 - p0");
  }

  #[test]
  fn test_small_lagrange_negative_values() {
    let p0: i32 = 100;
    let p1: i32 = 50;

    let extended = extend_for_test::<i32, 2>(&[p0, p1]);

    assert_eq!(extended[0], -50i32);
    assert_eq!(extended[1], p0);
    assert_eq!(extended[2], p1);

    // Verify field version handles negative correctly
    let input_field: Vec<Scalar> = vec![Scalar::from(100u64), Scalar::from(50u64)];
    let field_ext = extend_for_test::<Scalar, 2>(&input_field);
    assert_eq!(field_ext[0], -Scalar::from(50u64));
  }
}
