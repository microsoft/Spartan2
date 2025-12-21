// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2
#![allow(dead_code)]

//! Accumulator data structures for Algorithm 6 small-value sumcheck optimization.
//!
//! This module defines:
//! - [`RoundAccumulator`]: Single round accumulator A_i(v, u) with flat storage
//! - [`SmallValueAccumulators`]: Collection of accumulators for all ℓ₀ rounds
//! - [`QuadraticTAccumulators`]: Type alias for Spartan's t_i sumcheck (D=2)

use crate::{
  accumulator_index::compute_idx4,
  csr::Csr,
  lagrange::{LagrangeCoeff, LagrangeEvaluatedMultilinearPolynomial, UdHatPoint, UdTuple},
  polys::{
    eq::{EqPolynomial, compute_suffix_eq_pyramid},
    multilinear::MultilinearPolynomial,
  },
};
use ff::PrimeField;
use rayon::prelude::*;

/// Polynomial degree D for Spartan's small-value sumcheck.
/// For Spartan's cubic relation (A·B - C), D=2 yields quadratic t_i.
pub const SPARTAN_T_DEGREE: usize = 2;

/// A single round's accumulator A_i(v, u) with flat contiguous storage.
///
/// For round i (0-indexed), this stores:
/// - (D+1)^i prefixes (one per v ∈ U_D^i)
/// - Each prefix has D values (one per u ∈ Û_D = {∞, 0, 2, ..., D-1})
///
/// Storage: `Vec<[Scalar; D]>` — one allocation, contiguous
/// Access: data[v_idx][u_idx]
///
/// This flat storage design provides:
/// - Cache-friendly memory access patterns
/// - Vectorizable merge operations
/// - No runtime bounds checks on inner dimension (compile-time D)
pub struct RoundAccumulator<Scalar: PrimeField, const D: usize> {
  /// Flat storage: data[v_idx] = [A_i(v, ∞), A_i(v, 0), A_i(v, 2), ...]
  data: Vec<[Scalar; D]>,
}

impl<Scalar: PrimeField, const D: usize> RoundAccumulator<Scalar, D> {
  /// Base of the Lagrange domain U_D (compile-time constant)
  const BASE: usize = D + 1;

  /// Create a new accumulator for the given round (0-indexed).
  ///
  /// Allocates (D+1)^round prefix entries, each with D values.
  pub fn new(round: usize) -> Self {
    let num_prefixes = Self::BASE.pow(round as u32);
    Self {
      data: vec![[Scalar::ZERO; D]; num_prefixes],
    }
  }

  /// O(1) indexed accumulation into bucket (v_idx, u_idx).
  #[inline]
  pub fn accumulate(&mut self, v_idx: usize, u_idx: usize, value: Scalar) {
    self.data[v_idx][u_idx] += value;
  }

  /// O(1) indexed read from bucket (v_idx, u_idx).
  #[inline]
  pub fn get(&self, v_idx: usize, u_idx: usize) -> Scalar {
    self.data[v_idx][u_idx]
  }

  /// Accumulate by domain types (type-safe path).
  ///
  /// # Arguments
  /// * `v` - Prefix tuple in U_D^i
  /// * `u` - Point in Û_D
  /// * `value` - Value to accumulate
  #[inline]
  pub fn accumulate_by_domain(&mut self, v: &UdTuple<D>, u: UdHatPoint<D>, value: Scalar) {
    let v_idx = v.to_flat_index();
    let u_idx = u.to_index();
    self.data[v_idx][u_idx] += value;
  }

  /// Read by domain types (type-safe path).
  #[inline]
  pub fn get_by_domain(&self, v: &UdTuple<D>, u: UdHatPoint<D>) -> Scalar {
    let v_idx = v.to_flat_index();
    let u_idx = u.to_index();
    self.data[v_idx][u_idx]
  }

  /// Evaluate t_i(u) = ⟨R_i, A_i(·, u)⟩ for this round.
  pub fn eval_t_at_u(&self, coeff: &LagrangeCoeff<Scalar, D>, u: UdHatPoint<D>) -> Scalar {
    debug_assert_eq!(
      self.data.len(),
      coeff.len(),
      "R_i length must match number of prefixes"
    );
    let u_idx = u.to_index();
    coeff
      .as_slice()
      .iter()
      .zip(self.data.iter())
      .map(|(c, row)| *c * row[u_idx])
      .sum()
  }

  /// Evaluate t_i(u) for all u ∈ Û_D in a single pass.
  pub fn eval_t_all_u(&self, coeff: &LagrangeCoeff<Scalar, D>) -> [Scalar; D] {
    debug_assert_eq!(self.data.len(), coeff.len());
    let mut acc = [Scalar::ZERO; D];
    for (c, row) in coeff.as_slice().iter().zip(self.data.iter()) {
      let scaled = *c;
      for i in 0..D {
        acc[i] += scaled * row[i];
      }
    }
    acc
  }

  /// Element-wise merge (tight loop, compiler can vectorize).
  ///
  /// Used in the reduce phase of parallel fold-reduce.
  pub fn merge(&mut self, other: &Self) {
    for (a, b) in self.data.iter_mut().zip(&other.data) {
      for i in 0..D {
        a[i] += b[i];
      }
    }
  }

  /// Number of prefix entries.
  pub fn num_prefixes(&self) -> usize {
    self.data.len()
  }
}

/// Collection of accumulators for all ℓ₀ rounds.
///
/// Each thread gets its own copy during parallel execution.
/// After processing, thread-local copies are merged via `merge()`.
///
/// Type parameter D is the degree bound for t_i(X) (D=2 for Spartan).
pub struct SmallValueAccumulators<Scalar: PrimeField, const D: usize> {
  /// Number of rounds using small-value optimization
  l0: usize,
  /// rounds[i] contains A_{i+1} (the accumulator for 1-indexed round i+1)
  rounds: Vec<RoundAccumulator<Scalar, D>>,
}

#[derive(Clone, Copy)]
struct CachedPrefixIndex {
  round_0: usize,
  v_idx: usize,
  u_idx: usize,
  y_idx: usize,
}

/// Type alias for Spartan's quadratic t_i sumcheck (D=2).
///
/// For quadratic polynomials, we evaluate at U_2 = {∞, 0, 1} (3 points)
/// and store Û_2 = {∞, 0} (2 points, excluding 1).
pub type QuadraticTAccumulators<Scalar> = SmallValueAccumulators<Scalar, SPARTAN_T_DEGREE>;

impl<Scalar: PrimeField, const D: usize> SmallValueAccumulators<Scalar, D> {
  /// Create a fresh accumulator (used per-thread in fold).
  ///
  /// # Arguments
  /// * `l0` - Number of rounds using small-value optimization
  pub fn new(l0: usize) -> Self {
    let rounds = (0..l0).map(RoundAccumulator::new).collect();
    Self { l0, rounds }
  }

  /// O(1) accumulation into bucket (round, v_idx, u_idx).
  #[inline]
  pub fn accumulate(&mut self, round: usize, v_idx: usize, u_idx: usize, value: Scalar) {
    self.rounds[round].accumulate(v_idx, u_idx, value);
  }

  /// Read A_i(v, u).
  #[inline]
  pub fn get(&self, round: usize, v_idx: usize, u_idx: usize) -> Scalar {
    self.rounds[round].get(v_idx, u_idx)
  }

  /// Merge another accumulator into this one (for reduce phase).
  pub fn merge(&mut self, other: &Self) {
    for (self_round, other_round) in self.rounds.iter_mut().zip(&other.rounds) {
      self_round.merge(other_round);
    }
  }

  /// Accumulate by domain types (type-safe path).
  #[inline]
  pub fn accumulate_by_domain(
    &mut self,
    round: usize,
    v: &UdTuple<D>,
    u: UdHatPoint<D>,
    value: Scalar,
  ) {
    self.rounds[round].accumulate_by_domain(v, u, value);
  }

  /// Read A_i(v, u) by domain types (type-safe path).
  #[inline]
  pub fn get_by_domain(&self, round: usize, v: &UdTuple<D>, u: UdHatPoint<D>) -> Scalar {
    self.rounds[round].get_by_domain(v, u)
  }

  /// Number of rounds.
  pub fn num_rounds(&self) -> usize {
    self.l0
  }

  /// Get read-only access to a specific round's accumulator.
  pub fn round(&self, i: usize) -> &RoundAccumulator<Scalar, D> {
    &self.rounds[i]
  }
}

/// Procedure 9: Build accumulators A_i(v, u) for Spartan's first sum-check (Algorithm 6).
///
/// Computes accumulators for: g(X) = eq(τ, X) · (Az(X) · Bz(X) - Cz(X))
///
/// D is the degree bound of t_i(X) (not s_i); for Spartan, D = 2.
///
/// Parallelism strategy:
/// - Outer parallel loop over x_out values (using Rayon fold-reduce)
/// - Each thread maintains thread-local accumulators
/// - Final reduction merges all thread-local results via element-wise addition
///
/// Spartan-specific optimizations (D=2):
/// - Skip cz_ext entirely: for binary betas, use cz_pref directly
/// - Skip binary betas: for satisfying witnesses, Az·Bz = Cz on {0,1}^n, so they contribute 0
/// - Only process betas with ∞ (where Cz doesn't contribute anyway)
pub fn build_accumulators_spartan<S: PrimeField + Send + Sync>(
  az: &MultilinearPolynomial<S>,
  bz: &MultilinearPolynomial<S>,
  taus: &[S],
  l0: usize,
) -> SmallValueAccumulators<S, 2> {
  let base: usize = 3; // D + 1 = 2 + 1 = 3
  let l = az.Z.len().trailing_zeros() as usize;
  debug_assert_eq!(az.Z.len(), 1usize << l, "poly size must be power of 2");
  debug_assert_eq!(az.Z.len(), bz.Z.len());
  debug_assert_eq!(taus.len(), l, "taus must have length ℓ");
  debug_assert_eq!(l % 2, 0, "Algorithm 6 split expects even ℓ");

  let half = l / 2;
  let xout_vars = half.checked_sub(l0).expect("l0 must be <= ℓ/2");
  debug_assert!(l0 <= half, "l0 must be <= ℓ/2");

  let num_x_out = 1usize << xout_vars;
  let num_betas = base.pow(l0 as u32);
  let suffix_vars = l - l0;
  let prefix_size = 1usize << l0;

  // Precompute eq tables
  let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + half]); // |{0,1}|^{ℓ/2}
  let e_xout = EqPolynomial::evals_from_points(&taus[half + l0..]); // |{0,1}|^{ℓ/2-ℓ0}
  let e_y = compute_suffix_eq_pyramid(&taus[..l0], l0); // Vec per round

  let beta_has_infinity: Vec<bool> = (0..num_betas)
    .map(|mut t| {
      for _ in 0..l0 {
        if t % base == 0 {
          return true;
        }
        t /= base;
      }
      false
    })
    .collect();

  let mut beta_prefix_cache: Csr<CachedPrefixIndex> = Csr::with_capacity(num_betas, num_betas * l0);
  for b in 0..num_betas {
    let beta = UdTuple::<2>::from_flat_index(b, l0);
    let entries: Vec<_> = compute_idx4(&beta, l0)
      .into_iter()
      .map(|entry| CachedPrefixIndex {
        round_0: entry.round_0idx(),
        v_idx: entry.v_idx,
        u_idx: entry.u.to_index(),
        y_idx: entry.y_idx,
      })
      .collect();
    beta_prefix_cache.push(&entries);
  }

  // Parallel over x_out
  (0..num_x_out)
    .into_par_iter()
    .fold(
      || SmallValueAccumulators::<S, 2>::new(l0),
      |mut acc, x_out_bits| {
        // Partial sums indexed by β, accumulated over x_in loop
        let mut beta_partial_sums = vec![S::ZERO; num_betas];

        // Pre-allocate buffers outside inner loop (reused across iterations)
        // Note: cz_pref not needed - we skip binary betas entirely
        let mut az_pref = vec![S::ZERO; prefix_size];
        let mut bz_pref = vec![S::ZERO; prefix_size];
        let ext_size = base.pow(l0 as u32); // (D+1)^l0
        let mut buf_a = vec![S::ZERO; ext_size];
        let mut buf_b = vec![S::ZERO; ext_size];

        // Inner loop over x_in
        for (x_in_bits, &ein) in e_in.iter().enumerate() {
          let suffix = (x_in_bits << xout_vars) | x_out_bits;

          // Fill by index assignment (no allocation)
          // Note: only az and bz needed - cz is skipped
          for prefix in 0..prefix_size {
            let idx = (prefix << suffix_vars) | suffix;
            az_pref[prefix] = az.Z[idx];
            bz_pref[prefix] = bz.Z[idx];
          }

          // Extend Az and Bz to Lagrange domain using ping-pong buffers.
          // Skip Cz extension entirely - binary betas yield 0, ∞ betas don't use Cz.
          let az_ext =
            LagrangeEvaluatedMultilinearPolynomial::<S, 2>::from_boolean_evals_with_buffer_reusing(
              &az_pref, &mut buf_a, &mut buf_b,
            );
          let bz_ext =
            LagrangeEvaluatedMultilinearPolynomial::<S, 2>::from_boolean_evals_with_buffer_reusing(
              &bz_pref, &mut buf_a, &mut buf_b,
            );

          // Only process betas with ∞ - binary betas contribute 0 for satisfying witnesses
          // (Az·Bz = Cz on {0,1}^n). For ∞ betas, Cz doesn't contribute anyway.
          for (beta_idx, sum) in beta_partial_sums.iter_mut().enumerate() {
            if !beta_has_infinity[beta_idx] {
              continue; // Skip binary betas - they contribute 0
            }
            // For ∞ betas: Cz doesn't contribute, just compute Az·Bz
            let prod = az_ext.get(beta_idx) * bz_ext.get(beta_idx);
            *sum += ein * prod;
          }
        }

        // Distribute beta_partial_sums → A_i(v,u) via idx4
        let ex = e_xout[x_out_bits];
        for (beta_idx, &val) in beta_partial_sums.iter().enumerate() {
          if val.is_zero().into() {
            continue;
          }
          let ex_val = ex * val;
          for pref in &beta_prefix_cache[beta_idx] {
            let ey = e_y[pref.round_0][pref.y_idx];
            acc.accumulate(pref.round_0, pref.v_idx, pref.u_idx, ey * ex_val);
          }
        }

        acc
      },
    )
    .reduce(
      || SmallValueAccumulators::<S, 2>::new(l0),
      |mut a, b| {
        a.merge(&b);
        a
      },
    )
}

/// Generic Procedure 9: Build accumulators A_i(v, u) for Algorithm 6.
///
/// Computes accumulators for: g(X) = eq(τ, X) · ∏_{k=1}^d p_k(X)
///
/// This is the general algorithm that works for any number of polynomials
/// and any degree bound D.
///
/// # Arguments
/// * `polys` - Slice of multilinear polynomials to multiply
/// * `taus` - Random challenge points (length ℓ)
/// * `l0` - Number of small-value rounds
pub fn build_accumulators<S: PrimeField + Send + Sync, const D: usize>(
  polys: &[&MultilinearPolynomial<S>],
  taus: &[S],
  l0: usize,
) -> SmallValueAccumulators<S, D> {
  assert!(!polys.is_empty(), "must have at least one polynomial");
  let base: usize = D + 1;
  let l = polys[0].Z.len().trailing_zeros() as usize;
  debug_assert_eq!(
    polys[0].Z.len(),
    1usize << l,
    "poly size must be power of 2"
  );
  for poly in polys.iter().skip(1) {
    debug_assert_eq!(
      poly.Z.len(),
      polys[0].Z.len(),
      "all polys must have same size"
    );
  }
  debug_assert_eq!(taus.len(), l, "taus must have length ℓ");
  debug_assert_eq!(l % 2, 0, "Algorithm 6 split expects even ℓ");

  let half = l / 2;
  let xout_vars = half.checked_sub(l0).expect("l0 must be <= ℓ/2");
  debug_assert!(l0 <= half, "l0 must be <= ℓ/2");

  let num_x_out = 1usize << xout_vars;
  let num_betas = base.pow(l0 as u32);
  let suffix_vars = l - l0;
  let prefix_size = 1usize << l0;
  let d = polys.len();

  // Precompute eq tables
  let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + half]); // |{0,1}|^{ℓ/2}
  let e_xout = EqPolynomial::evals_from_points(&taus[half + l0..]); // |{0,1}|^{ℓ/2-ℓ0}
  let e_y = compute_suffix_eq_pyramid(&taus[..l0], l0); // Vec per round

  let mut beta_prefix_cache: Csr<CachedPrefixIndex> = Csr::with_capacity(num_betas, num_betas * l0);
  for b in 0..num_betas {
    let beta = UdTuple::<D>::from_flat_index(b, l0);
    let entries: Vec<_> = compute_idx4(&beta, l0)
      .into_iter()
      .map(|entry| CachedPrefixIndex {
        round_0: entry.round_0idx(),
        v_idx: entry.v_idx,
        u_idx: entry.u.to_index(),
        y_idx: entry.y_idx,
      })
      .collect();
    beta_prefix_cache.push(&entries);
  }

  // Parallel over x_out
  (0..num_x_out)
    .into_par_iter()
    .fold(
      || SmallValueAccumulators::<S, D>::new(l0),
      |mut acc, x_out_bits| {
        // Partial sums indexed by β, accumulated over x_in loop
        let mut beta_partial_sums = vec![S::ZERO; num_betas];

        // Pre-allocate buffers outside inner loop (reused across iterations)
        let mut poly_prefs: Vec<Vec<S>> = (0..d).map(|_| vec![S::ZERO; prefix_size]).collect();
        let ext_size = base.pow(l0 as u32);
        let mut scratch_buf_a = vec![S::ZERO; ext_size];
        let mut scratch_buf_b = vec![S::ZERO; ext_size];

        // Inner loop over x_in
        for (x_in_bits, &ein) in e_in.iter().enumerate() {
          let suffix = (x_in_bits << xout_vars) | x_out_bits;

          // Fill all d prefix buffers by index assignment
          for prefix in 0..prefix_size {
            let idx = (prefix << suffix_vars) | suffix;
            for (k, poly) in polys.iter().enumerate() {
              poly_prefs[k][prefix] = poly.Z[idx];
            }
          }

          // Extend all d polynomials using shared scratch buffers
          let exts: Vec<_> = poly_prefs
            .iter()
            .map(|pref| {
              LagrangeEvaluatedMultilinearPolynomial::<S, D>::from_boolean_evals_with_buffer_reusing(
                pref,
                &mut scratch_buf_a,
                &mut scratch_buf_b,
              )
            })
            .collect();

          // Compute ∏ p_k(β) for each beta
          for (beta_idx, sum) in beta_partial_sums.iter_mut().enumerate() {
            let prod: S = exts.iter().map(|ext| ext.get(beta_idx)).product();
            *sum += ein * prod;
          }
        }

        // Distribute beta_partial_sums → A_i(v,u) via idx4
        let ex = e_xout[x_out_bits];
        for (beta_idx, &val) in beta_partial_sums.iter().enumerate() {
          if val.is_zero().into() {
            continue;
          }
          let ex_val = ex * val;
          for pref in &beta_prefix_cache[beta_idx] {
            let ey = e_y[pref.round_0][pref.y_idx];
            acc.accumulate(pref.round_0, pref.v_idx, pref.u_idx, ey * ex_val);
          }
        }

        acc
      },
    )
    .reduce(
      || SmallValueAccumulators::<S, D>::new(l0),
      |mut a, b| {
        a.merge(&b);
        a
      },
    )
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{lagrange::UdPoint, polys::eq::EqPolynomial, provider::pasta::pallas};
  use ff::Field;

  type Scalar = pallas::Scalar;

  // Use the shared constant for polynomial degree in tests
  const D: usize = SPARTAN_T_DEGREE;

  // === RoundAccumulator tests ===

  #[test]
  fn test_round_accumulator_new() {
    // D=3: round 0 → 4^0=1 prefix, round 1 → 4^1=4 prefixes, round 2 → 4^2=16 prefixes
    let acc0: RoundAccumulator<Scalar, 3> = RoundAccumulator::new(0);
    assert_eq!(acc0.num_prefixes(), 1);

    let acc1: RoundAccumulator<Scalar, 3> = RoundAccumulator::new(1);
    assert_eq!(acc1.num_prefixes(), 4);

    let acc2: RoundAccumulator<Scalar, 3> = RoundAccumulator::new(2);
    assert_eq!(acc2.num_prefixes(), 16);
  }

  #[test]
  fn test_round_accumulator_accumulate_get() {
    let mut acc: RoundAccumulator<Scalar, 3> = RoundAccumulator::new(1); // 4 prefixes

    // Initially all zeros
    for v_idx in 0..4 {
      for u_idx in 0..3 {
        assert_eq!(acc.get(v_idx, u_idx), Scalar::ZERO);
      }
    }

    // Accumulate some values
    let val1 = Scalar::from(7u64);
    let val2 = Scalar::from(13u64);

    acc.accumulate(0, 0, val1);
    acc.accumulate(0, 0, val2);
    assert_eq!(acc.get(0, 0), val1 + val2);

    acc.accumulate(2, 1, val1);
    assert_eq!(acc.get(2, 1), val1);

    // Other entries unchanged
    assert_eq!(acc.get(0, 1), Scalar::ZERO);
    assert_eq!(acc.get(1, 0), Scalar::ZERO);
  }

  #[test]
  fn test_round_accumulator_merge() {
    let mut acc1: RoundAccumulator<Scalar, 3> = RoundAccumulator::new(1); // 4 prefixes
    let mut acc2: RoundAccumulator<Scalar, 3> = RoundAccumulator::new(1);

    let val1 = Scalar::from(5u64);
    let val2 = Scalar::from(11u64);
    let val3 = Scalar::from(17u64);

    acc1.accumulate(0, 0, val1);
    acc1.accumulate(1, 2, val2);

    acc2.accumulate(0, 0, val3);
    acc2.accumulate(2, 1, val1);

    acc1.merge(&acc2);

    // Check merged values
    assert_eq!(acc1.get(0, 0), val1 + val3);
    assert_eq!(acc1.get(1, 2), val2);
    assert_eq!(acc1.get(2, 1), val1);
    assert_eq!(acc1.get(3, 0), Scalar::ZERO);
  }

  #[test]
  fn test_round_accumulator_domain_methods() {
    let mut acc: RoundAccumulator<Scalar, 3> = RoundAccumulator::new(1); // 4 prefixes

    // v = (Finite(1),) -> flat index = 2 (base 4: ∞=0, 0=1, 1=2, 2=3)
    let v = UdTuple::<3>(vec![UdPoint::Finite(1)]);
    let u = UdHatPoint::<3>::Infinity; // index 0

    let val = Scalar::from(42u64);
    acc.accumulate_by_domain(&v, u, val);

    assert_eq!(acc.get_by_domain(&v, u), val);
    // Verify same via raw indices
    assert_eq!(acc.get(2, 0), val);
  }

  // === SmallValueAccumulators tests ===

  #[test]
  fn test_small_value_accumulators_new() {
    // For D=2 (base=3), ℓ₀=3
    // Round 0: 3^0 = 1 prefix
    // Round 1: 3^1 = 3 prefixes
    // Round 2: 3^2 = 9 prefixes
    let acc: SmallValueAccumulators<Scalar, D> = SmallValueAccumulators::new(3);

    assert_eq!(acc.num_rounds(), 3);
    assert_eq!(acc.round(0).num_prefixes(), 1);
    assert_eq!(acc.round(1).num_prefixes(), 3);
    assert_eq!(acc.round(2).num_prefixes(), 9);
  }

  #[test]
  fn test_small_value_accumulators_accumulate_get() {
    let mut acc: SmallValueAccumulators<Scalar, D> = SmallValueAccumulators::new(3);

    let val1 = Scalar::from(19u64);
    let val2 = Scalar::from(23u64);

    // Accumulate into different rounds
    acc.accumulate(0, 0, 0, val1);
    acc.accumulate(1, 2, 1, val2);
    acc.accumulate(2, 6, 1, val1);

    assert_eq!(acc.get(0, 0, 0), val1);
    assert_eq!(acc.get(1, 2, 1), val2);
    assert_eq!(acc.get(2, 6, 1), val1);
    assert_eq!(acc.get(2, 0, 0), Scalar::ZERO);
  }

  #[test]
  fn test_small_value_accumulators_merge() {
    let mut acc1: SmallValueAccumulators<Scalar, D> = SmallValueAccumulators::new(3);
    let mut acc2: SmallValueAccumulators<Scalar, D> = SmallValueAccumulators::new(3);

    let val1 = Scalar::from(7u64);
    let val2 = Scalar::from(11u64);
    let val3 = Scalar::from(13u64);

    acc1.accumulate(0, 0, 0, val1);
    acc1.accumulate(1, 1, 0, val2);

    acc2.accumulate(0, 0, 0, val3);
    acc2.accumulate(2, 4, 1, val1);

    acc1.merge(&acc2);

    assert_eq!(acc1.get(0, 0, 0), val1 + val3);
    assert_eq!(acc1.get(1, 1, 0), val2);
    assert_eq!(acc1.get(2, 4, 1), val1);
  }

  #[test]
  fn test_small_value_accumulators_domain_methods() {
    let mut acc: SmallValueAccumulators<Scalar, D> = SmallValueAccumulators::new(2);

    // Round 1 has 3 prefixes (base^1)
    // v = (Finite(0),) -> flat index = 1 (∞=0, 0=1, 1=2)
    let v = UdTuple::<2>(vec![UdPoint::Finite(0)]);
    let u = UdHatPoint::<D>::Infinity; // index 0

    let val = Scalar::from(99u64);
    acc.accumulate_by_domain(1, &v, u, val);

    assert_eq!(acc.get_by_domain(1, &v, u), val);
    // Verify same via raw indices
    assert_eq!(acc.get(1, 1, 0), val);
  }

  #[test]
  fn test_quadratic_t_accumulators_alias() {
    // Verify the type alias works
    let acc: QuadraticTAccumulators<Scalar> = QuadraticTAccumulators::new(2);
    assert_eq!(acc.num_rounds(), 2);
    assert_eq!(acc.round(0).num_prefixes(), 1); // 3^0
    assert_eq!(acc.round(1).num_prefixes(), 3); // 3^1
  }

  #[test]
  fn test_accumulator_sizes_match_spec() {
    // For D=2, ℓ₀=3 should have total 26 elements
    // Round 0: 1 * 2 = 2
    // Round 1: 3 * 2 = 6
    // Round 2: 9 * 2 = 18
    // Total: 26
    let acc: SmallValueAccumulators<Scalar, D> = SmallValueAccumulators::new(3);

    let total_elements: usize = (0..3).map(|i| acc.round(i).num_prefixes() * 2).sum();
    assert_eq!(total_elements, 26);
  }

  /// End-to-end correctness for build_accumulators_spartan on a tiny instance.
  ///
  /// ℓ = 4, ℓ0 = 2, D = 2.
  /// Uses a satisfying witness (Az * Bz = Cz) to test the optimized Spartan path.
  /// Verifies against a straightforward (non-parallel) implementation of Procedure 9.
  #[test]
  fn test_build_accumulators_spartan_matches_naive() {
    let l0 = 2;
    let l = 4;
    let half = l / 2;

    // Define deterministic Az, Bz, Cz over {0,1}^4
    // Use a SATISFYING witness: Cz = Az * Bz
    let eval = |bits: usize| -> Scalar {
      // Simple affine: a0 x0 + a1 x1 + a2 x2 + a3 x3 + const
      let x0 = (bits >> 3) & 1;
      let x1 = (bits >> 2) & 1;
      let x2 = (bits >> 1) & 1;
      let x3 = bits & 1;
      Scalar::from((x0 + 2 * x1 + 3 * x2 + 4 * x3 + 5) as u64)
    };
    let az_vals: Vec<Scalar> = (0..16).map(eval).collect();
    let bz_vals: Vec<Scalar> = (0..16).map(|b| eval(b) + Scalar::from(7u64)).collect();
    // Satisfying witness: cz = az * bz
    let cz_vals: Vec<Scalar> = az_vals
      .iter()
      .zip(bz_vals.iter())
      .map(|(a, b)| *a * *b)
      .collect();

    let az = MultilinearPolynomial::new(az_vals.clone());
    let bz = MultilinearPolynomial::new(bz_vals.clone());
    let cz = MultilinearPolynomial::new(cz_vals.clone());

    // Taus (length ℓ)
    let taus: Vec<Scalar> = vec![
      Scalar::from(5u64),
      Scalar::from(7u64),
      Scalar::from(11u64),
      Scalar::from(13u64),
    ];

    // Implementation under test
    let acc_impl = build_accumulators_spartan(&az, &bz, &taus, l0);

    // Precompute eq tables for naive computation
    let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + half]); // τ[2..4]
    let e_xout = EqPolynomial::evals_from_points(&taus[half + l0..]); // τ[4..] (empty -> [1])
    let e_y = compute_suffix_eq_pyramid(&taus[..l0], l0);

    let num_betas = (D + 1).pow(l0 as u32);
    let idx4_cache: Vec<Vec<_>> = (0..num_betas)
      .map(|b| compute_idx4(&UdTuple::<D>::from_flat_index(b, l0), l0))
      .collect();

    // Naive accumulators
    let mut acc_naive: SmallValueAccumulators<Scalar, D> = SmallValueAccumulators::new(l0);

    // x_out domain size = 1 (xout_vars = 0)
    let ex = e_xout[0];

    for x_in_bits in 0..(1 << half) {
      let suffix = x_in_bits; // x_out = 0

      let az_pref = az.gather_prefix_evals(l0, suffix);
      let bz_pref = bz.gather_prefix_evals(l0, suffix);
      let cz_pref = cz.gather_prefix_evals(l0, suffix);

      let az_ext = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&az_pref);
      let bz_ext = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&bz_pref);
      let cz_ext = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&cz_pref);

      let ein = e_in[x_in_bits];

      for beta_idx in 0..num_betas {
        let beta_tuple = az_ext.to_domain_tuple(beta_idx);
        let ab = az_ext.get(beta_idx) * bz_ext.get(beta_idx);
        let prod = if beta_tuple.has_infinity() {
          ab
        } else {
          ab - cz_ext.get(beta_idx)
        };
        let val = ein * prod;

        for pref in &idx4_cache[beta_idx] {
          let ey = e_y[pref.round_0idx()][pref.y_idx];
          acc_naive.accumulate(
            pref.round_0idx(),
            pref.v_idx,
            pref.u.to_index(),
            ey * ex * val,
          );
        }
      }
    }

    // Compare all buckets
    for round in 0..l0 {
      let num_v = (D + 1).pow(round as u32);
      for v_idx in 0..num_v {
        for u_idx in 0..D {
          let got = acc_impl.get(round, v_idx, u_idx);
          let expect = acc_naive.get(round, v_idx, u_idx);
          assert_eq!(
            got, expect,
            "Mismatch at round {}, v_idx {}, u_idx {}",
            round, v_idx, u_idx
          );
        }
      }
    }
  }

  /// Check the ∞ rule: constant polynomials have zero leading coefficient at ∞.
  ///
  /// ℓ = 2, ℓ0 = 1, D = 2.
  /// Tests product of two constant polynomials (1 and -1) = -1 everywhere.
  /// At ∞: leading coefficient of constant poly is 0.
  /// At finite points: evaluates to the constant value.
  ///
  /// Uses generic `build_accumulators` (Procedure 9).
  #[test]
  fn test_infinity_drops_cz() {
    let l0 = 1;
    let l = 2;
    let half = l / 2;

    // Two constant polynomials: 1 and -1, product = -1
    let ones = MultilinearPolynomial::new(vec![Scalar::ONE; 1 << l]);
    let neg_ones = MultilinearPolynomial::new(vec![-Scalar::ONE; 1 << l]);

    let taus: Vec<Scalar> = vec![Scalar::from(5u64), Scalar::from(7u64)];

    // Use generic build_accumulators with D=2 for product of two polynomials
    let acc = build_accumulators::<Scalar, 2>(&[&ones, &neg_ones], &taus, l0);

    // Compute e_in sum = Σ eq(τ[l0..l0+half], xin), here half=1, l0=1 -> slice τ[1]
    let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + half]);
    let ein_sum: Scalar = e_in.iter().copied().sum();

    // Round 0, v_idx=0
    let u_infinity_idx = UdHatPoint::<2>::Infinity.to_index(); // 0
    let u_zero_idx = UdHatPoint::<2>::Finite(0).to_index(); // 1

    let acc_inf = acc.get(0, 0, u_infinity_idx);
    let acc_zero = acc.get(0, 0, u_zero_idx);

    // At ∞: leading coefficient of constant poly is 0
    assert_eq!(
      acc_inf,
      Scalar::ZERO,
      "Constant poly has zero leading coeff at ∞"
    );
    // At 0: product = 1 * (-1) = -1
    assert_eq!(
      acc_zero,
      ein_sum * (-Scalar::ONE),
      "Should equal sum * (-1)"
    );
  }

  /// Binary-β zero shortcut: Az=Bz=Cz=first variable (x0), so Az·Bz−Cz=0 on binary β.
  /// Non-binary β (∞) should yield non-zero in some bucket.
  #[test]
  fn test_binary_beta_zero_shortcut_behavior() {
    // Use l0=1 so round 0 buckets are fed only by β of length 1 (easy to reason about).
    let l0 = 1;
    let l = 2;

    // Az = Bz = top bit x0 (most significant of 2 bits)
    // For satisfying witness, Cz = Az * Bz = Az (since Az ∈ {0,1} and Az = Bz)
    let az_vals: Vec<Scalar> = (0..(1 << l))
      .map(|bits| {
        let x0 = (bits >> (l - 1)) & 1;
        Scalar::from(x0 as u64)
      })
      .collect();
    let bz_vals = az_vals.clone();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    let taus: Vec<Scalar> = vec![Scalar::from(3u64), Scalar::from(5u64)];

    let acc = build_accumulators_spartan(&az, &bz, &taus, l0);

    // Only round 0 exists (v is empty). β ranges over U_d with binary {0,1} and non-binary {∞}.
    // Buckets for u = 0 should be zero (binary β), bucket for u = ∞ should be non-zero.
    let u_inf = UdHatPoint::<D>::Infinity.to_index(); // 0
    let u_zero = UdHatPoint::<D>::Finite(0).to_index(); // 1

    assert!(
      bool::from(acc.get(0, 0, u_zero).is_zero()),
      "binary β should give zero for u=0"
    );
    assert!(
      !bool::from(acc.get(0, 0, u_inf).is_zero()),
      "non-binary β (∞) should give non-zero"
    );
  }

  /// Test generic build_accumulators (Procedure 9) with a product of 3 polynomials.
  ///
  /// ℓ = 10, ℓ0 = 3, D = 3 (degree bound for product of 3 polynomials).
  /// Verifies that accumulators are computed correctly by comparing against naive computation.
  #[test]
  fn test_build_accumulators_product_of_three() {
    use ff::Field;
    use rand::{rngs::StdRng, SeedableRng};

    const L: usize = 10;
    const L0: usize = 3;
    const D: usize = 3; // Degree bound for product of 3 linear polynomials

    let n = 1usize << L;
    let half = L / 2;
    let xout_vars = half - L0;
    let num_betas = (D + 1).pow(L0 as u32);

    let mut rng = StdRng::seed_from_u64(42);

    // Create 3 random multilinear polynomials
    let p1_vals: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let p2_vals: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let p3_vals: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();

    let p1 = MultilinearPolynomial::new(p1_vals);
    let p2 = MultilinearPolynomial::new(p2_vals);
    let p3 = MultilinearPolynomial::new(p3_vals);

    // Random taus
    let taus: Vec<Scalar> = (0..L).map(|_| Scalar::random(&mut rng)).collect();

    // Build accumulators using generic Procedure 9
    let acc_impl = build_accumulators::<Scalar, D>(&[&p1, &p2, &p3], &taus, L0);

    // ===== Naive computation for comparison =====
    let e_in = EqPolynomial::evals_from_points(&taus[L0..L0 + half]);
    let e_xout = EqPolynomial::evals_from_points(&taus[half + L0..]);
    let e_y = compute_suffix_eq_pyramid(&taus[..L0], L0);

    let idx4_cache: Vec<Vec<_>> = (0..num_betas)
      .map(|b| compute_idx4(&UdTuple::<D>::from_flat_index(b, L0), L0))
      .collect();

    let mut acc_naive: SmallValueAccumulators<Scalar, D> = SmallValueAccumulators::new(L0);

    for x_out_bits in 0..(1 << xout_vars) {
      let ex = e_xout[x_out_bits];

      for x_in_bits in 0..(1 << half) {
        let suffix = (x_in_bits << xout_vars) | x_out_bits;

        // Gather prefix evaluations and extend to Lagrange domain
        let p1_pref = p1.gather_prefix_evals(L0, suffix);
        let p2_pref = p2.gather_prefix_evals(L0, suffix);
        let p3_pref = p3.gather_prefix_evals(L0, suffix);

        let p1_ext = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&p1_pref);
        let p2_ext = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&p2_pref);
        let p3_ext = LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&p3_pref);

        let ein = e_in[x_in_bits];

        for beta_idx in 0..num_betas {
          // Compute product p1(β) * p2(β) * p3(β)
          let prod = p1_ext.get(beta_idx) * p2_ext.get(beta_idx) * p3_ext.get(beta_idx);
          let val = ein * prod;

          // Distribute to accumulators via idx4
          for pref in &idx4_cache[beta_idx] {
            let ey = e_y[pref.round_0idx()][pref.y_idx];
            acc_naive.accumulate(pref.round_0idx(), pref.v_idx, pref.u.to_index(), ey * ex * val);
          }
        }
      }
    }

    // ===== Compare all accumulator buckets =====
    for round in 0..L0 {
      let num_v = (D + 1).pow(round as u32);
      for v_idx in 0..num_v {
        for u_idx in 0..D {
          let got = acc_impl.get(round, v_idx, u_idx);
          let expect = acc_naive.get(round, v_idx, u_idx);
          assert_eq!(
            got, expect,
            "Mismatch at round {}, v_idx {}, u_idx {}",
            round, v_idx, u_idx
          );
        }
      }
    }
  }
}
