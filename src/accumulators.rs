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
/// Important correctness points:
/// - If any coordinate of β is ∞, drop the Cz term (eq·Cz is degree 2, so it has no X³ coefficient).
/// - Do not double-count eq: E_in is applied inside the x_in loop; E_out is applied when distributing via idx4.
pub fn build_accumulators<S: PrimeField + Send + Sync, const D: usize>(
  az: &MultilinearPolynomial<S>,
  bz: &MultilinearPolynomial<S>,
  cz: &MultilinearPolynomial<S>,
  taus: &[S],
  l0: usize,
) -> SmallValueAccumulators<S, D> {
  let base: usize = D + 1;
  let l = az.Z.len().trailing_zeros() as usize;
  debug_assert_eq!(az.Z.len(), 1usize << l, "poly size must be power of 2");
  debug_assert_eq!(az.Z.len(), bz.Z.len());
  debug_assert_eq!(az.Z.len(), cz.Z.len());
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
        // tA[β]
        let mut tA = vec![S::ZERO; num_betas];

        // Pre-allocate buffers outside inner loop (reused across iterations)
        let mut az_pref = vec![S::ZERO; prefix_size];
        let mut bz_pref = vec![S::ZERO; prefix_size];
        let mut cz_pref = vec![S::ZERO; prefix_size];
        let ext_size = (D + 1).pow(l0 as u32);
        let mut buf_a = vec![S::ZERO; ext_size];
        let mut buf_b = vec![S::ZERO; ext_size];

        // Inner loop over x_in
        for (x_in_bits, &ein) in e_in.iter().enumerate() {
          let suffix = (x_in_bits << xout_vars) | x_out_bits;

          // Fill by index assignment (no allocation)
          for prefix in 0..prefix_size {
            let idx = (prefix << suffix_vars) | suffix;
            az_pref[prefix] = az.Z[idx];
            bz_pref[prefix] = bz.Z[idx];
            cz_pref[prefix] = cz.Z[idx];
          }

          // Extend to Lagrange domain using ping-pong buffers. This reduces allocations
          // from O(num_x_in × num_x_out) to O(num_threads) by reusing buf_a/buf_b.
          let az_ext =
            LagrangeEvaluatedMultilinearPolynomial::<S, D>::from_boolean_evals_with_buffer_reusing(&az_pref, &mut buf_a, &mut buf_b);
          let bz_ext =
            LagrangeEvaluatedMultilinearPolynomial::<S, D>::from_boolean_evals_with_buffer_reusing(&bz_pref, &mut buf_a, &mut buf_b);
          let cz_ext =
            LagrangeEvaluatedMultilinearPolynomial::<S, D>::from_boolean_evals_with_buffer_reusing(&cz_pref, &mut buf_a, &mut buf_b);

          for (beta_idx, tA_slot) in tA.iter_mut().enumerate() {
            let ab = az_ext.get(beta_idx) * bz_ext.get(beta_idx);
            // ∞ rule: drop Cz at ∞
            let prod = if beta_has_infinity[beta_idx] {
              ab
            } else {
              ab - cz_ext.get(beta_idx)
            };
            *tA_slot += ein * prod;
          }
        }

        // Distribute tA → A_i(v,u) via idx4
        let ex = e_xout[x_out_bits];
        for (beta_idx, &val) in tA.iter().enumerate() {
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

  /// End-to-end correctness for build_accumulators on a tiny instance.
  ///
  /// ℓ = 4, ℓ0 = 2, D = 2.
  /// Verifies against a straightforward (non-parallel) implementation of Procedure 9.
  #[test]
  fn test_build_accumulators_matches_naive() {
    let l0 = 2;
    let l = 4;
    let half = l / 2;

    // Define deterministic Az, Bz, Cz over {0,1}^4
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
    let cz_vals: Vec<Scalar> = (0..16)
      .map(|b| eval(b) * Scalar::from(3u64) + Scalar::from(11u64))
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
    let acc_impl = build_accumulators::<Scalar, D>(&az, &bz, &cz, &taus, l0);

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

  /// Check the ∞ rule: Cz must be dropped when any β coordinate is ∞.
  ///
  /// ℓ = 2, ℓ0 = 1, D = 2.
  /// Az = 1, Bz = 1, Cz = const. Taus arbitrary.
  /// Accumulator for u=∞ should equal Σ_xin e_in[xin] * (1) (drop Cz).
  /// Accumulator for u=0 should equal Σ_xin e_in[xin] * (1 - Cz).
  #[test]
  fn test_infinity_drops_cz() {
    let l0 = 1;
    let l = 2;
    let half = l / 2;

    // Polys: Az=Bz=1 everywhere, Cz=const everywhere
    let az_vals: Vec<Scalar> = vec![Scalar::ONE; 1 << l];
    let bz_vals: Vec<Scalar> = vec![Scalar::ONE; 1 << l];
    let cz_const = Scalar::from(2u64);
    let cz_vals: Vec<Scalar> = vec![cz_const; 1 << l];

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);
    let cz = MultilinearPolynomial::new(cz_vals);

    let taus: Vec<Scalar> = vec![Scalar::from(5u64), Scalar::from(7u64)];

    // Build accumulators
    let acc = build_accumulators::<Scalar, D>(&az, &bz, &cz, &taus, l0);

    // Compute e_in sum = Σ eq(τ[l0..l0+half], xin), here half=1, l0=1 -> slice τ[1]
    let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + half]);
    let ein_sum: Scalar = e_in.iter().copied().sum();

    // Round 0, v_idx=0
    let u_infinity_idx = UdHatPoint::<D>::Infinity.to_index(); // 0
    let u_zero_idx = UdHatPoint::<D>::Finite(0).to_index(); // 1

    let acc_inf = acc.get(0, 0, u_infinity_idx);
    let acc_zero = acc.get(0, 0, u_zero_idx);

    // At ∞: leading coefficient of constant poly is 0, Cz dropped
    assert_eq!(acc_inf, Scalar::ZERO, "Cz should be dropped at ∞ (leading coeff of const poly is 0)");
    // At 0: product = 1 - cz_const
    assert_eq!(acc_zero, ein_sum * (Scalar::ONE - cz_const), "Cz should be subtracted at finite points");
  }

  /// Binary-β zero shortcut: Az=Bz=Cz=first variable (x0), so Az·Bz−Cz=0 on binary β.
  /// Non-binary β (∞) should yield non-zero in some bucket.
  #[test]
  fn test_binary_beta_zero_shortcut_behavior() {
    // Use l0=1 so round 0 buckets are fed only by β of length 1 (easy to reason about).
    let l0 = 1;
    let l = 2;

    // Az = Bz = Cz = top bit x0 (most significant of 2 bits)
    let az_vals: Vec<Scalar> = (0..(1 << l))
      .map(|bits| {
        let x0 = (bits >> (l - 1)) & 1;
        Scalar::from(x0 as u64)
      })
      .collect();
    let bz_vals = az_vals.clone();
    let cz_vals = az_vals.clone();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);
    let cz = MultilinearPolynomial::new(cz_vals);

    let taus: Vec<Scalar> = vec![Scalar::from(3u64), Scalar::from(5u64)];

    let acc = build_accumulators::<Scalar, D>(&az, &bz, &cz, &taus, l0);

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
}
