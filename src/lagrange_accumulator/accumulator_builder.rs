// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Builder functions for constructing Lagrange accumulators (Procedure 9).
//!
//! This module provides:
//! - [`build_accumulators_spartan`]: Optimized builder for Spartan's cubic relation
//! - [`build_accumulators_neutronnova`]: Optimized builder for NeutronNova

use super::{
  accumulator::LagrangeAccumulators,
  domain::LagrangeIndex,
  extension::extend_to_lagrange_domain,
  index::CachedPrefixIndex,
  thread_state::{NeutronNovaThreadState, SpartanThreadState},
};
use crate::{
  csr::Csr,
  polys::{
    eq::{EqPolynomial, compute_suffix_eq_pyramid},
    multilinear::MultilinearPolynomial,
  },
  small_field::{DelayedReduction, SmallValueField, WideMul},
};
use ff::PrimeField;
use num_traits::Zero;
use rayon::prelude::*;
use std::ops::{Add, Sub};

use super::index::compute_idx4;

/// Polynomial degree D for Spartan's small-value sumcheck.
/// For Spartan's cubic relation (A·B - C), D=2 yields quadratic t_i.
pub const SPARTAN_T_DEGREE: usize = 2;

struct EqSplitTables<S: PrimeField> {
  e_in: Vec<S>,
  e_xout: Vec<S>,
  e_y: Vec<Vec<S>>,
}

pub(crate) struct BetaPrefixCache {
  cache: Csr<CachedPrefixIndex>,
  num_betas: usize,
}

/// Procedure 9: Build accumulators A_i(v, u) for Spartan's first sum-check (Algorithm 6).
///
/// Computes accumulators for: g(X) = eq(τ, X) · (Az(X) · Bz(X) - Cz(X))
///
/// D is the degree bound of t_i(X) (not s_i); for Spartan, D = 2.
///
/// # Type Parameters
///
/// - `F`: Field type with small-value and delayed reduction support
/// - `SmallValue`: Witness value type (i32 or i64)
///
/// # Parallelism strategy
///
/// - Outer parallel loop over x_out values (using Rayon fold-reduce)
/// - Each thread maintains thread-local accumulators
/// - Final reduction merges all thread-local results via element-wise addition
///
/// # Spartan-specific optimizations (D=2)
///
/// - Skip cz_ext entirely: for binary betas, use cz_pref directly
/// - Skip binary betas: for satisfying witnesses, Az·Bz = Cz on {0,1}^n, so they contribute 0
/// - Only process betas with ∞ (where Cz doesn't contribute anyway)
pub fn build_accumulators_spartan<F, SmallValue>(
  az: &MultilinearPolynomial<SmallValue>,
  bz: &MultilinearPolynomial<SmallValue>,
  taus: &[F],
  l0: usize,
) -> LagrangeAccumulators<F, 2>
where
  F: PrimeField
    + SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
  SmallValue: WideMul
    + Copy
    + Default
    + Zero
    + Add<Output = SmallValue>
    + Sub<Output = SmallValue>
    + Send
    + Sync,
{
  let base: usize = 3; // D + 1 = 2 + 1 = 3
  let l = az.Z.len().trailing_zeros() as usize;
  debug_assert_eq!(az.Z.len(), 1usize << l, "poly size must be power of 2");
  debug_assert_eq!(az.Z.len(), bz.Z.len());
  debug_assert_eq!(taus.len(), l, "taus must have length ℓ");
  debug_assert!(l0 < l, "l0 must be < ℓ");

  let suffix_vars = l - l0;
  let prefix_size = 1usize << l0;

  let (eq_tables, _in_vars, xout_vars) = precompute_eq_tables(taus, l0);
  let num_x_out = 1usize << xout_vars;
  let BetaPrefixCache {
    cache: beta_prefix_cache,
    num_betas,
  } = build_beta_cache::<2>(l0);

  // Only betas containing at least one ∞ coordinate contribute non-zero values.
  // On binary inputs {0,1}^n, Az·Bz = Cz (R1CS identity), so Az·Bz - Cz = 0.
  // The ∞ coordinate corresponds to the "leading coefficient" of the Lagrange polynomial,
  // which is non-zero only for non-constant polynomials. Non-binary evaluations (those
  // with ∞) are where we accumulate the sumcheck.
  let betas_with_infty: Vec<usize> = (0..num_betas)
    .filter(|&i| (0..l0).any(|d| (i / base.pow(d as u32)).is_multiple_of(base)))
    .collect();

  let ext_size = base.pow(l0 as u32); // (D+1)^l0

  // Build eq cache: precomputes ex * ey products for all combinations.
  let eq_cache: Vec<Vec<F>> = eq_tables
    .e_y
    .iter()
    .map(|round_ey| {
      round_ey
        .iter()
        .flat_map(|ey| eq_tables.e_xout.iter().map(|ex| *ey * *ex))
        .collect()
    })
    .collect();

  // Precompute e_in values.
  let e_in_cache: Vec<F> = eq_tables.e_in.clone();

  // Parallel over x_out with thread-local state (zero per-iteration allocations)
  type State<F2, SV2> = SpartanThreadState<F2, SV2, 2>;

  let fold_results: Vec<State<F, SmallValue>> = (0..num_x_out)
    .into_par_iter()
    .fold(
      || State::<F, SmallValue>::new(l0, num_betas, prefix_size, ext_size),
      |mut state: State<F, SmallValue>, x_out_bits| {
        // Reset partial sums for this x_out iteration
        state.reset_partial_sums();

        // Inner loop over x_in - accumulate into UNREDUCED form
        // Each beta_partial_sums[beta_idx] accumulates 2^(l/2) terms per x_out.
        // Safety bound for SignedWideLimbs<N> (N limbs, 64 bits per limb):
        //   field_bits + product_bits + (l/2) < 64*N
        // i32 path: N=5, product_bits<=62; i64 path: N=6, product_bits<=126.
        for (x_in_bits, e_in_eval) in e_in_cache.iter().enumerate() {
          let suffix = (x_in_bits << xout_vars) | x_out_bits;

          // Fill prefix buffers by index assignment (no allocation)
          #[allow(clippy::needless_range_loop)]
          for prefix in 0..prefix_size {
            let idx = (prefix << suffix_vars) | suffix;
            state.az_prefix_boolean_evals[prefix] = az.Z[idx];
            state.bz_prefix_boolean_evals[prefix] = bz.Z[idx];
          }

          // Extend Az and Bz to Lagrange domain in-place (zero allocation)
          let az_size = extend_to_lagrange_domain::<SmallValue, 2>(
            &state.az_prefix_boolean_evals,
            &mut state.az_extended_evals,
            &mut state.az_extended_scratch,
          );
          let az_ext = &state.az_extended_evals[..az_size];

          let bz_size = extend_to_lagrange_domain::<SmallValue, 2>(
            &state.bz_prefix_boolean_evals,
            &mut state.bz_extended_evals,
            &mut state.bz_extended_scratch,
          );
          let bz_ext = &state.bz_extended_evals[..bz_size];

          // Only process betas with ∞ - binary betas contribute 0 for satisfying witnesses
          // Uses delayed modular reduction: accumulates into unreduced wide-limb form.
          // wide_mul computes small × small → product, then unreduced_multiply_accumulate adds field × product
          for &beta_idx in &betas_with_infty {
            let prod = SmallValue::wide_mul(az_ext[beta_idx], bz_ext[beta_idx]);
            F::unreduced_multiply_accumulate(&mut state.partial_sums[beta_idx], e_in_eval, &prod);
          }
        }

        // Pre-compute and filter: reduce all non-zero betas upfront
        // This eliminates closure call overhead in the accumulator building loop
        // Reuse pre-allocated buffer to avoid per-iteration allocations
        for &beta_idx in &betas_with_infty {
          if state.partial_sums[beta_idx].is_zero() {
            continue;
          }
          // Reduce partial sum to field element
          let val =
            <F as DelayedReduction<SmallValue::Product>>::reduce(&state.partial_sums[beta_idx]);
          if val == F::ZERO {
            continue;
          }
          state.beta_values.push((beta_idx, val));
        }

        // Distribute beta values → A_i(v,u) via idx4 using precomputed eq cache
        // Multiply-accumulate into wide accumulator (Montgomery REDC at end)
        for &(beta_idx, ref val) in &state.beta_values {
          for pref in &beta_prefix_cache[beta_idx] {
            let eq_eval = eq_cache[pref.round_0][pref.y_idx * num_x_out + x_out_bits];
            <F as DelayedReduction<F>>::unreduced_multiply_accumulate(
              &mut state.acc.rounds[pref.round_0].data_mut()[pref.v_idx][pref.u_idx],
              val,
              &eq_eval,
            );
          }
        }

        state
      },
    )
    .collect();

  // Sequential merge: avoids parallel reduce tree overhead and identity allocations.
  // Each fold task's State is merged one by one, spreading deallocation cost.
  // Using std::iter::Iterator::reduce (not rayon's) - no extra state allocations.
  let merged = fold_results
    .into_iter()
    .reduce(|mut a, b| {
      a.acc.merge(&b.acc);
      a
    })
    .expect("num_x_out > 0 guarantees non-empty fold results");

  // Finalize: reduce each bucket from wide 9-limb to field element
  let mut result: LagrangeAccumulators<F, 2> = LagrangeAccumulators::new(l0);
  for (round_idx, round) in merged.acc.rounds.iter().enumerate() {
    for (v_idx, row) in round.data().iter().enumerate() {
      for (u_idx, elem) in row.iter().enumerate() {
        if !elem.is_zero() {
          result.rounds[round_idx].data_mut()[v_idx][u_idx] =
            <F as DelayedReduction<F>>::reduce(elem);
        }
      }
    }
  }
  result
}

/// Build accumulators for NeutronNova's NIFS small-value sumcheck.
///
/// Computes accumulators for: g(b) = Σ_{x} eq(τ, x) · (Ã_b(x) · B̃_b(x) − C̃_b(x))
/// where b ranges over U₂^{ℓ_b} and x ranges over the constraint space.
///
/// Note: Cz is not passed because on the boolean hypercube Az·Bz = Cz (R1CS identity),
/// so the Az·Bz - Cz term is zero for all binary evaluation points. We only need
/// non-binary evaluations (infinity-containing betas) where Cz doesn't contribute.
///
/// Unlike Spartan (which gathers from a single polynomial), NeutronNova gathers
/// across instances: `a_layers[p][x_R * left + x_L]` for each instance p.
///
/// All ℓ_b rounds are Lagrange (l0 = ℓ_b). Uses fused three-way DMR in the inner
/// loop and non-Montgomery scatter with precomputed `e_rb_cache`.
///
/// # Arguments
/// * `a_layers` - Az evaluations per instance, each of length `left * right`
/// * `b_layers` - Bz evaluations per instance
/// * `e_eq` - Pre-computed power polynomial split evals: `[e_left | e_right]` where
///   `e_left[x_l] = tau^{x_l}` and `e_right[x_r] = tau^{x_r * left}`
/// * `left` - Size of left tensor component (from `compute_tensor_decomp`)
/// * `right` - Size of right tensor component (from `compute_tensor_decomp`)
/// * `rhos` - Instance-folding challenges (ρ₁, ..., ρ_{ℓ_b}) for the NIFS folding sumcheck.
///   These are the verifier challenges for folding n = 2^{ℓ_b} instances, not the main
///   constraint-satisfiability sumcheck challenges.
///
/// # Scatter optimization
///
/// All values stay in Montgomery form throughout:
/// 1. `e_rb_cache` is precomputed as Montgomery field elements: `e_right[x_r] * e_b[round][y]`
/// 2. Partial sums are reduced to Montgomery field elements
/// 3. Scatter accumulates Montgomery×Montgomery products into `WideLimbs<9>` (R²-scaled)
/// 4. Final Montgomery REDC per bucket converts back to field elements
pub fn build_accumulators_neutronnova<F, SmallValue>(
  a_layers: &[Vec<SmallValue>],
  b_layers: &[Vec<SmallValue>],
  e_eq: &[F],
  left: usize,
  right: usize,
  rhos: &[F],
) -> LagrangeAccumulators<F, 2>
where
  F: PrimeField
    + SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
  SmallValue: WideMul
    + Copy
    + Default
    + Zero
    + Add<Output = SmallValue>
    + Sub<Output = SmallValue>
    + Send
    + Sync,
{
  let n = a_layers.len();
  let l0 = n.trailing_zeros() as usize; // ℓ_b = log2(n)
  debug_assert_eq!(n, 1 << l0, "number of instances must be power of 2");
  debug_assert_eq!(b_layers.len(), n);
  debug_assert_eq!(rhos.len(), l0);
  debug_assert_eq!(
    e_eq.len(),
    left + right,
    "E_eq must have length left + right"
  );
  debug_assert_eq!(a_layers[0].len(), left * right);

  let base: usize = 3; // D + 1 = 2 + 1 = 3
  let prefix_size = n; // 2^l_b

  // Suffix eq weights over instance-folding challenges ρ.
  let e_b = compute_suffix_eq_pyramid(rhos, l0);

  // Extract e_left and e_right from pre-computed E_eq
  // Uses the same tensor decomposition split as the field path
  let e_left_slice = &e_eq[..left];
  let e_right = &e_eq[left..];

  // Precompute e_rb_cache: e_right[x_r] * e_b[round][y] products.
  // Layout: e_rb_cache[round][y * right + x_r] = e_right[x_r] * e_b[round][y]
  let e_rb_cache: Vec<Vec<F>> = e_b
    .iter()
    .map(|round_ey| {
      round_ey
        .iter()
        .flat_map(|ey| e_right.iter().map(|er| *er * *ey))
        .collect()
    })
    .collect();

  // Build bit-reversal permutation
  let bit_rev: Vec<usize> = (0..prefix_size)
    .map(|p| p.reverse_bits() >> (usize::BITS as usize - l0))
    .collect();
  let ext_size = base.pow(l0 as u32); // 3^l_b

  let BetaPrefixCache {
    cache: beta_prefix_cache,
    num_betas,
  } = build_beta_cache::<2>(l0);

  // Precompute which betas have infinity (same as Spartan builder)
  // Only betas containing at least one ∞ coordinate contribute non-zero values.
  // On binary inputs {0,1}^n, Az·Bz = Cz (R1CS identity), so Az·Bz - Cz = 0.
  // The ∞ coordinate corresponds to the "leading coefficient" of the Lagrange polynomial,
  // which is non-zero only for non-constant polynomials. Non-binary evaluations (those
  // with ∞) are where we accumulate the sumcheck.
  let betas_with_infty: Vec<usize> = (0..num_betas)
    .filter(|&i| (0..l0).any(|d| (i / base.pow(d as u32)).is_multiple_of(base)))
    .collect();

  type State<F2, SV2> = NeutronNovaThreadState<
    F2,
    SV2,
    <F2 as DelayedReduction<<SV2 as WideMul>::Product>>::Accumulator,
    2,
  >;

  let fold_results: Vec<State<F, SmallValue>> = (0..right)
    .into_par_iter()
    .fold(
      || State::<F, SmallValue>::new(l0, num_betas, prefix_size, ext_size),
      |mut state: State<F, SmallValue>, x_r| {
        state.reset_partial_sums();

        for (x_l, &e_l) in e_left_slice.iter().enumerate() {
          // Gather: collect Az_p(x_L, x_R) for each instance p (stays in small value type)
          #[allow(clippy::needless_range_loop)]
          for p in 0..prefix_size {
            let idx = x_r * left + x_l;
            let layer = bit_rev[p];
            state.az_prefix_boolean_evals[p] = a_layers[layer][idx];
            state.bz_prefix_boolean_evals[p] = b_layers[layer][idx];
          }

          // Extend to U₂^{ℓ_b} — integer add/sub only, no field arithmetic
          // Values stay in small value type throughout (safe due to bound check in vec_to_small_for_extension)
          let az_size = extend_to_lagrange_domain::<SmallValue, 2>(
            &state.az_prefix_boolean_evals,
            &mut state.az_extended_evals,
            &mut state.az_extended_scratch,
          );
          let az_ext = &state.az_extended_evals[..az_size];

          let bz_size = extend_to_lagrange_domain::<SmallValue, 2>(
            &state.bz_prefix_boolean_evals,
            &mut state.bz_extended_evals,
            &mut state.bz_extended_scratch,
          );
          let bz_ext = &state.bz_extended_evals[..bz_size];

          // Fused DMR: acc += e_L × az_ext × bz_ext with zero field reductions.
          // wide_mul computes small × small → product, then unreduced_multiply_accumulate adds field × product
          for &beta_idx in &betas_with_infty {
            let prod = SmallValue::wide_mul(az_ext[beta_idx], bz_ext[beta_idx]);
            F::unreduced_multiply_accumulate(&mut state.partial_sums[beta_idx], &e_l, &prod);
          }
        }

        // Reduce partial sums to field elements and filter non-zero
        for &beta_idx in &betas_with_infty {
          let unreduced = &state.partial_sums[beta_idx];
          if !unreduced.is_zero() {
            let val = <F as DelayedReduction<SmallValue::Product>>::reduce(unreduced);
            state.beta_values.push((beta_idx, val));
          }
        }

        // Scatter: multiply-accumulate into wide accumulator (Montgomery REDC at end)
        for &(beta_idx, val) in &state.beta_values {
          for pref in &beta_prefix_cache[beta_idx] {
            let e_rb = e_rb_cache[pref.round_0][pref.y_idx * right + x_r];
            <F as DelayedReduction<F>>::unreduced_multiply_accumulate(
              &mut state.scatter_acc.rounds[pref.round_0].data[pref.v_idx][pref.u_idx],
              &val,
              &e_rb,
            );
          }
        }

        state
      },
    )
    .collect();

  // Merge thread-local scatter accumulators
  let merged = fold_results
    .into_iter()
    .reduce(|mut a, b| {
      a.scatter_acc.merge(&b.scatter_acc);
      a
    })
    .expect("right > 0 guarantees non-empty fold results");

  // Reduce each bucket from wide 9-limb to field element
  merged
    .scatter_acc
    .map(|acc| <F as DelayedReduction<F>>::reduce(acc))
}

// =============================================================================
// Helper functions
// =============================================================================

/// Precompute eq polynomial tables with balanced split for e_in and e_xout.
///
/// Returns (tables, in_vars, xout_vars) where:
/// - in_vars = ceil((l - l0) / 2) - variables for inner loop (e_in)
/// - xout_vars = floor((l - l0) / 2) - variables for outer loop (e_xout)
///
/// The balanced split reduces precomputation cost by ~33% compared to the
/// asymmetric l/2 split, and enables odd number of rounds.
fn precompute_eq_tables<S: PrimeField>(taus: &[S], l0: usize) -> (EqSplitTables<S>, usize, usize) {
  let l = taus.len();
  let suffix_vars = l - l0;
  let in_vars = suffix_vars.div_ceil(2); // ceiling: e_in larger (inner loop, sequential access)
  let xout_vars = suffix_vars - in_vars; // floor: e_xout smaller (outer loop, reused)

  let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + in_vars]); // 2^in_vars entries
  let e_xout = EqPolynomial::evals_from_points(&taus[l0 + in_vars..]); // 2^xout_vars entries
  let e_y = compute_suffix_eq_pyramid(&taus[..l0], l0); // Vec per round, total 2^l0 - 1

  (EqSplitTables { e_in, e_xout, e_y }, in_vars, xout_vars)
}

pub(crate) fn build_beta_cache<const D: usize>(l0: usize) -> BetaPrefixCache {
  let base: usize = D + 1;
  let num_betas = base.pow(l0 as u32);
  let mut cache: Csr<CachedPrefixIndex> = Csr::with_capacity(num_betas, num_betas * l0);
  for b in 0..num_betas {
    let beta = LagrangeIndex::<D>::from_flat_index(b, l0);
    let entries: Vec<_> = compute_idx4(&beta)
      .into_iter()
      .map(|entry| CachedPrefixIndex {
        round_0: entry.round_0idx(),
        v_idx: entry.v_idx,
        u_idx: entry.u.to_index(),
        y_idx: entry.y_idx,
      })
      .collect();
    cache.push(&entries);
  }

  BetaPrefixCache { cache, num_betas }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{lagrange_accumulator::domain::LagrangeHatPoint, provider::pasta::pallas};
  use ff::Field;

  type Scalar = pallas::Scalar;

  // Use the shared constant for polynomial degree in tests
  const D: usize = SPARTAN_T_DEGREE;

  /// Binary-β zero shortcut: Az=Bz=Cz=first variable (x0), so Az·Bz−Cz=0 on binary β.
  /// Non-binary β (∞) should yield non-zero in some bucket.
  #[test]
  fn test_binary_beta_zero_shortcut_behavior() {
    // Use l0=1 so round 0 buckets are fed only by β of length 1 (easy to reason about).
    let l0 = 1;
    let l = 2;

    // Az = Bz = top bit x0 (most significant of 2 bits)
    // For satisfying witness, Cz = Az * Bz = Az (since Az ∈ {0,1} and Az = Bz)
    let az_vals: Vec<i32> = (0..(1 << l)).map(|bits| (bits >> (l - 1)) & 1).collect();
    let bz_vals = az_vals.clone();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    let taus: Vec<Scalar> = vec![Scalar::from(3u64), Scalar::from(5u64)];

    let acc = build_accumulators_spartan(&az, &bz, &taus, l0);

    // Only round 0 exists (v is empty). β ranges over U_d with binary {0,1} and non-binary {∞}.
    // Buckets for u = 0 should be zero (binary β), bucket for u = ∞ should be non-zero.
    let u_inf = LagrangeHatPoint::<D>::Infinity.to_index(); // 0
    let u_zero = LagrangeHatPoint::<D>::Finite(0).to_index(); // 1

    assert!(
      bool::from(acc.get(0, 0, u_zero).is_zero()),
      "binary β should give zero for u=0"
    );
    assert!(
      !bool::from(acc.get(0, 0, u_inf).is_zero()),
      "non-binary β (∞) should give non-zero"
    );
  }

  /// Test build_accumulators_spartan with i32 witnesses produces consistent results.
  ///
  /// Verifies that running the same computation twice produces the same output.
  #[test]
  fn test_build_accumulators_spartan_small_consistent() {
    let l0 = 2;

    // Define deterministic Az, Bz over {0,1}^4 using small values
    let eval = |bits: usize| -> i32 {
      let x0 = (bits >> 3) & 1;
      let x1 = (bits >> 2) & 1;
      let x2 = (bits >> 1) & 1;
      let x3 = bits & 1;
      (x0 + 2 * x1 + 3 * x2 + 4 * x3 + 5) as i32
    };

    let az_vals: Vec<i32> = (0..16).map(&eval).collect();
    let bz_vals: Vec<i32> = (0..16).map(|b| eval(b) + 7).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    // Taus (length ℓ)
    let taus: Vec<Scalar> = vec![
      Scalar::from(5u64),
      Scalar::from(7u64),
      Scalar::from(11u64),
      Scalar::from(13u64),
    ];

    // Build accumulators twice
    let acc1 = build_accumulators_spartan(&az, &bz, &taus, l0);
    let acc2 = build_accumulators_spartan(&az, &bz, &taus, l0);

    // Compare all buckets
    for round in 0..l0 {
      let num_v = (D + 1).pow(round as u32);
      for v_idx in 0..num_v {
        for u_idx in 0..D {
          let got = acc1.get(round, v_idx, u_idx);
          let expect = acc2.get(round, v_idx, u_idx);
          assert_eq!(
            got, expect,
            "Mismatch at round {}, v_idx {}, u_idx {}",
            round, v_idx, u_idx
          );
        }
      }
    }
  }

  /// Test build_accumulators_spartan with i32 witnesses using larger inputs to stress test.
  #[test]
  fn test_build_accumulators_spartan_small_larger() {
    let l0 = 3;
    let l = 10;
    let n = 1 << l;

    // Create polynomials with varying small values
    let az_vals: Vec<i32> = (0..n).map(|i| (i % 1000) + 1).collect();
    let bz_vals: Vec<i32> = (0..n).map(|i| ((i * 7) % 1000) + 1).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    // Random-looking taus
    let taus: Vec<Scalar> = (0..l).map(|i| Scalar::from((i * 7 + 3) as u64)).collect();

    // Build accumulators twice to verify consistency
    let acc1 = build_accumulators_spartan(&az, &bz, &taus, l0);
    let acc2 = build_accumulators_spartan(&az, &bz, &taus, l0);

    for round in 0..l0 {
      let num_v = (D + 1).pow(round as u32);
      for v_idx in 0..num_v {
        for u_idx in 0..D {
          let got = acc1.get(round, v_idx, u_idx);
          let expect = acc2.get(round, v_idx, u_idx);
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
