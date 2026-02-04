// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Builder functions for constructing Lagrange accumulators (Procedure 9).
//!
//! This module provides:
//! - [`build_accumulators_spartan`]: Optimized builder for Spartan's cubic relation
//! - [`build_accumulators`]: Generic builder for arbitrary polynomial products

use super::{
  accumulator::LagrangeAccumulators,
  domain::LagrangeIndex,
  extension::LagrangeEvaluatedMultilinearPolynomial,
  index::CachedPrefixIndex,
  thread_state::{GenericThreadState, NeutronNovaThreadState, SpartanThreadState},
};
use crate::{
  csr::Csr,
  polys::{
    eq::{EqPolynomial, compute_suffix_eq_pyramid},
    multilinear::MultilinearPolynomial,
  },
};
use crate::small_field::{DelayedReduction, SmallValueField};
use std::{
  fmt::Debug,
  ops::{Add, AddAssign, Neg, Sub, SubAssign},
};
use ff::PrimeField;
use num_traits::Zero;
use rayon::prelude::*;

use super::index::compute_idx4;

/// Polynomial degree D for Spartan's small-value sumcheck.
/// For Spartan's cubic relation (A·B - C), D=2 yields quadratic t_i.
pub const SPARTAN_T_DEGREE: usize = 2;

struct EqSplitTables<S: PrimeField> {
  e_in: Vec<S>,
  e_xout: Vec<S>,
  e_y: Vec<Vec<S>>,
  e_y_sizes: Vec<usize>,
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
/// - `S`: Field type implementing `DelayedReduction<V>`
/// - `V`: Witness value type (i32 or i64)
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
pub fn build_accumulators_spartan<S, V>(
  az: &MultilinearPolynomial<V>,
  bz: &MultilinearPolynomial<V>,
  taus: &[S],
  l0: usize,
) -> LagrangeAccumulators<S, 2>
where
  S: PrimeField + DelayedReduction<V> + Send + Sync,
  V: Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + Eq
    + Add<Output = V>
    + Sub<Output = V>
    + Neg<Output = V>
    + AddAssign
    + SubAssign
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

  let betas_with_infty: Vec<usize> = (0..num_betas)
    .filter(|&i| (0..l0).any(|d| (i / base.pow(d as u32)).is_multiple_of(base)))
    .collect();

  let ext_size = base.pow(l0 as u32); // (D+1)^l0

  // Build eq cache: precomputes ex * ey as non-R-scaled raw limbs for all combinations.
  // This eliminates Montgomery multiplications from the scatter hot path.
  let eq_cache: Vec<Vec<S::UnreducedField>> = eq_tables
    .e_y
    .iter()
    .map(|round_ey| {
      round_ey
        .iter()
        .flat_map(|ey| eq_tables.e_xout.iter().map(|ex| (*ey * *ex).to_unreduced()))
        .collect()
    })
    .collect();

  // Precompute e_in cache as non-R-scaled raw limbs.
  // This eliminates from_mont conversion in the inner loop reduction.
  let e_in_cache: Vec<S::UnreducedField> = eq_tables.e_in.iter().map(|e| e.to_unreduced()).collect();

  // Parallel over x_out with thread-local state (zero per-iteration allocations)
  type State<S, V> = SpartanThreadState<S, V, 2>;

  let fold_results: Vec<State<S, V>> = (0..num_x_out)
    .into_par_iter()
    .fold(
      || State::<S, V>::new(l0, num_betas, prefix_size, ext_size),
      |mut state: State<S, V>, x_out_bits| {
        // Reset partial sums for this x_out iteration
        state.reset_partial_sums();

        // Inner loop over x_in - accumulate into UNREDUCED form
        // Each beta_partial_sums[beta_idx] accumulates 2^(l/2) terms per x_out.
        // Safety bound for UnreducedFieldInt (N limbs, 64 bits per limb):
        //   field_bits + product_bits + (l/2) < 64*N
        // i32 path: N=6, product_bits<=62; i64 path: N=8, product_bits<=126.
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
          let az_size = LagrangeEvaluatedMultilinearPolynomial::<V, 2>::extend_in_place(
            &state.az_prefix_boolean_evals,
            &mut state.az_extended_evals,
            &mut state.az_extended_scratch,
          );
          let az_ext = &state.az_extended_evals[..az_size];

          let bz_size = LagrangeEvaluatedMultilinearPolynomial::<V, 2>::extend_in_place(
            &state.bz_prefix_boolean_evals,
            &mut state.bz_extended_evals,
            &mut state.bz_extended_scratch,
          );
          let bz_ext = &state.bz_extended_evals[..bz_size];

          // Only process betas with ∞ - binary betas contribute 0 for satisfying witnesses
          // Uses delayed modular reduction: accumulates into unreduced wide-limb form.
          // Pass small values directly for raw limb accumulation (no pre-multiplication)
          for &beta_idx in &betas_with_infty {
            S::accumulate_field_small_small_products(
              &mut state.partial_sums[beta_idx],
              e_in_eval,
              az_ext[beta_idx],
              bz_ext[beta_idx],
            );
          }
        }

        // Pre-compute and filter: reduce all non-zero betas upfront
        // This eliminates closure call overhead in the accumulator building loop
        // Reuse pre-allocated buffer to avoid per-iteration allocations
        for &beta_idx in &betas_with_infty {
          if state.partial_sums[beta_idx].is_zero() {
            continue;
          }
          // Barrett-reduce to non-R-scaled raw limbs (accumulator is already non-R-scaled)
          let val = S::reduce_raw_field_int_to_unreduced(&state.partial_sums[beta_idx]);
          if val == S::UnreducedField::default() {
            continue;
          }
          state.beta_values.push((beta_idx, val));
        }

        // Distribute beta values → A_i(v,u) via idx4 using precomputed eq cache
        // Raw limb multiply-accumulate (no Montgomery ops), final Barrett reduction after merge
        for &(beta_idx, ref val) in &state.beta_values {
          for pref in &beta_prefix_cache[beta_idx] {
            let eq_eval = &eq_cache[pref.round_0][pref.y_idx * num_x_out + x_out_bits];
            S::accumulate_raw_field_field_products(
              &mut state.acc.rounds[pref.round_0].data_mut()[pref.v_idx][pref.u_idx],
              val,
              eq_eval,
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

  // Finalize: Barrett-reduce each bucket from raw 9-limb to field element
  let mut result: LagrangeAccumulators<S, 2> = LagrangeAccumulators::new(l0);
  for (round_idx, round) in merged.acc.rounds.iter().enumerate() {
    for (v_idx, row) in round.data().iter().enumerate() {
      for (u_idx, elem) in row.iter().enumerate() {
        if !elem.is_zero() {
          result.rounds[round_idx].data_mut()[v_idx][u_idx] = S::reduce_unreduced_field_field(elem);
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
/// * `rhos` - Instance-folding challenges (ρ₁, ..., ρ_{ℓ_b}), one per sumcheck round over instance index b
///
/// # Scatter optimization
///
/// All values stay in Montgomery form throughout (no conversions to raw limbs):
/// 1. `e_rb_cache` is precomputed as Montgomery field elements: `e_right[x_r] * e_b[round][y]`
/// 2. Partial sums are reduced to Montgomery field elements (`UnreducedField = Self`)
/// 3. Scatter accumulates Montgomery×Montgomery products into `WideLimbs<9>` (R²-scaled)
/// 4. Final Montgomery REDC per bucket converts back to field elements
pub fn build_accumulators_neutronnova<S>(
  a_layers: &[Vec<i64>],
  b_layers: &[Vec<i64>],
  e_eq: &[S],
  left: usize,
  right: usize,
  rhos: &[S],
) -> LagrangeAccumulators<S, 2>
where
  S: PrimeField + SmallValueField<i64, IntermediateSmallValue = i128> + DelayedReduction<i64, IntermediateSmallValue = i128> + Send + Sync,
{
  let n = a_layers.len();
  let l0 = n.trailing_zeros() as usize; // ℓ_b = log2(n)
  debug_assert_eq!(n, 1 << l0, "number of instances must be power of 2");
  debug_assert_eq!(b_layers.len(), n);
  debug_assert_eq!(rhos.len(), l0);
  debug_assert_eq!(e_eq.len(), left + right, "E_eq must have length left + right");
  debug_assert_eq!(a_layers[0].len(), left * right);

  let base: usize = 3; // D + 1 = 2 + 1 = 3
  let prefix_size = n; // 2^l_b

  // Suffix eq weights over instance-folding challenges ρ.
  let e_b = compute_suffix_eq_pyramid(rhos, l0);

  // Extract e_left and e_right from pre-computed E_eq
  // Uses the same tensor decomposition split as the field path
  let e_left_slice = &e_eq[..left];
  let e_right = &e_eq[left..];

  // Precompute e_rb_cache: non-R-scaled raw limbs of e_right[x_r] * e_b[round][y].
  // Layout: e_rb_cache[round][y * right + x_r] = from_mont(e_right[x_r] * e_b[round][y])
  let e_rb_cache: Vec<Vec<S::UnreducedField>> = e_b
    .iter()
    .map(|round_ey| {
      round_ey
        .iter()
        .flat_map(|ey| e_right.iter().map(|er| (*er * *ey).to_unreduced()))
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
  let betas_with_infty: Vec<usize> = (0..num_betas)
    .filter(|&i| (0..l0).any(|d| (i / base.pow(d as u32)).is_multiple_of(base)))
    .collect();

  type State<S> = NeutronNovaThreadState<S, i128, <S as DelayedReduction<i64>>::UnreducedFieldInt, 2>;

  let fold_results: Vec<State<S>> = (0..right)
    .into_par_iter()
    .fold(
      || State::<S>::new(l0, num_betas, prefix_size, ext_size),
      |mut state: State<S>, x_r| {
        state.reset_partial_sums();

        for (x_l, &e_l) in e_left_slice.iter().enumerate() {
          // Gather: collect Az_p(x_L, x_R) for each instance p, widen to IntermediateSmallValue
          #[allow(clippy::needless_range_loop)]
          for p in 0..prefix_size {
            let idx = x_r * left + x_l;
            let layer = bit_rev[p];
            state.az_prefix_boolean_evals[p] = S::small_to_intermediate(a_layers[layer][idx]);
            state.bz_prefix_boolean_evals[p] = S::small_to_intermediate(b_layers[layer][idx]);
          }

          // Extend to U₂^{ℓ_b} — integer add/sub only, no field arithmetic
          let az_size =
            LagrangeEvaluatedMultilinearPolynomial::<S::IntermediateSmallValue, 2>::extend_in_place(
              &state.az_prefix_boolean_evals,
              &mut state.az_extended_evals,
              &mut state.az_extended_scratch,
            );
          let az_ext = &state.az_extended_evals[..az_size];

          let bz_size =
            LagrangeEvaluatedMultilinearPolynomial::<S::IntermediateSmallValue, 2>::extend_in_place(
              &state.bz_prefix_boolean_evals,
              &mut state.bz_extended_evals,
              &mut state.bz_extended_scratch,
            );
          let bz_ext = &state.bz_extended_evals[..bz_size];

          // Fused DMR: acc += e_L × az_ext × bz_ext with zero field reductions.
          for &beta_idx in &betas_with_infty {
            S::unreduced_field_ext_mul_add(
              &mut state.partial_sums[beta_idx],
              &e_l,
              az_ext[beta_idx],
              bz_ext[beta_idx],
            );
          }
        }

        // Reduce partial sums to non-R-scaled raw limbs and filter non-zero
        for &beta_idx in &betas_with_infty {
          let unreduced = &state.partial_sums[beta_idx];
          if !unreduced.is_zero() {
            let val = S::reduce_field_int_to_unreduced(unreduced);
            state.beta_values.push((beta_idx, val));
          }
        }

        // Scatter: raw limb multiply-accumulate (no Montgomery multiplications)
        // val is non-R-scaled, e_rb_cache is non-R-scaled → product is non-R-scaled
        for &(beta_idx, val) in &state.beta_values {
          for pref in &beta_prefix_cache[beta_idx] {
            let e_rb = &e_rb_cache[pref.round_0][pref.y_idx * right + x_r];
            S::accumulate_raw_field_field_products(
              &mut state.scatter_acc.rounds[pref.round_0].data[pref.v_idx][pref.u_idx],
              &val,
              e_rb,
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

  // Barrett-reduce each bucket from raw 9-limb to field element
  merged.scatter_acc.map(|acc| S::reduce_unreduced_field_field(acc))
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
#[allow(dead_code)]
pub fn build_accumulators<S: PrimeField + Send + Sync, const D: usize>(
  polys: &[&MultilinearPolynomial<S>],
  taus: &[S],
  l0: usize,
) -> LagrangeAccumulators<S, D> {
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
  debug_assert!(l0 < l, "l0 must be < ℓ");

  let suffix_vars = l - l0;
  let prefix_size = 1usize << l0;
  let d = polys.len();

  let (eq_tables, in_vars, xout_vars) = precompute_eq_tables(taus, l0);
  let num_x_out = 1usize << xout_vars;
  let _num_x_in = 1usize << in_vars;
  let BetaPrefixCache {
    cache: beta_prefix_cache,
    num_betas,
  } = build_beta_cache::<D>(l0);

  let ext_size = base.pow(l0 as u32);

  // Parallel over x_out with thread-local state (zero per-iteration allocations)
  (0..num_x_out)
    .into_par_iter()
    .fold(
      || {
        GenericThreadState::<S, D>::new(
          l0,
          num_betas,
          prefix_size,
          ext_size,
          d,
          &eq_tables.e_y_sizes,
        )
      },
      |mut state, x_out_bits| {
        // Reset partial sums for this x_out iteration
        state.reset_partial_sums();

        // Compute eyx = ey * ex on-the-fly for this x_out_bits (tiny, stays hot in L1)
        let ex = eq_tables.e_xout[x_out_bits];
        fill_eyx(ex, &eq_tables.e_y, &mut state.eyx);

        // Inner loop over x_in
        for (x_in_bits, &e_in_eval) in eq_tables.e_in.iter().enumerate() {
          let suffix = (x_in_bits << xout_vars) | x_out_bits;

          // Fill all d prefix buffers by index assignment
          #[allow(clippy::needless_range_loop)]
          for prefix in 0..prefix_size {
            let idx = (prefix << suffix_vars) | suffix;
            for (k, poly) in polys.iter().enumerate() {
              state.poly_prefs[k][prefix] = poly.Z[idx];
            }
          }

          // Extend all d polynomials in-place (zero allocation)
          // Result is always in buf_curr (first element of each pair)
          for (pref, (buf_curr, buf_scratch)) in
            state.poly_prefs.iter().zip(state.buf_pairs.iter_mut())
          {
            LagrangeEvaluatedMultilinearPolynomial::<S, D>::extend_in_place(
              pref,
              buf_curr,
              buf_scratch,
            );
          }

          // Compute ∏ p_k(β) for each beta
          for (beta_idx, sum) in state.beta_partial_sums.iter_mut().enumerate() {
            let prod: S = state
              .buf_pairs
              .iter()
              .map(|(buf_curr, _)| buf_curr[beta_idx])
              .product();
            *sum += e_in_eval * prod;
          }
        }

        // Distribute beta_partial_sums → A_i(v,u) via idx4
        scatter_beta_contributions(
          0..num_betas,
          &beta_prefix_cache,
          &state.eyx,
          &mut state.acc,
          |beta_idx| {
            let val = state.beta_partial_sums[beta_idx];
            if ff::Field::is_zero(&val).into() {
              None
            } else {
              Some(val)
            }
          },
        );

        state
      },
    )
    .map(|state| state.acc)
    .reduce(
      || LagrangeAccumulators::<S, D>::new(l0),
      |mut a, b| {
        a.merge(&b);
        a
      },
    )
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
  let e_y_sizes: Vec<usize> = e_y.iter().map(|v| v.len()).collect();

  (
    EqSplitTables {
      e_in,
      e_xout,
      e_y,
      e_y_sizes,
    },
    in_vars,
    xout_vars,
  )
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

#[inline]
fn fill_eyx<S: PrimeField>(ex: S, e_y: &[Vec<S>], eyx: &mut [Vec<S>]) {
  debug_assert_eq!(e_y.len(), eyx.len());
  for (round, ey_round) in e_y.iter().enumerate() {
    let dst = &mut eyx[round];
    debug_assert_eq!(dst.len(), ey_round.len());
    for (dst_i, &ey) in dst.iter_mut().zip(ey_round.iter()) {
      *dst_i = ey * ex;
    }
  }
}

/// Legacy scatter function using eyx precomputation with immediate F×F reduction.
/// Each contribution does F×F multiply with internal Montgomery reduction.
#[allow(dead_code)] // Kept for reference; new code uses scatter_beta_contributions_unreduced
#[inline]
fn scatter_beta_contributions<S: PrimeField, const D: usize, I, F>(
  beta_indices: I,
  beta_prefix_cache: &Csr<CachedPrefixIndex>,
  eyx: &[Vec<S>],
  acc: &mut LagrangeAccumulators<S, D>,
  mut value_for_beta: F,
) where
  I: IntoIterator<Item = usize>,
  F: FnMut(usize) -> Option<S>,
{
  for beta_idx in beta_indices {
    let Some(val) = value_for_beta(beta_idx) else {
      continue;
    };
    for pref in &beta_prefix_cache[beta_idx] {
      let eyx_val = eyx[pref.round_0][pref.y_idx];
      acc.accumulate(pref.round_0, pref.v_idx, pref.u_idx, eyx_val * val);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    lagrange_accumulator::domain::LagrangeHatPoint,
    polys::eq::EqPolynomial,
    provider::pasta::pallas,
  };
  use ff::Field;

  type Scalar = pallas::Scalar;

  // Use the shared constant for polynomial degree in tests
  const D: usize = SPARTAN_T_DEGREE;

  /// End-to-end correctness for build_accumulators_spartan on a tiny instance.
  ///
  /// ℓ = 4, ℓ0 = 2, D = 2.
  /// Uses a satisfying witness (Az * Bz = Cz) to test the optimized Spartan path.
  /// Verifies against a straightforward (non-parallel) implementation of Procedure 9.
  #[test]
  fn test_build_accumulators_spartan_matches_naive() {
    let l0 = 2;
    let l = 4;

    // Balanced split for eq tables (matches precompute_eq_tables)
    let suffix_vars = l - l0; // 2
    let in_vars = (suffix_vars + 1) / 2; // 1
    let xout_vars = suffix_vars - in_vars; // 1

    // Define deterministic Az, Bz, Cz over {0,1}^4 using small values
    // Use a SATISFYING witness: Cz = Az * Bz
    let eval = |bits: usize| -> i32 {
      // Simple affine: a0 x0 + a1 x1 + a2 x2 + a3 x3 + const
      let x0 = (bits >> 3) & 1;
      let x1 = (bits >> 2) & 1;
      let x2 = (bits >> 1) & 1;
      let x3 = bits & 1;
      (x0 + 2 * x1 + 3 * x2 + 4 * x3 + 5) as i32
    };
    let az_vals: Vec<i32> = (0..16).map(eval).collect();
    let bz_vals: Vec<i32> = (0..16).map(|b| eval(b) + 7).collect();
    // Satisfying witness: cz = az * bz (for naive reference)
    let cz_vals: Vec<Scalar> = az_vals
      .iter()
      .zip(bz_vals.iter())
      .map(|(a, b)| Scalar::from((*a as i64 * *b as i64) as u64))
      .collect();

    let az = MultilinearPolynomial::new(az_vals.clone());
    let bz = MultilinearPolynomial::new(bz_vals.clone());
    let cz = MultilinearPolynomial::new(cz_vals.clone());

    // Convert to field for naive reference computation
    let az_field: Vec<Scalar> = az_vals.iter().map(|&v| Scalar::from(v as u64)).collect();
    let bz_field: Vec<Scalar> = bz_vals.iter().map(|&v| Scalar::from(v as u64)).collect();
    let az_poly = MultilinearPolynomial::new(az_field);
    let bz_poly = MultilinearPolynomial::new(bz_field);

    // Taus (length ℓ)
    let taus: Vec<Scalar> = vec![
      Scalar::from(5u64),
      Scalar::from(7u64),
      Scalar::from(11u64),
      Scalar::from(13u64),
    ];

    // Implementation under test
    let acc_impl = build_accumulators_spartan(&az, &bz, &taus, l0);

    // Precompute eq tables for naive computation (balanced split)
    let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + in_vars]); // τ[2..3]
    let e_xout = EqPolynomial::evals_from_points(&taus[l0 + in_vars..]); // τ[3..4]
    let e_y = compute_suffix_eq_pyramid(&taus[..l0], l0);

    let num_betas = (D + 1).pow(l0 as u32);
    let idx4_cache: Vec<Vec<_>> = (0..num_betas)
      .map(|b| compute_idx4(&LagrangeIndex::<D>::from_flat_index(b, l0)))
      .collect();

    // Naive accumulators
    let mut acc_naive: LagrangeAccumulators<Scalar, D> = LagrangeAccumulators::new(l0);

    // Iterate over x_out and x_in with balanced split
    #[allow(clippy::needless_range_loop)]
    for x_out_bits in 0..(1 << xout_vars) {
      let ex = e_xout[x_out_bits];

      for x_in_bits in 0..(1 << in_vars) {
        let suffix = (x_in_bits << xout_vars) | x_out_bits;

        let az_prefix_boolean_evals = az_poly.gather_prefix_evals(l0, suffix);
        let bz_prefix_boolean_evals = bz_poly.gather_prefix_evals(l0, suffix);
        let cz_pref = cz.gather_prefix_evals(l0, suffix);

        let az_ext =
          LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&az_prefix_boolean_evals);
        let bz_ext =
          LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&bz_prefix_boolean_evals);
        let cz_ext =
          LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&cz_pref);

        let e_in_eval = e_in[x_in_bits];

        #[allow(clippy::needless_range_loop)]
        for beta_idx in 0..num_betas {
          let beta_tuple = az_ext.to_domain_tuple(beta_idx);
          let ab = az_ext.get(beta_idx) * bz_ext.get(beta_idx);
          let prod = if beta_tuple.has_infinity() {
            ab
          } else {
            ab - cz_ext.get(beta_idx)
          };
          let val = e_in_eval * prod;

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

    // Balanced split for eq tables
    let suffix_vars = l - l0; // 1
    let in_vars = (suffix_vars + 1) / 2; // 1

    // Two constant polynomials: 1 and -1, product = -1
    let ones = MultilinearPolynomial::new(vec![Scalar::ONE; 1 << l]);
    let neg_ones = MultilinearPolynomial::new(vec![-Scalar::ONE; 1 << l]);

    let taus: Vec<Scalar> = vec![Scalar::from(5u64), Scalar::from(7u64)];

    // Use generic build_accumulators with D=2 for product of two polynomials
    let acc = build_accumulators::<Scalar, 2>(&[&ones, &neg_ones], &taus, l0);

    // Compute e_in sum = Σ eq(τ[l0..l0+in_vars], xin), here in_vars=1, l0=1 -> slice τ[1..2]
    let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + in_vars]);
    let e_in_eval_sum: Scalar = e_in.iter().copied().sum();

    // Round 0, v_idx=0
    let u_infinity_idx = LagrangeHatPoint::<2>::Infinity.to_index(); // 0
    let u_zero_idx = LagrangeHatPoint::<2>::Finite(0).to_index(); // 1

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
      e_in_eval_sum * (-Scalar::ONE),
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
    let az_vals: Vec<i32> = (0..(1 << l))
      .map(|bits| {
        let x0 = (bits >> (l - 1)) & 1;
        x0 as i32
      })
      .collect();
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

  /// Test generic build_accumulators (Procedure 9) with a product of 3 polynomials.
  ///
  /// ℓ = 10, ℓ0 = 3, D = 3 (degree bound for product of 3 polynomials).
  /// Verifies that accumulators are computed correctly by comparing against naive computation.
  #[test]
  fn test_build_accumulators_product_of_three() {
    use rand::{SeedableRng, rngs::StdRng};

    const L: usize = 10;
    const L0: usize = 3;
    const D: usize = 3; // Degree bound for product of 3 linear polynomials

    let n = 1usize << L;

    // Balanced split for eq tables (matches precompute_eq_tables)
    let suffix_vars = L - L0; // 7
    let in_vars = (suffix_vars + 1) / 2; // 4
    let xout_vars = suffix_vars - in_vars; // 3

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

    // ===== Naive computation for comparison (balanced split) =====
    let e_in = EqPolynomial::evals_from_points(&taus[L0..L0 + in_vars]);
    let e_xout = EqPolynomial::evals_from_points(&taus[L0 + in_vars..]);
    let e_y = compute_suffix_eq_pyramid(&taus[..L0], L0);

    let idx4_cache: Vec<Vec<_>> = (0..num_betas)
      .map(|b| compute_idx4(&LagrangeIndex::<D>::from_flat_index(b, L0)))
      .collect();

    let mut acc_naive: LagrangeAccumulators<Scalar, D> = LagrangeAccumulators::new(L0);

    #[allow(clippy::needless_range_loop)]
    for x_out_bits in 0..(1 << xout_vars) {
      let ex = e_xout[x_out_bits];

      #[allow(clippy::needless_range_loop)]
      for x_in_bits in 0..(1 << in_vars) {
        let suffix = (x_in_bits << xout_vars) | x_out_bits;

        // Gather prefix evaluations and extend to Lagrange domain
        let p1_pref = p1.gather_prefix_evals(L0, suffix);
        let p2_pref = p2.gather_prefix_evals(L0, suffix);
        let p3_pref = p3.gather_prefix_evals(L0, suffix);

        let p1_ext =
          LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&p1_pref);
        let p2_ext =
          LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&p2_pref);
        let p3_ext =
          LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&p3_pref);

        let e_in_eval = e_in[x_in_bits];

        #[allow(clippy::needless_range_loop)]
        for beta_idx in 0..num_betas {
          // Compute product p1(β) * p2(β) * p3(β)
          let prod = p1_ext.get(beta_idx) * p2_ext.get(beta_idx) * p3_ext.get(beta_idx);
          let val = e_in_eval * prod;

          // Distribute to accumulators via idx4
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
    let az_vals: Vec<i32> = (0..n).map(|i| ((i % 1000) + 1) as i32).collect();
    let bz_vals: Vec<i32> = (0..n).map(|i| (((i * 7) % 1000) + 1) as i32).collect();

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

  /// Test that odd number of rounds works correctly with balanced split.
  ///
  /// ℓ = 11 (odd), ℓ0 = 3, D = 2.
  /// This tests the new balanced split which enables odd rounds.
  /// With l=11, l0=3: suffix_vars=8, in_vars=4, xout_vars=4 (perfectly balanced!)
  #[test]
  fn test_build_accumulators_odd_rounds() {
    use rand::{SeedableRng, rngs::StdRng};

    const L: usize = 11; // Odd number of rounds
    const L0: usize = 3;

    let n = 1usize << L;

    // Balanced split for eq tables
    let suffix_vars = L - L0; // 8
    let in_vars = (suffix_vars + 1) / 2; // 4
    let xout_vars = suffix_vars - in_vars; // 4

    let num_betas = (D + 1).pow(L0 as u32);

    let mut rng = StdRng::seed_from_u64(123);

    // Create witnesses using small values (i32) for the optimized path
    // Use deterministic small values for reproducibility
    let az_vals: Vec<i32> = (0..n).map(|i| ((i % 1000) as i32) - 500).collect();
    let bz_vals: Vec<i32> = (0..n).map(|i| (((i * 7) % 1000) as i32) - 500).collect();

    let az = MultilinearPolynomial::new(az_vals.clone());
    let bz = MultilinearPolynomial::new(bz_vals.clone());

    // Field versions for naive reference
    let az_field: Vec<Scalar> = az_vals
      .iter()
      .map(|&v| {
        if v >= 0 {
          Scalar::from(v as u64)
        } else {
          -Scalar::from((-v) as u64)
        }
      })
      .collect();
    let bz_field: Vec<Scalar> = bz_vals
      .iter()
      .map(|&v| {
        if v >= 0 {
          Scalar::from(v as u64)
        } else {
          -Scalar::from((-v) as u64)
        }
      })
      .collect();

    // Satisfying witness: cz = az * bz (for naive reference)
    let cz_vals: Vec<Scalar> = az_field
      .iter()
      .zip(bz_field.iter())
      .map(|(a, b)| *a * *b)
      .collect();

    let az_poly = MultilinearPolynomial::new(az_field);
    let bz_poly = MultilinearPolynomial::new(bz_field);
    let cz = MultilinearPolynomial::new(cz_vals);

    // Random taus of length 11 (odd)
    let taus: Vec<Scalar> = (0..L).map(|_| Scalar::random(&mut rng)).collect();

    // Build accumulators using optimized Spartan path
    let acc_impl = build_accumulators_spartan(&az, &bz, &taus, L0);

    // ===== Naive computation for comparison (balanced split) =====
    let e_in = EqPolynomial::evals_from_points(&taus[L0..L0 + in_vars]);
    let e_xout = EqPolynomial::evals_from_points(&taus[L0 + in_vars..]);
    let e_y = compute_suffix_eq_pyramid(&taus[..L0], L0);

    let idx4_cache: Vec<Vec<_>> = (0..num_betas)
      .map(|b| compute_idx4(&LagrangeIndex::<D>::from_flat_index(b, L0)))
      .collect();

    let mut acc_naive: LagrangeAccumulators<Scalar, D> = LagrangeAccumulators::new(L0);

    #[allow(clippy::needless_range_loop)]
    for x_out_bits in 0..(1 << xout_vars) {
      let ex = e_xout[x_out_bits];

      for x_in_bits in 0..(1 << in_vars) {
        let suffix = (x_in_bits << xout_vars) | x_out_bits;

        let az_prefix_boolean_evals = az_poly.gather_prefix_evals(L0, suffix);
        let bz_prefix_boolean_evals = bz_poly.gather_prefix_evals(L0, suffix);
        let cz_pref = cz.gather_prefix_evals(L0, suffix);

        let az_ext =
          LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&az_prefix_boolean_evals);
        let bz_ext =
          LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&bz_prefix_boolean_evals);
        let cz_ext =
          LagrangeEvaluatedMultilinearPolynomial::<Scalar, D>::from_multilinear(&cz_pref);

        let e_in_eval = e_in[x_in_bits];

        #[allow(clippy::needless_range_loop)]
        for beta_idx in 0..num_betas {
          let beta_tuple = az_ext.to_domain_tuple(beta_idx);
          let ab = az_ext.get(beta_idx) * bz_ext.get(beta_idx);
          let prod = if beta_tuple.has_infinity() {
            ab
          } else {
            ab - cz_ext.get(beta_idx)
          };
          let val = e_in_eval * prod;

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
            "Mismatch at round {}, v_idx {}, u_idx {} for odd ℓ={}",
            round, v_idx, u_idx, L
          );
        }
      }
    }
  }
}
