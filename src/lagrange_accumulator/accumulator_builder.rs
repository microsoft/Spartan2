// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Builder functions for constructing Lagrange accumulators (Procedure 9).
//!
//! This module provides:
//! - [`build_accumulators_spartan_satisfying`]: Optimized builder for Spartan's
//!   satisfying-witness cubic relation

use super::{
  accumulator::LagrangeAccumulators, csr::Csr, domain::LagrangeIndex,
  extension::extend_to_lagrange_domain, index::AccumulatorPrefixIndex,
  thread_state::SpartanThreadState,
};
use crate::{
  big_num::{DelayedReduction, SmallValue, SmallValueEngine},
  polys::{eq::build_eq_pyramid, eq::compute_suffix_eq_pyramid, multilinear::MultilinearPolynomial},
};
use ff::PrimeField;
use num_traits::Zero;
use rayon::prelude::*;

use super::index::compute_idx4;

/// Polynomial degree D for Spartan's small-value sumcheck.
/// For Spartan's cubic relation (A·B - C), D=2 yields quadratic t_i.
pub const SPARTAN_T_DEGREE: usize = 2;

/// Precomputed eq polynomial pyramids with balanced split.
struct EqSplitTables<F: PrimeField> {
  /// Full pyramid for inner variables: eq(τ[l0..l0+in_vars], ·)
  /// Layer k has size 2^k, layer in_vars is the full table (size 2^in_vars)
  e_in_pyramid: Vec<Vec<F>>,
  /// Full pyramid for outer variables: eq(τ[l0+in_vars..], ·)
  /// Layer k has size 2^k, layer xout_vars is the full table (size 2^xout_vars)
  e_xout_pyramid: Vec<Vec<F>>,
  /// Suffix eq pyramid for prefix variables, Vec per round
  e_y: Vec<Vec<F>>,
}

/// Cached prefix indices for O(1) scatter access.
pub(crate) struct BetaPrefixCache {
  cache: Csr<AccumulatorPrefixIndex>,
  num_betas: usize,
}

/// Precompute eq polynomial pyramids with balanced split for e_in and e_xout.
///
/// Returns (tables, in_vars, xout_vars) where:
/// - in_vars = ceil((l - l0) / 2) - variables for inner loop (e_in)
/// - xout_vars = floor((l - l0) / 2) - variables for outer loop (e_xout)
///
/// The balanced split reduces precomputation cost by ~33% compared to the
/// asymmetric l/2 split, and enables odd number of rounds.
///
/// Both e_in and e_xout are returned as full pyramids (not just top layers),
/// enabling reuse by EqSumCheckInstance for the remaining sumcheck rounds.
fn precompute_eq_tables<F: PrimeField>(taus: &[F], l0: usize) -> (EqSplitTables<F>, usize, usize) {
  let l = taus.len();
  let suffix_vars = l - l0;
  let in_vars = suffix_vars.div_ceil(2); // ceiling: e_in larger (inner loop, sequential access)
  let xout_vars = suffix_vars - in_vars; // floor: e_xout smaller (outer loop, reused)

  // Build full pyramids (not just top layers) for reuse by EqSumCheckInstance
  let e_in_pyramid = build_eq_pyramid(&taus[l0..l0 + in_vars]); // in_vars+1 layers
  let e_xout_pyramid = build_eq_pyramid(&taus[l0 + in_vars..]); // xout_vars+1 layers
  let e_y = compute_suffix_eq_pyramid(&taus[..l0], l0); // Vec per round, total 2^l0 - 1

  (
    EqSplitTables {
      e_in_pyramid,
      e_xout_pyramid,
      e_y,
    },
    in_vars,
    xout_vars,
  )
}

/// Build beta → prefix index cache for O(1) scatter access.
pub(crate) fn build_beta_cache<const D: usize>(l0: usize) -> BetaPrefixCache {
  let base: usize = D + 1;
  let num_betas = base.pow(l0 as u32);
  let mut cache: Csr<AccumulatorPrefixIndex> = Csr::with_capacity(num_betas, num_betas * l0);
  for b in 0..num_betas {
    let beta = LagrangeIndex::<D>::from_flat_index(b, l0);
    let entries = compute_idx4(&beta);
    cache.push(&entries);
  }

  BetaPrefixCache { cache, num_betas }
}

/// Procedure 9: Build accumulators A_i(v, u) for Spartan's satisfying-witness
/// first sum-check (Algorithm 6).
///
/// Computes accumulators for: g(X) = eq(τ, X) · (Az(X) · Bz(X) - Cz(X))
///
/// This specialized builder assumes `Az * Bz = Cz` on `{0,1}^n`. Under that
/// satisfying-witness invariant, all binary-β contributions vanish, so the
/// optimized prefix rounds can omit explicit `C` handling and only accumulate
/// the non-binary β values containing `∞`.
///
/// D is the degree bound of t_i(X) (not s_i); for Spartan, D = 2.
///
/// # Type Parameters
///
/// - `F`: Field type with small-value and delayed reduction support
/// - `SV`: Witness value type (i32 or i64)
///
/// # Returns
///
/// A tuple of (accumulators, e_in_pyramid, e_xout_pyramid) where:
/// - accumulators: The computed Lagrange accumulators for small-value rounds
/// - e_in_pyramid: Full eq pyramid for inner variables τ[l0..l0+in_vars], has in_vars+1 layers
/// - e_xout_pyramid: Full eq pyramid for outer variables τ[l0+in_vars..ℓ], has xout_vars+1 layers
///
/// The pyramids can be reused by EqSumCheckInstance for the remaining sumcheck rounds,
/// avoiding redundant eq polynomial computation.
///
/// # Parallelism strategy
///
/// - Outer parallel loop over x_out values (using Rayon fold-reduce)
/// - Each thread maintains thread-local accumulators
/// - Final reduction merges all thread-local results via element-wise addition
///
/// # Spartan-specific optimizations (D=2)
///
/// - Skip binary betas: for satisfying witnesses, Az·Bz = Cz on {0,1}^n, so Az·Bz - Cz = 0
/// - Only process betas containing ∞ (non-binary evaluations where contributions are non-zero)
pub fn build_accumulators_spartan_satisfying<F, SV>(
  az: &MultilinearPolynomial<SV>,
  bz: &MultilinearPolynomial<SV>,
  taus: &[F],
  l0: usize,
) -> (LagrangeAccumulators<F, 2>, Vec<Vec<F>>, Vec<Vec<F>>)
where
  F: SmallValueEngine<SV>,
  SV: SmallValue,
{
  let base: usize = 3; // D + 1 = 2 + 1 = 3
  let l = az.Z.len().trailing_zeros() as usize;
  debug_assert_eq!(az.Z.len(), 1usize << l, "poly size must be power of 2");
  debug_assert_eq!(az.Z.len(), bz.Z.len());
  debug_assert_eq!(taus.len(), l, "taus must have length ℓ");
  debug_assert!(l0 < l, "l0 must be < ℓ");

  let suffix_vars = l - l0;
  let prefix_size = 1usize << l0;

  // Precompute eq pyramids with balanced split
  let (eq_tables, in_vars, xout_vars) = precompute_eq_tables(taus, l0);
  let num_x_out = 1usize << xout_vars;

  // Get top layers (full eq tables) for accumulator computation
  let e_in = eq_tables.e_in_pyramid.last().expect("e_in_pyramid non-empty");
  let e_xout = eq_tables.e_xout_pyramid.last().expect("e_xout_pyramid non-empty");
  debug_assert_eq!(e_in.len(), 1 << in_vars);
  debug_assert_eq!(e_xout.len(), 1 << xout_vars);

  // Build beta → prefix index cache
  let BetaPrefixCache {
    cache: beta_prefix_cache,
    num_betas,
  } = build_beta_cache::<2>(l0);

  // Only betas containing at least one ∞ coordinate contribute non-zero values.
  // This builder intentionally omits explicit C terms: on satisfying witnesses,
  // Az·Bz = Cz on {0,1}^n, so all binary-β contributions are zero already.
  let betas_with_infty: Vec<usize> = (0..num_betas)
    .filter(|&i| (0..l0).any(|d| (i / base.pow(d as u32)) % base == 0))
    .collect();

  let ext_size = base.pow(l0 as u32); // (D+1)^l0

  // Build eq_cache: precomputes e_xout[x_out] * e_y[round][y] products.
  // Layout: eq_cache[round][x_out * num_y + y] for cache-friendly access.
  // Each parallel task (fixed x_out) accesses a contiguous block of size num_y.
  let eq_cache: Vec<Vec<F>> = eq_tables
    .e_y
    .iter()
    .map(|round_ey| {
      e_xout
        .iter()
        .flat_map(|ex| round_ey.iter().map(|ey| *ex * *ey))
        .collect()
    })
    .collect();

  // Precompute num_y per round for transposed access
  let num_y_per_round: Vec<usize> = eq_tables.e_y.iter().map(|ey| ey.len()).collect();

  // Parallel over x_out with thread-local state (zero per-iteration allocations)
  type State<F2, SV2> = SpartanThreadState<F2, SV2, 2>;

  let fold_results: Vec<State<F, SV>> = (0..num_x_out)
    .into_par_iter()
    .fold(
      || State::<F, SV>::new(l0, num_betas, prefix_size, ext_size),
      |mut state: State<F, SV>, x_out_bits| {
        // Reset partial sums for this x_out iteration
        state.reset_partial_sums();

        // Inner loop over x_in - accumulate into UNREDUCED form
        for (x_in_bits, e_in_eval) in e_in.iter().enumerate() {
          let suffix = (x_in_bits << xout_vars) | x_out_bits;

          // Fill prefix buffers by index assignment (no allocation)
          #[allow(clippy::needless_range_loop)]
          for prefix in 0..prefix_size {
            let idx = (prefix << suffix_vars) | suffix;
            state.az_prefix_boolean_evals[prefix] = az.Z[idx];
            state.bz_prefix_boolean_evals[prefix] = bz.Z[idx];
          }

          // Extend Az and Bz to Lagrange domain in-place (zero allocation)
          let az_size = extend_to_lagrange_domain::<SV, 2>(
            &state.az_prefix_boolean_evals,
            &mut state.az_extended_evals,
            &mut state.az_extended_scratch,
          );
          let az_ext = &state.az_extended_evals[..az_size];

          let bz_size = extend_to_lagrange_domain::<SV, 2>(
            &state.bz_prefix_boolean_evals,
            &mut state.bz_extended_evals,
            &mut state.bz_extended_scratch,
          );
          let bz_ext = &state.bz_extended_evals[..bz_size];

          // Only process betas with ∞. Omitting explicit C terms is intentional here:
          // the satisfying-witness invariant makes every binary-β contribution vanish.
          // Uses delayed modular reduction: accumulates into unreduced wide-limb form.
          for &beta_idx in &betas_with_infty {
            let prod = SV::wide_mul(az_ext[beta_idx], bz_ext[beta_idx]);
            F::unreduced_multiply_accumulate(&mut state.partial_sums[beta_idx], e_in_eval, &prod);
          }
        }

        // Pre-compute and filter: reduce all non-zero betas upfront
        for &beta_idx in &betas_with_infty {
          if state.partial_sums[beta_idx].is_zero() {
            continue;
          }
          // Reduce partial sum to field element
          let val = <F as DelayedReduction<SV::Product>>::reduce(&state.partial_sums[beta_idx]);
          if val == F::ZERO {
            continue;
          }
          state.beta_values.push((beta_idx, val));
        }

        // Distribute beta values → A_i(v,u) via idx4 using precomputed eq_cache
        // Multiply-accumulate into wide accumulator (Montgomery REDC at end)
        for &(beta_idx, ref val) in &state.beta_values {
          for pref in &beta_prefix_cache[beta_idx] {
            // Transposed layout: eq_cache[round][x_out * num_y + y] for contiguous y access
            let num_y = num_y_per_round[pref.round_0 as usize];
            let eq_eval = eq_cache[pref.round_0 as usize][x_out_bits * num_y + pref.y_idx as usize];
            <F as DelayedReduction<F>>::unreduced_multiply_accumulate(
              &mut state.acc.rounds[pref.round_0 as usize].data_mut()[pref.v_idx as usize]
                [pref.u_idx as usize],
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
  let merged = fold_results
    .into_iter()
    .reduce(|mut a, b| {
      a.acc.merge(&b.acc);
      a
    })
    .expect("num_x_out > 0 guarantees non-empty fold results");

  // Finalize: reduce each bucket from wide 9-limb to field element
  let accumulators = merged
    .acc
    .map(|acc| <F as DelayedReduction<F>>::reduce(acc));

  // Return accumulators along with full pyramids for EqSumCheckInstance reuse
  (
    accumulators,
    eq_tables.e_in_pyramid,
    eq_tables.e_xout_pyramid,
  )
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
  fn test_binary_beta_zero_shortcut_behavior_for_satisfying_witness() {
    // Use l0=1 so round 0 buckets are fed only by β of length 1 (easy to reason about).
    let l0 = 1;
    let l = 2;

    // Az = Bz = top bit x0 (most significant of 2 bits)
    // For satisfying witness, Cz = Az * Bz = Az (since Az ∈ {0,1} and Az = Bz)
    let az_vals: Vec<i32> = (0..(1 << l)).map(|bits| (bits >> (l - 1)) & 1).collect();
    let bz_vals: Vec<i32> = (0..(1 << l)).map(|bits| (bits >> (l - 1)) & 1).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    let taus: Vec<Scalar> = vec![Scalar::from(3u64), Scalar::from(5u64)];

    let (acc, _, _) = build_accumulators_spartan_satisfying(&az, &bz, &taus, l0);

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

  /// Test `build_accumulators_spartan_satisfying` with i32 satisfying witnesses
  /// produces consistent results.
  ///
  /// Verifies that running the same computation twice produces the same output.
  #[test]
  fn test_build_accumulators_spartan_satisfying_small_consistent() {
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
    let (acc1, _, _) = build_accumulators_spartan_satisfying(&az, &bz, &taus, l0);
    let (acc2, _, _) = build_accumulators_spartan_satisfying(&az, &bz, &taus, l0);

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

  /// Test `build_accumulators_spartan_satisfying` with i32 satisfying witnesses
  /// using larger inputs to stress the satisfying-witness shortcut.
  #[test]
  fn test_build_accumulators_spartan_satisfying_small_larger() {
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
    let (acc1, _, _) = build_accumulators_spartan_satisfying(&az, &bz, &taus, l0);
    let (acc2, _, _) = build_accumulators_spartan_satisfying(&az, &bz, &taus, l0);

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
