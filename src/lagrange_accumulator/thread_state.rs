// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Thread-local scratch buffers for accumulator building.
//!
//! These structs eliminate per-iteration heap allocations in the parallel fold loops
//! of `build_accumulators_spartan` and `build_accumulators`. By hoisting buffer
//! allocations to the fold identity closure (called once per Rayon thread subdivision),
//! we reduce allocations from O(num_x_out) to O(num_threads).

use super::accumulator::LagrangeAccumulators;
use ff::PrimeField;
use std::ops::AddAssign;

/// Thread-local scratch buffers for `build_accumulators_spartan`.
///
/// # Motivation
///
/// Without this optimization, the fold closure allocates 5 vectors on every x_out iteration:
/// ```ignore
/// |mut acc, x_out_bits| {
///     let mut beta_partial_sums = vec![S::ZERO; num_betas];     // ALLOC
///     let mut az_pref = vec![...];                               // ALLOC
///     let mut bz_pref = vec![...];                               // ALLOC
///     let mut buf_a = vec![...];                                 // ALLOC
///     let mut buf_b = vec![...];                                 // ALLOC
///     ...
/// }
/// ```
///
/// For typical workloads (l=20, l0=4), num_x_out = 2^6 = 64, causing 320 allocations
/// per parallel task. With Rayon's work-stealing, this leads to significant allocator
/// contention and cache pollution.
///
/// # Solution
///
/// By hoisting these buffers into a struct created once per Rayon thread subdivision
/// (in the fold identity closure), we reduce allocations from O(num_x_out) to O(num_threads).
/// The `reset_partial_sums()` method zeros the sums between iterations (cheap memset).
///
/// # Buffer Layout
///
/// - `az_buf_curr/scratch`, `bz_buf_curr/scratch`: Separate buffer pairs for Az and Bz
///   Lagrange extensions. After `extend_in_place`, the result is always in `*_buf_curr`.
///   We need 4 buffers (not 2) because both extension results must be available
///   simultaneously to compute Az(β) × Bz(β) for each β.
/// - `acc_unred`: Unreduced F×F bucket accumulators for scatter phase optimization.
///
/// Note: `eyx` precomputation has been removed. We now use `e_y` directly in the
/// scatter phase, computing `z_beta = ex * tA_red` once per beta instead of
/// precomputing `eyx = ey * ex` for all y indices.
///
/// # Type Parameters
///
/// - `S`: Field type for final accumulator values
/// - `V`: Witness value type (i32 for small-value, S for field)
/// - `U`: Unreduced sum type for delayed modular reduction (field × int products)
/// - `UFF`: Unreduced F×F accumulator type (9 limbs, 576 bits, 2R-scaled)
/// - `D`: Polynomial degree bound
pub(crate) struct SpartanThreadState<
  S: PrimeField,
  V: Copy + Default,
  U: Copy + Default,
  UFF: Copy + Default,
  const D: usize,
> {
  /// Reduced accumulator (kept for compatibility, not used in optimized path)
  #[allow(dead_code)]
  pub acc: LagrangeAccumulators<S, D>,
  /// Unreduced F×F bucket accumulators for scatter phase.
  /// Shape: [round][v_idx][u_idx] matching LagrangeAccumulators layout.
  /// Final reduction is done once after all threads are merged.
  pub acc_unred: Vec<Vec<[UFF; D]>>,
  /// Partial sums indexed by β, accumulated over the x_in loop (unreduced form).
  /// Reset each x_out iteration.
  pub beta_partial_sums: Vec<U>,
  /// Prefix evaluations of Az for current suffix. Size: 2^l0
  pub az_pref: Vec<V>,
  /// Prefix evaluations of Bz for current suffix. Size: 2^l0
  pub bz_pref: Vec<V>,
  /// Result buffer for Az Lagrange extension. After `extend_in_place`, contains the extended values.
  pub az_buf_curr: Vec<V>,
  /// Scratch buffer for Az Lagrange extension. Used during iterative extension.
  pub az_buf_scratch: Vec<V>,
  /// Result buffer for Bz Lagrange extension. After `extend_in_place`, contains the extended values.
  pub bz_buf_curr: Vec<V>,
  /// Scratch buffer for Bz Lagrange extension. Used during iterative extension.
  pub bz_buf_scratch: Vec<V>,
}

impl<
  S: PrimeField,
  V: Copy + Default,
  U: Copy + Clone + Default + AddAssign,
  UFF: Copy + Clone + Default + AddAssign,
  const D: usize,
> SpartanThreadState<S, V, U, UFF, D>
{
  /// Base of the Lagrange domain U_D (compile-time constant)
  const BASE: usize = D + 1;

  pub fn new(l0: usize, num_betas: usize, prefix_size: usize, ext_size: usize) -> Self {
    // Allocate unreduced accumulators matching LagrangeAccumulators shape
    let acc_unred = (0..l0)
      .map(|round| {
        let num_prefixes = Self::BASE.pow(round as u32);
        vec![[UFF::default(); D]; num_prefixes]
      })
      .collect();

    Self {
      acc: LagrangeAccumulators::new(l0),
      acc_unred,
      beta_partial_sums: vec![U::default(); num_betas],
      az_pref: vec![V::default(); prefix_size],
      bz_pref: vec![V::default(); prefix_size],
      az_buf_curr: vec![V::default(); ext_size],
      az_buf_scratch: vec![V::default(); ext_size],
      bz_buf_curr: vec![V::default(); ext_size],
      bz_buf_scratch: vec![V::default(); ext_size],
    }
  }

  /// Zero out partial sums for the next x_out iteration.
  /// This is O(num_betas) but much cheaper than reallocating.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    self.beta_partial_sums.fill(U::default());
  }

  /// Zero out unreduced bucket accumulators.
  /// Not used in the optimized path (we keep unreduced until final merge).
  #[inline]
  #[allow(dead_code)]
  pub fn reset_acc_unred(&mut self) {
    for round in &mut self.acc_unred {
      for row in round.iter_mut() {
        *row = [UFF::default(); D];
      }
    }
  }
}

/// Thread-local scratch buffers for the generic `build_accumulators`.
///
/// Similar to `SpartanThreadState`, but handles a variable number of polynomials (d).
/// Each polynomial needs its own buffer pair for Lagrange extension since all d
/// extension results must be available simultaneously to compute ∏ p_k(β).
///
/// See `SpartanThreadState` documentation for the full motivation.
pub(crate) struct GenericThreadState<S: PrimeField, const D: usize> {
  /// Accumulator being built (the actual output)
  pub acc: LagrangeAccumulators<S, D>,
  /// Partial sums indexed by β. Reset each x_out iteration.
  pub beta_partial_sums: Vec<S>,
  /// Prefix evaluations for each of the d polynomials. Size: d × 2^l0
  pub poly_prefs: Vec<Vec<S>>,
  /// Buffer pairs for each polynomial's Lagrange extension: (result, scratch).
  /// After `extend_in_place`, the result is always in the first element of each pair.
  /// Size: d × 2 × (D+1)^l0
  pub buf_pairs: Vec<(Vec<S>, Vec<S>)>,
  /// On-the-fly computed ey*ex scratch buffer. Size per round: 2^{l0-1-round}.
  /// Total size: 2^l0 - 1 (e.g., 7 for l0=3). Stays hot in L1 cache.
  pub eyx: Vec<Vec<S>>,
}

impl<S: PrimeField, const D: usize> GenericThreadState<S, D> {
  pub fn new(
    l0: usize,
    num_betas: usize,
    prefix_size: usize,
    ext_size: usize,
    num_polys: usize,
    e_y_sizes: &[usize],
  ) -> Self {
    Self {
      acc: LagrangeAccumulators::new(l0),
      beta_partial_sums: vec![S::ZERO; num_betas],
      poly_prefs: (0..num_polys).map(|_| vec![S::ZERO; prefix_size]).collect(),
      buf_pairs: (0..num_polys)
        .map(|_| (vec![S::ZERO; ext_size], vec![S::ZERO; ext_size]))
        .collect(),
      eyx: e_y_sizes.iter().map(|&sz| vec![S::ZERO; sz]).collect(),
    }
  }

  /// Zero out partial sums for the next x_out iteration.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    self.beta_partial_sums.fill(S::ZERO);
  }
}
