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

use super::{
  accumulator::LagrangeAccumulators, delay_modular_reduction_mode::DelayedModularReductionMode,
  mat_vec_mle::MatVecMLE,
};
use ff::PrimeField;

/// Thread-local scratch buffers for `build_accumulators_spartan`.
///
/// # Motivation
///
/// Without this optimization, the fold closure allocates 5 vectors on every x_out iteration:
/// ```ignore
/// |mut acc, x_out_bits| {
///     let mut beta_partial_sums = vec![S::ZERO; num_betas];     // ALLOC
///     let mut az_prefix_boolean_evals = vec![...];                               // ALLOC
///     let mut bz_prefix_boolean_evals = vec![...];                               // ALLOC
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
/// - `az_extended_evals/scratch`, `bz_extended_evals/scratch`: Separate buffer pairs for Az and Bz
///   Lagrange extensions. After `extend_in_place`, the result is always in `*_buf_curr`.
///   We need 4 buffers (not 2) because both extension results must be available
///   simultaneously to compute Az(β) × Bz(β) for each β.
/// - `scatter_acc`: Bucket accumulators for scatter phase (type determined by Mode).
///
/// Note: `eyx` precomputation has been removed. We now use `e_y` directly in the
/// scatter phase, computing `z_beta = ex * tA_red` once per beta instead of
/// precomputing `eyx = ey * ex` for all y indices.
///
/// # Type Parameters
///
/// - `S`: Field type for final accumulator values
/// - `V`: Witness value type (i32 for small-value, S for field)
/// - `P`: Polynomial type implementing [`MatVecMLE`]
/// - `Mode`: Delayed modular reduction mode selection ([`super::DelayedModularReductionEnabled`] or [`super::DelayedModularReductionDisabled`])
/// - `D`: Polynomial degree bound
pub(crate) struct SpartanThreadState<S, V, P, Mode, const D: usize>
where
  S: PrimeField + Send + Sync,
  V: Copy + Default,
  P: MatVecMLE<S>,
  Mode: DelayedModularReductionMode<S, P, D>,
{
  /// Partial sums indexed by β, accumulated over the x_in loop.
  /// Type determined by Mode: unreduced for DelayedModularReductionEnabled, reduced for DelayedModularReductionDisabled.
  /// Reset each x_out iteration.
  pub partial_sums: Vec<Mode::PartialSum>,
  /// Bucket accumulators for scatter phase.
  /// Type determined by Mode: unreduced F×F for DelayedModularReductionEnabled, LagrangeAccumulator for DelayedModularReductionDisabled.
  pub scatter_acc: LagrangeAccumulators<Mode::ScatterElement, D>,
  /// Prefix evaluations of Az for current suffix. Size: 2^l0
  pub az_prefix_boolean_evals: Vec<V>,
  /// Prefix evaluations of Bz for current suffix. Size: 2^l0
  pub bz_prefix_boolean_evals: Vec<V>,
  /// Result buffer for Az Lagrange extension. After `extend_in_place`, contains the extended values.
  pub az_extended_evals: Vec<V>,
  /// Scratch buffer for Az Lagrange extension. Used during iterative extension.
  pub az_extended_scratch: Vec<V>,
  /// Result buffer for Bz Lagrange extension. After `extend_in_place`, contains the extended values.
  pub bz_extended_evals: Vec<V>,
  /// Scratch buffer for Bz Lagrange extension. Used during iterative extension.
  pub bz_extended_scratch: Vec<V>,
  /// Reusable buffer for filtered (beta_idx, reduced_value) pairs in scatter phase.
  /// Eliminates per-x_out allocation overhead.
  pub beta_values: Vec<(usize, S)>,
}

impl<S, V, P, Mode, const D: usize> SpartanThreadState<S, V, P, Mode, D>
where
  S: PrimeField + Send + Sync,
  V: Copy + Default,
  P: MatVecMLE<S>,
  Mode: DelayedModularReductionMode<S, P, D>,
{
  pub fn new(l0: usize, num_betas: usize, prefix_size: usize, ext_size: usize) -> Self {
    Self {
      partial_sums: vec![Mode::PartialSum::default(); num_betas],
      scatter_acc: LagrangeAccumulators::new(l0),
      az_prefix_boolean_evals: vec![V::default(); prefix_size],
      bz_prefix_boolean_evals: vec![V::default(); prefix_size],
      az_extended_evals: vec![V::default(); ext_size],
      az_extended_scratch: vec![V::default(); ext_size],
      bz_extended_evals: vec![V::default(); ext_size],
      bz_extended_scratch: vec![V::default(); ext_size],
      beta_values: Vec::with_capacity(num_betas),
    }
  }

  /// Zero out partial sums for the next x_out iteration.
  /// This is O(num_betas) but much cheaper than reallocating.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    for sum in &mut self.partial_sums {
      *sum = Mode::PartialSum::default();
    }
    self.beta_values.clear();
  }
}

/// Thread-local scratch buffers for `build_accumulators_neutronnova`.
///
/// Simplified version of `SpartanThreadState` — no `Mode` generics, no `MatVecMLE`.
/// Phase 1 MVP uses immediate reduction (field elements for partial_sums/scatter).
/// Pref/extension buffers use `IntermediateSmallValue` (i64/i128) to avoid overflow
/// during ExtendToU₂ which adds up to ℓ_b bits of growth.
///
/// # Type Parameters
///
/// - `S`: Field type for partial sums and scatter accumulators
/// - `V`: Intermediate value type for pref/extension buffers (i64 for i32 path, i128 for i64 path)
/// - `D`: Polynomial degree bound
pub(crate) struct NeutronNovaThreadState<S: PrimeField, V: Copy + Default, PS: Copy + Default, const D: usize>
{
  /// Partial sums indexed by β, accumulated over the x_L loop. Reset each x_R iteration.
  /// Type is `S` for immediate reduction, or `UnreducedFieldInt` for delayed reduction.
  pub partial_sums: Vec<PS>,
  /// Bucket accumulators for scatter phase.
  pub scatter_acc: LagrangeAccumulators<S, D>,
  /// Prefix evaluations of Az for current x_R. Size: 2^l_b
  pub az_prefix_boolean_evals: Vec<V>,
  /// Prefix evaluations of Bz for current x_R. Size: 2^l_b
  pub bz_prefix_boolean_evals: Vec<V>,
  /// Result buffer for Az Lagrange extension. Size: 3^l_b
  pub az_extended_evals: Vec<V>,
  /// Scratch buffer for Az Lagrange extension.
  pub az_extended_scratch: Vec<V>,
  /// Result buffer for Bz Lagrange extension.
  pub bz_extended_evals: Vec<V>,
  /// Scratch buffer for Bz Lagrange extension.
  pub bz_extended_scratch: Vec<V>,
  /// Reusable buffer for filtered (beta_idx, reduced_value) pairs in scatter phase.
  pub beta_values: Vec<(usize, S)>,
}

impl<S: PrimeField, V: Copy + Default, PS: Copy + Default, const D: usize>
  NeutronNovaThreadState<S, V, PS, D>
{
  pub fn new(l0: usize, num_betas: usize, prefix_size: usize, ext_size: usize) -> Self {
    Self {
      partial_sums: vec![PS::default(); num_betas],
      scatter_acc: LagrangeAccumulators::new(l0),
      az_prefix_boolean_evals: vec![V::default(); prefix_size],
      bz_prefix_boolean_evals: vec![V::default(); prefix_size],
      az_extended_evals: vec![V::default(); ext_size],
      az_extended_scratch: vec![V::default(); ext_size],
      bz_extended_evals: vec![V::default(); ext_size],
      bz_extended_scratch: vec![V::default(); ext_size],
      beta_values: Vec::with_capacity(num_betas),
    }
  }

  /// Zero out partial sums for the next x_R iteration.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    self.partial_sums.fill(PS::default());
    self.beta_values.clear();
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
