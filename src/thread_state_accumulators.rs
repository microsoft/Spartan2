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

use crate::accumulators::SmallValueAccumulators;
use ff::PrimeField;

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
/// - `az_buf_a/b`, `bz_buf_a/b`: Separate ping-pong buffer pairs for Az and Bz Lagrange
///   extensions. We need 4 buffers (not 2) because both extension results must be
///   available simultaneously to compute Az(β) × Bz(β) for each β.
pub(crate) struct SpartanThreadState<S: PrimeField, V: Copy + Default, const D: usize> {
  /// Accumulator being built (the actual output)
  pub acc: SmallValueAccumulators<S, D>,
  /// Partial sums indexed by β, accumulated over the x_in loop. Reset each x_out iteration.
  pub beta_partial_sums: Vec<S>,
  /// Prefix evaluations of Az for current suffix. Size: 2^l0
  pub az_pref: Vec<V>,
  /// Prefix evaluations of Bz for current suffix. Size: 2^l0
  pub bz_pref: Vec<V>,
  /// Ping-pong buffers for Az Lagrange extension. Size: (D+1)^l0 each
  pub az_buf_a: Vec<V>,
  pub az_buf_b: Vec<V>,
  /// Ping-pong buffers for Bz Lagrange extension. Size: (D+1)^l0 each
  pub bz_buf_a: Vec<V>,
  pub bz_buf_b: Vec<V>,
}

impl<S: PrimeField, V: Copy + Default, const D: usize> SpartanThreadState<S, V, D> {
  pub fn new(l0: usize, num_betas: usize, prefix_size: usize, ext_size: usize) -> Self {
    Self {
      acc: SmallValueAccumulators::new(l0),
      beta_partial_sums: vec![S::ZERO; num_betas],
      az_pref: vec![V::default(); prefix_size],
      bz_pref: vec![V::default(); prefix_size],
      az_buf_a: vec![V::default(); ext_size],
      az_buf_b: vec![V::default(); ext_size],
      bz_buf_a: vec![V::default(); ext_size],
      bz_buf_b: vec![V::default(); ext_size],
    }
  }

  /// Zero out partial sums for the next x_out iteration.
  /// This is O(num_betas) but much cheaper than reallocating.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    self.beta_partial_sums.fill(S::ZERO);
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
  pub acc: SmallValueAccumulators<S, D>,
  /// Partial sums indexed by β. Reset each x_out iteration.
  pub beta_partial_sums: Vec<S>,
  /// Prefix evaluations for each of the d polynomials. Size: d × 2^l0
  pub poly_prefs: Vec<Vec<S>>,
  /// Ping-pong buffer pairs for each polynomial's Lagrange extension. Size: d × 2 × (D+1)^l0
  pub buf_pairs: Vec<(Vec<S>, Vec<S>)>,
}

impl<S: PrimeField, const D: usize> GenericThreadState<S, D> {
  pub fn new(l0: usize, num_betas: usize, prefix_size: usize, ext_size: usize, num_polys: usize) -> Self {
    Self {
      acc: SmallValueAccumulators::new(l0),
      beta_partial_sums: vec![S::ZERO; num_betas],
      poly_prefs: (0..num_polys).map(|_| vec![S::ZERO; prefix_size]).collect(),
      buf_pairs: (0..num_polys)
        .map(|_| (vec![S::ZERO; ext_size], vec![S::ZERO; ext_size]))
        .collect(),
    }
  }

  /// Zero out partial sums for the next x_out iteration.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    self.beta_partial_sums.fill(S::ZERO);
  }
}
