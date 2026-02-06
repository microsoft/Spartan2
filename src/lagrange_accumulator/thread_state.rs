// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Thread-local scratch buffers for accumulator building.
//!
//! These structs eliminate per-iteration heap allocations in the parallel fold loops
//! of `build_accumulators_spartan` and `build_accumulators_neutronnova`. By hoisting buffer
//! allocations to the fold identity closure (called once per Rayon thread subdivision),
//! we reduce allocations from O(num_x_out) to O(num_threads).

use super::accumulator::LagrangeAccumulators;
use crate::small_field::{DelayedReduction, SmallValueField, WideMul};
use ff::PrimeField;
use num_traits::Zero;
use std::ops::{Add, Sub};

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
/// - `az_extended_evals/scratch`, `bz_extended_evals/scratch`: Two buffer pairs each for Az and Bz
///   (4 buffers total per polynomial × 2 polynomials = 8 buffer fields total). Both Az and Bz
///   extension results must be available simultaneously to compute Az(β) × Bz(β) for each β.
/// - `acc`: Bucket accumulators for scatter phase (unreduced F×F).
///
/// # Type Parameters
///
/// - `F`: Field type with small-value and delayed reduction support
/// - `SmallValue`: Witness value type (i32 or i64 for small-value witnesses)
/// - `D`: Polynomial degree bound
pub(crate) struct SpartanThreadState<F, SmallValue, const D: usize>
where
  F: PrimeField
    + SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
  SmallValue: WideMul + Copy + Default + num_traits::Zero + Add<Output = SmallValue> + Sub<Output = SmallValue> + Send + Sync,
{
  /// Partial sums indexed by β, accumulated over the x_in loop.
  /// Uses unreduced wide-limb form for delayed modular reduction.
  /// Reset each x_out iteration.
  pub partial_sums: Vec<<F as DelayedReduction<SmallValue::Product>>::Accumulator>,
  /// Bucket accumulators for accumulator building phase.
  /// Uses unreduced F×F form (accumulator for field × field products).
  pub acc: LagrangeAccumulators<<F as DelayedReduction<F>>::Accumulator, D>,
  /// Prefix evaluations of Az for current suffix. Size: 2^l0
  pub az_prefix_boolean_evals: Vec<SmallValue>,
  /// Prefix evaluations of Bz for current suffix. Size: 2^l0
  pub bz_prefix_boolean_evals: Vec<SmallValue>,
  /// Result buffer for Az Lagrange extension. After `extend_in_place`, contains the extended values.
  pub az_extended_evals: Vec<SmallValue>,
  /// Scratch buffer for Az Lagrange extension. Used during iterative extension.
  pub az_extended_scratch: Vec<SmallValue>,
  /// Result buffer for Bz Lagrange extension. After `extend_in_place`, contains the extended values.
  pub bz_extended_evals: Vec<SmallValue>,
  /// Scratch buffer for Bz Lagrange extension. Used during iterative extension.
  pub bz_extended_scratch: Vec<SmallValue>,
  /// Reusable buffer for filtered (beta_idx, reduced_value) pairs in accumulator building phase.
  /// Values are field elements from reducing partial sums.
  /// Eliminates per-x_out allocation overhead.
  pub beta_values: Vec<(usize, F)>,
}

impl<F, SmallValue, const D: usize> SpartanThreadState<F, SmallValue, D>
where
  F: PrimeField
    + SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
  SmallValue: WideMul + Copy + Default + num_traits::Zero + Add<Output = SmallValue> + Sub<Output = SmallValue> + Send + Sync,
{
  pub fn new(l0: usize, num_betas: usize, prefix_size: usize, ext_size: usize) -> Self {
    Self {
      partial_sums: vec![<F as DelayedReduction<SmallValue::Product>>::Accumulator::zero(); num_betas],
      acc: LagrangeAccumulators::new(l0),
      az_prefix_boolean_evals: vec![SmallValue::zero(); prefix_size],
      bz_prefix_boolean_evals: vec![SmallValue::zero(); prefix_size],
      az_extended_evals: vec![SmallValue::zero(); ext_size],
      az_extended_scratch: vec![SmallValue::zero(); ext_size],
      bz_extended_evals: vec![SmallValue::zero(); ext_size],
      bz_extended_scratch: vec![SmallValue::zero(); ext_size],
      beta_values: Vec::with_capacity(num_betas),
    }
  }

  /// Zero out partial sums for the next x_out iteration.
  /// This is O(num_betas) but much cheaper than reallocating.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    for sum in &mut self.partial_sums {
      *sum = <F as DelayedReduction<SmallValue::Product>>::Accumulator::zero();
    }
    self.beta_values.clear();
  }
}

/// Thread-local scratch buffers for `build_accumulators_neutronnova`.
///
/// Supports both immediate reduction (`PS = F`) and delayed reduction
/// (`PS = F::Accumulator`). Extension buffers use small values throughout,
/// validated by `vec_to_small_for_extension` to stay within bounds after
/// Lagrange extension (3^ℓ_b growth factor for D=2).
///
/// # Type Parameters
///
/// - `F`: Field type for partial sums and scatter accumulators
/// - `SmallValue`: Value type for pref/extension buffers (i32, i64, etc.)
/// - `PS`: Partial sum type (Accumulator for delayed reduction)
/// - `D`: Polynomial degree bound
pub(crate) struct NeutronNovaThreadState<F, SmallValue, PS: Copy + Default + Zero, const D: usize>
where
  F: PrimeField
    + SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
  SmallValue: WideMul + Copy + Default + num_traits::Zero + Add<Output = SmallValue> + Sub<Output = SmallValue> + Send + Sync,
{
  /// Partial sums indexed by β, accumulated over the x_L loop. Reset each x_R iteration.
  /// Type is `F` for immediate reduction, or `Accumulator` for delayed reduction.
  pub partial_sums: Vec<PS>,
  /// Bucket accumulators for scatter phase (accumulator for field × field products).
  pub scatter_acc: LagrangeAccumulators<<F as DelayedReduction<F>>::Accumulator, D>,
  /// Prefix evaluations of Az for current x_R. Size: 2^l_b
  pub az_prefix_boolean_evals: Vec<SmallValue>,
  /// Prefix evaluations of Bz for current x_R. Size: 2^l_b
  pub bz_prefix_boolean_evals: Vec<SmallValue>,
  /// Result buffer for Az Lagrange extension. Size: 3^l_b
  pub az_extended_evals: Vec<SmallValue>,
  /// Scratch buffer for Az Lagrange extension.
  pub az_extended_scratch: Vec<SmallValue>,
  /// Result buffer for Bz Lagrange extension.
  pub bz_extended_evals: Vec<SmallValue>,
  /// Scratch buffer for Bz Lagrange extension.
  pub bz_extended_scratch: Vec<SmallValue>,
  /// Reusable buffer for filtered (beta_idx, reduced_value) pairs in scatter phase.
  /// Values are field elements from reducing partial sums.
  pub beta_values: Vec<(usize, F)>,
}

impl<F, SmallValue, PS: Copy + Default + Zero, const D: usize> NeutronNovaThreadState<F, SmallValue, PS, D>
where
  F: PrimeField
    + SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
  SmallValue: WideMul + Copy + Default + num_traits::Zero + Add<Output = SmallValue> + Sub<Output = SmallValue> + Send + Sync,
{
  pub fn new(l0: usize, num_betas: usize, prefix_size: usize, ext_size: usize) -> Self {
    Self {
      partial_sums: vec![PS::zero(); num_betas],
      scatter_acc: LagrangeAccumulators::new(l0),
      az_prefix_boolean_evals: vec![SmallValue::zero(); prefix_size],
      bz_prefix_boolean_evals: vec![SmallValue::zero(); prefix_size],
      az_extended_evals: vec![SmallValue::zero(); ext_size],
      az_extended_scratch: vec![SmallValue::zero(); ext_size],
      bz_extended_evals: vec![SmallValue::zero(); ext_size],
      bz_extended_scratch: vec![SmallValue::zero(); ext_size],
      beta_values: Vec::with_capacity(num_betas),
    }
  }

  /// Zero out partial sums for the next x_R iteration.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    self.partial_sums.fill(PS::zero());
    self.beta_values.clear();
  }
}

