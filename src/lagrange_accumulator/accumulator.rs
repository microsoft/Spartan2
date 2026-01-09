// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Accumulator data structures for Algorithm 6 small-value sumcheck optimization.
//!
//! This module defines:
//! - [`RoundAccumulator`]: Single round accumulator A_i(v, u) with flat storage
//! - [`LagrangeAccumulators`]: Collection of accumulators for all ℓ₀ rounds

use super::{basis::LagrangeCoeff, evals::LagrangeHatEvals};
use ff::PrimeField;

#[cfg(test)]
use super::domain::{LagrangeHatPoint, LagrangeIndex};

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

  /// Evaluate t_i(u) for all u ∈ Û_D in a single pass.
  pub fn eval_t_all_u(&self, coeff: &LagrangeCoeff<Scalar, D>) -> LagrangeHatEvals<Scalar, D> {
    debug_assert_eq!(self.data.len(), coeff.len());
    let mut acc = [Scalar::ZERO; D];
    for (c, row) in coeff.as_slice().iter().zip(self.data.iter()) {
      let scaled = *c;
      for i in 0..D {
        acc[i] += scaled * row[i];
      }
    }
    LagrangeHatEvals::from_array(acc)
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
}

/// Test-only helper methods for RoundAccumulator.
#[cfg(test)]
impl<Scalar: PrimeField, const D: usize> RoundAccumulator<Scalar, D> {
  /// O(1) indexed read from bucket (v_idx, u_idx).
  #[inline]
  pub fn get(&self, v_idx: usize, u_idx: usize) -> Scalar {
    self.data[v_idx][u_idx]
  }

  /// Accumulate by domain types (type-safe path).
  #[inline]
  pub fn accumulate_by_domain(
    &mut self,
    v: &LagrangeIndex<D>,
    u: LagrangeHatPoint<D>,
    value: Scalar,
  ) {
    let v_idx = v.to_flat_index();
    let u_idx = u.to_index();
    self.data[v_idx][u_idx] += value;
  }

  /// Read by domain types (type-safe path).
  #[inline]
  pub fn get_by_domain(&self, v: &LagrangeIndex<D>, u: LagrangeHatPoint<D>) -> Scalar {
    let v_idx = v.to_flat_index();
    let u_idx = u.to_index();
    self.data[v_idx][u_idx]
  }

  /// Number of prefix entries.
  #[inline]
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
pub struct LagrangeAccumulators<Scalar: PrimeField, const D: usize> {
  /// rounds[i] contains A_{i+1} (the accumulator for 1-indexed round i+1)
  rounds: Vec<RoundAccumulator<Scalar, D>>,
}

impl<Scalar: PrimeField, const D: usize> LagrangeAccumulators<Scalar, D> {
  /// Create a fresh accumulator (used per-thread in fold).
  ///
  /// # Arguments
  /// * `l0` - Number of rounds using small-value optimization
  pub fn new(l0: usize) -> Self {
    let rounds = (0..l0).map(RoundAccumulator::new).collect();
    Self { rounds }
  }

  /// O(1) accumulation into bucket (round, v_idx, u_idx).
  #[inline]
  pub fn accumulate(&mut self, round: usize, v_idx: usize, u_idx: usize, value: Scalar) {
    self.rounds[round].accumulate(v_idx, u_idx, value);
  }

  /// Merge another accumulator into this one (for reduce phase).
  pub fn merge(&mut self, other: &Self) {
    for (self_round, other_round) in self.rounds.iter_mut().zip(&other.rounds) {
      self_round.merge(other_round);
    }
  }

  /// Get read-only access to a specific round's accumulator.
  pub fn round(&self, i: usize) -> &RoundAccumulator<Scalar, D> {
    &self.rounds[i]
  }
}

/// Test-only helper methods for LagrangeAccumulators.
#[cfg(test)]
impl<Scalar: PrimeField, const D: usize> LagrangeAccumulators<Scalar, D> {
  /// Read A_i(v, u).
  #[inline]
  pub fn get(&self, round: usize, v_idx: usize, u_idx: usize) -> Scalar {
    self.rounds[round].get(v_idx, u_idx)
  }

  /// Accumulate by domain types (type-safe path).
  #[inline]
  pub fn accumulate_by_domain(
    &mut self,
    round: usize,
    v: &LagrangeIndex<D>,
    u: LagrangeHatPoint<D>,
    value: Scalar,
  ) {
    self.rounds[round].accumulate_by_domain(v, u, value);
  }

  /// Read A_i(v, u) by domain types (type-safe path).
  #[inline]
  pub fn get_by_domain(
    &self,
    round: usize,
    v: &LagrangeIndex<D>,
    u: LagrangeHatPoint<D>,
  ) -> Scalar {
    self.rounds[round].get_by_domain(v, u)
  }

  /// Number of rounds.
  pub fn num_rounds(&self) -> usize {
    self.rounds.len()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{lagrange_accumulator::domain::LagrangePoint, provider::pasta::pallas};
  use ff::Field;

  type Scalar = pallas::Scalar;

  // Use D=2 for Spartan
  const D: usize = 2;

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
    let v = LagrangeIndex::<3>(vec![LagrangePoint::Finite(1)]);
    let u = LagrangeHatPoint::<3>::Infinity; // index 0

    let val = Scalar::from(42u64);
    acc.accumulate_by_domain(&v, u, val);

    assert_eq!(acc.get_by_domain(&v, u), val);
    // Verify same via raw indices
    assert_eq!(acc.get(2, 0), val);
  }

  // === LagrangeAccumulators tests ===

  #[test]
  fn test_lagrange_accumulators_new() {
    // For D=2 (base=3), ℓ₀=3
    // Round 0: 3^0 = 1 prefix
    // Round 1: 3^1 = 3 prefixes
    // Round 2: 3^2 = 9 prefixes
    let acc: LagrangeAccumulators<Scalar, D> = LagrangeAccumulators::new(3);

    assert_eq!(acc.num_rounds(), 3);
    assert_eq!(acc.round(0).num_prefixes(), 1);
    assert_eq!(acc.round(1).num_prefixes(), 3);
    assert_eq!(acc.round(2).num_prefixes(), 9);
  }

  #[test]
  fn test_lagrange_accumulators_accumulate_get() {
    let mut acc: LagrangeAccumulators<Scalar, D> = LagrangeAccumulators::new(3);

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
  fn test_lagrange_accumulators_merge() {
    let mut acc1: LagrangeAccumulators<Scalar, D> = LagrangeAccumulators::new(3);
    let mut acc2: LagrangeAccumulators<Scalar, D> = LagrangeAccumulators::new(3);

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
  fn test_lagrange_accumulators_domain_methods() {
    let mut acc: LagrangeAccumulators<Scalar, D> = LagrangeAccumulators::new(2);

    // Round 1 has 3 prefixes (base^1)
    // v = (Finite(0),) -> flat index = 1 (∞=0, 0=1, 1=2)
    let v = LagrangeIndex::<2>(vec![LagrangePoint::Finite(0)]);
    let u = LagrangeHatPoint::<D>::Infinity; // index 0

    let val = Scalar::from(99u64);
    acc.accumulate_by_domain(1, &v, u, val);

    assert_eq!(acc.get_by_domain(1, &v, u), val);
    // Verify same via raw indices
    assert_eq!(acc.get(1, 1, 0), val);
  }

  #[test]
  fn test_accumulator_sizes_match_spec() {
    // For D=2, ℓ₀=3 should have total 26 elements
    // Round 0: 1 * 2 = 2
    // Round 1: 3 * 2 = 6
    // Round 2: 9 * 2 = 18
    // Total: 26
    let acc: LagrangeAccumulators<Scalar, D> = LagrangeAccumulators::new(3);

    let total_elements: usize = (0..3).map(|i| acc.round(i).num_prefixes() * 2).sum();
    assert_eq!(total_elements, 26);
  }
}
