// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Index mapping for Algorithm 6 small-value sumcheck optimization (Definition A.5).
//!
//! Maps evaluation prefixes β ∈ U_d^ℓ₀ to accumulator contributions.

use super::domain::{LagrangeIndex, LagrangePoint};

/// Pre-computed indices for O(1) accumulator access in inner loops.
///
/// Represents the decomposition of β ∈ U_d^ℓ₀ into:
/// - Round i (0-indexed for direct array access)
/// - Prefix v = (β₁, ..., β_{i-1}) as flat index
/// - Coordinate u = βᵢ ∈ Û_d as flat index
/// - Binary suffix y = (β_{i+1}, ..., β_{ℓ₀}) as flat index
///
/// Compact 12-byte layout for cache efficiency (5 entries per 64-byte cache line).
/// Field sizes chosen for practical bounds:
/// - `v_idx`: up to 3^ℓ₀ where ℓ₀ ≤ 20 → max ~3.5B, fits u32
/// - `y_idx`: up to 2^ℓ₀ where ℓ₀ ≤ 32 → fits u32
/// - `round_0`: 0..ℓ₀ where ℓ₀ ≤ 255 → fits u8
/// - `u_idx`: 0..D where D ≤ 255 → fits u8
#[derive(Clone, Copy)]
pub(crate) struct AccumulatorPrefixIndex {
  /// Prefix v as flat index in U_d^{i-1}
  pub v_idx: u32,
  /// Binary suffix y as flat index in {0,1}^{ℓ₀-i}
  pub y_idx: u32,
  /// Round (0-indexed for direct array access)
  pub round_0: u8,
  /// Coordinate u as flat index in Û_d
  pub u_idx: u8,
}

/// Computes accumulator indices for β ∈ U_d^ℓ₀ (Definition A.5).
///
/// For each round i where:
/// 1. The suffix β[i..] is binary (values in {0,1})
/// 2. The coordinate u = β[i-1] is in Û_d (i.e., u ≠ 1)
pub(crate) fn compute_idx4<const D: usize>(beta: &LagrangeIndex<D>) -> Vec<AccumulatorPrefixIndex> {
  let l0 = beta.len();
  let mut result = Vec::new();
  let base = LagrangePoint::<D>::BASE;

  // Phase 1: Compute prefix indices (forward pass)
  let mut prefix_idx = vec![0usize; l0 + 1];
  for i in 0..l0 {
    prefix_idx[i + 1] = prefix_idx[i] * base + beta.0[i].to_index();
  }

  // Phase 2: Compute suffix properties (backward pass)
  let mut suffix_is_binary = vec![true; l0 + 1];
  let mut suffix_idx = vec![0usize; l0 + 1];
  for i in (0..l0).rev() {
    let point = beta.0[i];
    if !point.is_binary() {
      suffix_is_binary[i] = false;
      continue;
    }
    suffix_is_binary[i] = suffix_is_binary[i + 1];
    let bit = match point {
      LagrangePoint::Finite(0) => 0,
      LagrangePoint::Finite(1) => 1,
      _ => unreachable!(),
    };
    suffix_idx[i] = suffix_idx[i + 1] | (bit << (l0 - 1 - i));
  }

  // Phase 3: Generate contributions for valid rounds
  for i in 1..=l0 {
    if !suffix_is_binary[i] {
      continue;
    }

    let u = beta.0[i - 1];
    let Some(u_hat) = u.to_ud_hat() else {
      continue; // u = Finite(1), not in Û_d
    };

    result.push(AccumulatorPrefixIndex {
      v_idx: prefix_idx[i - 1] as u32,
      y_idx: suffix_idx[i] as u32,
      round_0: (i - 1) as u8,
      u_idx: u_hat.to_index() as u8,
    });
  }

  result
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::lagrange_accumulator::domain::LagrangeHatPoint;

  fn tuple_from_indices<const D: usize>(indices: &[usize]) -> LagrangeIndex<D> {
    LagrangeIndex(
      indices
        .iter()
        .map(|&i| LagrangePoint::from_index(i))
        .collect(),
    )
  }

  #[test]
  fn test_struct_size() {
    assert_eq!(std::mem::size_of::<AccumulatorPrefixIndex>(), 12);
  }

  #[test]
  fn test_compute_idx4_basic() {
    // β = (Finite(0), Finite(1), Finite(0)) → point (0, 1, 0)
    // Round 2 filtered because u = value 1 ∉ Û_d
    let beta = tuple_from_indices::<3>(&[1, 2, 1]);
    let contributions = compute_idx4(&beta);

    assert_eq!(contributions.len(), 2);

    // Round 1 (0-indexed: 0): v=(), u=Finite(0), y=(1,0) → y_idx=2
    let c0 = contributions.iter().find(|c| c.round_0 == 0).unwrap();
    assert_eq!(c0.v_idx, 0);
    assert_eq!(c0.u_idx, LagrangeHatPoint::<3>::Finite(0).to_index() as u8);
    assert_eq!(c0.y_idx, 2);

    // Round 3 (0-indexed: 2): v=(0,1), u=Finite(0), y=()
    let c2 = contributions.iter().find(|c| c.round_0 == 2).unwrap();
    assert_eq!(c2.v_idx, 6); // 1*4 + 2 = 6
    assert_eq!(c2.u_idx, LagrangeHatPoint::<3>::Finite(0).to_index() as u8);
    assert_eq!(c2.y_idx, 0);
  }

  #[test]
  fn test_compute_idx4_with_infinity() {
    // β = (∞, 0, 1) → point (∞, 0, 1)
    let beta = tuple_from_indices::<3>(&[0, 1, 2]);
    let contributions = compute_idx4(&beta);

    assert_eq!(contributions.len(), 2);

    // Round 1: u=∞, y=(0,1) → y_idx=1
    let c0 = contributions.iter().find(|c| c.round_0 == 0).unwrap();
    assert_eq!(c0.u_idx, LagrangeHatPoint::<3>::Infinity.to_index() as u8);
    assert_eq!(c0.y_idx, 1);

    // Round 2: u=Finite(0), y=(1,) → y_idx=1
    let c1 = contributions.iter().find(|c| c.round_0 == 1).unwrap();
    assert_eq!(c1.u_idx, LagrangeHatPoint::<3>::Finite(0).to_index() as u8);
    assert_eq!(c1.y_idx, 1);
  }

  #[test]
  fn test_compute_idx4_all_ones_empty() {
    // β = (1, 1, 1) → all rounds filtered (u = 1 ∉ Û_d)
    let beta = tuple_from_indices::<3>(&[2, 2, 2]);
    let contributions = compute_idx4(&beta);
    assert!(contributions.is_empty());
  }

  #[test]
  fn test_compute_idx4_index_bounds() {
    let l0 = 4;
    const D: usize = 3;
    let base = D + 1;
    let num_betas = base.pow(l0 as u32);

    for beta_idx in 0..num_betas {
      let beta = LagrangeIndex::<D>::from_flat_index(beta_idx, l0);
      let contributions = compute_idx4(&beta);

      for c in contributions {
        assert!((c.round_0 as usize) < l0);
        assert!((c.u_idx as usize) < D);

        let prefix_len = c.round_0 as usize;
        let suffix_len = l0 - 1 - prefix_len;
        assert!((c.v_idx as usize) < base.pow(prefix_len as u32));
        assert!((c.y_idx as usize) < (1 << suffix_len));
      }
    }
  }
}
