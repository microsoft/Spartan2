// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Index mapping for Algorithm 6 small-value sumcheck optimization (Definition A.5).
//!
//! This module defines:
//! - [`AccumulatorPrefixIndex`]: Describes how an evaluation prefix β contributes to accumulators
//! - [`compute_idx4`]: Maps evaluation prefixes β ∈ U_d^ℓ₀ to accumulator contributions
//! - [`CubicAccumulatorPrefixIndex`]: Type alias for Spartan's cubic sumcheck (D=3)

// Allow dead code until later chunks use these types
#![allow(dead_code)]

use crate::lagrange::{UdHatPoint, UdPoint, UdTuple};

/// A single contribution from β to an accumulator A_i(v, u).
///
/// Represents the decomposition of β ∈ U_d^ℓ₀ into:
/// - Round i (which accumulator)
/// - Prefix v = (β₁, ..., β_{i-1}) ∈ U_d^{i-1}
/// - Coordinate u = βᵢ ∈ Û_d
/// - Binary suffix y = (β_{i+1}, ..., β_{ℓ₀}) ∈ {0,1}^{ℓ₀-i}
///
/// This set identifies all accumulators A_i(v, u) to which the product term
/// computed using the prefix β contributes. The y component will be summed
/// over when computing the accumulators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccumulatorPrefixIndex<const D: usize> {
  /// Total number of small-value rounds (ℓ₀)
  pub l0: usize,

  /// Round index i ∈ [1, ℓ₀] (1-indexed as in paper)
  pub round: usize,

  /// Prefix v = (β₁, ..., β_{i-1}) as flat index in U_d^{i-1}
  pub v_idx: usize,

  /// Coordinate u = βᵢ ∈ Û_d (type-safe, excludes value 1)
  pub u: UdHatPoint<D>,

  /// Binary suffix y = (β_{i+1}, ..., β_{ℓ₀}) as flat index in {0,1}^{ℓ₀-i}
  pub y_idx: usize,
}

impl<const D: usize> AccumulatorPrefixIndex<D> {
  /// Round as 0-indexed (for array access)
  #[inline]
  pub fn round_0idx(&self) -> usize {
    self.round - 1
  }

  /// Length of prefix v
  #[inline]
  pub fn prefix_len(&self) -> usize {
    self.round - 1
  }

  /// Length of binary suffix y
  #[inline]
  pub fn suffix_len(&self) -> usize {
    self.l0 - self.round
  }
}

/// Type alias for Spartan's cubic sumcheck index mapping (D=3).
pub type CubicAccumulatorPrefixIndex = AccumulatorPrefixIndex<3>;

/// Computes accumulator indices for β ∈ U_d^ℓ₀ (Definition A.5).
///
/// For each round i where:
/// 1. The suffix β[i..] is binary (values in {0,1})
/// 2. The coordinate u = β[i-1] is in Û_d (i.e., u ≠ 1)
///
/// Returns a list of `AccumulatorPrefixIndex` describing contributions.
///
/// # Type-Safe API
///
/// This function accepts `UdTuple<D>` for type safety. For performance-critical
/// code using flat indices, convert via `UdTuple::from_flat_index`.
///
/// # Arguments
/// * `beta` - Tuple in U_d^ℓ₀ as a UdTuple
/// * `l0` - Number of small-value rounds
///
/// # Example
///
/// For β = (Finite(0), Finite(1), Finite(0)) with l0=3, d=3:
/// - Round 1: v=(), u=Finite(0)∈Û_d, suffix=(1,0) binary → contributes
/// - Round 2: v=(0,), u=Finite(1)∉Û_d → filtered out
/// - Round 3: v=(0,1), u=Finite(0)∈Û_d, suffix=() binary → contributes
pub fn compute_idx4<const D: usize>(beta: &UdTuple<D>, l0: usize) -> Vec<AccumulatorPrefixIndex<D>> {
  debug_assert_eq!(beta.len(), l0, "β length must equal l0");

  let mut result = Vec::new();

  for i in 1..=l0 {
    // Check if suffix β[i..] is all binary
    // Binary means the value is Finite(0) or Finite(1)
    let suffix_is_binary = beta.0[i..].iter().all(|p| p.is_binary());

    if !suffix_is_binary {
      continue;
    }

    // u = β[i-1] — try to convert to Û_d point
    // If u = Finite(1), to_ud_hat() returns None and we skip this round
    let u = beta.0[i - 1];
    let Some(u_hat) = u.to_ud_hat() else {
      continue; // u = Finite(1), not in Û_d
    };

    // v = prefix β[0..i-1] as flat index
    let prefix = UdTuple(beta.0[0..(i - 1)].to_vec());
    let v_idx = prefix.to_flat_index();

    // y = suffix converted to binary index
    // Finite(0) → bit 0, Finite(1) → bit 1
    // MSB-first encoding: leftmost suffix element is MSB
    let suffix = &beta.0[i..];
    let y_idx = suffix.iter().fold(0usize, |acc, p| {
      let bit = match p {
        UdPoint::Finite(0) => 0,
        UdPoint::Finite(1) => 1,
        _ => unreachable!("suffix should be binary"),
      };
      (acc << 1) | bit
    });

    result.push(AccumulatorPrefixIndex {
      l0,
      round: i, // 1-indexed
      v_idx,
      u: u_hat,
      y_idx,
    });
  }

  result
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Helper: construct UdTuple from flat indices
  /// 0 → ∞, 1 → 0, 2 → 1, 3 → 2, ...
  fn tuple_from_indices<const D: usize>(indices: &[usize]) -> UdTuple<D> {
    UdTuple(indices.iter().map(|&i| UdPoint::from_index(i)).collect())
  }

  #[test]
  fn test_accumulator_prefix_index_helpers() {
    let idx: AccumulatorPrefixIndex<3> = AccumulatorPrefixIndex {
      l0: 5,
      round: 3,
      v_idx: 10,
      u: UdHatPoint::Finite(2),
      y_idx: 3,
    };

    assert_eq!(idx.round_0idx(), 2); // 3 - 1
    assert_eq!(idx.prefix_len(), 2); // 3 - 1
    assert_eq!(idx.suffix_len(), 2); // 5 - 3
  }

  #[test]
  fn test_compute_idx4_example_mixed_binary() {
    // β = (Finite(0), Finite(1), Finite(0)) → point (0, 1, 0)
    // Round 2 filtered because u = value 1 ∉ Û_d
    let l0 = 3;

    let beta = tuple_from_indices::<3>(&[1, 2, 1]); // indices → (0, 1, 0)
    let contributions = compute_idx4(&beta, l0);

    assert_eq!(contributions.len(), 2); // Round 2 filtered

    // Round 1: v=(), u=value 0, y=(Finite(1),Finite(0)) → y_idx = 2
    let round1 = contributions.iter().find(|c| c.round == 1).unwrap();
    assert_eq!(round1.v_idx, 0);
    assert_eq!(round1.u, UdHatPoint::Finite(0));
    assert_eq!(round1.y_idx, 2); // bits (1,0) = 2
    assert_eq!(round1.prefix_len(), 0);
    assert_eq!(round1.suffix_len(), 2);

    // Round 2: FILTERED (u = Finite(1) ∉ Û_d)
    assert!(contributions.iter().all(|c| c.round != 2));

    // Round 3: v=(Finite(0),Finite(1)), u=value 0, y=() → y_idx = 0
    let round3 = contributions.iter().find(|c| c.round == 3).unwrap();
    assert_eq!(round3.v_idx, 6); // 1*4 + 2 = 6
    assert_eq!(round3.u, UdHatPoint::Finite(0));
    assert_eq!(round3.y_idx, 0);
    assert_eq!(round3.prefix_len(), 2);
    assert_eq!(round3.suffix_len(), 0);
  }

  #[test]
  fn test_compute_idx4_example_with_infinity() {
    // β = (∞, 0, 1) → point (∞, 0, 1)
    // Round 3 filtered because u = value 1 ∉ Û_d
    let l0 = 3;

    let beta = tuple_from_indices::<3>(&[0, 1, 2]); // indices → (∞, 0, 1)
    let contributions = compute_idx4(&beta, l0);

    assert_eq!(contributions.len(), 2); // Round 3 filtered

    // Round 1: v=(), u=∞, y=(Finite(0),Finite(1)) → y_idx = 1
    let round1 = contributions.iter().find(|c| c.round == 1).unwrap();
    assert_eq!(round1.v_idx, 0);
    assert_eq!(round1.u, UdHatPoint::Infinity);
    assert_eq!(round1.y_idx, 1); // bits (0,1) = 1

    // Round 2: v=(∞,), u=value 0, y=(Finite(1),) → y_idx = 1
    let round2 = contributions.iter().find(|c| c.round == 2).unwrap();
    assert_eq!(round2.v_idx, 0);
    assert_eq!(round2.u, UdHatPoint::Finite(0));
    assert_eq!(round2.y_idx, 1); // bits (1,) = 1

    // Round 3: FILTERED (u = value 1 ∉ Û_d)
    assert!(contributions.iter().all(|c| c.round != 3));
  }

  #[test]
  fn test_compute_idx4_example_double_infinity() {
    // β = (∞, ∞, 0) → point (∞, ∞, 0)
    // Round 1 skipped because suffix contains ∞
    let l0 = 3;

    let beta = tuple_from_indices::<3>(&[0, 0, 1]); // indices → (∞, ∞, 0)
    let contributions = compute_idx4(&beta, l0);

    assert_eq!(contributions.len(), 2);

    // Round 1 should be missing (suffix has ∞)
    assert!(contributions.iter().all(|c| c.round != 1));

    // Round 2: v=(∞,), u=∞, y=(Finite(0),) → y_idx = 0
    let round2 = contributions.iter().find(|c| c.round == 2).unwrap();
    assert_eq!(round2.v_idx, 0);
    assert_eq!(round2.u, UdHatPoint::Infinity);
    assert_eq!(round2.y_idx, 0);

    // Round 3: v=(∞,∞), u=Finite(0), y=() → y_idx = 0
    let round3 = contributions.iter().find(|c| c.round == 3).unwrap();
    assert_eq!(round3.v_idx, 0); // 0*4 + 0 = 0
    assert_eq!(round3.u, UdHatPoint::Finite(0));
    assert_eq!(round3.y_idx, 0);
  }

  #[test]
  fn test_compute_idx4_example_all_extrapolated() {
    // β = (Finite(2), Finite(2), Finite(2)) → point (2, 2, 2)
    // Only last round (empty suffix is vacuously binary)
    let l0 = 3;

    let beta = tuple_from_indices::<3>(&[3, 3, 3]); // indices → (2, 2, 2)
    let contributions = compute_idx4(&beta, l0);

    assert_eq!(contributions.len(), 1);

    let only = &contributions[0];
    assert_eq!(only.round, 3);
    assert_eq!(only.v_idx, 15); // 3*4 + 3 = 15
    assert_eq!(only.u, UdHatPoint::Finite(2));
    assert_eq!(only.y_idx, 0);
    assert_eq!(only.suffix_len(), 0);
  }

  #[test]
  fn test_compute_idx4_all_ones_has_no_contributions() {
    // β = (Finite(1), Finite(1), Finite(1)) → point (1, 1, 1)
    // ALL rounds filtered because u = value 1 ∉ Û_d at every position
    let l0 = 3;

    let beta = tuple_from_indices::<3>(&[2, 2, 2]); // indices → (1, 1, 1)
    let contributions = compute_idx4(&beta, l0);

    assert!(
      contributions.is_empty(),
      "β=(1,1,1) should have no contributions"
    );
  }

  #[test]
  fn test_compute_idx4_contribution_counts() {
    // Not every β contributes! β where all elements are Finite(1)
    // has NO contributions because u=1 ∉ Û_d for all rounds.
    let l0 = 3;
    const D: usize = 3;
    let base = D + 1;
    let prefix_ud_size = base.pow(l0 as u32);

    let mut zero_contribution_count = 0;

    for beta_idx in 0..prefix_ud_size {
      let beta = UdTuple::<D>::from_flat_index(beta_idx, l0);
      let contributions = compute_idx4(&beta, l0);

      // All contributions should have correct l0
      for c in &contributions {
        assert_eq!(c.l0, l0);
      }

      if contributions.is_empty() {
        zero_contribution_count += 1;
      }
    }

    // β = (Finite(1), Finite(1), Finite(1)) is the only β with zero contributions
    assert_eq!(
      zero_contribution_count, 1,
      "Only β=(1,1,1) should have zero contributions"
    );
  }

  #[test]
  fn test_compute_idx4_binary_beta_filtered_by_u() {
    // Binary β (Finite(0) or Finite(1)) DON'T all contribute to all rounds!
    // Round i is filtered when β[i-1] = Finite(1) (i.e., u = value 1 ∉ Û_d)
    let l0 = 3;
    const D: usize = 3;

    // Iterate over all binary β
    for b in 0..(1 << l0) {
      // Construct beta with Finite(0) or Finite(1) values
      let indices: Vec<usize> = (0..l0)
        .map(|j| ((b >> (l0 - 1 - j)) & 1) + 1) // 0→index 1 (Finite(0)), 1→index 2 (Finite(1))
        .collect();
      let beta = tuple_from_indices::<D>(&indices);

      let contributions = compute_idx4(&beta, l0);

      // Count how many positions have Finite(1) (value 1)
      let num_ones = beta
        .0
        .iter()
        .filter(|&&p| p == UdPoint::Finite(1))
        .count();

      // Expected contributions = l0 - num_ones (each position with value 1 filters that round)
      let expected_len = l0 - num_ones;
      assert_eq!(
        contributions.len(),
        expected_len,
        "β={:?} should have {} contributions (filtering {} rounds with u=1)",
        beta,
        expected_len,
        num_ones
      );

      // Verify correct rounds are present/missing
      for round in 1..=l0 {
        let has_round = contributions.iter().any(|c| c.round == round);
        let u = beta.0[round - 1];

        if u == UdPoint::Finite(1) {
          // u = 1 ∉ Û_d → round should be filtered
          assert!(!has_round, "β={:?} should NOT have round {} (u=1)", beta, round);
        } else {
          // u ∈ Û_d → round should be present (suffix is always binary for binary β)
          assert!(has_round, "β={:?} should have round {} (u≠1)", beta, round);
        }
      }
    }
  }

  #[test]
  fn test_compute_idx4_index_bounds() {
    let l0 = 4;
    const D: usize = 3;
    let base = D + 1;
    let prefix_ud_size = base.pow(l0 as u32);

    for beta_idx in 0..prefix_ud_size {
      let beta = UdTuple::<D>::from_flat_index(beta_idx, l0);
      let contributions = compute_idx4(&beta, l0);

      for c in contributions {
        // l0 should match
        assert_eq!(c.l0, l0);

        // Round bound: i ∈ [1, ℓ₀]
        assert!(
          c.round >= 1 && c.round <= l0,
          "round {} out of [1, {}] for β={:?}",
          c.round,
          l0,
          beta
        );

        // v_idx bound: v ∈ U_d^{i-1}, so v_idx < (d+1)^{i-1}
        let max_v_idx = base.pow(c.prefix_len() as u32);
        assert!(
          c.v_idx < max_v_idx,
          "v_idx {} >= {} for round {} β={:?}",
          c.v_idx,
          max_v_idx,
          c.round,
          beta
        );

        // u should be valid UdHatPoint (this is enforced by type system)
        let u_idx = c.u.to_index();
        assert!(u_idx < D, "u_idx {} >= {} for β={:?}", u_idx, D, beta);

        // y_idx bound: y ∈ {0,1}^{ℓ₀-i}, so y_idx < 2^{ℓ₀-i}
        let max_y_idx = 1usize << c.suffix_len();
        assert!(
          c.y_idx < max_y_idx,
          "y_idx {} >= {} for round {} β={:?}",
          c.y_idx,
          max_y_idx,
          c.round,
          beta
        );
      }
    }
  }

  #[test]
  fn test_compute_idx4_v_u_decode() {
    let l0 = 3;
    const D: usize = 3;

    // Test cases that have at least some contributions
    // Note: all Finite(1) has NO contributions so we exclude it
    let test_cases: Vec<UdTuple<D>> = vec![
      tuple_from_indices(&[1, 1, 1]), // all u=Finite(0), 3 contributions
      tuple_from_indices(&[0, 1, 1]), // u=∞,Finite(0),Finite(0), 3 contributions
      tuple_from_indices(&[3, 0, 1]), // u=Finite(2),∞,Finite(0), 3 contributions
      tuple_from_indices(&[3, 3, 3]), // u=Finite(2)×3, only round 3 due to non-binary suffix
    ];

    for beta in test_cases {
      let contributions = compute_idx4(&beta, l0);

      for c in contributions {
        // v should be β[0..round-1]
        let expected_v = UdTuple(beta.0[0..c.prefix_len()].to_vec());
        let expected_v_idx = expected_v.to_flat_index();
        assert_eq!(
          c.v_idx, expected_v_idx,
          "v_idx mismatch for round {} β={:?}",
          c.round, beta
        );

        // u should be the Û_d point from β[round-1]
        let u = beta.0[c.round - 1];
        let expected_u = u.to_ud_hat().unwrap();
        assert_eq!(
          c.u, expected_u,
          "u mismatch for round {} β={:?}: expected {:?} (from {:?})",
          c.round, beta, expected_u, u
        );
      }
    }
  }

  #[test]
  fn test_compute_idx4_y_idx_encoding() {
    let l0 = 4;
    const D: usize = 3;

    // β = (Finite(0), Finite(1), Finite(0), Finite(1)) → point (0, 1, 0, 1)
    // Rounds 2 and 4 filtered because u = value 1 ∉ Û_d
    let beta = tuple_from_indices::<D>(&[1, 2, 1, 2]);
    let contributions = compute_idx4(&beta, l0);

    assert_eq!(contributions.len(), 2); // Only rounds 1 and 3

    // Round 1: u=Finite(0) ∈ Û_d, y = suffix → bits (1,0,1) → y_idx = 5
    let c1 = contributions.iter().find(|c| c.round == 1).unwrap();
    assert_eq!(c1.y_idx, 0b101);
    assert_eq!(c1.suffix_len(), 3);
    assert_eq!(c1.u, UdHatPoint::Finite(0));

    // Round 2: FILTERED (u = Finite(1) ∉ Û_d)
    assert!(contributions.iter().all(|c| c.round != 2));

    // Round 3: u=Finite(0) ∈ Û_d, y = suffix → bits (1,) → y_idx = 1
    let c3 = contributions.iter().find(|c| c.round == 3).unwrap();
    assert_eq!(c3.y_idx, 0b1);
    assert_eq!(c3.suffix_len(), 1);
    assert_eq!(c3.u, UdHatPoint::Finite(0));

    // Round 4: FILTERED (u = Finite(1) ∉ Û_d)
    assert!(contributions.iter().all(|c| c.round != 4));
  }

  #[test]
  fn test_compute_idx4_y_idx_encoding_no_filtering() {
    // Test y_idx encoding with a β that has no filtering
    let l0 = 4;
    const D: usize = 3;

    // β = (Finite(0), Finite(0), Finite(0), Finite(0)) → all zeros
    // All u = Finite(0) ∈ Û_d, so all rounds present
    let beta = tuple_from_indices::<D>(&[1, 1, 1, 1]);
    let contributions = compute_idx4(&beta, l0);

    assert_eq!(contributions.len(), l0);

    // Round 1: y = suffix → bits (0,0,0) → y_idx = 0
    let c1 = contributions.iter().find(|c| c.round == 1).unwrap();
    assert_eq!(c1.y_idx, 0b000);
    assert_eq!(c1.suffix_len(), 3);

    // Round 2: y = suffix → bits (0,0) → y_idx = 0
    let c2 = contributions.iter().find(|c| c.round == 2).unwrap();
    assert_eq!(c2.y_idx, 0b00);
    assert_eq!(c2.suffix_len(), 2);

    // Round 3: y = suffix → bits (0,) → y_idx = 0
    let c3 = contributions.iter().find(|c| c.round == 3).unwrap();
    assert_eq!(c3.y_idx, 0b0);
    assert_eq!(c3.suffix_len(), 1);

    // Round 4: y = suffix → () → y_idx = 0
    let c4 = contributions.iter().find(|c| c.round == 4).unwrap();
    assert_eq!(c4.y_idx, 0);
    assert_eq!(c4.suffix_len(), 0);
  }

  #[test]
  fn test_compute_idx4_suffix_must_be_binary_values() {
    // Two conditions filter rounds:
    // 1. The suffix y must consist only of binary values (Finite(0) or Finite(1))
    // 2. The coordinate u must be in Û_d (i.e., u ≠ Finite(1))

    let l0 = 3;
    const D: usize = 3;

    // β with non-binary value (Finite(2)) in suffix position
    // indices (1, 3, 2) → values (Finite(0), Finite(2), Finite(1))
    let beta = tuple_from_indices::<D>(&[1, 3, 2]);
    let contributions = compute_idx4(&beta, l0);

    // Round 1: u=Finite(0) ∈ Û_d ✓, suffix has Finite(2) → SKIP (non-binary suffix)
    // Round 2: u=Finite(2) ∈ Û_d ✓, suffix has Finite(1) binary ✓ → OK
    // Round 3: u=Finite(1) ∉ Û_d ✗ → SKIP (u not in Û_d)
    assert_eq!(contributions.len(), 1); // Only round 2
    assert!(contributions.iter().any(|c| c.round == 2));
    assert!(contributions.iter().all(|c| c.round != 1 && c.round != 3));

    // β with ∞ in suffix position
    // indices (2, 0, 1) → values (Finite(1), ∞, Finite(0))
    let beta = tuple_from_indices::<D>(&[2, 0, 1]);
    let contributions = compute_idx4(&beta, l0);

    // Round 1: u=Finite(1) ∉ Û_d ✗ → SKIP (u not in Û_d)
    // Round 2: u=∞ ∈ Û_d ✓, suffix has Finite(0) binary ✓ → OK
    // Round 3: u=Finite(0) ∈ Û_d ✓, suffix=() binary ✓ → OK
    assert_eq!(contributions.len(), 2); // Rounds 2 and 3
    assert!(contributions.iter().all(|c| c.round != 1));
    assert!(contributions.iter().any(|c| c.round == 2));
    assert!(contributions.iter().any(|c| c.round == 3));
  }

  #[test]
  fn test_cubic_accumulator_prefix_index_alias() {
    // Verify the type alias works
    let idx: CubicAccumulatorPrefixIndex = AccumulatorPrefixIndex {
      l0: 3,
      round: 2,
      v_idx: 5,
      u: UdHatPoint::Infinity,
      y_idx: 1,
    };

    assert_eq!(idx.round_0idx(), 1);
    assert_eq!(idx.prefix_len(), 1);
    assert_eq!(idx.suffix_len(), 1);
  }
}
