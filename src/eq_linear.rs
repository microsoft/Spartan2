// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Utilities for computing the per-round linear equality factor in sum-check.
#![allow(dead_code)]

use ff::PrimeField;

/// The per-round linear equality factor values ℓ_i(0), ℓ_i(1), and ℓ_i(∞).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EqRoundValues<F: PrimeField> {
  /// ℓ_i(0) = α_i · (1 − w_i).
  pub(crate) l0: F,
  /// ℓ_i(1) = α_i · w_i.
  pub(crate) l1: F,
  /// ℓ_i(∞) = α_i · (2w_i − 1).
  pub(crate) linf: F,
}

impl<F: PrimeField> EqRoundValues<F> {
  /// Evaluates ℓ_i at an arbitrary point u.
  pub(crate) fn eval_at(&self, u: F) -> F {
    self.linf * u + self.l0
  }
}

/// Tracks α_i = eqe(w_{<i}, r_{<i}) to build ℓ_i values each round.
pub(crate) struct EqRoundFactor<F: PrimeField> {
  alpha: F,
}

impl<F: PrimeField> EqRoundFactor<F> {
  /// Creates a new tracker with α_0 = 1.
  pub(crate) fn new() -> Self {
    Self { alpha: F::ONE }
  }

  /// Returns the current prefix product α_i.
  pub(crate) fn alpha(&self) -> F {
    self.alpha
  }

  /// Returns ℓ_i(0), ℓ_i(1), ℓ_i(∞) for the provided w_i.
  pub(crate) fn values(&self, w_i: F) -> EqRoundValues<F> {
    let l0 = self.alpha * (F::ONE - w_i);
    let l1 = self.alpha * w_i;
    let linf = self.alpha * (w_i.double() - F::ONE);
    EqRoundValues { l0, l1, linf }
  }

  /// Advances α using ℓ_i values and the verifier challenge r_i.
  pub(crate) fn advance_from_values(&mut self, li: &EqRoundValues<F>, r_i: F) {
    self.alpha = li.eval_at(r_i);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;
  use ff::Field;

  type F = pallas::Scalar;

  fn eqe_bit(w: F, x: F) -> F {
    F::ONE - w - x + (w * x).double()
  }

  // Round 0 has alpha = 1; check the base formulas for l0, l1, linf.
  #[test]
  fn test_values_round0_basic() {
    let w = F::from(3u64);
    let tracker = EqRoundFactor::new();
    let v = tracker.values(w);

    assert_eq!(v.l0, F::ONE - w);
    assert_eq!(v.l1, w);
    assert_eq!(v.linf, w.double() - F::ONE);
  }

  // Invariants: l0 + l1 = alpha and linf = l1 - l0 after one advance.
  #[test]
  fn test_values_relations_hold() {
    let w0 = F::from(7u64);
    let w1 = F::from(5u64);
    let r0 = F::from(9u64);
    let mut tracker = EqRoundFactor::new();

    let v0 = tracker.values(w0);
    tracker.advance_from_values(&v0, r0);

    let v1 = tracker.values(w1);
    assert_eq!(v1.l0 + v1.l1, tracker.alpha());
    assert_eq!(v1.linf, v1.l1 - v1.l0);
  }

  // eval_at(u) should agree with stored points and derived l(2).
  #[test]
  fn test_li_at_matches_values() {
    let w = F::from(11u64);
    let tracker = EqRoundFactor::new();
    let v = tracker.values(w);

    assert_eq!(v.eval_at(F::ZERO), v.l0);
    assert_eq!(v.eval_at(F::ONE), v.l1);
    assert_eq!(v.eval_at(F::from(2u64)), v.linf.double() + v.l0);
  }

  // advance_from_values should update alpha by eqe(w, r).
  #[test]
  fn test_advance_updates_alpha() {
    let w0 = F::from(4u64);
    let r0 = F::from(6u64);
    let mut tracker = EqRoundFactor::new();
    let alpha0 = tracker.alpha();

    let v0 = tracker.values(w0);
    tracker.advance_from_values(&v0, r0);

    let expected = alpha0 * eqe_bit(w0, r0);
    assert_eq!(tracker.alpha(), expected);
  }

  // Repeated updates should match the product of eqe(w_i, r_i).
  #[test]
  fn test_alpha_matches_product() {
    let taus = vec![F::from(2u64), F::from(5u64), F::from(8u64)];
    let rs = vec![F::from(3u64), F::from(4u64), F::from(7u64)];
    let mut tracker = EqRoundFactor::new();

    let mut expected = F::ONE;
    for (tau, r) in taus.into_iter().zip(rs.into_iter()) {
      let v = tracker.values(tau);
      tracker.advance_from_values(&v, r);
      expected *= eqe_bit(tau, r);
      assert_eq!(tracker.alpha(), expected);
    }
  }

  // Degenerate endpoints: w=0 and w=1 should yield expected l0/l1/linf.
  #[test]
  fn test_values_degenerate_case_for_w_zero_and_one() {
    let tracker = EqRoundFactor::<F>::new();

    let v0 = tracker.values(F::ZERO);
    assert_eq!(v0.l0, F::ONE);
    assert_eq!(v0.l1, F::ZERO);
    assert_eq!(v0.linf, -F::ONE);

    let v1 = tracker.values(F::ONE);
    assert_eq!(v1.l0, F::ZERO);
    assert_eq!(v1.l1, F::ONE);
    assert_eq!(v1.linf, F::ONE);
  }

  // For w = 1/2, slope should be zero (linf = 0).
  #[test]
  fn test_slope_zero_at_half() {
    let half = F::from(2u64).invert().unwrap();
    let tracker = EqRoundFactor::new();
    let v = tracker.values(half);

    assert_eq!(v.linf, F::ZERO);
    assert_eq!(v.l0 + v.l1, F::ONE);
  }
}
