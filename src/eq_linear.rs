// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Utilities for computing the per-round linear equality factor in sum-check.
#![allow(dead_code)]

use crate::lagrange::UdEvaluations;
use ff::PrimeField;

/// Derives t_i(1) using the sumcheck relation: claim = ℓ_i(0)·t(0) + ℓ_i(1)·t(1).
///
/// Returns `None` if `l1` is zero (non-invertible).
pub(crate) fn derive_t1<F: PrimeField>(l0: F, l1: F, claim_prev: F, t0: F) -> Option<F> {
  let s0 = l0 * t0;
  let s1 = claim_prev - s0;
  l1.invert().into_option().map(|inv| s1 * inv)
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

  /// Returns ℓ_i evaluated at U_2 = {∞, 0, 1} for the provided w_i.
  ///
  /// - `infinity` = ℓ_i(∞) = α_i · (2w_i − 1)
  /// - `finite[0]` = ℓ_i(0) = α_i · (1 − w_i)
  /// - `finite[1]` = ℓ_i(1) = α_i · w_i
  pub(crate) fn values(&self, w_i: F) -> UdEvaluations<F, 2> {
    let l0 = self.alpha * (F::ONE - w_i);
    let l1 = self.alpha * w_i;
    let linf = self.alpha * (w_i.double() - F::ONE);
    UdEvaluations::new(linf, [l0, l1])
  }

  /// Advances α using ℓ_i(r_i) = linf * r_i + l0.
  pub(crate) fn advance(&mut self, li: &UdEvaluations<F, 2>, r_i: F) {
    self.alpha = li.eval_linear_at(r_i);
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

    assert_eq!(v.at_zero(), F::ONE - w);
    assert_eq!(v.at_one(), w);
    assert_eq!(v.at_infinity(), w.double() - F::ONE);
  }

  // Invariants: l0 + l1 = alpha and linf = l1 - l0 after one advance.
  #[test]
  fn test_values_relations_hold() {
    let w0 = F::from(7u64);
    let w1 = F::from(5u64);
    let r0 = F::from(9u64);
    let mut tracker = EqRoundFactor::new();

    let v0 = tracker.values(w0);
    tracker.advance(&v0, r0);

    let v1 = tracker.values(w1);
    assert_eq!(v1.at_zero() + v1.at_one(), tracker.alpha());
    assert_eq!(v1.at_infinity(), v1.at_one() - v1.at_zero());
  }

  // eval_linear_at(u) should agree with stored points and derived l(2).
  #[test]
  fn test_li_at_matches_values() {
    let w = F::from(11u64);
    let tracker = EqRoundFactor::new();
    let v = tracker.values(w);

    assert_eq!(v.eval_linear_at(F::ZERO), v.at_zero());
    assert_eq!(v.eval_linear_at(F::ONE), v.at_one());
    assert_eq!(v.eval_linear_at(F::from(2u64)), v.at_infinity().double() + v.at_zero());
  }

  // advance should update alpha by eqe(w, r).
  #[test]
  fn test_advance_updates_alpha() {
    let w0 = F::from(4u64);
    let r0 = F::from(6u64);
    let mut tracker = EqRoundFactor::new();
    let alpha0 = tracker.alpha();

    let v0 = tracker.values(w0);
    tracker.advance(&v0, r0);

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
      tracker.advance(&v, r);
      expected *= eqe_bit(tau, r);
      assert_eq!(tracker.alpha(), expected);
    }
  }

  // Degenerate endpoints: w=0 and w=1 should yield expected l0/l1/linf.
  #[test]
  fn test_values_degenerate_case_for_w_zero_and_one() {
    let tracker = EqRoundFactor::<F>::new();

    let v0 = tracker.values(F::ZERO);
    assert_eq!(v0.at_zero(), F::ONE);
    assert_eq!(v0.at_one(), F::ZERO);
    assert_eq!(v0.at_infinity(), -F::ONE);

    let v1 = tracker.values(F::ONE);
    assert_eq!(v1.at_zero(), F::ZERO);
    assert_eq!(v1.at_one(), F::ONE);
    assert_eq!(v1.at_infinity(), F::ONE);
  }

  // For w = 1/2, slope should be zero (linf = 0).
  #[test]
  fn test_slope_zero_at_half() {
    let half = F::from(2u64).invert().unwrap();
    let tracker = EqRoundFactor::new();
    let v = tracker.values(half);

    assert_eq!(v.at_infinity(), F::ZERO);
    assert_eq!(v.at_zero() + v.at_one(), F::ONE);
  }

  // derive_t1 should return s1 / l1 for non-zero l1.
  #[test]
  fn test_derive_t1_returns_value() {
    let l0 = F::from(2u64);
    let l1 = F::from(5u64);
    let t0 = F::from(11u64);
    let claim = F::from(97u64);

    let s0 = l0 * t0;
    let s1 = claim - s0;
    let expected = s1 * l1.invert().unwrap();

    assert_eq!(derive_t1(l0, l1, claim, t0), Some(expected));
  }

  // derive_t1 should return None when l1 == 0.
  #[test]
  fn test_derive_t1_returns_none_on_zero_l1() {
    let l0 = F::from(3u64);
    let l1 = F::ZERO;
    let t0 = F::from(4u64);
    let claim = F::from(10u64);

    assert_eq!(derive_t1(l0, l1, claim, t0), None);
  }
}
