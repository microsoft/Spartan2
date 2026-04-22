// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value sumcheck implementation for the first ℓ₀ rounds of Spartan's
//! outer cubic sum-check.
//!
//! This module provides optimized sumcheck proving using native integer
//! arithmetic for polynomials with small coefficients. The key optimization
//! is replacing expensive field multiplications with native integer operations
//! during the initial rounds when polynomial values are guaranteed to be small.
//!
//! This implementation is not a generic small-value prover for arbitrary
//! `A(X) * B(X) - C(X)` relations. It relies on the Spartan outer-sumcheck
//! structure:
//! - `A(x) * B(x) - C(x) = 0` on the Boolean hypercube for satisfying witnesses
//! - contributions with an `∞` prefix coordinate only need the highest-degree
//!   term, so the linear `C` term drops out there
//!
//! Based on "Speeding Up Sum-Check Proving" by Suyash Bagad, Quang Dao,
//! Yuval Domb, and Justin Thaler. <https://eprint.iacr.org/2025/1117.pdf>
//!
//! # Overview
//!
//! The main entry point is [`prove_cubic_small_value`], which implements

use crate::{
  big_num::{DelayedReduction, SmallValue, SmallValueEngine, SmallValueField},
  errors::SpartanError,
  lagrange_accumulator::{
    LagrangeAccumulators, LagrangeBasisFactory, LagrangeCoeff, LagrangeDomainEvals,
    ReducedLagrangeDomainEvals, SPARTAN_T_DEGREE, build_accumulators_spartan,
  },
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial, univariate::UniPoly},
  start_span,
  sumcheck::{SumcheckProof, eq_sumcheck},
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use ff::PrimeField;
use itertools::Itertools;
use num_traits::Zero;
use rayon::prelude::*;
use tracing::info;

use crate::sumcheck::PAR_THRESHOLD;

/// Tracks the small-value sum-check state for the first ℓ₀ rounds.
///
/// This struct maintains the precomputed accumulators and running state
/// needed to efficiently evaluate round polynomials using native integer
/// arithmetic instead of field operations.
struct SmallValueSumCheck<Scalar: PrimeField, const D: usize> {
  accumulators: LagrangeAccumulators<Scalar, D>,
  coeff: LagrangeCoeff<Scalar, D>,
  eq_alpha: Scalar,
  basis_factory: LagrangeBasisFactory<Scalar, D>,
}


/// Prove Spartan's outer `poly_A * poly_B - poly_C` relation using Algorithm 6
/// (EqPoly-SmallValueSC).
///
/// This function combines small-value optimization (Algorithm 4) for the first ℓ₀ rounds
/// with eq-poly optimization (Algorithm 5) for the remaining rounds.
///
/// Field-element polynomials are created internally via batched eq-weighted binding
/// from the small-value inputs, eliminating the need for pre-allocated field polys.
///
/// This path is Spartan-outer-specific. It assumes:
/// - the witness satisfies `A(x) * B(x) - C(x) = 0` on `{0,1}^n`
/// - for evaluation points containing `∞`, only the highest-degree term matters,
///   so `C` does not contribute to the accumulator
///
/// Generic over `SmallValue` to support both i32/i64 and i64/i128 configurations.
///
/// # Type Parameters
///
/// - `LB`: Number of small-value rounds (ℓ₀). The actual number of rounds used is
///   `min(LB, num_rounds.saturating_sub(1))`, so the transition path always leaves
///   at least one suffix variable for the standard eq-sumcheck continuation.
///   Caller should ensure input values are bounded by roughly `SV::MAX / 3^LB`
///   for safe Lagrange extension, since extending a Boolean prefix to
///   `U_2 = {∞, 0, 1}` can grow magnitudes by a factor of about `3^LB`.
///   Typical values are 3-4 for practical instances (3^4 = 81× growth factor).
pub fn prove_cubic_small_value<E, SV, const LB: usize>(
  claim: &E::Scalar,
  taus: Vec<E::Scalar>,
  poly_A_small: &MultilinearPolynomial<SV>,
  poly_B_small: &MultilinearPolynomial<SV>,
  poly_C_small: &MultilinearPolynomial<SV>,
  transcript: &mut E::TE,
) -> Result<(SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  let num_rounds = taus.len();
  let mut r: Vec<E::Scalar> = Vec::with_capacity(num_rounds);
  let mut polys: Vec<crate::polys::univariate::CompressedUniPoly<E::Scalar>> =
    Vec::with_capacity(num_rounds);
  let mut claim_per_round = *claim;

  // Determine ℓ₀: number of small-value rounds (at most LB, but must be < num_rounds
  // to leave at least one suffix variable for the transition phase)
  let l0 = std::cmp::min(LB, num_rounds.saturating_sub(1));

  // If l0 is 0, the small-value optimization is not applicable
  if l0 == 0 {
    let mut poly_a = small_poly_to_field::<E::Scalar, SV>(poly_A_small);
    let mut poly_b = small_poly_to_field::<E::Scalar, SV>(poly_B_small);
    let mut poly_c = small_poly_to_field::<E::Scalar, SV>(poly_C_small);
    return SumcheckProof::prove_cubic_with_three_inputs(
      claim,
      taus,
      &mut poly_a,
      &mut poly_b,
      &mut poly_c,
      transcript,
    );
  }

  // ===== Pre-computation Phase =====
  // Build accumulators A_i(v, u) for all i ∈ [ℓ₀] using small-value arithmetic.
  // Also builds eq pyramids for reuse by EqSumCheckInstance.
  // Internally computes eq tables with balanced split and precomputed eq_cache.
  // Uses: small × small → intermediate (for Az·Bz products),
  // then intermediate × field (for eq weighting via DelayedReduction).
  let (accumulators, _e_in_pyramid, _e_xout_pyramid) =
    build_accumulators_spartan(poly_A_small, poly_B_small, &taus, l0);

  let mut small_value_sumcheck =
    SmallValueSumCheck::<E::Scalar, SPARTAN_T_DEGREE>::from_accumulators(accumulators);
  let mut transition_round = l0;

  // ===== Small-Value Rounds (0 to ℓ₀-1) =====
  // During these rounds, we use the precomputed accumulators. Polynomials are NOT bound
  // during these rounds - that will happen in the transition phase.
  #[allow(clippy::needless_range_loop)]
  for round in 0..l0 {
    let (_round_span, round_t) = start_span!("sumcheck_smallvalue_round", round = round);

    // 1. Get t_i evaluations from accumulators
    let t_all = small_value_sumcheck.eval_t_all_u(round);
    let t_inf = t_all.at_infinity();
    let t0 = t_all.at_zero();

    // 2. Get eq factor values ℓ_i(0), ℓ_i(1), ℓ_i(∞)
    let li = small_value_sumcheck.eq_round_values(taus[round]);

    // 3. Derive t(1) from the sumcheck constraint. If ℓ_i(1)=0, the optimized
    // path cannot recover t_i(1), so we fall back to the standard prover from
    // this round onward.
    let Some(t1) = derive_t1(li.at_zero(), li.at_one(), claim_per_round, t0) else {
      transition_round = round;
      break;
    };

    // 4. Build round polynomial s_i(X) = ℓ_i(X) · t_i(X)
    let poly = build_univariate_round_polynomial(&li, t0, t1, t_inf);

    // 5. Transcript interaction
    transcript.absorb(b"p", &poly);
    let r_i = transcript.squeeze(b"c")?;
    r.push(r_i);
    polys.push(poly.compress());

    // 6. Update claim
    claim_per_round = poly.evaluate(&r_i);

    // 7. Advance small-value state (updates R_{i+1} and the prefix eq factor)
    small_value_sumcheck.advance(&li, r_i);

    info!(
      elapsed_ms = %round_t.elapsed().as_millis(),
      round = round,
      "sumcheck_smallvalue_round"
    );
  }

  // ===== Transition Phase =====
  // Create bound field-element polynomials via batched eq-weighted small-value accumulation.
  // Binds all completed small-value rounds in a single pass using field×int
  // unreduced accumulation.
  let (_bind_span, bind_t) = start_span!("bind_poly_vars_transition");
  let (mut poly_A, mut poly_B, mut poly_C) = bind_three_polys_batched_small_value(
    poly_A_small,
    poly_B_small,
    poly_C_small,
    &r[..transition_round],
  );
  info!(
    elapsed_ms = %bind_t.elapsed().as_millis(),
    "bind_poly_vars_transition"
  );

  // ===== Remaining Rounds (ℓ₀ to ℓ-1) =====
  let mut eq_instance = eq_sumcheck::EqSumCheckInstance::<E>::new_with_eval_eq_left(
    &taus[transition_round..],
    small_value_sumcheck.eq_alpha(),
  );

  // Continue with the remaining rounds using the standard eq instance seeded with the
  // accumulated prefix eq factor from the small-value rounds.
  for round in transition_round..num_rounds {
    let (_round_span, round_t) = start_span!("sumcheck_round", round = round);

    let poly = {
      let (_eval_span, eval_t) = start_span!("compute_eval_points");
      let (eval_point_0, eval_point_2, eval_point_3) = eq_instance
        .evaluation_points_cubic_with_three_inputs(&poly_A, &poly_B, &poly_C, claim_per_round);
      if eval_t.elapsed().as_millis() > 0 {
        info!(elapsed_ms = %eval_t.elapsed().as_millis(), "compute_eval_points");
      }

      let evals = [
        eval_point_0,
        claim_per_round - eval_point_0,
        eval_point_2,
        eval_point_3,
      ];
      UniPoly::from_evals(&evals)?
    };

    // Transcript interaction
    transcript.absorb(b"p", &poly);
    let r_i = transcript.squeeze(b"c")?;
    r.push(r_i);
    polys.push(poly.compress());

    // Update claim
    claim_per_round = poly.evaluate(&r_i);

    // Bind polynomials and advance eq instance
    let (_bind_span, bind_t) = start_span!("bind_poly_vars");
    poly_A.bind_poly_var_top(&r_i);
    poly_B.bind_poly_var_top(&r_i);
    poly_C.bind_poly_var_top(&r_i);
    eq_instance.bound(&r_i);
    info!(elapsed_ms = %bind_t.elapsed().as_millis(), "bind_poly_vars");
    info!(elapsed_ms = %round_t.elapsed().as_millis(), round = round, "sumcheck_round");
  }

  Ok((
    SumcheckProof {
      compressed_polys: polys,
    },
    r,
    vec![poly_A[0], poly_B[0], poly_C[0]],
  ))
}

impl<Scalar: PrimeField, const D: usize> SmallValueSumCheck<Scalar, D> {
  /// Create a new small-value round tracker with precomputed accumulators.
  fn new(
    accumulators: LagrangeAccumulators<Scalar, D>,
    basis_factory: LagrangeBasisFactory<Scalar, D>,
  ) -> Self {
    Self {
      accumulators,
      coeff: LagrangeCoeff::new(),
      eq_alpha: Scalar::ONE,
      basis_factory,
    }
  }

  /// Create from accumulators with the standard Lagrange basis (0, 1, 2, ...).
  fn from_accumulators(accumulators: LagrangeAccumulators<Scalar, D>) -> Self {
    let basis_factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));
    Self::new(accumulators, basis_factory)
  }

  /// Evaluate t_i(u) for all u ∈ Û_D in a single pass for round i.
  fn eval_t_all_u(&self, round: usize) -> ReducedLagrangeDomainEvals<Scalar, D> {
    self.accumulators.eval_t_all_u(round, &self.coeff)
  }

  /// Compute ℓ_i values for the provided w_i.
  fn eq_round_values(&self, w_i: Scalar) -> LagrangeDomainEvals<Scalar, 2> {
    let l0 = self.eq_alpha * (Scalar::ONE - w_i);
    let l1 = self.eq_alpha * w_i;
    let linf = self.eq_alpha * (w_i.double() - Scalar::ONE);
    LagrangeDomainEvals::new(linf, [l0, l1])
  }

  /// Advance the round state with the verifier challenge r_i.
  fn advance(&mut self, li: &LagrangeDomainEvals<Scalar, 2>, r_i: Scalar) {
    self.eq_alpha = li.eval_linear_at(r_i);
    self.coeff.extend(&self.basis_factory.basis_at(r_i));
  }

  /// Returns the accumulated eq factor α = eq(τ_{0..i}, r_{0..i}).
  ///
  /// After l0 rounds, this gives the eq factor that must be incorporated
  /// into the remaining sumcheck rounds.
  fn eq_alpha(&self) -> Scalar {
    self.eq_alpha
  }
}

/// Derive `t_i(1)` from the sumcheck relation
/// `claim_prev = ℓ_i(0) · t_i(0) + ℓ_i(1) · t_i(1)`.
///
/// Returns `None` when `ℓ_i(1) = 0`, since the optimized path cannot recover
/// `t_i(1)` in that case.
fn derive_t1<F: PrimeField>(l0: F, l1: F, claim_prev: F, t0: F) -> Option<F> {
  let s0 = l0 * t0;
  let s1 = claim_prev - s0;
  l1.invert().into_option().map(|inv| s1 * inv)
}

/// Build the cubic round polynomial s_i(X) in coefficient form for Spartan.
///
/// Constructs s_i(X) = ℓ_i(X) · t_i(X) where:
/// - ℓ_i(X) is the linear eq factor
/// - t_i(X) is the degree-2 polynomial from accumulators
fn build_univariate_round_polynomial<F: PrimeField>(
  li: &LagrangeDomainEvals<F, 2>,
  t0: F,
  t1: F,
  t_inf: F,
) -> UniPoly<F> {
  // Reconstruct t_i(X) = aX^2 + bX + c using:
  // - a = t_i(∞) (leading coefficient for degree-2 polynomials)
  // - c = t_i(0)
  // - t_i(1) = a + b + c ⇒ b = t_i(1) − a − c
  let a = t_inf;
  let c = t0;
  let b = t1 - a - c;

  let linf = li.at_infinity();
  let l0 = li.at_zero();

  // Multiply s_i(X) = ℓ_i(X)·t_i(X) with ℓ_i(X)=ℓ_∞X+ℓ_0 and collect coefficients.
  let s3 = linf * a;
  let s2 = linf * b + l0 * a;
  let s1 = linf * c + l0 * b;
  let s0 = l0 * c;

  UniPoly {
    coeffs: vec![s0, s1, s2, s3],
  }
}

/// Batch-bind l0 top variables of three polynomials using eq-weighted small-value accumulation.
///
/// Instead of l0 sequential passes with field×field muls, computes:
///   `poly_out[s] = Σ_{p ∈ {0,1}^l0} eq(challenges, p) · poly_small[p * stride + s]`
/// in one pass using field×int unreduced accumulation with a single final reduction.
fn bind_three_polys_batched_small_value<F, SV>(
  poly_a_small: &MultilinearPolynomial<SV>,
  poly_b_small: &MultilinearPolynomial<SV>,
  poly_c_small: &MultilinearPolynomial<SV>,
  challenges: &[F],
) -> (
  MultilinearPolynomial<F>,
  MultilinearPolynomial<F>,
  MultilinearPolynomial<F>,
)
where
  SV: Copy + Send + Sync,
  F: PrimeField + DelayedReduction<SV>,
{
  let l0 = challenges.len();
  let n = poly_a_small.Z.len();
  debug_assert_eq!(poly_b_small.Z.len(), n);
  debug_assert_eq!(poly_c_small.Z.len(), n);
  debug_assert_eq!(n % (1 << l0), 0);

  let num_prefixes = 1usize << l0;
  let stride = n >> l0;

  // Precompute eq(challenges, p) for all p ∈ {0,1}^l0
  let eq_table = EqPolynomial::evals_from_points(challenges);
  debug_assert_eq!(eq_table.len(), num_prefixes);

  type Acc<F2, SV2> = <F2 as DelayedReduction<SV2>>::Accumulator;

  // Suffix-outer parallel loop: accumulators live on stack per thread
  let compute = |s: usize| -> (F, F, F) {
    let mut acc_a = Acc::<F, SV>::zero();
    let mut acc_b = Acc::<F, SV>::zero();
    let mut acc_c = Acc::<F, SV>::zero();

    for (p, eq_p) in eq_table.iter().enumerate() {
      let idx = p * stride + s;

      // Single-value accumulation: field × small with delayed reduction
      F::unreduced_multiply_accumulate(&mut acc_a, eq_p, &poly_a_small.Z[idx]);
      F::unreduced_multiply_accumulate(&mut acc_b, eq_p, &poly_b_small.Z[idx]);
      F::unreduced_multiply_accumulate(&mut acc_c, eq_p, &poly_c_small.Z[idx]);
    }

    (F::reduce(&acc_a), F::reduce(&acc_b), F::reduce(&acc_c))
  };

  let results: Vec<(F, F, F)> = if stride >= PAR_THRESHOLD {
    (0..stride).into_par_iter().map(compute).collect()
  } else {
    (0..stride).map(compute).collect()
  };

  let (out_a, out_b, out_c): (Vec<_>, Vec<_>, Vec<_>) = results.into_iter().multiunzip();

  (
    MultilinearPolynomial::new(out_a),
    MultilinearPolynomial::new(out_b),
    MultilinearPolynomial::new(out_c),
  )
}

/// Convert a small-value polynomial to its field-valued representation.
fn small_poly_to_field<F, SV>(poly: &MultilinearPolynomial<SV>) -> MultilinearPolynomial<F>
where
  F: PrimeField + SmallValueField<SV>,
  SV: Copy,
{
  MultilinearPolynomial::new(poly.Z.iter().copied().map(F::small_to_field).collect())
}


#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    big_num::{SmallValue, SmallValueEngine, SmallValueField},
    polys::multilinear::MultilinearPolynomial,
    provider::PallasHyraxEngine,
    sumcheck::eq_sumcheck::EqSumCheckInstance,
    traits::{Engine, transcript::TranscriptEngineTrait},
  };
  use ff::Field;
  use rand::{Rng, SeedableRng, rngs::StdRng};
  use std::{fmt::Debug, ops::Mul};

  type E = PallasHyraxEngine;
  type F = <E as Engine>::Scalar;

  fn eqe_bit(w: F, x: F) -> F {
    F::ONE - w - x + (w * x).double()
  }

  #[test]
  fn test_eq_round_values_matches_formula() {
    let mut small_value = SmallValueSumCheck::<F, SPARTAN_T_DEGREE>::new(
      LagrangeAccumulators::new(1),
      LagrangeBasisFactory::new(|i| F::from(i as u64)),
    );
    small_value.eq_alpha = F::from(13u64);

    let w = F::from(7u64);
    let li = small_value.eq_round_values(w);

    assert_eq!(li.at_zero(), small_value.eq_alpha * (F::ONE - w));
    assert_eq!(li.at_one(), small_value.eq_alpha * w);
    assert_eq!(
      li.at_infinity(),
      small_value.eq_alpha * (w.double() - F::ONE)
    );
  }

  #[test]
  fn test_advance_matches_prefix_eq_product() {
    let mut small_value = SmallValueSumCheck::<F, SPARTAN_T_DEGREE>::new(
      LagrangeAccumulators::new(4),
      LagrangeBasisFactory::new(|i| F::from(i as u64)),
    );
    let taus = [F::from(2u64), F::from(5u64), F::from(8u64)];
    let rs = [F::from(3u64), F::from(4u64), F::from(7u64)];

    let mut expected = F::ONE;
    for (tau, r) in taus.into_iter().zip(rs) {
      let li = small_value.eq_round_values(tau);
      small_value.advance(&li, r);
      expected *= eqe_bit(tau, r);
      assert_eq!(small_value.eq_alpha(), expected);
    }
  }

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

  #[test]
  fn test_derive_t1_returns_none_on_zero_l1() {
    let l0 = F::from(3u64);
    let l1 = F::ZERO;
    let t0 = F::from(4u64);
    let claim = F::from(10u64);

    assert_eq!(derive_t1(l0, l1, claim, t0), None);
  }

  /// Generic helper to test that SmallValueSumCheck produces the same polynomial
  /// evaluations as EqSumCheckInstance across multiple rounds.
  fn run_smallvalue_round_test<SV>()
  where
    SV: SmallValue + Mul<Output = SV> + TryFrom<usize>,
    <SV as TryFrom<usize>>::Error: Debug,
    F: SmallValueEngine<SV>,
  {
    const NUM_VARS: usize = 6;
    const SMALL_VALUE_ROUNDS: usize = 3;

    let n = 1usize << NUM_VARS;
    let taus = (0..NUM_VARS)
      .map(|i| F::from((i + 2) as u64))
      .collect::<Vec<_>>();

    // Small-value polynomials for build_accumulators_spartan
    let az_small: Vec<SV> = (0..n).map(|i| SV::try_from(i + 1).unwrap()).collect();
    let bz_small: Vec<SV> = (0..n).map(|i| SV::try_from(i + 3).unwrap()).collect();
    let cz_small: Vec<SV> = az_small
      .iter()
      .zip(bz_small.iter())
      .map(|(&a, &b)| a * b)
      .collect();

    let az_poly = MultilinearPolynomial::new(az_small.clone());
    let bz_poly = MultilinearPolynomial::new(bz_small.clone());

    // Field polynomials for reference computation
    let az_vals: Vec<F> = az_small.iter().map(|&v| F::small_to_field(v)).collect();
    let bz_vals: Vec<F> = bz_small.iter().map(|&v| F::small_to_field(v)).collect();
    let cz_vals: Vec<F> = cz_small.iter().map(|&v| F::small_to_field(v)).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);
    let cz = MultilinearPolynomial::new(cz_vals);

    // Claim = 0 for satisfying witness (Az·Bz = Cz)
    let mut claim = F::ZERO;

    // Build accumulators using the simplified API
    let (accs, _, _) = build_accumulators_spartan(&az_poly, &bz_poly, &taus, SMALL_VALUE_ROUNDS);
    let mut small_value = SmallValueSumCheck::from_accumulators(accs);

    // Full eq_instance for verification against standard sumcheck
    let mut eq_instance = EqSumCheckInstance::<E>::new(&taus);
    let mut poly_A = az.clone();
    let mut poly_B = bz.clone();
    let mut poly_C = cz.clone();

    for (round, &tau_round) in taus.iter().enumerate().take(SMALL_VALUE_ROUNDS) {
      // Get expected evaluations from standard method
      let (expected_eval_0, expected_eval_2, expected_eval_3) = eq_instance
        .evaluation_points_cubic_with_three_inputs(&poly_A, &poly_B, &poly_C, claim);
      let expected_eval_1 = claim - expected_eval_0; // s(0) + s(1) = claim

      // Build small-value polynomial
      let li = small_value.eq_round_values(tau_round);
      let t_all = small_value.eval_t_all_u(round);
      let t_inf = t_all.at_infinity();
      let t0 = t_all.at_zero();
      let t1 = derive_t1(li.at_zero(), li.at_one(), claim, t0)
        .expect("l1 should be non-zero for chosen taus");

      let poly = build_univariate_round_polynomial(&li, t0, t1, t_inf);

      // Check all 4 evaluation points
      assert_eq!(
        poly.evaluate(&F::ZERO),
        expected_eval_0,
        "s(0) mismatch at round {}",
        round
      );
      assert_eq!(
        poly.evaluate(&F::ONE),
        expected_eval_1,
        "s(1) mismatch at round {}",
        round
      );
      assert_eq!(
        poly.evaluate(&F::from(2u64)),
        expected_eval_2,
        "s(2) mismatch at round {}",
        round
      );
      assert_eq!(
        poly.evaluate(&F::from(3u64)),
        expected_eval_3,
        "s(3) mismatch at round {}",
        round
      );

      // Advance to next round with a fixed challenge
      let r_i = F::from((round + 7) as u64);
      claim = poly.evaluate(&r_i);

      poly_A.bind_poly_var_top(&r_i);
      poly_B.bind_poly_var_top(&r_i);
      poly_C.bind_poly_var_top(&r_i);
      eq_instance.bound(&r_i);
      small_value.advance(&li, r_i);
    }
  }

  #[test]
  fn test_smallvalue_round_matches_eq_instance_evals_i32() {
    run_smallvalue_round_test::<i32>();
  }

  #[test]
  fn test_smallvalue_round_matches_eq_instance_evals_i64() {
    run_smallvalue_round_test::<i64>();
  }

  /// Test that prove_cubic_small_value produces identical
  /// output to prove_cubic_with_three_inputs using synthetic small-value polynomials.
  ///
  /// Uses synthetic Az, Bz values in a small range and computes Cz = Az * Bz.
  fn run_equivalence_test<SV>(num_vars: usize)
  where
    SV: SmallValue + Mul<Output = SV> + TryFrom<i32>,
    <SV as TryFrom<i32>>::Error: Debug,
    F: SmallValueEngine<SV>,
  {
    const SEED: u64 = 0xDEADBEEF;
    let mut rng = StdRng::seed_from_u64(SEED);
    let n = 1usize << num_vars;

    // Generate synthetic small-value polynomials
    // Use values in [-100, 100] range to ensure products fit in i32/i64
    let az_small: Vec<SV> = (0..n)
      .map(|_| SV::try_from(rng.gen_range(-100i32..=100i32)).unwrap())
      .collect();
    let bz_small: Vec<SV> = (0..n)
      .map(|_| SV::try_from(rng.gen_range(-100i32..=100i32)).unwrap())
      .collect();
    // Cz = Az * Bz computed in the small domain
    let cz_small: Vec<SV> = az_small
      .iter()
      .zip(&bz_small)
      .map(|(&a, &b)| a * b)
      .collect();

    // Random taus
    let taus: Vec<F> = (0..num_vars).map(|_| F::random(&mut rng)).collect();

    run_equivalence_test_with_taus::<SV>(taus, az_small, bz_small, cz_small);
  }

  fn run_equivalence_test_with_taus<SV>(
    taus: Vec<F>,
    az_small: Vec<SV>,
    bz_small: Vec<SV>,
    cz_small: Vec<SV>,
  ) where
    SV: SmallValue + Mul<Output = SV>,
    F: SmallValueEngine<SV>,
  {
    let num_vars = taus.len();
    let az_vals: Vec<F> = az_small.iter().map(|&v| F::small_to_field(v)).collect();
    let bz_vals: Vec<F> = bz_small.iter().map(|&v| F::small_to_field(v)).collect();
    let cz_vals: Vec<F> = cz_small.iter().map(|&v| F::small_to_field(v)).collect();

    // Claim = 0 for satisfying witness (Az·Bz = Cz on {0,1}^n)
    let claim: F = F::ZERO;

    // Small-value polynomials
    let az_small_poly = MultilinearPolynomial::new(az_small);
    let bz_small_poly = MultilinearPolynomial::new(bz_small);
    let cz_small_poly = MultilinearPolynomial::new(cz_small);

    // Polynomials for standard method
    let mut az1 = MultilinearPolynomial::new(az_vals);
    let mut bz1 = MultilinearPolynomial::new(bz_vals);
    let mut cz1 = MultilinearPolynomial::new(cz_vals);

    // Fresh transcripts with same seed
    let mut transcript1 = <E as Engine>::TE::new(b"test");
    let mut transcript2 = <E as Engine>::TE::new(b"test");

    // Run standard method
    let (proof1, r1, evals1) = SumcheckProof::<E>::prove_cubic_with_three_inputs(
      &claim,
      taus.clone(),
      &mut az1,
      &mut bz1,
      &mut cz1,
      &mut transcript1,
    )
    .expect("standard prove should succeed");

    // Run small-value method
    let (proof2, r2, evals2) = prove_cubic_small_value::<E, SV, 3>(
      &claim,
      taus.clone(),
      &az_small_poly,
      &bz_small_poly,
      &cz_small_poly,
      &mut transcript2,
    )
    .expect("small-value prove should succeed");

    // Verify all outputs match
    assert_eq!(r1, r2, "challenges must match for num_vars={}", num_vars);
    assert_eq!(
      proof1, proof2,
      "proofs must match for num_vars={}",
      num_vars
    );
    assert_eq!(
      evals1, evals2,
      "final evals must match for num_vars={}",
      num_vars
    );

    // Verify the proof
    let mut transcript_v = <E as Engine>::TE::new(b"test");
    let (final_claim, r_v) = proof1
      .verify(claim, num_vars, 3, &mut transcript_v)
      .expect("verification should succeed");
    assert_eq!(r_v, r1, "verify challenges must match prover");
    let tau_eval = EqPolynomial::new(taus).evaluate(&r_v);
    let expected = tau_eval * (evals1[0] * evals1[1] - evals1[2]);
    assert_eq!(final_claim, expected, "final claim mismatch");
  }

  /// Test small-value sumcheck equivalence with synthetic polynomials.
  ///
  /// Tests multiple sizes to ensure equivalence holds for various l0 values.
  /// With LB=3, l0 = min(3, num_vars - 1):
  /// - num_vars=2: l0=1, suffix_vars=1
  /// - num_vars=3: l0=2, suffix_vars=1
  /// - num_vars=4: l0=3, suffix_vars=1
  /// - num_vars=6: l0=3, suffix_vars=3
  /// - num_vars=10: l0=3, suffix_vars=7
  #[test]
  fn test_sumcheck_equivalence_with_synthetic_i32() {
    for num_vars in [2, 3, 4, 6, 10] {
      run_equivalence_test::<i32>(num_vars);
    }
  }

  #[test]
  fn test_sumcheck_equivalence_with_synthetic_i64() {
    for num_vars in [2, 3, 4, 6, 10] {
      run_equivalence_test::<i64>(num_vars);
    }
  }

  #[test]
  fn test_sumcheck_equivalence_when_first_tau_is_zero() {
    const NUM_VARS: usize = 6;
    const SEED: u64 = 0xDEADBEEF;
    let mut rng = StdRng::seed_from_u64(SEED);
    let n = 1usize << NUM_VARS;

    let az_small: Vec<i32> = (0..n).map(|_| rng.gen_range(-100i32..=100i32)).collect();
    let bz_small: Vec<i32> = (0..n).map(|_| rng.gen_range(-100i32..=100i32)).collect();
    let cz_small: Vec<i32> = az_small
      .iter()
      .zip(&bz_small)
      .map(|(&a, &b)| a * b)
      .collect();

    let mut taus: Vec<F> = (0..NUM_VARS).map(|_| F::random(&mut rng)).collect();
    taus[0] = F::ZERO;

    run_equivalence_test_with_taus::<i32>(taus, az_small, bz_small, cz_small);
  }
}

#[cfg(test)]
mod perf_tests {
  use super::*;
  use crate::{
    big_num::SmallValueEngine, polys::multilinear::MultilinearPolynomial, start_span,
    traits::Engine,
  };
  use ff::Field;
  use rand::{Rng, SeedableRng, rngs::StdRng};
  use tracing::info;
  use tracing_subscriber::EnvFilter;

  // Test sizes: smaller for debug builds, full range for release
  #[cfg(debug_assertions)]
  const TEST_SIZES: &[usize] = &[16, 18];

  #[cfg(not(debug_assertions))]
  const TEST_SIZES: &[usize] = &[16, 18, 20, 22, 24];

  fn test_small_value_sumcheck_with<E: Engine>()
  where
    E::Scalar: SmallValueEngine<i64>,
  {
    const SEED: u64 = 0xDEADBEEF;
    let field_name = std::any::type_name::<E::Scalar>()
      .split("::")
      .last()
      .unwrap_or("unknown");

    for &num_vars in TEST_SIZES {
      let len = 1 << num_vars;
      let mut rng = StdRng::seed_from_u64(SEED);

      // Generate synthetic small-value polynomials
      let az_small: Vec<i64> = (0..len).map(|_| rng.gen_range(-100i64..=100i64)).collect();
      let bz_small: Vec<i64> = (0..len).map(|_| rng.gen_range(-100i64..=100i64)).collect();
      let cz_small: Vec<i64> = az_small
        .iter()
        .zip(&bz_small)
        .map(|(&a, &b)| a * b)
        .collect();

      let az_poly = MultilinearPolynomial::new(az_small);
      let bz_poly = MultilinearPolynomial::new(bz_small);
      let cz_poly = MultilinearPolynomial::new(cz_small);

      let taus: Vec<E::Scalar> = (0..num_vars).map(|_| E::Scalar::random(&mut rng)).collect();
      let mut transcript = E::TE::new(b"perf_test");

      let (_span, t) = start_span!(
        "small_value_sumcheck_prove",
        field = field_name,
        num_vars = num_vars
      );

      let (proof, _r, _evals) = prove_cubic_small_value::<E, _, 3>(
        &E::Scalar::ZERO,
        taus.clone(),
        &az_poly,
        &bz_poly,
        &cz_poly,
        &mut transcript,
      )
      .expect("proof generation should succeed");

      info!(field = field_name, num_vars, n = len, ms = ?t.elapsed().as_millis(), "completed");

      // Verify proof with fresh transcript
      let mut verifier_transcript = E::TE::new(b"perf_test");
      proof
        .verify(E::Scalar::ZERO, num_vars, 3, &mut verifier_transcript)
        .expect("proof verification should succeed");
    }
  }

  #[test]
  fn test_small_value_sumcheck_perf() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();

    use crate::provider::Bn254Engine;

    // Always test with BN254
    test_small_value_sumcheck_with::<Bn254Engine>();

    // Additional engines only in release builds
    #[cfg(not(debug_assertions))]
    {
      use crate::provider::{PallasHyraxEngine, T256HyraxEngine};
      test_small_value_sumcheck_with::<PallasHyraxEngine>();
      test_small_value_sumcheck_with::<T256HyraxEngine>();
    }
  }
}
