// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value sumcheck implementation for the first ℓ₀ rounds.
//!
//! This module provides optimized sumcheck proving using native integer
//! arithmetic for polynomials with small coefficients. The key optimization
//! is replacing expensive field multiplications with native integer operations
//! during the initial rounds when polynomial values are guaranteed to be small.
//!
//! Based on "Speeding Up Sum-Check Proving" by Suyash Bagad, Quang Dao,
//! Yuval Domb, and Justin Thaler. <https://eprint.iacr.org/2025/1117.pdf>
//!
//! # Overview
//!
//! The main entry point is [`prove_cubic_small_value`], which implements
//! Algorithm 6 (EqPoly-SmallValueSC) combining:
//! - Small-value optimization (Algorithm 4) for the first ℓ₀ rounds
//! - Eq-poly optimization (Algorithm 5) for the remaining rounds
//!
//! # Key Components
//!
//! - [`SmallValueSumCheck`]: Tracks state during small-value rounds
//! - [`build_univariate_round_polynomial`]: Constructs the cubic round polynomial

use crate::{
  errors::SpartanError,
  lagrange_accumulator::{
    EqRoundFactor, LagrangeAccumulators, LagrangeBasisFactory, LagrangeCoeff, LagrangeEvals,
    LagrangeHatEvals, SPARTAN_T_DEGREE, build_accumulators_spartan, derive_t1,
  },
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial, univariate::UniPoly},
  small_field::{DelayedReduction, SmallValueField, WideMul},
  start_span,
  sumcheck::{SumcheckProof, eq_sumcheck},
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use ff::PrimeField;
use num_traits::Zero;
use rayon::prelude::*;
use tracing::info;

use crate::sumcheck::PAR_THRESHOLD;

/// Tracks the small-value sum-check state for the first ℓ₀ rounds.
///
/// This struct maintains the precomputed accumulators and running state
/// needed to efficiently evaluate round polynomials using native integer
/// arithmetic instead of field operations.
pub struct SmallValueSumCheck<Scalar: PrimeField, const D: usize> {
  accumulators: LagrangeAccumulators<Scalar, D>,
  coeff: LagrangeCoeff<Scalar, D>,
  eq_factor: EqRoundFactor<Scalar>,
  basis_factory: LagrangeBasisFactory<Scalar, D>,
}

impl<Scalar: PrimeField, const D: usize> SmallValueSumCheck<Scalar, D> {
  /// Create a new small-value round tracker with precomputed accumulators.
  pub fn new(
    accumulators: LagrangeAccumulators<Scalar, D>,
    basis_factory: LagrangeBasisFactory<Scalar, D>,
  ) -> Self {
    Self {
      accumulators,
      coeff: LagrangeCoeff::new(),
      eq_factor: EqRoundFactor::new(),
      basis_factory,
    }
  }

  /// Create from accumulators with the standard Lagrange basis (0, 1, 2, ...).
  pub fn from_accumulators(accumulators: LagrangeAccumulators<Scalar, D>) -> Self {
    let basis_factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));
    Self::new(accumulators, basis_factory)
  }

  /// Evaluate t_i(u) for all u ∈ Û_D in a single pass for round i.
  pub fn eval_t_all_u(&self, round: usize) -> LagrangeHatEvals<Scalar, D> {
    self.accumulators.round(round).eval_t_all_u(&self.coeff)
  }

  /// Compute ℓ_i values for the provided w_i.
  pub fn eq_round_values(&self, w_i: Scalar) -> LagrangeEvals<Scalar, 2> {
    self.eq_factor.values(w_i)
  }

  /// Advance the round state with the verifier challenge r_i.
  pub fn advance(&mut self, li: &LagrangeEvals<Scalar, 2>, r_i: Scalar) {
    self.eq_factor.advance(li, r_i);
    self.coeff.extend(&self.basis_factory.basis_at(r_i));
  }
}

/// Build the cubic round polynomial s_i(X) in coefficient form for Spartan.
///
/// Constructs s_i(X) = ℓ_i(X) · t_i(X) where:
/// - ℓ_i(X) is the linear eq factor
/// - t_i(X) is the degree-2 polynomial from accumulators
pub(crate) fn build_univariate_round_polynomial<F: PrimeField>(
  li: &LagrangeEvals<F, 2>,
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

    (
      F::reduce(&acc_a),
      F::reduce(&acc_b),
      F::reduce(&acc_c),
    )
  };

  let results: Vec<(F, F, F)> = if stride >= PAR_THRESHOLD {
    (0..stride).into_par_iter().map(compute).collect()
  } else {
    (0..stride).map(compute).collect()
  };

  let mut out_a = Vec::with_capacity(stride);
  let mut out_b = Vec::with_capacity(stride);
  let mut out_c = Vec::with_capacity(stride);
  for (a, b, c) in results {
    out_a.push(a);
    out_b.push(b);
    out_c.push(c);
  }

  (
    MultilinearPolynomial::new(out_a),
    MultilinearPolynomial::new(out_b),
    MultilinearPolynomial::new(out_c),
  )
}

/// Prove poly_A * poly_B - poly_C using Algorithm 6 (EqPoly-SmallValueSC).
///
/// This function combines small-value optimization (Algorithm 4) for the first ℓ₀ rounds
/// with eq-poly optimization (Algorithm 5) for the remaining rounds.
///
/// Field-element polynomials are created internally via batched eq-weighted binding
/// from the small-value inputs, eliminating the need for pre-allocated field polys.
///
/// Generic over `SmallValue` to support both i32/i64 and i64/i128 configurations.
///
/// # Type Parameters
///
/// - `LB`: Number of small-value rounds (ℓ₀). The actual number of rounds used is
///   `min(LB, num_rounds / 2)`. Caller should ensure input values are bounded by
///   `i64::MAX / 3^LB` for safe Lagrange extension (see `vec_to_small_for_extension`).
///   Typical values are 3-4 for practical instances (3^4 = 81× growth factor).
pub fn prove_cubic_small_value<E, SmallValue, const LB: usize>(
  claim: &E::Scalar,
  taus: Vec<E::Scalar>,
  poly_A_small: &MultilinearPolynomial<SmallValue>,
  poly_B_small: &MultilinearPolynomial<SmallValue>,
  poly_C_small: &MultilinearPolynomial<SmallValue>,
  transcript: &mut E::TE,
) -> Result<(SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError>
where
  E: Engine,
  SmallValue: WideMul + Copy + Default + num_traits::Zero + std::ops::Add<Output = SmallValue> + std::ops::Sub<Output = SmallValue> + Send + Sync,
  E::Scalar: SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<E::Scalar>,
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
    return Err(SpartanError::SmallValueRoundsZero {
      num_rounds,
      lb: LB,
    });
  }

  // ===== Pre-computation Phase =====
  // Build accumulators A_i(v, u) for all i ∈ [ℓ₀] using small-value arithmetic.
  // Internally uses: small × small → intermediate (for Az·Bz products),
  // then intermediate × field (for eq weighting via DelayedReduction).
  let accumulators = build_accumulators_spartan(poly_A_small, poly_B_small, &taus, l0);
  let mut small_value =
    SmallValueSumCheck::<E::Scalar, SPARTAN_T_DEGREE>::from_accumulators(accumulators);

  // ===== Small-Value Rounds (0 to ℓ₀-1) =====
  // During these rounds, we use the precomputed accumulators. Polynomials are NOT bound
  // during these rounds - that will happen in the transition phase.
  #[allow(clippy::needless_range_loop)]
  for round in 0..l0 {
    let (_round_span, round_t) = start_span!("sumcheck_smallvalue_round", round = round);

    // 1. Get t_i evaluations from accumulators
    let t_all = small_value.eval_t_all_u(round);
    let t_inf = t_all.at_infinity();
    let t0 = t_all.at_zero();

    // 2. Get eq factor values ℓ_i(0), ℓ_i(1), ℓ_i(∞)
    let li = small_value.eq_round_values(taus[round]);

    // 3. Derive t(1) from sumcheck constraint: s(0) + s(1) = claim
    let t1 = derive_t1(li.at_zero(), li.at_one(), claim_per_round, t0)
      .ok_or(SpartanError::InvalidSumcheckProof)?;

    // 4. Build round polynomial s_i(X) = ℓ_i(X) · t_i(X)
    let poly = build_univariate_round_polynomial(&li, t0, t1, t_inf);

    // 5. Transcript interaction
    transcript.absorb(b"p", &poly);
    let r_i = transcript.squeeze(b"c")?;
    r.push(r_i);
    polys.push(poly.compress());

    // 6. Update claim
    claim_per_round = poly.evaluate(&r_i);

    // 7. Advance small-value state (updates R_{i+1} and eq_factor)
    small_value.advance(&li, r_i);

    info!(
      elapsed_ms = %round_t.elapsed().as_millis(),
      round = round,
      "sumcheck_smallvalue_round"
    );
  }

  // ===== Transition Phase =====
  // Create bound field-element polynomials via batched eq-weighted small-value accumulation.
  // This replaces l0 sequential bind_three_polys_top calls with a single pass using
  // field×int unreduced accumulation — ~6-9x faster for l0=3.
  let (_bind_span, bind_t) = start_span!("bind_poly_vars_transition");
  let (mut poly_A, mut poly_B, mut poly_C) =
    bind_three_polys_batched_small_value(poly_A_small, poly_B_small, poly_C_small, &r[..l0]);
  info!(
    elapsed_ms = %bind_t.elapsed().as_millis(),
    "bind_poly_vars_transition"
  );

  // Create EqSumCheckInstance with ALL taus and advance it by l0 rounds.
  let mut eq_instance = eq_sumcheck::EqSumCheckInstance::<E>::new(taus.clone());
  for r_i in &r[..l0] {
    eq_instance.bound(r_i);
  }

  // ===== Remaining Rounds (ℓ₀ to ℓ-1) =====
  // Continue using the same eq_instance which is now in the correct state
  for round in l0..num_rounds {
    let (_round_span, round_t) = start_span!("sumcheck_round", round = round);

    let poly = {
      let (_eval_span, eval_t) = start_span!("compute_eval_points");
      let (eval_point_0, eval_point_2, eval_point_3) = eq_instance
        .evaluation_points_cubic_with_three_inputs_delayed(round, &poly_A, &poly_B, &poly_C);
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
    crate::sumcheck::bind_three_polys_top(&mut poly_A, &mut poly_B, &mut poly_C, &r_i);
    eq_instance.bound(&r_i);
    info!(elapsed_ms = %bind_t.elapsed().as_millis(), "bind_poly_vars");
    info!(elapsed_ms = %round_t.elapsed().as_millis(), round = round, "sumcheck_round");
  }

  Ok((
    SumcheckProof::new(polys),
    r,
    vec![poly_A[0], poly_B[0], poly_C[0]],
  ))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    gadgets::CubicChainCircuit,
    polys::multilinear::MultilinearPolynomial,
    sha256_circuits::SmallSha256Circuit,
    provider::PallasHyraxEngine,
    small_field::{DelayedReduction, SmallValueField, WideMul},
    spartan::SpartanSNARK,
    sumcheck::eq_sumcheck::EqSumCheckInstance,
    traits::{Engine, snark::R1CSSNARKTrait, transcript::TranscriptEngineTrait},
  };
  use ff::Field;
  use std::ops::{Add, Mul, Sub};

  type E = PallasHyraxEngine;
  type F = <E as Engine>::Scalar;

  /// Generic helper to test that SmallValueSumCheck produces the same polynomial
  /// evaluations as EqSumCheckInstance across multiple rounds.
  fn run_smallvalue_round_test<V>()
  where
    V: WideMul + Copy + Default + num_traits::Zero + Add<Output = V> + Sub<Output = V> + Mul<Output = V> + Send + Sync + TryFrom<usize>,
    <V as TryFrom<usize>>::Error: std::fmt::Debug,
    F: SmallValueField<V> + DelayedReduction<V> + DelayedReduction<V::Product> + DelayedReduction<F>,
  {
    const NUM_VARS: usize = 6;
    const SMALL_VALUE_ROUNDS: usize = 3;

    let n = 1usize << NUM_VARS;
    let taus = (0..NUM_VARS)
      .map(|i| F::from((i + 2) as u64))
      .collect::<Vec<_>>();

    // Small-value polynomials for build_accumulators_spartan
    let az_small: Vec<V> = (0..n).map(|i| V::try_from(i + 1).unwrap()).collect();
    let bz_small: Vec<V> = (0..n).map(|i| V::try_from(i + 3).unwrap()).collect();
    let cz_small: Vec<V> = az_small
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

    let accs = build_accumulators_spartan(&az_poly, &bz_poly, &taus, SMALL_VALUE_ROUNDS);
    let mut small_value = SmallValueSumCheck::from_accumulators(accs);

    let mut eq_instance = EqSumCheckInstance::<E>::new(taus.clone());
    let mut poly_A = az.clone();
    let mut poly_B = bz.clone();
    let mut poly_C = cz.clone();

    for (round, &tau_round) in taus.iter().enumerate().take(SMALL_VALUE_ROUNDS) {
      // Get expected evaluations from standard method
      let (expected_eval_0, expected_eval_2, expected_eval_3) =
        eq_instance.evaluation_points_cubic_with_three_inputs(round, &poly_A, &poly_B, &poly_C);
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
  /// output to prove_cubic_with_three_inputs using real R1CS polynomials
  /// derived from CubicChainCircuit.
  ///
  /// Uses the generic SmallValueField trait for field-to-small conversion.
  fn run_equivalence_test(num_rounds: usize) {
    // Create a circuit targeting num_rounds sumcheck rounds
    let circuit = CubicChainCircuit::for_rounds(num_rounds);

    // Setup SpartanSNARK
    let (pk, _vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup should succeed");

    // Prep prove (is_small=true to get small values in witness)
    let prep_snark =
      SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).expect("prep_prove should succeed");

    // Extract Az, Bz, Cz, tau from the circuit
    let (az_vals, bz_vals, cz_vals, taus) =
      SpartanSNARK::<E>::extract_outer_sumcheck_inputs(&pk, circuit, &prep_snark)
        .expect("extract_outer_sumcheck_inputs should succeed");

    let num_vars = taus.len();

    // Convert field values to small values using the generic SmallValueField trait.
    // This validates that all values fit in the small value type (i64).
    let az_small: Vec<i64> = az_vals
      .iter()
      .map(|v| {
        F::try_field_to_small(v).expect("Az values should fit in small type for CubicChainCircuit")
      })
      .collect();
    let bz_small: Vec<i64> = bz_vals
      .iter()
      .map(|v| {
        F::try_field_to_small(v).expect("Bz values should fit in small type for CubicChainCircuit")
      })
      .collect();
    // Cz is Az * Bz computed in the small domain
    let cz_small: Vec<i64> = az_small
      .iter()
      .zip(&bz_small)
      .map(|(&a, &b)| a * b)
      .collect();

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
    let (proof2, r2, evals2) = prove_cubic_small_value::<E, _, 3>(
      &claim,
      taus,
      &az_small_poly,
      &bz_small_poly,
      &cz_small_poly,
      &mut transcript2,
    )
    .expect("small-value prove should succeed");

    // Verify all outputs match
    assert_eq!(
      r1, r2,
      "challenges must match for num_rounds={} (num_vars={})",
      num_rounds, num_vars
    );
    assert_eq!(
      proof1, proof2,
      "proofs must match for num_rounds={} (num_vars={})",
      num_rounds, num_vars
    );
    assert_eq!(
      evals1, evals2,
      "final evals must match for num_rounds={} (num_vars={})",
      num_rounds, num_vars
    );
  }

  /// Test small-value sumcheck equivalence with real R1CS polynomials from CubicChainCircuit.
  ///
  /// Tests multiple round counts to ensure equivalence holds for:
  /// - Small circuits (rounds < l0=3)
  /// - Medium circuits (rounds == l0)
  /// - Large circuits (rounds > l0)
  #[test]
  fn test_sumcheck_equivalence_with_circuit() {
    // Test various round counts:
    // - 4: small circuit, few rounds after l0
    // - 6: medium circuit
    // - 10: larger circuit with more rounds
    for num_rounds in [4, 6, 10] {
      run_equivalence_test(num_rounds);
    }
  }

  /// Test that prove_cubic_small_value produces identical output to
  /// prove_cubic_with_three_inputs using SmallSha256Circuit.
  ///
  /// Tests both NoBatchEq (use_batching=false) and BatchingEq<21> (use_batching=true) modes.
  fn run_sha256_equivalence_test(preimage_len: usize, use_batching: bool) {
    // 1. Create SmallSha256Circuit
    let circuit = SmallSha256Circuit::<F>::new(vec![0u8; preimage_len], use_batching);

    // 2. Setup and prep_prove
    let (pk, _vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup");
    let prep_snark = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).expect("prep_prove");

    // 3. Extract Az, Bz, Cz, tau
    let (az_vals, bz_vals, cz_vals, taus) =
      SpartanSNARK::<E>::extract_outer_sumcheck_inputs(&pk, circuit, &prep_snark)
        .expect("extract_outer_sumcheck_inputs");

    // 4. Convert to small values (i64)
    let az_small: Vec<i64> = az_vals
      .iter()
      .map(|v| F::try_field_to_small(v).expect("Az should fit in i64"))
      .collect();
    let bz_small: Vec<i64> = bz_vals
      .iter()
      .map(|v| F::try_field_to_small(v).expect("Bz should fit in i64"))
      .collect();
    let cz_small: Vec<i64> = az_small
      .iter()
      .zip(&bz_small)
      .map(|(&a, &b)| a * b)
      .collect();

    // 5. Run both methods with fresh transcripts
    let claim = F::ZERO;
    let mut transcript1 = <E as Engine>::TE::new(b"test_sha256");
    let mut transcript2 = <E as Engine>::TE::new(b"test_sha256");

    let mut az1 = MultilinearPolynomial::new(az_vals);
    let mut bz1 = MultilinearPolynomial::new(bz_vals);
    let mut cz1 = MultilinearPolynomial::new(cz_vals);

    let (proof1, r1, evals1) = SumcheckProof::<E>::prove_cubic_with_three_inputs(
      &claim,
      taus.clone(),
      &mut az1,
      &mut bz1,
      &mut cz1,
      &mut transcript1,
    )
    .expect("standard prove");

    let az_small_poly = MultilinearPolynomial::new(az_small);
    let bz_small_poly = MultilinearPolynomial::new(bz_small);
    let cz_small_poly = MultilinearPolynomial::new(cz_small);

    let (proof2, r2, evals2) = prove_cubic_small_value::<E, _, 3>(
      &claim,
      taus.clone(),
      &az_small_poly,
      &bz_small_poly,
      &cz_small_poly,
      &mut transcript2,
    )
    .expect("small-value prove");

    // 6. Assert equivalence
    assert_eq!(r1, r2, "challenges must match");
    assert_eq!(proof1, proof2, "proofs must match");
    assert_eq!(evals1, evals2, "final evals must match");

    // 7. Verify the proof
    let num_vars = taus.len();
    let mut transcript_v = <E as Engine>::TE::new(b"test_sha256");
    let (final_claim, r_v) = proof1
      .verify(claim, num_vars, 3, &mut transcript_v)
      .expect("verification");
    assert_eq!(r_v, r1, "verify challenges must match prover");
    let tau_eval = EqPolynomial::new(taus).evaluate(&r_v);
    let expected = tau_eval * (evals1[0] * evals1[1] - evals1[2]);
    assert_eq!(final_claim, expected, "final claim mismatch");
  }

  /// Test small-value sumcheck equivalence with SmallSha256Circuit.
  /// Tests both NoBatchEq and BatchingEq<21> modes with a 64-byte message.
  #[test]
  fn test_sumcheck_equivalence_with_sha256_circuit() {
    // Use 64 bytes (smaller than example's 1024 for faster tests)
    let preimage_len = 64;

    // Test NoBatchEq (i32 path)
    run_sha256_equivalence_test(preimage_len, false);

    // Test BatchingEq<21> (i64 path)
    run_sha256_equivalence_test(preimage_len, true);
  }
}
