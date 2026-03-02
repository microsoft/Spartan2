// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module implements NeutronNova's folding scheme for folding together a batch of R1CS instances
//! This implementation focuses on a non-recursive version of NeutronNova and targets the case where the batch size is moderately large.
//! Since we are in the non-recursive setting, we simply fold a batch of instances into one (all at once, via multi-folding)
//! and then use Spartan to prove that folded instance.
//! The proof system implemented here provides zero-knowledge via Nova's folding scheme.
use std::ops::{Add, Sub};

use ff::PrimeField;

use crate::{
  CommitmentKey,
  bellpepper::{
    r1cs::{
      MultiRoundSpartanShape, MultiRoundSpartanWitness, PrecommittedState, RerandomizationTrait,
      SpartanShape, SpartanWitness,
    },
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  digest::{DigestComputer, SimpleDigestible},
  errors::SpartanError,
  lagrange_accumulator::{build_accumulators_neutronnova, derive_t1},
  math::Math,
  nifs::NovaNIFS,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
    power::PowPolynomial,
    univariate::UniPoly,
  },
  r1cs::{
    R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
    SplitMultiRoundR1CSInstance, SplitMultiRoundR1CSShape, SplitR1CSInstance, SplitR1CSShape,
  },
  small_field::{DelayedReduction, SmallValueField, WideMul, vec_to_small_for_extension},
  small_sumcheck::{SmallValueSumCheck, build_univariate_round_polynomial},
  start_span,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::SpartanCircuit,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    snark::{DigestHelperTrait, SpartanDigest},
    transcript::TranscriptEngineTrait,
  },
  zk::NeutronNovaVerifierCircuit,
};
use ff::Field;
use num_traits::Zero;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

fn compute_tensor_decomp(n: usize) -> (usize, usize, usize) {
  let ell = n.next_power_of_two().log_2();
  // we split ell into ell1 and ell2 such that ell1 + ell2 = ell and ell1 >= ell2
  let ell1 = ell.div_ceil(2); // This ensures ell1 >= ell2
  let ell2 = ell / 2;
  let left = 1 << ell1;
  let right = 1 << ell2;

  (ell, left, right)
}

/// A type that holds the NeutronNova NIFS (Non-Interactive Folding Scheme)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaNIFS<E: Engine> {
  polys: Vec<UniPoly<E::Scalar>>,
}

#[inline]
#[allow(clippy::needless_range_loop)]
fn suffix_weight_full<F: Field>(t: usize, ell_b: usize, pair_idx: usize, rhos: &[F]) -> F {
  let mut w = F::ONE;
  let mut k = pair_idx;
  for s in (t + 1)..ell_b {
    let bit = (k & 1) as u8; // LSB-first
    w *= if bit == 0 { F::ONE - rhos[s] } else { rhos[s] };
    k >>= 1;
  }
  w
}

#[inline]
fn mul_opt<F: Field>(a: &F, b: &F) -> F {
  if a == &F::ZERO || b == &F::ZERO {
    F::ZERO
  } else if a == &F::ONE {
    *b
  } else if b == &F::ONE {
    *a
  } else {
    *a * *b
  }
}

/// Build witness vector z = [W | 1 | X] for matrix-vector multiplication (field elements).
#[inline]
fn build_z<E: Engine>(w: &[E::Scalar], x: &[E::Scalar]) -> Vec<E::Scalar> {
  let mut z = Vec::with_capacity(w.len() + 1 + x.len());
  z.extend_from_slice(w);
  z.push(E::Scalar::ONE);
  z.extend_from_slice(x);
  z
}

/// Build witness vector z = [W | 1 | X] for matrix-vector multiplication (small values).
#[inline]
fn build_z_small<SV: Copy + num_traits::One>(w: &[SV], x: &[SV]) -> Vec<SV> {
  let mut z = Vec::with_capacity(w.len() + 1 + x.len());
  z.extend_from_slice(w);
  z.push(SV::one());
  z.extend_from_slice(x);
  z
}

/// Shared NIFS setup: padding, transcript absorption, tau/rhos squeezing.
fn prepare_nifs_inputs<E: Engine>(
  Us: &[R1CSInstance<E>],
  Ws: &[R1CSWitness<E>],
  transcript: &mut E::TE,
) -> Result<
  (
    Vec<R1CSInstance<E>>, // Padded Us
    Vec<R1CSWitness<E>>,  // Padded Ws
    usize,                // ell_b
    E::Scalar,            // tau
    Vec<E::Scalar>,       // rhos
  ),
  SpartanError,
>
where
  E::PCS: FoldingEngineTrait<E>,
{
  let n = Us.len();
  let n_padded = n.next_power_of_two();
  let ell_b = n_padded.log_2();

  info!(
    "NeutronNova NIFS prove for {} instances padded to {}",
    n, n_padded
  );

  let mut Us = Us.to_vec();
  let mut Ws = Ws.to_vec();
  if Us.len() < n_padded {
    Us.extend(vec![Us[0].clone(); n_padded - n]);
    Ws.extend(vec![Ws[0].clone(); n_padded - n]);
  }

  for U in Us.iter() {
    transcript.absorb(b"U", U);
  }
  transcript.absorb(b"T", &E::Scalar::ZERO);

  let tau = transcript.squeeze(b"tau")?;
  let rhos: Vec<_> = (0..ell_b)
    .map(|_| transcript.squeeze(b"rho"))
    .collect::<Result<_, _>>()?;

  Ok((Us, Ws, ell_b, tau, rhos))
}

/// Fold witnesses and instances, update VC with T_out and eq values.
fn fold_and_update_vc<E: Engine>(
  r_bs: &[E::Scalar],
  T_cur: E::Scalar,
  acc_eq: E::Scalar,
  Us: &[R1CSInstance<E>],
  Ws: &[R1CSWitness<E>],
  ell_b: usize,
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
) -> Result<(R1CSWitness<E>, R1CSInstance<E>), SpartanError>
where
  E::PCS: FoldingEngineTrait<E>,
{
  // T_out = poly_last(r_last) / eq(r_b, rho)
  let T_out = T_cur
    * acc_eq
      .invert()
      .into_option()
      .ok_or(SpartanError::ProofVerifyError {
        reason: "acc_eq is zero".to_string(),
      })?;
  vc.t_out_step = T_out;
  vc.eq_rho_at_rb = acc_eq;
  SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, ell_b, transcript)?;

  let (_fold_span, fold_t) = start_span!("fold_witnesses");
  let folded_W = R1CSWitness::fold_multiple(r_bs, Ws)?;
  info!(elapsed_ms = %fold_t.elapsed().as_millis(), "fold_witnesses");

  let (_fold_span, fold_t) = start_span!("fold_instances");
  let folded_U = R1CSInstance::fold_multiple(r_bs, Us)?;
  info!(elapsed_ms = %fold_t.elapsed().as_millis(), "fold_instances");

  Ok((folded_W, folded_U))
}

/// Fold multiple small-value vectors using delayed reduction.
/// Result[j] = Σ_i weights[i] * vectors[i][j]
fn fold_small_value_vectors<F, SV>(weights: &[F], vectors: &[Vec<SV>]) -> Vec<F>
where
  F: PrimeField + DelayedReduction<SV>,
  SV: Send + Sync,
{
  let dim = vectors[0].len();
  (0..dim)
    .into_par_iter()
    .map(|j| {
      let mut acc = <F as DelayedReduction<SV>>::Accumulator::zero();
      for (i, wi) in weights.iter().enumerate() {
        <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc, wi, &vectors[i][j]);
      }
      <F as DelayedReduction<SV>>::reduce(&acc)
    })
    .collect()
}

/// Fold witnesses and instances using small-value optimization, update VC.
#[allow(dead_code)]
fn fold_and_update_vc_small<E, SV>(
  r_bs: &[E::Scalar],
  T_cur: E::Scalar,
  acc_eq: E::Scalar,
  ws_small: &[Vec<SV>],
  xs_small: &[Vec<SV>],
  Us: &[R1CSInstance<E>],
  Ws: &[R1CSWitness<E>],
  ell_b: usize,
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
) -> Result<(R1CSWitness<E>, R1CSInstance<E>), SpartanError>
where
  E: Engine,
  E::Scalar: DelayedReduction<SV>,
  E::PCS: FoldingEngineTrait<E>,
  SV: Send + Sync,
{
  use crate::r1cs::weights_from_r;

  // T_out = poly_last(r_last) / eq(r_b, rho)
  let T_out = T_cur
    * acc_eq
      .invert()
      .into_option()
      .ok_or(SpartanError::ProofVerifyError {
        reason: "acc_eq is zero".to_string(),
      })?;
  vc.t_out_step = T_out;
  vc.eq_rho_at_rb = acc_eq;
  SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, ell_b, transcript)?;

  // Compute weights once for all folding operations
  let w = weights_from_r::<E::Scalar>(r_bs, Us.len());

  // Fold witnesses using small-value optimization
  let (_fold_span, fold_t) = start_span!("fold_witnesses");
  let r_Ws: Vec<_> = Ws.iter().map(|w| w.r_W.clone()).collect();
  let folded_W_vec = fold_small_value_vectors(&w, ws_small);
  let folded_r_W = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(&r_Ws, &w)?;
  let folded_W = R1CSWitness {
    W: folded_W_vec,
    r_W: folded_r_W,
    is_small: false, // After folding with random challenges, no longer small
  };
  info!(elapsed_ms = %fold_t.elapsed().as_millis(), "fold_witnesses");

  // Fold instances using small-value optimization for X, group ops for commitments
  let (_fold_span, fold_t) = start_span!("fold_instances");
  let folded_X = fold_small_value_vectors(&w, xs_small);
  let comm_W_acc = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
    &Us.iter().map(|u| u.comm_W.clone()).collect::<Vec<_>>(),
    &w,
  )?;
  let folded_U = R1CSInstance {
    X: folded_X,
    comm_W: comm_W_acc,
  };
  info!(elapsed_ms = %fold_t.elapsed().as_millis(), "fold_instances");

  Ok((folded_W, folded_U))
}

/// Eq-weighted fold with field×small delayed modular reduction.
///
/// Computes `folded[k] = Σ_i eq_evals[i] * layers[i][k]` for each coordinate k,
/// using accumulate_field_small_prod to delay modular reduction until the end.
///
/// Generic over `SV` to support different small value types (i32, i64).
fn small_value_eq_weighted_fold<E, SV>(
  eq_evals: &[E::Scalar],
  layers: &[Vec<SV>],
  num_cons: usize,
) -> Vec<E::Scalar>
where
  E: Engine,
  E::Scalar: DelayedReduction<SV>,
  SV: Copy + Send + Sync,
{
  let n_inst = eq_evals.len();
  (0..num_cons)
    .into_par_iter()
    .map(|k| {
      let mut acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
      for i in 0..n_inst {
        // Single-value accumulation (field × small with delayed reduction)
        <E::Scalar as DelayedReduction<SV>>::unreduced_multiply_accumulate(
          &mut acc,
          &eq_evals[i],
          &layers[i][k],
        );
      }
      <E::Scalar as DelayedReduction<SV>>::reduce(&acc)
    })
    .collect()
}

/// Fold pairs of field element vectors using a single round's challenge with DMR.
/// Folds m vectors → m/2 vectors: result[i] = (1-r) × vectors[2i] + r × vectors[2i+1]
fn fold_field_vectors_round_dmr<F>(vectors: &[Vec<F>], r: F) -> Vec<Vec<F>>
where
  F: PrimeField + DelayedReduction<F>,
{
  let pairs = vectors.len() / 2;
  let one_minus_r = F::ONE - r;

  (0..pairs)
    .into_par_iter()
    .map(|i| {
      let lo = &vectors[2 * i];
      let hi = &vectors[2 * i + 1];
      let dim = lo.len();

      (0..dim)
        .map(|j| {
          let mut acc = <F as DelayedReduction<F>>::Accumulator::zero();
          <F as DelayedReduction<F>>::unreduced_multiply_accumulate(&mut acc, &one_minus_r, &lo[j]);
          <F as DelayedReduction<F>>::unreduced_multiply_accumulate(&mut acc, &r, &hi[j]);
          <F as DelayedReduction<F>>::reduce(&acc)
        })
        .collect()
    })
    .collect()
}

/// Fold pairs of witnesses using a single round's challenge with DMR.
/// Folds m witnesses → m/2 witnesses.
fn fold_witnesses_round_dmr<E>(Ws: &[R1CSWitness<E>], r: E::Scalar) -> Vec<R1CSWitness<E>>
where
  E: Engine,
  E::Scalar: DelayedReduction<E::Scalar>,
  E::PCS: FoldingEngineTrait<E>,
{
  let pairs = Ws.len() / 2;
  let one_minus_r = E::Scalar::ONE - r;

  (0..pairs)
    .into_par_iter()
    .map(|i| {
      let W_lo = &Ws[2 * i];
      let W_hi = &Ws[2 * i + 1];
      let dim = W_lo.W.len();

      // Fold witness vectors with DMR
      let W_folded: Vec<E::Scalar> = (0..dim)
        .map(|j| {
          let mut acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
          <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
            &mut acc,
            &one_minus_r,
            &W_lo.W[j],
          );
          <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
            &mut acc,
            &r,
            &W_hi.W[j],
          );
          <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc)
        })
        .collect();

      // Fold blinds (group operation, no DMR)
      let r_W_folded = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
        &[W_lo.r_W.clone(), W_hi.r_W.clone()],
        &[one_minus_r, r],
      )
      .expect("fold_blinds");

      R1CSWitness {
        W: W_folded,
        r_W: r_W_folded,
        is_small: false,
      }
    })
    .collect()
}

impl<E: Engine> NeutronNovaNIFS<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Computes the evaluations of the sum-check polynomial at 0, 2, and 3
  #[inline]
  fn prove_helper(
    round: usize,
    (left, right): (usize, usize),
    e: &[E::Scalar],
    Az1: &[E::Scalar],
    Bz1: &[E::Scalar],
    Cz1: &[E::Scalar],
    Az2: &[E::Scalar],
    Bz2: &[E::Scalar],
    Cz2: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar) {
    // sanity check sizes
    assert_eq!(e.len(), left + right);
    assert_eq!(Az1.len(), left * right);
    assert_eq!(Bz1.len(), left * right);
    assert_eq!(Cz1.len(), left * right);
    assert_eq!(Az2.len(), left * right);
    assert_eq!(Bz2.len(), left * right);
    assert_eq!(Cz2.len(), left * right);

    let comb_func = |c1: &E::Scalar, c2: &E::Scalar, c3: &E::Scalar, c4: &E::Scalar| -> E::Scalar {
      *c1 * (*c2 * *c3 - *c4)
    };

    let (eval_at_0, quad_coeff) = (0..right)
      .into_par_iter()
      .map(|i| {
        let (mut i_eval_at_0, mut i_quad_coeff) = (0..left)
          .into_par_iter()
          .map(|j| {
            // Turn the two dimensional (i, j) into a single dimension index
            let k = i * left + j;
            let poly_e_bound_point = e[j];

            // eval 0: bound_func is A(low)
            // Optimization: In round 0, the target value T_cur = 0. The sumcheck polynomial
            // constructed from eval_point_0 and quad_coeff must satisfy T_cur when evaluated
            // at rho_t. Since T_cur = 0 in the first round, we can skip computing eval_point_0
            // (which would be e[j] * (Az1[k] * Bz1[k] - Cz1[k])) and use ZERO directly without
            // affecting the correctness of the folding protocol.
            let eval_point_0 = if round == 0 {
              E::Scalar::ZERO
            } else {
              comb_func(&poly_e_bound_point, &Az1[k], &Bz1[k], &Cz1[k])
            };

            // quad coeff
            let poly_Az_bound_point = Az2[k] - Az1[k];
            let poly_Bz_bound_point = Bz2[k] - Bz1[k];
            let quad_coeff = mul_opt(
              &mul_opt(&poly_Az_bound_point, &poly_Bz_bound_point),
              &poly_e_bound_point,
            );

            (eval_point_0, quad_coeff)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1),
          );

        let f = &e[left..];

        let poly_f_bound_point = f[i];

        // eval 0: bound_func is A(low)
        i_eval_at_0 *= poly_f_bound_point;

        // quad coeff
        i_quad_coeff *= poly_f_bound_point;

        (i_eval_at_0, i_quad_coeff)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1),
      );

    (eval_at_0, quad_coeff)
  }

  /// Computes the evaluations of the sum-check polynomial at 0 and the quadratic coefficient.
  /// Uses two-level delayed modular reduction (inner + middle levels).
  ///
  /// This is the DMR-optimized version of `prove_helper` that reduces Montgomery reductions
  /// from O(left × right) to O(right).
  #[inline]
  fn prove_helper_dmr(
    round: usize,
    (left, right): (usize, usize),
    e: &[E::Scalar],
    Az1: &[E::Scalar],
    Bz1: &[E::Scalar],
    Cz1: &[E::Scalar],
    Az2: &[E::Scalar],
    Bz2: &[E::Scalar],
    _Cz2: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar)
  where
    E::Scalar: DelayedReduction<E::Scalar>,
  {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    // sanity check sizes
    assert_eq!(e.len(), left + right);
    assert_eq!(Az1.len(), left * right);

    let f = &e[left..];

    // Middle level: accumulate over i (0..right)
    let (acc_e0, acc_quad) = (0..right)
      .into_par_iter()
      .fold(
        || (Acc::<E::Scalar>::zero(), Acc::<E::Scalar>::zero()),
        |mut middle_acc, i| {
          // Inner level: accumulate over j (0..left) - NO reductions in this loop!
          let mut inner_e0 = Acc::<E::Scalar>::zero();
          let mut inner_quad = Acc::<E::Scalar>::zero();

          for (j, e_j) in e[..left].iter().enumerate() {
            let k = i * left + j;

            // eval_at_0: skip in round 0 since T_cur = 0
            if round != 0 {
              let inner_val = Az1[k] * Bz1[k] - Cz1[k];
              <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
                &mut inner_e0,
                e_j,
                &inner_val,
              );
            }

            // quad_coeff = (Az2 - Az1) × (Bz2 - Bz1)
            let az_diff = Az2[k] - Az1[k];
            let bz_diff = Bz2[k] - Bz1[k];
            let quad_val = mul_opt(&az_diff, &bz_diff);
            <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
              &mut inner_quad,
              e_j,
              &quad_val,
            );
          }

          // ONE reduction per i (end of inner loop)
          let inner_e0_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_e0);
          let inner_quad_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_quad);

          // Accumulate f[i] × inner_result into middle (no reduction yet)
          let f_i = &f[i];
          <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
            &mut middle_acc.0,
            f_i,
            &inner_e0_red,
          );
          <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
            &mut middle_acc.1,
            f_i,
            &inner_quad_red,
          );

          middle_acc
        },
      )
      .reduce(
        || (Acc::<E::Scalar>::zero(), Acc::<E::Scalar>::zero()),
        |mut a, b| {
          a.0 += b.0;
          a.1 += b.1;
          a
        },
      );

    // ONE final reduction each
    (
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_e0),
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad),
    )
  }

  /// Small-value NIFS sumcheck using Lagrange accumulators.
  ///
  /// Builds accumulators from small-value Az/Bz/Cz layers and runs all ℓ_b sumcheck rounds.
  /// All rounds are Lagrange rounds — no transition phase, no remaining rounds.
  ///
  /// # Arguments
  /// * `a_layers` - Az evaluations per instance, each of length `left * right`
  /// * `b_layers` - Bz evaluations per instance
  /// * `e_eq` - Pre-computed power polynomial split evals: `[e_left | e_right]`
  /// * `left` - Size of left tensor component (from `compute_tensor_decomp`)
  /// * `right` - Size of right tensor component (from `compute_tensor_decomp`)
  /// * `rhos` - Instance-folding challenges
  ///
  /// Returns (polys, r_bs, T_cur, acc_eq).
  pub fn prove_neutronnova_small_value_sumcheck<SmallValue>(
    a_layers: &[Vec<SmallValue>],
    b_layers: &[Vec<SmallValue>],
    e_eq: &[E::Scalar],
    left: usize,
    right: usize,
    rhos: &[E::Scalar],
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<
    (
      Vec<UniPoly<E::Scalar>>,
      Vec<E::Scalar>,
      E::Scalar,
      E::Scalar,
    ),
    SpartanError,
  >
  where
    E::Scalar: SmallValueField<SmallValue>
      + DelayedReduction<SmallValue>
      + DelayedReduction<SmallValue::Product>
      + DelayedReduction<E::Scalar>,
    SmallValue: WideMul
      + Copy
      + Default
      + Zero
      + Add<Output = SmallValue>
      + Sub<Output = SmallValue>
      + Send
      + Sync,
  {
    let ell_b = rhos.len();

    // 1. Build accumulators (takes pre-computed E_eq, not tau)
    let (_acc_span, acc_t) = start_span!("build_accumulators_neutronnova");
    let accumulators = build_accumulators_neutronnova(a_layers, b_layers, e_eq, left, right, rhos, ell_b);
    info!(
      elapsed_ms = %acc_t.elapsed().as_millis(),
      "build_accumulators_neutronnova"
    );

    // 2. Create SmallValueSumCheck state
    let mut small_value = SmallValueSumCheck::<E::Scalar, 2>::from_accumulators(accumulators);

    let mut polys: Vec<UniPoly<E::Scalar>> = Vec::with_capacity(ell_b);
    let mut r_bs: Vec<E::Scalar> = Vec::with_capacity(ell_b);
    let mut T_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;
    // 3. Run ℓ_b sumcheck rounds
    for (i, rho_i) in rhos.iter().enumerate() {
      let (_round_span, round_t) = start_span!("nifs_smallvalue_round", round = i);

      // Evaluate at Û₂
      let t_all = small_value.eval_t_all_u(i);
      let t0 = t_all.at_zero();
      let t_inf = t_all.at_infinity();

      // Eq factor values
      let li = small_value.eq_round_values(*rho_i);

      // Recover t(1) from sumcheck identity
      let t1 = derive_t1(li.at_zero(), li.at_one(), T_cur, t0)
        .ok_or(SpartanError::InvalidSumcheckProof)?;

      // Build cubic round polynomial
      let poly = build_univariate_round_polynomial(&li, t0, t1, t_inf);

      // Feed into verifier circuit
      let c = &poly.coeffs;
      vc.nifs_polys[i] = [c[0], c[1], c[2], c[3]];

      // Transcript interaction (vc_commit)
      let (_vc_span, vc_t) = start_span!("vc_commit");
      let chals =
        SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, i, transcript)?;
      info!(elapsed_ms = %vc_t.elapsed().as_millis(), "vc_commit");
      let r_i = chals[0];

      // Update state
      T_cur = poly.evaluate(&r_i);
      acc_eq *= (E::Scalar::ONE - r_i) * (E::Scalar::ONE - *rho_i) + r_i * *rho_i;

      r_bs.push(r_i);
      polys.push(poly);

      // Advance accumulator state
      small_value.advance(&li, r_i);

      info!(
        elapsed_ms = %round_t.elapsed().as_millis(),
        round = i,
        "nifs_smallvalue_round"
      );
    }
    Ok((polys, r_bs, T_cur, acc_eq))
  }

  /// Small-value optimized NIFS prove.
  ///
  /// Uses Lagrange accumulators and delayed modular reduction for better performance
  /// when witness values fit in a small value type (i32 or i64).
  ///
  /// Generic over `SV` to support different small value types.
  ///
  /// Returns (E_eq, Az folded, Bz folded, Cz folded, folded witness, folded instance).
  #[allow(dead_code)]
  fn prove_small_value<SV>(
    S: &SplitR1CSShape<E>,
    Us: &[R1CSInstance<E>],
    Ws: &[R1CSWitness<E>],
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<
    (
      Vec<E::Scalar>,  // E_eq (split evals, length left+right)
      Vec<E::Scalar>,  // Az layer 0
      Vec<E::Scalar>,  // Bz layer 0
      Vec<E::Scalar>,  // Cz layer 0
      R1CSWitness<E>,  // final folded witness
      R1CSInstance<E>, // final folded instance
    ),
    SpartanError,
  >
  where
    SV: WideMul
      + Copy
      + Default
      + num_traits::Zero
      + num_traits::One
      + std::ops::Add<Output = SV>
      + std::ops::Sub<Output = SV>
      + num_traits::Bounded
      + Into<SV::Product>
      + Send
      + Sync,
    SV::Product: Copy
      + Ord
      + num_traits::Signed
      + std::ops::Div<Output = SV::Product>
      + std::ops::Mul<Output = SV::Product>
      + num_traits::One
      + From<i32>,
    E::Scalar: SmallValueField<SV>
      + DelayedReduction<SV>
      + DelayedReduction<SV::Product>
      + DelayedReduction<E::Scalar>,
  {
    let (_nifs_total_span, _nifs_total_t) = start_span!("nifs_prove");

    // === SHARED SETUP ===
    let (Us, Ws, ell_b, tau, rhos) = prepare_nifs_inputs::<E>(Us, Ws, transcript)?;
    let n_padded = Us.len();

    // === EXTRACT SMALL VECTORS EARLY ===
    // Convert all W and X to small values with Lagrange extension bound check
    let (_convert_span, convert_t) = start_span!("convert_to_small", instances = n_padded);
    let ws_small: Vec<Vec<SV>> = Ws
      .par_iter()
      .map(|w| vec_to_small_for_extension::<E::Scalar, SV, 2>(&w.W, ell_b))
      .collect::<Result<_, _>>()?;
    let xs_small: Vec<Vec<SV>> = Us
      .par_iter()
      .map(|u| vec_to_small_for_extension::<E::Scalar, SV, 2>(&u.X, ell_b))
      .collect::<Result<_, _>>()?;
    info!(elapsed_ms = %convert_t.elapsed().as_millis(), "convert_to_small");

    // === TENSOR DECOMPOSITION (same split as field path) ===
    // E_eq split must match what callers expect (compute_tensor_decomp's split)
    let (ell_cons, left, right) = compute_tensor_decomp(S.num_cons);
    let E_eq = PowPolynomial::split_evals(tau, ell_cons, left, right);

    // === MATRIX-VECTOR MULTIPLY (small-value optimized) ===
    // Build z_small directly from small W and X, then use field × small multiplication
    let (_matrix_span, matrix_t) =
      start_span!("matrix_vector_multiply_instances", instances = n_padded);

    let smalls: Vec<(Vec<SV>, Vec<SV>, Vec<SV>)> = (0..n_padded)
      .into_par_iter()
      .map(|i| {
        let z_small = build_z_small(&ws_small[i], &xs_small[i]);
        let Az = S.A.multiply_vec_small::<2, SV>(&z_small, ell_b)?;
        let Bz = S.B.multiply_vec_small::<2, SV>(&z_small, ell_b)?;
        let Cz = S.C.multiply_vec_small::<2, SV>(&z_small, ell_b)?;
        Ok((Az, Bz, Cz))
      })
      .collect::<Result<Vec<_>, SpartanError>>()?;

    // Unzip into separate layer vectors
    let (a_small, b_small, c_small): (Vec<Vec<SV>>, Vec<Vec<SV>>, Vec<Vec<SV>>) =
      smalls.into_iter().fold(
        (
          Vec::with_capacity(n_padded),
          Vec::with_capacity(n_padded),
          Vec::with_capacity(n_padded),
        ),
        |(mut a, mut b, mut c), (az, bz, cz)| {
          a.push(az);
          b.push(bz);
          c.push(cz);
          (a, b, c)
        },
      );
    info!(
      elapsed_ms = %matrix_t.elapsed().as_millis(),
      instances = n_padded,
      "matrix_vector_multiply_instances"
    );

    // === SMALL-VALUE SUMCHECK (with pre-computed E_eq) ===
    let (_nifs_rounds_span, nifs_rounds_t) = start_span!("nifs_folding_rounds", rounds = ell_b);
    let (_polys, r_bs, T_cur, acc_eq) = Self::prove_neutronnova_small_value_sumcheck(
      &a_small, &b_small, &E_eq, left, right, &rhos, vc, vc_state, vc_shape, vc_ck, transcript,
    )?;
    info!(
      elapsed_ms = %nifs_rounds_t.elapsed().as_millis(),
      rounds = ell_b,
      "nifs_folding_rounds"
    );

    // === EQ-WEIGHTED FOLD (in small-value land) ===
    let (_fold_span, fold_t) = start_span!("nifs_eq_fold");
    // evals_from_points uses big-endian bit ordering (r_bs[0] = MSB of index),
    // but the vanilla fold uses little-endian (layer index bit 0 = round 0 = r_bs[0]).
    // Reverse r_bs so that evals_from_points produces little-endian ordering.
    let r_bs_rev: Vec<_> = r_bs.iter().rev().cloned().collect();
    let eq_evals = EqPolynomial::evals_from_points(&r_bs_rev);
    let num_cons = a_small[0].len();

    let (az_folded, (bz_folded, cz_folded)) = rayon::join(
      || small_value_eq_weighted_fold::<E, SV>(&eq_evals, &a_small, num_cons),
      || {
        rayon::join(
          || small_value_eq_weighted_fold::<E, SV>(&eq_evals, &b_small, num_cons),
          || small_value_eq_weighted_fold::<E, SV>(&eq_evals, &c_small, num_cons),
        )
      },
    );
    info!(elapsed_ms = %fold_t.elapsed().as_millis(), "nifs_eq_fold");

    // === FOLD WITNESSES/INSTANCES AND UPDATE VC (small-value optimized) ===
    let (folded_W, folded_U) = fold_and_update_vc_small::<E, SV>(
      &r_bs, T_cur, acc_eq, &ws_small, &xs_small, &Us, &Ws, ell_b, vc, vc_state, vc_shape, vc_ck,
      transcript,
    )?;

    info!(
      elapsed_ms = %_nifs_total_t.elapsed().as_millis(),
      "nifs_prove"
    );

    Ok((E_eq, az_folded, bz_folded, cz_folded, folded_W, folded_U))
  }

  /// Field-based NIFS prove (vanilla path).
  ///
  /// Performs NIFS folding using standard field arithmetic.
  ///
  /// Returns (E_eq, Az folded, Bz folded, Cz folded, folded witness, folded instance).
  fn prove_regular(
    S: &SplitR1CSShape<E>,
    Us: &[R1CSInstance<E>],
    Ws: &[R1CSWitness<E>],
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<
    (
      Vec<E::Scalar>,  // E_eq (split evals, length left+right)
      Vec<E::Scalar>,  // Az layer 0
      Vec<E::Scalar>,  // Bz layer 0
      Vec<E::Scalar>,  // Cz layer 0
      R1CSWitness<E>,  // final folded witness
      R1CSInstance<E>, // final folded instance
    ),
    SpartanError,
  > {
    let (_nifs_total_span, _nifs_total_t) = start_span!("nifs_prove");

    // === SHARED SETUP ===
    let (Us, Ws, ell_b, tau, rhos) = prepare_nifs_inputs::<E>(Us, Ws, transcript)?;
    let n_padded = Us.len();

    // === TENSOR DECOMPOSITION ===
    let (ell_cons, left, right) = compute_tensor_decomp(S.num_cons);
    let E_eq = PowPolynomial::split_evals(tau, ell_cons, left, right);

    // === MATRIX-VECTOR MULTIPLY (field elements) ===
    let (_matrix_span, matrix_t) =
      start_span!("matrix_vector_multiply_instances", instances = n_padded);
    let triples: Vec<_> = (0..n_padded)
      .into_par_iter()
      .map(|i| {
        let z = build_z::<E>(&Ws[i].W, &Us[i].X);
        S.multiply_vec(&z)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Unzip into separate layer vectors
    let (mut A_layers, mut B_layers, mut C_layers): (
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
    ) = triples.into_iter().fold(
      (
        Vec::with_capacity(n_padded),
        Vec::with_capacity(n_padded),
        Vec::with_capacity(n_padded),
      ),
      |(mut a, mut b, mut c), (az, bz, cz)| {
        a.push(az);
        b.push(bz);
        c.push(cz);
        (a, b, c)
      },
    );
    info!(
      elapsed_ms = %matrix_t.elapsed().as_millis(),
      instances = n_padded,
      "matrix_vector_multiply_instances"
    );

    // === VANILLA NIFS ROUNDS ===
    let (_nifs_rounds_span, nifs_rounds_t) = start_span!("nifs_folding_rounds", rounds = ell_b);
    let mut polys: Vec<UniPoly<E::Scalar>> = Vec::with_capacity(ell_b);
    let mut r_bs: Vec<E::Scalar> = Vec::with_capacity(ell_b);
    let mut T_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;
    let mut m = n_padded;

    for t in 0..ell_b {
      let rho_t = rhos[t];
      let pairs = m / 2;

      let (e0, quad_coeff) = A_layers
        .par_chunks(2)
        .zip(B_layers.par_chunks(2))
        .zip(C_layers.par_chunks(2))
        .enumerate()
        .map(|(pair_idx, ((pair_a, pair_b), pair_c))| {
          let (e0, quad_coeff) = Self::prove_helper(
            t,
            (left, right),
            &E_eq,
            &pair_a[0],
            &pair_b[0],
            &pair_c[0],
            &pair_a[1],
            &pair_b[1],
            &pair_c[1],
          );
          let w = suffix_weight_full::<E::Scalar>(t, ell_b, pair_idx, &rhos);
          (e0 * w, quad_coeff * w)
        })
        .reduce(
          || (E::Scalar::ZERO, E::Scalar::ZERO),
          |a, b| (a.0 + b.0, a.1 + b.1),
        );

      let one_minus_rho = E::Scalar::ONE - rho_t;
      let two_rho_minus_one = rho_t - one_minus_rho;
      let c = e0 * acc_eq;
      let a = quad_coeff * acc_eq;
      let a_b_c = (T_cur - c * one_minus_rho)
        * rho_t
          .invert()
          .into_option()
          .ok_or(SpartanError::ProofVerifyError {
            reason: "rho_t is zero".to_string(),
          })?;
      let b = a_b_c - a - c;
      let new_a = a * two_rho_minus_one;
      let new_b = b * two_rho_minus_one + a * one_minus_rho;
      let new_c = c * two_rho_minus_one + b * one_minus_rho;
      let new_d = c * one_minus_rho;

      let poly_t = UniPoly {
        coeffs: vec![new_d, new_c, new_b, new_a],
      };
      polys.push(poly_t.clone());

      let c = &poly_t.coeffs;
      vc.nifs_polys[t] = [c[0], c[1], c[2], c[3]];

      // Debug: print polynomial values for regular path
      debug!(
        "Regular round {}: e0={:?}, quad={:?}, T_cur={:?}, poly=[{:?}, {:?}, {:?}, {:?}]",
        t, e0, quad_coeff, T_cur, c[0], c[1], c[2], c[3]
      );

      let (_vc_span, vc_t) = start_span!("vc_commit");
      let chals =
        SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, t, transcript)?;
      info!(elapsed_ms = %vc_t.elapsed().as_millis(), "vc_commit");
      let r_b = chals[0];
      r_bs.push(r_b);

      acc_eq *= (E::Scalar::ONE - r_b) * (E::Scalar::ONE - rho_t) + r_b * rho_t;
      T_cur = poly_t.evaluate(&r_b);

      // Fold A/B/C layers for next round
      let mut next_A = vec![vec![]; m];
      let mut next_B = vec![vec![]; m];
      let mut next_C = vec![vec![]; m];
      for i in 0..m {
        let t = if i & 1 == 0 { i >> 1 } else { (i >> 1) + pairs };
        next_A[t] = std::mem::take(&mut A_layers[i]);
        next_B[t] = std::mem::take(&mut B_layers[i]);
        next_C[t] = std::mem::take(&mut C_layers[i]);
      }
      A_layers = next_A;
      B_layers = next_B;
      C_layers = next_C;

      for matrix_layer in [&mut A_layers, &mut B_layers, &mut C_layers] {
        let (low, high) = matrix_layer.split_at_mut(pairs);
        low.iter_mut().zip(high.iter()).for_each(|(lo, hi)| {
          lo.iter_mut().zip(hi.iter()).for_each(|(l, h)| {
            *l += mul_opt(&(*h - *l), &r_b);
          });
        });
        matrix_layer.truncate(pairs);
      }

      m = pairs;
    }
    info!(
      elapsed_ms = %nifs_rounds_t.elapsed().as_millis(),
      rounds = ell_b,
      "nifs_folding_rounds"
    );

    let (az_folded, bz_folded, cz_folded) = (
      A_layers[0].clone(),
      B_layers[0].clone(),
      C_layers[0].clone(),
    );

    // === FOLD WITNESSES/INSTANCES AND UPDATE VC ===
    let (folded_W, folded_U) = fold_and_update_vc(
      &r_bs, T_cur, acc_eq, &Us, &Ws, ell_b, vc, vc_state, vc_shape, vc_ck, transcript,
    )?;

    info!(elapsed_ms = %_nifs_total_t.elapsed().as_millis(), "nifs_prove");

    Ok((E_eq, az_folded, bz_folded, cz_folded, folded_W, folded_U))
  }

  /// NIFS prove with configurable l0 rounds of small-value sumcheck.
  ///
  /// - `l0 = 0`: Large-value mode (standard field arithmetic, no small-value optimization)
  /// - `l0 > 0`: Decoupled mode:
  ///   - Phase 1: Small-value sumcheck for l0 rounds using Lagrange accumulators
  ///   - Transition: Fold 2^ℓ_b → 2^(ℓ_b-l0) instances using DMR
  ///   - Phase 2: SumFold with three-level DMR for remaining (ℓ_b - l0) rounds
  ///   - Final: One big MSM to fold all commitments
  ///
  /// Returns (E_eq, Az folded, Bz folded, Cz folded, folded witness, folded instance).
  pub fn prove<SmallValue>(
    S: &SplitR1CSShape<E>,
    Us: &[R1CSInstance<E>],
    Ws: &[R1CSWitness<E>],
    l0: usize,
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<
    (
      Vec<E::Scalar>,  // E_eq (split evals, length left+right)
      Vec<E::Scalar>,  // Az layer 0
      Vec<E::Scalar>,  // Bz layer 0
      Vec<E::Scalar>,  // Cz layer 0
      R1CSWitness<E>,  // final folded witness
      R1CSInstance<E>, // final folded instance
    ),
    SpartanError,
  >
  where
    SmallValue: WideMul
      + Copy
      + Default
      + Zero
      + num_traits::One
      + num_traits::Bounded
      + Add<Output = SmallValue>
      + Sub<Output = SmallValue>
      + Into<SmallValue::Product>
      + Send
      + Sync,
    SmallValue::Product: Copy
      + Ord
      + num_traits::Signed
      + std::ops::Div<Output = SmallValue::Product>
      + std::ops::Mul<Output = SmallValue::Product>
      + num_traits::One
      + From<i32>,
    E::Scalar: SmallValueField<SmallValue>
      + DelayedReduction<SmallValue>
      + DelayedReduction<SmallValue::Product>
      + DelayedReduction<E::Scalar>,
  {
    // l0 = 0 means large-value mode (no small-value optimization)
    if l0 == 0 {
      return Self::prove_regular(S, Us, Ws, vc, vc_state, vc_shape, vc_ck, transcript);
    }

    // Compute ell_b early to check if we need to fall back
    // (must be done BEFORE prepare_nifs_inputs to avoid corrupting transcript)
    let ell_b_early = Us.len().next_power_of_two().trailing_zeros() as usize;

    // Validate l0
    if l0 > ell_b_early {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("l0 ({}) must be <= ell_b ({})", l0, ell_b_early),
      });
    }

    // Currently, the small-value accumulator builder only supports l0 == ell_b.
    // For l0 < ell_b, fall back to regular mode (no small-value optimization).
    // TODO: Implement correct l0 < ell_b path (requires computing Az*Bz products before
    // eq-weighted folding, which defeats the small-value optimization).
    if l0 < ell_b_early {
      info!(
        "l0 ({}) < ell_b ({}): falling back to regular mode (no small-value optimization)",
        l0, ell_b_early
      );
      return Self::prove_regular(S, Us, Ws, vc, vc_state, vc_shape, vc_ck, transcript);
    }

    use crate::r1cs::weights_from_r;

    let (_nifs_total_span, _nifs_total_t) = start_span!("nifs_prove");

    // === SHARED SETUP ===
    let (Us, Ws, ell_b, tau, rhos) = prepare_nifs_inputs::<E>(&Us, &Ws, transcript)?;
    let n_padded = Us.len();
    debug_assert_eq!(ell_b, ell_b_early, "ell_b should match early computation");

    info!(
      "Decoupled NIFS: {} total rounds, {} small-value rounds, {} SumFold rounds",
      ell_b,
      l0,
      ell_b - l0
    );

    // === TENSOR DECOMPOSITION ===
    let (ell_cons, left, right) = compute_tensor_decomp(S.num_cons);
    let E_eq = PowPolynomial::split_evals(tau, ell_cons, left, right);

    // === CONVERT TO SMALL VALUES ===
    let (_convert_span, convert_t) = start_span!("convert_to_small");
    let ws_small: Vec<Vec<SmallValue>> = Ws
      .iter()
      .map(|w| vec_to_small_for_extension::<E::Scalar, SmallValue, 2>(&w.W, l0))
      .collect::<Result<Vec<_>, _>>()?;
    let xs_small: Vec<Vec<SmallValue>> = Us
      .iter()
      .map(|u| vec_to_small_for_extension::<E::Scalar, SmallValue, 2>(&u.X, l0))
      .collect::<Result<Vec<_>, _>>()?;
    info!(elapsed_ms = %convert_t.elapsed().as_millis(), "convert_to_small");

    // === MATRIX-VECTOR MULTIPLY (small values) ===
    // NOTE: Use .collect() to preserve index order (fold+reduce doesn't preserve order!)
    let (_matrix_span, matrix_t) = start_span!("matrix_vector_multiply_small", instances = n_padded);
    let triples: Vec<_> = (0..n_padded)
      .into_par_iter()
      .map(|i| {
        let z_small = build_z_small(&ws_small[i], &xs_small[i]);
        let az = S.A.multiply_vec_small::<2, SmallValue>(&z_small, l0).unwrap();
        let bz = S.B.multiply_vec_small::<2, SmallValue>(&z_small, l0).unwrap();
        let cz = S.C.multiply_vec_small::<2, SmallValue>(&z_small, l0).unwrap();
        (az, bz, cz)
      })
      .collect();
    let (a_small, b_small, c_small): (Vec<Vec<SmallValue>>, Vec<Vec<SmallValue>>, Vec<Vec<SmallValue>>) =
      triples.into_iter().fold(
        (
          Vec::with_capacity(n_padded),
          Vec::with_capacity(n_padded),
          Vec::with_capacity(n_padded),
        ),
        |(mut a, mut b, mut c), (az, bz, cz)| {
          a.push(az);
          b.push(bz);
          c.push(cz);
          (a, b, c)
        },
      );
    info!(elapsed_ms = %matrix_t.elapsed().as_millis(), instances = n_padded, "matrix_vector_multiply_small");

    // === PHASE 1: SMALL-VALUE SUMCHECK (l0 rounds) ===
    let (_phase1_span, phase1_t) = start_span!("phase1_small_value", rounds = l0);

    // Build accumulators for l0 rounds
    // Pass full rhos (length ℓ_b) and l0 to enable eq-weighted folding of suffix instances
    let accumulators = build_accumulators_neutronnova(
      &a_small,
      &b_small,
      &E_eq,
      left,
      right,
      &rhos,
      l0,
    );

    // Run l0 rounds of small-value sumcheck
    let mut small_value = SmallValueSumCheck::<E::Scalar, 2>::from_accumulators(accumulators);
    let mut polys: Vec<UniPoly<E::Scalar>> = Vec::with_capacity(ell_b);
    let mut r_bs: Vec<E::Scalar> = Vec::with_capacity(ell_b);
    let mut T_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;

    for t in 0..l0 {
      let rho_t = rhos[t];

      // Evaluate t_i at Û_2
      let t_all = small_value.eval_t_all_u(t);
      let t0 = t_all.at_zero();
      let t_inf = t_all.at_infinity();

      // Compute eq factor ℓ_i(X) for challenge ρ_t
      let li = small_value.eq_round_values(rho_t);

      // Derive t(1) from sumcheck constraint
      let t1 = derive_t1(li.at_zero(), li.at_one(), T_cur, t0)
        .ok_or(SpartanError::InvalidSumcheckProof)?;

      // Build cubic polynomial
      let poly = build_univariate_round_polynomial(&li, t0, t1, t_inf);
      polys.push(poly.clone());

      // Store polynomial coefficients for verifier circuit
      let c = &poly.coeffs;
      vc.nifs_polys[t] = [c[0], c[1], c[2], c[3]];

      // Debug: print polynomial values for Phase 1
      debug!(
        "Phase1 round {}: t0={:?}, t1={:?}, t_inf={:?}, T_cur={:?}, poly=[{:?}, {:?}, {:?}, {:?}]",
        t, t0, t1, t_inf, T_cur, c[0], c[1], c[2], c[3]
      );

      // Get challenge from transcript
      let chals =
        SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, t, transcript)?;
      let r_t = chals[0];
      r_bs.push(r_t);

      // Update running values
      T_cur = poly.evaluate(&r_t);
      acc_eq *= (E::Scalar::ONE - r_t) * (E::Scalar::ONE - rho_t) + r_t * rho_t;

      // Advance accumulator state
      small_value.advance(&li, r_t);
    }
    info!(elapsed_ms = %phase1_t.elapsed().as_millis(), rounds = l0, "phase1_small_value");

    // === TRANSITION: FOLD TO FIELD ELEMENTS ===
    let (_transition_span, transition_t) = start_span!("transition_fold");

    // Compute eq weights for transition: fold 2^ℓ_b → 2^(ℓ_b - l0) instances
    // r_bs[0..l0] are the challenges from Phase 1
    let r_bs_rev: Vec<_> = r_bs[..l0].iter().rev().cloned().collect();
    let eq_evals = EqPolynomial::evals_from_points(&r_bs_rev);
    let num_cons = a_small[0].len();
    let n_intermediate = 1 << (ell_b - l0);

    // Fold Az/Bz/Cz layers from small-value to field elements
    // Each intermediate instance j gets contribution from instances j*2^l0..(j+1)*2^l0
    // NOTE: Use .collect() to preserve index order (fold+reduce doesn't preserve order!)
    let fold_az_bz_cz = |a_small: &[Vec<SmallValue>],
                          b_small: &[Vec<SmallValue>],
                          c_small: &[Vec<SmallValue>]|
     -> (Vec<Vec<E::Scalar>>, Vec<Vec<E::Scalar>>, Vec<Vec<E::Scalar>>) {
      let group_size = 1 << l0;
      let triples: Vec<_> = (0..n_intermediate)
        .into_par_iter()
        .map(|j| {
          let start = j * group_size;
          let group_eq = &eq_evals[..group_size];
          let group_a = &a_small[start..start + group_size];
          let group_b = &b_small[start..start + group_size];
          let group_c = &c_small[start..start + group_size];

          let az_folded = small_value_eq_weighted_fold::<E, SmallValue>(group_eq, group_a, num_cons);
          let bz_folded = small_value_eq_weighted_fold::<E, SmallValue>(group_eq, group_b, num_cons);
          let cz_folded = small_value_eq_weighted_fold::<E, SmallValue>(group_eq, group_c, num_cons);

          (az_folded, bz_folded, cz_folded)
        })
        .collect();
      // Unzip into separate layer vectors (preserves order)
      triples.into_iter().fold(
        (
          Vec::with_capacity(n_intermediate),
          Vec::with_capacity(n_intermediate),
          Vec::with_capacity(n_intermediate),
        ),
        |(mut a, mut b, mut c), (az, bz, cz)| {
          a.push(az);
          b.push(bz);
          c.push(cz);
          (a, b, c)
        },
      )
    };

    let (mut A_layers, mut B_layers, mut C_layers) = fold_az_bz_cz(&a_small, &b_small, &c_small);

    // Fold witnesses from small-value to field elements
    let w_transition = weights_from_r::<E::Scalar>(&r_bs[..l0], n_padded);
    let group_size = 1 << l0;
    let mut Ws_intermediate: Vec<R1CSWitness<E>> = (0..n_intermediate)
      .into_par_iter()
      .map(|j| {
        let start = j * group_size;
        let group_weights = &w_transition[start..start + group_size];
        let group_ws_small = &ws_small[start..start + group_size];

        let W_folded = fold_small_value_vectors(group_weights, group_ws_small);

        // Fold blinds
        let r_Ws: Vec<_> = Ws[start..start + group_size]
          .iter()
          .map(|w| w.r_W.clone())
          .collect();
        let r_W_folded =
          <E::PCS as FoldingEngineTrait<E>>::fold_blinds(&r_Ws, group_weights).unwrap();

        R1CSWitness {
          W: W_folded,
          r_W: r_W_folded,
          is_small: false,
        }
      })
      .collect();

    // Fold instance X vectors from small-value to field elements
    let mut Xs_intermediate: Vec<Vec<E::Scalar>> = (0..n_intermediate)
      .into_par_iter()
      .map(|j| {
        let start = j * group_size;
        let group_weights = &w_transition[start..start + group_size];
        let group_xs_small = &xs_small[start..start + group_size];
        fold_small_value_vectors(group_weights, group_xs_small)
      })
      .collect();

    info!(elapsed_ms = %transition_t.elapsed().as_millis(), "transition_fold");

    // === PHASE 2: SUMFOLD WITH THREE-LEVEL DMR (ℓ_b - l0 rounds) ===
    let (_phase2_span, phase2_t) = start_span!("phase2_sumfold", rounds = ell_b - l0);
    let mut m = n_intermediate;

    for t in l0..ell_b {
      let rho_t = rhos[t];
      let pairs = m / 2;

      // Three-level DMR: outer accumulation over pairs
      type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

      let (acc_e0, acc_quad) = A_layers
        .par_chunks(2)
        .zip(B_layers.par_chunks(2))
        .zip(C_layers.par_chunks(2))
        .enumerate()
        .fold(
          || (Acc::<E::Scalar>::zero(), Acc::<E::Scalar>::zero()),
          |mut outer_acc, (pair_idx, ((pair_a, pair_b), pair_c))| {
            // Two-level DMR prove_helper
            // Pass global round index t (not t-l0) so round 0 optimization only
            // triggers when T_cur is actually 0 (i.e., the very first round overall)
            let (e0, quad_coeff) = Self::prove_helper_dmr(
              t,
              (left, right),
              &E_eq,
              &pair_a[0],
              &pair_b[0],
              &pair_c[0],
              &pair_a[1],
              &pair_b[1],
              &pair_c[1],
            );

            // Suffix weight for outer accumulation
            let w = suffix_weight_full::<E::Scalar>(t, ell_b, pair_idx, &rhos);

            // Outer-level DMR accumulation
            <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
              &mut outer_acc.0,
              &w,
              &e0,
            );
            <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
              &mut outer_acc.1,
              &w,
              &quad_coeff,
            );

            outer_acc
          },
        )
        .reduce(
          || (Acc::<E::Scalar>::zero(), Acc::<E::Scalar>::zero()),
          |mut a, b| {
            a.0 += b.0;
            a.1 += b.1;
            a
          },
        );

      let e0 = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_e0);
      let quad_coeff = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad);

      // Build cubic polynomial
      let one_minus_rho = E::Scalar::ONE - rho_t;
      let two_rho_minus_one = rho_t - one_minus_rho;
      let c = e0 * acc_eq;
      let a = quad_coeff * acc_eq;
      let a_b_c = (T_cur - c * one_minus_rho)
        * rho_t
          .invert()
          .into_option()
          .ok_or(SpartanError::ProofVerifyError {
            reason: "rho_t is zero".to_string(),
          })?;
      let b = a_b_c - a - c;
      let new_a = a * two_rho_minus_one;
      let new_b = b * two_rho_minus_one + a * one_minus_rho;
      let new_c = c * two_rho_minus_one + b * one_minus_rho;
      let new_d = c * one_minus_rho;

      let poly_t = UniPoly {
        coeffs: vec![new_d, new_c, new_b, new_a],
      };
      polys.push(poly_t.clone());

      // Store polynomial coefficients for verifier circuit
      let c = &poly_t.coeffs;
      vc.nifs_polys[t] = [c[0], c[1], c[2], c[3]];

      // Get challenge from transcript
      let chals =
        SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, t, transcript)?;
      let r_t = chals[0];
      r_bs.push(r_t);

      // Update running values
      acc_eq *= (E::Scalar::ONE - r_t) * (E::Scalar::ONE - rho_t) + r_t * rho_t;
      T_cur = poly_t.evaluate(&r_t);

      // Fold Az/Bz/Cz layers for next round using DMR
      A_layers = fold_field_vectors_round_dmr(&A_layers, r_t);
      B_layers = fold_field_vectors_round_dmr(&B_layers, r_t);
      C_layers = fold_field_vectors_round_dmr(&C_layers, r_t);

      // Fold witnesses for next round using DMR
      Ws_intermediate = fold_witnesses_round_dmr::<E>(&Ws_intermediate, r_t);

      // Fold instance X vectors for next round using DMR
      Xs_intermediate = fold_field_vectors_round_dmr(&Xs_intermediate, r_t);

      m = pairs;
    }
    info!(elapsed_ms = %phase2_t.elapsed().as_millis(), rounds = ell_b - l0, "phase2_sumfold");

    // === FINAL: ONE BIG MSM FOR COMMITMENTS ===
    let (_final_span, final_t) = start_span!("final_commitment_fold");

    // Compute final weights from all challenges
    let final_weights = weights_from_r::<E::Scalar>(&r_bs, n_padded);

    // One big MSM for all original commitments
    let all_comms: Vec<_> = Us.iter().map(|u| u.comm_W.clone()).collect();
    let final_comm =
      <E::PCS as FoldingEngineTrait<E>>::fold_commitments(&all_comms, &final_weights)?;

    // T_out = T_cur / acc_eq
    let T_out = T_cur
      * acc_eq
        .invert()
        .into_option()
        .ok_or(SpartanError::ProofVerifyError {
          reason: "acc_eq is zero".to_string(),
        })?;
    vc.t_out_step = T_out;
    vc.eq_rho_at_rb = acc_eq;
    SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, ell_b, transcript)?;

    // Final folded witness and instance
    let folded_W = Ws_intermediate.into_iter().next().unwrap();
    let folded_U = R1CSInstance {
      X: Xs_intermediate.into_iter().next().unwrap(),
      comm_W: final_comm,
    };

    let (az_folded, bz_folded, cz_folded) = (
      A_layers.into_iter().next().unwrap(),
      B_layers.into_iter().next().unwrap(),
      C_layers.into_iter().next().unwrap(),
    );

    info!(elapsed_ms = %final_t.elapsed().as_millis(), "final_commitment_fold");
    info!(elapsed_ms = %_nifs_total_t.elapsed().as_millis(), "nifs_prove");

    Ok((E_eq, az_folded, bz_folded, cz_folded, folded_W, folded_U))
  }
}

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaProverKey<E: Engine> {
  /// Commitment key
  pub ck: CommitmentKey<E>,
  /// Step circuit R1CS shape
  pub S_step: SplitR1CSShape<E>,
  /// Core circuit R1CS shape
  pub S_core: SplitR1CSShape<E>,
  /// Digest of the verifier's key
  pub vk_digest: SpartanDigest,
  /// Verifier circuit multi-round shape
  pub vc_shape: SplitMultiRoundR1CSShape<E>,
  /// Verifier circuit regular shape
  pub vc_shape_regular: R1CSShape<E>,
  /// Verifier circuit commitment key
  pub vc_ck: CommitmentKey<E>,
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaVerifierKey<E: Engine> {
  ck: CommitmentKey<E>,
  vk_ee: <E::PCS as PCSEngineTrait<E>>::VerifierKey,
  S_step: SplitR1CSShape<E>,
  S_core: SplitR1CSShape<E>,
  vc_shape: SplitMultiRoundR1CSShape<E>,
  vc_shape_regular: R1CSShape<E>,
  vc_ck: CommitmentKey<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<SpartanDigest>,
}

impl<E: Engine> SimpleDigestible for NeutronNovaVerifierKey<E> {}

impl<E: Engine> DigestHelperTrait<E> for NeutronNovaVerifierKey<E> {
  /// Returns the digest of the verifier's key.
  fn digest(&self) -> Result<SpartanDigest, SpartanError> {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::<_>::new(self);
        dc.digest()
      })
      .cloned()
      .map_err(|_| SpartanError::DigestError {
        reason: "Unable to compute digest for SpartanVerifierKey".to_string(),
      })
  }
}

/// A type that holds the pre-processed state for proving
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaPrepZkSNARK<E: Engine> {
  /// Pre-committed state for each step circuit
  pub ps_step: Vec<PrecommittedState<E>>,
  /// Pre-committed state for the core circuit
  pub ps_core: PrecommittedState<E>,
}

/// Holds the proof produced by the NeutronNova folding scheme followed by NeutronNova SNARK
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaZkSNARK<E: Engine> {
  step_instances: Vec<SplitR1CSInstance<E>>,
  core_instance: SplitR1CSInstance<E>,
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
  U_verifier: SplitMultiRoundR1CSInstance<E>,
  nifs: NovaNIFS<E>,
  random_U: RelaxedR1CSInstance<E>,
  folded_W: RelaxedR1CSWitness<E>,
}

impl<E: Engine> NeutronNovaZkSNARK<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Sets up the NeutronNova SNARK for a batch of circuits of type `C1` and a single circuit of type `C2`
  ///
  /// # Parameters
  /// - `step_circuit`: The circuit to be folded in the batch
  /// - `core_circuit`: The core circuit that connects the batch together
  /// - `num_steps`: The number of step circuits in the batch (will be padded to next power of two internally)
  pub fn setup<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    step_circuit: &C1,
    core_circuit: &C2,
    num_steps: usize,
  ) -> Result<(NeutronNovaProverKey<E>, NeutronNovaVerifierKey<E>), SpartanError> {
    let (_setup_span, setup_t) = start_span!("neutronnova_setup");

    let (_r1cs_span, r1cs_t) = start_span!("r1cs_shape_generation");
    debug!("Synthesizing step circuit");
    let mut S_step = ShapeCS::r1cs_shape(step_circuit)?;
    debug!("Finished synthesizing step circuit");

    debug!("Synthesizing core circuit");
    let mut S_core = ShapeCS::r1cs_shape(core_circuit)?;
    debug!("Finished synthesizing core circuit");

    SplitR1CSShape::equalize(&mut S_step, &mut S_core);

    info!(
      "Step circuit's witness sizes: shared = {}, precommitted = {}, rest = {}",
      S_step.num_shared, S_step.num_precommitted, S_step.num_rest
    );
    info!(
      "Core circuit's witness sizes: shared = {}, precommitted = {}, rest = {}",
      S_core.num_shared, S_core.num_precommitted, S_core.num_rest
    );
    info!(elapsed_ms = %r1cs_t.elapsed().as_millis(), "r1cs_shape_generation");

    let (_ck_span, ck_t) = start_span!("commitment_key_generation");
    let (ck, vk_ee) = SplitR1CSShape::commitment_key(&[&S_step, &S_core])?;
    info!(elapsed_ms = %ck_t.elapsed().as_millis(), "commitment_key_generation");

    // Calculate num_rounds_b from num_steps by padding to next power of two
    let (_vc_span, vc_t) = start_span!("verifier_circuit_setup");
    let num_rounds_b = num_steps.next_power_of_two().log_2();

    let num_vars = S_step.num_shared + S_step.num_precommitted + S_step.num_rest;
    let num_rounds_x =
      usize::try_from(S_step.num_cons.ilog2()).expect("constraint count log2 fits in usize");
    let num_rounds_y = usize::try_from(num_vars.ilog2()).expect("num_vars log2 fits in usize") + 1;
    let vc = NeutronNovaVerifierCircuit::<E>::default(num_rounds_b, num_rounds_x, num_rounds_y);
    let (vc_shape, vc_ck, _vk_mr) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&vc)?;
    let vc_shape_regular = vc_shape.to_regular_shape();
    info!(elapsed_ms = %vc_t.elapsed().as_millis(), "verifier_circuit_setup");

    let vk: NeutronNovaVerifierKey<E> = NeutronNovaVerifierKey {
      ck: ck.clone(),
      S_step: S_step.clone(),
      S_core: S_core.clone(),
      vk_ee,
      vc_shape: vc_shape.clone(),
      vc_shape_regular: vc_shape_regular.clone(),
      vc_ck: vc_ck.clone(),
      digest: OnceCell::new(),
    };

    let pk = NeutronNovaProverKey {
      ck,
      S_step,
      S_core,
      vc_shape,
      vc_shape_regular,
      vc_ck,
      vk_digest: vk.digest()?,
    };

    info!(elapsed_ms = %setup_t.elapsed().as_millis(), "neutronnova_setup");
    Ok((pk, vk))
  }

  /// Prepares the pre-processed state for proving
  /// Prepare witnesses for NeutronNova proving.
  /// `l0 > 0` requires small-value compatible witnesses.
  pub fn prep_prove<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    l0: usize,
  ) -> Result<NeutronNovaPrepZkSNARK<E>, SpartanError> {
    let is_small = l0 > 0;
    let (_prep_span, prep_t) = start_span!("neutronnova_prep_prove");

    // we synthesize shared witness for the first circuit; every other circuit including the core circuit shares this witness
    let (_shared_span, shared_t) = start_span!("generate_shared_witness");
    let mut ps =
      SatisfyingAssignment::shared_witness(&pk.S_step, &pk.ck, &step_circuits[0], is_small)?;
    info!(elapsed_ms = %shared_t.elapsed().as_millis(), "generate_shared_witness");

    let (_precommit_span, precommit_t) = start_span!(
      "generate_precommitted_witnesses",
      circuits = step_circuits.len() + 1
    );
    let ps_step = (0..step_circuits.len())
      .into_par_iter()
      .map(|i| {
        // copy ps to avoid mutating the original shared witness
        let mut ps_i = ps.clone();
        SatisfyingAssignment::precommitted_witness(
          &mut ps_i,
          &pk.S_step,
          &pk.ck,
          &step_circuits[i],
          is_small,
        )?;
        Ok(ps_i)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // we don't need to make a copy of ps for the core circuit, as it will be used only once
    SatisfyingAssignment::precommitted_witness(
      &mut ps,
      &pk.S_core,
      &pk.ck,
      core_circuit,
      is_small,
    )?;
    info!(elapsed_ms = %precommit_t.elapsed().as_millis(), circuits = step_circuits.len() + 1, "generate_precommitted_witnesses");

    info!(elapsed_ms = %prep_t.elapsed().as_millis(), "neutronnova_prep_prove");
    Ok(NeutronNovaPrepZkSNARK {
      ps_step,
      ps_core: ps,
    })
  }

  /// Prove the folding of a batch of R1CS instances and a core circuit that connects them together
  ///
  /// # Arguments
  /// * `l0` - Number of small-value sumcheck rounds:
  ///   - `l0 = 0`: Large-value mode (standard field arithmetic)
  ///   - `l0 > 0`: Decoupled mode with l0 small-value rounds (requires witness values to fit in i64)
  pub fn prove<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    prep_snark: &NeutronNovaPrepZkSNARK<E>,
    l0: usize,
  ) -> Result<Self, SpartanError>
  where
    E::Scalar: SmallValueField<i64>
      + DelayedReduction<i64>
      + DelayedReduction<i128>
      + DelayedReduction<E::Scalar>,
  {
    // l0 > 0 requires small-value compatible witnesses
    let is_small = l0 > 0;
    let (_prove_span, prove_t) = start_span!("neutronnova_prove");

    // rerandomize prep state: we first rerandomize core, then step circuits by reusing shared commitments
    let (_rerandomize_span, rerandomize_t) = start_span!("rerandomize_prep_state");
    let mut ps_core = prep_snark.ps_core.rerandomize(&pk.ck, &pk.S_core)?;
    let mut ps_step = prep_snark
      .ps_step
      .par_iter()
      .map(|ps_i| {
        ps_i.rerandomize_with_shared(
          &pk.ck,
          &pk.S_step,
          &ps_core.comm_W_shared,
          &ps_core.r_W_shared,
        )
      })
      .collect::<Result<Vec<_>, _>>()?;
    info!(elapsed_ms = %rerandomize_t.elapsed().as_millis(), "rerandomize_prep_state");

    // Parallel generation of instances and witnesses
    // Build instances and witnesses in one parallel pass
    let (_gen_span, gen_t) = start_span!(
      "generate_instances_witnesses",
      step_circuits = step_circuits.len()
    );
    let (res_steps, res_core) = rayon::join(
      || {
        ps_step
          .par_iter_mut()
          .zip(step_circuits.par_iter().enumerate())
          .map(|(pre_state, (i, circuit))| {
            let mut transcript = E::TE::new(b"neutronnova_prove");
            transcript.absorb(b"vk", &pk.vk_digest);
            transcript.absorb(
              b"num_circuits",
              &E::Scalar::from(step_circuits.len() as u64),
            );
            transcript.absorb(b"circuit_index", &E::Scalar::from(i as u64));

            let public_values =
              circuit
                .public_values()
                .map_err(|e| SpartanError::SynthesisError {
                  reason: format!("Circuit does not provide public IO: {e}"),
                })?;
            transcript.absorb(b"public_values", &public_values.as_slice());

            SatisfyingAssignment::r1cs_instance_and_witness(
              pre_state,
              &pk.S_step,
              &pk.ck,
              circuit,
              is_small,
              &mut transcript,
            )
          })
          .try_fold(
            || (Vec::new(), Vec::new()),
            |mut acc, res: Result<_, SpartanError>| {
              let (u, w) = res?;
              acc.0.push(u);
              acc.1.push(w);
              Ok(acc)
            },
          )
          .try_reduce(
            || (Vec::new(), Vec::new()),
            |mut a, mut b| {
              a.0.append(&mut b.0);
              a.1.append(&mut b.1);
              Ok(a)
            },
          )
      },
      || {
        // synthesize the core instance
        let mut transcript = E::TE::new(b"neutronnova_prove");
        transcript.absorb(b"vk", &pk.vk_digest);
        let public_values_core =
          core_circuit
            .public_values()
            .map_err(|e| SpartanError::SynthesisError {
              reason: format!("Core circuit does not provide public IO: {e}"),
            })?;
        transcript.absorb(b"public_values", &public_values_core.as_slice());
        SatisfyingAssignment::r1cs_instance_and_witness(
          &mut ps_core,
          &pk.S_core,
          &pk.ck,
          core_circuit,
          is_small,
          &mut transcript,
        )
      },
    );

    let ((step_instances, step_witnesses), (core_instance, core_witness)) = (res_steps?, res_core?);
    info!(elapsed_ms = %gen_t.elapsed().as_millis(), step_circuits = step_circuits.len(), "generate_instances_witnesses");

    let (_reg_span, reg_t) = start_span!("convert_to_regular_instances");
    let step_instances_regular = step_instances
      .iter()
      .map(|u| u.to_regular_instance())
      .collect::<Result<Vec<_>, _>>()?;

    let core_instance_regular = core_instance.to_regular_instance()?;
    info!(elapsed_ms = %reg_t.elapsed().as_millis(), "convert_to_regular_instances");

    // We start a new transcript for the NeutronNova NIFS proof
    // All instances will be absorbed into the transcript
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &pk.vk_digest);

    // absorb the core instance; NIFS will absorb the step instances
    transcript.absorb(b"core_instance", &core_instance_regular);

    let n_padded = step_instances_regular.len().next_power_of_two();
    let num_vars = pk.S_step.num_shared + pk.S_step.num_precommitted + pk.S_step.num_rest;
    let num_rounds_b = n_padded.log_2();
    let num_rounds_x = pk.S_step.num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let mut vc = NeutronNovaVerifierCircuit::<E>::default(num_rounds_b, num_rounds_x, num_rounds_y);
    let mut vc_state = SatisfyingAssignment::<E>::initialize_multiround_witness(&pk.vc_shape)?;

    // Perform ZK NIFS prove and collect outputs
    let (_nifs_span, nifs_t) = start_span!("NIFS");
    let (E_eq, Az_step, Bz_step, Cz_step, folded_W, folded_U) = NeutronNovaNIFS::<E>::prove::<i64>(
      &pk.S_step,
      &step_instances_regular,
      &step_witnesses,
      l0,
      &mut vc,
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &mut transcript,
    )?;
    info!(elapsed_ms = %nifs_t.elapsed().as_millis(), "NIFS");

    let (_tensor_span, tensor_t) = start_span!("compute_tensor_and_poly_tau");
    let (_ell, left, _right) = compute_tensor_decomp(pk.S_step.num_cons);
    let mut E1 = E_eq;
    let E2 = E1.split_off(left);

    let mut poly_tau_left = MultilinearPolynomial::new(E1);
    let poly_tau_right = MultilinearPolynomial::new(E2);

    info!(elapsed_ms = %tensor_t.elapsed().as_millis(), "compute_tensor_and_poly_tau");

    // outer sum-check preparation
    let (_mp_span, mp_t) = start_span!("prepare_multilinear_polys");
    let (mut poly_Az_step, mut poly_Bz_step, mut poly_Cz_step) = (
      MultilinearPolynomial::new(Az_step),
      MultilinearPolynomial::new(Bz_step),
      MultilinearPolynomial::new(Cz_step),
    );

    let (mut poly_Az_core, mut poly_Bz_core, mut poly_Cz_core) = {
      let (_core_span, core_t) = start_span!("compute_core_polys");
      let z = [
        core_witness.W.clone(),
        vec![E::Scalar::ONE],
        core_instance.public_values.clone(),
        core_instance.challenges.clone(),
      ]
      .concat();

      let (Az, Bz, Cz) = pk.S_core.multiply_vec(&z)?;
      info!(elapsed_ms = %core_t.elapsed().as_millis(), "compute_core_polys");
      (
        MultilinearPolynomial::new(Az),
        MultilinearPolynomial::new(Bz),
        MultilinearPolynomial::new(Cz),
      )
    };

    info!(elapsed_ms = %mp_t.elapsed().as_millis(), "prepare_multilinear_polys");
    let outer_start_index = num_rounds_b + 1;
    // outer sum-check (batched)
    let (_sc_span, sc_t) = start_span!("outer_sumcheck_batched");
    let r_x = SumcheckProof::<E>::prove_cubic_with_additive_term_batched_zk(
      num_rounds_x,
      &mut poly_tau_left,
      &poly_tau_right,
      &mut poly_Az_step,
      &mut poly_Az_core,
      &mut poly_Bz_step,
      &mut poly_Bz_core,
      &mut poly_Cz_step,
      &mut poly_Cz_core,
      &mut vc,
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &mut transcript,
      outer_start_index,
    )?;
    info!(elapsed_ms = %sc_t.elapsed().as_millis(), "outer_sumcheck_batched");

    vc.claim_Az_step = poly_Az_step[0];
    vc.claim_Bz_step = poly_Bz_step[0];
    vc.claim_Cz_step = poly_Cz_step[0];
    vc.claim_Az_core = poly_Az_core[0];
    vc.claim_Bz_core = poly_Bz_core[0];
    vc.claim_Cz_core = poly_Cz_core[0];
    vc.tau_at_rx = poly_tau_left[0];

    let chals = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &vc,
      outer_start_index + num_rounds_x,
      &mut transcript,
    )?;
    let r = chals[0];

    // inner sum-check preparation
    let claim_inner_joint_step = vc.claim_Az_step + r * vc.claim_Bz_step + r * r * vc.claim_Cz_step;
    let claim_inner_joint_core = vc.claim_Az_core + r * vc.claim_Bz_core + r * r * vc.claim_Cz_core;

    let (_eval_rx_span, eval_rx_t) = start_span!("compute_eval_rx");
    let evals_rx = EqPolynomial::evals_from_points(&r_x);
    info!(elapsed_ms = %eval_rx_t.elapsed().as_millis(), "compute_eval_rx");

    let (_sparse_span, sparse_t) = start_span!("compute_eval_table_sparse");
    let (evals_A_step, evals_B_step, evals_C_step) = pk.S_step.bind_row_vars(&evals_rx);
    let (evals_A_core, evals_B_core, evals_C_core) = pk.S_core.bind_row_vars(&evals_rx);
    info!(elapsed_ms = %sparse_t.elapsed().as_millis(), "compute_eval_table_sparse");

    let (_abc_span, abc_t) = start_span!("prepare_poly_ABC");
    let poly_ABC_step = (0..evals_A_step.len())
      .into_par_iter()
      .map(|i| evals_A_step[i] + r * evals_B_step[i] + r * r * evals_C_step[i])
      .collect::<Vec<E::Scalar>>();
    let poly_ABC_core = (0..evals_A_core.len())
      .into_par_iter()
      .map(|i| evals_A_core[i] + r * evals_B_core[i] + r * r * evals_C_core[i])
      .collect::<Vec<E::Scalar>>();
    info!(elapsed_ms = %abc_t.elapsed().as_millis(), "prepare_poly_ABC");

    // inner sum-check
    let (_sc2_span, sc2_t) = start_span!("inner_sumcheck_batched");

    debug!("Proving inner sum-check with {} rounds", num_rounds_y);
    debug!(
      "Inner sum-check sizes - poly_ABC_step: {}, poly_ABC_core: {}",
      poly_ABC_step.len(),
      poly_ABC_core.len()
    );
    let (r_y, evals) = SumcheckProof::<E>::prove_quad_batched_zk(
      &[claim_inner_joint_step, claim_inner_joint_core],
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC_step),
      &mut MultilinearPolynomial::new(poly_ABC_core),
      &mut MultilinearPolynomial::new({
        let mut v = vec![E::Scalar::ZERO; num_vars * 2];
        let w_len = folded_W.W.len();
        v[..w_len].copy_from_slice(&folded_W.W);
        v[w_len] = E::Scalar::ONE;
        let x_len = folded_U.X.len();
        v[w_len + 1..w_len + 1 + x_len].copy_from_slice(&folded_U.X);
        v
      }),
      &mut MultilinearPolynomial::new({
        let mut v = vec![E::Scalar::ZERO; num_vars * 2];
        let w_len = core_witness.W.len();
        v[..w_len].copy_from_slice(&core_witness.W);
        v[w_len] = E::Scalar::ONE;
        let x_len = core_instance_regular.X.len();
        v[w_len + 1..w_len + 1 + x_len].copy_from_slice(&core_instance_regular.X);
        v
      }),
      &mut vc,
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &mut transcript,
      outer_start_index + num_rounds_x + 1,
    )?;
    info!(elapsed_ms = %sc2_t.elapsed().as_millis(), "inner_sumcheck_batched");

    let eval_Z_step = evals[2];
    let eval_Z_core = evals[3];

    let eval_X_step = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(folded_U.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).expect("num_vars log2 fits in usize");
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };
    let eval_X_core = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(core_instance_regular.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).expect("num_vars log2 fits in usize");
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };
    let eval_W_step = (eval_Z_step - r_y[0] * eval_X_step)
      * (E::Scalar::ONE - r_y[0])
        .invert()
        .expect("1 - r_y[0] is non-zero");
    let eval_W_core = (eval_Z_core - r_y[0] * eval_X_core)
      * (E::Scalar::ONE - r_y[0])
        .invert()
        .expect("1 - r_y[0] is non-zero");

    vc.eval_W_step = eval_W_step;
    vc.eval_W_core = eval_W_core;
    vc.eval_X_step = eval_X_step;
    vc.eval_X_core = eval_X_core;

    // Inner final equality round
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &vc,
      outer_start_index + num_rounds_x + 1 + num_rounds_y,
      &mut transcript,
    )?;

    // Commit eval_W_step
    let eval_w_step_commit_round = outer_start_index + num_rounds_x + 1 + num_rounds_y + 1;
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &vc,
      eval_w_step_commit_round,
      &mut transcript,
    )?;

    // Commit eval_W_core
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &vc,
      eval_w_step_commit_round + 1,
      &mut transcript,
    )?;

    let (U_verifier, W_verifier) =
      SatisfyingAssignment::<E>::finalize_multiround_witness(&mut vc_state, &pk.vc_shape)?;

    let U_verifier_regular = U_verifier.to_regular_instance()?;
    let (random_U, random_W) = pk
      .vc_shape_regular
      .sample_random_instance_witness(&pk.vc_ck)?;

    let (nifs, folded_W_verifier) = NovaNIFS::<E>::prove(
      &pk.vc_ck,
      &pk.vc_shape_regular,
      &random_U,
      &random_W,
      &U_verifier_regular,
      &W_verifier,
      &mut transcript,
    )?;

    // access two claimed commitments to evaluations of W_step and W_core
    let comm_eval_W_step = U_verifier.comm_w_per_round[eval_w_step_commit_round].clone();
    let blind_eval_W_step = vc_state.r_w_per_round[eval_w_step_commit_round].clone();

    let comm_eval_W_core = U_verifier.comm_w_per_round[eval_w_step_commit_round + 1].clone();
    let blind_eval_W_core = vc_state.r_w_per_round[eval_w_step_commit_round + 1].clone();

    // the commitments are already absorbed in the transcript, so we simply squeeze the challenge
    let c_eval = transcript.squeeze(b"c_eval")?;

    // fold evaluation claims into one
    let (_fold_eval_span, fold_eval_t) = start_span!("fold_evaluation_claims");
    let comm = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &[folded_U.comm_W, core_instance_regular.comm_W],
      &[E::Scalar::ONE, c_eval],
    )?;
    let blind = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
      &[folded_W.r_W.clone(), core_witness.r_W.clone()],
      &[E::Scalar::ONE, c_eval],
    )?;
    let W = folded_W
      .W
      .par_iter()
      .zip(core_witness.W.par_iter())
      .map(|(w1, w2)| *w1 + c_eval * *w2)
      .collect::<Vec<_>>();
    let comm_eval = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &[comm_eval_W_step, comm_eval_W_core],
      &[E::Scalar::ONE, c_eval],
    )?;
    let blind_eval = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
      &[blind_eval_W_step, blind_eval_W_core],
      &[E::Scalar::ONE, c_eval],
    )?;
    info!(elapsed_ms = %fold_eval_t.elapsed().as_millis(), "fold_evaluation_claims");

    let (_pcs_span, pcs_t) = start_span!("pcs_prove");
    let eval_arg = E::PCS::prove(
      &pk.ck,
      &pk.vc_ck,
      &mut transcript,
      &comm,
      &W,
      &blind,
      &r_y[1..],
      &comm_eval,
      &blind_eval,
    )?;
    info!(elapsed_ms = %pcs_t.elapsed().as_millis(), "pcs_prove");

    let result = Self {
      step_instances,
      core_instance,
      eval_arg,
      U_verifier,
      nifs,
      random_U,
      folded_W: folded_W_verifier,
    };

    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "neutronnova_prove");
    Ok(result)
  }

  /// Verifies the NeutronNovaZkSNARK and returns the public IO from the instances
  pub fn verify(
    &self,
    vk: &NeutronNovaVerifierKey<E>,
    num_instances: usize,
  ) -> Result<(Vec<Vec<E::Scalar>>, Vec<E::Scalar>), SpartanError> {
    let (_verify_span, _verify_t) = start_span!("neutronnova_verify");
    if num_instances != self.step_instances.len() {
      return Err(SpartanError::ProofVerifyError {
        reason: format!(
          "Expected {} instances, got {}",
          num_instances,
          self.step_instances.len()
        ),
      });
    }

    // validate the step instances
    let (_validate_span, validate_t) =
      start_span!("validate_instances", instances = self.step_instances.len());
    for (i, u) in self.step_instances.iter().enumerate() {
      let mut transcript = E::TE::new(b"neutronnova_prove");
      transcript.absorb(b"vk", &vk.digest()?);
      transcript.absorb(
        b"num_circuits",
        &E::Scalar::from(self.step_instances.len() as u64),
      );
      transcript.absorb(b"circuit_index", &E::Scalar::from(i as u64));
      // absorb the public IO into the transcript
      transcript.absorb(b"public_values", &u.public_values.as_slice());

      u.validate(&vk.S_step, &mut transcript)?;
    }

    // validate the core instance
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &vk.digest()?);
    // absorb the public IO into the transcript
    transcript.absorb(
      b"public_values",
      &self.core_instance.public_values.as_slice(),
    );

    self.core_instance.validate(&vk.S_core, &mut transcript)?;
    info!(elapsed_ms = %validate_t.elapsed().as_millis(), instances = self.step_instances.len(), "validate_instances");

    // we require all step instances to have the same shared commitment and match the shared commitment of the core instance
    for u in &self.step_instances {
      if u.comm_W_shared != self.core_instance.comm_W_shared {
        return Err(SpartanError::ProofVerifyError {
          reason: "All instances must have the same shared commitment".to_string(),
        });
      }
    }

    let (_convert_span, convert_t) = start_span!("convert_to_regular_verify");
    let mut step_instances_padded = self.step_instances.clone();
    if step_instances_padded.len() != step_instances_padded.len().next_power_of_two() {
      step_instances_padded.extend(std::iter::repeat_n(
        step_instances_padded[0].clone(),
        step_instances_padded.len().next_power_of_two() - step_instances_padded.len(),
      ));
    }
    let step_instances_regular = step_instances_padded
      .par_iter()
      .map(|u| u.to_regular_instance())
      .collect::<Result<Vec<_>, _>>()?;

    let core_instance_regular = self.core_instance.to_regular_instance()?;
    info!(elapsed_ms = %convert_t.elapsed().as_millis(), "convert_to_regular_verify");

    // We start a new transcript for the NeutronNova NIFS proof
    let mut transcript = E::TE::new(b"neutronnova_prove");

    // absorb the verifier key and instances
    transcript.absorb(b"vk", &vk.digest()?);
    transcript.absorb(b"core_instance", &core_instance_regular);
    for U in step_instances_regular.iter() {
      transcript.absorb(b"U", U);
    }
    transcript.absorb(b"T", &E::Scalar::ZERO); // we always have T=0 in NeutronNova

    // compute the number of rounds of NIFS, outer sum-check, and inner sum-check
    let num_rounds_b = step_instances_regular.len().log_2();
    let num_vars = vk.S_step.num_shared + vk.S_step.num_precommitted + vk.S_step.num_rest;
    let num_rounds_x = vk.S_step.num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    // we need num_rounds_b challenges for folding the step instances; we also need tau to compress multiple R1CS checks
    let tau = transcript.squeeze(b"tau")?;
    let rhos = (0..num_rounds_b)
      .map(|_| transcript.squeeze(b"rho"))
      .collect::<Result<Vec<_>, _>>()?;

    // validate the provided multi-round verifier instance and advance transcript
    self.U_verifier.validate(&vk.vc_shape, &mut transcript)?;

    let U_verifier_regular = self.U_verifier.to_regular_instance()?;

    // extract challenges and public IO from U_verifier's public IO
    let num_public_values = 6usize;
    let num_challenges = num_rounds_b + num_rounds_x + 1 + num_rounds_y;
    if U_verifier_regular.X.len() != num_challenges + num_public_values {
      return Err(SpartanError::ProofVerifyError {
        reason: format!(
          "Verifier instance has incorrect number of public IO: expected {}, got {}",
          num_challenges + num_public_values,
          U_verifier_regular.X.len()
        ),
      });
    }

    let challenges = &U_verifier_regular.X[0..num_challenges];
    let public_values = &U_verifier_regular.X[num_challenges..num_challenges + 6];

    let r_b = challenges[0..num_rounds_b].to_vec();
    let r_x = challenges[num_rounds_b..num_rounds_b + num_rounds_x].to_vec();
    let r = challenges[num_rounds_b + num_rounds_x]; // r for combining inner claims
    let r_y = challenges[num_rounds_b + num_rounds_x + 1..].to_vec();

    let folded_U = R1CSInstance::fold_multiple(&r_b, &step_instances_regular)?;

    let folded_U_verifier =
      self
        .nifs
        .verify(&mut transcript, &self.random_U, &U_verifier_regular)?;

    vk.vc_shape_regular
      .is_sat_relaxed(&vk.vc_ck, &folded_U_verifier, &self.folded_W)
      .map_err(|e| SpartanError::ProofVerifyError {
        reason: format!("Folded instance not satisfiable: {e}"),
      })?;

    let (_matrix_eval_span, matrix_eval_t) = start_span!("matrix_evaluations");
    let (eval_A_step, eval_B_step, eval_C_step, eval_A_core, eval_B_core, eval_C_core) = {
      let T_x = EqPolynomial::evals_from_points(&r_x);
      let T_y = EqPolynomial::evals_from_points(&r_y);
      let (eval_A_step, eval_B_step, eval_C_step) = vk.S_step.evaluate_with_tables(&T_x, &T_y);
      let (eval_A_core, eval_B_core, eval_C_core) = vk.S_core.evaluate_with_tables(&T_x, &T_y);

      (
        eval_A_step,
        eval_B_step,
        eval_C_step,
        eval_A_core,
        eval_B_core,
        eval_C_core,
      )
    };
    info!(elapsed_ms = %matrix_eval_t.elapsed().as_millis(), "matrix_evaluations");

    let eval_X_step = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(folded_U.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).expect("num_vars log2 fits in usize");
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };
    let eval_X_core = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(core_instance_regular.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).expect("num_vars log2 fits in usize");
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };

    // Compute quotient_* = (eval_A + r*eval_B + r^2*eval_C) for both branches
    let quotient_step = eval_A_step + r * eval_B_step + r * r * eval_C_step;
    let quotient_core = eval_A_core + r * eval_B_core + r * r * eval_C_core;
    let tau_at_rx = PowPolynomial::new(&tau, r_x.len()).evaluate(&r_x)?;
    let eq_rho_at_rb = EqPolynomial::new(r_b).evaluate(&rhos);

    if public_values[0] != tau_at_rx
      || public_values[1] != eval_X_step
      || public_values[2] != eval_X_core
      || public_values[3] != eq_rho_at_rb
      || public_values[4] != quotient_step
      || public_values[5] != quotient_core
    {
      return Err(SpartanError::ProofVerifyError {
        reason:
          "Verifier instance public tau_at_rx/eval_X_step/eq_rho_at_rb/eval_X_core/quotients do not match recomputation"
            .to_string(),
      });
    }

    // verify PCS eval
    let c_eval = transcript.squeeze(b"c_eval")?;

    let eval_w_step_commit_round = num_rounds_b + 1 + num_rounds_x + 1 + num_rounds_y + 1;
    let comm_eval_W_step = self.U_verifier.comm_w_per_round[eval_w_step_commit_round].clone();
    let comm_eval_W_core = self.U_verifier.comm_w_per_round[eval_w_step_commit_round + 1].clone();

    let comm = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &[folded_U.comm_W, core_instance_regular.comm_W],
      &[E::Scalar::ONE, c_eval],
    )?;
    let comm_eval = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &[comm_eval_W_step, comm_eval_W_core],
      &[E::Scalar::ONE, c_eval],
    )?;

    let (_pcs_verify_span, pcs_verify_t) = start_span!("pcs_verify");
    E::PCS::verify(
      &vk.vk_ee,
      &vk.vc_ck,
      &mut transcript,
      &comm,
      &r_y[1..],
      &comm_eval,
      &self.eval_arg,
    )?;
    info!(elapsed_ms = %pcs_verify_t.elapsed().as_millis(), "pcs_verify");

    info!(elapsed_ms = %_verify_t.elapsed().as_millis(), "neutronnova_verify");

    let public_values_step = self
      .step_instances
      .iter()
      .take(num_instances)
      .map(|u| u.public_values.clone())
      .collect::<Vec<Vec<_>>>();

    let public_values_core = self.core_instance.public_values.clone();

    // return a vector of public values
    Ok((public_values_step, public_values_core))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{gadgets::CubicChainCircuit, provider::T256HyraxEngine};
  use bellpepper::gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    sha256::sha256,
  };
  use bellpepper_core::{ConstraintSystem, SynthesisError};
  use core::marker::PhantomData;

  #[derive(Clone, Debug)]
  struct Sha256Circuit<E: Engine> {
    preimage: Vec<u8>,
    _p: PhantomData<E>,
  }

  impl<E: Engine> SpartanCircuit<E> for Sha256Circuit<E> {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
      Ok(vec![E::Scalar::ZERO]) // Placeholder, we don't use public values in this example
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![]) // Placeholder, we don't use shared variables in this example
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![]) // Placeholder, we don't use precommitted variables in this example
    }

    fn num_challenges(&self) -> usize {
      0 // Placeholder, we don't use challenges in this example
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _shared: &[AllocatedNum<E::Scalar>],
      _precommitted: &[AllocatedNum<E::Scalar>],
      _challenges: Option<&[E::Scalar]>, // challenges from the verifier
    ) -> Result<(), SynthesisError> {
      // we write a circuit that checks if the input is a SHA256 preimage
      let bit_values: Vec<_> = self
        .preimage
        .clone()
        .into_iter()
        .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
        .map(Some)
        .collect();
      assert_eq!(bit_values.len(), self.preimage.len() * 8);

      let preimage_bits = bit_values
        .into_iter()
        .enumerate()
        .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
        .map(|b| b.map(Boolean::from))
        .collect::<Result<Vec<_>, _>>()?;

      let _ = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::ZERO))?;
      x.inputize(cs.namespace(|| "inputize x"))?;

      Ok(())
    }
  }

  fn generate_sha_r1cs<E: Engine>(
    num_circuits: usize,
    len: usize,
  ) -> (
    NeutronNovaProverKey<E>,
    NeutronNovaVerifierKey<E>,
    Vec<Sha256Circuit<E>>,
  )
  where
    E::PCS: FoldingEngineTrait<E>, // Ensure that the PCS supports folding
  {
    let circuit = Sha256Circuit::<E> {
      preimage: vec![0u8; len],
      _p: Default::default(),
    };

    let (pk, vk) = NeutronNovaZkSNARK::<E>::setup(&circuit, &circuit, num_circuits).unwrap();

    let circuits = (0..num_circuits)
      .map(|i| Sha256Circuit::<E> {
        preimage: vec![i as u8; len],
        _p: Default::default(),
      })
      .collect::<Vec<_>>();

    (pk, vk, circuits)
  }

  fn test_neutron_inner<E: Engine, C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    name: &str,
    pk: &NeutronNovaProverKey<E>,
    vk: &NeutronNovaVerifierKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    l0: usize,
  ) where
    E::PCS: FoldingEngineTrait<E>,
    E::Scalar: SmallValueField<i64>
      + DelayedReduction<i64>
      + DelayedReduction<i128>
      + DelayedReduction<E::Scalar>,
  {
    println!(
      "[bench_neutron_inner] name: {name}, num_circuits: {}, l0: {l0}",
      step_circuits.len()
    );

    let ps = NeutronNovaZkSNARK::<E>::prep_prove(pk, step_circuits, core_circuit, l0).unwrap();
    let res = NeutronNovaZkSNARK::prove(pk, step_circuits, core_circuit, &ps, l0);
    assert!(res.is_ok());

    let snark = res.unwrap();
    let res = snark.verify(vk, step_circuits.len());
    println!(
      "[bench_neutron_inner] name: {name}, num_circuits: {}, verify res: {:?}",
      step_circuits.len(),
      res
    );
    assert!(res.is_ok());

    let (public_values_step, _public_values_core) = res.unwrap();
    assert_eq!(public_values_step.len(), step_circuits.len());
  }

  #[test]
  fn test_neutron_sha256() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true)
      .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
      .try_init();

    type E = T256HyraxEngine;

    for num_circuits in [2, 7, 32, 64] {
      for len in [32, 64].iter() {
        let (pk, vk, circuits) = generate_sha_r1cs::<E>(num_circuits, *len);
        test_neutron_inner(
          &format!("sha256_num_circuits={num_circuits}_len={len}"),
          &pk,
          &vk,
          &circuits,
          &circuits[0], // core circuit is the first one, for test purposes
          0,            // SHA256 witnesses don't fit in i64, use large-value mode
        );
      }
    }
  }

  /// A simple circuit with guaranteed small witness values.
  /// Computes x^2 + x + 5 = y where x is a small input.
  #[derive(Clone, Debug)]
  struct SmallWitnessCircuit<E: Engine> {
    x: u64,
    _p: PhantomData<E>,
  }

  impl<E: Engine> SpartanCircuit<E> for SmallWitnessCircuit<E> {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
      Ok(vec![E::Scalar::from(self.x)])
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![])
    }

    fn num_challenges(&self) -> usize {
      0
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _shared: &[AllocatedNum<E::Scalar>],
      _precommitted: &[AllocatedNum<E::Scalar>],
      _challenges: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::from(self.x)))?;
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let sum = AllocatedNum::alloc(cs.namespace(|| "sum"), || {
        Ok(E::Scalar::from(self.x * self.x + self.x + 5))
      })?;
      // x^2 + x + 5 = sum
      cs.enforce(
        || "x_sq + x + 5 = sum",
        |lc| {
          lc + x_sq.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + sum.get_variable(),
      );
      // Expose x as public input
      x.inputize(cs.namespace(|| "input x"))?;
      Ok(())
    }
  }

  #[test]
  fn test_neutronnova_small_value_proves_and_verifies() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true)
      .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
      .try_init();

    type E = T256HyraxEngine;

    let num_circuits = 4;
    let template = SmallWitnessCircuit::<E> {
      x: 0,
      _p: Default::default(),
    };
    let (pk, vk) = NeutronNovaZkSNARK::<E>::setup(&template, &template, num_circuits).unwrap();

    let circuits: Vec<_> = (0..num_circuits as u64)
      .map(|i| SmallWitnessCircuit::<E> {
        x: i + 1,
        _p: Default::default(),
      })
      .collect();

    // Prep once (can be shared between both prove paths)
    let ps = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &circuits[0], 0).unwrap();

    // Verify large-value path works (l0=0)
    {
      let snark_v = NeutronNovaZkSNARK::prove(&pk, &circuits, &circuits[0], &ps, 0)
        .expect("large-value prove should succeed");
      let res_v = snark_v.verify(&vk, circuits.len());
      assert!(
        res_v.is_ok(),
        "large-value proof should verify: {:?}",
        res_v.err()
      );
    }

    // Verify small-value path works (l0=ell_b for fully decoupled mode)
    {
      let ell_b = num_circuits.next_power_of_two().trailing_zeros() as usize;
      let snark_s = NeutronNovaZkSNARK::prove(&pk, &circuits, &circuits[0], &ps, ell_b)
        .expect("small-value prove should succeed");
      let res_s = snark_s.verify(&vk, circuits.len());
      assert!(
        res_s.is_ok(),
        "small-value proof should verify: {:?}",
        res_s.err()
      );
    }
  }

  /// Test NeutronNovaZkSNARK equivalence (and implicitly NIFS equivalence) with varying circuit counts.
  ///
  /// This tests both the regular field path (is_small=false) and the small-value optimized
  /// path (is_small=true) across different numbers of folded circuits to ensure both produce valid proofs.
  #[test]
  fn test_neutronnova_equivalence_varying_num_circuits() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true)
      .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
      .try_init();

    type E = T256HyraxEngine;

    // Test with different numbers of circuits being folded
    // Includes non-power-of-2 counts to test padding logic
    for num_circuits in [2, 3, 4, 5, 7, 8, 16] {
      test_neutronnova_equivalence_for_num_circuits::<E>(num_circuits);
    }
  }

  fn test_neutronnova_equivalence_for_num_circuits<E: Engine>(num_circuits: usize)
  where
    E::PCS: FoldingEngineTrait<E>,
    E::Scalar: SmallValueField<i64>
      + DelayedReduction<i64>
      + DelayedReduction<i128>
      + DelayedReduction<E::Scalar>,
  {
    const NUM_ROUNDS: usize = 6; // Fixed circuit size with ~64 constraints
    let circuit = CubicChainCircuit::for_rounds(NUM_ROUNDS);
    let expected_output = circuit.expected_output::<E::Scalar>();

    // Setup with template circuit
    let (pk, vk) = NeutronNovaZkSNARK::<E>::setup(&circuit, &circuit, num_circuits).unwrap();

    // Create circuit instances (all identical for this test)
    let circuits: Vec<_> = (0..num_circuits).map(|_| circuit.clone()).collect();

    // Prep once (can be shared between both prove paths)
    let ps = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &circuits[0], 0).unwrap();

    // Compute ell_b for small-value l0 constraint
    let ell_b = num_circuits.next_power_of_two().trailing_zeros() as usize;
    let small_l0 = std::cmp::min(3, ell_b);

    // Helper to test prove and verify
    #[allow(clippy::expect_fun_call)]
    let assert_prove_and_verify = |l0: usize, path_name: &str| {
      let snark = NeutronNovaZkSNARK::prove(&pk, &circuits, &circuits[0], &ps, l0).expect(
        &format!("{path_name} prove should succeed for num_circuits={num_circuits}"),
      );
      let res = snark.verify(&vk, circuits.len());
      assert!(
        res.is_ok(),
        "{path_name} proof should verify for num_circuits={num_circuits}: {:?}",
        res.err()
      );
      let (public_values, _) = res.unwrap();
      assert_eq!(
        public_values.len(),
        num_circuits,
        "should have {num_circuits} public values"
      );
      for (i, pv) in public_values.iter().enumerate() {
        assert_eq!(
          pv,
          &[expected_output],
          "{path_name} path: circuit {i} output mismatch for num_circuits={num_circuits}"
        );
      }
    };

    assert_prove_and_verify(0, "large-value");
    assert_prove_and_verify(small_l0, "small-value");
  }

  /// Test that NIFS sumcheck polynomial generation produces identical results
  /// between the regular (field) path and the small-value optimized path.
  ///
  /// Uses CubicChainCircuit to generate realistic R1CS data with small witness values.
  fn run_nifs_sumcheck_polynomial_equivalence_test<E: Engine>(num_instances: usize)
  where
    E::PCS: FoldingEngineTrait<E>,
    E::Scalar: SmallValueField<i64>
      + DelayedReduction<i64>
      + DelayedReduction<i128>
      + DelayedReduction<E::Scalar>,
  {
    // 1. Setup using CubicChainCircuit (guarantees small witness values)
    const NUM_ROUNDS: usize = 6; // ~64 constraints
    let circuit = CubicChainCircuit::for_rounds(NUM_ROUNDS);
    let (pk, _vk) = NeutronNovaZkSNARK::<E>::setup(&circuit, &circuit, num_instances).unwrap();
    let circuits: Vec<_> = (0..num_instances).map(|_| circuit.clone()).collect();

    // 2. Generate witnesses using prep_prove
    let ps = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &circuits[0], 0).unwrap();

    // 3. Synthesize full instances and witnesses
    let mut instances_witnesses: Vec<(R1CSInstance<E>, R1CSWitness<E>)> = Vec::new();
    for (i, circuit) in circuits.iter().enumerate() {
      let mut ps_clone = ps.ps_step[i].clone();
      let mut transcript = E::TE::new(b"test_transcript");
      transcript.absorb(b"index", &E::Scalar::from(i as u64));
      let (instance, witness) = SatisfyingAssignment::<E>::r1cs_instance_and_witness(
        &mut ps_clone,
        &pk.S_step,
        &pk.ck,
        circuit,
        true,
        &mut transcript,
      )
      .unwrap();
      instances_witnesses.push((instance.to_regular_instance().unwrap(), witness));
    }

    // 4. Pad to power of 2
    let n_padded = num_instances.next_power_of_two();
    let ell_b = n_padded.log_2();
    while instances_witnesses.len() < n_padded {
      instances_witnesses.push(instances_witnesses[0].clone());
    }

    // 5. Generate deterministic tau, rhos, and challenges
    let tau = E::Scalar::from(42u64);
    let rhos: Vec<E::Scalar> = (0..ell_b)
      .map(|i| E::Scalar::from((i + 3) as u64))
      .collect();
    let challenges: Vec<E::Scalar> = (0..ell_b)
      .map(|i| E::Scalar::from((i + 7) as u64))
      .collect();

    // 6. Compute E_eq via tensor decomposition
    let (ell_cons, left, right) = compute_tensor_decomp(pk.S_step.num_cons);
    let E_eq = PowPolynomial::split_evals(tau, ell_cons, left, right);

    // 7. Compute Az, Bz, Cz for all instances (field version)
    let triples: Vec<_> = instances_witnesses
      .iter()
      .map(|(u, w)| {
        let z = build_z::<E>(&w.W, &u.X);
        pk.S_step.multiply_vec(&z).unwrap()
      })
      .collect();

    let (mut A_layers, mut B_layers, mut C_layers): (
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
    ) = triples.into_iter().fold(
      (
        Vec::with_capacity(n_padded),
        Vec::with_capacity(n_padded),
        Vec::with_capacity(n_padded),
      ),
      |(mut a, mut b, mut c), (az, bz, cz)| {
        a.push(az);
        b.push(bz);
        c.push(cz);
        (a, b, c)
      },
    );

    // 8. Compute small-value versions
    let smalls: Vec<_> = instances_witnesses
      .iter()
      .map(|(u, w)| {
        let w_small: Vec<i64> = w
          .W
          .iter()
          .map(|v| E::Scalar::try_field_to_small(v).unwrap())
          .collect();
        let x_small: Vec<i64> = u
          .X
          .iter()
          .map(|v| E::Scalar::try_field_to_small(v).unwrap())
          .collect();
        let z_small = build_z_small(&w_small, &x_small);
        let az = pk
          .S_step
          .A
          .multiply_vec_small::<2, i64>(&z_small, ell_b)
          .unwrap();
        let bz = pk
          .S_step
          .B
          .multiply_vec_small::<2, i64>(&z_small, ell_b)
          .unwrap();
        (az, bz)
      })
      .collect();

    let (a_small, b_small): (Vec<Vec<i64>>, Vec<Vec<i64>>) = smalls.into_iter().unzip();

    // Run REGULAR path sumcheck
    let mut polys_regular: Vec<UniPoly<E::Scalar>> = Vec::with_capacity(ell_b);
    let mut T_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;
    let mut m = n_padded;

    for t in 0..ell_b {
      let rho_t = rhos[t];
      let pairs = m / 2;

      // Compute (e0, quad_coeff) via prove_helper for each pair
      let (e0, quad_coeff) = A_layers
        .chunks(2)
        .zip(B_layers.chunks(2))
        .zip(C_layers.chunks(2))
        .enumerate()
        .map(|(pair_idx, ((pair_a, pair_b), pair_c))| {
          let (e0, quad_coeff) = NeutronNovaNIFS::<E>::prove_helper(
            t,
            (left, right),
            &E_eq,
            &pair_a[0],
            &pair_b[0],
            &pair_c[0],
            &pair_a[1],
            &pair_b[1],
            &pair_c[1],
          );
          let w = suffix_weight_full::<E::Scalar>(t, ell_b, pair_idx, &rhos);
          (e0 * w, quad_coeff * w)
        })
        .fold((E::Scalar::ZERO, E::Scalar::ZERO), |a, b| {
          (a.0 + b.0, a.1 + b.1)
        });

      // Build polynomial (same formula as prove_regular)
      let one_minus_rho = E::Scalar::ONE - rho_t;
      let two_rho_minus_one = rho_t - one_minus_rho;
      let c = e0 * acc_eq;
      let a = quad_coeff * acc_eq;
      let a_b_c = (T_cur - c * one_minus_rho) * rho_t.invert().unwrap();
      let b = a_b_c - a - c;
      let poly_t = UniPoly {
        coeffs: vec![
          c * one_minus_rho,                         // d
          c * two_rho_minus_one + b * one_minus_rho, // c
          b * two_rho_minus_one + a * one_minus_rho, // b
          a * two_rho_minus_one,                     // a
        ],
      };
      polys_regular.push(poly_t.clone());

      // Use pre-determined challenge
      let r_b = challenges[t];
      acc_eq *= (E::Scalar::ONE - r_b) * (E::Scalar::ONE - rho_t) + r_b * rho_t;
      T_cur = poly_t.evaluate(&r_b);

      // Fold layers for next round
      let mut next_A = vec![vec![]; m];
      let mut next_B = vec![vec![]; m];
      let mut next_C = vec![vec![]; m];
      for i in 0..m {
        let target = if i & 1 == 0 { i >> 1 } else { (i >> 1) + pairs };
        next_A[target] = std::mem::take(&mut A_layers[i]);
        next_B[target] = std::mem::take(&mut B_layers[i]);
        next_C[target] = std::mem::take(&mut C_layers[i]);
      }
      A_layers = next_A;
      B_layers = next_B;
      C_layers = next_C;

      for matrix_layer in [&mut A_layers, &mut B_layers, &mut C_layers] {
        let (low, high) = matrix_layer.split_at_mut(pairs);
        low.iter_mut().zip(high.iter()).for_each(|(lo, hi)| {
          lo.iter_mut().zip(hi.iter()).for_each(|(l, h)| {
            *l += mul_opt(&(*h - *l), &r_b);
          });
        });
        matrix_layer.truncate(pairs);
      }

      m = pairs;
    }

    // Run SMALL-VALUE path sumcheck
    let accumulators =
      build_accumulators_neutronnova(&a_small, &b_small, &E_eq, left, right, &rhos, ell_b);
    let mut small_value = SmallValueSumCheck::<E::Scalar, 2>::from_accumulators(accumulators);

    let mut polys_small: Vec<UniPoly<E::Scalar>> = Vec::with_capacity(ell_b);
    let mut T_cur_small = E::Scalar::ZERO;

    for (i, rho_i) in rhos.iter().enumerate() {
      let t_all = small_value.eval_t_all_u(i);
      let t0 = t_all.at_zero();
      let t_inf = t_all.at_infinity();

      let li = small_value.eq_round_values(*rho_i);
      let t1 = derive_t1(li.at_zero(), li.at_one(), T_cur_small, t0).unwrap();

      let poly = build_univariate_round_polynomial(&li, t0, t1, t_inf);
      polys_small.push(poly.clone());

      let r_i = challenges[i]; // Same challenge as regular path
      T_cur_small = poly.evaluate(&r_i);
      small_value.advance(&li, r_i);
    }

    // Compare polynomial coefficients round-by-round
    for (round, (p_reg, p_small)) in polys_regular.iter().zip(&polys_small).enumerate() {
      assert_eq!(
        p_reg.coeffs, p_small.coeffs,
        "round {round} polynomial mismatch for num_instances={num_instances}\n\
         regular: {:?}\n\
         small:   {:?}",
        p_reg.coeffs, p_small.coeffs
      );
    }
  }

  #[test]
  fn test_nifs_sumcheck_polynomial_equivalence() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true)
      .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
      .try_init();

    type E = T256HyraxEngine;

    // Test with both power-of-2 and non-power-of-2 instance counts
    // Non-power-of-2 tests the padding logic
    for num_instances in [2, 3, 4, 5, 7, 8, 16] {
      run_nifs_sumcheck_polynomial_equivalence_test::<E>(num_instances);
    }
  }
}
