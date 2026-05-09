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
use crate::start_span;
use crate::{
  Commitment, CommitmentKey, DEFAULT_COMMITMENT_WIDTH, VerifierKey,
  bellpepper::{
    r1cs::{
      MultiRoundSpartanShape, MultiRoundSpartanWitness, PrecommittedState, SpartanShape,
      SpartanWitness,
    },
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  big_num::{DelayedReduction, SmallAccumulator, small_value_field::to_small_vec_or_zero},
  digest::DigestComputer,
  errors::SpartanError,
  math::Math,
  nifs::NovaNIFS,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
    power::PowPolynomial,
    univariate::UniPoly,
  },
  r1cs::{
    R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, SplitMultiRoundR1CSInstance,
    SplitMultiRoundR1CSShape, SplitR1CSInstance, SplitR1CSShape, weights_from_r,
  },
  small_constraint_system::SmallShapeCS,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::{SmallSpartanCircuit, SpartanCircuit},
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    snark::{DigestHelperTrait, SpartanDigest},
    transcript::TranscriptEngineTrait,
  },
  zk::NeutronNovaVerifierCircuit,
};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

#[path = "small_neutronnova_zk.rs"]
mod small_neutronnova_zk;

pub use small_neutronnova_zk::{
  NeutronNovaAccumulatorPrepZkSNARK, NeutronNovaSmallAccumulatorPrepZkSNARK,
};

type NeutronNovaNIFSOutput<E> = (
  Vec<<E as Engine>::Scalar>,
  Vec<<E as Engine>::Scalar>,
  Vec<<E as Engine>::Scalar>,
  Vec<<E as Engine>::Scalar>,
  R1CSWitness<E>,
  R1CSInstance<E>,
);

type MultiRoundState<E> = <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState;

fn compute_tensor_decomp(n: usize) -> (usize, usize, usize) {
  let ell = n.next_power_of_two().log_2();
  // we split ell into ell1 and ell2 such that ell1 + ell2 = ell and ell1 >= ell2
  let ell1 = ell.div_ceil(2); // This ensures ell1 >= ell2
  let ell2 = ell / 2;
  let left = 1 << ell1;
  let right = 1 << ell2;

  (ell, left, right)
}

#[inline]
fn build_z<E: Engine>(w: &[E::Scalar], x: &[E::Scalar]) -> Vec<E::Scalar> {
  let mut z = Vec::with_capacity(w.len() + 1 + x.len());
  z.extend_from_slice(w);
  z.push(E::Scalar::ONE);
  z.extend_from_slice(x);
  z
}

#[allow(dead_code)]
fn prepare_nifs_inputs<E: Engine>(
  Us: &[R1CSInstance<E>],
  Ws: &[R1CSWitness<E>],
  transcript: &mut E::TE,
) -> Result<
  (
    Vec<R1CSInstance<E>>,
    Vec<R1CSWitness<E>>,
    usize,
    E::Scalar,
    Vec<E::Scalar>,
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
    "NeutronNova NIFS prove for {} instances and padded to {} instances",
    Us.len(),
    n_padded
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
  let rhos = (0..ell_b)
    .map(|_| transcript.squeeze(b"rho"))
    .collect::<Result<Vec<_>, _>>()?;

  Ok((Us, Ws, ell_b, tau, rhos))
}

/// A type that holds the NeutronNova NIFS (Non-Interactive Folding Scheme)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaNIFS<E: Engine> {
  polys: Vec<UniPoly<E::Scalar>>,
}

#[inline(always)]
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

impl<E: Engine> NeutronNovaNIFS<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Computes the evaluations of the sum-check polynomial at 0, 2, and 3
  /// Uses two-level delayed modular reduction (inner + middle levels).
  /// Note: Outer level (over pairs) uses regular field arithmetic since there are few pairs.
  #[inline(always)]
  #[allow(clippy::needless_range_loop)]
  fn prove_helper(
    round: usize,
    (left, right): (usize, usize),
    e: &[E::Scalar],
    Az1: &[E::Scalar],
    Bz1: &[E::Scalar],
    Cz1: &[E::Scalar],
    Az2: &[E::Scalar],
    Bz2: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar) {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    // sanity check sizes
    assert_eq!(e.len(), left + right);
    assert_eq!(Az1.len(), left * right);

    let f = &e[left..];
    let e_left = &e[..left];
    let compute_e0 = round != 0;

    let mut acc_e0 = Acc::<E::Scalar>::default();
    let mut acc_quad = Acc::<E::Scalar>::default();

    for i in 0..right {
      let base = i * left;
      let mut inner_e0 = Acc::<E::Scalar>::default();
      let mut inner_quad = Acc::<E::Scalar>::default();

      if compute_e0 {
        for j in 0..left {
          let k = base + j;
          let inner_val = Az1[k] * Bz1[k] - Cz1[k];
          <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
            &mut inner_e0,
            &e_left[j],
            &inner_val,
          );
          let az_diff = Az2[k] - Az1[k];
          let bz_diff = Bz2[k] - Bz1[k];
          let quad_val = az_diff * bz_diff;
          <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
            &mut inner_quad,
            &e_left[j],
            &quad_val,
          );
        }
      } else {
        for j in 0..left {
          let k = base + j;
          let az_diff = Az2[k] - Az1[k];
          let bz_diff = Bz2[k] - Bz1[k];
          let quad_val = az_diff * bz_diff;
          <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
            &mut inner_quad,
            &e_left[j],
            &quad_val,
          );
        }
      }

      let inner_e0_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_e0);
      let inner_quad_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_quad);

      let f_i = &f[i];
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_e0,
        f_i,
        &inner_e0_red,
      );
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_quad,
        f_i,
        &inner_quad_red,
      );
    }

    (
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_e0),
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad),
    )
  }

  /// Computes the AB-only terms for the equality-weighted NIFS fold-line extension.
  ///
  /// The two input layers define the ordinary NIFS fold line:
  ///
  /// `Az(r) = Az1 + r * (Az2 - Az1)`
  /// `Bz(r) = Bz1 + r * (Bz2 - Bz1)`
  ///
  /// The sum is over the tensor-decomposed constraint-evaluation domain:
  /// `i in [0, right)`, `j in [0, left)`, and `k = i * left + j`.
  /// The equality vector is stored in split form, where `e[j]` is the left
  /// tensor factor and `e[left + i]` is the right tensor factor.
  ///
  /// This helper evaluates the AB part:
  ///
  /// `sum_i sum_j e[left + i] * e[j] * Az(r)[k] * Bz(r)[k]`
  ///
  /// and returns:
  /// - the constant term `e0_ab`, i.e. the AB value at `r = 0`;
  /// - the quadratic coefficient of `r^2`.
  ///
  /// It intentionally omits the `Cz` contribution. Callers that need the full
  /// R1CS term subtract the separately computed/evaluated `Cz` contribution.
  #[inline(always)]
  #[allow(clippy::needless_range_loop)]
  fn compute_tensor_eq_ab_fold_extension_terms(
    (left, right): (usize, usize),
    e: &[E::Scalar],
    Az1: &[E::Scalar],
    Bz1: &[E::Scalar],
    Az2: &[E::Scalar],
    Bz2: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar) {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    let f = &e[left..];
    let e_left = &e[..left];

    let mut acc_e0_ab = Acc::<E::Scalar>::default();
    let mut acc_quad = Acc::<E::Scalar>::default();

    for i in 0..right {
      let base = i * left;
      let mut inner_e0 = Acc::<E::Scalar>::default();
      let mut inner_quad = Acc::<E::Scalar>::default();

      for j in 0..left {
        let k = base + j;
        let ab_val = Az1[k] * Bz1[k];
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut inner_e0,
          &e_left[j],
          &ab_val,
        );
        let az_diff = Az2[k] - Az1[k];
        let bz_diff = Bz2[k] - Bz1[k];
        let quad_val = az_diff * bz_diff;
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut inner_quad,
          &e_left[j],
          &quad_val,
        );
      }

      let inner_e0_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_e0);
      let inner_quad_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_quad);

      let f_i = &f[i];
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_e0_ab,
        f_i,
        &inner_e0_red,
      );
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_quad,
        f_i,
        &inner_quad_red,
      );
    }

    (
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_e0_ab),
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad),
    )
  }

  /// Small-value variant of prove_helper for round 0 (compute_e0=false).
  ///
  /// Uses integer arithmetic for the inner loop: i64 subtraction + i128 multiplication,
  /// then `SmallAccumulator` for `field_mont * i128` accumulation.
  /// Large-value positions (where Az/Bz didn't fit i64) are corrected with field arithmetic.
  ///
  /// Returns only quad_coeff (e0 is always zero for round 0).
  #[inline(always)]
  #[allow(clippy::needless_range_loop)]
  fn prove_helper_small(
    (left, right): (usize, usize),
    e: &[E::Scalar],
    Az1: &[E::Scalar],
    Bz1: &[E::Scalar],
    Az2: &[E::Scalar],
    Bz2: &[E::Scalar],
    Az1_i64: &[i64],
    Bz1_i64: &[i64],
    Az2_i64: &[i64],
    Bz2_i64: &[i64],
    large_positions: &[usize],
  ) -> E::Scalar
  where
    E::Scalar: DelayedReduction<i128>,
  {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    let f = &e[left..];
    let e_left = &e[..left];
    let total = left * right;

    let mut acc_quad = Acc::<E::Scalar>::default();

    for i in 0..right {
      let base = i * left;
      let mut inner_acc = SmallAccumulator::<E::Scalar>::default();

      for j in 0..left {
        let k = base + j;
        let az_diff = Az2_i64[k] as i128 - Az1_i64[k] as i128;
        let bz_diff = Bz2_i64[k] as i128 - Bz1_i64[k] as i128;
        let quad_val = az_diff * bz_diff;
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut inner_acc,
          &e_left[j],
          &quad_val,
        );
      }

      let inner_quad_red = <E::Scalar as DelayedReduction<i128>>::reduce(&inner_acc);
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_quad,
        &f[i],
        &inner_quad_red,
      );
    }

    let mut quad = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad);

    // Correction for large-value positions: add field arithmetic for positions
    // where the i64 path contributed 0 instead of the correct value.
    if !large_positions.is_empty() {
      for &k in large_positions {
        if k >= total {
          continue;
        }
        let i = k / left;
        let j = k % left;
        let az_diff = Az2[k] - Az1[k];
        let bz_diff = Bz2[k] - Bz1[k];
        quad += f[i] * e_left[j] * az_diff * bz_diff;
      }
    }

    quad
  }

  /// Small-value prove_helper for rounds 1+ using cross-product decomposition.
  ///
  /// Instead of working on folded field data, computes (e0_ab, quad) directly from
  /// 4 original i64 layers per prove pair. The folded values are:
  ///   Az_lo[k] = (1-r_0)*a_0[k] + r_0*a_1[k]
  ///   Az_hi[k] = (1-r_0)*a_2[k] + r_0*a_3[k]
  ///
  /// The products decompose as cross-product sums with 3 weight classes:
  ///   c_0_0 = (1-r_0)^2, c_0_1 = (1-r_0)*r_0, c_1_1 = r_0^2
  #[inline(always)]
  #[allow(clippy::needless_range_loop)]
  fn prove_helper_ab_cross(
    (left, right): (usize, usize),
    e: &[E::Scalar],
    a_i64: [&[i64]; 4],
    b_i64: [&[i64]; 4],
    a_field: [&[E::Scalar]; 4],
    b_field: [&[E::Scalar]; 4],
    c00: &E::Scalar,
    c01: &E::Scalar,
    c11: &E::Scalar,
    r0: &E::Scalar,
    large_positions: &[usize],
  ) -> (E::Scalar, E::Scalar)
  where
    E::Scalar: DelayedReduction<i128>,
  {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    let f = &e[left..];
    let e_left = &e[..left];
    let total = left * right;

    let mut acc_e0 = Acc::<E::Scalar>::default();
    let mut acc_quad = Acc::<E::Scalar>::default();

    for i in 0..right {
      let base = i * left;

      // Process e0 cross-product terms (Az_lo * Bz_lo)
      let mut sa_e0_00 = SmallAccumulator::<E::Scalar>::default();
      let mut sa_e0_01 = SmallAccumulator::<E::Scalar>::default();
      let mut sa_e0_11 = SmallAccumulator::<E::Scalar>::default();

      for j in 0..left {
        let k = base + j;
        let field = &e_left[j];
        let (a0, a1) = (a_i64[0][k] as i128, a_i64[1][k] as i128);
        let (b0, b1) = (b_i64[0][k] as i128, b_i64[1][k] as i128);
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut sa_e0_00,
          field,
          &(a0 * b0),
        );
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut sa_e0_01,
          field,
          &(a0 * b1 + a1 * b0),
        );
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut sa_e0_11,
          field,
          &(a1 * b1),
        );
      }

      let e0_inner = *c00 * <E::Scalar as DelayedReduction<i128>>::reduce(&sa_e0_00)
        + *c01 * <E::Scalar as DelayedReduction<i128>>::reduce(&sa_e0_01)
        + *c11 * <E::Scalar as DelayedReduction<i128>>::reduce(&sa_e0_11);
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_e0,
        &f[i],
        &e0_inner,
      );

      // Process quad cross-product terms ((Az_hi-Az_lo) * (Bz_hi-Bz_lo))
      let mut sa_q_00 = SmallAccumulator::<E::Scalar>::default();
      let mut sa_q_01 = SmallAccumulator::<E::Scalar>::default();
      let mut sa_q_11 = SmallAccumulator::<E::Scalar>::default();

      for j in 0..left {
        let k = base + j;
        let field = &e_left[j];
        let (da0, da1) = (
          a_i64[2][k] as i128 - a_i64[0][k] as i128,
          a_i64[3][k] as i128 - a_i64[1][k] as i128,
        );
        let (db0, db1) = (
          b_i64[2][k] as i128 - b_i64[0][k] as i128,
          b_i64[3][k] as i128 - b_i64[1][k] as i128,
        );
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut sa_q_00,
          field,
          &(da0 * db0),
        );
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut sa_q_01,
          field,
          &(da0 * db1 + da1 * db0),
        );
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut sa_q_11,
          field,
          &(da1 * db1),
        );
      }

      let quad_inner = *c00 * <E::Scalar as DelayedReduction<i128>>::reduce(&sa_q_00)
        + *c01 * <E::Scalar as DelayedReduction<i128>>::reduce(&sa_q_01)
        + *c11 * <E::Scalar as DelayedReduction<i128>>::reduce(&sa_q_11);
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_quad,
        &f[i],
        &quad_inner,
      );
    }

    let mut e0 = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_e0);
    let mut quad = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad);

    // Correct for large-value positions
    if !large_positions.is_empty() {
      let one_minus_r0 = E::Scalar::ONE - *r0;
      for &k in large_positions {
        if k >= total {
          continue;
        }
        let i = k / left;
        let j = k % left;
        let ej_fi = e_left[j] * f[i];

        let az_lo = one_minus_r0 * a_field[0][k] + *r0 * a_field[1][k];
        let az_hi = one_minus_r0 * a_field[2][k] + *r0 * a_field[3][k];
        let bz_lo = one_minus_r0 * b_field[0][k] + *r0 * b_field[1][k];
        let bz_hi = one_minus_r0 * b_field[2][k] + *r0 * b_field[3][k];

        e0 += ej_fi * az_lo * bz_lo;
        quad += ej_fi * (az_hi - az_lo) * (bz_hi - bz_lo);
      }
    }

    (e0, quad)
  }

  /// Parallel fold of A/B layer chunks of size 4. Each chunk folds
  /// indices [0,1] into [0] and indices [2,3] into [2] using `r_b`.
  /// The resulting folded vectors are at positions 4j and 4j+2 after this call.
  /// Use `compact_folded_layers` to move them to 2j and 2j+1.
  fn par_fold_ab_chunks(a: &mut [Vec<E::Scalar>], b: &mut [Vec<E::Scalar>], r_b: E::Scalar) {
    a.par_chunks_mut(4)
      .zip(b.par_chunks_mut(4))
      .for_each(|(a_chunk, b_chunk)| {
        {
          let (lo, hi) = a_chunk.split_at_mut(1);
          lo[0]
            .iter_mut()
            .zip(hi[0].iter())
            .for_each(|(l, h)| *l += r_b * (*h - *l));
        }
        {
          let (lo, hi) = a_chunk.split_at_mut(3);
          lo[2]
            .iter_mut()
            .zip(hi[0].iter())
            .for_each(|(l, h)| *l += r_b * (*h - *l));
        }
        {
          let (lo, hi) = b_chunk.split_at_mut(1);
          lo[0]
            .iter_mut()
            .zip(hi[0].iter())
            .for_each(|(l, h)| *l += r_b * (*h - *l));
        }
        {
          let (lo, hi) = b_chunk.split_at_mut(3);
          lo[2]
            .iter_mut()
            .zip(hi[0].iter())
            .for_each(|(l, h)| *l += r_b * (*h - *l));
        }
      });
  }

  /// Compact folded results from positions [4j, 4j+2] (for j in 0..prove_pairs)
  /// down to positions [2j, 2j+1]. Runs serially but only does O(prove_pairs)
  /// swaps of `Vec` handles (pointer swaps, not data copies).
  fn compact_folded_layers(a: &mut [Vec<E::Scalar>], b: &mut [Vec<E::Scalar>], prove_pairs: usize) {
    for j in 0..prove_pairs {
      a.swap(2 * j, 4 * j);
      a.swap(2 * j + 1, 4 * j + 2);
      b.swap(2 * j, 4 * j);
      b.swap(2 * j + 1, 4 * j + 2);
    }
  }

  /// Like `compact_folded_layers` but also handles C layers (for non-i64 path).
  fn compact_folded_layers_abc(
    a: &mut [Vec<E::Scalar>],
    b: &mut [Vec<E::Scalar>],
    c: &mut [Vec<E::Scalar>],
    prove_pairs: usize,
  ) {
    for j in 0..prove_pairs {
      a.swap(2 * j, 4 * j);
      a.swap(2 * j + 1, 4 * j + 2);
      b.swap(2 * j, 4 * j);
      b.swap(2 * j + 1, 4 * j + 2);
      c.swap(2 * j, 4 * j);
      c.swap(2 * j + 1, 4 * j + 2);
    }
  }

  /// ZK version of NeutronNova NIFS prove. This function performs the NIFS folding
  /// rounds while interacting with the multi-round verifier circuit/state to derive
  /// per-round challenges via Fiat-Shamir, and populates the verifier circuit's
  /// NIFS-related public values. It returns:
  /// - the constructed NIFS (list of cubic univariate polynomials),
  /// - the split equality polynomial evaluations E (length left+right),
  /// - the final A/B/C layers after folding (as multilinear tables),
  /// - the final outer claim T_out for the step branch, and
  /// - the sequence of challenges r_b used to fold instances/witnesses.
  pub fn prove(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    Us: Vec<R1CSInstance<E>>,
    Ws: Vec<R1CSWitness<E>>,
    cached_matvec: Option<Vec<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>)>>,
    cached_i64: Option<Vec<(Vec<i64>, Vec<i64>, Vec<i64>)>>,
    large_positions: &[usize],
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
    E::Scalar: DelayedReduction<i128>,
  {
    // Determine padding and NIFS rounds
    let n = Us.len();
    let n_padded = Us.len().next_power_of_two();
    let ell_b = n_padded.log_2();

    info!(
      "NeutronNova NIFS prove for {} instances and padded to {} instances",
      Us.len(),
      n_padded
    );

    let mut Us = Us;
    let mut Ws = Ws;
    if Us.len() < n_padded {
      Us.extend(vec![Us[0].clone(); n_padded - n]);
      Ws.extend(vec![Ws[0].clone(); n_padded - n]);
    }
    for U in Us.iter() {
      transcript.absorb(b"U", U);
    }
    let T = E::Scalar::ZERO;
    transcript.absorb(b"T", &T);

    // Squeeze tau and rhos fresh inside this function (like ZK sum-check APIs)
    let (ell_cons, left, right) = compute_tensor_decomp(S.num_cons);
    let tau = transcript.squeeze(b"tau")?;

    let E_eq = PowPolynomial::split_evals(tau, ell_cons, left, right);

    let mut rhos = Vec::with_capacity(ell_b);
    for _ in 0..ell_b {
      rhos.push(transcript.squeeze(b"rho")?);
    }

    // Build Az, Bz, Cz tables for each (possibly padded) instance

    // Split cached matvec: consume owned triples for cached instances, compute rest
    let mut A_layers: Vec<Vec<E::Scalar>> = Vec::with_capacity(n_padded);
    let mut B_layers: Vec<Vec<E::Scalar>> = Vec::with_capacity(n_padded);
    let mut C_layers: Vec<Vec<E::Scalar>> = Vec::with_capacity(n_padded);

    let n_cached = cached_matvec.as_ref().map_or(0, |c| c.len());
    if let Some(cached) = cached_matvec {
      for (a, b, c) in cached {
        A_layers.push(a);
        B_layers.push(b);
        C_layers.push(c);
      }
    }
    // Compute matvec for any remaining (padded) instances
    for i in n_cached..n_padded {
      let w = &Ws[i].W;
      let x = &Us[i].X;
      let z = build_z::<E>(w, x);
      let (a, b, c) = S.multiply_vec(&z)?;
      A_layers.push(a);
      B_layers.push(b);
      C_layers.push(c);
    }
    // Build i64 layers for small-value NIFS optimization
    let n_i64_cached = cached_i64.as_ref().map_or(0, |c| c.len());
    let mut A_i64_layers: Vec<Vec<i64>> = Vec::with_capacity(n_padded);
    let mut B_i64_layers: Vec<Vec<i64>> = Vec::with_capacity(n_padded);
    let mut C_i64_layers: Vec<Vec<i64>> = Vec::with_capacity(n_padded);
    let has_i64 = cached_i64.is_some();
    if let Some(cached) = cached_i64 {
      for (a, b, c) in cached {
        A_i64_layers.push(a);
        B_i64_layers.push(b);
        C_i64_layers.push(c);
      }
    }
    // For padded instances, convert from field layers.
    // Padded instances are clones of Us[0], so their large positions should be a subset
    // of the global large_positions. We still zero at all global positions for safety.
    for i in n_i64_cached..n_padded {
      if has_i64 {
        let (mut a_i64, a_large) = to_small_vec_or_zero(&A_layers[i]);
        let (mut b_i64, b_large) = to_small_vec_or_zero(&B_layers[i]);
        let (mut c_i64, c_large) = to_small_vec_or_zero(&C_layers[i]);
        // Verify padded instance large positions are covered by global large_positions
        debug_assert!(
          a_large.iter().all(|p| large_positions.contains(p)),
          "padded instance has large position not in global set"
        );
        debug_assert!(
          b_large.iter().all(|p| large_positions.contains(p)),
          "padded instance has large position not in global set"
        );
        debug_assert!(
          c_large.iter().all(|p| large_positions.contains(p)),
          "padded instance has large position not in global set"
        );
        // Zero at global large_positions to maintain the invariant
        for &pos in large_positions {
          a_i64[pos] = 0;
          b_i64[pos] = 0;
          c_i64[pos] = 0;
        }
        A_i64_layers.push(a_i64);
        B_i64_layers.push(b_i64);
        C_i64_layers.push(c_i64);
      }
    }

    // Execute NIFS rounds, generating cubic polynomials and driving r_b via multi-round state

    // Precompute C_val[b] = sum_k E_eq[k] * Cz_b[k] for each instance b.
    // This lets us skip C in fold_abc_pair and prove_helper, computing
    // the C contribution to e0 as a weighted sum of these scalars instead.
    // Uses two-level structure: E[k] = e_left[j] * f[i] where k = i*left + j.
    let c_vals: Vec<E::Scalar> = if has_i64 {
      let e_left = &E_eq[..left];
      let f = &E_eq[left..];

      let mut vals: Vec<E::Scalar> = (0..n_padded)
        .into_par_iter()
        .map(|b| {
          let c_i64 = &C_i64_layers[b];
          type Acc<S> = <S as DelayedReduction<S>>::Accumulator;
          let mut acc = Acc::<E::Scalar>::default();
          #[allow(clippy::needless_range_loop)]
          for i in 0..right {
            let base = i * left;
            let mut inner = SmallAccumulator::<E::Scalar>::default();
            for j in 0..left {
              <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
                &mut inner,
                &e_left[j],
                &(c_i64[base + j] as i128),
              );
            }
            let inner_red = <E::Scalar as DelayedReduction<i128>>::reduce(&inner);
            <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
              &mut acc, &f[i], &inner_red,
            );
          }
          <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc)
        })
        .collect();

      // Correct for large positions where i64 was 0 instead of actual value
      if !large_positions.is_empty() {
        let total = left * right;
        for &k in large_positions {
          if k >= total {
            continue;
          }
          let i = k / left;
          let j = k % left;
          let ej_fi = e_left[j] * f[i];
          for b in 0..n_padded {
            vals[b] += ej_fi * C_layers[b][k];
          }
        }
      }
      vals
    } else {
      vec![]
    };

    let mut polys: Vec<UniPoly<E::Scalar>> = Vec::with_capacity(ell_b);
    let mut r_bs: Vec<E::Scalar> = Vec::with_capacity(ell_b);
    let mut T_cur = E::Scalar::ZERO; // the current target value, starts at 0
    let mut acc_eq = E::Scalar::ONE;
    let mut m = n_padded;

    // Helper closure: build polynomial, process round, extract r_b
    // (factored out since it's identical for standalone and merged rounds)
    macro_rules! finish_round {
      ($t:expr, $e0:expr, $quad_coeff:expr) => {{
        let rho_t = rhos[$t];
        let one_minus_rho = E::Scalar::ONE - rho_t;
        let two_rho_minus_one = rho_t - one_minus_rho;
        let c = $e0 * acc_eq;
        let a = $quad_coeff * acc_eq;
        let rho_t_inv: Option<E::Scalar> = rho_t.invert().into();
        let a_b_c = (T_cur - c * one_minus_rho) * rho_t_inv.ok_or(SpartanError::DivisionByZero)?;
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
        vc.nifs_polys[$t] = [c[0], c[1], c[2], c[3]];

        let chals =
          SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, $t, transcript)?;
        let r_b = chals[0];
        r_bs.push(r_b);

        acc_eq *= (E::Scalar::ONE - r_b) * (E::Scalar::ONE - rho_t) + r_b * rho_t;
        T_cur = poly_t.evaluate(&r_b);
        r_b
      }};
    }

    // Helper closure: fold one A/B pair from src indices to dest index.
    // C layers are NOT folded -- C contribution is handled via precomputed c_vals when has_i64.
    macro_rules! fold_ab_pair {
      ($src_even:expr, $src_odd:expr, $dest:expr, $r_b:expr) => {{
        {
          let even = std::mem::take(&mut A_layers[$src_even]);
          let odd = &A_layers[$src_odd];
          let mut folded = even;
          folded.iter_mut().zip(odd.iter()).for_each(|(l, h)| {
            *l += $r_b * (*h - *l);
          });
          A_layers[$dest] = folded;
        }
        {
          let even = std::mem::take(&mut B_layers[$src_even]);
          let odd = &B_layers[$src_odd];
          let mut folded = even;
          folded.iter_mut().zip(odd.iter()).for_each(|(l, h)| {
            *l += $r_b * (*h - *l);
          });
          B_layers[$dest] = folded;
        }
      }};
    }

    // Full A/B/C fold for fallback (non-i64) path
    macro_rules! fold_abc_pair {
      ($src_even:expr, $src_odd:expr, $dest:expr, $r_b:expr) => {{
        fold_ab_pair!($src_even, $src_odd, $dest, $r_b);
        {
          let even = std::mem::take(&mut C_layers[$src_even]);
          let odd = &C_layers[$src_odd];
          let mut folded = even;
          folded.iter_mut().zip(odd.iter()).for_each(|(l, h)| {
            *l += $r_b * (*h - *l);
          });
          C_layers[$dest] = folded;
        }
      }};
    }

    // Round 0: prove_helper (compute_e0 = false for round 0)
    // Uses small-value integer arithmetic when i64 data is available.
    {
      let pairs = m / 2;
      let (e0, quad_coeff) = if has_i64 {
        // Small-value fast path: i64 subtraction + i128 multiplication
        let quad_coeff = A_layers
          .par_chunks(2)
          .zip(B_layers.par_chunks(2))
          .zip(A_i64_layers.par_chunks(2))
          .zip(B_i64_layers.par_chunks(2))
          .enumerate()
          .map(|(pair_idx, (((pair_a, pair_b), pair_a_i64), pair_b_i64))| {
            let qc = Self::prove_helper_small(
              (left, right),
              &E_eq,
              &pair_a[0],
              &pair_b[0],
              &pair_a[1],
              &pair_b[1],
              &pair_a_i64[0],
              &pair_b_i64[0],
              &pair_a_i64[1],
              &pair_b_i64[1],
              large_positions,
            );
            let w = suffix_weight_full::<E::Scalar>(0, ell_b, pair_idx, &rhos);
            qc * w
          })
          .reduce(|| E::Scalar::ZERO, |a, b| a + b);
        (E::Scalar::ZERO, quad_coeff)
      } else {
        // Standard field arithmetic path
        A_layers
          .par_chunks(2)
          .zip(B_layers.par_chunks(2))
          .zip(C_layers.par_chunks(2))
          .enumerate()
          .map(|(pair_idx, ((pair_a, pair_b), pair_c))| {
            let (e0, quad_coeff) = Self::prove_helper(
              0,
              (left, right),
              &E_eq,
              &pair_a[0],
              &pair_b[0],
              &pair_c[0],
              &pair_a[1],
              &pair_b[1],
            );
            let w = suffix_weight_full::<E::Scalar>(0, ell_b, pair_idx, &rhos);
            (e0 * w, quad_coeff * w)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1),
          )
      };
      let r_b = finish_round!(0, e0, quad_coeff);

      if ell_b == 1 {
        for i in 0..pairs {
          if has_i64 {
            fold_ab_pair!(2 * i, 2 * i + 1, i, r_b);
          } else {
            fold_abc_pair!(2 * i, 2 * i + 1, i, r_b);
          }
        }
        A_layers.truncate(pairs);
        B_layers.truncate(pairs);
        if !has_i64 {
          C_layers.truncate(pairs);
        }
        m = pairs;
      }
    }

    // Rounds 1..ell_b-1: merged fold(prev round) + prove_helper(current round)
    // When has_i64: skip C folds, use AB-only fold extension terms, subtract precomputed c_vals.
    if ell_b > 1 {
      let mut prev_r_b = r_bs[0];

      // Build prefix_coeffs incrementally: eq(v, (r_0, r_1, ...))
      // After round 0, the first challenge r_0 gives prefix = [(1-r_0), r_0]
      let mut prefix_coeffs: Vec<E::Scalar> = if has_i64 {
        let r0 = r_bs[0];
        vec![E::Scalar::ONE - r0, r0]
      } else {
        vec![]
      };

      for t in 1..ell_b {
        let fold_pairs = m / 2;
        let prove_pairs = fold_pairs / 2;
        let mut e0_acc = E::Scalar::ZERO;
        let mut quad_acc = E::Scalar::ZERO;

        if has_i64 {
          let n_prefix = prefix_coeffs.len(); // 2^t

          // Round 1 special case: use SA cross-product from i64 originals
          if t == 1 {
            let r0 = prev_r_b;
            let one_minus_r0 = E::Scalar::ONE - r0;
            let c00 = one_minus_r0 * one_minus_r0;
            let c01 = one_minus_r0 * r0;
            let c11 = r0 * r0;

            // Parallel prove (read-only of A_layers / A_i64_layers etc.)
            let prefix_coeffs_ref = &prefix_coeffs;
            let c_vals_ref = &c_vals;
            let e_eq_ref = &E_eq;
            let rhos_ref = &rhos;
            let a_layers_ref = &A_layers;
            let b_layers_ref = &B_layers;
            let a_i64_ref = &A_i64_layers;
            let b_i64_ref = &B_i64_layers;
            let (e0_sum, qc_sum) = (0..prove_pairs)
              .into_par_iter()
              .map(|j| {
                let (e0_ab, qc) = Self::prove_helper_ab_cross(
                  (left, right),
                  e_eq_ref,
                  [
                    &a_i64_ref[4 * j],
                    &a_i64_ref[4 * j + 1],
                    &a_i64_ref[4 * j + 2],
                    &a_i64_ref[4 * j + 3],
                  ],
                  [
                    &b_i64_ref[4 * j],
                    &b_i64_ref[4 * j + 1],
                    &b_i64_ref[4 * j + 2],
                    &b_i64_ref[4 * j + 3],
                  ],
                  [
                    &a_layers_ref[4 * j],
                    &a_layers_ref[4 * j + 1],
                    &a_layers_ref[4 * j + 2],
                    &a_layers_ref[4 * j + 3],
                  ],
                  [
                    &b_layers_ref[4 * j],
                    &b_layers_ref[4 * j + 1],
                    &b_layers_ref[4 * j + 2],
                    &b_layers_ref[4 * j + 3],
                  ],
                  &c00,
                  &c01,
                  &c11,
                  &r0,
                  large_positions,
                );
                let lo_base = (2 * j) * n_prefix;
                let mut c_val_lo = E::Scalar::ZERO;
                for v in 0..n_prefix {
                  c_val_lo += prefix_coeffs_ref[v] * c_vals_ref[lo_base + v];
                }
                let e0 = e0_ab - c_val_lo;
                let w = suffix_weight_full::<E::Scalar>(t, ell_b, j, rhos_ref);
                (e0 * w, qc * w)
              })
              .reduce(
                || (E::Scalar::ZERO, E::Scalar::ZERO),
                |a, b| (a.0 + b.0, a.1 + b.1),
              );
            e0_acc += e0_sum;
            quad_acc += qc_sum;

            // Parallel fold of all prove_pairs chunks (field arithmetic)
            if prove_pairs > 0 {
              Self::par_fold_ab_chunks(
                &mut A_layers[..4 * prove_pairs],
                &mut B_layers[..4 * prove_pairs],
                prev_r_b,
              );
              // Compact folded results from positions [4j, 4j+2] into [2j, 2j+1]
              Self::compact_folded_layers(&mut A_layers, &mut B_layers, prove_pairs);
            }
            // Tail fold (for odd fold_pairs / non-power-of-two cases)
            for i in (2 * prove_pairs)..fold_pairs {
              fold_ab_pair!(2 * i, 2 * i + 1, i, prev_r_b);
            }
          } else {
            // Rounds 2+: merged parallel fold + prove from field data
            let prefix_coeffs_ref = &prefix_coeffs;
            let c_vals_ref = &c_vals;
            let e_eq_ref = &E_eq;
            let rhos_ref = &rhos;

            let (a_head, _) = A_layers.split_at_mut(4 * prove_pairs);
            let (b_head, _) = B_layers.split_at_mut(4 * prove_pairs);

            let (e0_sum, qc_sum) = a_head
              .par_chunks_mut(4)
              .zip(b_head.par_chunks_mut(4))
              .enumerate()
              .map(|(j, (a_chunk, b_chunk))| {
                // Fold a_chunk[0] += r * (a_chunk[1] - a_chunk[0])
                {
                  let (lo, hi) = a_chunk.split_at_mut(1);
                  lo[0]
                    .iter_mut()
                    .zip(hi[0].iter())
                    .for_each(|(l, h)| *l += prev_r_b * (*h - *l));
                }
                // Fold a_chunk[2] += r * (a_chunk[3] - a_chunk[2])
                {
                  let (lo, hi) = a_chunk.split_at_mut(3);
                  lo[2]
                    .iter_mut()
                    .zip(hi[0].iter())
                    .for_each(|(l, h)| *l += prev_r_b * (*h - *l));
                }
                // Fold b_chunk[0] and b_chunk[2] similarly
                {
                  let (lo, hi) = b_chunk.split_at_mut(1);
                  lo[0]
                    .iter_mut()
                    .zip(hi[0].iter())
                    .for_each(|(l, h)| *l += prev_r_b * (*h - *l));
                }
                {
                  let (lo, hi) = b_chunk.split_at_mut(3);
                  lo[2]
                    .iter_mut()
                    .zip(hi[0].iter())
                    .for_each(|(l, h)| *l += prev_r_b * (*h - *l));
                }
                // Prove from folded positions [0] and [2]
                let (e0_ab, qc) = Self::compute_tensor_eq_ab_fold_extension_terms(
                  (left, right),
                  e_eq_ref,
                  &a_chunk[0],
                  &b_chunk[0],
                  &a_chunk[2],
                  &b_chunk[2],
                );
                let lo_base = (2 * j) * n_prefix;
                let mut c_val_lo = E::Scalar::ZERO;
                for v in 0..n_prefix {
                  c_val_lo += prefix_coeffs_ref[v] * c_vals_ref[lo_base + v];
                }
                let e0 = e0_ab - c_val_lo;
                let w = suffix_weight_full::<E::Scalar>(t, ell_b, j, rhos_ref);
                (e0 * w, qc * w)
              })
              .reduce(
                || (E::Scalar::ZERO, E::Scalar::ZERO),
                |a, b| (a.0 + b.0, a.1 + b.1),
              );
            e0_acc += e0_sum;
            quad_acc += qc_sum;

            // Compact folded results from positions [4j, 4j+2] into [2j, 2j+1]
            Self::compact_folded_layers(&mut A_layers, &mut B_layers, prove_pairs);

            for i in (2 * prove_pairs)..fold_pairs {
              fold_ab_pair!(2 * i, 2 * i + 1, i, prev_r_b);
            }
          }
        } else {
          // Parallel merged fold + prove for the non-i64 path (mirrors the i64 structure above).
          let e_eq_ref = &E_eq;
          let rhos_ref = &rhos;

          let (a_head, _) = A_layers.split_at_mut(4 * prove_pairs);
          let (b_head, _) = B_layers.split_at_mut(4 * prove_pairs);
          let (c_head, _) = C_layers.split_at_mut(4 * prove_pairs);

          let (e0_sum, qc_sum) = a_head
            .par_chunks_mut(4)
            .zip(b_head.par_chunks_mut(4))
            .zip(c_head.par_chunks_mut(4))
            .enumerate()
            .map(|(j, ((a_chunk, b_chunk), c_chunk))| {
              // Fold [0] += r * ([1] - [0]) and [2] += r * ([3] - [2]) for A, B, C
              for chunk in [&mut *a_chunk, &mut *b_chunk, &mut *c_chunk] {
                {
                  let (lo, hi) = chunk.split_at_mut(1);
                  lo[0]
                    .iter_mut()
                    .zip(hi[0].iter())
                    .for_each(|(l, h)| *l += prev_r_b * (*h - *l));
                }
                {
                  let (lo, hi) = chunk.split_at_mut(3);
                  lo[2]
                    .iter_mut()
                    .zip(hi[0].iter())
                    .for_each(|(l, h)| *l += prev_r_b * (*h - *l));
                }
              }
              // Prove from folded positions [0] and [2]
              let (e0, qc) = Self::prove_helper(
                t,
                (left, right),
                e_eq_ref,
                &a_chunk[0],
                &b_chunk[0],
                &c_chunk[0],
                &a_chunk[2],
                &b_chunk[2],
              );
              let w = suffix_weight_full::<E::Scalar>(t, ell_b, j, rhos_ref);
              (e0 * w, qc * w)
            })
            .reduce(
              || (E::Scalar::ZERO, E::Scalar::ZERO),
              |a, b| (a.0 + b.0, a.1 + b.1),
            );
          e0_acc += e0_sum;
          quad_acc += qc_sum;

          // Compact folded results from positions [4j, 4j+2] into [2j, 2j+1]
          Self::compact_folded_layers_abc(&mut A_layers, &mut B_layers, &mut C_layers, prove_pairs);

          for i in (2 * prove_pairs)..fold_pairs {
            fold_abc_pair!(2 * i, 2 * i + 1, i, prev_r_b);
          }
        }

        A_layers.truncate(fold_pairs);
        B_layers.truncate(fold_pairs);
        if !has_i64 {
          C_layers.truncate(fold_pairs);
        }
        m = fold_pairs;
        prev_r_b = finish_round!(t, e0_acc, quad_acc);

        // Extend prefix_coeffs: each c splits into c*(1-r_t) and c*r_t
        if has_i64 {
          let r_t = prev_r_b;
          let one_minus_r_t = E::Scalar::ONE - r_t;
          let old = std::mem::take(&mut prefix_coeffs);
          prefix_coeffs = Vec::with_capacity(old.len() * 2);
          // Concatenate: first old*(1-r_t), then old*r_t.
          // This matches the fold structure where bit t occupies the UPPER half
          // of each group: v = v_old + n_old * b_t.
          for c in &old {
            prefix_coeffs.push(*c * one_minus_r_t);
          }
          for c in &old {
            prefix_coeffs.push(*c * r_t);
          }
        }
      }

      // Final fold: fold remaining A/B layers
      let final_pairs = m / 2;
      if has_i64 && final_pairs > 0 {
        // Parallel fold over pairs (2*final_pairs elements → final_pairs elements)
        A_layers[..2 * final_pairs]
          .par_chunks_mut(2)
          .zip(B_layers[..2 * final_pairs].par_chunks_mut(2))
          .for_each(|(a_chunk, b_chunk)| {
            {
              let (lo, hi) = a_chunk.split_at_mut(1);
              lo[0]
                .iter_mut()
                .zip(hi[0].iter())
                .for_each(|(l, h)| *l += prev_r_b * (*h - *l));
            }
            {
              let (lo, hi) = b_chunk.split_at_mut(1);
              lo[0]
                .iter_mut()
                .zip(hi[0].iter())
                .for_each(|(l, h)| *l += prev_r_b * (*h - *l));
            }
          });
        // After parallel fold, folded results are at positions 2i (i in 0..final_pairs).
        // Compact to positions [0..final_pairs].
        for i in 0..final_pairs {
          A_layers.swap(i, 2 * i);
          B_layers.swap(i, 2 * i);
        }
      } else {
        for i in 0..final_pairs {
          if has_i64 {
            fold_ab_pair!(2 * i, 2 * i + 1, i, prev_r_b);
          } else {
            fold_abc_pair!(2 * i, 2 * i + 1, i, prev_r_b);
          }
        }
      }
      A_layers.truncate(final_pairs);
      B_layers.truncate(final_pairs);
      if !has_i64 {
        C_layers.truncate(final_pairs);
      }
    }

    // Compute final Cz_step from original C_i64 layers using SmallAccumulator
    if has_i64 {
      let final_weights = weights_from_r::<E::Scalar>(&r_bs, n_padded);
      let total = left * right;

      // Parallel across k: for each k, accumulate across all b serially.
      let mut cz_step: Vec<E::Scalar> = (0..total)
        .into_par_iter()
        .map(|k| {
          let mut sa = SmallAccumulator::<E::Scalar>::default();
          for b in 0..n_padded {
            <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
              &mut sa,
              &final_weights[b],
              &(C_i64_layers[b][k] as i128),
            );
          }
          <E::Scalar as DelayedReduction<i128>>::reduce(&sa)
        })
        .collect();

      // Correct large positions with field arithmetic
      if !large_positions.is_empty() {
        for &k in large_positions {
          if k >= total {
            continue;
          }
          let mut val = E::Scalar::ZERO;
          for b in 0..n_padded {
            val += final_weights[b] * C_layers[b][k];
          }
          cz_step[k] = val;
        }
      }

      C_layers = vec![cz_step];
    }
    // T_out = poly_last(r_last) / eq(r_b, rho)
    let acc_eq_inv: Option<E::Scalar> = acc_eq.invert().into();
    let T_out = T_cur * acc_eq_inv.ok_or(SpartanError::DivisionByZero)?;
    vc.t_out_step = T_out;
    vc.eq_rho_at_rb = acc_eq;
    let _ =
      SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, ell_b, transcript)?;

    // Truncate witness W vectors to skip zero rest portion before folding.
    // The rest portion (indices effective_len..) is all zero for step circuits,
    // so the folded result there is also zero. We resize back after folding.
    // Only apply when shared+precommitted > 0 (otherwise truncation would zero everything).
    let effective_len = S.num_shared + S.num_precommitted;
    let use_truncated_fold = effective_len > 0;
    if use_truncated_fold {
      for w in Ws.iter_mut() {
        w.W.truncate(effective_len);
      }
    }

    let (_fold_final_span, fold_final_t) = start_span!("fold_witnesses");
    let mut folded_W = R1CSWitness::fold_multiple(&r_bs, &Ws)?;
    if use_truncated_fold {
      let full_dim = S.num_shared + S.num_precommitted + S.num_rest;
      folded_W.W.resize(full_dim, E::Scalar::ZERO);
    }
    info!(elapsed_ms = %fold_final_t.elapsed().as_millis(), "fold_witnesses");

    // Optimized instance fold: only MSM data rows (shared+precommitted),
    // compute rest rows from folded blind + h (field arithmetic instead of MSM).
    // Fall back to full fold when shared+precommitted=0.
    let (_fold_final_span, fold_final_t) = start_span!("fold_instances");
    let w = weights_from_r::<E::Scalar>(&r_bs, Us.len());
    let d = Us[0].X.len();

    let mut X_acc = vec![E::Scalar::ZERO; d];
    for (i, Ui) in Us.iter().enumerate() {
      let wi = w[i];
      for (j, Uij) in Ui.X.iter().enumerate() {
        X_acc[j] += wi * Uij;
      }
    }

    let comms: Vec<_> = Us.iter().map(|U| U.comm_W.clone()).collect();
    let comm_acc = if use_truncated_fold {
      let num_data_rows = (S.num_shared + S.num_precommitted).div_ceil(DEFAULT_COMMITMENT_WIDTH);
      <E::PCS as FoldingEngineTrait<E>>::fold_commitments_partial(
        &comms,
        &w,
        num_data_rows,
        &folded_W.r_W,
        ck,
      )?
    } else {
      <E::PCS as FoldingEngineTrait<E>>::fold_commitments(&comms, &w)?
    };
    let folded_U = R1CSInstance::<E>::new_unchecked(comm_acc, X_acc)?;
    info!(elapsed_ms = %fold_final_t.elapsed().as_millis(), "fold_instances");

    Ok((
      E_eq,
      std::mem::take(&mut A_layers[0]),
      std::mem::take(&mut B_layers[0]),
      std::mem::take(&mut C_layers[0]),
      folded_W,
      folded_U,
    ))
  }
}

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaProverKey<E: Engine> {
  ck: CommitmentKey<E>,
  S_step: SplitR1CSShape<E>,
  S_core: SplitR1CSShape<E>,
  vk_digest: SpartanDigest, // digest of the verifier's key
  vc_shape: SplitMultiRoundR1CSShape<E>,
  vc_shape_regular: R1CSShape<E>,
  vc_ck: CommitmentKey<E>,
}

/// Prover key for the native small-value step/core path.
#[derive(Serialize, Deserialize)]
#[serde(bound(
  serialize = "Coeff: Serialize",
  deserialize = "Coeff: Deserialize<'de>"
))]
pub struct NeutronNovaSmallProverKey<E: Engine, Coeff> {
  field: NeutronNovaProverKey<E>,
  S_step_small: SplitR1CSShape<E, Coeff>,
  S_core_small: SplitR1CSShape<E, Coeff>,
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
  vc_vk: VerifierKey<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<SpartanDigest>,
}

impl<E: Engine> crate::digest::Digestible for NeutronNovaVerifierKey<E> {
  fn write_bytes<W: Sized + std::io::Write>(&self, w: &mut W) -> Result<(), std::io::Error> {
    use bincode::Options;
    let config = bincode::DefaultOptions::new()
      .with_little_endian()
      .with_fixint_encoding();
    config
      .serialize_into(&mut *w, &self.ck)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    config
      .serialize_into(&mut *w, &self.vk_ee)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // Use fast raw-byte path for the R1CS shapes
    self.S_step.write_bytes(w)?;
    self.S_core.write_bytes(w)?;
    // Serialize remaining small fields with bincode
    config
      .serialize_into(&mut *w, &self.vc_shape)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    config
      .serialize_into(&mut *w, &self.vc_shape_regular)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    config
      .serialize_into(&mut *w, &self.vc_ck)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    config
      .serialize_into(&mut *w, &self.vc_vk)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(())
  }
}

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
  ps_step: Vec<PrecommittedState<E>>,
  ps_core: PrecommittedState<E>,
  /// Cached partial matrix-vector products for shared+precommitted columns per step circuit (deterministic).
  cached_step_matvec: Option<Vec<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>)>>,
  /// Small-value (i64) cache of Az/Bz/Cz for NIFS integer arithmetic.
  /// Large values are stored as 0 and corrected via field arithmetic using `large_positions`.
  cached_step_i64: Option<Vec<(Vec<i64>, Vec<i64>, Vec<i64>)>>,
  /// Positions where ANY instance's Az/Bz/Cz didn't fit i64 (union across all instances).
  /// i64 vectors are zeroed at these positions; correction uses field values.
  large_positions: Vec<usize>,
  /// Public values used when computing cached_step_matvec, for validation in prove.
  /// Non-empty when the matvec cache is active; prove checks that step circuits produce the same values.
  cached_step_public_values: Vec<Vec<E::Scalar>>,
}

/// Holds the proof produced by the NeutronNova folding scheme followed by NeutronNova SNARK
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaZkSNARK<E: Engine> {
  /// Shared commitment stored once (same for all step instances and core).
  comm_W_shared: Option<Commitment<E>>,
  step_instances: Vec<SplitR1CSInstance<E>>,
  core_instance: SplitR1CSInstance<E>,
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
  U_verifier: SplitMultiRoundR1CSInstance<E>,
  nifs: NovaNIFS<E>,
  random_U: RelaxedR1CSInstance<E>,
  relaxed_snark: crate::spartan_relaxed::RelaxedR1CSSpartanProof<E>,
}

struct PrepStepArtifacts<E: Engine> {
  ps_step: Vec<PrecommittedState<E>>,
  ps_core: PrecommittedState<E>,
  cached_step_public_values: Vec<Vec<E::Scalar>>,
}

impl<E: Engine> NeutronNovaZkSNARK<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  fn small_r1cs_shape<C: SmallSpartanCircuit<E, bool, i32>>(
    circuit: &C,
  ) -> Result<SplitR1CSShape<E, i32>, SpartanError> {
    let num_challenges = circuit.num_challenges();
    if num_challenges != 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "small-value NeutronNova setup only supports zero challenges, got {num_challenges}"
        ),
      });
    }

    let mut cs = SmallShapeCS::<i32>::new();
    let shared = circuit
      .shared(&mut cs)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to allocate small shared variables: {e}"),
      })?;
    let num_shared = cs.num_vars();

    let precommitted =
      circuit
        .precommitted(&mut cs, &shared)
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Unable to allocate small precommitted variables: {e}"),
        })?;
    let num_precommitted = cs.num_vars() - num_shared;

    circuit
      .synthesize(&mut cs, &shared, &precommitted, None)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize small rest variables: {e}"),
      })?;

    let num_vars = cs.num_vars();
    let num_rest = num_vars - num_shared - num_precommitted;
    let num_public = cs.num_inputs().saturating_sub(1);
    let num_cons = cs.num_constraints();
    let (A, B, C) = cs.to_matrices();

    SplitR1CSShape::<E, i32>::new_int(
      num_cons,
      num_shared,
      num_precommitted,
      num_rest,
      num_public,
      num_challenges,
      A,
      B,
      C,
    )
  }

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
    E::PCS::precompute_ck(&ck);
    info!(elapsed_ms = %ck_t.elapsed().as_millis(), "commitment_key_generation");

    // Calculate num_rounds_b from num_steps by padding to next power of two
    let (_vc_span, vc_t) = start_span!("verifier_circuit_setup");
    let num_rounds_b = num_steps.next_power_of_two().log_2();

    let num_vars = S_step.num_shared + S_step.num_precommitted + S_step.num_rest;
    let num_rounds_x = usize::try_from(S_step.num_cons.ilog2()).unwrap();
    let num_rounds_y = usize::try_from(num_vars.ilog2()).unwrap() + 1;
    let vc = NeutronNovaVerifierCircuit::<E>::default(num_rounds_b, num_rounds_x, num_rounds_y, 32);
    let (vc_shape, vc_ck, vc_vk) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&vc)?;
    let vc_shape_regular = vc_shape.to_regular_shape();
    info!(elapsed_ms = %vc_t.elapsed().as_millis(), "verifier_circuit_setup");
    // Eagerly init FixedBaseMul table before cloning so both pk/vk get it
    E::PCS::precompute_ck(&vc_ck);
    let vk: NeutronNovaVerifierKey<E> = NeutronNovaVerifierKey {
      ck: ck.clone(),
      S_step: S_step.clone(),
      S_core: S_core.clone(),
      vk_ee,
      vc_shape: vc_shape.clone(),
      vc_shape_regular: vc_shape_regular.clone(),
      vc_ck: vc_ck.clone(),
      vc_vk: vc_vk.clone(),
      digest: OnceCell::new(),
    };

    let vk_digest = vk.digest()?;
    let pk = NeutronNovaProverKey {
      ck,
      S_step,
      S_core,
      vc_shape,
      vc_shape_regular,
      vc_ck,
      vk_digest,
    };

    // Eagerly precompute sparse matrix data for the step and core circuits
    pk.S_step.precompute();
    pk.S_core.precompute();
    vk.S_step.precompute();
    vk.S_core.precompute();
    info!(elapsed_ms = %setup_t.elapsed().as_millis(), "neutronnova_setup");
    Ok((pk, vk))
  }

  /// Sets up NeutronNova with native small-value step/core circuits.
  pub fn setup_small<C1, C2>(
    step_circuit: &C1,
    core_circuit: &C2,
    num_steps: usize,
  ) -> Result<(NeutronNovaSmallProverKey<E, i32>, NeutronNovaVerifierKey<E>), SpartanError>
  where
    C1: SmallSpartanCircuit<E, bool, i32>,
    C2: SmallSpartanCircuit<E, bool, i32>,
  {
    let (_setup_span, setup_t) = start_span!("neutronnova_setup_small");

    let (_r1cs_span, r1cs_t) = start_span!("small_r1cs_shape_generation");
    debug!("Synthesizing small step circuit");
    let mut S_step_small = Self::small_r1cs_shape(step_circuit)?;
    debug!("Finished synthesizing small step circuit");

    debug!("Synthesizing small core circuit");
    let mut S_core_small = Self::small_r1cs_shape(core_circuit)?;
    debug!("Finished synthesizing small core circuit");

    SplitR1CSShape::<E, i32>::equalize_int(&mut S_step_small, &mut S_core_small);
    let S_step = S_step_small.to_field_shape();
    let S_core = S_core_small.to_field_shape();

    info!(
      "Small step circuit's witness sizes: shared = {}, precommitted = {}, rest = {}",
      S_step.num_shared, S_step.num_precommitted, S_step.num_rest
    );
    info!(
      "Small core circuit's witness sizes: shared = {}, precommitted = {}, rest = {}",
      S_core.num_shared, S_core.num_precommitted, S_core.num_rest
    );
    info!(elapsed_ms = %r1cs_t.elapsed().as_millis(), "small_r1cs_shape_generation");

    let (_ck_span, ck_t) = start_span!("commitment_key_generation");
    let (ck, vk_ee) = SplitR1CSShape::commitment_key(&[&S_step, &S_core])?;
    E::PCS::precompute_ck(&ck);
    info!(elapsed_ms = %ck_t.elapsed().as_millis(), "commitment_key_generation");

    let (_vc_span, vc_t) = start_span!("verifier_circuit_setup");
    let num_rounds_b = num_steps.next_power_of_two().log_2();
    let num_vars = S_step.num_shared + S_step.num_precommitted + S_step.num_rest;
    let num_rounds_x = usize::try_from(S_step.num_cons.ilog2()).unwrap();
    let num_rounds_y = usize::try_from(num_vars.ilog2()).unwrap() + 1;
    let vc = NeutronNovaVerifierCircuit::<E>::default(num_rounds_b, num_rounds_x, num_rounds_y, 32);
    let (vc_shape, vc_ck, vc_vk) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&vc)?;
    let vc_shape_regular = vc_shape.to_regular_shape();
    info!(elapsed_ms = %vc_t.elapsed().as_millis(), "verifier_circuit_setup");
    E::PCS::precompute_ck(&vc_ck);

    let vk: NeutronNovaVerifierKey<E> = NeutronNovaVerifierKey {
      ck: ck.clone(),
      S_step: S_step.clone(),
      S_core: S_core.clone(),
      vk_ee,
      vc_shape: vc_shape.clone(),
      vc_shape_regular: vc_shape_regular.clone(),
      vc_ck: vc_ck.clone(),
      vc_vk: vc_vk.clone(),
      digest: OnceCell::new(),
    };

    let vk_digest = vk.digest()?;
    let field = NeutronNovaProverKey {
      ck,
      S_step,
      S_core,
      vc_shape,
      vc_shape_regular,
      vc_ck,
      vk_digest,
    };

    field.S_step.precompute();
    field.S_core.precompute();
    vk.S_step.precompute();
    vk.S_core.precompute();

    let pk = NeutronNovaSmallProverKey {
      field,
      S_step_small,
      S_core_small,
    };

    info!(elapsed_ms = %setup_t.elapsed().as_millis(), "neutronnova_setup_small");
    Ok((pk, vk))
  }

  fn prepare_prep_step_artifacts<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    is_small: bool,
  ) -> Result<PrepStepArtifacts<E>, SpartanError> {
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

    SatisfyingAssignment::precommitted_witness(
      &mut ps,
      &pk.S_core,
      &pk.ck,
      core_circuit,
      is_small,
    )?;
    info!(
      elapsed_ms = %precommit_t.elapsed().as_millis(),
      circuits = step_circuits.len() + 1,
      "generate_precommitted_witnesses"
    );

    let cached_step_public_values = if Self::can_cache_step_matvec(pk) {
      step_circuits
        .iter()
        .map(|c| {
          c.public_values().map_err(|e| SpartanError::SynthesisError {
            reason: format!("Circuit does not provide public IO: {e}"),
          })
        })
        .collect::<Result<Vec<_>, _>>()?
    } else {
      info!(
        "Step circuit has rest_unpadded={} challenges={}, skipping matvec/i64 caching",
        pk.S_step.num_rest_unpadded, pk.S_step.num_challenges
      );
      Vec::new()
    };

    Ok(PrepStepArtifacts {
      ps_step,
      ps_core: ps,
      cached_step_public_values,
    })
  }

  #[inline]
  fn can_cache_step_matvec(pk: &NeutronNovaProverKey<E>) -> bool {
    pk.S_step.num_challenges == 0 && pk.S_step.num_rest_unpadded == 0
  }

  fn build_cached_step_matvec(
    S: &SplitR1CSShape<E>,
    ps_step: &[PrecommittedState<E>],
    step_public_values: &[Vec<E::Scalar>],
  ) -> Result<Vec<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>)>, SpartanError> {
    if ps_step.len() != step_public_values.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "cached matvec needs {} public-value rows, got {}",
          ps_step.len(),
          step_public_values.len()
        ),
      });
    }

    if rayon::current_num_threads() > 1 {
      (0..ps_step.len())
        .into_par_iter()
        .map(|i| {
          let z = build_z::<E>(&ps_step[i].W, &step_public_values[i]);
          S.multiply_vec(&z)
        })
        .collect::<Result<Vec<_>, _>>()
    } else {
      (0..ps_step.len())
        .map(|i| {
          let z = build_z::<E>(&ps_step[i].W, &step_public_values[i]);
          S.multiply_vec(&z)
        })
        .collect::<Result<Vec<_>, _>>()
    }
  }

  fn build_round0_i64_cache(
    matvec: &[(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>)],
  ) -> (Vec<(Vec<i64>, Vec<i64>, Vec<i64>)>, Vec<usize>) {
    let mut all_i64 = Vec::with_capacity(matvec.len());
    let mut large_pos_set = std::collections::BTreeSet::new();
    for (az, bz, cz) in matvec {
      let (az_i64, az_large) = to_small_vec_or_zero(az);
      let (bz_i64, bz_large) = to_small_vec_or_zero(bz);
      let (cz_i64, cz_large) = to_small_vec_or_zero(cz);
      for pos in az_large {
        large_pos_set.insert(pos);
      }
      for pos in bz_large {
        large_pos_set.insert(pos);
      }
      for pos in cz_large {
        large_pos_set.insert(pos);
      }
      all_i64.push((az_i64, bz_i64, cz_i64));
    }

    let lp: Vec<usize> = large_pos_set.into_iter().collect();
    if !matvec.is_empty() {
      info!(
        n_large = lp.len(),
        total = matvec[0].0.len(),
        "i64_conversion_stats"
      );
    }

    if !lp.is_empty() {
      for (az_i64, bz_i64, cz_i64) in &mut all_i64 {
        for &pos in &lp {
          az_i64[pos] = 0;
          bz_i64[pos] = 0;
          cz_i64[pos] = 0;
        }
      }
    }

    (all_i64, lp)
  }

  /// Prepares the pre-processed state for proving
  pub fn prep_prove<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<NeutronNovaPrepZkSNARK<E>, SpartanError> {
    let (_prep_span, prep_t) = start_span!("neutronnova_prep_prove");
    let PrepStepArtifacts {
      ps_step,
      ps_core,
      cached_step_public_values,
    } = Self::prepare_prep_step_artifacts(pk, step_circuits, core_circuit, is_small)?;
    let cached_step_matvec = if Self::can_cache_step_matvec(pk) {
      Some(Self::build_cached_step_matvec(
        &pk.S_step,
        &ps_step,
        &cached_step_public_values,
      )?)
    } else {
      None
    };
    let (cached_step_i64, large_positions) = if let Some(matvec) = cached_step_matvec.as_ref() {
      let (all_i64, large_positions) = Self::build_round0_i64_cache(matvec);
      (Some(all_i64), large_positions)
    } else {
      (None, Vec::new())
    };

    info!(elapsed_ms = %prep_t.elapsed().as_millis(), "neutronnova_prep_prove");
    Ok(NeutronNovaPrepZkSNARK {
      ps_step,
      ps_core,
      cached_step_matvec,
      cached_step_i64,
      large_positions,
      cached_step_public_values,
    })
  }

  /// Prove the folding of a batch of R1CS instances and a core circuit that connects them together.
  /// Takes ownership of `prep_snark` to avoid cloning large witness vectors.
  pub fn prove<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    mut prep_snark: NeutronNovaPrepZkSNARK<E>,
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<(Self, NeutronNovaPrepZkSNARK<E>), SpartanError>
  where
    E::Scalar: DelayedReduction<i128>,
  {
    let (_prove_span, prove_t) = start_span!("neutronnova_prove");

    let (_rerandomize_span, rerandomize_t) = start_span!("rerandomize_prep_state");
    prep_snark
      .ps_core
      .rerandomize_in_place(&pk.ck, &pk.S_core)?;
    let comm_W_shared = prep_snark.ps_core.comm_W_shared.clone();
    let r_W_shared = prep_snark.ps_core.r_W_shared.clone();
    prep_snark.ps_step.par_iter_mut().try_for_each(|ps_i| {
      ps_i.rerandomize_with_shared_in_place(&pk.ck, &pk.S_step, &comm_W_shared, &r_W_shared)
    })?;
    info!(elapsed_ms = %rerandomize_t.elapsed().as_millis(), "rerandomize_prep_state");

    if !prep_snark.cached_step_public_values.is_empty() {
      if prep_snark.cached_step_public_values.len() != step_circuits.len() {
        return Err(SpartanError::InternalError {
          reason: format!(
            "Cached matvec was computed for {} step circuits, but prove received {}",
            prep_snark.cached_step_public_values.len(),
            step_circuits.len()
          ),
        });
      }
      for (i, circuit) in step_circuits.iter().enumerate() {
        let current_pv = circuit
          .public_values()
          .map_err(|e| SpartanError::SynthesisError {
            reason: format!("Circuit does not provide public IO: {e}"),
          })?;
        if prep_snark.cached_step_public_values[i] != current_pv {
          return Err(SpartanError::InternalError {
            reason: format!("Step circuit {i} public values changed between prep_prove and prove"),
          });
        }
      }
    }

    let (_gen_span, gen_t) = start_span!(
      "generate_instances_witnesses",
      step_circuits = step_circuits.len()
    );
    let (res_steps, res_core) = rayon::join(
      || {
        prep_snark
          .ps_step
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
          .collect::<Result<Vec<_>, _>>()
          .map(|pairs| {
            let (instances, witnesses): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
            (instances, witnesses)
          })
      },
      || {
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
          &mut prep_snark.ps_core,
          &pk.S_core,
          &pk.ck,
          core_circuit,
          is_small,
          &mut transcript,
        )
      },
    );

    let ((step_instances, step_witnesses), (core_instance, core_witness)) = (res_steps?, res_core?);
    info!(
      elapsed_ms = %gen_t.elapsed().as_millis(),
      step_circuits = step_circuits.len(),
      "generate_instances_witnesses"
    );

    let (_reg_span, reg_t) = start_span!("convert_to_regular_instances");
    let step_instances_regular = step_instances
      .iter()
      .map(|u| u.to_regular_field_instance())
      .collect::<Result<Vec<_>, _>>()?;
    let core_instance_regular = core_instance.to_regular_field_instance()?;
    info!(elapsed_ms = %reg_t.elapsed().as_millis(), "convert_to_regular_instances");

    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"core_instance", &core_instance_regular);

    let n_padded = step_instances_regular.len().next_power_of_two();
    let num_vars = pk.S_step.num_shared + pk.S_step.num_precommitted + pk.S_step.num_rest;
    let num_rounds_b = n_padded.log_2();
    let num_rounds_x = pk.S_step.num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let mut vc = NeutronNovaVerifierCircuit::<E>::default(
      num_rounds_b,
      num_rounds_x,
      num_rounds_y,
      pk.vc_shape.commitment_width,
    );
    let mut vc_state = SatisfyingAssignment::<E>::initialize_multiround_witness(&pk.vc_shape)?;

    let cached_matvec = prep_snark
      .cached_step_matvec
      .as_ref()
      .map(|v| v.par_iter().cloned().collect::<Vec<_>>());
    let cached_i64 = prep_snark
      .cached_step_i64
      .as_ref()
      .map(|v| v.par_iter().cloned().collect::<Vec<_>>());

    let (_nifs_span, nifs_t) = start_span!("NIFS");
    let (E_eq, Az_step, Bz_step, Cz_step, folded_W, folded_U) = NeutronNovaNIFS::<E>::prove(
      &pk.S_step,
      &pk.ck,
      step_instances_regular,
      step_witnesses,
      cached_matvec,
      cached_i64,
      &prep_snark.large_positions,
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
    let (poly_ABC_step, step_lo_eff, step_hi_eff) =
      pk.S_step.bind_and_prepare_poly_ABC_full(&evals_rx, &r);
    let (poly_ABC_core, core_lo_eff, core_hi_eff) =
      pk.S_core.bind_and_prepare_poly_ABC_full(&evals_rx, &r);
    info!(elapsed_ms = %sparse_t.elapsed().as_millis(), "compute_eval_table_sparse");
    // inner sum-check
    let (_sc2_span, sc2_t) = start_span!("inner_sumcheck_batched");

    debug!("Proving inner sum-check with {} rounds", num_rounds_y);
    debug!(
      "Inner sum-check sizes - poly_ABC_step: {}, poly_ABC_core: {}",
      poly_ABC_step.len(),
      poly_ABC_core.len()
    );

    // Build z vectors for the folded and core instances.
    // Non-zero prefix = w_len + 1 + x_len (witness + u + public inputs).
    let (z_folded_vec, z_folded_lo, z_folded_hi) = {
      let mut v = vec![E::Scalar::ZERO; num_vars * 2];
      let w_len = folded_W.W.len();
      v[..w_len].copy_from_slice(&folded_W.W);
      v[w_len] = E::Scalar::ONE;
      let x_len = folded_U.X.len();
      v[w_len + 1..w_len + 1 + x_len].copy_from_slice(&folded_U.X);
      let last_nz = w_len + 1 + x_len;
      (v, last_nz.min(num_vars), last_nz.saturating_sub(num_vars))
    };
    let (z_core_vec, z_core_lo, z_core_hi) = {
      let mut v = vec![E::Scalar::ZERO; num_vars * 2];
      let w_len = core_witness.W.len();
      v[..w_len].copy_from_slice(&core_witness.W);
      v[w_len] = E::Scalar::ONE;
      let x_len = core_instance_regular.X.len();
      v[w_len + 1..w_len + 1 + x_len].copy_from_slice(&core_instance_regular.X);
      let last_nz = w_len + 1 + x_len;
      (v, last_nz.min(num_vars), last_nz.saturating_sub(num_vars))
    };

    // Use actual X length for hi_eff (num_public in SplitR1CSShape may not include shared inputs)
    let step_hi_eff = step_hi_eff.max(z_folded_hi);
    let core_hi_eff = core_hi_eff.max(z_core_hi);

    let (r_y, evals) = SumcheckProof::<E>::prove_quad_batched_zk(
      &[claim_inner_joint_step, claim_inner_joint_core],
      num_rounds_y,
      &mut MultilinearPolynomial::new_with_halves(poly_ABC_step, step_lo_eff, step_hi_eff),
      &mut MultilinearPolynomial::new_with_halves(poly_ABC_core, core_lo_eff, core_hi_eff),
      &mut MultilinearPolynomial::new_with_halves(z_folded_vec, z_folded_lo, z_folded_hi),
      &mut MultilinearPolynomial::new_with_halves(z_core_vec, z_core_lo, z_core_hi),
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
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).unwrap();
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };
    let eval_X_core = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(core_instance_regular.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).unwrap();
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };
    let inv: Option<E::Scalar> = (E::Scalar::ONE - r_y[0]).invert().into();
    let one_minus_ry0_inv = inv.ok_or(SpartanError::DivisionByZero)?;
    let eval_W_step = (eval_Z_step - r_y[0] * eval_X_step) * one_minus_ry0_inv;
    let eval_W_core = (eval_Z_core - r_y[0] * eval_X_core) * one_minus_ry0_inv;

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

    // Sample fresh random instance/witness for ZK (must be done per-prove to preserve zero-knowledge).
    let (random_U, random_W) = pk
      .vc_shape_regular
      .sample_random_instance_witness(&pk.vc_ck)?;
    let (nifs, folded_W_verifier, folded_u, folded_X) = NovaNIFS::<E>::prove(
      &pk.vc_ck,
      &pk.vc_shape_regular,
      &random_U,
      &random_W,
      &U_verifier_regular,
      &W_verifier,
      &mut transcript,
    )?;

    // Prove satisfiability of the folded VC instance via relaxed R1CS Spartan
    let relaxed_snark = crate::spartan_relaxed::RelaxedR1CSSpartanProof::prove(
      &pk.vc_shape_regular,
      &pk.vc_ck,
      &folded_u,
      &folded_X,
      &folded_W_verifier,
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

    // Extract shared commitment (same for all step instances and core) and strip from instances
    let comm_W_shared = step_instances.first().and_then(|u| u.comm_W_shared.clone());
    let step_instances = step_instances
      .into_iter()
      .map(|mut u| {
        u.comm_W_shared = None;
        u
      })
      .collect::<Vec<_>>();
    let mut core_instance = core_instance;
    core_instance.comm_W_shared = None;

    let result = Self {
      comm_W_shared,
      step_instances,
      core_instance,
      eval_arg,
      U_verifier,
      nifs,
      random_U,
      relaxed_snark,
    };

    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "neutronnova_prove");
    Ok((result, prep_snark))
  }

  /// Verifies the NeutronNovaZkSNARK and returns the public IO from the instances
  pub fn verify(
    &self,
    vk: &NeutronNovaVerifierKey<E>,
    num_instances: usize,
  ) -> Result<(Vec<Vec<E::Scalar>>, Vec<E::Scalar>), SpartanError> {
    let (_verify_span, _verify_t) = start_span!("neutronnova_verify");
    if num_instances == 0 || num_instances != self.step_instances.len() {
      return Err(SpartanError::ProofVerifyError {
        reason: format!(
          "Expected {} instances (non-zero), got {}",
          num_instances,
          self.step_instances.len()
        ),
      });
    }

    // Reconstruct step instances and core instance with the shared commitment
    let step_instances: Vec<SplitR1CSInstance<E>> = self
      .step_instances
      .iter()
      .map(|u| {
        let mut u = u.clone();
        u.comm_W_shared = self.comm_W_shared.clone();
        u
      })
      .collect();
    let mut core_instance = self.core_instance.clone();
    core_instance.comm_W_shared = self.comm_W_shared.clone();

    // validate the step instances
    let (_validate_span, validate_t) =
      start_span!("validate_instances", instances = step_instances.len());
    for (i, u) in step_instances.iter().enumerate() {
      let mut transcript = E::TE::new(b"neutronnova_prove");
      transcript.absorb(b"vk", &vk.digest()?);
      transcript.absorb(
        b"num_circuits",
        &E::Scalar::from(step_instances.len() as u64),
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
    transcript.absorb(b"public_values", &core_instance.public_values.as_slice());

    core_instance.validate(&vk.S_core, &mut transcript)?;
    info!(elapsed_ms = %validate_t.elapsed().as_millis(), instances = step_instances.len(), "validate_instances");

    // shared commitment consistency was enforced at construction -- all step instances share comm_W_shared
    // also verify it matches the core instance
    for u in &step_instances {
      if u.comm_W_shared != core_instance.comm_W_shared {
        return Err(SpartanError::ProofVerifyError {
          reason: "All instances must have the same shared commitment".to_string(),
        });
      }
    }

    let (_convert_span, convert_t) = start_span!("convert_to_regular_verify");
    let mut step_instances_padded = step_instances.clone();
    if step_instances_padded.len() != step_instances_padded.len().next_power_of_two() {
      step_instances_padded.extend(std::iter::repeat_n(
        step_instances_padded[0].clone(),
        step_instances_padded.len().next_power_of_two() - step_instances_padded.len(),
      ));
    }
    let step_instances_regular = step_instances_padded
      .par_iter()
      .map(|u| u.to_regular_field_instance())
      .collect::<Result<Vec<_>, _>>()?;

    let core_instance_regular = core_instance.to_regular_field_instance()?;
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

    // fold_multiple and nifs.verify are independent: overlap them
    let (folded_U_result, folded_U_verifier_result) = rayon::join(
      || R1CSInstance::fold_multiple(&r_b, &step_instances_regular),
      || {
        self
          .nifs
          .verify(&mut transcript, &self.random_U, &U_verifier_regular)
      },
    );
    let folded_U = folded_U_result?;
    let folded_U_verifier = folded_U_verifier_result?;
    self
      .relaxed_snark
      .verify(
        &vk.vc_shape_regular,
        &vk.vc_vk,
        &folded_U_verifier,
        &mut transcript,
      )
      .map_err(|e| SpartanError::ProofVerifyError {
        reason: format!("Relaxed Spartan verify failed: {e}"),
      })?;
    let (_matrix_eval_span, matrix_eval_t) = start_span!("matrix_evaluations");
    let (eval_A_step, eval_B_step, eval_C_step, eval_A_core, eval_B_core, eval_C_core) = {
      let T_x = EqPolynomial::evals_from_points(&r_x);
      let T_y = EqPolynomial::evals_from_points(&r_y);
      let (eval_A_step, eval_B_step, eval_C_step) = vk.S_step.evaluate_with_tables_fast(&T_x, &T_y);
      let (eval_A_core, eval_B_core, eval_C_core) = vk.S_core.evaluate_with_tables_fast(&T_x, &T_y);

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
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).unwrap();
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };
    let eval_X_core = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(core_instance_regular.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).unwrap();
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

    let public_values_step = step_instances
      .iter()
      .take(num_instances)
      .map(|u| u.public_values.clone())
      .collect::<Vec<Vec<_>>>();

    let public_values_core = core_instance.public_values.clone();

    // return a vector of public values
    Ok((public_values_step, public_values_core))
  }
}

#[cfg(test)]
mod tests {
  use super::small_neutronnova_zk::fold_small_value_vectors;
  use super::*;
  use crate::big_num::SmallValueField;
  use crate::gadgets::SmallBit;
  use crate::lagrange_accumulator::build_accumulators_neutronnova;
  use crate::provider::T256HyraxEngine;
  use crate::small_constraint_system::{SmallConstraintSystem, SmallLinearCombination};
  use crate::small_sumcheck::{SmallValueSumCheck, build_univariate_round_polynomial, derive_t1};
  use bellpepper::gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    sha256::sha256,
  };
  use bellpepper_core::{ConstraintSystem, SynthesisError, Variable};
  use core::marker::PhantomData;

  #[derive(Clone, Debug)]
  struct Sha256Circuit<E: Engine> {
    preimage: Vec<u8>,
    _p: PhantomData<E>,
  }

  #[derive(Clone, Debug)]
  struct TinyCubicCircuit<E: Engine> {
    x: u64,
    _p: PhantomData<E>,
  }

  #[derive(Clone, Debug)]
  struct TinySmallBoolCircuit<E: Engine> {
    bit: bool,
    _p: PhantomData<E>,
  }

  impl<E: Engine> TinySmallBoolCircuit<E> {
    fn new(bit: bool) -> Self {
      Self {
        bit,
        _p: PhantomData,
      }
    }
  }

  impl<E: Engine> TinyCubicCircuit<E> {
    fn new(x: u64) -> Self {
      Self { x, _p: PhantomData }
    }
  }

  impl<E: Engine> SpartanCircuit<E> for TinyCubicCircuit<E> {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
      Ok(vec![E::Scalar::ZERO])
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      let x_val = E::Scalar::from(self.x);
      let y_val = x_val * x_val;
      let z_val = y_val * x_val;
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(x_val))?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || Ok(y_val))?;
      let z = AllocatedNum::alloc(cs.namespace(|| "z"), || Ok(z_val))?;

      cs.enforce(
        || "x squared",
        |lc| lc + x.get_variable(),
        |lc| lc + x.get_variable(),
        |lc| lc + y.get_variable(),
      );
      cs.enforce(
        || "x cubed",
        |lc| lc + y.get_variable(),
        |lc| lc + x.get_variable(),
        |lc| lc + z.get_variable(),
      );

      let public = AllocatedNum::alloc(cs.namespace(|| "public zero"), || Ok(E::Scalar::ZERO))?;
      public.inputize(cs.namespace(|| "inputize public zero"))?;

      Ok(vec![x, y, z])
    }

    fn num_challenges(&self) -> usize {
      0
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
      _: &[AllocatedNum<E::Scalar>],
      _: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
      Ok(())
    }
  }

  impl<E: Engine> SmallSpartanCircuit<E, bool, i32> for TinySmallBoolCircuit<E> {
    fn public_values(&self) -> Result<Vec<bool>, SynthesisError> {
      Ok(vec![false])
    }

    fn shared<CS: SmallConstraintSystem<bool, i32>>(
      &self,
      _: &mut CS,
    ) -> Result<Vec<Variable>, SynthesisError> {
      Ok(vec![])
    }

    fn precommitted<CS: SmallConstraintSystem<bool, i32>>(
      &self,
      cs: &mut CS,
      _: &[Variable],
    ) -> Result<Vec<Variable>, SynthesisError> {
      let bit = SmallBit::alloc(cs, Some(self.bit))?;
      let public = cs.alloc_input(|| "public zero", || Ok(false))?;
      cs.enforce(
        || "public is zero",
        SmallLinearCombination::from_variable(public, 1i32),
        SmallLinearCombination::one(1i32),
        SmallLinearCombination::zero(),
      );
      cs.enforce(
        || "dummy zero 0",
        SmallLinearCombination::zero(),
        SmallLinearCombination::one(1i32),
        SmallLinearCombination::zero(),
      );
      cs.enforce(
        || "dummy zero 1",
        SmallLinearCombination::zero(),
        SmallLinearCombination::one(1i32),
        SmallLinearCombination::zero(),
      );
      Ok(vec![bit.get_variable()])
    }

    fn num_challenges(&self) -> usize {
      0
    }

    fn synthesize<CS: SmallConstraintSystem<bool, i32>>(
      &self,
      _: &mut CS,
      _: &[Variable],
      _: &[Variable],
      _: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
      Ok(())
    }
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
  ) where
    E::PCS: FoldingEngineTrait<E>,
    E::Scalar: DelayedReduction<i128>,
  {
    println!(
      "[bench_neutron_inner] name: {name}, num_circuits: {}",
      step_circuits.len()
    );

    let ps = NeutronNovaZkSNARK::<E>::prep_prove(pk, step_circuits, core_circuit, true).unwrap();
    let res = NeutronNovaZkSNARK::prove(pk, step_circuits, core_circuit, ps, true);
    assert!(res.is_ok());

    let (snark, _ps) = res.unwrap();
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

  fn small_to_field_vec<E: Engine>(vals: &[i64]) -> Vec<E::Scalar>
  where
    E::Scalar: SmallValueField<i64>,
  {
    vals
      .iter()
      .copied()
      .map(<E::Scalar as SmallValueField<i64>>::small_to_field)
      .collect()
  }

  fn nifs_round_poly_from_direct_terms<F: ff::PrimeField>(
    rho_t: F,
    acc_eq: F,
    T_cur: F,
    e0: F,
    quad_coeff: F,
  ) -> UniPoly<F> {
    let one_minus_rho = F::ONE - rho_t;
    let two_rho_minus_one = rho_t - one_minus_rho;
    let c = e0 * acc_eq;
    let a = quad_coeff * acc_eq;
    let rho_t_inv = rho_t.invert().unwrap();
    let a_b_c = (T_cur - c * one_minus_rho) * rho_t_inv;
    let b = a_b_c - a - c;

    UniPoly {
      coeffs: vec![
        c * one_minus_rho,
        c * two_rho_minus_one + b * one_minus_rho,
        b * two_rho_minus_one + a * one_minus_rho,
        a * two_rho_minus_one,
      ],
    }
  }

  fn fold_layers_once<F: Field>(layers: &[Vec<F>], r: F) -> Vec<Vec<F>> {
    layers
      .chunks_exact(2)
      .map(|pair| {
        pair[0]
          .iter()
          .zip(&pair[1])
          .map(|(lo, hi)| *lo + r * (*hi - *lo))
          .collect()
      })
      .collect()
  }

  fn run_nifs_sumcheck_polynomial_equivalence_test<E: Engine>(num_instances: usize)
  where
    E::PCS: FoldingEngineTrait<E>,
    E::Scalar: crate::big_num::SmallValueEngine<i64>,
  {
    let n_padded = num_instances.next_power_of_two();
    let ell_b = n_padded.log_2();
    assert!(ell_b > 0);

    let (ell_cons, left, right) = compute_tensor_decomp(16);
    let num_cons = left * right;

    let mut a_small: Vec<Vec<i64>> = (0..num_instances)
      .map(|inst| {
        (0..num_cons)
          .map(|k| ((inst as i64 * 11 + k as i64 * 7) % 9) - 4)
          .collect()
      })
      .collect();
    let mut b_small: Vec<Vec<i64>> = (0..num_instances)
      .map(|inst| {
        (0..num_cons)
          .map(|k| ((inst as i64 * 5 + k as i64 * 3 + 2) % 11) - 5)
          .collect()
      })
      .collect();
    while a_small.len() < n_padded {
      a_small.push(a_small[0].clone());
      b_small.push(b_small[0].clone());
    }

    let c_small: Vec<Vec<i64>> = a_small
      .iter()
      .zip(&b_small)
      .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x * y).collect())
      .collect();

    let mut A_layers = a_small
      .iter()
      .map(|v| small_to_field_vec::<E>(v))
      .collect::<Vec<_>>();
    let mut B_layers = b_small
      .iter()
      .map(|v| small_to_field_vec::<E>(v))
      .collect::<Vec<_>>();
    let mut C_layers = c_small
      .iter()
      .map(|v| small_to_field_vec::<E>(v))
      .collect::<Vec<_>>();

    let tau = E::Scalar::from(42u64);
    let E_eq = PowPolynomial::split_evals(tau, ell_cons, left, right);
    let rhos = (0..ell_b)
      .map(|i| E::Scalar::from((i as u64) + 3))
      .collect::<Vec<_>>();
    let challenges = (0..ell_b)
      .map(|i| E::Scalar::from((2 * i as u64) + 7))
      .collect::<Vec<_>>();

    let mut direct_polys = Vec::with_capacity(ell_b);
    let mut T_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;
    let mut m = n_padded;

    for t in 0..ell_b {
      let pairs = m / 2;
      let (e0, quad_coeff) = A_layers
        .chunks_exact(2)
        .zip(B_layers.chunks_exact(2))
        .zip(C_layers.chunks_exact(2))
        .take(pairs)
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
          );
          let w = suffix_weight_full::<E::Scalar>(t, ell_b, pair_idx, &rhos);
          (e0 * w, quad_coeff * w)
        })
        .fold((E::Scalar::ZERO, E::Scalar::ZERO), |a, b| {
          (a.0 + b.0, a.1 + b.1)
        });

      let poly = nifs_round_poly_from_direct_terms(rhos[t], acc_eq, T_cur, e0, quad_coeff);
      direct_polys.push(poly.clone());

      let r_b = challenges[t];
      acc_eq *= (E::Scalar::ONE - r_b) * (E::Scalar::ONE - rhos[t]) + r_b * rhos[t];
      T_cur = poly.evaluate(&r_b);

      A_layers = fold_layers_once(&A_layers[..m], r_b);
      B_layers = fold_layers_once(&B_layers[..m], r_b);
      C_layers = fold_layers_once(&C_layers[..m], r_b);
      m = pairs;
    }

    let accumulators =
      build_accumulators_neutronnova(&a_small, &b_small, &E_eq, left, right, &rhos, ell_b);
    let mut small_value = SmallValueSumCheck::<E::Scalar, 2>::from_accumulators(accumulators);
    let mut small_polys = Vec::with_capacity(ell_b);
    let mut T_cur_small = E::Scalar::ZERO;

    for (i, rho_i) in rhos.iter().enumerate() {
      let t_all = small_value.eval_t_all_u(i);
      let t0 = t_all.at_zero();
      let t_inf = t_all.at_infinity();
      let li = small_value.eq_round_values(*rho_i);
      let t1 = derive_t1(li.at_zero(), li.at_one(), T_cur_small, t0).unwrap();
      let poly = build_univariate_round_polynomial(&li, t0, t1, t_inf);
      small_polys.push(poly.clone());
      let r_i = challenges[i];
      T_cur_small = poly.evaluate(&r_i);
      small_value.advance(&li, r_i);
    }

    for (round, (direct, small)) in direct_polys.iter().zip(&small_polys).enumerate() {
      assert_eq!(
        direct.coeffs, small.coeffs,
        "round {round} polynomial mismatch for num_instances={num_instances}"
      );
    }

    let r_bs_rev = challenges.iter().rev().copied().collect::<Vec<_>>();
    let eq_evals = EqPolynomial::evals_from_points(&r_bs_rev);
    assert_eq!(eq_evals, weights_from_r::<E::Scalar>(&challenges, n_padded));

    let folded_a = fold_small_value_vectors(&eq_evals, &a_small);
    assert_eq!(folded_a, A_layers[0]);
  }

  #[test]
  fn test_nifs_sumcheck_polynomial_equivalence() {
    type E = T256HyraxEngine;

    for num_instances in [2, 3, 4, 5, 7, 8] {
      run_nifs_sumcheck_polynomial_equivalence_test::<E>(num_instances);
    }
  }

  #[test]
  fn test_baseline_and_accumulator_routes_verify() {
    type E = T256HyraxEngine;

    let num_circuits = 3;
    let proto = TinyCubicCircuit::<E>::new(2);
    let core = TinyCubicCircuit::<E>::new(1);
    let (pk, vk) = NeutronNovaZkSNARK::<E>::setup(&proto, &core, num_circuits).unwrap();
    let circuits = (0..num_circuits)
      .map(|i| TinyCubicCircuit::<E>::new((i + 2) as u64))
      .collect::<Vec<_>>();
    let ell_b = num_circuits.next_power_of_two().log_2();

    let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core, true).unwrap();
    let (snark, _) = NeutronNovaZkSNARK::<E>::prove(&pk, &circuits, &core, prep, true).unwrap();
    snark.verify(&vk, num_circuits).unwrap();

    for l0 in [1, ell_b] {
      let prep = NeutronNovaAccumulatorPrepZkSNARK::<
        E,
        i64,
        <E as Engine>::Scalar,
        PrecommittedState<E>,
      >::prep_prove(&pk, &circuits, &core, l0)
      .unwrap();
      let (snark, _) = prep.prove(&pk, &circuits, &core).unwrap();
      snark.verify(&vk, num_circuits).unwrap();
    }

    let err = NeutronNovaAccumulatorPrepZkSNARK::<
      E,
      i64,
      <E as Engine>::Scalar,
      PrecommittedState<E>,
    >::prep_prove(&pk, &circuits, &core, 0)
    .map(|_| ())
    .unwrap_err();
    assert!(matches!(err, SpartanError::InvalidInputLength { .. }));
  }

  #[test]
  fn test_accumulator_i32_tiny_circuit_verify() {
    type E = T256HyraxEngine;

    let num_circuits = 3;
    let proto = TinyCubicCircuit::<E>::new(2);
    let core = TinyCubicCircuit::<E>::new(1);
    let (pk, vk) = NeutronNovaZkSNARK::<E>::setup(&proto, &core, num_circuits).unwrap();
    let circuits = (0..num_circuits)
      .map(|i| TinyCubicCircuit::<E>::new((i + 2) as u64))
      .collect::<Vec<_>>();

    let prep = NeutronNovaAccumulatorPrepZkSNARK::<
      E,
      i32,
      <E as Engine>::Scalar,
      PrecommittedState<E>,
    >::prep_prove(&pk, &circuits, &core, 1)
    .unwrap();
    let (snark, _) = prep.prove(&pk, &circuits, &core).unwrap();
    snark.verify(&vk, num_circuits).unwrap();
  }

  #[test]
  fn test_small_bool_accumulator_routes_verify_l0_cases() {
    type E = T256HyraxEngine;

    let num_circuits = 3;
    let proto = TinySmallBoolCircuit::<E>::new(false);
    let core = TinySmallBoolCircuit::<E>::new(false);
    let (pk, vk) = NeutronNovaZkSNARK::<E>::setup_small(&proto, &core, num_circuits).unwrap();
    let circuits = (0..num_circuits)
      .map(|i| TinySmallBoolCircuit::<E>::new(i % 2 == 1))
      .collect::<Vec<_>>();
    let ell_b = num_circuits.next_power_of_two().log_2();

    for l0 in [1, ell_b] {
      let prep = NeutronNovaSmallAccumulatorPrepZkSNARK::<E, i64>::prep_prove_small(
        &pk, &circuits, &core, l0,
      )
      .unwrap();
      let (snark, _) = prep.prove_small(&pk, &circuits, &core).unwrap();
      snark.verify(&vk, num_circuits).unwrap();
    }
  }

  #[test]
  fn test_prefix_accumulator_cache_roundtrip() {
    type E = T256HyraxEngine;

    let num_circuits = 17;
    let proto = TinyCubicCircuit::<E>::new(2);
    let core = TinyCubicCircuit::<E>::new(1);
    let (pk, vk) = NeutronNovaZkSNARK::<E>::setup(&proto, &core, num_circuits).unwrap();
    let circuits = (0..num_circuits)
      .map(|i| TinyCubicCircuit::<E>::new((i + 2) as u64))
      .collect::<Vec<_>>();

    for l0 in [3usize, 4usize] {
      let prep = NeutronNovaAccumulatorPrepZkSNARK::<
        E,
        i64,
        <E as Engine>::Scalar,
        PrecommittedState<E>,
      >::prep_prove(&pk, &circuits, &core, l0)
      .unwrap();
      assert_eq!(prep.l0(), l0);

      let (snark, _) = prep.prove(&pk, &circuits, &core).unwrap();
      snark.verify(&vk, num_circuits).unwrap();
    }
  }

  #[test]
  fn test_accumulator_prefix_mle_inputs_error_on_large_row_sum() {
    type E = T256HyraxEngine;

    let num_circuits = 2;
    let proto = TinyCubicCircuit::<E>::new(2);
    let core = TinyCubicCircuit::<E>::new(1);
    let (pk, _) = NeutronNovaZkSNARK::<E>::setup(&proto, &core, num_circuits).unwrap();
    let circuits = vec![
      TinyCubicCircuit::<E>::new(i32::MAX as u64),
      TinyCubicCircuit::<E>::new(i32::MAX as u64),
    ];

    let err = NeutronNovaAccumulatorPrepZkSNARK::<
      E,
      i32,
      <E as Engine>::Scalar,
      PrecommittedState<E>,
    >::prep_prove(&pk, &circuits, &core, 1)
    .unwrap_err();

    match err {
      SpartanError::SmallValueOverflow { context, .. } => {
        assert!(context.contains("accumulator prep"));
      }
      other => panic!("expected SmallValueOverflow, got {other:?}"),
    }
  }

  #[test]
  fn test_baseline_prep_has_no_accumulator_cache_fields() {
    let source = include_str!("neutronnova_zk.rs");
    let start = source.find("pub struct NeutronNovaPrepZkSNARK").unwrap();
    let end = source[start..]
      .find("/// Holds the proof produced by the NeutronNova folding scheme")
      .map(|offset| start + offset)
      .unwrap();
    let baseline_prep = &source[start..end];
    assert!(!baseline_prep.contains("AccumulatorNifsCache"));
    assert!(!baseline_prep.contains("cached_step_ext"));
    assert!(!baseline_prep.contains("cached_step_prefix"));
    assert!(!baseline_prep.contains("cached_step_a_lagrange"));
    assert!(!baseline_prep.contains("cached_step_b_lagrange"));
  }

  #[test]
  fn test_small_neutronnova_zk_has_no_custom_macros() {
    let source = include_str!("small_neutronnova_zk.rs");
    for needle in [
      "macro_rules!",
      "finish_round!",
      "fold_ab_pair!",
      "fold_abc_pair!",
    ] {
      assert!(
        !source.contains(needle),
        "small_neutronnova_zk.rs should not contain {needle}"
      );
    }
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
        );
      }
    }
  }
}
