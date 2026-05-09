// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value accumulator path for NeutronNova's ZK folding scheme.

use super::{
  MultiRoundState, NeutronNovaNIFS, NeutronNovaNIFSOutput, NeutronNovaProverKey,
  NeutronNovaZkSNARK, PrepStepArtifacts, build_z, compute_tensor_decomp, prepare_nifs_inputs,
  suffix_weight_full,
};
use crate::{
  CommitmentKey,
  bellpepper::{
    r1cs::{MultiRoundSpartanWitness, PrecommittedState, SpartanWitness},
    solver::SatisfyingAssignment,
  },
  big_num::{
    DelayedReduction, ExtensionSmallValue, SmallValue, SmallValueField, vec_to_small_for_extension,
  },
  errors::SpartanError,
  lagrange_accumulator::{
    build_accumulators_neutronnova, build_accumulators_neutronnova_preextended,
    extension::{bit_rev_prefix_table, gather_and_extend_prefix},
  },
  math::Math,
  nifs::NovaNIFS,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
    power::PowPolynomial,
    univariate::UniPoly,
  },
  r1cs::{R1CSInstance, R1CSWitness, SplitMultiRoundR1CSShape, SplitR1CSShape, weights_from_r},
  small_sumcheck::{SmallValueSumCheck, build_univariate_round_polynomial, derive_t1},
  start_span,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::SpartanCircuit,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    transcript::TranscriptEngineTrait,
  },
  zk::NeutronNovaVerifierCircuit,
};
use ff::Field;
use num_traits::Zero;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use tracing::{info, info_span};

/// Pre-processed state for the accumulator/l0 NIFS proving path.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "SmallAbc<SV>: Serialize, ExtendedPrefixMleEvals<SV>: Serialize",
  deserialize = "SmallAbc<SV>: Deserialize<'de>, ExtendedPrefixMleEvals<SV>: Deserialize<'de>"
))]
pub struct NeutronNovaAccumulatorPrepZkSNARK<E: Engine, SV> {
  ps_step: Vec<PrecommittedState<E>>,
  ps_core: PrecommittedState<E>,
  small_abc: SmallAbc<SV>,
  extended_mle_evals: Option<ExtendedPrefixMleEvals<SV>>,
  cached_step_public_values: Vec<Vec<E::Scalar>>,
}

impl<E: Engine, SV> NeutronNovaAccumulatorPrepZkSNARK<E, SV> {
  /// Returns the accumulator prefix length used by this prepared state.
  pub fn l0(&self) -> usize {
    self.small_abc.l0
  }
}

impl<E: Engine, SV> NeutronNovaAccumulatorPrepZkSNARK<E, SV>
where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<SV> + Sync,
  SV: ExtensionSmallValue,
{
  fn validate_l0(
    pk: &NeutronNovaProverKey<E>,
    l0: usize,
    ell_b: usize,
  ) -> Result<(), SpartanError> {
    if l0 == 0 || l0 > ell_b {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("accumulator l0 ({}) must be in 1..={}", l0, ell_b),
      });
    }
    if !NeutronNovaZkSNARK::<E>::can_cache_step_matvec(pk) {
      return Err(SpartanError::InvalidInputLength {
        reason: "accumulator prep requires step circuits without rest/challenge columns".into(),
      });
    }
    Ok(())
  }

  fn build_small_abc(
    pk: &NeutronNovaProverKey<E>,
    ps_step: &[PrecommittedState<E>],
    step_public_values: &[Vec<E::Scalar>],
    l0: usize,
  ) -> Result<SmallAbc<SV>, SpartanError> {
    let num_instances = ps_step.len();
    if num_instances == 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "cannot build accumulator cache from empty step batch".into(),
      });
    }
    if step_public_values.len() != num_instances {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "accumulator cache needs {} public-value rows, got {}",
          num_instances,
          step_public_values.len()
        ),
      });
    }

    let num_constraints = pk.S_step.num_cons;
    let rows = (0..num_instances)
      .into_par_iter()
      .map(|idx| {
        let span = info_span!("accumulator_prep", layer = idx);
        let _enter = span.enter();
        let prep_ctx = |matrix: &str, e: SpartanError| match e {
          SpartanError::SmallValueOverflow { value, context } => SpartanError::SmallValueOverflow {
            value,
            context: format!("accumulator prep {matrix} layer={idx}: {context}"),
          },
          other => other,
        };
        let z = build_z::<E>(&ps_step[idx].W, &step_public_values[idx]);
        let (az, bz, cz) = pk.S_step.multiply_vec(&z)?;
        let az = info_span!("matrix", name = "Az")
          .in_scope(|| vec_to_small_for_extension::<E::Scalar, SV, 2>(&az, l0))
          .map_err(|e| prep_ctx("Az", e))?;
        let bz = info_span!("matrix", name = "Bz")
          .in_scope(|| vec_to_small_for_extension::<E::Scalar, SV, 2>(&bz, l0))
          .map_err(|e| prep_ctx("Bz", e))?;
        let cz = info_span!("matrix", name = "Cz")
          .in_scope(|| vec_to_small_for_extension::<E::Scalar, SV, 2>(&cz, l0))
          .map_err(|e| prep_ctx("Cz", e))?;
        Ok::<_, SpartanError>((az, bz, cz))
      })
      .collect::<Result<Vec<_>, _>>()?;

    let mut a = Vec::with_capacity(num_instances * num_constraints);
    let mut b = Vec::with_capacity(num_instances * num_constraints);
    let mut c = Vec::with_capacity(num_instances * num_constraints);
    for (idx, (az, bz, cz)) in rows.into_iter().enumerate() {
      if az.len() != num_constraints || bz.len() != num_constraints || cz.len() != num_constraints {
        return Err(SpartanError::InvalidInputLength {
          reason: format!("accumulator cache row {idx} does not match step constraint count"),
        });
      }
      a.extend(az);
      b.extend(bz);
      c.extend(cz);
    }

    Ok(SmallAbc {
      l0,
      num_instances,
      num_constraints,
      a,
      b,
      c,
    })
  }

  /// Prepares the accumulator/l0 proving state.
  pub fn prep_prove<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    l0: usize,
  ) -> Result<NeutronNovaAccumulatorPrepZkSNARK<E, SV>, SpartanError> {
    let ell_b = step_circuits.len().next_power_of_two().log_2();
    Self::validate_l0(pk, l0, ell_b)?;

    let (_prep_span, prep_t) = start_span!("neutronnova_accumulator_prep_prove");
    let PrepStepArtifacts {
      ps_step,
      ps_core,
      cached_step_public_values,
    } =
      NeutronNovaZkSNARK::<E>::prepare_prep_step_artifacts(pk, step_circuits, core_circuit, true)?;
    let (_cache_span, cache_t) = start_span!("prep_accumulator_nifs_cache", l0 = l0);
    let small_abc = Self::build_small_abc(pk, &ps_step, &cached_step_public_values, l0)?;
    let extended_mle_evals = if l0 == ell_b {
      Some(build_extended_prefix_mle_evals(&small_abc, l0)?)
    } else {
      None
    };
    info!(
      elapsed_ms = %cache_t.elapsed().as_millis(),
      instances = step_circuits.len(),
      l0 = l0,
      "prep_accumulator_nifs_cache"
    );

    info!(elapsed_ms = %prep_t.elapsed().as_millis(), "neutronnova_accumulator_prep_prove");
    Ok(NeutronNovaAccumulatorPrepZkSNARK {
      ps_step,
      ps_core,
      small_abc,
      extended_mle_evals,
      cached_step_public_values,
    })
  }

  /// Proves through the accumulator/l0 NIFS path.
  pub fn prove<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    self,
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
  ) -> Result<(NeutronNovaZkSNARK<E>, Self), SpartanError>
  where
    E::Scalar: DelayedReduction<SV> + DelayedReduction<SV::Product> + DelayedReduction<E::Scalar>,
  {
    let mut prep_snark = self;
    let l0 = prep_snark.small_abc.l0;
    let ell_b = step_circuits.len().next_power_of_two().log_2();
    Self::validate_l0(pk, l0, ell_b)?;
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

    if prep_snark.cached_step_public_values.len() != step_circuits.len() {
      return Err(SpartanError::InternalError {
        reason: format!(
          "Accumulator cache was computed for {} step circuits, but prove received {}",
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
              true,
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
          true,
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
      .map(|u| u.to_regular_instance())
      .collect::<Result<Vec<_>, _>>()?;
    let core_instance_regular = core_instance.to_regular_instance()?;
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

    let (_nifs_span, nifs_t) = start_span!("NIFS");
    let (E_eq, Az_step, Bz_step, Cz_step, folded_W, folded_U) =
      NeutronNovaNIFS::<E>::prove_accumulator_with_l0::<SV>(
        &pk.S_step,
        &pk.ck,
        step_instances_regular,
        step_witnesses,
        &prep_snark.small_abc,
        prep_snark.extended_mle_evals.as_ref(),
        &mut vc,
        &mut vc_state,
        &pk.vc_shape,
        &pk.vc_ck,
        &mut transcript,
        l0,
      )?;
    info!(elapsed_ms = %nifs_t.elapsed().as_millis(), "NIFS");

    let (_tensor_span, tensor_t) = start_span!("compute_tensor_and_poly_tau");
    let (_ell, left, _right) = compute_tensor_decomp(pk.S_step.num_cons);
    let mut E1 = E_eq;
    let E2 = E1.split_off(left);

    let mut poly_tau_left = MultilinearPolynomial::new(E1);
    let poly_tau_right = MultilinearPolynomial::new(E2);

    info!(elapsed_ms = %tensor_t.elapsed().as_millis(), "compute_tensor_and_poly_tau");

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

    let (_sc2_span, sc2_t) = start_span!("inner_sumcheck_batched");
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

    let _ = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &vc,
      outer_start_index + num_rounds_x + 1 + num_rounds_y,
      &mut transcript,
    )?;

    let eval_w_step_commit_round = outer_start_index + num_rounds_x + 1 + num_rounds_y + 1;
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &vc,
      eval_w_step_commit_round,
      &mut transcript,
    )?;

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
    let (nifs, folded_W_verifier, folded_u, folded_X) = NovaNIFS::<E>::prove(
      &pk.vc_ck,
      &pk.vc_shape_regular,
      &random_U,
      &random_W,
      &U_verifier_regular,
      &W_verifier,
      &mut transcript,
    )?;

    let relaxed_snark = crate::spartan_relaxed::RelaxedR1CSSpartanProof::prove(
      &pk.vc_shape_regular,
      &pk.vc_ck,
      &folded_u,
      &folded_X,
      &folded_W_verifier,
      &mut transcript,
    )?;
    let comm_eval_W_step = U_verifier.comm_w_per_round[eval_w_step_commit_round].clone();
    let blind_eval_W_step = vc_state.r_w_per_round[eval_w_step_commit_round].clone();

    let comm_eval_W_core = U_verifier.comm_w_per_round[eval_w_step_commit_round + 1].clone();
    let blind_eval_W_core = vc_state.r_w_per_round[eval_w_step_commit_round + 1].clone();

    let c_eval = transcript.squeeze(b"c_eval")?;

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

    let result = NeutronNovaZkSNARK {
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
}

impl<E: Engine> NeutronNovaNIFS<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  fn prove_neutronnova_small_value_sumcheck<SV, A, B>(
    a_layers: &[A],
    b_layers: &[B],
    preextended_ab: Option<(&[SV], &[SV])>,
    e_eq: &[E::Scalar],
    left: usize,
    right: usize,
    rhos: &[E::Scalar],
    l0: usize,
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut MultiRoundState<E>,
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
    E::Scalar: SmallValueField<SV>
      + DelayedReduction<SV>
      + DelayedReduction<SV::Product>
      + DelayedReduction<E::Scalar>,
    A: AsRef<[SV]> + Sync,
    B: AsRef<[SV]> + Sync,
    SV: SmallValue,
  {
    let ell_b = rhos.len();
    debug_assert!(l0 > 0 && l0 <= ell_b, "l0 must be in 1..=ell_b");

    let mut polys = Vec::with_capacity(l0);
    let mut r_bs = Vec::with_capacity(l0);
    let mut T_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;

    let (_acc_span, acc_t) = start_span!("build_accumulators_neutronnova");
    let accumulators = if let Some((a_ext, b_ext)) = preextended_ab
      && l0 == ell_b
    {
      build_accumulators_neutronnova_preextended(a_ext, b_ext, e_eq, left, right, rhos, ell_b)
    } else {
      build_accumulators_neutronnova(a_layers, b_layers, e_eq, left, right, rhos, l0)
    };
    info!(
      elapsed_ms = %acc_t.elapsed().as_millis(),
      "build_accumulators_neutronnova"
    );

    let mut small_value = SmallValueSumCheck::<E::Scalar, 2>::from_accumulators(accumulators);
    for (i, rho_i) in rhos.iter().take(l0).enumerate() {
      let (_round_span, round_t) = start_span!("nifs_smallvalue_round", round = i);
      let t_all = small_value.eval_t_all_u(i);
      let t0 = t_all.at_zero();
      let t_inf = t_all.at_infinity();
      let li = small_value.eq_round_values(*rho_i);
      let t1 = derive_t1(li.at_zero(), li.at_one(), T_cur, t0)
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      let poly = build_univariate_round_polynomial(&li, t0, t1, t_inf);

      let c = &poly.coeffs;
      vc.nifs_polys[i] = [c[0], c[1], c[2], c[3]];

      let (_vc_span, vc_t) = start_span!("vc_commit");
      let chals =
        SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, i, transcript)?;
      info!(elapsed_ms = %vc_t.elapsed().as_millis(), "vc_commit");
      let r_i = chals[0];

      T_cur = poly.evaluate(&r_i);
      acc_eq *= (E::Scalar::ONE - r_i) * (E::Scalar::ONE - *rho_i) + r_i * *rho_i;
      r_bs.push(r_i);
      polys.push(poly);
      small_value.advance(&li, r_i);

      info!(
        elapsed_ms = %round_t.elapsed().as_millis(),
        round = i,
        "nifs_smallvalue_round"
      );
    }

    Ok((polys, r_bs, T_cur, acc_eq))
  }

  fn prove_accumulator_full_batch<SV>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    Us: &[R1CSInstance<E>],
    Ws: &[R1CSWitness<E>],
    small_abc: &SmallAbc<SV>,
    extended_mle_evals: &ExtendedPrefixMleEvals<SV>,
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut MultiRoundState<E>,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<NeutronNovaNIFSOutput<E>, SpartanError>
  where
    E::Scalar: SmallValueField<SV>
      + DelayedReduction<SV>
      + DelayedReduction<SV::Product>
      + DelayedReduction<E::Scalar>,
    SV: ExtensionSmallValue,
  {
    let (_nifs_total_span, nifs_total_t) = start_span!("nifs_prove");
    let (Us, Ws, ell_b, tau, rhos) = prepare_nifs_inputs::<E>(Us, Ws, transcript)?;
    let n_padded = Us.len();
    if small_abc.l0 != ell_b {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "full-batch accumulator cache was built for l0 {}, but ell_b is {}",
          small_abc.l0, ell_b
        ),
      });
    }
    if small_abc.num_constraints != S.num_cons || extended_mle_evals.num_constraints != S.num_cons {
      return Err(SpartanError::InvalidInputLength {
        reason: "full-batch accumulator cache shape does not match step R1CS shape".into(),
      });
    }

    let (ell_cons, left, right) = compute_tensor_decomp(S.num_cons);
    let E_eq = PowPolynomial::split_evals(tau, ell_cons, left, right);

    let (_matrix_span, matrix_t) =
      start_span!("matrix_vector_multiply_instances", instances = n_padded);
    let preextended_ab = Some((
      extended_mle_evals.a.as_slice(),
      extended_mle_evals.b.as_slice(),
    ));
    let mut a_small = Vec::with_capacity(n_padded);
    let mut b_small = Vec::with_capacity(n_padded);
    let mut c_small = Vec::with_capacity(n_padded);
    for idx in 0..n_padded {
      let row_idx = small_abc.padded_row_idx(idx)?;
      a_small.push(Cow::Borrowed(small_abc.a_row(row_idx)));
      b_small.push(Cow::Borrowed(small_abc.b_row(row_idx)));
      c_small.push(Cow::Borrowed(small_abc.c_row(row_idx)));
    }
    info!(
      elapsed_ms = %matrix_t.elapsed().as_millis(),
      instances = n_padded,
      used_preextended = true,
      "matrix_vector_multiply_instances"
    );

    let (_rounds_span, rounds_t) = start_span!("nifs_folding_rounds", rounds = ell_b);
    let (_polys, r_bs, T_cur, acc_eq) = Self::prove_neutronnova_small_value_sumcheck::<SV, _, _>(
      &a_small,
      &b_small,
      preextended_ab,
      &E_eq,
      left,
      right,
      &rhos,
      ell_b,
      vc,
      vc_state,
      vc_shape,
      vc_ck,
      transcript,
    )?;
    info!(
      elapsed_ms = %rounds_t.elapsed().as_millis(),
      rounds = ell_b,
      "nifs_folding_rounds"
    );

    let (_fold_span, fold_t) = start_span!("nifs_eq_fold");
    let r_bs_rev: Vec<_> = r_bs.iter().rev().copied().collect();
    let eq_evals = EqPolynomial::evals_from_points(&r_bs_rev);

    let (az_folded, (bz_folded, cz_folded)) = rayon::join(
      || fold_small_value_vectors(&eq_evals, &a_small),
      || {
        rayon::join(
          || fold_small_value_vectors(&eq_evals, &b_small),
          || fold_small_value_vectors(&eq_evals, &c_small),
        )
      },
    );
    info!(elapsed_ms = %fold_t.elapsed().as_millis(), "nifs_eq_fold");

    let (folded_W, folded_U) = fold_and_update_vc_field::<E>(
      S, ck, &r_bs, T_cur, acc_eq, &Us, &Ws, ell_b, vc, vc_state, vc_shape, vc_ck, transcript,
    )?;

    info!(elapsed_ms = %nifs_total_t.elapsed().as_millis(), "nifs_prove");
    Ok((E_eq, az_folded, bz_folded, cz_folded, folded_W, folded_U))
  }

  fn continue_neutronnova_field_sumcheck(
    a_layers: &mut Vec<Vec<E::Scalar>>,
    b_layers: &mut Vec<Vec<E::Scalar>>,
    c_layers: &mut Vec<Vec<E::Scalar>>,
    e_eq: &[E::Scalar],
    left: usize,
    right: usize,
    rhos: &[E::Scalar],
    start_round: usize,
    r_bs: &mut Vec<E::Scalar>,
    t_cur: &mut E::Scalar,
    acc_eq: &mut E::Scalar,
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut MultiRoundState<E>,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<(), SpartanError>
  where
    E::Scalar: DelayedReduction<E::Scalar>,
  {
    let ell_b = rhos.len();
    let mut m = a_layers.len();
    if start_round >= ell_b {
      return Ok(());
    }

    {
      let pairs = m / 2;
      let (e0, quad_coeff) = a_layers[..2 * pairs]
        .par_chunks(2)
        .zip(b_layers[..2 * pairs].par_chunks(2))
        .zip(c_layers[..2 * pairs].par_chunks(2))
        .enumerate()
        .map(|(pair_idx, ((pair_a, pair_b), pair_c))| {
          let (e0, quad_coeff) = Self::prove_helper(
            start_round,
            (left, right),
            e_eq,
            &pair_a[0],
            &pair_b[0],
            &pair_c[0],
            &pair_a[1],
            &pair_b[1],
          );
          let w = suffix_weight_full::<E::Scalar>(start_round, ell_b, pair_idx, rhos);
          (e0 * w, quad_coeff * w)
        })
        .reduce(
          || (E::Scalar::ZERO, E::Scalar::ZERO),
          |a, b| (a.0 + b.0, a.1 + b.1),
        );

      let r_b = finish_field_sumcheck_round::<E>(
        start_round,
        e0,
        quad_coeff,
        rhos,
        r_bs,
        t_cur,
        acc_eq,
        vc,
        vc_state,
        vc_shape,
        vc_ck,
        transcript,
      )?;

      if start_round + 1 == ell_b {
        fold_final_abc_pairs::<E>(a_layers, b_layers, c_layers, pairs, r_b);
        a_layers.truncate(pairs);
        b_layers.truncate(pairs);
        c_layers.truncate(pairs);
        return Ok(());
      }
    }

    let mut prev_r_b = *r_bs.last().ok_or(SpartanError::InvalidSumcheckProof)?;

    for t in (start_round + 1)..ell_b {
      let fold_pairs = m / 2;
      let prove_pairs = fold_pairs / 2;
      let mut e0_acc = E::Scalar::ZERO;
      let mut quad_acc = E::Scalar::ZERO;

      if prove_pairs > 0 {
        let (a_head, _) = a_layers.split_at_mut(4 * prove_pairs);
        let (b_head, _) = b_layers.split_at_mut(4 * prove_pairs);
        let (c_head, _) = c_layers.split_at_mut(4 * prove_pairs);

        let e_eq_ref = e_eq;
        let rhos_ref = rhos;
        let (e0_sum, qc_sum) = a_head
          .par_chunks_mut(4)
          .zip(b_head.par_chunks_mut(4))
          .zip(c_head.par_chunks_mut(4))
          .enumerate()
          .map(|(j, ((a_chunk, b_chunk), c_chunk))| {
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

        compact_folded_layers_abc::<E>(a_layers, b_layers, c_layers, prove_pairs);
      }

      for i in (2 * prove_pairs)..fold_pairs {
        fold_abc_pair_into::<E>(a_layers, b_layers, c_layers, 2 * i, 2 * i + 1, i, prev_r_b);
      }

      a_layers.truncate(fold_pairs);
      b_layers.truncate(fold_pairs);
      c_layers.truncate(fold_pairs);
      m = fold_pairs;

      prev_r_b = finish_field_sumcheck_round::<E>(
        t, e0_acc, quad_acc, rhos, r_bs, t_cur, acc_eq, vc, vc_state, vc_shape, vc_ck, transcript,
      )?;
    }

    let final_pairs = m / 2;
    if final_pairs > 0 {
      fold_final_abc_pairs::<E>(a_layers, b_layers, c_layers, final_pairs, prev_r_b);
    }
    a_layers.truncate(final_pairs);
    b_layers.truncate(final_pairs);
    c_layers.truncate(final_pairs);

    Ok(())
  }

  fn prove_accumulator_prefix_small<SV>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    Us: &[R1CSInstance<E>],
    Ws: &[R1CSWitness<E>],
    small_abc: &SmallAbc<SV>,
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut MultiRoundState<E>,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
    l0: usize,
  ) -> Result<NeutronNovaNIFSOutput<E>, SpartanError>
  where
    E::Scalar: SmallValueField<SV>
      + DelayedReduction<SV>
      + DelayedReduction<SV::Product>
      + DelayedReduction<E::Scalar>,
    SV: ExtensionSmallValue,
  {
    let (_nifs_total_span, nifs_total_t) = start_span!("nifs_prove");
    let (Us, Ws, ell_b, tau, rhos) = prepare_nifs_inputs::<E>(Us, Ws, transcript)?;
    debug_assert!(l0 > 0 && l0 < ell_b);
    let n_padded = Us.len();
    if small_abc.l0 != l0 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "prefix accumulator cache was built for l0 {}, but prove requested l0 {}",
          small_abc.l0, l0
        ),
      });
    }
    if small_abc.num_constraints != S.num_cons {
      return Err(SpartanError::InvalidInputLength {
        reason: "prefix accumulator cache shape does not match step R1CS shape".into(),
      });
    }

    let (ell_cons, left, right) = compute_tensor_decomp(S.num_cons);
    let E_eq = PowPolynomial::split_evals(tau, ell_cons, left, right);

    let (_matrix_span, matrix_t) =
      start_span!("matrix_vector_multiply_instances", instances = n_padded);
    let prefix_size = 1usize << l0;
    let (mut a_layers, mut b_layers, mut c_layers, a_small, b_small, c_small): (
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
      Vec<Cow<'_, [SV]>>,
      Vec<Cow<'_, [SV]>>,
      Vec<Cow<'_, [SV]>>,
    ) = {
      let mut a_small = Vec::with_capacity(n_padded);
      let mut b_small = Vec::with_capacity(n_padded);
      let mut c_small = Vec::with_capacity(n_padded);
      for idx in 0..n_padded {
        let row_idx = small_abc.padded_row_idx(idx)?;
        a_small.push(Cow::Borrowed(small_abc.a_row(row_idx)));
        b_small.push(Cow::Borrowed(small_abc.b_row(row_idx)));
        c_small.push(Cow::Borrowed(small_abc.c_row(row_idx)));
      }
      info!(
        elapsed_ms = %matrix_t.elapsed().as_millis(),
        instances = n_padded,
        used_small_cache = true,
        "matrix_vector_multiply_instances"
      );
      (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        a_small,
        b_small,
        c_small,
      )
    };

    let (_rounds_span, rounds_t) = start_span!("nifs_folding_rounds", rounds = l0);
    let (_polys, mut r_bs, mut T_cur, mut acc_eq) =
      Self::prove_neutronnova_small_value_sumcheck::<SV, _, _>(
        &a_small, &b_small, None, &E_eq, left, right, &rhos, l0, vc, vc_state, vc_shape, vc_ck,
        transcript,
      )?;
    info!(
      elapsed_ms = %rounds_t.elapsed().as_millis(),
      rounds = l0,
      "nifs_folding_rounds"
    );

    let (_fold_prefix_span, fold_prefix_t) = start_span!("nifs_prefix_fold", rounds = l0);
    let prefix_weights = weights_from_r::<E::Scalar>(&r_bs, prefix_size);
    if a_layers.is_empty() {
      let (ab_folded, c_folded) = rayon::join(
        || {
          let (_ab_span, ab_t) = start_span!("nifs_prefix_fold_ab_small");
          let out = rayon::join(
            || {
              fold_small_layers_by_prefix::<E::Scalar, SV, _>(
                &prefix_weights,
                &a_small,
                prefix_size,
              )
            },
            || {
              fold_small_layers_by_prefix::<E::Scalar, SV, _>(
                &prefix_weights,
                &b_small,
                prefix_size,
              )
            },
          );
          info!(
            elapsed_ms = %ab_t.elapsed().as_millis(),
            "nifs_prefix_fold_ab_small"
          );
          out
        },
        || {
          let (_c_span, c_t) = start_span!("nifs_prefix_fold_c_small");
          let out =
            fold_small_layers_by_prefix::<E::Scalar, SV, _>(&prefix_weights, &c_small, prefix_size);
          info!(elapsed_ms = %c_t.elapsed().as_millis(), "nifs_prefix_fold_c_small");
          Ok(out)
        },
      );
      let (a_folded, b_folded) = ab_folded;
      a_layers = a_folded;
      b_layers = b_folded;
      c_layers = c_folded?;
    } else {
      let (a_folded, (b_folded, c_folded)) = rayon::join(
        || fold_field_layers_by_prefix::<E>(&prefix_weights, &a_layers, prefix_size),
        || {
          rayon::join(
            || fold_field_layers_by_prefix::<E>(&prefix_weights, &b_layers, prefix_size),
            || fold_field_layers_by_prefix::<E>(&prefix_weights, &c_layers, prefix_size),
          )
        },
      );
      a_layers = a_folded;
      b_layers = b_folded;
      c_layers = c_folded;
    }
    info!(elapsed_ms = %fold_prefix_t.elapsed().as_millis(), "nifs_prefix_fold");

    let (_suffix_span, suffix_t) = start_span!("nifs_suffix_rounds", rounds = ell_b - l0);
    Self::continue_neutronnova_field_sumcheck(
      &mut a_layers,
      &mut b_layers,
      &mut c_layers,
      &E_eq,
      left,
      right,
      &rhos,
      l0,
      &mut r_bs,
      &mut T_cur,
      &mut acc_eq,
      vc,
      vc_state,
      vc_shape,
      vc_ck,
      transcript,
    )?;
    info!(
      elapsed_ms = %suffix_t.elapsed().as_millis(),
      rounds = ell_b - l0,
      "nifs_suffix_rounds"
    );

    let az_folded = a_layers.pop().ok_or(SpartanError::InvalidInputLength {
      reason: "partial-l0 NIFS produced no folded A layer".into(),
    })?;
    let bz_folded = b_layers.pop().ok_or(SpartanError::InvalidInputLength {
      reason: "partial-l0 NIFS produced no folded B layer".into(),
    })?;
    let cz_folded = c_layers.pop().ok_or(SpartanError::InvalidInputLength {
      reason: "partial-l0 NIFS produced no folded C layer".into(),
    })?;

    let (folded_W, folded_U) = fold_and_update_vc_field::<E>(
      S, ck, &r_bs, T_cur, acc_eq, &Us, &Ws, ell_b, vc, vc_state, vc_shape, vc_ck, transcript,
    )?;

    info!(elapsed_ms = %nifs_total_t.elapsed().as_millis(), "nifs_prove");
    Ok((E_eq, az_folded, bz_folded, cz_folded, folded_W, folded_U))
  }

  fn prove_accumulator_with_l0<SV>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    Us: Vec<R1CSInstance<E>>,
    Ws: Vec<R1CSWitness<E>>,
    small_abc: &SmallAbc<SV>,
    extended_mle_evals: Option<&ExtendedPrefixMleEvals<SV>>,
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut MultiRoundState<E>,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
    l0: usize,
  ) -> Result<NeutronNovaNIFSOutput<E>, SpartanError>
  where
    E::Scalar: SmallValueField<SV>
      + DelayedReduction<SV>
      + DelayedReduction<SV::Product>
      + DelayedReduction<E::Scalar>,
    SV: ExtensionSmallValue,
  {
    let ell_b = Us.len().next_power_of_two().log_2();
    if l0 == 0 || l0 > ell_b {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("accumulator l0 ({}) must be in 1..={}", l0, ell_b),
      });
    }
    if small_abc.l0 != l0 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "accumulator prep cache was built for l0 {}, but prove requested l0 {}",
          small_abc.l0, l0
        ),
      });
    }

    if l0 < ell_b {
      Self::prove_accumulator_prefix_small::<SV>(
        S, ck, &Us, &Ws, small_abc, vc, vc_state, vc_shape, vc_ck, transcript, l0,
      )
    } else {
      match extended_mle_evals {
        Some(extended_mle_evals) => Self::prove_accumulator_full_batch::<SV>(
          S,
          ck,
          &Us,
          &Ws,
          small_abc,
          extended_mle_evals,
          vc,
          vc_state,
          vc_shape,
          vc_ck,
          transcript,
        ),
        None => Err(SpartanError::InvalidInputLength {
          reason: "full-batch accumulator prove requires preextended cache".into(),
        }),
      }
    }
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SmallAbc<SV> {
  l0: usize,
  num_instances: usize,
  num_constraints: usize,
  a: Vec<SV>,
  b: Vec<SV>,
  c: Vec<SV>,
}

impl<SV> SmallAbc<SV> {
  fn row<'a>(&'a self, table: &'a [SV], idx: usize) -> &'a [SV] {
    let start = idx * self.num_constraints;
    &table[start..start + self.num_constraints]
  }

  fn padded_row_idx(&self, idx: usize) -> Result<usize, SpartanError> {
    if self.num_instances == 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "accumulator cache has no step-instance rows".into(),
      });
    }
    Ok(if idx < self.num_instances { idx } else { 0 })
  }

  fn a_row(&self, idx: usize) -> &[SV] {
    self.row(&self.a, idx)
  }

  fn b_row(&self, idx: usize) -> &[SV] {
    self.row(&self.b, idx)
  }

  fn c_row(&self, idx: usize) -> &[SV] {
    self.row(&self.c, idx)
  }
}

/// `Az` and `Bz` evaluations extended from `{0,1}^l0` to `U_2^l0`.
///
/// The vectors are constraint-major, with each constraint owning one contiguous
/// slice of length `3^l0`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ExtendedPrefixMleEvals<SV> {
  num_constraints: usize,
  domain_size: usize,
  a: Vec<SV>,
  b: Vec<SV>,
}

fn build_extended_prefix_mle_evals<SV>(
  mle_inputs: &SmallAbc<SV>,
  l0: usize,
) -> Result<ExtendedPrefixMleEvals<SV>, SpartanError>
where
  SV: SmallValue,
{
  let prefix_size = 1usize << l0;
  let ext_size = 3usize.pow(l0 as u32);
  let num_constraints = mle_inputs.num_constraints;
  if mle_inputs.num_instances == 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: "cannot precompute full-batch extension cache for empty step batch".into(),
    });
  }

  let mut a_layers: Vec<&[SV]> = (0..mle_inputs.num_instances)
    .map(|idx| mle_inputs.a_row(idx))
    .collect();
  let mut b_layers: Vec<&[SV]> = (0..mle_inputs.num_instances)
    .map(|idx| mle_inputs.b_row(idx))
    .collect();
  if a_layers.len() < prefix_size {
    let first_a = *a_layers.first().ok_or(SpartanError::InvalidInputLength {
      reason: "cannot pad empty full-batch A cache".into(),
    })?;
    let first_b = *b_layers.first().ok_or(SpartanError::InvalidInputLength {
      reason: "cannot pad empty full-batch B cache".into(),
    })?;
    a_layers.resize(prefix_size, first_a);
    b_layers.resize(prefix_size, first_b);
  }

  let bit_rev = bit_rev_prefix_table(l0);

  let mut a_ext = vec![SV::default(); num_constraints * ext_size];
  let mut b_ext = vec![SV::default(); num_constraints * ext_size];
  if rayon::current_num_threads() <= 1 {
    let mut a_prefix = vec![SV::default(); prefix_size];
    let mut b_prefix = vec![SV::default(); prefix_size];
    let mut a_buf = vec![SV::default(); ext_size];
    let mut a_scratch = vec![SV::default(); ext_size];
    let mut b_buf = vec![SV::default(); ext_size];
    let mut b_scratch = vec![SV::default(); ext_size];
    for idx in 0..num_constraints {
      let a_size = gather_and_extend_prefix(
        &a_layers,
        &bit_rev,
        0,
        idx,
        &mut a_prefix,
        &mut a_buf,
        &mut a_scratch,
      );
      let b_size = gather_and_extend_prefix(
        &b_layers,
        &bit_rev,
        0,
        idx,
        &mut b_prefix,
        &mut b_buf,
        &mut b_scratch,
      );
      debug_assert_eq!(a_size, ext_size);
      debug_assert_eq!(b_size, ext_size);
      let start = idx * ext_size;
      let end = start + ext_size;
      a_ext[start..end].copy_from_slice(&a_buf[..a_size]);
      b_ext[start..end].copy_from_slice(&b_buf[..b_size]);
    }
  } else {
    a_ext
      .par_chunks_mut(ext_size)
      .zip(b_ext.par_chunks_mut(ext_size))
      .enumerate()
      .for_each_init(
        || {
          (
            vec![SV::default(); prefix_size],
            vec![SV::default(); prefix_size],
            vec![SV::default(); ext_size],
            vec![SV::default(); ext_size],
            vec![SV::default(); ext_size],
            vec![SV::default(); ext_size],
          )
        },
        |(a_prefix, b_prefix, a_buf, a_scratch, b_buf, b_scratch), (idx, (a_chunk, b_chunk))| {
          let a_size =
            gather_and_extend_prefix(&a_layers, &bit_rev, 0, idx, a_prefix, a_buf, a_scratch);
          let b_size =
            gather_and_extend_prefix(&b_layers, &bit_rev, 0, idx, b_prefix, b_buf, b_scratch);
          debug_assert_eq!(a_size, ext_size);
          debug_assert_eq!(b_size, ext_size);
          a_chunk.copy_from_slice(&a_buf[..a_size]);
          b_chunk.copy_from_slice(&b_buf[..b_size]);
        },
      );
  }

  Ok(ExtendedPrefixMleEvals {
    num_constraints,
    domain_size: ext_size,
    a: a_ext,
    b: b_ext,
  })
}

pub(super) fn fold_small_value_vectors<F, SV, V>(weights: &[F], vectors: &[V]) -> Vec<F>
where
  F: Field + DelayedReduction<SV>,
  V: AsRef<[SV]> + Sync,
  SV: Send + Sync,
{
  let dim = vectors[0].as_ref().len();
  (0..dim)
    .into_par_iter()
    .map(|j| {
      let mut acc = <F as DelayedReduction<SV>>::Accumulator::zero();
      for (wi, vector) in weights.iter().zip(vectors.iter()) {
        let vector = vector.as_ref();
        <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc, wi, &vector[j]);
      }
      <F as DelayedReduction<SV>>::reduce(&acc)
    })
    .collect()
}

fn fold_field_layers_by_prefix<E: Engine>(
  weights: &[E::Scalar],
  layers: &[Vec<E::Scalar>],
  prefix_size: usize,
) -> Vec<Vec<E::Scalar>>
where
  E::Scalar: DelayedReduction<E::Scalar>,
{
  debug_assert!(prefix_size > 0);
  debug_assert_eq!(layers.len() % prefix_size, 0);
  let suffix_groups = layers.len() / prefix_size;

  (0..suffix_groups)
    .into_par_iter()
    .map(|suffix_idx| {
      let start = suffix_idx * prefix_size;
      let end = start + prefix_size;
      fold_small_value_vectors::<E::Scalar, E::Scalar, _>(weights, &layers[start..end])
    })
    .collect()
}

fn fold_small_layers_by_prefix<F, SV, V>(
  weights: &[F],
  layers: &[V],
  prefix_size: usize,
) -> Vec<Vec<F>>
where
  F: Field + DelayedReduction<SV>,
  V: AsRef<[SV]> + Sync,
  SV: Send + Sync,
{
  debug_assert!(prefix_size > 0);
  debug_assert_eq!(layers.len() % prefix_size, 0);
  let suffix_groups = layers.len() / prefix_size;

  (0..suffix_groups)
    .into_par_iter()
    .map(|suffix_idx| {
      let start = suffix_idx * prefix_size;
      let end = start + prefix_size;
      fold_small_value_vectors::<F, SV, _>(weights, &layers[start..end])
    })
    .collect()
}

fn fold_layer_pair_into<F: Field>(
  layers: &mut [Vec<F>],
  src_even: usize,
  src_odd: usize,
  dest: usize,
  r: F,
) {
  let even = std::mem::take(&mut layers[src_even]);
  let odd = &layers[src_odd];
  let mut folded = even;
  folded
    .iter_mut()
    .zip(odd.iter())
    .for_each(|(lo, hi)| *lo += r * (*hi - *lo));
  layers[dest] = folded;
}

fn fold_abc_pair_into<E: Engine>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  c_layers: &mut [Vec<E::Scalar>],
  src_even: usize,
  src_odd: usize,
  dest: usize,
  r: E::Scalar,
) {
  fold_layer_pair_into(a_layers, src_even, src_odd, dest, r);
  fold_layer_pair_into(b_layers, src_even, src_odd, dest, r);
  fold_layer_pair_into(c_layers, src_even, src_odd, dest, r);
}

fn compact_folded_layers_abc<E: Engine>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  c_layers: &mut [Vec<E::Scalar>],
  prove_pairs: usize,
) {
  for j in 0..prove_pairs {
    a_layers.swap(2 * j, 4 * j);
    a_layers.swap(2 * j + 1, 4 * j + 2);
    b_layers.swap(2 * j, 4 * j);
    b_layers.swap(2 * j + 1, 4 * j + 2);
    c_layers.swap(2 * j, 4 * j);
    c_layers.swap(2 * j + 1, 4 * j + 2);
  }
}

fn fold_final_abc_pairs<E: Engine>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  c_layers: &mut [Vec<E::Scalar>],
  pairs: usize,
  r: E::Scalar,
) {
  a_layers[..2 * pairs]
    .par_chunks_mut(2)
    .zip(b_layers[..2 * pairs].par_chunks_mut(2))
    .zip(c_layers[..2 * pairs].par_chunks_mut(2))
    .for_each(|((a_chunk, b_chunk), c_chunk)| {
      for chunk in [&mut *a_chunk, &mut *b_chunk, &mut *c_chunk] {
        let (lo, hi) = chunk.split_at_mut(1);
        lo[0]
          .iter_mut()
          .zip(hi[0].iter())
          .for_each(|(l, h)| *l += r * (*h - *l));
      }
    });

  for i in 0..pairs {
    a_layers.swap(i, 2 * i);
    b_layers.swap(i, 2 * i);
    c_layers.swap(i, 2 * i);
  }
}

fn finish_field_sumcheck_round<E>(
  round: usize,
  e0: E::Scalar,
  quad_coeff: E::Scalar,
  rhos: &[E::Scalar],
  r_bs: &mut Vec<E::Scalar>,
  t_cur: &mut E::Scalar,
  acc_eq: &mut E::Scalar,
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut MultiRoundState<E>,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
) -> Result<E::Scalar, SpartanError>
where
  E: Engine,
{
  let rho_t = rhos[round];
  let one_minus_rho = E::Scalar::ONE - rho_t;
  let two_rho_minus_one = rho_t - one_minus_rho;
  let c = e0 * *acc_eq;
  let a = quad_coeff * *acc_eq;
  let rho_t_inv: Option<E::Scalar> = rho_t.invert().into();
  let a_b_c = (*t_cur - c * one_minus_rho) * rho_t_inv.ok_or(SpartanError::DivisionByZero)?;
  let b = a_b_c - a - c;
  let new_a = a * two_rho_minus_one;
  let new_b = b * two_rho_minus_one + a * one_minus_rho;
  let new_c = c * two_rho_minus_one + b * one_minus_rho;
  let new_d = c * one_minus_rho;
  let poly_t = UniPoly {
    coeffs: vec![new_d, new_c, new_b, new_a],
  };
  let coeffs = &poly_t.coeffs;
  vc.nifs_polys[round] = [coeffs[0], coeffs[1], coeffs[2], coeffs[3]];

  let chals =
    SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, round, transcript)?;
  let r_b = chals[0];
  r_bs.push(r_b);
  *acc_eq *= (E::Scalar::ONE - r_b) * (E::Scalar::ONE - rho_t) + r_b * rho_t;
  *t_cur = poly_t.evaluate(&r_b);

  Ok(r_b)
}

fn fold_and_update_vc_field<E>(
  _S: &SplitR1CSShape<E>,
  _ck: &CommitmentKey<E>,
  r_bs: &[E::Scalar],
  T_cur: E::Scalar,
  acc_eq: E::Scalar,
  Us: &[R1CSInstance<E>],
  Ws: &[R1CSWitness<E>],
  ell_b: usize,
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut MultiRoundState<E>,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
) -> Result<(R1CSWitness<E>, R1CSInstance<E>), SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
{
  let T_out = T_cur
    * acc_eq
      .invert()
      .into_option()
      .ok_or(SpartanError::DivisionByZero)?;
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
