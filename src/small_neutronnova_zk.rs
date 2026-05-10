// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value accumulator path for NeutronNova's ZK folding scheme.

use super::{
  MultiRoundState, NeutronNovaNIFS, NeutronNovaNIFSOutput, NeutronNovaSmallProverKey,
  NeutronNovaZkSNARK, compute_tensor_decomp, suffix_weight_full,
};
use crate::{
  CommitmentKey, DEFAULT_COMMITMENT_WIDTH, PCS,
  bellpepper::{
    r1cs::{MultiRoundSpartanWitness, SmallPrecommittedState},
    solver::SatisfyingAssignment,
  },
  big_num::{DelayedReduction, ExtensionSmallValue, SmallValue, SmallValueField, WideMul},
  errors::SpartanError,
  lagrange_accumulator::{
    build_accumulators_neutronnova, build_accumulators_neutronnova_from_prefix_ext_workspace,
    build_accumulators_neutronnova_preextended,
    extension::{bit_rev_prefix_table, extend_to_lagrange_domain, gather_and_extend_prefix},
  },
  math::Math,
  nifs::NovaNIFS,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
    power::PowPolynomial,
    univariate::UniPoly,
  },
  r1cs::{
    R1CSInstance, R1CSValue, R1CSWitness, SplitMultiRoundR1CSShape, SplitR1CSInstance,
    SplitR1CSShape, weights_from_r,
  },
  small_constraint_system::{SmallCoeff, SmallSatisfyingAssignment},
  small_sumcheck::{SmallValueSumCheck, build_univariate_round_polynomial, derive_t1},
  start_span,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::SmallSpartanCircuit,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    transcript::TranscriptEngineTrait,
  },
  zk::NeutronNovaVerifierCircuit,
};
use ff::Field;
use num_traits::Zero;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, marker::PhantomData, ops::AddAssign, time::Duration};
use tracing::info;

/// Pre-processed state for the accumulator/l0 NIFS proving path.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "PS: Serialize, W: Serialize, SmallAbc<SV>: Serialize, ExtendedPrefixMleEvals<SV>: Serialize, PrefixWorkspace<SV>: Serialize",
  deserialize = "PS: Deserialize<'de>, W: Deserialize<'de>, SmallAbc<SV>: Deserialize<'de>, ExtendedPrefixMleEvals<SV>: Deserialize<'de>, PrefixWorkspace<SV>: Deserialize<'de>"
))]
pub struct NeutronNovaAccumulatorPrepZkSNARK<E: Engine, SV: WideMul, W, PS> {
  ps_step: Vec<PS>,
  ps_core: PS,
  small_abc: SmallAbc<SV>,
  extended_mle_evals: Option<ExtendedPrefixMleEvals<SV>>,
  prefix_workspace: Option<PrefixWorkspace<SV>>,
  cached_step_public_values: Vec<Vec<W>>,
  cached_core_public_values: Vec<W>,
  #[serde(skip)]
  cached_core_witness_field: Option<Vec<E::Scalar>>,
  #[serde(skip)]
  _p: PhantomData<E>,
}

/// Small-value accumulator prep state for typed native small witnesses/public values.
pub type NeutronNovaSmallAccumulatorPrepZkSNARK<E, SV, W> =
  NeutronNovaAccumulatorPrepZkSNARK<E, SV, W, SmallPrecommittedState<E, W>>;

/// Witness/public-value type supported by the native small accumulator path.
pub trait SmallWitnessValue<E>:
  R1CSValue<E>
  + Copy
  + Default
  + From<bool>
  + Send
  + Sync
  + PartialEq
  + Eq
  + Serialize
  + for<'de> Deserialize<'de>
where
  E: Engine,
{
  /// Additive identity used for padded witness slots.
  fn zero() -> Self {
    Self::from(false)
  }

  /// Multiplicative identity used for the constant-one slot in `z = (W, 1, X)`.
  fn one() -> Self {
    Self::from(true)
  }

  /// Commit to a native small witness vector using the PCS specialization for this type.
  fn commit_witness(
    ck: &CommitmentKey<E>,
    v: &[Self],
    r: &<E::PCS as PCSEngineTrait<E>>::Blind,
  ) -> Result<<E::PCS as PCSEngineTrait<E>>::Commitment, SpartanError>;
}

impl<E: Engine> SmallWitnessValue<E> for bool {
  fn commit_witness(
    ck: &CommitmentKey<E>,
    v: &[Self],
    r: &<E::PCS as PCSEngineTrait<E>>::Blind,
  ) -> Result<<E::PCS as PCSEngineTrait<E>>::Commitment, SpartanError> {
    PCS::<E>::commit_bool(ck, v, r)
  }
}

impl<E: Engine> SmallWitnessValue<E> for i8 {
  fn commit_witness(
    ck: &CommitmentKey<E>,
    v: &[Self],
    r: &<E::PCS as PCSEngineTrait<E>>::Blind,
  ) -> Result<<E::PCS as PCSEngineTrait<E>>::Commitment, SpartanError> {
    PCS::<E>::commit_i8(ck, v, r)
  }
}

fn prepare_nifs_inputs_typed<E, X, W>(
  Us: &[R1CSInstance<E, X>],
  Ws: &[R1CSWitness<E, W>],
  transcript: &mut E::TE,
) -> Result<
  (
    Vec<R1CSInstance<E, X>>,
    Vec<R1CSWitness<E, W>>,
    usize,
    E::Scalar,
    Vec<E::Scalar>,
  ),
  SpartanError,
>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  X: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
  W: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
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

#[inline]
fn build_z_small<W: Copy + From<bool>>(w: &[W], x: &[W]) -> Vec<W> {
  let mut z = Vec::with_capacity(w.len() + 1 + x.len());
  z.extend_from_slice(w);
  z.push(W::from(true));
  z.extend_from_slice(x);
  z
}

fn small_shared_witness<E, W, Coeff, C>(
  S: &SplitR1CSShape<E, Coeff>,
  ck: &CommitmentKey<E>,
  circuit: &C,
) -> Result<SmallPrecommittedState<E, W>, SpartanError>
where
  E: Engine,
  W: SmallWitnessValue<E>,
  C: SmallSpartanCircuit<E, W, Coeff>,
{
  let mut cs = SmallSatisfyingAssignment::<W>::new();
  let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
  let mut W_vec = vec![W::zero(); num_vars];

  let shared = circuit
    .shared(&mut cs)
    .map_err(|e| SpartanError::SynthesisError {
      reason: format!("Unable to allocate small shared variables: {e}"),
    })?;
  if cs.aux_assignment.len() < S.num_shared_unpadded {
    return Err(SpartanError::SynthesisError {
      reason: "Small shared variables are not allocated correctly".to_string(),
    });
  }
  W_vec[..S.num_shared_unpadded].copy_from_slice(&cs.aux_assignment[..S.num_shared_unpadded]);

  let (comm_W_shared, r_W_shared) = if S.num_shared_unpadded > 0 {
    let r_W_shared = PCS::<E>::blind(ck, S.num_shared);
    let comm_W_shared = W::commit_witness(ck, &W_vec[..S.num_shared], &r_W_shared)?;
    (Some(comm_W_shared), Some(r_W_shared))
  } else {
    (None, None)
  };

  Ok(SmallPrecommittedState {
    cs,
    shared,
    precommitted: vec![],
    comm_W_shared,
    r_W_shared,
    comm_W_precommitted: None,
    r_W_precommitted: None,
    W: W_vec,
  })
}

fn small_synthesize_precommitted_witness<E, W, Coeff, C>(
  ps: &mut SmallPrecommittedState<E, W>,
  S: &SplitR1CSShape<E, Coeff>,
  circuit: &C,
) -> Result<(), SpartanError>
where
  E: Engine,
  W: SmallWitnessValue<E>,
  C: SmallSpartanCircuit<E, W, Coeff>,
{
  let (_synth_span, synth_t) = start_span!("small_precommitted_witness_synthesize");
  let precommitted =
    circuit
      .precommitted(&mut ps.cs, &ps.shared)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to allocate small precommitted variables: {e}"),
      })?;
  info!(
    elapsed_ms = %synth_t.elapsed().as_millis(),
    "small_precommitted_witness_synthesize"
  );

  if ps.cs.aux_assignment[S.num_shared_unpadded..].len() < S.num_precommitted_unpadded {
    return Err(SpartanError::SynthesisError {
      reason: "Small precommitted variables are not allocated correctly".to_string(),
    });
  }
  ps.W[S.num_shared..S.num_shared + S.num_precommitted_unpadded].copy_from_slice(
    &ps.cs.aux_assignment
      [S.num_shared_unpadded..S.num_shared_unpadded + S.num_precommitted_unpadded],
  );

  ps.precommitted = precommitted;
  Ok(())
}

fn small_commit_precommitted_witness<E, W, Coeff>(
  ps: &mut SmallPrecommittedState<E, W>,
  S: &SplitR1CSShape<E, Coeff>,
  ck: &CommitmentKey<E>,
) -> Result<(), SpartanError>
where
  E: Engine,
  W: SmallWitnessValue<E>,
{
  let (_commit_span, commit_t) = start_span!("commit_small_witness_precommitted");
  let (comm_W_precommitted, r_W_precommitted) = if S.num_precommitted_unpadded > 0 {
    let r_W_precommitted = PCS::<E>::blind(ck, S.num_precommitted);
    let comm_W_precommitted = W::commit_witness(
      ck,
      &ps.W[S.num_shared..S.num_shared + S.num_precommitted],
      &r_W_precommitted,
    )?;
    (Some(comm_W_precommitted), Some(r_W_precommitted))
  } else {
    (None, None)
  };
  info!(
    elapsed_ms = %commit_t.elapsed().as_millis(),
    "commit_small_witness_precommitted"
  );

  ps.comm_W_precommitted = comm_W_precommitted;
  ps.r_W_precommitted = r_W_precommitted;
  Ok(())
}

fn small_r1cs_instance_and_witness<E, W, Coeff, C>(
  ps: &mut SmallPrecommittedState<E, W>,
  S: &SplitR1CSShape<E, Coeff>,
  ck: &CommitmentKey<E>,
  circuit: &C,
  cached_public_values: Option<&[W]>,
  transcript: &mut E::TE,
) -> Result<(SplitR1CSInstance<E, W>, R1CSWitness<E, W>), SpartanError>
where
  E: Engine,
  W: SmallWitnessValue<E>,
  C: SmallSpartanCircuit<E, W, Coeff>,
{
  if S.num_challenges != 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: format!(
        "small-value witness generation only supports zero challenges, got {}",
        S.num_challenges
      ),
    });
  }

  if let Some(comm_W_shared) = &ps.comm_W_shared {
    transcript.absorb(b"comm_W_shared", comm_W_shared);
  }
  if let Some(comm_W_precommitted) = &ps.comm_W_precommitted {
    transcript.absorb(b"comm_W_precommitted", comm_W_precommitted);
  }

  let challenges = Vec::new();
  let skip_synthesize = S.num_rest_unpadded == 0;
  if !skip_synthesize {
    let prep_aux_len = S.num_shared_unpadded + S.num_precommitted_unpadded;
    ps.cs.aux_assignment.truncate(prep_aux_len);
    ps.cs.input_assignment.truncate(1);
    circuit
      .synthesize(&mut ps.cs, &ps.shared, &ps.precommitted, None)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize small witness: {e}"),
      })?;

    ps.W
      [S.num_shared + S.num_precommitted..S.num_shared + S.num_precommitted + S.num_rest_unpadded]
      .copy_from_slice(
        &ps.cs.aux_assignment[S.num_shared_unpadded + S.num_precommitted_unpadded
          ..S.num_shared_unpadded + S.num_precommitted_unpadded + S.num_rest_unpadded],
      );
  }

  let r_W_rest = PCS::<E>::blind(ck, S.num_rest);
  let comm_W_rest = if S.num_rest_unpadded == 0 {
    W::commit_witness(ck, &vec![W::zero(); S.num_rest], &r_W_rest)?
  } else {
    W::commit_witness(
      ck,
      &ps.W[S.num_shared + S.num_precommitted..S.num_shared + S.num_precommitted + S.num_rest],
      &r_W_rest,
    )?
  };
  transcript.absorb(b"comm_W_rest", &comm_W_rest);

  let public_values = if !skip_synthesize {
    small_public_values_from_assignment(ps, S)?
  } else if let Some(public_values) = cached_public_values {
    public_values.to_vec()
  } else {
    circuit
      .public_values()
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Small circuit does not provide public IO: {e}"),
      })?
  };

  let U = SplitR1CSInstance::<E, W>::new(
    S,
    ps.comm_W_shared.clone(),
    ps.comm_W_precommitted.clone(),
    comm_W_rest,
    public_values,
    challenges,
  )?;

  let mut blinds = Vec::with_capacity(3);
  if let Some(r_W_shared) = &ps.r_W_shared {
    blinds.push(r_W_shared.clone());
  }
  if let Some(r_W_precommitted) = &ps.r_W_precommitted {
    blinds.push(r_W_precommitted.clone());
  }
  blinds.push(r_W_rest);
  let r_W = PCS::<E>::combine_blinds(&blinds)?;

  let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
  let w_vec = std::mem::replace(&mut ps.W, vec![W::zero(); num_vars]);
  ps.W[..S.num_shared_unpadded].copy_from_slice(&ps.cs.aux_assignment[..S.num_shared_unpadded]);
  ps.W[S.num_shared..S.num_shared + S.num_precommitted_unpadded].copy_from_slice(
    &ps.cs.aux_assignment
      [S.num_shared_unpadded..S.num_shared_unpadded + S.num_precommitted_unpadded],
  );

  let W = R1CSWitness::<E, W> {
    W: w_vec,
    r_W,
    is_small: true,
  };

  Ok((U, W))
}

fn small_public_values_from_assignment<E, W, Coeff>(
  ps: &SmallPrecommittedState<E, W>,
  S: &SplitR1CSShape<E, Coeff>,
) -> Result<Vec<W>, SpartanError>
where
  E: Engine,
  W: SmallWitnessValue<E>,
{
  let end = 1 + S.num_public;
  ps.cs
    .input_assignment
    .get(1..end)
    .map(|public_values| public_values.to_vec())
    .ok_or_else(|| SpartanError::SynthesisError {
      reason: format!(
        "Small circuit exposed {} public IO values, expected {}",
        ps.cs.input_assignment.len().saturating_sub(1),
        S.num_public
      ),
    })
}

impl<E: Engine, SV: WideMul, W, PS> NeutronNovaAccumulatorPrepZkSNARK<E, SV, W, PS> {
  /// Returns the accumulator prefix length used by this prepared state.
  pub fn l0(&self) -> usize {
    self.small_abc.l0
  }
}

impl<E: Engine, SV, W> NeutronNovaSmallAccumulatorPrepZkSNARK<E, SV, W>
where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<SV> + Sync,
  SV: ExtensionSmallValue,
  <SV as WideMul>::Output: Copy + Zero + Send + Sync + Serialize + for<'de> Deserialize<'de>,
  W: SmallWitnessValue<E>,
{
  fn validate_l0_small<Coeff>(
    pk: &NeutronNovaSmallProverKey<E, Coeff>,
    l0: usize,
    ell_b: usize,
  ) -> Result<(), SpartanError> {
    if l0 == 0 || l0 > ell_b {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("accumulator l0 ({}) must be in 1..={}", l0, ell_b),
      });
    }
    if pk.S_step_small.num_challenges != 0 || pk.S_step_small.num_rest_unpadded != 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "small accumulator prep requires step circuits without rest/challenge columns"
          .into(),
      });
    }
    Ok(())
  }

  fn build_small_abc_small<Coeff>(
    pk: &NeutronNovaSmallProverKey<E, Coeff>,
    ps_step: &[SmallPrecommittedState<E, W>],
    step_public_values: &[Vec<W>],
    l0: usize,
  ) -> Result<SmallAbc<SV>, SpartanError>
  where
    Coeff: SmallCoeff + WideMul<W, Output = SV>,
    W: Copy + Sync,
    SV: Copy + Default + AddAssign + Send,
  {
    let num_instances = ps_step.len();
    if num_instances == 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "cannot build accumulator cache from empty step batch".into(),
      });
    }
    if step_public_values.len() != num_instances {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "small accumulator cache needs {} public-value rows, got {}",
          num_instances,
          step_public_values.len()
        ),
      });
    }

    let num_constraints = pk.S_step_small.num_cons;
    let rows = (0..num_instances)
      .into_par_iter()
      .map(|idx| {
        let z = build_z_small(&ps_step[idx].W, &step_public_values[idx]);
        pk.S_step_small.multiply_vec_small(&z)
      })
      .collect::<Result<Vec<_>, _>>()?;

    let mut a = Vec::with_capacity(num_instances * num_constraints);
    let mut b = Vec::with_capacity(num_instances * num_constraints);
    let mut c = Vec::with_capacity(num_instances * num_constraints);
    for (idx, (az, bz, cz)) in rows.into_iter().enumerate() {
      if az.len() != num_constraints || bz.len() != num_constraints || cz.len() != num_constraints {
        return Err(SpartanError::InvalidInputLength {
          reason: format!("small accumulator cache row {idx} does not match step constraint count"),
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

  /// Prepares the native small-value accumulator proving state.
  pub fn prep_prove_small<C1, C2, Coeff>(
    pk: &NeutronNovaSmallProverKey<E, Coeff>,
    step_circuits: &[C1],
    core_circuit: &C2,
    l0: usize,
  ) -> Result<Self, SpartanError>
  where
    Coeff: SmallCoeff + WideMul<W, Output = SV>,
    W: Copy + Sync,
    SV: Copy + Default + AddAssign + Send,
    C1: SmallSpartanCircuit<E, W, Coeff>,
    C2: SmallSpartanCircuit<E, W, Coeff>,
  {
    let ell_b = step_circuits.len().next_power_of_two().log_2();
    Self::validate_l0_small(pk, l0, ell_b)?;

    let (_prep_span, prep_t) = start_span!("neutronnova_small_accumulator_prep_prove");
    let (_shared_span, shared_t) = start_span!("generate_small_shared_witness");
    let mut ps = small_shared_witness(&pk.S_step_small, &pk.field.ck, &step_circuits[0])?;
    info!(elapsed_ms = %shared_t.elapsed().as_millis(), "generate_small_shared_witness");

    let (_precommit_span, precommit_t) = start_span!(
      "generate_small_precommitted_witnesses",
      circuits = step_circuits.len() + 1
    );
    let (_witness_span, witness_t) = start_span!("prep_witness_generation");
    let (_step_synth_span, step_synth_t) = start_span!(
      "prep_step_witness_synthesis",
      circuits = step_circuits.len()
    );
    let mut ps_step = (0..step_circuits.len())
      .into_par_iter()
      .map(|i| {
        let mut ps_i = ps.clone();
        small_synthesize_precommitted_witness(&mut ps_i, &pk.S_step_small, &step_circuits[i])?;
        Ok(ps_i)
      })
      .collect::<Result<Vec<_>, _>>()?;
    info!(
      elapsed_ms = %step_synth_t.elapsed().as_millis(),
      circuits = step_circuits.len(),
      "prep_step_witness_synthesis"
    );

    let (_core_synth_span, core_synth_t) = start_span!("prep_core_witness_synthesis");
    small_synthesize_precommitted_witness(&mut ps, &pk.S_core_small, core_circuit)?;
    info!(
      elapsed_ms = %core_synth_t.elapsed().as_millis(),
      "prep_core_witness_synthesis"
    );
    info!(elapsed_ms = %witness_t.elapsed().as_millis(), "prep_witness_generation");

    let (_commit_span, commit_t) = start_span!("prep_witness_commit");
    let (_step_commit_span, step_commit_t) =
      start_span!("prep_step_witness_commit", circuits = step_circuits.len());
    ps_step.par_iter_mut().try_for_each(|ps_i| {
      small_commit_precommitted_witness(ps_i, &pk.S_step_small, &pk.field.ck)
    })?;
    info!(
      elapsed_ms = %step_commit_t.elapsed().as_millis(),
      circuits = step_circuits.len(),
      "prep_step_witness_commit"
    );

    let (_core_commit_span, core_commit_t) = start_span!("prep_core_witness_commit");
    small_commit_precommitted_witness(&mut ps, &pk.S_core_small, &pk.field.ck)?;
    info!(
      elapsed_ms = %core_commit_t.elapsed().as_millis(),
      "prep_core_witness_commit"
    );
    info!(elapsed_ms = %commit_t.elapsed().as_millis(), "prep_witness_commit");

    info!(
      elapsed_ms = %precommit_t.elapsed().as_millis(),
      circuits = step_circuits.len() + 1,
      "generate_small_precommitted_witnesses"
    );

    let cached_step_public_values = step_circuits
      .iter()
      .map(|c| {
        c.public_values().map_err(|e| SpartanError::SynthesisError {
          reason: format!("Small circuit does not provide public IO: {e}"),
        })
      })
      .collect::<Result<Vec<_>, _>>()?;

    let (_cache_span, cache_t) = start_span!("prep_small_accumulator_nifs_cache", l0 = l0);
    let small_abc = Self::build_small_abc_small(pk, &ps_step, &cached_step_public_values, l0)?;
    let extended_mle_evals = if l0 == ell_b {
      Some(build_extended_prefix_mle_evals(&small_abc, l0)?)
    } else {
      None
    };
    let prefix_workspace = if l0 < ell_b {
      Some(PrefixWorkspace::build(
        &small_abc,
        l0,
        step_circuits.len().next_power_of_two(),
      )?)
    } else {
      None
    };
    info!(
      elapsed_ms = %cache_t.elapsed().as_millis(),
      instances = step_circuits.len(),
      l0 = l0,
      "prep_small_accumulator_nifs_cache"
    );

    let cached_core_public_values =
      core_circuit
        .public_values()
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Small core circuit does not provide public IO: {e}"),
        })?;
    let cached_core_witness_field = if pk.S_core_small.num_rest_unpadded == 0 {
      Some(
        ps.W
          .par_iter()
          .copied()
          .map(R1CSValue::<E>::to_scalar)
          .collect::<Vec<_>>(),
      )
    } else {
      None
    };

    info!(elapsed_ms = %prep_t.elapsed().as_millis(), "neutronnova_small_accumulator_prep_prove");
    Ok(NeutronNovaAccumulatorPrepZkSNARK {
      ps_step,
      ps_core: ps,
      small_abc,
      extended_mle_evals,
      prefix_workspace,
      cached_step_public_values,
      cached_core_public_values,
      cached_core_witness_field,
      _p: PhantomData,
    })
  }

  /// Proves through the native small-value accumulator path.
  pub fn prove_small<C1, C2, Coeff>(
    self,
    pk: &NeutronNovaSmallProverKey<E, Coeff>,
    step_circuits: &[C1],
    core_circuit: &C2,
  ) -> Result<(NeutronNovaZkSNARK<E>, Self), SpartanError>
  where
    Coeff: SmallCoeff + WideMul<W, Output = SV>,
    W: Copy + Sync,
    SV: Copy + Default + AddAssign + Send,
    C1: SmallSpartanCircuit<E, W, Coeff>,
    C2: SmallSpartanCircuit<E, W, Coeff>,
    E::Scalar: DelayedReduction<SV>
      + DelayedReduction<<SV as WideMul>::Output>
      + DelayedReduction<E::Scalar>,
  {
    let mut prep_snark = self;
    let l0 = prep_snark.small_abc.l0;
    let ell_b = step_circuits.len().next_power_of_two().log_2();
    Self::validate_l0_small(pk, l0, ell_b)?;
    let field_pk = &pk.field;
    let (_prove_span, prove_t) = start_span!("neutronnova_prove_small");

    let (_rerandomize_span, rerandomize_t) = start_span!("rerandomize_small_prep_state");
    prep_snark
      .ps_core
      .rerandomize_in_place(&field_pk.ck, &pk.S_core_small)?;
    let comm_W_shared = prep_snark.ps_core.comm_W_shared.clone();
    let r_W_shared = prep_snark.ps_core.r_W_shared.clone();
    prep_snark.ps_step.par_iter_mut().try_for_each(|ps_i| {
      ps_i.rerandomize_with_shared_in_place(
        &field_pk.ck,
        &pk.S_step_small,
        &comm_W_shared,
        &r_W_shared,
      )
    })?;
    info!(elapsed_ms = %rerandomize_t.elapsed().as_millis(), "rerandomize_small_prep_state");

    if prep_snark.cached_step_public_values.len() != step_circuits.len() {
      return Err(SpartanError::InternalError {
        reason: format!(
          "Small accumulator cache was computed for {} step circuits, but prove received {}",
          prep_snark.cached_step_public_values.len(),
          step_circuits.len()
        ),
      });
    }
    for (i, circuit) in step_circuits.iter().enumerate() {
      let current_pv = circuit
        .public_values()
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Small circuit does not provide public IO: {e}"),
        })?;
      if prep_snark.cached_step_public_values[i] != current_pv {
        return Err(SpartanError::InternalError {
          reason: format!("Step circuit {i} public values changed between prep_prove and prove"),
        });
      }
    }
    let current_core_public_values =
      core_circuit
        .public_values()
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Small core circuit does not provide public IO: {e}"),
        })?;
    if prep_snark.cached_core_public_values != current_core_public_values {
      return Err(SpartanError::InternalError {
        reason: "Core circuit public values changed between prep_prove and prove".to_string(),
      });
    }

    let (_gen_span, gen_t) = start_span!(
      "generate_small_instances_witnesses",
      step_circuits = step_circuits.len()
    );
    let cached_step_public_values = &prep_snark.cached_step_public_values;
    let cached_core_public_values = prep_snark.cached_core_public_values.clone();
    let (res_steps, res_core) = rayon::join(
      || {
        prep_snark
          .ps_step
          .par_iter_mut()
          .zip(step_circuits.par_iter().enumerate())
          .map(|(pre_state, (i, circuit))| {
            let mut transcript = E::TE::new(b"neutronnova_prove");
            transcript.absorb(b"vk", &field_pk.vk_digest);
            transcript.absorb(
              b"num_circuits",
              &E::Scalar::from(step_circuits.len() as u64),
            );
            transcript.absorb(b"circuit_index", &E::Scalar::from(i as u64));

            let public_values = &cached_step_public_values[i];
            let public_values_field = public_values
              .iter()
              .copied()
              .map(R1CSValue::<E>::to_scalar)
              .collect::<Vec<E::Scalar>>();
            transcript.absorb(b"public_values", &public_values_field.as_slice());

            small_r1cs_instance_and_witness(
              pre_state,
              &pk.S_step_small,
              &field_pk.ck,
              circuit,
              Some(public_values.as_slice()),
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
        transcript.absorb(b"vk", &field_pk.vk_digest);
        let public_values_core_field = cached_core_public_values
          .iter()
          .copied()
          .map(R1CSValue::<E>::to_scalar)
          .collect::<Vec<E::Scalar>>();
        transcript.absorb(b"public_values", &public_values_core_field.as_slice());
        small_r1cs_instance_and_witness(
          &mut prep_snark.ps_core,
          &pk.S_core_small,
          &field_pk.ck,
          core_circuit,
          Some(cached_core_public_values.as_slice()),
          &mut transcript,
        )
      },
    );

    let ((step_instances, step_witnesses), (core_instance, core_witness)) = (res_steps?, res_core?);
    info!(
      elapsed_ms = %gen_t.elapsed().as_millis(),
      step_circuits = step_circuits.len(),
      "generate_small_instances_witnesses"
    );

    let (_reg_span, reg_t) = start_span!("convert_small_core_instance");
    let step_instances_regular = step_instances
      .iter()
      .map(|u| u.to_regular_instance())
      .collect::<Result<Vec<_>, _>>()?;
    let core_instance_regular = core_instance.to_regular_field_instance()?;
    info!(elapsed_ms = %reg_t.elapsed().as_millis(), "convert_small_core_instance");

    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &field_pk.vk_digest);
    transcript.absorb(b"core_instance", &core_instance_regular);

    let n_padded = step_instances_regular.len().next_power_of_two();
    let num_vars =
      field_pk.S_step.num_shared + field_pk.S_step.num_precommitted + field_pk.S_step.num_rest;
    let num_rounds_b = n_padded.log_2();
    let num_rounds_x = field_pk.S_step.num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let mut vc = NeutronNovaVerifierCircuit::<E>::default(
      num_rounds_b,
      num_rounds_x,
      num_rounds_y,
      field_pk.vc_shape.commitment_width,
    );
    let mut vc_state =
      SatisfyingAssignment::<E>::initialize_multiround_witness(&field_pk.vc_shape)?;

    let (_nifs_span, nifs_t) = start_span!("NIFS");
    let (E_eq, Az_step, Bz_step, Cz_step, folded_W, folded_U) =
      NeutronNovaNIFS::<E>::prove_accumulator_with_l0::<SV, W, W>(
        &field_pk.S_step,
        &field_pk.ck,
        step_instances_regular,
        step_witnesses,
        &prep_snark.small_abc,
        prep_snark.prefix_workspace.as_ref(),
        prep_snark.extended_mle_evals.as_ref(),
        &mut vc,
        &mut vc_state,
        &field_pk.vc_shape,
        &field_pk.vc_ck,
        &mut transcript,
        l0,
      )?;
    info!(elapsed_ms = %nifs_t.elapsed().as_millis(), "NIFS");

    let (_tensor_span, tensor_t) = start_span!("compute_tensor_and_poly_tau");
    let (_ell, left, _right) = compute_tensor_decomp(field_pk.S_step.num_cons);
    let mut E1 = E_eq;
    let E2 = E1.split_off(left);

    let mut poly_tau_left = MultilinearPolynomial::new(E1);
    let poly_tau_right = MultilinearPolynomial::new(E2);
    info!(elapsed_ms = %tensor_t.elapsed().as_millis(), "compute_tensor_and_poly_tau");

    let (_mp_span, mp_t) = start_span!("prepare_multilinear_polys");
    let (mut poly_Az_step, mut poly_Bz_step, mut poly_Cz_step) = (
      multilinear_with_effective_halves(Az_step),
      multilinear_with_effective_halves(Bz_step),
      multilinear_with_effective_halves(Cz_step),
    );

    let (mut poly_Az_core, mut poly_Bz_core, mut poly_Cz_core) = {
      let (_core_span, core_t) = start_span!("compute_small_core_polys");
      let z = build_z_small(&core_witness.W, &core_instance.public_values);
      let (Az, Bz, Cz) = pk.S_core_small.multiply_vec_small(&z)?;
      let to_field = |values: Vec<SV>| {
        values
          .into_iter()
          .map(<E::Scalar as SmallValueField<SV>>::small_to_field)
          .collect::<Vec<_>>()
      };
      info!(elapsed_ms = %core_t.elapsed().as_millis(), "compute_small_core_polys");
      (
        multilinear_with_effective_halves(to_field(Az)),
        multilinear_with_effective_halves(to_field(Bz)),
        multilinear_with_effective_halves(to_field(Cz)),
      )
    };

    let core_witness_field: Cow<'_, [E::Scalar]> = if let Some(cached_core_witness_field) =
      prep_snark.cached_core_witness_field.as_ref()
      && cached_core_witness_field.len() == core_witness.W.len()
    {
      Cow::Borrowed(cached_core_witness_field.as_slice())
    } else {
      Cow::Owned(
        core_witness
          .W
          .par_iter()
          .copied()
          .map(R1CSValue::<E>::to_scalar)
          .collect(),
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
      &field_pk.vc_shape,
      &field_pk.vc_ck,
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
      &field_pk.vc_shape,
      &field_pk.vc_ck,
      &vc,
      outer_start_index + num_rounds_x,
      &mut transcript,
    )?;
    let r = chals[0];

    let claim_inner_joint_step = vc.claim_Az_step + r * vc.claim_Bz_step + r * r * vc.claim_Cz_step;
    let claim_inner_joint_core = vc.claim_Az_core + r * vc.claim_Bz_core + r * r * vc.claim_Cz_core;

    let (_eval_rx_span, eval_rx_t) = start_span!("compute_eval_rx");
    let evals_rx = EqPolynomial::evals_from_points(&r_x);
    info!(
      elapsed_ms = %eval_rx_t.elapsed().as_millis(),
      num_rounds_x,
      r_x_len = r_x.len(),
      evals_rx_len = evals_rx.len(),
      "compute_eval_rx"
    );

    let (_sparse_span, sparse_t) = start_span!("compute_eval_table_sparse");
    let (poly_ABC_step, step_lo_eff, step_hi_eff) = field_pk
      .S_step
      .bind_and_prepare_poly_ABC_full(&evals_rx, &r);
    let (poly_ABC_core, core_lo_eff, core_hi_eff) = field_pk
      .S_core
      .bind_and_prepare_poly_ABC_full(&evals_rx, &r);
    info!(
      elapsed_ms = %sparse_t.elapsed().as_millis(),
      evals_rx_len = evals_rx.len(),
      step_poly_len = poly_ABC_step.len(),
      core_poly_len = poly_ABC_core.len(),
      step_lo_eff,
      step_hi_eff,
      core_lo_eff,
      core_hi_eff,
      "compute_eval_table_sparse"
    );

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
      let w_len = core_witness_field.len();
      v[..w_len].copy_from_slice(core_witness_field.as_ref());
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
      &field_pk.vc_shape,
      &field_pk.vc_ck,
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
      &field_pk.vc_shape,
      &field_pk.vc_ck,
      &vc,
      outer_start_index + num_rounds_x + 1 + num_rounds_y,
      &mut transcript,
    )?;

    let eval_w_step_commit_round = outer_start_index + num_rounds_x + 1 + num_rounds_y + 1;
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &field_pk.vc_shape,
      &field_pk.vc_ck,
      &vc,
      eval_w_step_commit_round,
      &mut transcript,
    )?;

    let _ = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &field_pk.vc_shape,
      &field_pk.vc_ck,
      &vc,
      eval_w_step_commit_round + 1,
      &mut transcript,
    )?;

    let (U_verifier, W_verifier) =
      SatisfyingAssignment::<E>::finalize_multiround_witness(&mut vc_state, &field_pk.vc_shape)?;

    let U_verifier_regular = U_verifier.to_regular_instance()?;

    let (random_U, random_W) = field_pk
      .vc_shape_regular
      .sample_random_instance_witness(&field_pk.vc_ck)?;
    let (nifs, folded_W_verifier, folded_u, folded_X) = NovaNIFS::<E>::prove(
      &field_pk.vc_ck,
      &field_pk.vc_shape_regular,
      &random_U,
      &random_W,
      &U_verifier_regular,
      &W_verifier,
      &mut transcript,
    )?;

    let relaxed_snark = crate::spartan_relaxed::RelaxedR1CSSpartanProof::prove(
      &field_pk.vc_shape_regular,
      &field_pk.vc_ck,
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
      .zip(core_witness_field.par_iter())
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

    let (_eval_arg_span, eval_arg_t) = start_span!("prove_eval_arg");
    let eval_arg = E::PCS::prove(
      &field_pk.ck,
      &field_pk.vc_ck,
      &mut transcript,
      &comm,
      &W,
      &blind,
      &r_y[1..],
      &comm_eval,
      &blind_eval,
    )?;
    info!(elapsed_ms = %eval_arg_t.elapsed().as_millis(), "prove_eval_arg");

    let step_instances = step_instances
      .into_iter()
      .map(|u| u.to_field_split_instance())
      .collect();
    let core_instance = core_instance.to_field_split_instance();
    let snark = NeutronNovaZkSNARK {
      comm_W_shared: core_instance.comm_W_shared.clone(),
      step_instances,
      core_instance,
      eval_arg,
      U_verifier,
      nifs,
      random_U,
      relaxed_snark,
    };

    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "neutronnova_prove_small");
    Ok((snark, prep_snark))
  }
}

impl<E: Engine> NeutronNovaNIFS<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  fn prove_neutronnova_small_value_sumcheck<SV, Layer>(
    a_layers: &[Layer],
    b_layers: &[Layer],
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
      Duration,
    ),
    SpartanError,
  >
  where
    E::Scalar: SmallValueField<SV>
      + DelayedReduction<SV>
      + DelayedReduction<<SV as WideMul>::Output>
      + DelayedReduction<E::Scalar>,
    Layer: AsRef<[SV]> + Sync,
    SV: SmallValue,
  {
    let ell_b = rhos.len();
    debug_assert!(l0 > 0 && l0 <= ell_b, "l0 must be in 1..=ell_b");

    let mut polys = Vec::with_capacity(l0);
    let mut r_bs = Vec::with_capacity(l0);
    let mut T_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;
    let num_constraints = a_layers.first().map_or(0, |layer| layer.as_ref().len());

    let (_acc_span, acc_t) = start_span!("build_accumulators_neutronnova");
    let accumulators = if let Some((a_ext, b_ext)) = preextended_ab
      && l0 == ell_b
    {
      build_accumulators_neutronnova_preextended(a_ext, b_ext, e_eq, left, right, rhos, ell_b)
    } else {
      build_accumulators_neutronnova(a_layers, b_layers, e_eq, left, right, rhos, l0)
    };
    let acc_elapsed = acc_t.elapsed();
    info!(
      elapsed_ms = %acc_elapsed.as_millis(),
      l0,
      ell_b,
      instances = a_layers.len(),
      constraints = num_constraints,
      threads = rayon::current_num_threads(),
      "build_accumulators_neutronnova"
    );
    info!(
      elapsed_ms = %acc_elapsed.as_millis(),
      l0,
      ell_b,
      instances = a_layers.len(),
      constraints = num_constraints,
      threads = rayon::current_num_threads(),
      "nifs_build_acc"
    );

    let mut small_value = SmallValueSumCheck::<E::Scalar, 2>::from_accumulators(accumulators);
    let (_first_l0_span, first_l0_t) = start_span!("nifs_first_l0_rounds", rounds = l0);
    let mut vc_commit_total = Duration::default();
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
      let vc_elapsed = vc_t.elapsed();
      vc_commit_total += vc_elapsed;
      info!(elapsed_ms = %vc_elapsed.as_millis(), round = i, "vc_commit");
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
    info!(
      elapsed_ms = %first_l0_t.elapsed().as_millis(),
      rounds = l0,
      l0,
      ell_b,
      instances = a_layers.len(),
      constraints = num_constraints,
      threads = rayon::current_num_threads(),
      "nifs_first_l0_rounds"
    );

    Ok((polys, r_bs, T_cur, acc_eq, vc_commit_total))
  }

  fn prove_neutronnova_small_value_sumcheck_prefix_workspace<SV>(
    prefix_workspace: &PrefixWorkspace<SV>,
    e_eq: &[E::Scalar],
    left: usize,
    right: usize,
    rhos: &[E::Scalar],
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
      Duration,
    ),
    SpartanError,
  >
  where
    E::Scalar: SmallValueField<SV>
      + DelayedReduction<SV>
      + DelayedReduction<<SV as WideMul>::Output>
      + DelayedReduction<E::Scalar>,
    SV: SmallValue,
    <SV as WideMul>::Output: Sync,
  {
    let l0 = prefix_workspace.l0;
    let ell_b = rhos.len();
    debug_assert!(l0 > 0 && l0 <= ell_b, "l0 must be in 1..=ell_b");

    let mut polys = Vec::with_capacity(l0);
    let mut r_bs = Vec::with_capacity(l0);
    let mut T_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;

    let (_acc_span, acc_t) = start_span!("build_accumulators_neutronnova_workspace");
    let accumulators = build_accumulators_neutronnova_from_prefix_ext_workspace(
      &prefix_workspace.ab_ext,
      &prefix_workspace.beta_indices,
      prefix_workspace.num_constraints,
      prefix_workspace.suffix_groups,
      e_eq,
      left,
      right,
      rhos,
      l0,
    );
    let acc_elapsed = acc_t.elapsed();
    info!(
      elapsed_ms = %acc_elapsed.as_millis(),
      l0,
      ell_b,
      suffix_groups = prefix_workspace.suffix_groups,
      constraints = prefix_workspace.num_constraints,
      threads = rayon::current_num_threads(),
      "build_accumulators_neutronnova_workspace"
    );
    info!(
      elapsed_ms = %acc_elapsed.as_millis(),
      l0,
      ell_b,
      suffix_groups = prefix_workspace.suffix_groups,
      constraints = prefix_workspace.num_constraints,
      threads = rayon::current_num_threads(),
      "nifs_build_acc"
    );

    let mut small_value = SmallValueSumCheck::<E::Scalar, 2>::from_accumulators(accumulators);
    let (_first_l0_span, first_l0_t) = start_span!("nifs_first_l0_rounds", rounds = l0);
    let mut vc_commit_total = Duration::default();
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
      let vc_elapsed = vc_t.elapsed();
      vc_commit_total += vc_elapsed;
      info!(elapsed_ms = %vc_elapsed.as_millis(), round = i, "vc_commit");
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
    info!(
      elapsed_ms = %first_l0_t.elapsed().as_millis(),
      rounds = l0,
      l0,
      ell_b,
      suffix_groups = prefix_workspace.suffix_groups,
      constraints = prefix_workspace.num_constraints,
      threads = rayon::current_num_threads(),
      "nifs_first_l0_rounds"
    );

    Ok((polys, r_bs, T_cur, acc_eq, vc_commit_total))
  }

  fn fold_prefix_workspace_and_first_suffix_round<SV>(
    prefix_weights: &[E::Scalar],
    prefix_workspace: &PrefixWorkspace<SV>,
    l0: usize,
    ell_b: usize,
    left: usize,
    right: usize,
    e_eq: &[E::Scalar],
    rhos: &[E::Scalar],
    carry_c_layers: bool,
  ) -> (
    Vec<Vec<E::Scalar>>,
    Vec<Vec<E::Scalar>>,
    Vec<Vec<E::Scalar>>,
    Vec<E::Scalar>,
    E::Scalar,
    E::Scalar,
  )
  where
    E::Scalar: DelayedReduction<SV> + DelayedReduction<E::Scalar>,
    SV: WideMul + Send + Sync,
  {
    let pairs = prefix_workspace.suffix_groups / 2;
    debug_assert!(pairs > 0);
    debug_assert_eq!(prefix_workspace.suffix_groups % 2, 0);

    let pair_results: Vec<_> = (0..pairs)
      .into_par_iter()
      .map(|pair_idx| {
        let lo = 2 * pair_idx;
        let hi = lo + 1;
        let (a_lo, a_hi, b_lo, b_hi, c_lo_vec, c_hi_vec, c_lo, c_hi, e0, quad_coeff) =
          Self::fold_prefix_workspace_pair_and_first_suffix_round::<SV>(
            prefix_weights,
            prefix_workspace,
            lo,
            hi,
            left,
            right,
            e_eq,
            carry_c_layers,
          );
        let w = suffix_weight_full::<E::Scalar>(l0, ell_b, pair_idx, rhos);
        (
          pair_idx,
          a_lo,
          a_hi,
          b_lo,
          b_hi,
          c_lo_vec,
          c_hi_vec,
          c_lo,
          c_hi,
          e0 * w,
          quad_coeff * w,
        )
      })
      .collect();

    let mut a_layers = vec![Vec::new(); prefix_workspace.suffix_groups];
    let mut b_layers = vec![Vec::new(); prefix_workspace.suffix_groups];
    let mut c_layers = vec![Vec::new(); prefix_workspace.suffix_groups];
    let mut c_vals = vec![E::Scalar::ZERO; prefix_workspace.suffix_groups];
    let mut e0 = E::Scalar::ZERO;
    let mut quad_coeff = E::Scalar::ZERO;
    for (pair_idx, a_lo, a_hi, b_lo, b_hi, c_lo_vec, c_hi_vec, c_lo, c_hi, pair_e0, pair_qc) in
      pair_results
    {
      let lo = 2 * pair_idx;
      let hi = lo + 1;
      a_layers[lo] = a_lo;
      a_layers[hi] = a_hi;
      b_layers[lo] = b_lo;
      b_layers[hi] = b_hi;
      c_layers[lo] = c_lo_vec;
      c_layers[hi] = c_hi_vec;
      c_vals[lo] = c_lo;
      c_vals[hi] = c_hi;
      e0 += pair_e0;
      quad_coeff += pair_qc;
    }

    (a_layers, b_layers, c_layers, c_vals, e0, quad_coeff)
  }

  #[allow(clippy::type_complexity)]
  fn fold_prefix_workspace_pair_and_first_suffix_round<SV>(
    prefix_weights: &[E::Scalar],
    prefix_workspace: &PrefixWorkspace<SV>,
    lo_suffix: usize,
    hi_suffix: usize,
    left: usize,
    right: usize,
    e_eq: &[E::Scalar],
    carry_c_layers: bool,
  ) -> (
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    E::Scalar,
    E::Scalar,
    E::Scalar,
    E::Scalar,
  )
  where
    E::Scalar: DelayedReduction<SV> + DelayedReduction<E::Scalar>,
    SV: WideMul + Send + Sync,
  {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    let num_constraints = prefix_workspace.num_constraints;
    debug_assert_eq!(num_constraints, left * right);
    let mut a_lo = vec![E::Scalar::ZERO; num_constraints];
    let mut a_hi = vec![E::Scalar::ZERO; num_constraints];
    let mut b_lo = vec![E::Scalar::ZERO; num_constraints];
    let mut b_hi = vec![E::Scalar::ZERO; num_constraints];
    let mut c_lo_vec = if carry_c_layers {
      vec![E::Scalar::ZERO; num_constraints]
    } else {
      Vec::new()
    };
    let mut c_hi_vec = if carry_c_layers {
      vec![E::Scalar::ZERO; num_constraints]
    } else {
      Vec::new()
    };

    let e_left = &e_eq[..left];
    let e_right = &e_eq[left..];
    let row_terms: Vec<_> = if carry_c_layers {
      a_lo
        .par_chunks_mut(left)
        .zip(a_hi.par_chunks_mut(left))
        .zip(b_lo.par_chunks_mut(left))
        .zip(b_hi.par_chunks_mut(left))
        .zip(c_lo_vec.par_chunks_mut(left))
        .zip(c_hi_vec.par_chunks_mut(left))
        .enumerate()
        .map(
          |(i, (((((a_lo_row, a_hi_row), b_lo_row), b_hi_row), c_lo_row), c_hi_row))| {
            Self::fold_prefix_workspace_constraint_row::<SV>(
              prefix_weights,
              prefix_workspace,
              lo_suffix,
              hi_suffix,
              left,
              num_constraints,
              e_left,
              i,
              a_lo_row,
              a_hi_row,
              b_lo_row,
              b_hi_row,
              Some((c_lo_row, c_hi_row)),
            )
          },
        )
        .collect()
    } else {
      a_lo
        .par_chunks_mut(left)
        .zip(a_hi.par_chunks_mut(left))
        .zip(b_lo.par_chunks_mut(left))
        .zip(b_hi.par_chunks_mut(left))
        .enumerate()
        .map(|(i, (((a_lo_row, a_hi_row), b_lo_row), b_hi_row))| {
          Self::fold_prefix_workspace_constraint_row::<SV>(
            prefix_weights,
            prefix_workspace,
            lo_suffix,
            hi_suffix,
            left,
            num_constraints,
            e_left,
            i,
            a_lo_row,
            a_hi_row,
            b_lo_row,
            b_hi_row,
            None,
          )
        })
        .collect()
    };

    let mut acc_e0_ab = Acc::<E::Scalar>::default();
    let mut acc_quad = Acc::<E::Scalar>::default();
    let mut acc_c_lo = Acc::<E::Scalar>::default();
    let mut acc_c_hi = Acc::<E::Scalar>::default();
    for (i, inner_e0_ab, inner_quad, inner_c_lo, inner_c_hi) in row_terms {
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_e0_ab,
        &e_right[i],
        &inner_e0_ab,
      );
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_quad,
        &e_right[i],
        &inner_quad,
      );
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_c_lo,
        &e_right[i],
        &inner_c_lo,
      );
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_c_hi,
        &e_right[i],
        &inner_c_hi,
      );
    }

    let e0_ab = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_e0_ab);
    let c_lo_val = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_c_lo);
    let c_hi_val = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_c_hi);

    (
      a_lo,
      a_hi,
      b_lo,
      b_hi,
      c_lo_vec,
      c_hi_vec,
      c_lo_val,
      c_hi_val,
      e0_ab - c_lo_val,
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad),
    )
  }

  #[allow(clippy::too_many_arguments)]
  fn fold_prefix_workspace_constraint_row<SV>(
    prefix_weights: &[E::Scalar],
    prefix_workspace: &PrefixWorkspace<SV>,
    lo_suffix: usize,
    hi_suffix: usize,
    left: usize,
    num_constraints: usize,
    e_left: &[E::Scalar],
    row_idx: usize,
    a_lo_row: &mut [E::Scalar],
    a_hi_row: &mut [E::Scalar],
    b_lo_row: &mut [E::Scalar],
    b_hi_row: &mut [E::Scalar],
    mut c_rows: Option<(&mut [E::Scalar], &mut [E::Scalar])>,
  ) -> (usize, E::Scalar, E::Scalar, E::Scalar, E::Scalar)
  where
    E::Scalar: DelayedReduction<SV> + DelayedReduction<E::Scalar>,
    SV: WideMul + Send + Sync,
  {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    let base = row_idx * left;
    let mut inner_e0_ab = Acc::<E::Scalar>::default();
    let mut inner_quad = Acc::<E::Scalar>::default();
    let mut inner_c_lo = Acc::<E::Scalar>::default();
    let mut inner_c_hi = Acc::<E::Scalar>::default();

    for j in 0..left {
      let k = base + j;
      let (a0, a1, b0, b1, c0, c1) = fold_prefix_workspace_pair_values::<E::Scalar, SV>(
        prefix_weights,
        prefix_workspace,
        lo_suffix,
        hi_suffix,
        k,
        num_constraints,
      );

      a_lo_row[j] = a0;
      a_hi_row[j] = a1;
      b_lo_row[j] = b0;
      b_hi_row[j] = b1;
      if let Some((c_lo_row, c_hi_row)) = c_rows.as_mut() {
        c_lo_row[j] = c0;
        c_hi_row[j] = c1;
      }

      let inner_val = a0 * b0;
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut inner_e0_ab,
        &e_left[j],
        &inner_val,
      );
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut inner_c_lo,
        &e_left[j],
        &c0,
      );
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut inner_c_hi,
        &e_left[j],
        &c1,
      );
      let quad_val = (a1 - a0) * (b1 - b0);
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut inner_quad,
        &e_left[j],
        &quad_val,
      );
    }

    (
      row_idx,
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_e0_ab),
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_quad),
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_c_lo),
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_c_hi),
    )
  }

  fn prove_accumulator_full_batch<SV, X, W>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    Us: &[R1CSInstance<E, X>],
    Ws: &[R1CSWitness<E, W>],
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
      + DelayedReduction<<SV as WideMul>::Output>
      + DelayedReduction<E::Scalar>,
    SV: ExtensionSmallValue,
    <SV as WideMul>::Output: Copy + Zero + Send + Sync,
    X: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
    W: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
  {
    let (_nifs_total_span, nifs_total_t) = start_span!("nifs_prove");
    let (Us, Ws, ell_b, tau, rhos) = prepare_nifs_inputs_typed::<E, X, W>(Us, Ws, transcript)?;
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
    let (_polys, r_bs, T_cur, acc_eq, mut vc_commit_total) =
      Self::prove_neutronnova_small_value_sumcheck::<SV, _>(
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

    let (az_folded, (bz_folded, (cz_folded, final_c_elapsed))) = rayon::join(
      || fold_small_value_vectors(&eq_evals, &a_small),
      || {
        rayon::join(
          || fold_small_value_vectors(&eq_evals, &b_small),
          || {
            let (_final_c_span, final_c_t) = start_span!("nifs_final_c");
            let folded = fold_small_value_vectors(&eq_evals, &c_small);
            (folded, final_c_t.elapsed())
          },
        )
      },
    );
    info!(
      elapsed_ms = %final_c_elapsed.as_millis(),
      l0 = ell_b,
      ell_b,
      instances = n_padded,
      constraints = S.num_cons,
      threads = rayon::current_num_threads(),
      "nifs_final_c"
    );
    info!(
      elapsed_ms = %fold_t.elapsed().as_millis(),
      l0 = ell_b,
      ell_b,
      instances = n_padded,
      constraints = S.num_cons,
      threads = rayon::current_num_threads(),
      "nifs_eq_fold"
    );

    let (folded_W, folded_U, final_vc_elapsed) = fold_and_update_vc_field::<E, X, W>(
      S, ck, &r_bs, T_cur, acc_eq, &Us, &Ws, ell_b, vc, vc_state, vc_shape, vc_ck, transcript,
    )?;
    vc_commit_total += final_vc_elapsed;
    info!(
      elapsed_ms = %vc_commit_total.as_millis(),
      l0 = ell_b,
      ell_b,
      instances = n_padded,
      constraints = S.num_cons,
      threads = rayon::current_num_threads(),
      "nifs_vc_commit_total"
    );

    info!(
      elapsed_ms = %nifs_total_t.elapsed().as_millis(),
      l0 = ell_b,
      ell_b,
      instances = n_padded,
      constraints = S.num_cons,
      threads = rayon::current_num_threads(),
      "nifs_prove"
    );
    Ok((E_eq, az_folded, bz_folded, cz_folded, folded_W, folded_U))
  }

  fn prove_accumulator_prefix_small<SV, X, W>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    Us: &[R1CSInstance<E, X>],
    Ws: &[R1CSWitness<E, W>],
    small_abc: &SmallAbc<SV>,
    cached_prefix_workspace: Option<&PrefixWorkspace<SV>>,
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
      + DelayedReduction<<SV as WideMul>::Output>
      + DelayedReduction<E::Scalar>,
    SV: ExtensionSmallValue,
    <SV as WideMul>::Output: Copy + Zero + Send + Sync,
    X: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
    W: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
  {
    let (_nifs_total_span, nifs_total_t) = start_span!("nifs_prove");
    let (Us, Ws, ell_b, tau, rhos) = prepare_nifs_inputs_typed::<E, X, W>(Us, Ws, transcript)?;
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

    let prefix_size = 1usize << l0;
    let (_workspace_span, workspace_t) =
      start_span!("nifs_prefix_workspace_build", instances = n_padded, l0);
    let built_prefix_workspace;
    let prefix_workspace = match cached_prefix_workspace {
      Some(prefix_workspace) => {
        if prefix_workspace.l0 != l0
          || prefix_workspace.prefix_size != prefix_size
          || prefix_workspace.suffix_groups != n_padded / prefix_size
          || prefix_workspace.num_constraints != S.num_cons
        {
          return Err(SpartanError::InvalidInputLength {
            reason: "cached prefix workspace does not match accumulator prove shape".into(),
          });
        }
        prefix_workspace
      }
      None => {
        built_prefix_workspace = PrefixWorkspace::build(small_abc, l0, n_padded)?;
        &built_prefix_workspace
      }
    };
    info!(
      elapsed_ms = %workspace_t.elapsed().as_millis(),
      instances = n_padded,
      l0,
      suffix_groups = prefix_workspace.suffix_groups,
      constraints = prefix_workspace.num_constraints,
      cached = cached_prefix_workspace.is_some(),
      "nifs_prefix_workspace_build"
    );

    let (_rounds_span, rounds_t) = start_span!("nifs_folding_rounds", rounds = l0);
    let (_polys, mut r_bs, mut T_cur, mut acc_eq, mut vc_commit_total) =
      Self::prove_neutronnova_small_value_sumcheck_prefix_workspace::<SV>(
        prefix_workspace,
        &E_eq,
        left,
        right,
        &rhos,
        vc,
        vc_state,
        vc_shape,
        vc_ck,
        transcript,
      )?;
    info!(
      elapsed_ms = %rounds_t.elapsed().as_millis(),
      rounds = l0,
      "nifs_folding_rounds"
    );

    let (_fold_prefix_span, fold_prefix_t) = start_span!("nifs_prefix_fold", rounds = l0);
    let prefix_weights = weights_from_r::<E::Scalar>(&r_bs, prefix_size);
    let carry_c_layers = rayon::current_num_threads() <= 1;
    debug_assert_eq!(prefix_workspace.prefix_size, prefix_size);
    let (_fused_span, fused_t) = start_span!("nifs_prefix_fold_first_suffix_workspace");
    let (mut a_layers, mut b_layers, mut c_layers, mut c_vals, first_e0, first_quad_coeff) =
      Self::fold_prefix_workspace_and_first_suffix_round::<SV>(
        &prefix_weights,
        prefix_workspace,
        l0,
        ell_b,
        left,
        right,
        &E_eq,
        &rhos,
        carry_c_layers,
      );
    info!(
      elapsed_ms = %fused_t.elapsed().as_millis(),
      "nifs_prefix_fold_first_suffix_workspace"
    );
    info!(
      elapsed_ms = %fold_prefix_t.elapsed().as_millis(),
      l0,
      ell_b,
      instances = n_padded,
      suffix_groups = prefix_workspace.suffix_groups,
      constraints = prefix_workspace.num_constraints,
      threads = rayon::current_num_threads(),
      "nifs_prefix_fold"
    );

    let (_suffix_span, suffix_t) = start_span!("nifs_suffix_rounds", rounds = ell_b - l0);
    if l0 < ell_b {
      let mut m = a_layers.len();
      let mut finished_suffix = false;

      let pairs = m / 2;
      if pairs > 0 {
        let (r_b, vc_elapsed) = finish_field_sumcheck_round::<E>(
          l0,
          first_e0,
          first_quad_coeff,
          &rhos,
          &mut r_bs,
          &mut T_cur,
          &mut acc_eq,
          vc,
          vc_state,
          vc_shape,
          vc_ck,
          transcript,
        )?;
        vc_commit_total += vc_elapsed;

        if l0 + 1 == ell_b {
          fold_final_ab_pairs::<E>(&mut a_layers, &mut b_layers, pairs, r_b);
          if carry_c_layers {
            fold_final_layer_pairs(&mut c_layers, pairs, r_b);
          }
          a_layers.truncate(pairs);
          b_layers.truncate(pairs);
          if carry_c_layers {
            c_layers.truncate(pairs);
          }
          finished_suffix = true;
        }
      }

      if !finished_suffix {
        let mut prev_r_b = *r_bs.last().ok_or(SpartanError::InvalidSumcheckProof)?;

        for t in (l0 + 1)..ell_b {
          let fold_pairs = m / 2;
          let prove_pairs = fold_pairs / 2;
          let mut e0_acc = E::Scalar::ZERO;
          let mut quad_acc = E::Scalar::ZERO;

          if prove_pairs > 0 {
            let (a_head, _) = a_layers.split_at_mut(4 * prove_pairs);
            let (b_head, _) = b_layers.split_at_mut(4 * prove_pairs);
            let (c_head, _) = c_vals.split_at_mut(4 * prove_pairs);

            let (e0_sum, qc_sum) = if carry_c_layers {
              let (c_layer_head, _) = c_layers.split_at_mut(4 * prove_pairs);
              a_head
                .par_chunks_mut(4)
                .zip(b_head.par_chunks_mut(4))
                .zip(c_layer_head.par_chunks_mut(4))
                .zip(c_head.par_chunks_mut(4))
                .enumerate()
                .map(|(j, (((a_chunk, b_chunk), c_layer_chunk), c_chunk))| {
                  fold_suffix_round_chunk::<E>(
                    (left, right),
                    &E_eq,
                    t,
                    ell_b,
                    j,
                    &rhos,
                    prev_r_b,
                    a_chunk,
                    b_chunk,
                    c_chunk,
                    Some(c_layer_chunk),
                  )
                })
                .reduce(
                  || (E::Scalar::ZERO, E::Scalar::ZERO),
                  |a, b| (a.0 + b.0, a.1 + b.1),
                )
            } else {
              a_head
                .par_chunks_mut(4)
                .zip(b_head.par_chunks_mut(4))
                .zip(c_head.par_chunks_mut(4))
                .enumerate()
                .map(|(j, ((a_chunk, b_chunk), c_chunk))| {
                  fold_suffix_round_chunk::<E>(
                    (left, right),
                    &E_eq,
                    t,
                    ell_b,
                    j,
                    &rhos,
                    prev_r_b,
                    a_chunk,
                    b_chunk,
                    c_chunk,
                    None,
                  )
                })
                .reduce(
                  || (E::Scalar::ZERO, E::Scalar::ZERO),
                  |a, b| (a.0 + b.0, a.1 + b.1),
                )
            };
            e0_acc += e0_sum;
            quad_acc += qc_sum;

            compact_folded_layers_ab::<E>(&mut a_layers, &mut b_layers, prove_pairs);
            if carry_c_layers {
              compact_folded_layers(&mut c_layers, prove_pairs);
            }
            compact_folded_scalars::<E::Scalar>(&mut c_vals, prove_pairs);
          }

          for i in (2 * prove_pairs)..fold_pairs {
            fold_ab_pair_into::<E>(&mut a_layers, &mut b_layers, 2 * i, 2 * i + 1, i, prev_r_b);
            if carry_c_layers {
              fold_layer_pair_into(&mut c_layers, 2 * i, 2 * i + 1, i, prev_r_b);
            }
            fold_scalar_pair_into(&mut c_vals, 2 * i, 2 * i + 1, i, prev_r_b);
          }

          a_layers.truncate(fold_pairs);
          b_layers.truncate(fold_pairs);
          if carry_c_layers {
            c_layers.truncate(fold_pairs);
          }
          c_vals.truncate(fold_pairs);
          m = fold_pairs;

          let (next_r_b, vc_elapsed) = finish_field_sumcheck_round::<E>(
            t,
            e0_acc,
            quad_acc,
            &rhos,
            &mut r_bs,
            &mut T_cur,
            &mut acc_eq,
            vc,
            vc_state,
            vc_shape,
            vc_ck,
            transcript,
          )?;
          vc_commit_total += vc_elapsed;
          prev_r_b = next_r_b;
        }

        let final_pairs = m / 2;
        if final_pairs > 0 {
          fold_final_ab_pairs::<E>(&mut a_layers, &mut b_layers, final_pairs, prev_r_b);
          if carry_c_layers {
            fold_final_layer_pairs(&mut c_layers, final_pairs, prev_r_b);
          }
        }
        a_layers.truncate(final_pairs);
        b_layers.truncate(final_pairs);
        if carry_c_layers {
          c_layers.truncate(final_pairs);
        }
      }
    }
    info!(
      elapsed_ms = %suffix_t.elapsed().as_millis(),
      l0,
      ell_b,
      instances = n_padded,
      suffix_groups = prefix_workspace.suffix_groups,
      constraints = prefix_workspace.num_constraints,
      threads = rayon::current_num_threads(),
      rounds = ell_b - l0,
      "nifs_suffix_rounds"
    );

    let az_folded = a_layers.pop().ok_or(SpartanError::InvalidInputLength {
      reason: "partial-l0 NIFS produced no folded A layer".into(),
    })?;
    let bz_folded = b_layers.pop().ok_or(SpartanError::InvalidInputLength {
      reason: "partial-l0 NIFS produced no folded B layer".into(),
    })?;
    let (_final_c_span, final_c_t) = start_span!("nifs_final_c");
    let cz_folded = if carry_c_layers {
      c_layers.pop().ok_or(SpartanError::InvalidInputLength {
        reason: "partial-l0 NIFS produced no folded C layer".into(),
      })?
    } else {
      let final_weights = weights_from_r::<E::Scalar>(&r_bs, n_padded);
      fold_prefix_workspace_final_table::<E::Scalar, SV>(
        &final_weights,
        &prefix_workspace.c,
        prefix_workspace.num_constraints,
        prefix_workspace.prefix_size,
      )
    };
    info!(
      elapsed_ms = %final_c_t.elapsed().as_millis(),
      l0,
      ell_b,
      instances = n_padded,
      suffix_groups = prefix_workspace.suffix_groups,
      constraints = prefix_workspace.num_constraints,
      threads = rayon::current_num_threads(),
      "nifs_final_c"
    );

    let (folded_W, folded_U, final_vc_elapsed) = fold_and_update_vc_field::<E, X, W>(
      S, ck, &r_bs, T_cur, acc_eq, &Us, &Ws, ell_b, vc, vc_state, vc_shape, vc_ck, transcript,
    )?;
    vc_commit_total += final_vc_elapsed;
    info!(
      elapsed_ms = %vc_commit_total.as_millis(),
      l0,
      ell_b,
      instances = n_padded,
      suffix_groups = prefix_workspace.suffix_groups,
      constraints = prefix_workspace.num_constraints,
      threads = rayon::current_num_threads(),
      "nifs_vc_commit_total"
    );

    info!(
      elapsed_ms = %nifs_total_t.elapsed().as_millis(),
      l0,
      ell_b,
      instances = n_padded,
      suffix_groups = prefix_workspace.suffix_groups,
      constraints = prefix_workspace.num_constraints,
      threads = rayon::current_num_threads(),
      "nifs_prove"
    );
    Ok((E_eq, az_folded, bz_folded, cz_folded, folded_W, folded_U))
  }

  fn prove_accumulator_with_l0<SV, X, W>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    Us: Vec<R1CSInstance<E, X>>,
    Ws: Vec<R1CSWitness<E, W>>,
    small_abc: &SmallAbc<SV>,
    prefix_workspace: Option<&PrefixWorkspace<SV>>,
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
      + DelayedReduction<<SV as WideMul>::Output>
      + DelayedReduction<E::Scalar>,
    SV: ExtensionSmallValue,
    <SV as WideMul>::Output: Copy + Zero + Send + Sync,
    X: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
    W: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
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
      Self::prove_accumulator_prefix_small::<SV, X, W>(
        S,
        ck,
        &Us,
        &Ws,
        small_abc,
        prefix_workspace,
        vc,
        vc_state,
        vc_shape,
        vc_ck,
        transcript,
        l0,
      )
    } else {
      match extended_mle_evals {
        Some(extended_mle_evals) => Self::prove_accumulator_full_batch::<SV, X, W>(
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

/// Prove-local transposed view of `SmallAbc` for a partial-`l0` proof.
///
/// Each table uses the layout:
/// `suffix_group -> constraint -> prefix_values[0..2^l0]`.
///
/// This keeps prefix slices contiguous for both accumulator construction and
/// prefix folding without moving transcript-independent partial-`l0` work into
/// prep.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
  serialize = "SV: Serialize, <SV as WideMul>::Output: Serialize",
  deserialize = "SV: Deserialize<'de>, <SV as WideMul>::Output: Deserialize<'de>"
))]
struct PrefixWorkspace<SV: WideMul> {
  l0: usize,
  prefix_size: usize,
  ext_size: usize,
  suffix_groups: usize,
  num_constraints: usize,
  beta_indices: Vec<usize>,
  a: Vec<SV>,
  b: Vec<SV>,
  c: Vec<SV>,
  ab_ext: Vec<<SV as WideMul>::Output>,
}

impl<SV> std::fmt::Debug for PrefixWorkspace<SV>
where
  SV: WideMul + std::fmt::Debug,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("PrefixWorkspace")
      .field("l0", &self.l0)
      .field("prefix_size", &self.prefix_size)
      .field("ext_size", &self.ext_size)
      .field("suffix_groups", &self.suffix_groups)
      .field("num_constraints", &self.num_constraints)
      .field("beta_indices_len", &self.beta_indices.len())
      .field("a_len", &self.a.len())
      .field("b_len", &self.b.len())
      .field("c_len", &self.c.len())
      .field("ab_ext_len", &self.ab_ext.len())
      .finish()
  }
}

impl<SV> PrefixWorkspace<SV>
where
  SV: SmallValue,
  <SV as WideMul>::Output: Copy + Zero + Send + Sync,
{
  fn build(mle_inputs: &SmallAbc<SV>, l0: usize, n_padded: usize) -> Result<Self, SpartanError> {
    if mle_inputs.num_instances == 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "cannot build prefix workspace for empty step batch".into(),
      });
    }
    let prefix_size = 1usize << l0;
    if n_padded % prefix_size != 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "prefix workspace instance count {} is not divisible by prefix size {}",
          n_padded, prefix_size
        ),
      });
    }

    let suffix_groups = n_padded / prefix_size;
    let num_constraints = mle_inputs.num_constraints;
    let table_len = suffix_groups * num_constraints * prefix_size;
    let mut a = vec![SV::default(); table_len];
    let mut b = vec![SV::default(); table_len];
    let mut c = vec![SV::default(); table_len];

    rayon::join(
      || {
        Self::transpose_table(
          mle_inputs,
          &mle_inputs.a,
          &mut a,
          prefix_size,
          suffix_groups,
        )
      },
      || {
        rayon::join(
          || {
            Self::transpose_table(
              mle_inputs,
              &mle_inputs.b,
              &mut b,
              prefix_size,
              suffix_groups,
            )
          },
          || {
            Self::transpose_table(
              mle_inputs,
              &mle_inputs.c,
              &mut c,
              prefix_size,
              suffix_groups,
            )
          },
        )
      },
    );

    let ext_size = 3usize.pow(l0 as u32);
    let beta_indices = beta_indices_with_infty(l0);
    let ext_table_len = suffix_groups * num_constraints * beta_indices.len();
    let mut ab_ext = vec![<SV as WideMul>::Output::zero(); ext_table_len];
    Self::extend_prefix_product_table(&a, &b, &mut ab_ext, prefix_size, ext_size, &beta_indices);

    Ok(Self {
      l0,
      prefix_size,
      ext_size,
      suffix_groups,
      num_constraints,
      beta_indices,
      a,
      b,
      c,
      ab_ext,
    })
  }

  fn transpose_table(
    mle_inputs: &SmallAbc<SV>,
    source: &[SV],
    dest: &mut [SV],
    prefix_size: usize,
    suffix_groups: usize,
  ) {
    let num_constraints = mle_inputs.num_constraints;
    dest
      .par_chunks_mut(num_constraints * prefix_size)
      .take(suffix_groups)
      .enumerate()
      .for_each(|(suffix_idx, suffix_chunk)| {
        for prefix_idx in 0..prefix_size {
          let layer_idx = suffix_idx * prefix_size + prefix_idx;
          let row_idx = if layer_idx < mle_inputs.num_instances {
            layer_idx
          } else {
            0
          };
          let row_start = row_idx * num_constraints;
          let row = &source[row_start..row_start + num_constraints];
          for (constraint_idx, &value) in row.iter().enumerate() {
            suffix_chunk[constraint_idx * prefix_size + prefix_idx] = value;
          }
        }
      });
  }

  fn extend_prefix_product_table(
    a_source: &[SV],
    b_source: &[SV],
    dest: &mut [<SV as WideMul>::Output],
    prefix_size: usize,
    ext_size: usize,
    beta_indices: &[usize],
  ) {
    let bit_rev = bit_rev_prefix_table(prefix_size.log_2());
    dest
      .par_chunks_mut(beta_indices.len())
      .zip(a_source.par_chunks(prefix_size))
      .zip(b_source.par_chunks(prefix_size))
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
        |(a_prefix, b_prefix, a_ext, a_scratch, b_ext, b_scratch),
         ((dest_chunk, a_source_chunk), b_source_chunk)| {
          for (p, &rev) in bit_rev.iter().enumerate() {
            a_prefix[p] = a_source_chunk[rev];
            b_prefix[p] = b_source_chunk[rev];
          }
          let a_produced = extend_to_lagrange_domain::<SV, 2>(a_prefix, a_ext, a_scratch);
          let b_produced = extend_to_lagrange_domain::<SV, 2>(b_prefix, b_ext, b_scratch);
          debug_assert_eq!(a_produced, ext_size);
          debug_assert_eq!(b_produced, ext_size);
          for (slot, &beta_idx) in beta_indices.iter().enumerate() {
            dest_chunk[slot] = a_ext[beta_idx].wide_mul(b_ext[beta_idx]);
          }
        },
      );
  }
}

fn beta_indices_with_infty(l0: usize) -> Vec<usize> {
  let base = 3usize;
  let ext_size = base.pow(l0 as u32);
  (0..ext_size)
    .filter(|&idx| (0..l0).any(|d| (idx / base.pow(d as u32)) % base == 0))
    .collect()
}

#[allow(clippy::too_many_arguments)]
fn fold_suffix_round_chunk<E: Engine>(
  dims: (usize, usize),
  e_eq: &[E::Scalar],
  round: usize,
  ell_b: usize,
  pair_idx: usize,
  rhos: &[E::Scalar],
  prev_r_b: E::Scalar,
  a_chunk: &mut [Vec<E::Scalar>],
  b_chunk: &mut [Vec<E::Scalar>],
  c_chunk: &mut [E::Scalar],
  c_layer_chunk: Option<&mut [Vec<E::Scalar>]>,
) -> (E::Scalar, E::Scalar)
where
  E::PCS: FoldingEngineTrait<E>,
{
  for chunk in [&mut *a_chunk, &mut *b_chunk] {
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

  if let Some(c_layer_chunk) = c_layer_chunk {
    {
      let (lo, hi) = c_layer_chunk.split_at_mut(1);
      lo[0]
        .iter_mut()
        .zip(hi[0].iter())
        .for_each(|(l, h)| *l += prev_r_b * (*h - *l));
    }
    {
      let (lo, hi) = c_layer_chunk.split_at_mut(3);
      lo[2]
        .iter_mut()
        .zip(hi[0].iter())
        .for_each(|(l, h)| *l += prev_r_b * (*h - *l));
    }
  }

  {
    let c0 = c_chunk[0];
    c_chunk[0] += prev_r_b * (c_chunk[1] - c0);
    let c2 = c_chunk[2];
    c_chunk[2] += prev_r_b * (c_chunk[3] - c2);
  }

  let (e0_ab, qc) = NeutronNovaNIFS::<E>::compute_tensor_eq_ab_fold_extension_terms(
    dims,
    e_eq,
    &a_chunk[0],
    &b_chunk[0],
    &a_chunk[2],
    &b_chunk[2],
  );
  let e0 = e0_ab - c_chunk[0];
  let w = suffix_weight_full::<E::Scalar>(round, ell_b, pair_idx, rhos);
  (e0 * w, qc * w)
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

fn multilinear_with_effective_halves<F>(values: Vec<F>) -> MultilinearPolynomial<F>
where
  F: Field,
{
  let half = values.len() / 2;
  let lo_eff = effective_nonzero_prefix(&values[..half]);
  let hi_eff = effective_nonzero_prefix(&values[half..]);
  MultilinearPolynomial::new_with_halves(values, lo_eff, hi_eff)
}

fn effective_nonzero_prefix<F>(values: &[F]) -> usize
where
  F: Field,
{
  values
    .iter()
    .rposition(|value| !bool::from(value.is_zero()))
    .map_or(0, |idx| idx + 1)
}

fn fold_prefix_workspace_pair_values<F, SV>(
  weights: &[F],
  prefix_workspace: &PrefixWorkspace<SV>,
  lo_suffix: usize,
  hi_suffix: usize,
  constraint_idx: usize,
  num_constraints: usize,
) -> (F, F, F, F, F, F)
where
  F: Field + DelayedReduction<SV>,
  SV: WideMul,
{
  let prefix_size = weights.len();
  debug_assert!(prefix_size > 0);
  debug_assert!(num_constraints > 0);
  debug_assert_eq!(prefix_workspace.num_constraints, num_constraints);
  debug_assert_eq!(prefix_workspace.prefix_size, prefix_size);

  let lo_start = (lo_suffix * num_constraints + constraint_idx) * prefix_size;
  let hi_start = (hi_suffix * num_constraints + constraint_idx) * prefix_size;
  let a_lo = &prefix_workspace.a[lo_start..lo_start + prefix_size];
  let a_hi = &prefix_workspace.a[hi_start..hi_start + prefix_size];
  let b_lo = &prefix_workspace.b[lo_start..lo_start + prefix_size];
  let b_hi = &prefix_workspace.b[hi_start..hi_start + prefix_size];
  let c_lo = &prefix_workspace.c[lo_start..lo_start + prefix_size];
  let c_hi = &prefix_workspace.c[hi_start..hi_start + prefix_size];

  let mut acc_a_lo = <F as DelayedReduction<SV>>::Accumulator::zero();
  let mut acc_a_hi = <F as DelayedReduction<SV>>::Accumulator::zero();
  let mut acc_b_lo = <F as DelayedReduction<SV>>::Accumulator::zero();
  let mut acc_b_hi = <F as DelayedReduction<SV>>::Accumulator::zero();
  let mut acc_c_lo = <F as DelayedReduction<SV>>::Accumulator::zero();
  let mut acc_c_hi = <F as DelayedReduction<SV>>::Accumulator::zero();
  for ((((((weight, a0), a1), b0), b1), c0), c1) in weights
    .iter()
    .zip(a_lo.iter())
    .zip(a_hi.iter())
    .zip(b_lo.iter())
    .zip(b_hi.iter())
    .zip(c_lo.iter())
    .zip(c_hi.iter())
  {
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_a_lo, weight, a0);
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_a_hi, weight, a1);
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_b_lo, weight, b0);
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_b_hi, weight, b1);
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_c_lo, weight, c0);
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_c_hi, weight, c1);
  }
  (
    <F as DelayedReduction<SV>>::reduce(&acc_a_lo),
    <F as DelayedReduction<SV>>::reduce(&acc_a_hi),
    <F as DelayedReduction<SV>>::reduce(&acc_b_lo),
    <F as DelayedReduction<SV>>::reduce(&acc_b_hi),
    <F as DelayedReduction<SV>>::reduce(&acc_c_lo),
    <F as DelayedReduction<SV>>::reduce(&acc_c_hi),
  )
}

#[cfg(test)]
fn fold_prefix_workspace_table<F, SV>(
  weights: &[F],
  table: &[SV],
  num_constraints: usize,
) -> Vec<Vec<F>>
where
  F: Field + DelayedReduction<SV> + Send + Sync,
  SV: Send + Sync,
{
  let prefix_size = weights.len();
  debug_assert!(prefix_size > 0);
  debug_assert!(num_constraints > 0);
  debug_assert_eq!(table.len() % (num_constraints * prefix_size), 0);
  let num_prefix_rows = table.len() / prefix_size;
  let mut folded_flat = vec![F::ZERO; num_prefix_rows];

  folded_flat
    .par_iter_mut()
    .enumerate()
    .for_each(|(row_idx, out)| {
      let start = row_idx * prefix_size;
      let prefix = &table[start..start + prefix_size];
      let mut acc = <F as DelayedReduction<SV>>::Accumulator::zero();
      for (weight, value) in weights.iter().zip(prefix.iter()) {
        <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc, weight, value);
      }
      *out = <F as DelayedReduction<SV>>::reduce(&acc);
    });

  folded_flat
    .chunks(num_constraints)
    .map(|chunk| chunk.to_vec())
    .collect()
}

#[cfg(test)]
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;

  type Scalar = pallas::Scalar;

  #[test]
  fn test_prefix_workspace_fold_matches_layer_fold_with_padding() {
    let l0 = 2;
    let prefix_size = 1usize << l0;
    let n_padded = 8;
    let num_instances = 6;
    let num_constraints = 7;
    let make_table = |salt: i32| {
      (0..num_instances)
        .flat_map(|layer| {
          (0..num_constraints)
            .map(move |idx| ((layer as i32 * 13 + idx as i32 * 5 + salt) % 31) - 15)
        })
        .collect()
    };
    let small_abc = SmallAbc {
      l0,
      num_instances,
      num_constraints,
      a: make_table(0),
      b: make_table(3),
      c: make_table(7),
    };
    let workspace = PrefixWorkspace::build(&small_abc, l0, n_padded).unwrap();
    let weights = vec![
      Scalar::from(2u64),
      Scalar::from(3u64),
      Scalar::from(5u64),
      Scalar::from(7u64),
    ];

    let mut a_layers = Vec::with_capacity(n_padded);
    for idx in 0..n_padded {
      let row_idx = small_abc.padded_row_idx(idx).unwrap();
      a_layers.push(small_abc.a_row(row_idx));
    }

    let layer_fold =
      fold_small_layers_by_prefix::<Scalar, i32, _>(&weights, &a_layers, prefix_size);
    let workspace_fold =
      fold_prefix_workspace_table::<Scalar, i32>(&weights, &workspace.a, num_constraints);

    assert_eq!(layer_fold, workspace_fold);
  }
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

fn fold_ab_pair_into<E: Engine>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  src_even: usize,
  src_odd: usize,
  dest: usize,
  r: E::Scalar,
) {
  fold_layer_pair_into(a_layers, src_even, src_odd, dest, r);
  fold_layer_pair_into(b_layers, src_even, src_odd, dest, r);
}

fn compact_folded_layers_ab<E: Engine>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  prove_pairs: usize,
) {
  compact_folded_layers(a_layers, prove_pairs);
  compact_folded_layers(b_layers, prove_pairs);
}

fn compact_folded_layers<F>(layers: &mut [Vec<F>], prove_pairs: usize) {
  for j in 0..prove_pairs {
    layers.swap(2 * j, 4 * j);
    layers.swap(2 * j + 1, 4 * j + 2);
  }
}

fn fold_scalar_pair_into<F: Field>(
  values: &mut [F],
  src_even: usize,
  src_odd: usize,
  dest: usize,
  r: F,
) {
  let even = values[src_even];
  values[dest] = even + r * (values[src_odd] - even);
}

fn compact_folded_scalars<F>(values: &mut [F], prove_pairs: usize) {
  for j in 0..prove_pairs {
    values.swap(2 * j, 4 * j);
    values.swap(2 * j + 1, 4 * j + 2);
  }
}

fn fold_final_ab_pairs<E: Engine>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  pairs: usize,
  r: E::Scalar,
) {
  fold_final_layer_pairs(a_layers, pairs, r);
  fold_final_layer_pairs(b_layers, pairs, r);
}

fn fold_final_layer_pairs<F>(layers: &mut [Vec<F>], pairs: usize, r: F)
where
  F: Field + Send + Sync,
{
  if pairs == 1 {
    let (lo, hi) = layers.split_at_mut(1);
    lo[0]
      .par_iter_mut()
      .zip(hi[0].par_iter())
      .for_each(|(l, h)| *l += r * (*h - *l));
    return;
  }

  layers[..2 * pairs].par_chunks_mut(2).for_each(|chunk| {
    let (lo, hi) = chunk.split_at_mut(1);
    lo[0]
      .iter_mut()
      .zip(hi[0].iter())
      .for_each(|(l, h)| *l += r * (*h - *l));
  });

  for i in 0..pairs {
    layers.swap(i, 2 * i);
  }
}

fn fold_prefix_workspace_final_table<F, SV>(
  weights: &[F],
  table: &[SV],
  num_constraints: usize,
  prefix_size: usize,
) -> Vec<F>
where
  F: Field + DelayedReduction<SV> + Send + Sync,
  SV: Send + Sync,
{
  debug_assert!(num_constraints > 0);
  debug_assert!(prefix_size > 0);
  debug_assert_eq!(weights.len() % prefix_size, 0);
  let suffix_groups = weights.len() / prefix_size;
  debug_assert_eq!(table.len(), suffix_groups * num_constraints * prefix_size);

  (0..num_constraints)
    .into_par_iter()
    .map(|constraint_idx| {
      let mut acc = <F as DelayedReduction<SV>>::Accumulator::zero();
      for suffix_idx in 0..suffix_groups {
        let table_base = (suffix_idx * num_constraints + constraint_idx) * prefix_size;
        let weight_base = suffix_idx * prefix_size;
        for prefix_idx in 0..prefix_size {
          <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(
            &mut acc,
            &weights[weight_base + prefix_idx],
            &table[table_base + prefix_idx],
          );
        }
      }
      <F as DelayedReduction<SV>>::reduce(&acc)
    })
    .collect()
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
) -> Result<(E::Scalar, Duration), SpartanError>
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

  let (_vc_span, vc_t) = start_span!("nifs_vc_commit_round", round = round);
  let chals =
    SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, round, transcript)?;
  let vc_elapsed = vc_t.elapsed();
  info!(
    elapsed_ms = %vc_elapsed.as_millis(),
    round,
    "nifs_vc_commit_round"
  );
  let r_b = chals[0];
  r_bs.push(r_b);
  *acc_eq *= (E::Scalar::ONE - r_b) * (E::Scalar::ONE - rho_t) + r_b * rho_t;
  *t_cur = poly_t.evaluate(&r_b);

  Ok((r_b, vc_elapsed))
}

fn fold_and_update_vc_field<E, X, W>(
  S: &SplitR1CSShape<E>,
  ck: &CommitmentKey<E>,
  r_bs: &[E::Scalar],
  T_cur: E::Scalar,
  acc_eq: E::Scalar,
  Us: &[R1CSInstance<E, X>],
  Ws: &[R1CSWitness<E, W>],
  ell_b: usize,
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut MultiRoundState<E>,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
) -> Result<(R1CSWitness<E>, R1CSInstance<E>, Duration), SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  X: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
  W: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
{
  let T_out = T_cur
    * acc_eq
      .invert()
      .into_option()
      .ok_or(SpartanError::DivisionByZero)?;
  vc.t_out_step = T_out;
  vc.eq_rho_at_rb = acc_eq;
  let (_vc_span, vc_t) = start_span!("nifs_vc_commit_final", round = ell_b);
  SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, ell_b, transcript)?;
  let vc_elapsed = vc_t.elapsed();
  info!(
    elapsed_ms = %vc_elapsed.as_millis(),
    round = ell_b,
    "nifs_vc_commit_final"
  );

  let (_fold_wu_span, fold_wu_t) = start_span!("nifs_fold_wu");
  let weights = weights_from_r::<E::Scalar>(r_bs, Us.len());

  let full_dim = S.num_shared + S.num_precommitted + S.num_rest;
  let effective_len = S.num_shared + S.num_precommitted;
  let use_truncated_fold = effective_len > 0 && effective_len < full_dim;

  let (_fold_span, fold_t) = start_span!("fold_witnesses");
  let folded_W = if use_truncated_fold {
    fold_native_witness_prefix_into_field_with_weights::<E, W>(&weights, Ws, effective_len)?
  } else {
    R1CSWitness::<E, W>::fold_multiple_into_field_with_weights(&weights, Ws)?
  };
  info!(elapsed_ms = %fold_t.elapsed().as_millis(), "fold_witnesses");

  let (_fold_span, fold_t) = start_span!("fold_instances");
  let folded_U = fold_native_instance_into_field_with_partial_commitment::<E, X>(
    &weights,
    Us,
    use_truncated_fold,
    effective_len,
    &folded_W.r_W,
    ck,
  )?;
  info!(elapsed_ms = %fold_t.elapsed().as_millis(), "fold_instances");
  info!(
    elapsed_ms = %fold_wu_t.elapsed().as_millis(),
    instances = Us.len(),
    witness_len = full_dim,
    folded_witness_len = effective_len,
    truncated = use_truncated_fold,
    threads = rayon::current_num_threads(),
    "nifs_fold_wu"
  );

  Ok((folded_W, folded_U, vc_elapsed))
}

fn fold_native_witness_prefix_into_field_with_weights<E, W>(
  weights: &[E::Scalar],
  Ws: &[R1CSWitness<E, W>],
  effective_len: usize,
) -> Result<R1CSWitness<E>, SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  W: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
{
  let n = Ws.len();
  if n == 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: "fold_native_witness_prefix: empty witness list".into(),
    });
  }
  if weights.len() != n {
    return Err(SpartanError::InvalidInputLength {
      reason: "fold_native_witness_prefix: weights length mismatch".into(),
    });
  }
  let full_dim = Ws[0].W.len();
  if effective_len > full_dim {
    return Err(SpartanError::InvalidInputLength {
      reason: "fold_native_witness_prefix: effective length exceeds witness length".into(),
    });
  }
  if !Ws.iter().all(|w| w.W.len() == full_dim) {
    return Err(SpartanError::InvalidInputLength {
      reason: "fold_native_witness_prefix: all W vectors must have the same length".into(),
    });
  }

  let mut folded = vec![E::Scalar::ZERO; effective_len];
  folded.par_iter_mut().enumerate().for_each(|(j, acc)| {
    let mut out = E::Scalar::ZERO;
    for (i, weight) in weights.iter().enumerate() {
      let value = Ws[i].W[j].to_scalar();
      if value == E::Scalar::ZERO {
        continue;
      }
      if value == E::Scalar::ONE {
        out += *weight;
      } else {
        out += *weight * value;
      }
    }
    *acc = out;
  });
  folded.resize(full_dim, E::Scalar::ZERO);

  let folded_blind = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
    &Ws.iter().map(|w| w.r_W.clone()).collect::<Vec<_>>(),
    weights,
  )?;

  Ok(R1CSWitness::<E> {
    W: folded,
    r_W: folded_blind,
    is_small: false,
  })
}

fn fold_native_instance_into_field_with_partial_commitment<E, X>(
  weights: &[E::Scalar],
  Us: &[R1CSInstance<E, X>],
  use_partial_commitment: bool,
  effective_len: usize,
  folded_blind: &<E::PCS as PCSEngineTrait<E>>::Blind,
  ck: &CommitmentKey<E>,
) -> Result<R1CSInstance<E>, SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  X: R1CSValue<E> + Serialize + for<'de> Deserialize<'de>,
{
  let n = Us.len();
  if n == 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: "fold_native_instance: empty instance list".into(),
    });
  }
  if weights.len() != n {
    return Err(SpartanError::InvalidInputLength {
      reason: "fold_native_instance: weights length mismatch".into(),
    });
  }
  let dim = Us[0].X.len();
  if !Us.iter().all(|u| u.X.len() == dim) {
    return Err(SpartanError::InvalidInputLength {
      reason: "fold_native_instance: all X vectors must have the same length".into(),
    });
  }

  let mut X_acc = vec![E::Scalar::ZERO; dim];
  for (i, Ui) in Us.iter().enumerate() {
    let wi = weights[i];
    for (j, Uij) in Ui.X.iter().enumerate() {
      let value = Uij.to_scalar();
      if value == E::Scalar::ZERO {
        continue;
      }
      if value == E::Scalar::ONE {
        X_acc[j] += wi;
      } else {
        X_acc[j] += wi * value;
      }
    }
  }

  let comms: Vec<_> = Us.iter().map(|U| U.comm_W.clone()).collect();
  let comm_acc = if use_partial_commitment {
    let num_data_rows = effective_len.div_ceil(DEFAULT_COMMITMENT_WIDTH);
    <E::PCS as FoldingEngineTrait<E>>::fold_commitments_partial(
      &comms,
      weights,
      num_data_rows,
      folded_blind,
      ck,
    )?
  } else {
    <E::PCS as FoldingEngineTrait<E>>::fold_commitments(&comms, weights)?
  };

  R1CSInstance::<E>::new_unchecked(comm_acc, X_acc)
}
