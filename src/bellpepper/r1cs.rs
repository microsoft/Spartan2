//! Support for generating R1CS using bellpepper.
use crate::{
  Blind, CommitmentKey, PCS, PartialCommitment, VerifierKey,
  bellpepper::{shape_cs::ShapeCS, solver::SatisfyingAssignment},
  errors::SpartanError,
  r1cs::{R1CSWitness, SparseMatrix, SplitR1CSInstance, SplitR1CSShape},
  start_span,
  traits::{
    Engine, circuit::SpartanCircuit, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait,
  },
};
use bellpepper::gadgets::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, Index, LinearCombination};
use ff::{Field, PrimeField};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info, info_span};

/// `SpartanShape` provides methods for acquiring `SplitR1CSShape` and `CommitmentKey` from implementers.
pub trait SpartanShape<E: Engine> {
  /// Return an appropriate `SplitR1CSShape` and `CommitmentKey` structs.
  fn r1cs_shape<C: SpartanCircuit<E>>(
    circuit: &C,
  ) -> Result<(SplitR1CSShape<E>, CommitmentKey<E>, VerifierKey<E>), SpartanError>;
}

/// `SpartanWitness` provide a method for acquiring an `SplitR1CSInstance` and `R1CSWitness` from implementers.
pub trait SpartanWitness<E: Engine> {
  /// Holds the state of the prover after committing to the precommitted portions of the witness.
  type PrecommittedState: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Return partial commitments (to shared and precommitted variables) and the partial witness vector.
  fn precommitted_witness<C: SpartanCircuit<E>>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
  ) -> Result<Self::PrecommittedState, SpartanError>;

  /// Return an instance and witness, given a shape and ck.
  fn r1cs_instance_and_witness<C: SpartanCircuit<E>>(
    ps: &mut Self::PrecommittedState,
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
    transcript: &mut E::TE,
  ) -> Result<(SplitR1CSInstance<E>, R1CSWitness<E>), SpartanError>;
}

impl<E: Engine> SpartanShape<E> for ShapeCS<E> {
  fn r1cs_shape<C: SpartanCircuit<E>>(
    circuit: &C,
  ) -> Result<(SplitR1CSShape<E>, CommitmentKey<E>, VerifierKey<E>), SpartanError> {
    let num_challenges = circuit.num_challenges();

    let mut cs: Self = Self::new();

    // allocate shared variables
    let shared = circuit
      .shared(&mut cs)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to allocate shared variables: {e}"),
      })?;

    // determine the number of shared variables
    let num_shared = cs.num_aux();
    debug!("num_shared: {}", num_shared);
    debug!("shared.len(): {}", shared.len());
    if shared.len() > num_shared {
      return Err(SpartanError::SynthesisError {
        reason: "Shared variables are not allocated correctly".to_string(),
      });
    }

    // allocate precommitted variables
    let precommitted =
      circuit
        .precommitted(&mut cs, &shared)
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Unable to allocate precommitted variables: {e}"),
        })?;

    let num_precommitted = cs.num_aux() - num_shared;
    debug!("num_precommitted: {}", num_precommitted);
    debug!("precommitted.len(): {}", precommitted.len());
    if precommitted.len() > num_precommitted {
      return Err(SpartanError::SynthesisError {
        reason: "Precommitted variables are not allocated correctly".to_string(),
      });
    }

    // synthesize the circuit
    circuit
      .synthesize(&mut cs, &shared, &precommitted, None)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize circuit: {e}"),
      })?;

    let mut A = SparseMatrix::<E::Scalar>::empty();
    let mut B = SparseMatrix::<E::Scalar>::empty();
    let mut C = SparseMatrix::<E::Scalar>::empty();

    let mut num_cons_added = 0;
    let mut X = (&mut A, &mut B, &mut C, &mut num_cons_added);
    let num_inputs = cs.num_inputs();
    let num_constraints = cs.num_constraints();
    let num_vars = cs.num_aux();

    for constraint in cs.constraints.iter() {
      add_constraint(
        &mut X,
        num_vars,
        &constraint.0,
        &constraint.1,
        &constraint.2,
      );
    }
    assert_eq!(num_cons_added, num_constraints);
    assert!(num_inputs > num_challenges);

    A.cols = num_vars + num_inputs;
    B.cols = num_vars + num_inputs;
    C.cols = num_vars + num_inputs;

    let num_rest = num_vars - num_shared - num_precommitted;

    let width = E::PCS::width();

    // Don't count One as an input for shape's purposes.
    let S = SplitR1CSShape::new(
      width,
      num_constraints,
      num_shared,
      num_precommitted,
      num_rest,
      num_inputs - 1 - num_challenges,
      num_challenges,
      A,
      B,
      C,
    )
    .unwrap();
    let (ck, vk) = S.commitment_key();

    Ok((S, ck, vk))
  }
}

pub(crate) fn add_constraint<S: PrimeField>(
  X: &mut (
    &mut SparseMatrix<S>,
    &mut SparseMatrix<S>,
    &mut SparseMatrix<S>,
    &mut usize,
  ),
  num_vars: usize,
  a_lc: &LinearCombination<S>,
  b_lc: &LinearCombination<S>,
  c_lc: &LinearCombination<S>,
) {
  let (A, B, C, nn) = X;
  let n = **nn;
  assert_eq!(n + 1, A.indptr.len(), "A: invalid shape");
  assert_eq!(n + 1, B.indptr.len(), "B: invalid shape");
  assert_eq!(n + 1, C.indptr.len(), "C: invalid shape");

  let add_constraint_component = |index: Index, coeff: &S, M: &mut SparseMatrix<S>| {
    // we add constraints to the matrix only if the associated coefficient is non-zero
    if *coeff != S::ZERO {
      match index {
        Index::Input(idx) => {
          // Inputs come last, with input 0, representing 'one',
          // at position num_vars within the witness vector.
          let idx = idx + num_vars;
          M.data.push(*coeff);
          M.indices.push(idx);
        }
        Index::Aux(idx) => {
          M.data.push(*coeff);
          M.indices.push(idx);
        }
      }
    }
  };

  for (index, coeff) in a_lc.iter() {
    add_constraint_component(index.0, coeff, A);
  }
  A.indptr.push(A.indices.len());

  for (index, coeff) in b_lc.iter() {
    add_constraint_component(index.0, coeff, B)
  }
  B.indptr.push(B.indices.len());

  for (index, coeff) in c_lc.iter() {
    add_constraint_component(index.0, coeff, C)
  }
  C.indptr.push(C.indices.len());

  **nn += 1;
}

/// A type that holds the pre-processed state for proving
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PrecommittedState<E: Engine> {
  cs: SatisfyingAssignment<E>,
  shared: Vec<AllocatedNum<E::Scalar>>,
  precommitted: Vec<AllocatedNum<E::Scalar>>,
  comm_W_shared: Option<PartialCommitment<E>>,
  comm_W_precommitted: Option<PartialCommitment<E>>,
  W: Vec<E::Scalar>,
  r_W: Blind<E>,
  r_W_remaining: Blind<E>,
}

impl<E: Engine> SpartanWitness<E> for SatisfyingAssignment<E> {
  type PrecommittedState = PrecommittedState<E>;

  fn precommitted_witness<C: SpartanCircuit<E>>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
  ) -> Result<Self::PrecommittedState, SpartanError> {
    let (_synth_span, synth_t) = start_span!("precommitted_witness_synthesize");
    let mut cs: Self = Self::new();

    let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
    debug!("num_vars: {}", num_vars);

    // produce blinds for all commitments we will send
    let r_W = PCS::<E>::blind(ck);
    let mut W = vec![E::Scalar::ZERO; num_vars];

    // produce shared witness variables and commit to them
    let shared = circuit
      .shared(&mut cs)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to allocate shared variables: {e}"),
      })?;

    // we know that W is large enough to hold all shared variables
    if cs.aux_assignment.len() < S.num_shared_unpadded {
      return Err(SpartanError::SynthesisError {
        reason: "Shared variables are not allocated correctly".to_string(),
      });
    }
    W[..S.num_shared_unpadded].copy_from_slice(&cs.aux_assignment[..S.num_shared_unpadded]);

    // partial commitment to shared witness variables; we send None for full commitment as we don't have the full commitment yet
    let (_commit_span, commit_t) = start_span!("commit_witness_shared");
    let (comm_W_shared, r_W_remaining) = if S.num_shared_unpadded > 0 {
      let (comm_W_shared, r_remaining) =
        PCS::<E>::commit_partial(ck, &W[0..S.num_shared], &r_W, is_small)?;
      (Some(comm_W_shared), r_remaining)
    } else {
      (None, r_W.clone())
    };
    info!(elapsed_ms = %commit_t.elapsed().as_millis(), "commit_witness_shared");

    // produce precommitted witness variables
    let precommitted =
      circuit
        .precommitted(&mut cs, &shared)
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Unable to allocate precommitted variables: {e}"),
        })?;

    if cs.aux_assignment[S.num_shared_unpadded..].len() < S.num_precommitted_unpadded {
      return Err(SpartanError::SynthesisError {
        reason: "Precommitted variables are not allocated correctly".to_string(),
      });
    }
    W[S.num_shared..S.num_shared + S.num_precommitted_unpadded].copy_from_slice(
      &cs.aux_assignment
        [S.num_shared_unpadded..S.num_shared_unpadded + S.num_precommitted_unpadded],
    );

    // partial commitment to precommitted witness variables
    let (_commit_precommitted_span, commit_precommitted_t) =
      start_span!("commit_witness_precommitted");
    let (comm_W_precommitted, r_W_remaining) = if S.num_precommitted_unpadded > 0 {
      let (comm_W_precommitted, r_W_remaining) = PCS::<E>::commit_partial(
        ck,
        &W[S.num_shared..S.num_shared + S.num_precommitted],
        &r_W_remaining,
        is_small,
      )?;
      (Some(comm_W_precommitted), r_W_remaining)
    } else {
      (None, r_W_remaining)
    };
    info!(elapsed_ms = %commit_precommitted_t.elapsed().as_millis(), "commit_witness_precommitted");
    info!(elapsed_ms = %synth_t.elapsed().as_millis(), "precommitted_witness_synthesize");

    Ok(PrecommittedState {
      cs,
      shared,
      precommitted,
      comm_W_shared,
      comm_W_precommitted,
      W,
      r_W,
      r_W_remaining,
    })
  }

  fn r1cs_instance_and_witness<C: SpartanCircuit<E>>(
    ps: &mut Self::PrecommittedState,
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
    transcript: &mut E::TE,
  ) -> Result<(SplitR1CSInstance<E>, R1CSWitness<E>), SpartanError> {
    let (_synth_span, synth_t) = start_span!("circuit_synthesize_rest");

    // partial commitment to precommitted witness variables
    if let Some(comm_W_shared) = &ps.comm_W_shared {
      transcript.absorb(b"comm_W_shared", comm_W_shared);
    }
    if let Some(comm_W_precommitted) = &ps.comm_W_precommitted {
      transcript.absorb(b"comm_W_precommitted", comm_W_precommitted);
    }

    let challenges = (0..S.num_challenges)
      .map(|_| transcript.squeeze(b"challenge"))
      .collect::<Result<Vec<E::Scalar>, SpartanError>>()?;

    circuit
      .synthesize(&mut ps.cs, &ps.shared, &ps.precommitted, Some(&challenges))
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize witness: {e}"),
      })?;

    for (i, s) in ps.cs.aux_assignment[S.num_shared_unpadded + S.num_precommitted_unpadded..]
      .iter()
      .enumerate()
    {
      ps.W[S.num_shared + S.num_precommitted + i] = *s;
    }

    // commit to the rest with partial commitment
    let (_commit_rest_span, commit_rest_t) = start_span!("commit_witness_rest");
    let (comm_W_rest, _r_W_remaining) = PCS::<E>::commit_partial(
      ck,
      &ps.W[S.num_shared + S.num_precommitted..],
      &ps.r_W_remaining,
      is_small,
    )?;
    info!(elapsed_ms = %commit_rest_t.elapsed().as_millis(), "commit_witness_rest");
    transcript.absorb(b"comm_W_rest", &comm_W_rest); // add commitment to transcript

    let public_values = ps.cs.input_assignment[1..].to_vec()[..S.num_public].to_vec();
    let U = SplitR1CSInstance::<E>::new(
      S,
      ps.comm_W_shared.clone(),
      ps.comm_W_precommitted.clone(),
      comm_W_rest,
      public_values,
      challenges,
    )?;

    let W = R1CSWitness::<E>::new_unchecked(ps.W.clone(), ps.r_W.clone(), is_small)?;

    info!(elapsed_ms = %synth_t.elapsed().as_millis(), "circuit_synthesize_rest");

    Ok((U, W))
  }
}
