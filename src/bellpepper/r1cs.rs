//! Support for generating R1CS using bellpepper.
use crate::{
  CommitmentKey, PCS, VerifierKey,
  bellpepper::{shape_cs::ShapeCS, solver::SatisfyingAssignment},
  errors::SpartanError,
  r1cs::{R1CSWitness, SparseMatrix, SplitR1CSInstance, SplitR1CSShape},
  start_span,
  traits::{
    Engine, circuit::SpartanCircuit, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait,
  },
};
use bellpepper_core::{ConstraintSystem, Index, LinearCombination};
use ff::{Field, PrimeField};
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
  /// Return an instance and witness, given a shape and ck.
  fn r1cs_instance_and_witness<C: SpartanCircuit<E>>(
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

    // allocate precommitted variables
    let precommitted =
      circuit
        .precommitted(&mut cs, &shared)
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Unable to allocate precommitted variables: {e}"),
        })?;

    // synthesize the circuit
    circuit
      .synthesize(&mut cs, &shared, &precommitted, None)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize circuit: {e}"),
      })?;

    let num_shared = shared.len();
    let num_precommitted = precommitted.len();

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

impl<E: Engine> SpartanWitness<E> for SatisfyingAssignment<E> {
  fn r1cs_instance_and_witness<C: SpartanCircuit<E>>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
    transcript: &mut E::TE,
  ) -> Result<(SplitR1CSInstance<E>, R1CSWitness<E>), SpartanError> {
    let (_synth_span, synth_t) = start_span!("circuit_synthesize");
    let mut cs = Self::new();

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

    for (i, s) in shared.iter().enumerate() {
      W[i] = s.get_value().ok_or_else(|| SpartanError::SynthesisError {
        reason: "Shared variables are not allocated".to_string(),
      })?;
    }

    // partial commitment to shared witness variables; we send None for full commitment as we don't have the full commitment yet
    let (_commit_span, commit_t) = start_span!("commit_witness_shared");
    let (comm_W_shared, r_W_remaining) = if S.num_shared > 0 {
      let (comm_W_shared, r_remaining) =
        PCS::<E>::commit_partial(ck, &W[0..S.num_shared], &r_W, is_small)?;
      transcript.absorb(b"comm_W_shared", &comm_W_shared); // add commitment to transcript
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

    for (i, s) in precommitted.iter().enumerate() {
      W[S.num_shared + i] = s.get_value().ok_or_else(|| SpartanError::SynthesisError {
        reason: "Precommitted variables are not allocated".to_string(),
      })?;
    }

    // partial commitment to precommitted witness variables
    let (_commit_precommitted_span, commit_precommitted_t) =
      start_span!("commit_witness_precommitted");
    let (comm_W_precommitted, r_W_remaining) = if S.num_precommitted > 0 {
      let (comm_W_precommitted, r_W_remaining) = PCS::<E>::commit_partial(
        ck,
        &W[S.num_shared..S.num_shared + S.num_precommitted],
        &r_W_remaining,
        is_small,
      )?;
      transcript.absorb(b"comm_W_precommitted", &comm_W_precommitted); // add commitment to transcript
      (Some(comm_W_precommitted), r_W_remaining)
    } else {
      (None, r_W_remaining)
    };
    info!(elapsed_ms = %commit_precommitted_t.elapsed().as_millis(), "commit_witness_precommitted");

    let challenges = (0..S.num_challenges)
      .map(|_| transcript.squeeze(b"challenge"))
      .collect::<Result<Vec<E::Scalar>, SpartanError>>()?;

    circuit
      .synthesize(&mut cs, &shared, &precommitted, Some(&challenges))
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize witness: {e}"),
      })?;
    info!(elapsed_ms = %synth_t.elapsed().as_millis(), "circuit_synthesize");

    for (i, s) in cs.aux_assignment[shared.len() + precommitted.len()..]
      .iter()
      .enumerate()
    {
      W[S.num_shared + S.num_precommitted + i] = *s;
    }

    // commit to the rest with partial commitment
    let (_commit_rest_span, commit_rest_t) = start_span!("commit_witness_rest");
    let (comm_W_rest, _r_W_remaining) = PCS::<E>::commit_partial(
      ck,
      &W[S.num_shared + S.num_precommitted..],
      &r_W_remaining,
      is_small,
    )?;
    info!(elapsed_ms = %commit_rest_t.elapsed().as_millis(), "commit_witness_rest");
    transcript.absorb(b"comm_W_rest", &comm_W_rest); // add commitment to transcript

    let public_values = cs.input_assignment[1..].to_vec()[..S.num_public].to_vec();
    let U = SplitR1CSInstance::<E>::new(
      S,
      comm_W_shared,
      comm_W_precommitted,
      comm_W_rest,
      public_values,
      challenges,
    )?;
    let W = R1CSWitness::<E>::new_unchecked(W, r_W, is_small)?;

    Ok((U, W))
  }
}
