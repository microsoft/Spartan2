//! This module supports generating R1CS instance-witness pairs for test constraint systems.
//! This currently does not support generating split R1CS instances, but it can be extended to do so.
use crate::{
  CommitmentKey, VerifierKey,
  bellpepper::{r1cs::add_constraint, solver::SatisfyingAssignment, test_shape_cs::TestShapeCS},
  errors::SpartanError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, SparseMatrix},
  start_span,
  traits::Engine,
};
use std::time::Instant;
use tracing::{info, info_span};

/// `TestSpartanShape` provides methods for acquiring `R1CSShape` and `CommitmentKey` from implementers.
pub trait TestSpartanShape<E: Engine> {
  /// Return an appropriate `R1CSShape` and `CommitmentKey` structs.
  fn r1cs_shape(
    &mut self,
  ) -> Result<(R1CSShape<E>, CommitmentKey<E>, VerifierKey<E>), SpartanError>;
}

/// `TestSpartanWitness` provide a method for acquiring an `R1CSInstance` and `R1CSWitness` from implementers.
pub trait TestSpartanWitness<E: Engine> {
  /// Return an instance and witness, given a shape and ck.
  fn r1cs_instance_and_witness(
    &mut self,
    S: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
    is_small: bool,
  ) -> Result<(R1CSInstance<E>, R1CSWitness<E>), SpartanError>;
}

impl<E: Engine> TestSpartanShape<E> for TestShapeCS<E> {
  fn r1cs_shape(
    &mut self,
  ) -> Result<(R1CSShape<E>, CommitmentKey<E>, VerifierKey<E>), SpartanError> {
    let mut A = SparseMatrix::<E::Scalar>::empty();
    let mut B = SparseMatrix::<E::Scalar>::empty();
    let mut C = SparseMatrix::<E::Scalar>::empty();

    let mut num_cons_added = 0;
    let mut X = (&mut A, &mut B, &mut C, &mut num_cons_added);
    let num_inputs = self.num_inputs();
    let num_constraints = self.num_constraints();
    let num_vars = self.num_aux();

    for constraint in self.constraints.iter() {
      add_constraint(
        &mut X,
        num_vars,
        &constraint.0,
        &constraint.1,
        &constraint.2,
      );
    }
    assert_eq!(num_cons_added, num_constraints);

    A.cols = num_vars + num_inputs;
    B.cols = num_vars + num_inputs;
    C.cols = num_vars + num_inputs;

    // Don't count One as an input for shape's purposes.
    let S = R1CSShape::new(num_constraints, num_vars, num_inputs - 1, A, B, C).unwrap();
    let (ck, vk) = S.commitment_key();

    Ok((S, ck, vk))
  }
}

impl<E: Engine> TestSpartanWitness<E> for SatisfyingAssignment<E> {
  fn r1cs_instance_and_witness(
    &mut self,
    shape: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
    is_small: bool,
  ) -> Result<(R1CSInstance<E>, R1CSWitness<E>), SpartanError> {
    let (_witness_span, witness_t) = start_span!("create_r1cs_witness");
    let (W, comm_W) = R1CSWitness::<E>::new(ck, shape, &mut self.aux_assignment, is_small)?;
    info!(elapsed_ms = %witness_t.elapsed().as_millis(), "create_r1cs_witness");

    let (_instance_span, instance_t) = start_span!("create_r1cs_instance");
    let X = &self.input_assignment[1..];
    let instance = R1CSInstance::<E>::new(shape, &comm_W, X)?;
    info!(elapsed_ms = %instance_t.elapsed().as_millis(), "create_r1cs_instance");

    Ok((instance, W))
  }
}
