//! Support for generating R1CS using bellperson.

#![allow(non_snake_case)]

use super::{shape_cs::ShapeCS, test_shape_cs::TestShapeCS};
use crate::{
  CommitmentKey,
  VerifierKey,
  errors::SpartanError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, SparseMatrix},
  //start_span,
  traits::Engine,
};
use bellpepper_core::{Index, LinearCombination};
use ff::PrimeField;
//use std::time::Instant;
//use tracing::{info, info_span};

/// `SpartanShape` provides methods for acquiring `R1CSShape` and `CommitmentKey` from implementers.
pub trait SpartanShape<E: Engine> {
  /// Return an appropriate `R1CSShape` and `CommitmentKey` structs.
  fn r1cs_shape(
    &self,
    num_shared: usize,
    num_precommitted: usize,
  ) -> (R1CSShape<E>, CommitmentKey<E>, VerifierKey<E>);
}

/// `SpartanWitness` provide a method for acquiring an `R1CSInstance` and `R1CSWitness` from implementers.
pub trait SpartanWitness<E: Engine> {
  /// Return an instance and witness, given a shape and ck.
  fn r1cs_instance_and_witness(
    &mut self,
    shape: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
    is_small: bool,
  ) -> Result<(R1CSInstance<E>, R1CSWitness<E>), SpartanError>;
}

macro_rules! impl_spartan_shape {
  ( $name:ident) => {
    impl<E: Engine> SpartanShape<E> for $name<E>
    where
      E::Scalar: PrimeField,
    {
      fn r1cs_shape(
        &self,
        num_shared: usize,
        num_precommitted: usize,
      ) -> (R1CSShape<E>, CommitmentKey<E>, VerifierKey<E>) {
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

        let num_rest = num_vars - num_shared - num_precommitted;

        // Don't count One as an input for shape's purposes.
        let S = R1CSShape::new(
          num_constraints,
          num_shared,
          num_precommitted,
          num_rest,
          num_inputs - 1,
          A,
          B,
          C,
        )
        .unwrap();
        let (ck, vk) = S.commitment_key();

        (S, ck, vk)
      }
    }
  };
}

impl_spartan_shape!(ShapeCS);
impl_spartan_shape!(TestShapeCS);

fn add_constraint<S: PrimeField>(
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

/*
Move code from gen_witness to this
impl<E: Engine> SpartanWitness<E> for SatisfyingAssignment<E>
where
  E::Scalar: PrimeField,
{
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
*/
