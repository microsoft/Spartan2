//! Support for generating R1CS witness using bellpepper.
use crate::traits::Engine;
use bellpepper_core::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use ff::{Field, PrimeField};
use serde::{Deserialize, Serialize};

/// A `ConstraintSystem` which calculates witness values for a concrete instance of an R1CS circuit.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SatisfyingAssignment<E: Engine> {
  // Assignments of variables
  pub(crate) input_assignment: Vec<E::Scalar>,
  pub(crate) aux_assignment: Vec<E::Scalar>,
}
use std::fmt;

impl<E: Engine> fmt::Debug for SatisfyingAssignment<E>
where
  E::Scalar: PrimeField,
{
  fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt
      .debug_struct("SatisfyingAssignment")
      .field("input_assignment", &self.input_assignment)
      .field("aux_assignment", &self.aux_assignment)
      .finish()
  }
}

impl<E: Engine> PartialEq for SatisfyingAssignment<E> {
  fn eq(&self, other: &SatisfyingAssignment<E>) -> bool {
    self.input_assignment == other.input_assignment && self.aux_assignment == other.aux_assignment
  }
}

impl<E: Engine> ConstraintSystem<E::Scalar> for SatisfyingAssignment<E> {
  type Root = Self;

  fn new() -> Self {
    let input_assignment = vec![E::Scalar::ONE];

    Self {
      input_assignment,
      aux_assignment: vec![],
    }
  }

  fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.aux_assignment.push(f()?);

    Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
  }

  fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.input_assignment.push(f()?);

    Ok(Variable(Index::Input(self.input_assignment.len() - 1)))
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, _a: LA, _b: LB, _c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LB: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LC: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
  {
    // Do nothing: we don't care about linear-combination evaluations in this context.
  }

  fn push_namespace<NR, N>(&mut self, _: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    // Do nothing; we don't care about namespaces in this context.
  }

  fn pop_namespace(&mut self) {
    // Do nothing; we don't care about namespaces in this context.
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }

  fn is_extensible() -> bool {
    true
  }

  fn extend(&mut self, other: &Self) {
    self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[1..]);
    self.aux_assignment.extend(other.aux_assignment.clone());
  }

  fn is_witness_generator(&self) -> bool {
    true
  }

  fn extend_inputs(&mut self, new_inputs: &[E::Scalar]) {
    self.input_assignment.extend(new_inputs);
  }

  fn extend_aux(&mut self, new_aux: &[E::Scalar]) {
    self.aux_assignment.extend(new_aux);
  }

  fn allocate_empty(
    &mut self,
    aux_n: usize,
    inputs_n: usize,
  ) -> (&mut [E::Scalar], &mut [E::Scalar]) {
    let allocated_aux = {
      let i = self.aux_assignment.len();
      self.aux_assignment.resize(aux_n + i, E::Scalar::ZERO);
      &mut self.aux_assignment[i..]
    };

    let allocated_inputs = {
      let i = self.input_assignment.len();
      self.input_assignment.resize(inputs_n + i, E::Scalar::ZERO);
      &mut self.input_assignment[i..]
    };

    (allocated_aux, allocated_inputs)
  }

  fn inputs_slice(&self) -> &[E::Scalar] {
    &self.input_assignment
  }

  fn aux_slice(&self) -> &[E::Scalar] {
    &self.aux_assignment
  }
}
