// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Support for generating R1CS shape using bellpepper.
//! `TestShapeCS` implements a superset of `ShapeCS`, adding non-trivial namespace support for use in testing.

use crate::traits::Engine;
use bellpepper_core::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use ff::{Field, PrimeField};
use std::collections::HashMap;

#[derive(Debug)]
enum NamedObject {
  Constraint,
  Var,
  Namespace,
}

/// `TestShapeCS` is a `ConstraintSystem` for creating `R1CSShape`s for a circuit.
pub struct TestShapeCS<E: Engine>
where
  E::Scalar: PrimeField + Field,
{
  named_objects: HashMap<String, NamedObject>,
  current_namespace: Vec<String>,
  /// All constraints added to the `TestShapeCS`.
  pub constraints: Vec<(
    LinearCombination<E::Scalar>,
    LinearCombination<E::Scalar>,
    LinearCombination<E::Scalar>,
    String,
  )>,
  inputs: Vec<String>,
  aux: Vec<String>,
}

impl<E: Engine> TestShapeCS<E>
where
  E::Scalar: PrimeField,
{
  /// Create a new, default `TestShapeCS`,
  pub fn new() -> Self {
    TestShapeCS::default()
  }

  /// Returns the number of constraints defined for this `TestShapeCS`.
  pub fn num_constraints(&self) -> usize {
    self.constraints.len()
  }

  /// Returns the number of inputs defined for this `TestShapeCS`.
  pub fn num_inputs(&self) -> usize {
    self.inputs.len()
  }

  /// Returns the number of aux inputs defined for this `TestShapeCS`.
  pub fn num_aux(&self) -> usize {
    self.aux.len()
  }

  /// Associate `NamedObject` with `path`.
  /// `path` must not already have an associated object.
  fn set_named_obj(&mut self, path: String, to: NamedObject) {
    assert!(
      !self.named_objects.contains_key(&path),
      "tried to create object at existing path: {path}"
    );

    self.named_objects.insert(path, to);
  }
}

impl<E: Engine> Default for TestShapeCS<E>
where
  E::Scalar: PrimeField,
{
  fn default() -> Self {
    let mut map = HashMap::new();
    map.insert("ONE".into(), NamedObject::Var);
    TestShapeCS {
      named_objects: map,
      current_namespace: vec![],
      constraints: vec![],
      inputs: vec![String::from("ONE")],
      aux: vec![],
    }
  }
}

impl<E: Engine> ConstraintSystem<E::Scalar> for TestShapeCS<E>
where
  E::Scalar: PrimeField,
{
  type Root = Self;

  fn alloc<F, A, AR>(&mut self, annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let path = compute_path(&self.current_namespace, &annotation().into());
    self.aux.push(path);

    Ok(Variable::new_unchecked(Index::Aux(self.aux.len() - 1)))
  }

  fn alloc_input<F, A, AR>(&mut self, annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let path = compute_path(&self.current_namespace, &annotation().into());
    self.inputs.push(path);

    Ok(Variable::new_unchecked(Index::Input(self.inputs.len() - 1)))
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LB: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LC: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
  {
    let path = compute_path(&self.current_namespace, &annotation().into());
    self.set_named_obj(path.clone(), NamedObject::Constraint);

    let a = a(LinearCombination::zero());
    let b = b(LinearCombination::zero());
    let c = c(LinearCombination::zero());

    self.constraints.push((a, b, c, path));
  }

  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    let name = name_fn().into();
    let path = compute_path(&self.current_namespace, &name);
    self.set_named_obj(path, NamedObject::Namespace);
    self.current_namespace.push(name);
  }

  fn pop_namespace(&mut self) {
    assert!(self.current_namespace.pop().is_some());
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }
}

fn compute_path(ns: &[String], this: &str) -> String {
  assert!(
    !this.chars().any(|a| a == '/'),
    "'/' is not allowed in names"
  );

  let mut name = String::new();

  let mut needs_separation = false;
  for ns in ns.iter().chain(Some(this.to_string()).iter()) {
    if needs_separation {
      name += "/";
    }

    name += ns;
    needs_separation = true;
  }

  name
}
