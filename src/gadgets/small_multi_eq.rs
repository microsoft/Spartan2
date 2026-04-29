// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Batched equality constraints with bounded coefficients.

use super::{addmany, small_uint32::SmallUInt32};
use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError, Variable};
use ff::PrimeField;

/// Constraint-system extension for bounded-coefficient equality batching.
pub trait SmallMultiEq<Scalar: PrimeField>: ConstraintSystem<Scalar> {
  /// Enforce `lhs == rhs`.
  fn enforce_equal(&mut self, lhs: &LinearCombination<Scalar>, rhs: &LinearCombination<Scalar>);

  /// Flush any pending batched equality constraints.
  fn flush(&mut self);

  /// Add multiple 32-bit words modulo `2^32`.
  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError>;
}

/// Direct equality wrapper. This keeps coefficients small enough for i32 paths.
pub struct NoBatchEq<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>> {
  cs: &'a mut CS,
  ops: usize,
  addmany_count: usize,
  _marker: std::marker::PhantomData<Scalar>,
}

impl<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>> NoBatchEq<'a, Scalar, CS> {
  /// Wrap a constraint system without equality batching.
  pub fn new(cs: &'a mut CS) -> Self {
    Self {
      cs,
      ops: 0,
      addmany_count: 0,
      _marker: std::marker::PhantomData,
    }
  }
}

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>> SmallMultiEq<Scalar>
  for NoBatchEq<'_, Scalar, CS>
{
  fn enforce_equal(&mut self, lhs: &LinearCombination<Scalar>, rhs: &LinearCombination<Scalar>) {
    let ops = self.ops;
    self.cs.enforce(
      || format!("eq {ops}"),
      |_| lhs.clone(),
      |lc| lc + CS::one(),
      |_| rhs.clone(),
    );
    self.ops += 1;
  }

  fn flush(&mut self) {}

  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError> {
    assert!(Scalar::NUM_BITS >= 64);
    assert!((2..=10).contains(&operands.len()));
    if let Some(sum) = try_constant_sum(operands) {
      return Ok(SmallUInt32::constant(sum));
    }

    let count = self.addmany_count;
    self.addmany_count += 1;
    self.cs.push_namespace(|| format!("add{count}"));
    let result = addmany::limbed(self, operands);
    self.cs.pop_namespace();
    result
  }
}

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>> ConstraintSystem<Scalar>
  for NoBatchEq<'_, Scalar, CS>
{
  type Root = Self;

  fn one() -> Variable {
    CS::one()
  }

  fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.cs.alloc(annotation, f)
  }

  fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.cs.alloc_input(annotation, f)
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
  {
    self.cs.enforce(annotation, a, b, c);
  }

  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    self.cs.push_namespace(name_fn);
  }

  fn pop_namespace(&mut self) {
    self.cs.pop_namespace();
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }

  fn is_witness_generator(&self) -> bool {
    self.cs.is_witness_generator()
  }

  fn extend_inputs(&mut self, inputs: &[Scalar]) {
    self.cs.extend_inputs(inputs);
  }

  fn extend_aux(&mut self, aux: &[Scalar]) {
    self.cs.extend_aux(aux);
  }

  fn allocate_empty(&mut self, aux_n: usize, inputs_n: usize) -> (&mut [Scalar], &mut [Scalar]) {
    self.cs.allocate_empty(aux_n, inputs_n)
  }

  fn inputs_slice(&self) -> &[Scalar] {
    self.cs.inputs_slice()
  }

  fn aux_slice(&self) -> &[Scalar] {
    self.cs.aux_slice()
  }
}

/// Batching equality wrapper. `K` is the number of equalities packed per flush.
pub struct BatchingEq<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>, const K: usize> {
  cs: &'a mut CS,
  ops: usize,
  addmany_count: usize,
  bits_used: usize,
  lhs: LinearCombination<Scalar>,
  rhs: LinearCombination<Scalar>,
}

impl<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>, const K: usize>
  BatchingEq<'a, Scalar, CS, K>
{
  /// Wrap a constraint system with bounded equality batching.
  pub fn new(cs: &'a mut CS) -> Self {
    Self {
      cs,
      ops: 0,
      addmany_count: 0,
      bits_used: 0,
      lhs: LinearCombination::zero(),
      rhs: LinearCombination::zero(),
    }
  }

  fn do_flush(&mut self) {
    let ops = self.ops;
    let lhs = std::mem::replace(&mut self.lhs, LinearCombination::zero());
    let rhs = std::mem::replace(&mut self.rhs, LinearCombination::zero());

    if self.bits_used > 0 {
      self.cs.enforce(
        || format!("multieq {ops}"),
        |_| lhs,
        |lc| lc + CS::one(),
        |_| rhs,
      );
      self.ops += 1;
    }

    self.bits_used = 0;
  }
}

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>, const K: usize> SmallMultiEq<Scalar>
  for BatchingEq<'_, Scalar, CS, K>
{
  fn enforce_equal(&mut self, lhs: &LinearCombination<Scalar>, rhs: &LinearCombination<Scalar>) {
    if self.bits_used >= K {
      self.do_flush();
    }

    let coeff = Scalar::from(1u64 << self.bits_used);
    self.lhs = self.lhs.clone() + (coeff, lhs);
    self.rhs = self.rhs.clone() + (coeff, rhs);
    self.bits_used += 1;
  }

  fn flush(&mut self) {
    self.do_flush();
  }

  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError> {
    assert!(Scalar::NUM_BITS >= 64);
    assert!((2..=10).contains(&operands.len()));
    if let Some(sum) = try_constant_sum(operands) {
      return Ok(SmallUInt32::constant(sum));
    }

    let count = self.addmany_count;
    self.addmany_count += 1;
    self.cs.push_namespace(|| format!("add{count}"));
    let result = addmany::full(self, operands);
    self.cs.pop_namespace();
    result
  }
}

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>, const K: usize> Drop
  for BatchingEq<'_, Scalar, CS, K>
{
  fn drop(&mut self) {
    self.do_flush();
  }
}

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>, const K: usize> ConstraintSystem<Scalar>
  for BatchingEq<'_, Scalar, CS, K>
{
  type Root = Self;

  fn one() -> Variable {
    CS::one()
  }

  fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.cs.alloc(annotation, f)
  }

  fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.cs.alloc_input(annotation, f)
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
  {
    self.cs.enforce(annotation, a, b, c);
  }

  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    self.cs.push_namespace(name_fn);
  }

  fn pop_namespace(&mut self) {
    self.cs.pop_namespace();
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }

  fn is_witness_generator(&self) -> bool {
    self.cs.is_witness_generator()
  }

  fn extend_inputs(&mut self, inputs: &[Scalar]) {
    self.cs.extend_inputs(inputs);
  }

  fn extend_aux(&mut self, aux: &[Scalar]) {
    self.cs.extend_aux(aux);
  }

  fn allocate_empty(&mut self, aux_n: usize, inputs_n: usize) -> (&mut [Scalar], &mut [Scalar]) {
    self.cs.allocate_empty(aux_n, inputs_n)
  }

  fn inputs_slice(&self) -> &[Scalar] {
    self.cs.inputs_slice()
  }

  fn aux_slice(&self) -> &[Scalar] {
    self.cs.aux_slice()
  }
}

fn try_constant_sum(operands: &[SmallUInt32]) -> Option<u32> {
  if !operands.iter().all(|op| op.get_value().is_some()) {
    return None;
  }

  let all_constant = operands.iter().all(|op| {
    op.bits_le()
      .iter()
      .all(|b| matches!(b, bellpepper_core::boolean::Boolean::Constant(_)))
  });

  all_constant.then(|| {
    operands
      .iter()
      .map(|op| op.get_value().unwrap())
      .fold(0u32, |a, b| a.wrapping_add(b))
  })
}
