// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallMultiEq: Equality constraints with bounded coefficients.
//!
//! This module provides the `SmallMultiEq` trait and its implementation:
//! - [`NoBatchEq`]: Each equality constraint is enforced directly (i32 path)
//!
//! `BatchingEq` has been removed from the pure-integer path. All SHA-256
//! witnesses are bits (0/1), so NoBatchEq with limbed addition suffices.
//! Max coefficient ~2^18 fits in i32, enabling pure i128-free accumulation.
//!
//! # Why SmallMultiEq?
//!
//! Bellpepper's `MultiEq` batches equality constraints using powers of 2 as
//! coefficients, accumulating up to 2^237 after ~237 batched equalities.
//! This breaks small-value optimization because matrix entries would be huge.
//!
//! `NoBatchEq` keeps coefficients bounded at 2^18 (max from limbed addition).

use super::{addmany, small_uint32::SmallUInt32};
use crate::small_constraint_system::{
  SmallConstraintSystem, SmallLinearCombination, SmallNamespace,
};
use bellpepper_core::SynthesisError;

// ============================================================================
// SmallMultiEq Trait
// ============================================================================

/// Constraint system extension for equality constraints with bounded coefficients.
///
/// This trait extends `SmallConstraintSystem` with methods for enforcing equality
/// constraints and multi-operand addition.
pub trait SmallMultiEq<V>: SmallConstraintSystem<V> {
  /// Enforce that `lhs` equals `rhs`.
  fn enforce_equal(&mut self, lhs: &SmallLinearCombination<V>, rhs: &SmallLinearCombination<V>);

  /// Add multiple SmallUInt32 values together using limbed addition.
  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError>;
}

// ============================================================================
// NoBatchEq - Direct enforcement, limbed addition
// ============================================================================

/// Constraint system wrapper that enforces equality constraints directly.
///
/// Each call to `enforce_equal` immediately creates a constraint. Uses limbed
/// addition (max coefficient 2^18, fits i32) for multi-operand addition.
pub struct NoBatchEq<'a, V, CS: SmallConstraintSystem<V>> {
  pub(crate) cs: &'a mut CS,
  ops: usize,
  addmany_count: usize,
  _marker: std::marker::PhantomData<V>,
}

impl<'a, V, CS: SmallConstraintSystem<V>> NoBatchEq<'a, V, CS> {
  /// Create a new NoBatchEq wrapper around a constraint system.
  pub fn new(cs: &'a mut CS) -> Self {
    NoBatchEq {
      cs,
      ops: 0,
      addmany_count: 0,
      _marker: std::marker::PhantomData,
    }
  }
}

/// SmallMultiEq<i8> for witness generation path (enforce is no-op).
///
/// Allocates the same variables as the i32 shape path so witness indices match.
impl<CS: SmallConstraintSystem<i8>> SmallMultiEq<i8> for NoBatchEq<'_, i8, CS> {
  fn enforce_equal(
    &mut self,
    _lhs: &SmallLinearCombination<i8>,
    _rhs: &SmallLinearCombination<i8>,
  ) {
    // Witness generation: enforce is a no-op
  }

  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError> {
    assert!(operands.len() >= 2);
    assert!(operands.len() <= 10);

    if let Some(sum) = try_constant_sum(operands) {
      return Ok(SmallUInt32::constant(sum));
    }

    let count = self.addmany_count;
    self.addmany_count += 1;
    self.cs.push_namespace(|| format!("add{count}"));

    // Use limbed_witness: allocates same vars as limbed but no constraints
    let result = addmany::limbed_witness(self, operands);

    self.cs.pop_namespace();
    result
  }
}

impl<CS: SmallConstraintSystem<i32>> SmallMultiEq<i32> for NoBatchEq<'_, i32, CS> {
  fn enforce_equal(
    &mut self,
    lhs: &SmallLinearCombination<i32>,
    rhs: &SmallLinearCombination<i32>,
  ) {
    let ops = self.ops;
    // Enforce: lhs - rhs = 0, expressed as (lhs - rhs) * ONE = 0
    let mut diff = lhs.clone();
    for (var, coeff) in &rhs.terms {
      diff.add_term(*var, -*coeff);
    }
    self.cs.enforce(
      || format!("eq {ops}"),
      diff,
      SmallLinearCombination::one(1i32), // b = 1 * ONE
      SmallLinearCombination::zero(),    // c = 0
    );
    self.ops += 1;
  }

  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError> {
    assert!(operands.len() >= 2);
    assert!(operands.len() <= 10);

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

impl<V: Copy, CS: SmallConstraintSystem<V>> SmallConstraintSystem<V> for NoBatchEq<'_, V, CS> {
  type Root = CS::Root;

  fn alloc<A, AR, F>(
    &mut self,
    annotation: A,
    f: F,
  ) -> Result<bellpepper_core::Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<V, SynthesisError>,
  {
    self.cs.alloc(annotation, f)
  }

  fn alloc_input<A, AR, F>(
    &mut self,
    annotation: A,
    f: F,
  ) -> Result<bellpepper_core::Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<V, SynthesisError>,
  {
    self.cs.alloc_input(annotation, f)
  }

  fn enforce<A, AR>(
    &mut self,
    annotation: A,
    a: SmallLinearCombination<V>,
    b: SmallLinearCombination<V>,
    c: SmallLinearCombination<V>,
  ) where
    A: FnOnce() -> AR,
    AR: Into<String>,
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
    self.cs.get_root()
  }

  fn namespace<NR, N>(&mut self, name_fn: N) -> SmallNamespace<'_, V, Self::Root>
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    self.get_root().push_namespace(name_fn);
    SmallNamespace {
      inner: self.get_root(),
      _marker: std::marker::PhantomData,
    }
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Try to compute sum as a constant if all operands are constant.
fn try_constant_sum(operands: &[SmallUInt32]) -> Option<u32> {
  if !operands.iter().all(|op| op.get_value().is_some()) {
    return None;
  }

  let all_constant = operands.iter().all(|op| {
    op.bits_le()
      .iter()
      .all(|b| matches!(b, crate::gadgets::small_boolean::SmallBoolean::Constant(_)))
  });

  if all_constant {
    let sum: u32 = operands
      .iter()
      .map(|op| op.get_value().unwrap())
      .fold(0u32, |a, b| a.wrapping_add(b));
    Some(sum)
  } else {
    None
  }
}

// ============================================================================
// BatchingEq - kept for backward compatibility with bellpepper path
// ============================================================================

/// Constraint system wrapper that batches equality constraints (bellpepper path).
///
/// This is retained for any code still using the old bellpepper-based path.
/// For the pure-integer small-value path, use `NoBatchEq` instead.
pub struct BatchingEq<
  'a,
  Scalar: ff::PrimeField,
  CS: bellpepper_core::ConstraintSystem<Scalar>,
  const K: usize,
> {
  cs: &'a mut CS,
  ops: usize,
  #[allow(dead_code)]
  addmany_count: usize,
  bits_used: usize,
  lhs: bellpepper_core::LinearCombination<Scalar>,
  rhs: bellpepper_core::LinearCombination<Scalar>,
}

impl<'a, Scalar: ff::PrimeField, CS: bellpepper_core::ConstraintSystem<Scalar>, const K: usize>
  BatchingEq<'a, Scalar, CS, K>
{
  /// Create a new BatchingEq wrapper around a constraint system.
  pub fn new(cs: &'a mut CS) -> Self {
    BatchingEq {
      cs,
      ops: 0,
      addmany_count: 0,
      bits_used: 0,
      lhs: bellpepper_core::LinearCombination::zero(),
      rhs: bellpepper_core::LinearCombination::zero(),
    }
  }

  fn do_flush(&mut self) {
    let ops = self.ops;
    let lhs = std::mem::replace(&mut self.lhs, bellpepper_core::LinearCombination::zero());
    let rhs = std::mem::replace(&mut self.rhs, bellpepper_core::LinearCombination::zero());

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

impl<Scalar: ff::PrimeField, CS: bellpepper_core::ConstraintSystem<Scalar>, const K: usize> Drop
  for BatchingEq<'_, Scalar, CS, K>
{
  fn drop(&mut self) {
    self.do_flush();
  }
}

impl<Scalar: ff::PrimeField, CS: bellpepper_core::ConstraintSystem<Scalar>, const K: usize>
  bellpepper_core::ConstraintSystem<Scalar> for BatchingEq<'_, Scalar, CS, K>
{
  type Root = Self;

  fn one() -> bellpepper_core::Variable {
    CS::one()
  }

  fn alloc<F, A, AR>(
    &mut self,
    annotation: A,
    f: F,
  ) -> Result<bellpepper_core::Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.cs.alloc(annotation, f)
  }

  fn alloc_input<F, A, AR>(
    &mut self,
    annotation: A,
    f: F,
  ) -> Result<bellpepper_core::Variable, SynthesisError>
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
    LA: FnOnce(
      bellpepper_core::LinearCombination<Scalar>,
    ) -> bellpepper_core::LinearCombination<Scalar>,
    LB: FnOnce(
      bellpepper_core::LinearCombination<Scalar>,
    ) -> bellpepper_core::LinearCombination<Scalar>,
    LC: FnOnce(
      bellpepper_core::LinearCombination<Scalar>,
    ) -> bellpepper_core::LinearCombination<Scalar>,
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

/// Dummy SmallMultiEq impl for BatchingEq to satisfy old call sites if any.
/// In practice, BatchingEq is only used on the bellpepper path.
impl<Scalar: ff::PrimeField, CS: bellpepper_core::ConstraintSystem<Scalar>, const K: usize>
  BatchingEq<'_, Scalar, CS, K>
{
  /// Enforce equality using batched constraints (bellpepper path).
  pub fn enforce_equal_field(
    &mut self,
    lhs: &bellpepper_core::LinearCombination<Scalar>,
    rhs: &bellpepper_core::LinearCombination<Scalar>,
  ) {
    if self.bits_used >= K {
      self.do_flush();
    }
    let coeff = Scalar::from(1u64 << self.bits_used);
    self.lhs = self.lhs.clone() + (coeff, lhs);
    self.rhs = self.rhs.clone() + (coeff, rhs);
    self.bits_used += 1;
  }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::small_constraint_system::SmallShapeCS;

  #[test]
  fn test_no_batch_eq_basic() {
    let mut cs = SmallShapeCS::new();

    let a = cs.alloc(|| "a", || Ok(0i32)).unwrap();
    let b = cs.alloc(|| "b", || Ok(0i32)).unwrap();

    {
      let mut eq = NoBatchEq::<i32, _>::new(&mut cs);
      let lhs = SmallLinearCombination::from_variable(a, 1i32);
      let rhs = SmallLinearCombination::from_variable(b, 1i32);
      eq.enforce_equal(&lhs, &rhs);
    }

    // 1 constraint from alloc bool a, 1 from alloc bool b, 1 from enforce_equal = 3 total? No —
    // SmallShapeCS::alloc doesn't enforce boolean, just records variable.
    // enforce_equal adds 1 constraint.
    assert_eq!(cs.num_constraints(), 1);
  }

  #[test]
  fn test_no_batch_eq_addmany() {
    let mut cs = SmallShapeCS::new();

    let a = SmallUInt32::constant(100);
    let b = SmallUInt32::constant(200);
    let c = SmallUInt32::constant(300);

    let mut eq = NoBatchEq::<i32, _>::new(&mut cs);
    let result = eq.addmany(&[a, b, c]).unwrap();
    assert_eq!(result.get_value(), Some(600));
  }
}
