// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallMultiEq: Batched equality constraints with bounded coefficients.
//!
//! This module provides the `SmallMultiEq` trait and two implementations:
//! - [`NoBatchEq`]: Each equality constraint is enforced directly (for i32 path)
//! - [`BatchingEq`]: Batches up to K constraints before flushing (for i64 path)
//!
//! # Why SmallMultiEq?
//!
//! Bellpepper's `MultiEq` batches equality constraints using powers of 2 as
//! coefficients, accumulating up to 2^237 after ~237 batched equalities.
//! This breaks small-value optimization because:
//!
//! ```text
//! Az(x) = sum_y A(x,y) * z(y)
//! ```
//!
//! Even if z(y) is small (bits), if A(x,y) = 2^237, then Az(x) is huge.
//!
//! # Design
//!
//! The `SmallMultiEq` trait extends `ConstraintSystem` with:
//! - `enforce_equal`: Batch-aware equality constraints
//! - `flush`: Force pending constraints to be emitted
//! - `addmany`: Add multiple SmallUInt32 values (algorithm determined by impl)
//!
//! Each implementation pairs a batching strategy with its compatible addmany algorithm:
//! - `NoBatchEq`: No batching + limbed addition (max coeff 2^18, fits i32)
//! - `BatchingEq<K>`: Batch K constraints + full addition (max coeff 2^34, fits i64)

use super::{addmany, small_uint32::SmallUInt32};
use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError, Variable};
use ff::PrimeField;

// ============================================================================
// SmallMultiEq Trait
// ============================================================================

/// Constraint system extension for batched equality constraints with bounded coefficients.
///
/// This trait extends `ConstraintSystem` with methods for enforcing equality
/// constraints in a way that keeps coefficients within small-value bounds.
pub trait SmallMultiEq<Scalar: PrimeField>: ConstraintSystem<Scalar> {
  /// Enforce that `lhs` equals `rhs`.
  ///
  /// The implementation determines whether this is enforced directly or batched.
  fn enforce_equal(&mut self, lhs: &LinearCombination<Scalar>, rhs: &LinearCombination<Scalar>);

  /// Flush any pending batched constraints to the underlying constraint system.
  fn flush(&mut self);

  /// Add multiple SmallUInt32 values together.
  ///
  /// The implementation determines which addition algorithm is used:
  /// - `NoBatchEq`: Uses limbed addition (max coeff 2^18)
  /// - `BatchingEq<K>`: Uses full addition (max coeff 2^34)
  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError>;
}

// ============================================================================
// NoBatchEq - Direct enforcement, limbed addition
// ============================================================================

/// Constraint system wrapper that enforces equality constraints directly.
///
/// Each call to `enforce_equal` immediately creates a constraint. This is used
/// with the limbed addition algorithm which keeps coefficients within i32 bounds.
///
/// # Example
///
/// ```ignore
/// let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
/// eq.enforce_equal(&lhs, &rhs);  // Immediately enforced
/// let sum = eq.addmany(&[a, b, c])?;  // Uses limbed addition
/// ```
pub struct NoBatchEq<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>> {
  cs: &'a mut CS,
  ops: usize,
  addmany_count: usize,
  _marker: std::marker::PhantomData<Scalar>,
}

impl<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>> NoBatchEq<'a, Scalar, CS> {
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

  fn flush(&mut self) {
    // No-op: NoBatchEq enforces constraints directly
  }

  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError> {
    assert!(Scalar::NUM_BITS >= 64);
    assert!(operands.len() >= 2);
    assert!(operands.len() <= 10);

    // Check for all-constant case
    if let Some(sum) = try_constant_sum(operands) {
      return Ok(SmallUInt32::constant(sum));
    }

    // Create a unique namespace for this addmany call
    let count = self.addmany_count;
    self.addmany_count += 1;
    self.cs.push_namespace(|| format!("add{count}"));

    // Use limbed addition (max coeff 2^18, fits i32)
    let result = addmany::limbed(self, operands);

    self.cs.pop_namespace();
    result
  }
}

// Delegate ConstraintSystem to inner cs
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

// ============================================================================
// BatchingEq - Batched enforcement, full addition
// ============================================================================

/// Constraint system wrapper that batches equality constraints.
///
/// Accumulates up to K equality constraints with coefficients 2^0, 2^1, ..., 2^(K-1)
/// before flushing as a single batched constraint. This is used with the full
/// addition algorithm which keeps coefficients within i64 bounds.
///
/// # Type Parameter
///
/// - `K`: Maximum number of constraints to batch before flushing
///
/// # Why K=21?
///
/// Batching packs multiple equality constraints into fewer constraints by forming:
/// ```text
/// B(z) = Σⱼ 2ʲ · Lⱼ(z)  for j = 0..K-1
/// ```
/// and enforcing B(z) = 0. This is safe because powers of 2 give unique representation.
///
/// The limit K=21 comes from not overflowing i64 before converting to field:
/// ```text
/// Az ≤ 200 terms × 2^34 (positional) × 2^20 (batching) × 1 (witness)
///    = 2^8 × 2^34 × 2^20 × 2^0
///    = 2^62 < 2^63 (i64 signed max)
/// ```
///
/// # Example
///
/// ```ignore
/// let mut eq = BatchingEq::<Fq, _, 21>::new(&mut cs);
/// eq.enforce_equal(&lhs1, &rhs1);  // Batched with coeff 2^0
/// eq.enforce_equal(&lhs2, &rhs2);  // Batched with coeff 2^1
/// // ... up to 21 constraints batched together
/// let sum = eq.addmany(&[a, b, c])?;  // Uses full addition
/// drop(eq);  // Flushes remaining batched constraints
/// ```
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
  /// Create a new BatchingEq wrapper around a constraint system.
  pub fn new(cs: &'a mut CS) -> Self {
    BatchingEq {
      cs,
      ops: 0,
      addmany_count: 0,
      bits_used: 0,
      lhs: LinearCombination::zero(),
      rhs: LinearCombination::zero(),
    }
  }

  /// Flush the pending batched constraint to the underlying constraint system.
  fn do_flush(&mut self) {
    let ops = self.ops;
    let lhs = std::mem::replace(&mut self.lhs, LinearCombination::zero());
    let rhs = std::mem::replace(&mut self.rhs, LinearCombination::zero());

    // Only enforce if we have accumulated something
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

    // Compute the coefficient: 2^bits_used
    let coeff = Scalar::from(1u64 << self.bits_used);

    // Scale and accumulate
    self.lhs = self.lhs.clone() + (coeff, lhs);
    self.rhs = self.rhs.clone() + (coeff, rhs);

    self.bits_used += 1;
  }

  fn flush(&mut self) {
    self.do_flush();
  }

  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError> {
    assert!(Scalar::NUM_BITS >= 64);
    assert!(operands.len() >= 2);
    assert!(operands.len() <= 10);

    // Check for all-constant case
    if let Some(sum) = try_constant_sum(operands) {
      return Ok(SmallUInt32::constant(sum));
    }

    // Create a unique namespace for this addmany call
    let count = self.addmany_count;
    self.addmany_count += 1;
    self.cs.push_namespace(|| format!("add{count}"));

    // Use full addition (max coeff 2^34, fits i64)
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

// Delegate ConstraintSystem to inner cs
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

// ============================================================================
// Helper Functions
// ============================================================================

/// Try to compute sum as a constant if all operands are constant.
fn try_constant_sum(operands: &[SmallUInt32]) -> Option<u32> {
  // Check if all operands have known values
  if !operands.iter().all(|op| op.get_value().is_some()) {
    return None;
  }

  // Check if all bits are constant
  let all_constant = operands.iter().all(|op| {
    op.bits_le()
      .iter()
      .all(|b| matches!(b, bellpepper_core::boolean::Boolean::Constant(_)))
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::test_cs::TestConstraintSystem;
  use halo2curves::pasta::Fq;

  #[test]
  fn test_no_batch_eq_basic() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = cs.alloc(|| "a", || Ok(Fq::from(5u64))).unwrap();
    let b = cs.alloc(|| "b", || Ok(Fq::from(5u64))).unwrap();

    {
      let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
      let lhs = LinearCombination::zero() + a;
      let rhs = LinearCombination::zero() + b;
      eq.enforce_equal(&lhs, &rhs);
    }

    assert!(cs.is_satisfied());
    assert_eq!(cs.num_constraints(), 1);
  }

  #[test]
  fn test_batching_eq_basic() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = cs.alloc(|| "a", || Ok(Fq::from(5u64))).unwrap();
    let b = cs.alloc(|| "b", || Ok(Fq::from(5u64))).unwrap();

    {
      let mut eq = BatchingEq::<Fq, _, 21>::new(&mut cs);
      let lhs = LinearCombination::zero() + a;
      let rhs = LinearCombination::zero() + b;
      eq.enforce_equal(&lhs, &rhs);
    } // Drop flushes

    assert!(cs.is_satisfied());
    assert_eq!(cs.num_constraints(), 1);
  }

  #[test]
  fn test_no_batch_eq_multiple() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = cs.alloc(|| "a", || Ok(Fq::from(10u64))).unwrap();
    let b = cs.alloc(|| "b", || Ok(Fq::from(10u64))).unwrap();
    let c = cs.alloc(|| "c", || Ok(Fq::from(20u64))).unwrap();
    let d = cs.alloc(|| "d", || Ok(Fq::from(20u64))).unwrap();

    {
      let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
      eq.enforce_equal(
        &(LinearCombination::zero() + a),
        &(LinearCombination::zero() + b),
      );
      eq.enforce_equal(
        &(LinearCombination::zero() + c),
        &(LinearCombination::zero() + d),
      );
    }

    assert!(cs.is_satisfied());
    // NoBatchEq: 2 direct constraints
    assert_eq!(cs.num_constraints(), 2);
  }

  #[test]
  fn test_batching_eq_multiple() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = cs.alloc(|| "a", || Ok(Fq::from(10u64))).unwrap();
    let b = cs.alloc(|| "b", || Ok(Fq::from(10u64))).unwrap();
    let c = cs.alloc(|| "c", || Ok(Fq::from(20u64))).unwrap();
    let d = cs.alloc(|| "d", || Ok(Fq::from(20u64))).unwrap();

    {
      let mut eq = BatchingEq::<Fq, _, 21>::new(&mut cs);
      eq.enforce_equal(
        &(LinearCombination::zero() + a),
        &(LinearCombination::zero() + b),
      );
      eq.enforce_equal(
        &(LinearCombination::zero() + c),
        &(LinearCombination::zero() + d),
      );
    }

    assert!(cs.is_satisfied());
    // BatchingEq<21>: 2 constraints batched into 1
    assert_eq!(cs.num_constraints(), 1);
  }

  #[test]
  fn test_batching_eq_flush_at_capacity() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // Batch size 5 for testing
    let n = 12; // 12 constraints = 2 full batches (5+5) + 2 remaining

    let vars: Vec<_> = (0..n)
      .map(|i| {
        cs.alloc(|| format!("v{i}"), || Ok(Fq::from(42u64)))
          .unwrap()
      })
      .collect();

    let expected = cs.alloc(|| "expected", || Ok(Fq::from(42u64))).unwrap();

    {
      let mut eq = BatchingEq::<Fq, _, 5>::new(&mut cs);
      let expected_lc = LinearCombination::zero() + expected;

      for v in &vars {
        let lhs = LinearCombination::zero() + *v;
        eq.enforce_equal(&lhs, &expected_lc);
      }
    }

    assert!(cs.is_satisfied());
    // 12 constraints with batch size 5: 5+5+2 = 3 batched constraints
    assert_eq!(cs.num_constraints(), 3);
  }

  #[test]
  fn test_no_batch_eq_many() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let n = 10;

    let vars: Vec<_> = (0..n)
      .map(|i| {
        cs.alloc(|| format!("v{i}"), || Ok(Fq::from(42u64)))
          .unwrap()
      })
      .collect();

    let expected = cs.alloc(|| "expected", || Ok(Fq::from(42u64))).unwrap();

    {
      let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
      let expected_lc = LinearCombination::zero() + expected;

      for v in &vars {
        let lhs = LinearCombination::zero() + *v;
        eq.enforce_equal(&lhs, &expected_lc);
      }
    }

    assert!(cs.is_satisfied());
    // NoBatchEq: 10 direct constraints
    assert_eq!(cs.num_constraints(), 10);
  }

  #[test]
  fn test_batching_eq_unsatisfied() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = cs.alloc(|| "a", || Ok(Fq::from(5u64))).unwrap();
    let b = cs.alloc(|| "b", || Ok(Fq::from(10u64))).unwrap();

    {
      let mut eq = BatchingEq::<Fq, _, 21>::new(&mut cs);
      eq.enforce_equal(
        &(LinearCombination::zero() + a),
        &(LinearCombination::zero() + b),
      );
    }

    assert!(!cs.is_satisfied());
  }

  #[test]
  fn test_no_batch_eq_addmany() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::alloc(cs.namespace(|| "a"), Some(100)).unwrap();
    let b = SmallUInt32::alloc(cs.namespace(|| "b"), Some(200)).unwrap();
    let c = SmallUInt32::alloc(cs.namespace(|| "c"), Some(300)).unwrap();

    {
      let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
      let result = eq.addmany(&[a, b, c]).unwrap();
      assert_eq!(result.get_value(), Some(600));
    }

    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_batching_eq_addmany() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::alloc(cs.namespace(|| "a"), Some(100)).unwrap();
    let b = SmallUInt32::alloc(cs.namespace(|| "b"), Some(200)).unwrap();
    let c = SmallUInt32::alloc(cs.namespace(|| "c"), Some(300)).unwrap();

    {
      let mut eq = BatchingEq::<Fq, _, 21>::new(&mut cs);
      let result = eq.addmany(&[a, b, c]).unwrap();
      assert_eq!(result.get_value(), Some(600));
    }

    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_addmany_overflow() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::alloc(cs.namespace(|| "a"), Some(0xFFFFFFFF)).unwrap();
    let b = SmallUInt32::alloc(cs.namespace(|| "b"), Some(1)).unwrap();

    {
      let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
      let result = eq.addmany(&[a, b]).unwrap();
      // Should wrap to 0
      assert_eq!(result.get_value(), Some(0));
    }

    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_addmany_5_operands() {
    // SHA-256 uses 5-operand addition
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::alloc(cs.namespace(|| "a"), Some(0x12345678)).unwrap();
    let b = SmallUInt32::alloc(cs.namespace(|| "b"), Some(0x87654321)).unwrap();
    let c = SmallUInt32::alloc(cs.namespace(|| "c"), Some(0xDEADBEEF)).unwrap();
    let d = SmallUInt32::alloc(cs.namespace(|| "d"), Some(0xCAFEBABE)).unwrap();
    let e = SmallUInt32::alloc(cs.namespace(|| "e"), Some(0x01020304)).unwrap();

    let expected = 0x12345678u32
      .wrapping_add(0x87654321)
      .wrapping_add(0xDEADBEEF)
      .wrapping_add(0xCAFEBABE)
      .wrapping_add(0x01020304);

    {
      let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
      let result = eq.addmany(&[a, b, c, d, e]).unwrap();
      assert_eq!(result.get_value(), Some(expected));
    }

    assert!(cs.is_satisfied());
  }
}
