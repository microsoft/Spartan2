// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallMultiEq: Batched equality constraints with bounded coefficients.
//!
//! This module provides `SmallMultiEq`, a constraint system wrapper that batches
//! multiple equality constraints together while keeping coefficients within
//! `SmallMultiEqConfig` bounds. This enables the small-value sumcheck optimization.
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
//! SmallMultiEq solves this by either:
//! - For `NoBatching` (I32NoBatch<Fq>): Enforce each constraint directly
//! - For `Batching<K>` (I64Batch21<Fq>): Batch up to K constraints before flushing
//!
//! # Soundness
//!
//! SmallMultiEq is equally sound as MultiEq for bit constraints. The batching
//! uses powers of 2, so for constrained values that are bits (0 or 1), the only
//! way to make the batched sum zero is for all individual differences to be zero.
//! This is because 2^k > 2^(k-1) + ... + 2^0 = 2^k - 1.

use crate::small_field::{BatchingMode, SmallMultiEqConfig};
use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError, Variable};
use ff::PrimeField;
use std::marker::PhantomData;

/// SmallMultiEq: Batches equality constraints while keeping coefficients
/// within SmallMultiEqConfig bounds.
///
/// Unlike bellpepper's `MultiEq` which can accumulate 2^237 coefficients,
/// `SmallMultiEq` either:
/// - Enforces directly if `C::Batching = NoBatching`
/// - Flushes when approaching `MAX_COEFF_BITS` if `C::Batching = Batching<K>`
pub struct SmallMultiEq<Scalar: PrimeField, CS: ConstraintSystem<Scalar>, C: SmallMultiEqConfig> {
  cs: CS,
  /// Number of equality constraints enforced so far (for naming)
  ops: usize,
  /// Number of bits used in current batch (only relevant for Batching<K>)
  bits_used: usize,
  /// Accumulated left-hand side of batched equality
  lhs: LinearCombination<Scalar>,
  /// Accumulated right-hand side of batched equality
  rhs: LinearCombination<Scalar>,
  _config: PhantomData<C>,
}

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>, C: SmallMultiEqConfig>
  SmallMultiEq<Scalar, CS, C>
{
  /// Create a new SmallMultiEq wrapper around a constraint system.
  pub fn new(cs: CS) -> Self {
    SmallMultiEq {
      cs,
      ops: 0,
      bits_used: 0,
      lhs: LinearCombination::zero(),
      rhs: LinearCombination::zero(),
      _config: PhantomData,
    }
  }

  /// Flush the pending batched constraint to the underlying constraint system.
  /// Only used when batching is enabled.
  fn flush(&mut self) {
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

  /// Enforce that `lhs` equals `rhs`.
  ///
  /// Behavior depends on `C::Batching`:
  /// - `NoBatching`: Enforces directly as a single constraint
  /// - `Batching<K>`: Batches with coefficient 2^bits_used, flushes at K
  pub fn enforce_equal(
    &mut self,
    lhs: &LinearCombination<Scalar>,
    rhs: &LinearCombination<Scalar>,
  ) {
    // Const dispatch based on batching mode
    match <C::Batching as BatchingMode>::MAX_COEFF_BITS {
      None => {
        // NoBatching: enforce directly
        let ops = self.ops;
        self.cs.enforce(
          || format!("eq {ops}"),
          |_| lhs.clone(),
          |lc| lc + CS::one(),
          |_| rhs.clone(),
        );
        self.ops += 1;
      }
      Some(max_bits) => {
        // Batching<K>: accumulate with coefficients 2^bits_used
        if self.bits_used >= max_bits {
          self.flush();
        }

        // Compute the coefficient: 2^bits_used
        let coeff = Scalar::from(1u64 << self.bits_used);

        // Scale and accumulate
        self.lhs = self.lhs.clone() + (coeff, lhs);
        self.rhs = self.rhs.clone() + (coeff, rhs);

        self.bits_used += 1;
      }
    }
  }
}

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>, C: SmallMultiEqConfig> Drop
  for SmallMultiEq<Scalar, CS, C>
{
  fn drop(&mut self) {
    // Flush any pending constraints when going out of scope
    // (only relevant for Batching<K> mode)
    if <C::Batching as BatchingMode>::MAX_COEFF_BITS.is_some() {
      self.flush();
    }
  }
}

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>, C: SmallMultiEqConfig> ConstraintSystem<Scalar>
  for SmallMultiEq<Scalar, CS, C>
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
    // DO NOT flush here - R1CS constraints are unordered, so there's no
    // soundness reason to flush. Only flush on capacity limit or Drop.
    // This allows batching to actually work across enforce() calls.
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::small_field::{Batching, I32NoBatch, I64Batch21, SmallValueField};
  use bellpepper_core::test_cs::TestConstraintSystem;
  use halo2curves::pasta::Fq;
  use std::marker::PhantomData;

  // Test config with custom batching limit for testing flush behavior
  struct TestBatching5<S>(PhantomData<S>);
  impl<S: SmallValueField<i32>> SmallMultiEqConfig for TestBatching5<S> {
    type Scalar = S;
    type SmallValue = i32;
    type Batching = Batching<5>;
  }

  #[test]
  fn test_small_multi_eq_basic_small32() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // Create some variables
    let a = cs.alloc(|| "a", || Ok(Fq::from(5u64))).unwrap();
    let b = cs.alloc(|| "b", || Ok(Fq::from(5u64))).unwrap();

    {
      let mut multi_eq = SmallMultiEq::<_, _, I32NoBatch<Fq>>::new(&mut cs);

      // Enforce a == b
      let lhs = LinearCombination::zero() + a;
      let rhs = LinearCombination::zero() + b;
      multi_eq.enforce_equal(&lhs, &rhs);
    } // Drop

    assert!(cs.is_satisfied());
    // I32NoBatch<Fq> uses NoBatching, so there should be 1 direct constraint
    assert_eq!(cs.num_constraints(), 1);
  }

  #[test]
  fn test_small_multi_eq_basic_small64() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // Create some variables
    let a = cs.alloc(|| "a", || Ok(Fq::from(5u64))).unwrap();
    let b = cs.alloc(|| "b", || Ok(Fq::from(5u64))).unwrap();

    {
      let mut multi_eq = SmallMultiEq::<_, _, I64Batch21<Fq>>::new(&mut cs);

      // Enforce a == b
      let lhs = LinearCombination::zero() + a;
      let rhs = LinearCombination::zero() + b;
      multi_eq.enforce_equal(&lhs, &rhs);
    } // Drop flushes

    assert!(cs.is_satisfied());
    // I64Batch21<Fq> uses Batching<21>, so there should be 1 batched constraint after flush
    assert_eq!(cs.num_constraints(), 1);
  }

  #[test]
  fn test_small_multi_eq_multiple_small32() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // Create variables
    let a = cs.alloc(|| "a", || Ok(Fq::from(10u64))).unwrap();
    let b = cs.alloc(|| "b", || Ok(Fq::from(10u64))).unwrap();
    let c = cs.alloc(|| "c", || Ok(Fq::from(20u64))).unwrap();
    let d = cs.alloc(|| "d", || Ok(Fq::from(20u64))).unwrap();

    {
      let mut multi_eq = SmallMultiEq::<_, _, I32NoBatch<Fq>>::new(&mut cs);

      // Enforce a == b
      multi_eq.enforce_equal(
        &(LinearCombination::zero() + a),
        &(LinearCombination::zero() + b),
      );

      // Enforce c == d
      multi_eq.enforce_equal(
        &(LinearCombination::zero() + c),
        &(LinearCombination::zero() + d),
      );
    }

    assert!(cs.is_satisfied());
    // I32NoBatch<Fq> (NoBatching): 2 direct constraints
    assert_eq!(cs.num_constraints(), 2);
  }

  #[test]
  fn test_small_multi_eq_multiple_small64() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // Create variables
    let a = cs.alloc(|| "a", || Ok(Fq::from(10u64))).unwrap();
    let b = cs.alloc(|| "b", || Ok(Fq::from(10u64))).unwrap();
    let c = cs.alloc(|| "c", || Ok(Fq::from(20u64))).unwrap();
    let d = cs.alloc(|| "d", || Ok(Fq::from(20u64))).unwrap();

    {
      let mut multi_eq = SmallMultiEq::<_, _, I64Batch21<Fq>>::new(&mut cs);

      // Enforce a == b
      multi_eq.enforce_equal(
        &(LinearCombination::zero() + a),
        &(LinearCombination::zero() + b),
      );

      // Enforce c == d
      multi_eq.enforce_equal(
        &(LinearCombination::zero() + c),
        &(LinearCombination::zero() + d),
      );
    }

    assert!(cs.is_satisfied());
    // I64Batch21<Fq> (Batching<21>): 2 constraints batched into 1
    assert_eq!(cs.num_constraints(), 1);
  }

  #[test]
  fn test_small_multi_eq_flush_at_capacity() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // TestBatching5 flushes at 5 constraints
    let n = 12; // 12 constraints = 2 full batches (5+5) + 2 remaining

    let vars: Vec<_> = (0..n)
      .map(|i| {
        cs.alloc(|| format!("v{i}"), || Ok(Fq::from(42u64)))
          .unwrap()
      })
      .collect();

    // Allocate expected before creating SmallMultiEq
    let expected = cs.alloc(|| "expected", || Ok(Fq::from(42u64))).unwrap();

    {
      let mut multi_eq = SmallMultiEq::<_, _, TestBatching5<Fq>>::new(&mut cs);

      // Enforce v[i] == expected for all i
      let expected_lc = LinearCombination::zero() + expected;

      for v in &vars {
        let lhs = LinearCombination::zero() + *v;
        multi_eq.enforce_equal(&lhs, &expected_lc);
      }
    }

    assert!(cs.is_satisfied());
    // 12 constraints with batch size 5: 5+5+2 = 3 batched constraints
    assert_eq!(cs.num_constraints(), 3);
  }

  #[test]
  fn test_small_multi_eq_no_batching_many() {
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
      let mut multi_eq = SmallMultiEq::<_, _, I32NoBatch<Fq>>::new(&mut cs);

      let expected_lc = LinearCombination::zero() + expected;

      for v in &vars {
        let lhs = LinearCombination::zero() + *v;
        multi_eq.enforce_equal(&lhs, &expected_lc);
      }
    }

    assert!(cs.is_satisfied());
    // I32NoBatch<Fq> (NoBatching): 10 direct constraints
    assert_eq!(cs.num_constraints(), 10);
  }

  #[test]
  fn test_small_multi_eq_unsatisfied() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // Create variables with different values
    let a = cs.alloc(|| "a", || Ok(Fq::from(5u64))).unwrap();
    let b = cs.alloc(|| "b", || Ok(Fq::from(10u64))).unwrap();

    {
      let mut multi_eq = SmallMultiEq::<_, _, I64Batch21<Fq>>::new(&mut cs);

      // Enforce a == b (should fail because 5 != 10)
      multi_eq.enforce_equal(
        &(LinearCombination::zero() + a),
        &(LinearCombination::zero() + b),
      );
    }

    assert!(!cs.is_satisfied());
  }
}
