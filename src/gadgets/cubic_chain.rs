// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! A parameterized circuit with parallel cubic operations for testing.
//!
//! This circuit is designed to test that both small-value and regular proving
//! paths produce valid proofs across different sumcheck round counts.
//!
//! # Constraint Count
//!
//! Each cubic operation creates 3 constraints:
//! - `x * x = x_sq` (multiplication)
//! - `x_sq * x = x_cu` (multiplication)
//! - `x_cu + x + 5 = y` (linear constraint)
//!
//! Plus 1 constraint for the final `inputize`, giving: `constraints = 3 * num_ops + 1`
//!
//! # Usage
//!
//! ```ignore
//! // Create a circuit with ~2^6 = 64 constraints (6 sumcheck rounds)
//! let circuit = CubicChainCircuit::for_rounds(6);
//!
//! // Or specify num_ops directly
//! let circuit = CubicChainCircuit::new(21); // 3*21 + 1 = 64 constraints
//! ```

use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::PrimeField;

use crate::traits::{Engine, circuit::SpartanCircuit};

/// A parameterized circuit with parallel cubic operations.
///
/// All witness values are small (fit in i64), making this suitable for
/// testing small-value sumcheck optimizations.
#[derive(Clone, Debug)]
pub struct CubicChainCircuit {
  num_ops: usize,
}

impl CubicChainCircuit {
  /// Create a circuit with the specified number of parallel cubic operations.
  ///
  /// Total constraints = 3 * num_ops + 1
  pub fn new(num_ops: usize) -> Self {
    Self { num_ops: num_ops.max(1) }
  }

  /// Create a circuit targeting approximately `2^num_rounds` constraints.
  ///
  /// After Spartan's power-of-2 padding, this will result in `num_rounds` sumcheck rounds.
  pub fn for_rounds(num_rounds: usize) -> Self {
    // constraints = 3 * num_ops + 1, padded to 2^num_rounds
    // So num_ops = (2^num_rounds - 1) / 3
    let target = 1usize << num_rounds;
    let num_ops = (target.saturating_sub(1)) / 3;
    Self::new(num_ops)
  }

  /// Returns the number of parallel cubic operations.
  pub fn num_ops(&self) -> usize {
    self.num_ops
  }

  /// Returns the expected number of constraints (before padding).
  pub fn num_constraints(&self) -> usize {
    3 * self.num_ops + 1
  }

  /// Compute the expected public output value.
  ///
  /// We output the result of the last cubic operation: `(num_ops + 1)³ + (num_ops + 1) + 5`
  pub fn expected_output<F: PrimeField>(&self) -> F {
    let x = F::from((self.num_ops + 1) as u64);
    let x_sq = x * x;
    let x_cu = x_sq * x;
    x_cu + x + F::from(5u64)
  }
}

impl<E: Engine> SpartanCircuit<E> for CubicChainCircuit {
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    Ok(vec![self.expected_output()])
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
    _: &[AllocatedNum<E::Scalar>],
    _: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    let mut last_y = None;

    for i in 0..self.num_ops {
      // x = i + 2 (small value, fits in i64)
      let x_val = E::Scalar::from((i + 2) as u64);
      let x = AllocatedNum::alloc(cs.namespace(|| format!("x_{i}")), || Ok(x_val))?;

      // x² and x³
      let x_sq = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      let x_cu = x_sq.mul(cs.namespace(|| format!("x_cu_{i}")), &x)?;

      // y = x³ + x + 5
      // Compute y_val in closure to handle ShapeCS case where values may be None
      let y = AllocatedNum::alloc(cs.namespace(|| format!("y_{i}")), || {
        let x_cu_val = x_cu.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let x_val = x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(x_cu_val + x_val + E::Scalar::from(5u64))
      })?;

      cs.enforce(
        || format!("y_{i} = x_{i}^3 + x_{i} + 5"),
        |lc| {
          lc + x_cu.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + y.get_variable(),
      );

      last_y = Some(y);
    }

    // Output the last y value
    if let Some(y) = last_y {
      y.inputize(cs.namespace(|| "output"))?;
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_cubic_chain_constraint_count() {
    // Verify the constraint formula: 3 * num_ops + 1
    assert_eq!(CubicChainCircuit::new(1).num_constraints(), 4);
    assert_eq!(CubicChainCircuit::new(5).num_constraints(), 16);
    assert_eq!(CubicChainCircuit::new(21).num_constraints(), 64);
  }

  #[test]
  fn test_cubic_chain_for_rounds() {
    // for_rounds(r) should create <= 2^r constraints
    let c2 = CubicChainCircuit::for_rounds(2);
    assert!(c2.num_constraints() <= 4);

    let c4 = CubicChainCircuit::for_rounds(4);
    assert!(c4.num_constraints() <= 16);

    let c6 = CubicChainCircuit::for_rounds(6);
    assert!(c6.num_constraints() <= 64);

    let c10 = CubicChainCircuit::for_rounds(10);
    assert!(c10.num_constraints() <= 1024);
  }
}
