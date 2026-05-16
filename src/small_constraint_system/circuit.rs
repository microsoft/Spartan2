// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Circuit trait for the small-value (integer) proving path.

use bellpepper_core::{SynthesisError, Variable};

use crate::{small_constraint_system::SmallConstraintSystem, traits::Engine};

/// A helper trait for circuits that use the pure-integer small-value path.
///
/// Unlike `SpartanCircuit<E>`, this trait works over integer values without field
/// arithmetic. `W` is the witness value type and `C` is the constraint
/// coefficient type:
/// - `W = i8` for SHA-256 bit witnesses
/// - `C = i32` for SHA-256 constraint and matrix coefficients
///
/// All SHA-256 witnesses are bits (0/1), so `i8` is sufficient. Matrix coefficients
/// (powers of 2 up to 2^18) fit in `i32`.
pub trait SmallSpartanCircuit<E: Engine, W, C>: Send + Sync + Clone {
  /// Returns the public values of the circuit as W (usually i8 bit-values).
  fn public_values(&self) -> Result<Vec<W>, SynthesisError>;

  /// Allocates shared variables in the constraint system.
  fn shared<CS: SmallConstraintSystem<W, C>>(
    &self,
    cs: &mut CS,
  ) -> Result<Vec<Variable>, SynthesisError>;

  /// Allocates precommitted variables.
  fn precommitted<CS: SmallConstraintSystem<W, C>>(
    &self,
    cs: &mut CS,
    shared: &[Variable],
  ) -> Result<Vec<Variable>, SynthesisError>;

  /// Returns the number of verifier challenges this circuit expects.
  fn num_challenges(&self) -> usize;

  /// Allocates remaining variables and constraints.
  ///
  /// `challenges` remain field-typed since they come from the transcript.
  fn synthesize<CS: SmallConstraintSystem<W, C>>(
    &self,
    cs: &mut CS,
    shared: &[Variable],
    precommitted: &[Variable],
    challenges: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError>;
}
