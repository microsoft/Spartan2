// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Bridge: run `SmallConstraintSystem<W, C>` circuits inside a bellpepper
//! `ConstraintSystem<Scalar>`.
//!
//! `SmallToBellpepperCS` wraps a bellpepper `ConstraintSystem<Scalar>` and implements
//! `SmallConstraintSystem<W, C>`. This lets us use `small_sha256_int` with bit witnesses
//! and i32 coefficients inside the existing `SpartanCircuit<E>::precommitted` method,
//! ensuring the field-path shape matches the integer-path shape exactly.

use std::marker::PhantomData;

use bellpepper_core::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use ff::PrimeField;

use crate::small_constraint_system::{SmallCoeff, SmallConstraintSystem, SmallLinearCombination};

/// Wraps a bellpepper `ConstraintSystem<Scalar>` to implement `SmallConstraintSystem<W, C>`.
///
/// Variables are allocated using `W::to_field`.
/// Constraints are recorded as proper bellpepper constraints with `Scalar` coefficients.
pub struct SmallToBellpepperCS<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>> {
  pub(crate) cs: &'a mut CS,
  _marker: PhantomData<Scalar>,
}

impl<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>> SmallToBellpepperCS<'a, Scalar, CS> {
  /// Wrap a bellpepper constraint system.
  pub fn new(cs: &'a mut CS) -> Self {
    SmallToBellpepperCS {
      cs,
      _marker: PhantomData,
    }
  }

  /// Convert a `SmallLinearCombination<C>` to a bellpepper `LinearCombination<Scalar>`.
  fn to_lc<C: SmallCoeff>(lc: &SmallLinearCombination<C>) -> LinearCombination<Scalar> {
    let mut result = LinearCombination::zero();
    for (var, coeff) in &lc.terms {
      if *coeff == C::default() {
        continue;
      }
      result = result + (coeff.to_field::<Scalar>(), *var);
    }
    result
  }
}

impl<W: SmallCoeff, C: SmallCoeff, Scalar: PrimeField, CS: ConstraintSystem<Scalar>>
  SmallConstraintSystem<W, C> for SmallToBellpepperCS<'_, Scalar, CS>
{
  type Root = Self;

  fn alloc<A, AR, F>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<W, SynthesisError>,
  {
    self.cs.alloc(annotation, || {
      let val = f()?;
      Ok(val.to_field())
    })
  }

  fn alloc_input<A, AR, F>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<W, SynthesisError>,
  {
    self.cs.alloc_input(annotation, || {
      let val = f()?;
      Ok(val.to_field())
    })
  }

  fn enforce<A, AR>(
    &mut self,
    annotation: A,
    a: SmallLinearCombination<C>,
    b: SmallLinearCombination<C>,
    c: SmallLinearCombination<C>,
  ) where
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let lc_a = Self::to_lc(&a);
    let lc_b = Self::to_lc(&b);
    let lc_c = Self::to_lc(&c);
    self.cs.enforce(annotation, |_| lc_a, |_| lc_b, |_| lc_c);
  }

  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    self.cs.get_root().push_namespace(name_fn);
  }

  fn pop_namespace(&mut self) {
    self.cs.get_root().pop_namespace();
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }

  fn is_witness_generator(&self) -> bool {
    self.cs.is_witness_generator()
  }
}

/// A bellpepper `ConstraintSystem<Scalar>` wrapper that also implements
/// `SmallConstraintSystem<W, C>`.
///
/// This allows the field-path circuit (`SpartanCircuit<E>`) to use the same
/// `small_sha256_int` gadget as the integer-path circuit (`SmallSpartanCircuit`),
/// ensuring identical shapes.
impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>> ConstraintSystem<Scalar>
  for SmallToBellpepperCS<'_, Scalar, CS>
{
  type Root = Self;

  fn one() -> Variable {
    Variable::new_unchecked(Index::Input(0))
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
    self.cs.get_root().push_namespace(name_fn);
  }

  fn pop_namespace(&mut self) {
    self.cs.get_root().pop_namespace();
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
