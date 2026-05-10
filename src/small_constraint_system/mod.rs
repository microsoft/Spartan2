// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! # SmallConstraintSystem
//!
//! A constraint system trait that separates small witness values from small
//! constraint coefficients. This enables the pure-integer proving path where:
//! - Shape extraction uses `SmallShapeCS<i32>` → `SparseMatrix<i32>`
//! - Witness generation uses `SmallSatisfyingAssignment<i8>` → `Vec<i8>`
//! - No field elements are created until the inner sumcheck boundary

pub mod bridge;
pub mod circuit;

use std::{
  marker::PhantomData,
  ops::{Add, AddAssign, Neg, Sub},
};

pub use crate::r1cs::SparseMatrix;
pub use bridge::SmallToBellpepperCS;

use bellpepper_core::{Index, SynthesisError, Variable};

// ── SmallCoeff ───────────────────────────────────────────────────────────

/// A matrix coefficient type for the small-value proving path.
///
/// Abstracts over `i32` (and potentially `i16`, `i8`) for matrix coefficients.
/// The key method is `mul_field` which multiplies a field element by this coefficient,
/// with type-specific optimizations (e.g., Barrett reduction for `i32`).
pub trait SmallCoeff:
  Copy
  + Clone
  + Default
  + Send
  + Sync
  + Add<Output = Self>
  + Sub<Output = Self>
  + AddAssign
  + Neg<Output = Self>
  + PartialEq
  + PartialOrd
{
  /// Convert this small coefficient into a prime field element.
  fn to_field<F: ff::PrimeField>(self) -> F;

  /// Multiply a field element by this coefficient (optimized dispatch).
  /// Implementations should fast-path ±1 and use Barrett/limb tricks for larger values.
  fn mul_field<F: ff::PrimeField>(self, x: &F) -> F;

  /// Whether this value is ±1 (for unit partition optimization).
  fn is_unit(&self) -> bool;

  /// Whether this value is positive (> 0).
  fn is_positive(&self) -> bool;
}

impl SmallCoeff for i8 {
  #[inline(always)]
  fn to_field<F: ff::PrimeField>(self) -> F {
    let value = F::from(self.unsigned_abs() as u64);
    if self < 0 { -value } else { value }
  }

  #[inline(always)]
  fn mul_field<F: ff::PrimeField>(self, x: &F) -> F {
    match self {
      0 => F::ZERO,
      1 => *x,
      -1 => -*x,
      2 => *x + *x,
      _ => *x * self.to_field::<F>(),
    }
  }

  #[inline(always)]
  fn is_unit(&self) -> bool {
    *self == 1 || *self == -1
  }

  #[inline(always)]
  fn is_positive(&self) -> bool {
    *self > 0
  }
}

impl SmallCoeff for i32 {
  #[inline(always)]
  fn to_field<F: ff::PrimeField>(self) -> F {
    let value = F::from(self.unsigned_abs() as u64);
    if self < 0 { -value } else { value }
  }

  #[inline(always)]
  fn mul_field<F: ff::PrimeField>(self, x: &F) -> F {
    *x * self.to_field::<F>()
  }

  #[inline(always)]
  fn is_unit(&self) -> bool {
    *self == 1 || *self == -1
  }

  #[inline(always)]
  fn is_positive(&self) -> bool {
    *self > 0
  }
}

// ── SmallLinearCombination ─────────────────────────────────────────────────

/// A linear combination over small integer coefficients.
/// Replaces bellpepper's `LinearCombination<Scalar>` for the integer path.
#[derive(Clone, Debug)]
pub struct SmallLinearCombination<C> {
  pub(crate) terms: Vec<(Variable, C)>,
}

impl<C: Copy> SmallLinearCombination<C> {
  /// Empty linear combination (zero).
  pub fn zero() -> Self {
    SmallLinearCombination { terms: vec![] }
  }

  /// LC = coeff × ONE.
  pub fn one(coeff: C) -> Self {
    SmallLinearCombination {
      terms: vec![(Variable::new_unchecked(Index::Input(0)), coeff)],
    }
  }

  /// LC = coeff × var.
  pub fn from_variable(var: Variable, coeff: C) -> Self {
    SmallLinearCombination {
      terms: vec![(var, coeff)],
    }
  }

  /// Add a term: LC += coeff × var.
  pub fn add_term(&mut self, var: Variable, coeff: C) {
    self.terms.push((var, coeff));
  }

  /// Subtract a term: LC -= coeff × var (coeff must already be negated).
  pub fn sub_term(&mut self, var: Variable, neg_coeff: C) {
    self.terms.push((var, neg_coeff));
  }
}

// ── SmallConstraintSystem ──────────────────────────────────────────────────

/// A constraint system over small integer witnesses and coefficients.
///
/// Mirrors bellpepper's `ConstraintSystem<Scalar>` but without any field arithmetic.
/// Used for the pure-integer proving path.
pub trait SmallConstraintSystem<W, C>: Sized {
  /// The root constraint system type (for namespace delegation).
  type Root: SmallConstraintSystem<W, C>;

  /// Allocate an auxiliary variable.
  fn alloc<A, AR, F>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<W, SynthesisError>;

  /// Allocate an input variable.
  fn alloc_input<A, AR, F>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<W, SynthesisError>;

  /// Enforce a constraint: a(x) × b(x) = c(x).
  fn enforce<A, AR>(
    &mut self,
    annotation: A,
    a: SmallLinearCombination<C>,
    b: SmallLinearCombination<C>,
    c: SmallLinearCombination<C>,
  ) where
    A: FnOnce() -> AR,
    AR: Into<String>;

  /// Push a namespace scope for variable naming.
  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR;

  /// Pop the current namespace scope.
  fn pop_namespace(&mut self);

  /// Get a mutable reference to the root constraint system.
  fn get_root(&mut self) -> &mut Self::Root;

  /// Enter a named namespace (auto-pops on drop).
  fn namespace<NR, N>(&mut self, name_fn: N) -> SmallNamespace<'_, W, C, Self::Root>
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    self.get_root().push_namespace(name_fn);
    SmallNamespace {
      inner: self.get_root(),
      _marker: PhantomData,
    }
  }
}

// ── SmallNamespace ─────────────────────────────────────────────────────────

/// A scoped namespace within a SmallConstraintSystem.
pub struct SmallNamespace<'a, W, C, CS: SmallConstraintSystem<W, C>> {
  pub(crate) inner: &'a mut CS,
  pub(crate) _marker: PhantomData<(W, C)>,
}

impl<W, C, CS: SmallConstraintSystem<W, C>> SmallConstraintSystem<W, C>
  for SmallNamespace<'_, W, C, CS>
{
  type Root = CS::Root;

  fn alloc<A, AR, F>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<W, SynthesisError>,
  {
    self.inner.alloc(annotation, f)
  }

  fn alloc_input<A, AR, F>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<W, SynthesisError>,
  {
    self.inner.alloc_input(annotation, f)
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
    self.inner.enforce(annotation, a, b, c);
  }

  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    self.inner.push_namespace(name_fn);
  }

  fn pop_namespace(&mut self) {
    self.inner.pop_namespace();
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self.inner.get_root()
  }
}

impl<W, C, CS: SmallConstraintSystem<W, C>> Drop for SmallNamespace<'_, W, C, CS> {
  fn drop(&mut self) {
    self.inner.pop_namespace();
  }
}

// ── SmallSatisfyingAssignment ──────────────────────────────────────────────

/// Witness generation backend for the small-value path.
///
/// Allocates variables as i8 values (bits 0/1 for SHA-256).
/// `enforce` is a no-op — constraints are not checked, only witness values recorded.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SmallSatisfyingAssignment<V> {
  pub(crate) input_assignment: Vec<V>,
  pub(crate) aux_assignment: Vec<V>,
}

impl<V: From<bool>> SmallSatisfyingAssignment<V> {
  /// Create a new satisfying assignment with no variables.
  pub fn new() -> Self {
    SmallSatisfyingAssignment {
      // Input 0 is always ONE (the constant 1)
      input_assignment: vec![V::from(true)],
      aux_assignment: vec![],
    }
  }
}

impl<W, C> SmallConstraintSystem<W, C> for SmallSatisfyingAssignment<W> {
  type Root = Self;

  fn alloc<A, AR, F>(&mut self, _annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<W, SynthesisError>,
  {
    let val = f()?;
    self.aux_assignment.push(val);
    Ok(Variable::new_unchecked(Index::Aux(
      self.aux_assignment.len() - 1,
    )))
  }

  fn alloc_input<A, AR, F>(&mut self, _annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<W, SynthesisError>,
  {
    let val = f()?;
    self.input_assignment.push(val);
    Ok(Variable::new_unchecked(Index::Input(
      self.input_assignment.len() - 1,
    )))
  }

  fn enforce<A, AR>(
    &mut self,
    _annotation: A,
    _a: SmallLinearCombination<C>,
    _b: SmallLinearCombination<C>,
    _c: SmallLinearCombination<C>,
  ) where
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    // No-op: witness generation only, constraints not checked here
  }

  fn push_namespace<NR, N>(&mut self, _name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
  }

  fn pop_namespace(&mut self) {}

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }
}

// ── SmallShapeCS ───────────────────────────────────────────────────────────

/// Shape extraction backend — records constraints as small-integer linear combinations
/// and builds `SparseMatrix<C>` directly.
///
/// The type parameter `C` controls the coefficient type:
/// - `C = i32`: SHA-256 path (max ~2^18 for NoBatchEq)
/// - `C = i8`: Keccak path (coefficients in {-1, 0, 1, 2})
#[derive(Debug)]
pub struct SmallShapeCS<C: SmallCoeff> {
  pub(crate) constraints: Vec<(
    SmallLinearCombination<C>,
    SmallLinearCombination<C>,
    SmallLinearCombination<C>,
  )>,
  pub(crate) num_inputs: usize,
  pub(crate) num_aux: usize,
}

struct SmallCsrBuilder<C: SmallCoeff> {
  data: Vec<C>,
  indices: Vec<usize>,
  indptr: Vec<usize>,
  cols: usize,
}

impl<C: SmallCoeff> SmallCsrBuilder<C> {
  fn new(cols: usize, rows: usize) -> Self {
    let mut indptr = Vec::with_capacity(rows + 1);
    indptr.push(0);
    Self {
      data: Vec::new(),
      indices: Vec::new(),
      indptr,
      cols,
    }
  }

  fn push_row(&mut self, lc: &SmallLinearCombination<C>, num_aux: usize) {
    let row_start = self.data.len();
    for (var, coeff) in &lc.terms {
      if *coeff == C::default() {
        continue;
      }
      let col = match var.get_unchecked() {
        Index::Aux(i) => i,
        Index::Input(i) => num_aux + i,
      };
      self.data.push(*coeff);
      self.indices.push(col);
    }

    self.sort_and_merge_row(row_start);
    self.indptr.push(self.data.len());
  }

  fn sort_and_merge_row(&mut self, row_start: usize) {
    let row_end = self.data.len();
    let row_len = row_end - row_start;

    for i in 1..row_len {
      let mut j = row_start + i;
      while j > row_start && self.indices[j - 1] > self.indices[j] {
        self.indices.swap(j - 1, j);
        self.data.swap(j - 1, j);
        j -= 1;
      }
    }

    if row_len > 1 {
      let mut write = row_start;
      for read in (row_start + 1)..row_end {
        if self.indices[read] == self.indices[write] {
          let val = self.data[read];
          self.data[write] += val;
        } else {
          if self.data[write] != C::default() {
            write += 1;
          }
          self.data[write] = self.data[read];
          self.indices[write] = self.indices[read];
        }
      }

      if self.data[write] != C::default() {
        write += 1;
      }
      self.data.truncate(write);
      self.indices.truncate(write);
    }
  }

  fn finish(self) -> SparseMatrix<C> {
    SparseMatrix {
      data: self.data,
      indices: self.indices,
      indptr: self.indptr,
      cols: self.cols,
    }
  }
}

impl<C: SmallCoeff> Default for SmallShapeCS<C> {
  fn default() -> Self {
    Self::new()
  }
}

impl<C: SmallCoeff> SmallShapeCS<C> {
  /// Create a new shape constraint system.
  pub fn new() -> Self {
    SmallShapeCS {
      constraints: vec![],
      num_inputs: 1, // slot 0 = ONE
      num_aux: 0,
    }
  }

  /// Total number of auxiliary variables.
  /// Following bellpepper convention: input[0] = ONE, input[1..] = public inputs.
  pub fn num_vars(&self) -> usize {
    self.num_aux
  }

  /// Total number of input variables (including ONE at index 0).
  pub fn num_inputs(&self) -> usize {
    self.num_inputs
  }

  /// Total number of constraints.
  pub fn num_constraints(&self) -> usize {
    self.constraints.len()
  }

  /// Convert accumulated constraints to CSR sparse matrices.
  ///
  /// Variable layout (column index in matrix):
  /// - aux[0..num_aux]: columns 0..num_aux
  /// - input[0..num_inputs]: columns num_aux..num_aux+num_inputs
  ///   (input[0] = ONE = column num_aux)
  pub fn to_matrices(&self) -> (SparseMatrix<C>, SparseMatrix<C>, SparseMatrix<C>) {
    let num_cols = self.num_aux + self.num_inputs;
    let rows = self.constraints.len();
    let mut a = SmallCsrBuilder::new(num_cols, rows);
    let mut b = SmallCsrBuilder::new(num_cols, rows);
    let mut c = SmallCsrBuilder::new(num_cols, rows);

    for (a_lc, b_lc, c_lc) in &self.constraints {
      a.push_row(a_lc, self.num_aux);
      b.push_row(b_lc, self.num_aux);
      c.push_row(c_lc, self.num_aux);
    }

    (a.finish(), b.finish(), c.finish())
  }
}

impl<W, C: SmallCoeff> SmallConstraintSystem<W, C> for SmallShapeCS<C> {
  type Root = Self;

  fn alloc<A, AR, F>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<W, SynthesisError>,
  {
    let idx = self.num_aux;
    self.num_aux += 1;
    Ok(Variable::new_unchecked(Index::Aux(idx)))
  }

  fn alloc_input<A, AR, F>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<W, SynthesisError>,
  {
    let idx = self.num_inputs;
    self.num_inputs += 1;
    Ok(Variable::new_unchecked(Index::Input(idx)))
  }

  fn enforce<A, AR>(
    &mut self,
    _annotation: A,
    a: SmallLinearCombination<C>,
    b: SmallLinearCombination<C>,
    c: SmallLinearCombination<C>,
  ) where
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.constraints.push((a, b, c));
  }

  fn push_namespace<NR, N>(&mut self, _name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
  }

  fn pop_namespace(&mut self) {}

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }
}

#[cfg(test)]
mod tests {
  use super::{
    SmallCoeff, SmallConstraintSystem, SmallLinearCombination, SmallSatisfyingAssignment,
    SmallShapeCS, SmallToBellpepperCS,
  };
  use crate::{provider::Bn254Engine, traits::Engine};
  use bellpepper_core::test_cs::TestConstraintSystem;
  use ff::Field;

  type Scalar = <Bn254Engine as Engine>::Scalar;

  #[test]
  fn test_i8_small_coeff_to_field() {
    assert_eq!(0i8.to_field::<Scalar>(), Scalar::ZERO);
    assert_eq!(1i8.to_field::<Scalar>(), Scalar::ONE);
    assert_eq!((-1i8).to_field::<Scalar>(), -Scalar::ONE);
    assert_eq!(7i8.to_field::<Scalar>(), Scalar::from(7u64));
    assert_eq!((-7i8).to_field::<Scalar>(), -Scalar::from(7u64));
    assert_eq!(
      i8::MIN.to_field::<Scalar>(),
      -Scalar::from(i8::MIN.unsigned_abs() as u64)
    );
  }

  #[test]
  fn test_i32_small_coeff_to_field() {
    assert_eq!(0i32.to_field::<Scalar>(), Scalar::ZERO);
    assert_eq!(1i32.to_field::<Scalar>(), Scalar::ONE);
    assert_eq!((-1i32).to_field::<Scalar>(), -Scalar::ONE);
    assert_eq!(123_456i32.to_field::<Scalar>(), Scalar::from(123_456u64));
    assert_eq!(
      (-123_456i32).to_field::<Scalar>(),
      -Scalar::from(123_456u64)
    );
    assert_eq!(
      i32::MIN.to_field::<Scalar>(),
      -Scalar::from(i32::MIN.unsigned_abs() as u64)
    );
  }

  #[test]
  fn test_small_coeff_mul_field_uses_signed_conversion() {
    let x = Scalar::from(9u64);

    assert_eq!(3i8.mul_field(&x), Scalar::from(27u64));
    assert_eq!((-3i8).mul_field(&x), -Scalar::from(27u64));
    assert_eq!(5i32.mul_field(&x), Scalar::from(45u64));
    assert_eq!((-5i32).mul_field(&x), -Scalar::from(45u64));
  }

  fn synthesize_split_api<CS: SmallConstraintSystem<i8, i32>>(cs: &mut CS) {
    let bit = cs.alloc(|| "bit", || Ok(1i8)).unwrap();
    cs.enforce(
      || "bit_is_one",
      SmallLinearCombination::from_variable(bit, 1i32),
      SmallLinearCombination::one(1i32),
      SmallLinearCombination::one(1i32),
    );
  }

  #[test]
  fn test_small_shape_accepts_i8_witnesses_and_i32_coefficients() {
    let mut cs = SmallShapeCS::<i32>::new();
    synthesize_split_api(&mut cs);

    assert_eq!(cs.num_vars(), 1);
    assert_eq!(cs.num_inputs(), 1);
    assert_eq!(cs.num_constraints(), 1);

    let (a, b, c) = cs.to_matrices();
    assert_eq!(a.data, vec![1i32]);
    assert_eq!(b.data, vec![1i32]);
    assert_eq!(c.data, vec![1i32]);
  }

  #[test]
  fn test_small_satisfying_assignment_accepts_i32_linear_combinations() {
    let mut cs = SmallSatisfyingAssignment::<i8>::new();
    synthesize_split_api(&mut cs);

    assert_eq!(cs.aux_assignment, vec![1i8]);
    assert_eq!(cs.input_assignment, vec![1i8]);
  }

  #[test]
  fn test_small_to_bellpepper_accepts_i8_witnesses_and_i32_coefficients() {
    let mut inner = TestConstraintSystem::<Scalar>::new();
    {
      let mut cs = SmallToBellpepperCS::<Scalar, _>::new(&mut inner);
      synthesize_split_api(&mut cs);
    }

    assert!(inner.is_satisfied());
  }
}
