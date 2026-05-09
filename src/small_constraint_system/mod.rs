// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! # SmallConstraintSystem
//!
//! A constraint system trait that operates over small integer types (i8, i32)
//! instead of field elements. This enables the pure-integer proving path where:
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
pub struct SmallLinearCombination<V> {
  pub(crate) terms: Vec<(Variable, V)>,
}

impl<V: Copy> SmallLinearCombination<V> {
  /// Empty linear combination (zero).
  pub fn zero() -> Self {
    SmallLinearCombination { terms: vec![] }
  }

  /// LC = coeff × ONE.
  pub fn one(coeff: V) -> Self {
    SmallLinearCombination {
      terms: vec![(Variable::new_unchecked(Index::Input(0)), coeff)],
    }
  }

  /// LC = coeff × var.
  pub fn from_variable(var: Variable, coeff: V) -> Self {
    SmallLinearCombination {
      terms: vec![(var, coeff)],
    }
  }

  /// Add a term: LC += coeff × var.
  pub fn add_term(&mut self, var: Variable, coeff: V) {
    self.terms.push((var, coeff));
  }

  /// Subtract a term: LC -= coeff × var (coeff must already be negated).
  pub fn sub_term(&mut self, var: Variable, neg_coeff: V) {
    self.terms.push((var, neg_coeff));
  }
}

// ── SmallConstraintSystem ──────────────────────────────────────────────────

/// A constraint system over small integer values.
///
/// Mirrors bellpepper's `ConstraintSystem<Scalar>` but without any field arithmetic.
/// Used for the pure-integer proving path.
pub trait SmallConstraintSystem<V>: Sized {
  /// The root constraint system type (for namespace delegation).
  type Root: SmallConstraintSystem<V>;

  /// Allocate an auxiliary variable.
  fn alloc<A, AR, F>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<V, SynthesisError>;

  /// Allocate an input variable.
  fn alloc_input<A, AR, F>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<V, SynthesisError>;

  /// Enforce a constraint: a(x) × b(x) = c(x).
  fn enforce<A, AR>(
    &mut self,
    annotation: A,
    a: SmallLinearCombination<V>,
    b: SmallLinearCombination<V>,
    c: SmallLinearCombination<V>,
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
  fn namespace<NR, N>(&mut self, name_fn: N) -> SmallNamespace<'_, V, Self::Root>
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
pub struct SmallNamespace<'a, V, CS: SmallConstraintSystem<V>> {
  pub(crate) inner: &'a mut CS,
  pub(crate) _marker: PhantomData<V>,
}

impl<V, CS: SmallConstraintSystem<V>> SmallConstraintSystem<V> for SmallNamespace<'_, V, CS> {
  type Root = CS::Root;

  fn alloc<A, AR, F>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<V, SynthesisError>,
  {
    self.inner.alloc(annotation, f)
  }

  fn alloc_input<A, AR, F>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<V, SynthesisError>,
  {
    self.inner.alloc_input(annotation, f)
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

impl<V, CS: SmallConstraintSystem<V>> Drop for SmallNamespace<'_, V, CS> {
  fn drop(&mut self) {
    self.inner.pop_namespace();
  }
}

// ── SmallSatisfyingAssignment ──────────────────────────────────────────────

/// Witness generation backend for the small-value path.
///
/// Allocates variables as i8 values (bits 0/1 for SHA-256).
/// `enforce` is a no-op — constraints are not checked, only witness values recorded.
#[derive(Clone, Debug, Default)]
pub struct SmallSatisfyingAssignment<V> {
  pub(crate) input_assignment: Vec<V>,
  pub(crate) aux_assignment: Vec<V>,
}

impl<V: Copy + From<bool>> SmallSatisfyingAssignment<V> {
  /// Create a new satisfying assignment with no variables.
  pub fn new() -> Self {
    SmallSatisfyingAssignment {
      // Input 0 is always ONE (the constant 1)
      input_assignment: vec![V::from(true)],
      aux_assignment: vec![],
    }
  }
}

impl<V: Copy + Default> SmallConstraintSystem<V> for SmallSatisfyingAssignment<V> {
  type Root = Self;

  fn alloc<A, AR, F>(&mut self, _annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<V, SynthesisError>,
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
    F: FnOnce() -> Result<V, SynthesisError>,
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
    _a: SmallLinearCombination<V>,
    _b: SmallLinearCombination<V>,
    _c: SmallLinearCombination<V>,
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
    let _num_rows = self.constraints.len();

    let lc_to_csr = |get_lc: &dyn Fn(
      &(
        SmallLinearCombination<C>,
        SmallLinearCombination<C>,
        SmallLinearCombination<C>,
      ),
    ) -> &SmallLinearCombination<C>| {
      let mut data = vec![];
      let mut indices = vec![];
      let mut indptr = vec![0usize];

      for constraint in &self.constraints {
        let lc = get_lc(constraint);
        let row_start = data.len();
        for (var, coeff) in &lc.terms {
          if *coeff == C::default() {
            continue;
          }
          let col = match var.get_unchecked() {
            Index::Aux(i) => i,
            Index::Input(i) => self.num_aux + i,
          };
          data.push(*coeff);
          indices.push(col);
        }
        // Sort by column index for canonical CSR order
        let row_end = data.len();
        let row_slice_indices = &mut indices[row_start..row_end];
        let row_slice_data = &mut data[row_start..row_end];
        // Simple insertion sort (rows are small)
        for i in 1..(row_end - row_start) {
          let mut j = i;
          while j > 0 && row_slice_indices[j - 1] > row_slice_indices[j] {
            row_slice_indices.swap(j - 1, j);
            row_slice_data.swap(j - 1, j);
            j -= 1;
          }
        }
        // Merge duplicate column entries (sum coefficients, drop zeros)
        let row_len = row_end - row_start;
        if row_len > 1 {
          let mut write = row_start;
          for read in (row_start + 1)..row_end {
            if indices[read] == indices[write] {
              let val = data[read];
              data[write] += val;
            } else {
              if data[write] != C::default() {
                write += 1;
              }
              data[write] = data[read];
              indices[write] = indices[read];
            }
          }
          // Keep last element if non-zero
          if data[write] != C::default() {
            write += 1;
          }
          data.truncate(write);
          indices.truncate(write);
        }
        indptr.push(data.len());
      }

      SparseMatrix {
        data,
        indices,
        indptr,
        cols: num_cols,
      }
    };

    let a = lc_to_csr(&|(a, _, _)| a);
    let b = lc_to_csr(&|(_, b, _)| b);
    let c = lc_to_csr(&|(_, _, c)| c);

    (a, b, c)
  }
}

impl<C: SmallCoeff> SmallConstraintSystem<C> for SmallShapeCS<C> {
  type Root = Self;

  fn alloc<A, AR, F>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<C, SynthesisError>,
  {
    let idx = self.num_aux;
    self.num_aux += 1;
    Ok(Variable::new_unchecked(Index::Aux(idx)))
  }

  fn alloc_input<A, AR, F>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    F: FnOnce() -> Result<C, SynthesisError>,
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
  use super::SmallCoeff;
  use crate::{provider::Bn254Engine, traits::Engine};
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
}
