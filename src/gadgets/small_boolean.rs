// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! # SmallBoolean
//!
//! Boolean gadgets for the pure-integer constraint system path.
//! Mirrors bellpepper's `Boolean` but uses `SmallConstraintSystem<W, C>` and
//! `SmallLinearCombination<C>` instead of field elements.
//!
//! All allocated variables are bits (0 or 1), matching SHA-256's witness structure.
//!
//! The constraint system type parameter `W` is the witness value type (e.g. `i8`);
//! `C` is the coefficient type (e.g. `i32` for SHA-256).

use bellpepper_core::{Index, SynthesisError, Variable};

use crate::small_constraint_system::{SmallConstraintSystem, SmallLinearCombination};

// ── SmallBit ───────────────────────────────────────────────────────────────

/// An allocated bit variable in the small-value constraint system.
#[derive(Clone, Debug)]
pub struct SmallBit {
  pub(crate) variable: Variable,
  pub(crate) value: Option<bool>,
}

impl SmallBit {
  /// Allocate a bit and enforce the boolean constraint: bit × (1 - bit) = 0.
  ///
  /// The value type `W` must be able to represent `0` and `1`; constraints use
  /// coefficient type `C`.
  pub fn alloc<W, C, CS>(cs: &mut CS, value: Option<bool>) -> Result<Self, SynthesisError>
  where
    W: Copy + From<bool>,
    C: Copy + From<bool> + NegOne,
    CS: SmallConstraintSystem<W, C>,
  {
    let var = cs.alloc(
      || "bit",
      || value.map(W::from).ok_or(SynthesisError::AssignmentMissing),
    )?;

    // Enforce: bit * (1 - bit) = 0
    // a = 1 * bit
    // b = 1 * ONE - 1 * bit  (i.e. 1 - bit)
    // c = 0
    cs.enforce(
      || "bit_boolean",
      SmallLinearCombination::from_variable(var, C::from(true)),
      boolean_not_lc(var, C::from(true)),
      SmallLinearCombination::zero(),
    );

    Ok(SmallBit {
      variable: var,
      value,
    })
  }

  /// Get the underlying variable.
  pub fn get_variable(&self) -> Variable {
    self.variable
  }

  /// Get the value of this bit.
  pub fn get_value(&self) -> Option<bool> {
    self.value
  }

  /// Build a linear combination representing this bit's value: `coeff * bit`.
  pub fn lc<C: Copy>(&self, coeff: C) -> SmallLinearCombination<C> {
    SmallLinearCombination::from_variable(self.variable, coeff)
  }
}

// ── Helper: build (ONE - bit) LC ──────────────────────────────────────────

/// Build a linear combination for (1 - bit) where `pos_coeff` is the +1 coefficient.
///
/// This trait is implemented for coefficient types to provide -1 for the NOT term.
pub trait NegOne: Sized + Copy {
  /// Returns the negative of `pos_coeff`.
  fn neg(pos_coeff: Self) -> Self;
}

impl NegOne for i32 {
  fn neg(pos_coeff: i32) -> i32 {
    -pos_coeff
  }
}

impl NegOne for i8 {
  fn neg(pos_coeff: i8) -> i8 {
    pos_coeff.wrapping_neg()
  }
}

fn boolean_not_lc<C: Copy + NegOne>(var: Variable, pos_coeff: C) -> SmallLinearCombination<C> {
  let mut lc = SmallLinearCombination::one(pos_coeff); // 1 * ONE
  lc.add_term(var, C::neg(pos_coeff)); // + (-1) * var
  lc
}

// ── SmallBoolean ───────────────────────────────────────────────────────────

/// A boolean value in the small-value constraint system.
/// Mirrors bellpepper's `Boolean` enum.
#[derive(Clone, Debug)]
pub enum SmallBoolean {
  /// A constant boolean (not a circuit variable).
  Constant(bool),
  /// A positive boolean variable.
  Is(SmallBit),
  /// A negated boolean variable (NOT bit).
  Not(SmallBit),
}

impl SmallBoolean {
  /// Get the boolean value if known.
  pub fn get_value(&self) -> Option<bool> {
    match self {
      SmallBoolean::Constant(b) => Some(*b),
      SmallBoolean::Is(bit) => bit.value,
      SmallBoolean::Not(bit) => bit.value.map(|b| !b),
    }
  }

  /// Create a constant boolean.
  pub fn constant(val: bool) -> Self {
    SmallBoolean::Constant(val)
  }

  /// Negate this boolean.
  pub fn not(&self) -> Self {
    match self {
      SmallBoolean::Constant(b) => SmallBoolean::Constant(!b),
      SmallBoolean::Is(bit) => SmallBoolean::Not(bit.clone()),
      SmallBoolean::Not(bit) => SmallBoolean::Is(bit.clone()),
    }
  }

  /// Build a linear combination for this boolean's value.
  ///
  /// - `Constant(false)` → 0
  /// - `Constant(true)` → 1 × ONE
  /// - `Is(bit)` → 1 × bit
  /// - `Not(bit)` → 1 × ONE - 1 × bit
  pub fn lc<C: Copy + NegOne + From<bool>>(&self) -> SmallLinearCombination<C> {
    match self {
      SmallBoolean::Constant(false) => SmallLinearCombination::zero(),
      SmallBoolean::Constant(true) => SmallLinearCombination::one(C::from(true)),
      SmallBoolean::Is(bit) => SmallLinearCombination::from_variable(bit.variable, C::from(true)),
      SmallBoolean::Not(bit) => boolean_not_lc(bit.variable, C::from(true)),
    }
  }

  /// XOR two booleans.
  ///
  /// Constraint (bellpepper style): 2a * b = a + b - result
  pub fn xor<W, C, CS>(cs: &mut CS, a: &Self, b: &Self) -> Result<Self, SynthesisError>
  where
    W: Copy + From<bool>,
    C: Copy + From<bool> + NegOne + Double,
    CS: SmallConstraintSystem<W, C>,
  {
    match (a, b) {
      (SmallBoolean::Constant(a_val), _) => {
        if *a_val {
          Ok(b.not())
        } else {
          Ok(b.clone())
        }
      }
      (_, SmallBoolean::Constant(b_val)) => {
        if *b_val {
          Ok(a.not())
        } else {
          Ok(a.clone())
        }
      }
      _ => {
        let result_val = a.get_value().and_then(|av| b.get_value().map(|bv| av ^ bv));
        let result_bit = SmallBit::alloc(cs, result_val)?;

        // Enforce: 2a * b = a + b - result
        // lc_2a = 2 * a_var (for Is) or 2 * ONE - 2 * a_var (for Not)
        // We compute lc_2a by taking a.lc() and doubling each coefficient.
        let a_lc: SmallLinearCombination<C> = a.lc();
        let b_lc: SmallLinearCombination<C> = b.lc();

        // 2a: double the coefficients of a_lc
        // For i32: 2 = C::from(true) + C::from(true), but C has no Add.
        // Instead use a specialized two() fn.
        let lc_2a = double_lc::<C>(a_lc.clone());

        // c = a + b - result
        let mut lc_c: SmallLinearCombination<C> = a_lc;
        for (var, coeff) in &b_lc.terms {
          lc_c.add_term(*var, *coeff);
        }
        lc_c.add_term(result_bit.variable, C::neg(C::from(true))); // - result

        cs.enforce(|| "xor", lc_2a, b_lc, lc_c);
        Ok(SmallBoolean::Is(result_bit))
      }
    }
  }

  /// AND two booleans.
  ///
  /// Constraint: a * b = result
  pub fn and<W, C, CS>(cs: &mut CS, a: &Self, b: &Self) -> Result<Self, SynthesisError>
  where
    W: Copy + From<bool>,
    C: Copy + From<bool> + NegOne,
    CS: SmallConstraintSystem<W, C>,
  {
    match (a, b) {
      (SmallBoolean::Constant(false), _) | (_, SmallBoolean::Constant(false)) => {
        Ok(SmallBoolean::Constant(false))
      }
      (SmallBoolean::Constant(true), x) | (x, SmallBoolean::Constant(true)) => Ok(x.clone()),
      _ => {
        let result_val = a.get_value().and_then(|av| b.get_value().map(|bv| av & bv));
        let result_bit = SmallBit::alloc(cs, result_val)?;

        let lc_a: SmallLinearCombination<C> = a.lc();
        let lc_b: SmallLinearCombination<C> = b.lc();
        let lc_c = SmallLinearCombination::from_variable(result_bit.variable, C::from(true));

        cs.enforce(|| "and", lc_a, lc_b, lc_c);
        Ok(SmallBoolean::Is(result_bit))
      }
    }
  }

  /// SHA-256 CH function: (a AND b) XOR ((NOT a) AND c)
  ///
  /// Uses bellpepper's single-constraint form: (a - c) * b = result - c
  pub fn sha256_ch<W, C, CS>(
    cs: &mut CS,
    a: &Self,
    b: &Self,
    c: &Self,
  ) -> Result<Self, SynthesisError>
  where
    W: Copy + From<bool>,
    C: Copy + From<bool> + NegOne,
    CS: SmallConstraintSystem<W, C>,
  {
    let result_val = a.get_value().and_then(|av| {
      b.get_value()
        .and_then(|bv| c.get_value().map(|cv| (av & bv) ^ (!av & cv)))
    });

    match (a, b, c) {
      (SmallBoolean::Constant(av), SmallBoolean::Constant(bv), SmallBoolean::Constant(cv)) => {
        Ok(SmallBoolean::constant((av & bv) ^ (!av & cv)))
      }
      _ => {
        let result_bit = SmallBit::alloc(cs, result_val)?;

        // ch(a,b,c) = a*(b-c) + c  =>  constraint: a * (b - c) = result - c
        let a_lc: SmallLinearCombination<C> = a.lc();
        let c_lc: SmallLinearCombination<C> = c.lc();
        let b_lc: SmallLinearCombination<C> = b.lc();

        // lc_b_minus_c = b - c
        let mut lc_b_minus_c = b_lc;
        for (var, coeff) in &c_lc.terms {
          lc_b_minus_c.add_term(*var, C::neg(*coeff));
        }

        // lc_result_minus_c = result - c
        let mut lc_result_minus_c =
          SmallLinearCombination::from_variable(result_bit.variable, C::from(true));
        for (var, coeff) in &c_lc.terms {
          lc_result_minus_c.add_term(*var, C::neg(*coeff));
        }

        cs.enforce(|| "sha256_ch", a_lc, lc_b_minus_c, lc_result_minus_c);
        Ok(SmallBoolean::Is(result_bit))
      }
    }
  }

  /// SHA-256 MAJ function: (a AND b) XOR (a AND c) XOR (b AND c)
  ///
  /// Uses bellpepper's optimized form:
  /// allocate `bc = b * c`, then enforce `(2bc - b - c) * a = bc - result`.
  pub fn sha256_maj<W, C, CS>(
    cs: &mut CS,
    a: &Self,
    b: &Self,
    c: &Self,
  ) -> Result<Self, SynthesisError>
  where
    W: Copy + From<bool>,
    C: Copy + From<bool> + NegOne + Double,
    CS: SmallConstraintSystem<W, C>,
  {
    let result_val = a.get_value().and_then(|av| {
      b.get_value()
        .and_then(|bv| c.get_value().map(|cv| (av & bv) ^ (av & cv) ^ (bv & cv)))
    });

    match (a, b, c) {
      (SmallBoolean::Constant(av), SmallBoolean::Constant(bv), SmallBoolean::Constant(cv)) => {
        Ok(SmallBoolean::constant((av & bv) ^ (av & cv) ^ (bv & cv)))
      }
      _ => {
        let bc = SmallBoolean::and(cs.namespace(|| "bc").inner, b, c)?;
        let result_bit = SmallBit::alloc(cs, result_val)?;

        // maj(a,b,c) identity:
        //   (2bc - b - c) * a = bc - maj
        let a_lc: SmallLinearCombination<C> = a.lc();
        let b_lc: SmallLinearCombination<C> = b.lc();
        let c_lc: SmallLinearCombination<C> = c.lc();
        let bc_lc: SmallLinearCombination<C> = bc.lc();

        let mut lc_2bc_minus_b_minus_c = double_lc::<C>(bc_lc.clone());
        for (var, coeff) in &b_lc.terms {
          lc_2bc_minus_b_minus_c.add_term(*var, C::neg(*coeff));
        }
        for (var, coeff) in &c_lc.terms {
          lc_2bc_minus_b_minus_c.add_term(*var, C::neg(*coeff));
        }

        let mut lc_bc_minus_result = bc_lc;
        lc_bc_minus_result.add_term(result_bit.variable, C::neg(C::from(true)));

        cs.enforce(
          || "sha256_maj",
          lc_2bc_minus_b_minus_c,
          a_lc,
          lc_bc_minus_result,
        );
        Ok(SmallBoolean::Is(result_bit))
      }
    }
  }

  /// Get the variable if this is a variable (not a constant).
  pub fn get_variable(&self) -> Option<Variable> {
    match self {
      SmallBoolean::Constant(_) => None,
      SmallBoolean::Is(bit) | SmallBoolean::Not(bit) => Some(bit.variable),
    }
  }
}

impl From<SmallBit> for SmallBoolean {
  fn from(bit: SmallBit) -> Self {
    SmallBoolean::Is(bit)
  }
}

// ── Helper: double LC coefficients ────────────────────────────────────────

/// Trait for doubling a value (2×).
pub trait Double: Sized + Copy {
  /// Returns `2 * self`.
  fn double(self) -> Self;
}

impl Double for i32 {
  fn double(self) -> i32 {
    self * 2
  }
}

impl Double for i8 {
  fn double(self) -> i8 {
    self.wrapping_mul(2)
  }
}

fn double_lc<C: Copy + Double>(lc: SmallLinearCombination<C>) -> SmallLinearCombination<C> {
  SmallLinearCombination {
    terms: lc
      .terms
      .into_iter()
      .map(|(var, coeff)| (var, coeff.double()))
      .collect(),
  }
}

// ── ONE constant ───────────────────────────────────────────────────────────

/// The ONE constant variable (input[0]).
pub fn one() -> Variable {
  Variable::new_unchecked(Index::Input(0))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::Bn254Engine,
    small_constraint_system::{SmallShapeCS, SmallToBellpepperCS},
    traits::Engine,
  };
  use bellpepper_core::test_cs::TestConstraintSystem;

  type Scalar = <Bn254Engine as Engine>::Scalar;

  fn alloc_bit<CS: SmallConstraintSystem<i8, i32>>(
    cs: &mut CS,
    name: &'static str,
    value: bool,
  ) -> SmallBoolean {
    SmallBoolean::Is(
      SmallBit::alloc::<i8, i32, _>(&mut cs.namespace(|| name), Some(value)).unwrap(),
    )
  }

  #[test]
  fn test_sha256_maj_truth_table() {
    for a_val in [false, true] {
      for b_val in [false, true] {
        for c_val in [false, true] {
          let mut inner = TestConstraintSystem::<Scalar>::new();
          let result = {
            let mut cs = SmallToBellpepperCS::<Scalar, _>::new(&mut inner);
            let a = alloc_bit(&mut cs, "a", a_val);
            let b = alloc_bit(&mut cs, "b", b_val);
            let c = alloc_bit(&mut cs, "c", c_val);
            SmallBoolean::sha256_maj::<i8, i32, _>(&mut cs, &a, &b, &c).unwrap()
          };

          assert_eq!(
            result.get_value(),
            Some((a_val & b_val) ^ (a_val & c_val) ^ (b_val & c_val))
          );
          assert!(inner.is_satisfied());
        }
      }
    }
  }

  #[test]
  fn test_sha256_maj_shape_uses_bc_plus_custom_constraint() {
    let mut cs = SmallShapeCS::<i32>::new();
    let a = alloc_bit(&mut cs, "a", true);
    let b = alloc_bit(&mut cs, "b", false);
    let c = alloc_bit(&mut cs, "c", true);
    let before = cs.num_constraints();

    let _ = SmallBoolean::sha256_maj::<i8, i32, _>(&mut cs, &a, &b, &c).unwrap();

    // bc alloc boolean + bc relation + result alloc boolean + custom maj relation.
    assert_eq!(cs.num_constraints() - before, 4);
  }
}
