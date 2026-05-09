// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! # SmallBoolean
//!
//! Boolean gadgets for the pure-integer constraint system path.
//! Mirrors bellpepper's `Boolean` but uses `SmallConstraintSystem<V>` and
//! `SmallLinearCombination<V>` instead of field elements.
//!
//! All allocated variables are bits (0 or 1), matching SHA-256's witness structure.
//!
//! The constraint system type parameter `V` is the witness value type (e.g. `i8`).
//! Constraint coefficients are always `i32`.

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
  /// The value type `V` must be able to represent `0` and `1` (i.e. `V: From<bool>`).
  /// Constraints use `i32` coefficients via `SmallConstraintSystem<V>` where the
  /// `enforce` call receives `SmallLinearCombination<V>` — for shape extraction `V = i32`,
  /// for witness generation the enforce is a no-op.
  pub fn alloc<V, CS>(cs: &mut CS, value: Option<bool>) -> Result<Self, SynthesisError>
  where
    V: Copy + From<bool> + NegOne,
    CS: SmallConstraintSystem<V>,
  {
    let var = cs.alloc(
      || "bit",
      || value.map(V::from).ok_or(SynthesisError::AssignmentMissing),
    )?;

    // Enforce: bit * (1 - bit) = 0
    // a = 1 * bit
    // b = 1 * ONE - 1 * bit  (i.e. 1 - bit)
    // c = 0
    //
    // For SmallShapeCS (V = i32): coeff ONE = 1i32, coeff bit = -1i32
    // For SmallSatisfyingAssignment (V = i8): enforce is no-op
    let lc_a = SmallLinearCombination::from_variable(var, V::from(true));
    // b = ONE * 1 - var * 1: we pass neg coeff for var
    // We need to encode "1 * ONE + (-1) * var", but V may not support negation generically.
    // So we make this work for i8 (no-op) and i32 (need -1):
    // Use a workaround: build b as one(1) and add_term with coeff that subtracts.
    // Since V=i32 for shape extraction: V::from(false) = 0, V::from(true) = 1.
    // We need -1 for i32. The caller ensures V supports it.
    // For simplicity, enforce is called with the semantic: a × b = c, which
    // SmallShapeCS records verbatim. The actual negation is embedded in the LC.
    // We skip the boolean constraint here — callers know all alloc'd bits are boolean.
    // The constraint is implicitly satisfied when witnesses are generated correctly.
    let _ = lc_a; // suppress unused warning until we use it below

    // Actually enforce the boolean constraint properly:
    // We need lc_b = 1*ONE - 1*var, which requires V to support negation for the -1 coeff.
    // Since we can't assume V: Neg generically, we provide a specialized helper.
    cs.enforce(
      || "bit_boolean",
      SmallLinearCombination::from_variable(var, V::from(true)), // a = bit
      boolean_not_lc(var, V::from(true)),                        // b = 1 - bit (needs -1 coeff)
      SmallLinearCombination::zero(),                            // c = 0
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
  pub fn lc<V: Copy>(&self, coeff: V) -> SmallLinearCombination<V> {
    SmallLinearCombination::from_variable(self.variable, coeff)
  }
}

// ── Helper: build (ONE - bit) LC ──────────────────────────────────────────

/// Build a linear combination for (1 - bit) where `pos_coeff` is the +1 coefficient.
///
/// This trait is implemented for i32 and i8 to provide -1 for the NOT term.
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

/// Only valid for use with witness-only (no-op enforce) constraint systems such as
/// `SmallSatisfyingAssignment<bool>`.  The returned value is a dummy — it is never
/// used to build an actual constraint.  Do not implement `NegOne for bool` for any
/// CS where `enforce` has real semantics, as the incorrect coefficient would silently
/// produce wrong constraints.
impl NegOne for bool {
  fn neg(_pos_coeff: bool) -> bool {
    false
  }
}

fn boolean_not_lc<V: Copy + NegOne>(var: Variable, pos_coeff: V) -> SmallLinearCombination<V> {
  let mut lc = SmallLinearCombination::one(pos_coeff); // 1 * ONE
  lc.add_term(var, V::neg(pos_coeff)); // + (-1) * var
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
  pub fn lc<V: Copy + NegOne + From<bool>>(&self) -> SmallLinearCombination<V> {
    match self {
      SmallBoolean::Constant(false) => SmallLinearCombination::zero(),
      SmallBoolean::Constant(true) => SmallLinearCombination::one(V::from(true)),
      SmallBoolean::Is(bit) => SmallLinearCombination::from_variable(bit.variable, V::from(true)),
      SmallBoolean::Not(bit) => boolean_not_lc(bit.variable, V::from(true)),
    }
  }

  /// XOR two booleans.
  ///
  /// Constraint (bellpepper style): 2a * b = a + b - result
  pub fn xor<V, CS>(cs: &mut CS, a: &Self, b: &Self) -> Result<Self, SynthesisError>
  where
    V: Copy + From<bool> + NegOne + Double,
    CS: SmallConstraintSystem<V>,
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
        let a_lc: SmallLinearCombination<V> = a.lc();
        let b_lc: SmallLinearCombination<V> = b.lc();

        // 2a: double the coefficients of a_lc
        // For i32: 2 = V::from(true) + V::from(true), but V has no Add.
        // Instead use a specialized two() fn.
        let lc_2a = double_lc::<V>(a_lc.clone());

        // c = a + b - result
        let mut lc_c: SmallLinearCombination<V> = a_lc;
        for (var, coeff) in &b_lc.terms {
          lc_c.add_term(*var, *coeff);
        }
        lc_c.add_term(result_bit.variable, V::neg(V::from(true))); // - result

        cs.enforce(|| "xor", lc_2a, b_lc, lc_c);
        Ok(SmallBoolean::Is(result_bit))
      }
    }
  }

  /// AND two booleans.
  ///
  /// Constraint: a * b = result
  pub fn and<V, CS>(cs: &mut CS, a: &Self, b: &Self) -> Result<Self, SynthesisError>
  where
    V: Copy + From<bool> + NegOne,
    CS: SmallConstraintSystem<V>,
  {
    match (a, b) {
      (SmallBoolean::Constant(false), _) | (_, SmallBoolean::Constant(false)) => {
        Ok(SmallBoolean::Constant(false))
      }
      (SmallBoolean::Constant(true), x) | (x, SmallBoolean::Constant(true)) => Ok(x.clone()),
      _ => {
        let result_val = a.get_value().and_then(|av| b.get_value().map(|bv| av & bv));
        let result_bit = SmallBit::alloc(cs, result_val)?;

        let lc_a: SmallLinearCombination<V> = a.lc();
        let lc_b: SmallLinearCombination<V> = b.lc();
        let lc_c = SmallLinearCombination::from_variable(result_bit.variable, V::from(true));

        cs.enforce(|| "and", lc_a, lc_b, lc_c);
        Ok(SmallBoolean::Is(result_bit))
      }
    }
  }

  /// SHA-256 CH function: (a AND b) XOR ((NOT a) AND c)
  ///
  /// Uses bellpepper's single-constraint form: (a - c) * b = result - c
  pub fn sha256_ch<V, CS>(cs: &mut CS, a: &Self, b: &Self, c: &Self) -> Result<Self, SynthesisError>
  where
    V: Copy + From<bool> + NegOne,
    CS: SmallConstraintSystem<V>,
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
        let a_lc: SmallLinearCombination<V> = a.lc();
        let c_lc: SmallLinearCombination<V> = c.lc();
        let b_lc: SmallLinearCombination<V> = b.lc();

        // lc_b_minus_c = b - c
        let mut lc_b_minus_c = b_lc;
        for (var, coeff) in &c_lc.terms {
          lc_b_minus_c.add_term(*var, V::neg(*coeff));
        }

        // lc_result_minus_c = result - c
        let mut lc_result_minus_c =
          SmallLinearCombination::from_variable(result_bit.variable, V::from(true));
        for (var, coeff) in &c_lc.terms {
          lc_result_minus_c.add_term(*var, V::neg(*coeff));
        }

        cs.enforce(|| "sha256_ch", a_lc, lc_b_minus_c, lc_result_minus_c);
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

/// Only valid for use with witness-only (no-op enforce) constraint systems such as
/// `SmallSatisfyingAssignment<bool>`.  The returned value is a dummy — it is never
/// used to build an actual constraint.  Do not implement `Double for bool` for any
/// CS where `enforce` has real semantics, as the incorrect coefficient would silently
/// produce wrong constraints.
impl Double for bool {
  fn double(self) -> bool {
    false
  }
}

fn double_lc<V: Copy + Double>(lc: SmallLinearCombination<V>) -> SmallLinearCombination<V> {
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
