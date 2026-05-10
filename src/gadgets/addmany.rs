// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Addition algorithms for SmallUInt32 values.
//!
//! This module provides the limbed addition algorithm optimized for the
//! small-value sumcheck with i32 coefficients.
//!
//! - [`limbed`]: 16-bit limbed addition, max coefficient 2^18 (fits i32)
//!   Used by NoBatchEq for the pure-integer proving path.

use super::{small_multi_eq::SmallMultiEq, small_uint32::SmallUInt32};
use crate::{
  gadgets::small_boolean::{SmallBit, SmallBoolean},
  small_constraint_system::SmallLinearCombination,
};
use bellpepper_core::SynthesisError;

/// 16-bit limbed addition for i8 witness / i32 coefficient path.
///
/// Splits each 32-bit value into two 16-bit limbs and adds them separately.
/// This keeps the maximum coefficient at 2^18, which fits in i32.
///
/// Constraint 1 (low limb):
///   Σ(operand_lo) = result_lo + carry × 2^16
///
/// Constraint 2 (high limb):
///   Σ(operand_hi) + carry = result_hi + overflow × 2^16
pub(crate) fn limbed<W, M>(
  cs: &mut M,
  operands: &[SmallUInt32],
) -> Result<SmallUInt32, SynthesisError>
where
  W: Copy + From<bool>,
  M: SmallMultiEq<W, i32>,
{
  // For N operands, each 16-bit limb sum can be up to N * (2^16 - 1)
  // For 10 operands: 10 * 65535 = 655350, needs 20 bits
  // We allocate 16 result bits + up to 4 carry/overflow bits
  let num_carry_bits =
    64 - ((operands.len() as u64) * (u16::MAX as u64)).leading_zeros() as usize - 16;
  let num_carry_bits = num_carry_bits.max(1); // At least 1 carry bit

  // Compute limb sums in one witness pass.
  // IMPORTANT: Carry must be computed from the LOW LIMB SUM, not from the full result.
  let limb_sums: Option<(u64, u64)> = operands.iter().try_fold((0u64, 0u64), |(lo, hi), op| {
    op.get_value().map(|v| {
      (
        lo + ((v as u64) & 0xFFFF),
        hi + (((v as u64) >> 16) & 0xFFFF),
      )
    })
  });
  let lo_sum = limb_sums.map(|(lo, _)| lo);
  let hi_sum_with_carry = limb_sums.map(|(lo, hi)| hi + (lo >> 16));

  if cs.is_witness_generator() {
    let mut bits = [const { SmallBoolean::Constant(false) }; 32];

    for (i, slot) in bits.iter_mut().enumerate().take(16) {
      let bit = SmallBit::alloc(
        &mut cs.namespace(|| format!("lo{i}")),
        lo_sum.map(|v| (v >> i) & 1 == 1),
      )?;
      *slot = SmallBoolean::Is(bit);
    }

    for i in 0..num_carry_bits {
      let _ = SmallBit::alloc(
        &mut cs.namespace(|| format!("c{i}")),
        lo_sum.map(|v| (v >> (16 + i)) & 1 == 1),
      )?;
    }

    for i in 0..16 {
      let bit = SmallBit::alloc(
        &mut cs.namespace(|| format!("hi{i}")),
        hi_sum_with_carry.map(|v| (v >> i) & 1 == 1),
      )?;
      bits[16 + i] = SmallBoolean::Is(bit);
    }

    for i in 0..num_carry_bits {
      let _ = SmallBit::alloc(
        &mut cs.namespace(|| format!("o{i}")),
        hi_sum_with_carry.map(|v| (v >> (16 + i)) & 1 == 1),
      )?;
    }

    return Ok(SmallUInt32::from_bits_le(&bits));
  }

  // === LOW LIMB CONSTRAINT ===
  // Sum of low 16 bits of each operand = low 16 bits of result + carry × 2^16

  // Build LHS: sum of all operand low limbs
  // Coefficients are powers of 2: 1, 2, 4, ..., 2^15 (max 2^15, fits i32)
  let mut lo_operands_lc = SmallLinearCombination::zero();
  for op in operands.iter() {
    let mut coeff: i32 = 1;
    for bit in &op.bits_le()[0..16] {
      add_boolean_to_lc(&mut lo_operands_lc, bit, coeff);
      coeff *= 2;
    }
  }

  // Allocate result low bits (0..15) from lo_sum
  let mut lo_result_lc = SmallLinearCombination::zero();
  let mut bits = [const { SmallBoolean::Constant(false) }; 32];
  let mut carry_bits: Vec<SmallBit> = Vec::with_capacity(num_carry_bits);

  let mut coeff: i32 = 1;
  for (i, slot) in bits.iter_mut().enumerate().take(16) {
    let bit = SmallBit::alloc(
      &mut cs.namespace(|| format!("lo{i}")),
      lo_sum.map(|v| (v >> i) & 1 == 1),
    )?;
    lo_result_lc.add_term(bit.get_variable(), coeff);
    *slot = SmallBoolean::Is(bit);
    coeff *= 2;
  }

  // Allocate carry bits from lo_sum (bits 16+)
  for i in 0..num_carry_bits {
    let bit = SmallBit::alloc(
      &mut cs.namespace(|| format!("c{i}")),
      lo_sum.map(|v| (v >> (16 + i)) & 1 == 1),
    )?;
    lo_result_lc.add_term(bit.get_variable(), coeff);
    carry_bits.push(bit);
    coeff *= 2;
  }

  // Enforce: lo_operands_lc = lo_result_lc
  cs.enforce_equal(&lo_operands_lc, &lo_result_lc);

  // === HIGH LIMB CONSTRAINT ===
  // Sum of high 16 bits of each operand + carry = high 16 bits of result + overflow × 2^16

  // Build LHS: sum of all operand high limbs + carry
  let mut hi_operands_lc = SmallLinearCombination::zero();
  for op in operands.iter() {
    let mut coeff: i32 = 1;
    for bit in &op.bits_le()[16..32] {
      add_boolean_to_lc(&mut hi_operands_lc, bit, coeff);
      coeff *= 2;
    }
  }

  // Add carry from low limb
  let mut coeff: i32 = 1;
  for carry_bit in &carry_bits {
    hi_operands_lc.add_term(carry_bit.get_variable(), coeff);
    coeff *= 2;
  }

  // Allocate result high bits (16..31) from hi_sum_with_carry
  let mut hi_result_lc = SmallLinearCombination::zero();
  let mut coeff: i32 = 1;
  for i in 0..16 {
    let bit = SmallBit::alloc(
      &mut cs.namespace(|| format!("hi{i}")),
      hi_sum_with_carry.map(|v| (v >> i) & 1 == 1),
    )?;
    hi_result_lc.add_term(bit.get_variable(), coeff);
    bits[16 + i] = SmallBoolean::Is(bit);
    coeff *= 2;
  }

  // Allocate overflow bits from hi_sum_with_carry (bits 16+, discarded)
  for i in 0..num_carry_bits {
    let bit = SmallBit::alloc(
      &mut cs.namespace(|| format!("o{i}")),
      hi_sum_with_carry.map(|v| (v >> (16 + i)) & 1 == 1),
    )?;
    hi_result_lc.add_term(bit.get_variable(), coeff);
    coeff *= 2;
  }

  // Enforce: hi_operands_lc = hi_result_lc
  cs.enforce_equal(&hi_operands_lc, &hi_result_lc);

  Ok(SmallUInt32::from_bits_le(&bits))
}

/// Helper: add a SmallBoolean's contribution to a SmallLinearCombination<i32>.
///
/// - `Constant(false)` → nothing
/// - `Constant(true)` → add coeff to ONE term
/// - `Is(bit)` → add coeff * bit
/// - `Not(bit)` → add coeff * ONE, add -coeff * bit
fn add_boolean_to_lc(lc: &mut SmallLinearCombination<i32>, boolean: &SmallBoolean, coeff: i32) {
  use bellpepper_core::Index;
  let one_var = bellpepper_core::Variable::new_unchecked(Index::Input(0));
  match boolean {
    SmallBoolean::Constant(false) => {}
    SmallBoolean::Constant(true) => {
      lc.add_term(one_var, coeff);
    }
    SmallBoolean::Is(bit) => {
      lc.add_term(bit.get_variable(), coeff);
    }
    SmallBoolean::Not(bit) => {
      lc.add_term(one_var, coeff);
      lc.add_term(bit.get_variable(), -coeff);
    }
  }
}
