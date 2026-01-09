// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Addition algorithms for SmallUInt32 values.
//!
//! This module provides two addition algorithms optimized for different
//! coefficient bounds in the small-value sumcheck optimization:
//!
//! - [`limbed`]: 16-bit limbed addition, max coefficient 2^18 (fits i32)
//! - [`full`]: Full 35-bit addition, max coefficient 2^34 (fits i64)

use super::{small_multi_eq::SmallMultiEq, small_uint32::SmallUInt32};
use bellpepper_core::{
  LinearCombination, SynthesisError,
  boolean::{AllocatedBit, Boolean},
};
use ff::PrimeField;

/// Full 35-bit addition for i64 path.
///
/// Computes the sum of multiple SmallUInt32 operands, allocating enough bits
/// to represent the full sum before truncating to 32 bits.
///
/// Max coefficient: 2^34 (for 5 operands producing 35-bit result).
pub(crate) fn full<Scalar, M>(
  cs: &mut M,
  operands: &[SmallUInt32],
) -> Result<SmallUInt32, SynthesisError>
where
  Scalar: PrimeField,
  M: SmallMultiEq<Scalar>,
{
  // Compute the maximum value of the sum
  let max_value = (operands.len() as u64) * (u32::MAX as u64);

  // How many bits do we need to represent the result?
  let result_bits = 64 - max_value.leading_zeros() as usize;

  // Compute the value of the result
  let result_value = operands
    .iter()
    .try_fold(0u64, |acc, op| op.get_value().map(|v| acc + (v as u64)));

  // Allocate each bit of the result
  let mut result_bits_vec: Vec<Boolean> = Vec::with_capacity(result_bits);
  let mut coeff = Scalar::ONE;
  let mut lc = LinearCombination::zero();
  let mut all_operands_lc = LinearCombination::zero();

  for i in 0..result_bits {
    // Allocate the bit
    let bit = AllocatedBit::alloc(
      cs.namespace(|| format!("result bit {}", i)),
      result_value.map(|v| (v >> i) & 1 == 1),
    )?;

    // Add to linear combination
    lc = lc + (coeff, bit.get_variable());

    result_bits_vec.push(Boolean::from(bit));
    coeff = coeff.double();
  }

  // Compute linear combination of all operand bits
  for op in operands.iter() {
    let mut coeff = Scalar::ONE;
    for bit in op.bits_le() {
      all_operands_lc = all_operands_lc + &bit.lc(M::one(), coeff);
      coeff = coeff.double();
    }
  }

  // Enforce that the result equals the sum of operands
  cs.enforce_equal(&lc, &all_operands_lc);

  // Truncate to 32 bits
  let bits: [Boolean; 32] = result_bits_vec
    .into_iter()
    .take(32)
    .collect::<Vec<_>>()
    .try_into()
    .unwrap();

  Ok(SmallUInt32::from_bits_le(&bits))
}

/// 16-bit limbed addition for i32 path.
///
/// Splits each 32-bit value into two 16-bit limbs and adds them separately.
/// This keeps the maximum coefficient at 2^18, which fits in i32.
///
/// Constraint 1 (low limb):
///   Σ(operand_lo) = result_lo + carry × 2^16
///
/// Constraint 2 (high limb):
///   Σ(operand_hi) + carry = result_hi + overflow × 2^16
pub(crate) fn limbed<Scalar, M>(
  cs: &mut M,
  operands: &[SmallUInt32],
) -> Result<SmallUInt32, SynthesisError>
where
  Scalar: PrimeField,
  M: SmallMultiEq<Scalar>,
{
  // For N operands, each 16-bit limb sum can be up to N * (2^16 - 1)
  // For 10 operands: 10 * 65535 = 655350, needs 20 bits
  // We allocate 16 result bits + up to 4 carry/overflow bits
  let num_carry_bits =
    64 - ((operands.len() as u64) * (u16::MAX as u64)).leading_zeros() as usize - 16;
  let num_carry_bits = num_carry_bits.max(1); // At least 1 carry bit

  // Compute low limb sum (for witness generation)
  // IMPORTANT: Carry must be computed from the LOW LIMB SUM, not from the full result
  let lo_sum: Option<u64> = operands.iter().try_fold(0u64, |acc, op| {
    op.get_value().map(|v| acc + ((v as u64) & 0xFFFF))
  });

  // Compute high limb sum (for witness generation)
  // hi_sum = Σ(operand_hi) + carry
  let hi_sum: Option<u64> = operands.iter().try_fold(0u64, |acc, op| {
    op.get_value().map(|v| acc + (((v as u64) >> 16) & 0xFFFF))
  });
  let hi_sum_with_carry: Option<u64> = hi_sum.and_then(|h| lo_sum.map(|l| h + (l >> 16)));

  // === LOW LIMB CONSTRAINT ===
  // Sum of low 16 bits of each operand = low 16 bits of result + carry × 2^16

  // Build LHS: sum of all operand low limbs
  let mut lo_operands_lc = LinearCombination::zero();
  for op in operands.iter() {
    let mut coeff = Scalar::ONE;
    for bit in &op.bits_le()[0..16] {
      lo_operands_lc = lo_operands_lc + &bit.lc(M::one(), coeff);
      coeff = coeff.double();
    }
  }

  // Allocate result low bits (0..15) from lo_sum
  let mut lo_result_lc = LinearCombination::zero();
  let mut bits = [const { Boolean::Constant(false) }; 32];
  let mut carry_bits: Vec<AllocatedBit> = Vec::with_capacity(num_carry_bits);

  let mut coeff = Scalar::ONE;
  for (i, slot) in bits.iter_mut().enumerate().take(16) {
    let bit = AllocatedBit::alloc(
      cs.namespace(|| format!("lo{i}")),
      lo_sum.map(|v| (v >> i) & 1 == 1),
    )?;
    lo_result_lc = lo_result_lc + (coeff, bit.get_variable());
    *slot = Boolean::from(bit);
    coeff = coeff.double();
  }

  // Allocate carry bits from lo_sum (bits 16+)
  for i in 0..num_carry_bits {
    let bit = AllocatedBit::alloc(
      cs.namespace(|| format!("c{i}")),
      lo_sum.map(|v| (v >> (16 + i)) & 1 == 1),
    )?;
    lo_result_lc = lo_result_lc + (coeff, bit.get_variable());
    carry_bits.push(bit);
    coeff = coeff.double();
  }

  // Enforce: lo_operands_lc = lo_result_lc
  cs.enforce_equal(&lo_operands_lc, &lo_result_lc);

  // === HIGH LIMB CONSTRAINT ===
  // Sum of high 16 bits of each operand + carry = high 16 bits of result + overflow × 2^16

  // Build LHS: sum of all operand high limbs + carry
  let mut hi_operands_lc = LinearCombination::zero();
  for op in operands.iter() {
    let mut coeff = Scalar::ONE;
    for bit in &op.bits_le()[16..32] {
      hi_operands_lc = hi_operands_lc + &bit.lc(M::one(), coeff);
      coeff = coeff.double();
    }
  }

  // Add carry from low limb
  let mut coeff = Scalar::ONE;
  for carry_bit in &carry_bits {
    hi_operands_lc = hi_operands_lc + (coeff, carry_bit.get_variable());
    coeff = coeff.double();
  }

  // Allocate result high bits (16..31) from hi_sum_with_carry
  let mut hi_result_lc = LinearCombination::zero();
  let mut coeff = Scalar::ONE;
  for i in 0..16 {
    let bit = AllocatedBit::alloc(
      cs.namespace(|| format!("hi{i}")),
      hi_sum_with_carry.map(|v| (v >> i) & 1 == 1),
    )?;
    hi_result_lc = hi_result_lc + (coeff, bit.get_variable());
    bits[16 + i] = Boolean::from(bit);
    coeff = coeff.double();
  }

  // Allocate overflow bits from hi_sum_with_carry (bits 16+, discarded)
  for i in 0..num_carry_bits {
    let bit = AllocatedBit::alloc(
      cs.namespace(|| format!("o{i}")),
      hi_sum_with_carry.map(|v| (v >> (16 + i)) & 1 == 1),
    )?;
    hi_result_lc = hi_result_lc + (coeff, bit.get_variable());
    coeff = coeff.double();
  }

  // Enforce: hi_operands_lc = hi_result_lc
  cs.enforce_equal(&hi_operands_lc, &hi_result_lc);

  Ok(SmallUInt32::from_bits_le(&bits))
}
