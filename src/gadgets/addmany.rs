// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Addition algorithms for [`SmallUInt32`](super::SmallUInt32).

use super::{small_multi_eq::SmallMultiEq, small_uint32::SmallUInt32};
use bellpepper_core::{
  LinearCombination, SynthesisError,
  boolean::{AllocatedBit, Boolean},
};
use ff::PrimeField;

pub(crate) fn full<Scalar, M>(
  cs: &mut M,
  operands: &[SmallUInt32],
) -> Result<SmallUInt32, SynthesisError>
where
  Scalar: PrimeField,
  M: SmallMultiEq<Scalar>,
{
  let max_value = (operands.len() as u64) * (u32::MAX as u64);
  let result_bits = 64 - max_value.leading_zeros() as usize;
  let result_value = operands
    .iter()
    .try_fold(0u64, |acc, op| op.get_value().map(|v| acc + (v as u64)));

  let mut result_bits_vec = Vec::with_capacity(result_bits);
  let mut coeff = Scalar::ONE;
  let mut lc = LinearCombination::zero();
  let mut all_operands_lc = LinearCombination::zero();

  for i in 0..result_bits {
    let bit = AllocatedBit::alloc(
      cs.namespace(|| format!("result bit {}", i)),
      result_value.map(|v| (v >> i) & 1 == 1),
    )?;
    lc = lc + (coeff, bit.get_variable());
    result_bits_vec.push(Boolean::from(bit));
    coeff = coeff.double();
  }

  for op in operands {
    let mut coeff = Scalar::ONE;
    for bit in op.bits_le() {
      all_operands_lc = all_operands_lc + &bit.lc(M::one(), coeff);
      coeff = coeff.double();
    }
  }

  cs.enforce_equal(&lc, &all_operands_lc);

  let bits: [Boolean; 32] = result_bits_vec
    .into_iter()
    .take(32)
    .collect::<Vec<_>>()
    .try_into()
    .unwrap();
  Ok(SmallUInt32::from_bits_le(&bits))
}

pub(crate) fn limbed<Scalar, M>(
  cs: &mut M,
  operands: &[SmallUInt32],
) -> Result<SmallUInt32, SynthesisError>
where
  Scalar: PrimeField,
  M: SmallMultiEq<Scalar>,
{
  let num_carry_bits =
    64 - ((operands.len() as u64) * (u16::MAX as u64)).leading_zeros() as usize - 16;
  let num_carry_bits = num_carry_bits.max(1);

  let lo_sum = operands.iter().try_fold(0u64, |acc, op| {
    op.get_value().map(|v| acc + ((v as u64) & 0xFFFF))
  });
  let hi_sum = operands.iter().try_fold(0u64, |acc, op| {
    op.get_value().map(|v| acc + (((v as u64) >> 16) & 0xFFFF))
  });
  let hi_sum_with_carry = hi_sum.and_then(|h| lo_sum.map(|l| h + (l >> 16)));

  let mut lo_operands_lc = LinearCombination::zero();
  for op in operands {
    let mut coeff = Scalar::ONE;
    for bit in &op.bits_le()[0..16] {
      lo_operands_lc = lo_operands_lc + &bit.lc(M::one(), coeff);
      coeff = coeff.double();
    }
  }

  let mut lo_result_lc = LinearCombination::zero();
  let mut bits = [const { Boolean::Constant(false) }; 32];
  let mut carry_bits = Vec::with_capacity(num_carry_bits);

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

  for i in 0..num_carry_bits {
    let bit = AllocatedBit::alloc(
      cs.namespace(|| format!("c{i}")),
      lo_sum.map(|v| (v >> (16 + i)) & 1 == 1),
    )?;
    lo_result_lc = lo_result_lc + (coeff, bit.get_variable());
    carry_bits.push(bit);
    coeff = coeff.double();
  }

  cs.enforce_equal(&lo_operands_lc, &lo_result_lc);

  let mut hi_operands_lc = LinearCombination::zero();
  for op in operands {
    let mut coeff = Scalar::ONE;
    for bit in &op.bits_le()[16..32] {
      hi_operands_lc = hi_operands_lc + &bit.lc(M::one(), coeff);
      coeff = coeff.double();
    }
  }

  let mut coeff = Scalar::ONE;
  for carry_bit in &carry_bits {
    hi_operands_lc = hi_operands_lc + (coeff, carry_bit.get_variable());
    coeff = coeff.double();
  }

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

  for i in 0..num_carry_bits {
    let bit = AllocatedBit::alloc(
      cs.namespace(|| format!("o{i}")),
      hi_sum_with_carry.map(|v| (v >> (16 + i)) & 1 == 1),
    )?;
    hi_result_lc = hi_result_lc + (coeff, bit.get_variable());
    coeff = coeff.double();
  }

  cs.enforce_equal(&hi_operands_lc, &hi_result_lc);

  Ok(SmallUInt32::from_bits_le(&bits))
}
