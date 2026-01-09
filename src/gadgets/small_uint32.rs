// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallUInt32: 32-bit unsigned integer gadget using SmallMultiEq.
//!
//! This is a port of bellpepper's `UInt32` with one key change:
//! `addmany` uses `SmallMultiEq` instead of `MultiEq` for batched carry constraints.
//!
//! # Why SmallUInt32?
//!
//! Bellpepper's `UInt32::addmany` has:
//! ```ignore
//! where M: ConstraintSystem<Scalar, Root = MultiEq<Scalar, CS>>
//! ```
//!
//! It's hardcoded to bellpepper's `MultiEq`. We need our own copy that uses
//! `SmallMultiEq` to keep constraint coefficients within `SmallValueField` bounds.
//!
//! # SHA256 Operations
//!
//! | Operation | Uses SmallMultiEq? |
//! |-----------|-------------------|
//! | `addmany()` | **Yes** - carry constraints |
//! | `xor()` | No - delegates to Boolean |
//! | `rotr()` | No - just reorders bits |
//! | `shr()` | No - inserts zero bits |
//! | `sha256_ch/maj()` | No - uses AND/XOR |

use super::small_multi_eq::SmallMultiEq;
use crate::small_field::{SmallMultiEqConfig, SmallValueField};
use bellpepper_core::{
  ConstraintSystem, LinearCombination, SynthesisError,
  boolean::{AllocatedBit, Boolean},
};
use ff::PrimeField;

/// A 32-bit unsigned integer for circuits with small-value optimization.
#[derive(Clone, Debug)]
pub struct SmallUInt32 {
  /// Little-endian bit representation
  bits: Vec<Boolean>,
  /// Cached value (if known)
  value: Option<u32>,
}

impl SmallUInt32 {
  /// Construct a `SmallUInt32` from a `Boolean` vector.
  /// Bits are in little-endian order.
  pub fn from_bits_le(bits: &[Boolean]) -> Self {
    assert_eq!(bits.len(), 32);

    let value = bits.iter().rev().try_fold(0u32, |acc, bit| {
      bit
        .get_value()
        .map(|b| if b { (acc << 1) | 1 } else { acc << 1 })
    });

    SmallUInt32 {
      bits: bits.to_vec(),
      value,
    }
  }

  /// Construct a `SmallUInt32` from a `Boolean` vector in big-endian order.
  pub fn from_bits_be(bits: &[Boolean]) -> Self {
    assert_eq!(bits.len(), 32);
    let mut bits = bits.to_vec();
    bits.reverse();
    Self::from_bits_le(&bits)
  }

  /// Get the bits in little-endian order.
  pub fn bits_le(&self) -> &[Boolean] {
    &self.bits
  }

  /// Get the bits in big-endian order.
  pub fn into_bits_be(self) -> Vec<Boolean> {
    let mut bits = self.bits;
    bits.reverse();
    bits
  }

  /// Get the value if known.
  pub fn get_value(&self) -> Option<u32> {
    self.value
  }

  /// Create a constant `SmallUInt32`.
  pub fn constant(value: u32) -> Self {
    let bits: Vec<Boolean> = (0..32)
      .map(|i| Boolean::constant((value >> i) & 1 == 1))
      .collect();

    SmallUInt32 {
      bits,
      value: Some(value),
    }
  }

  /// Allocate a `SmallUInt32` in the constraint system.
  pub fn alloc<Scalar, CS>(mut cs: CS, value: Option<u32>) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let bits: Vec<Boolean> = (0..32)
      .map(|i| {
        AllocatedBit::alloc(
          cs.namespace(|| format!("bit {}", i)),
          value.map(|v| (v >> i) & 1 == 1),
        )
        .map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;

    Ok(SmallUInt32 { bits, value })
  }

  /// Right rotation.
  pub fn rotr(&self, by: usize) -> Self {
    let by = by % 32;
    let bits: Vec<Boolean> = self
      .bits
      .iter()
      .cycle()
      .skip(by)
      .take(32)
      .cloned()
      .collect();

    SmallUInt32 {
      bits,
      value: self.value.map(|v| v.rotate_right(by as u32)),
    }
  }

  /// Right shift.
  pub fn shr(&self, by: usize) -> Self {
    let bits: Vec<Boolean> = (0..32)
      .map(|i| {
        if i + by < 32 {
          self.bits[i + by].clone()
        } else {
          Boolean::constant(false)
        }
      })
      .collect();

    SmallUInt32 {
      bits,
      value: self.value.map(|v| v >> by),
    }
  }

  /// XOR with another `SmallUInt32`.
  pub fn xor<Scalar, CS>(&self, mut cs: CS, other: &Self) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let bits: Vec<Boolean> = self
      .bits
      .iter()
      .zip(other.bits.iter())
      .enumerate()
      .map(|(i, (a, b))| Boolean::xor(cs.namespace(|| format!("xor bit {}", i)), a, b))
      .collect::<Result<Vec<_>, _>>()?;

    Ok(SmallUInt32 {
      bits,
      value: self.value.and_then(|a| other.value.map(|b| a ^ b)),
    })
  }

  /// SHA-256 CH function: (a AND b) XOR ((NOT a) AND c)
  pub fn sha256_ch<Scalar, CS>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    c: &Self,
  ) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let bits: Vec<Boolean> = a
      .bits
      .iter()
      .zip(b.bits.iter())
      .zip(c.bits.iter())
      .enumerate()
      .map(|(i, ((a, b), c))| Boolean::sha256_ch(cs.namespace(|| format!("ch bit {}", i)), a, b, c))
      .collect::<Result<Vec<_>, _>>()?;

    Ok(SmallUInt32 {
      bits,
      value: a
        .value
        .and_then(|a| b.value.and_then(|b| c.value.map(|c| (a & b) ^ ((!a) & c)))),
    })
  }

  /// SHA-256 MAJ function: (a AND b) XOR (a AND c) XOR (b AND c)
  ///
  /// Optimized identity: Maj(a,b,c) = (a & b) ^ (c & (a ^ b))
  /// This uses 2 AND + 2 XOR per bit instead of 3 AND + 2 XOR.
  pub fn sha256_maj<Scalar, CS>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    c: &Self,
  ) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let bits: Vec<Boolean> = a
      .bits
      .iter()
      .zip(b.bits.iter())
      .zip(c.bits.iter())
      .enumerate()
      .map(|(i, ((a_bit, b_bit), c_bit))| {
        // Optimized: Maj(a,b,c) = (a & b) ^ (c & (a ^ b))
        // t = a ^ b
        let t = Boolean::xor(
          cs.namespace(|| format!("maj_axorb bit {}", i)),
          a_bit,
          b_bit,
        )?;
        // u = c & t
        let u = Boolean::and(cs.namespace(|| format!("maj_candt bit {}", i)), c_bit, &t)?;
        // v = a & b
        let v = Boolean::and(
          cs.namespace(|| format!("maj_aandb bit {}", i)),
          a_bit,
          b_bit,
        )?;
        // maj = v ^ u
        Boolean::xor(cs.namespace(|| format!("maj_result bit {}", i)), &v, &u)
      })
      .collect::<Result<Vec<_>, _>>()?;

    Ok(SmallUInt32 {
      bits,
      value: a.value.and_then(|a| {
        b.value
          .and_then(|b| c.value.map(|c| (a & b) ^ (a & c) ^ (b & c)))
      }),
    })
  }

  /// Add multiple `SmallUInt32`s together using `SmallMultiEq` for batched carry constraints.
  ///
  /// This is the key function that differs from bellpepper's `UInt32::addmany`:
  /// it uses `SmallMultiEq` instead of `MultiEq` to keep constraint coefficients bounded.
  ///
  /// For i32 (I32NoBatch<Fq>): Uses 16-bit limbed addition to keep max coefficient at 2^18.
  /// For i64 (I64Batch21<Fq>): Uses full 35-bit addition (max coefficient 2^34 fits in i64).
  pub fn addmany<Scalar, CS, C, M>(cs: M, operands: &[Self]) -> Result<Self, SynthesisError>
  where
    Scalar: SmallValueField<C::SmallValue>,
    CS: ConstraintSystem<Scalar>,
    C: SmallMultiEqConfig,
    M: ConstraintSystem<Scalar, Root = SmallMultiEq<Scalar, CS, C>>,
  {
    // Make some constraints about the number of operands
    assert!(Scalar::NUM_BITS >= 64);
    assert!(operands.len() >= 2);
    assert!(operands.len() <= 10); // Reasonable limit

    // If all operands are constant, return constant result
    if operands.iter().all(|op| op.value.is_some()) {
      let all_constant = operands
        .iter()
        .all(|op| op.bits.iter().all(|b| matches!(b, Boolean::Constant(_))));
      if all_constant {
        let sum: u32 = operands
          .iter()
          .map(|op| op.value.unwrap())
          .fold(0u32, |a, b| a.wrapping_add(b));
        return Ok(Self::constant(sum));
      }
    }

    // Dispatch based on small value type size:
    // - i32 (4 bytes): Use limbed addition (max coeff 2^18)
    // - i64 (8 bytes): Use full addition (max coeff 2^34)
    if std::mem::size_of::<C::SmallValue>() <= 4 {
      Self::addmany_limbed::<Scalar, CS, C, M>(cs, operands)
    } else {
      Self::addmany_full::<Scalar, CS, C, M>(cs, operands)
    }
  }

  /// Full 35-bit addition for i64 path.
  /// Max coefficient: 2^34 (for 5 operands producing 35-bit result).
  fn addmany_full<Scalar, CS, C, M>(mut cs: M, operands: &[Self]) -> Result<Self, SynthesisError>
  where
    Scalar: SmallValueField<C::SmallValue>,
    CS: ConstraintSystem<Scalar>,
    C: SmallMultiEqConfig,
    M: ConstraintSystem<Scalar, Root = SmallMultiEq<Scalar, CS, C>>,
  {
    // Compute the maximum value of the sum
    let max_value = (operands.len() as u64) * (u32::MAX as u64);

    // How many bits do we need to represent the result?
    let result_bits = 64 - max_value.leading_zeros() as usize;

    // Compute the value of the result
    let result_value = operands
      .iter()
      .try_fold(0u64, |acc, op| op.value.map(|v| acc + (v as u64)));

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
      for bit in &op.bits {
        all_operands_lc = all_operands_lc + &bit.lc(CS::one(), coeff);
        coeff = coeff.double();
      }
    }

    // Enforce that the result equals the sum of operands
    cs.get_root().enforce_equal(&lc, &all_operands_lc);

    // Truncate to 32 bits
    let bits: Vec<Boolean> = result_bits_vec.into_iter().take(32).collect();
    assert_eq!(bits.len(), 32);

    Ok(SmallUInt32 {
      bits,
      value: result_value.map(|v| v as u32),
    })
  }

  /// 16-bit limbed addition for i32 path.
  /// Splits each 32-bit value into two 16-bit limbs and adds them separately.
  /// Max coefficient: 2^18 (fits in i32).
  ///
  /// Constraint 1 (low limb):
  ///   Σ(operand_lo) = result_lo + carry × 2^16
  ///
  /// Constraint 2 (high limb):
  ///   Σ(operand_hi) + carry = result_hi + overflow × 2^16
  fn addmany_limbed<Scalar, CS, C, M>(mut cs: M, operands: &[Self]) -> Result<Self, SynthesisError>
  where
    Scalar: SmallValueField<C::SmallValue>,
    CS: ConstraintSystem<Scalar>,
    C: SmallMultiEqConfig,
    M: ConstraintSystem<Scalar, Root = SmallMultiEq<Scalar, CS, C>>,
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
      op.value.map(|v| acc + ((v as u64) & 0xFFFF))
    });

    // Compute carry value
    let carry_value: Option<u64> = lo_sum.map(|v| v >> 16);

    // Compute high limb sum (for witness generation)
    // hi_sum = Σ(operand_hi) + carry
    let hi_sum: Option<u64> = operands.iter().try_fold(0u64, |acc, op| {
      op.value.map(|v| acc + (((v as u64) >> 16) & 0xFFFF))
    });
    let hi_sum_with_carry: Option<u64> = hi_sum.and_then(|h| carry_value.map(|c| h + c));

    // Final result value (for returning)
    let result_value: Option<u64> = operands
      .iter()
      .try_fold(0u64, |acc, op| op.value.map(|v| acc + (v as u64)));

    // === LOW LIMB CONSTRAINT ===
    // Sum of low 16 bits of each operand = low 16 bits of result + carry × 2^16

    // Build LHS: sum of all operand low limbs
    let mut lo_operands_lc = LinearCombination::zero();
    for op in operands.iter() {
      let mut coeff = Scalar::ONE;
      for bit in &op.bits[0..16] {
        lo_operands_lc = lo_operands_lc + &bit.lc(CS::one(), coeff);
        coeff = coeff.double();
      }
    }

    // Allocate result low bits (0..15) from lo_sum
    let mut lo_result_lc = LinearCombination::zero();
    let mut result_bits: Vec<Boolean> = Vec::with_capacity(32);
    let mut carry_bits: Vec<AllocatedBit> = Vec::with_capacity(num_carry_bits);

    let mut coeff = Scalar::ONE;
    for i in 0..16 {
      let bit = AllocatedBit::alloc(
        cs.namespace(|| format!("lo_result_bit_{}", i)),
        lo_sum.map(|v| (v >> i) & 1 == 1),
      )?;
      lo_result_lc = lo_result_lc + (coeff, bit.get_variable());
      result_bits.push(Boolean::from(bit));
      coeff = coeff.double();
    }

    // Allocate carry bits from lo_sum (bits 16+)
    for i in 0..num_carry_bits {
      let bit = AllocatedBit::alloc(
        cs.namespace(|| format!("carry_bit_{}", i)),
        lo_sum.map(|v| (v >> (16 + i)) & 1 == 1),
      )?;
      lo_result_lc = lo_result_lc + (coeff, bit.get_variable());
      carry_bits.push(bit);
      coeff = coeff.double();
    }

    // Enforce: lo_operands_lc = lo_result_lc
    cs.get_root().enforce_equal(&lo_operands_lc, &lo_result_lc);

    // === HIGH LIMB CONSTRAINT ===
    // Sum of high 16 bits of each operand + carry = high 16 bits of result + overflow × 2^16

    // Build LHS: sum of all operand high limbs + carry
    let mut hi_operands_lc = LinearCombination::zero();
    for op in operands.iter() {
      let mut coeff = Scalar::ONE;
      for bit in &op.bits[16..32] {
        hi_operands_lc = hi_operands_lc + &bit.lc(CS::one(), coeff);
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
        cs.namespace(|| format!("hi_result_bit_{}", i)),
        hi_sum_with_carry.map(|v| (v >> i) & 1 == 1),
      )?;
      hi_result_lc = hi_result_lc + (coeff, bit.get_variable());
      result_bits.push(Boolean::from(bit));
      coeff = coeff.double();
    }

    // Allocate overflow bits from hi_sum_with_carry (bits 16+, discarded)
    for i in 0..num_carry_bits {
      let bit = AllocatedBit::alloc(
        cs.namespace(|| format!("overflow_bit_{}", i)),
        hi_sum_with_carry.map(|v| (v >> (16 + i)) & 1 == 1),
      )?;
      hi_result_lc = hi_result_lc + (coeff, bit.get_variable());
      coeff = coeff.double();
    }

    // Enforce: hi_operands_lc = hi_result_lc
    cs.get_root().enforce_equal(&hi_operands_lc, &hi_result_lc);

    assert_eq!(result_bits.len(), 32);

    Ok(SmallUInt32 {
      bits: result_bits,
      value: result_value.map(|v| v as u32),
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::small_field::I32NoBatch;
  use bellpepper_core::test_cs::TestConstraintSystem;
  use halo2curves::pasta::Fq;

  #[test]
  fn test_small_uint32_constant() {
    let u = SmallUInt32::constant(0x12345678);
    assert_eq!(u.get_value(), Some(0x12345678));
  }

  #[test]
  fn test_small_uint32_rotr() {
    let u = SmallUInt32::constant(0x80000001);
    let rotated = u.rotr(1);
    assert_eq!(rotated.get_value(), Some(0xC0000000));
  }

  #[test]
  fn test_small_uint32_shr() {
    let u = SmallUInt32::constant(0x80000000);
    let shifted = u.shr(1);
    assert_eq!(shifted.get_value(), Some(0x40000000));
  }

  #[test]
  fn test_small_uint32_xor() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::constant(0xAAAAAAAA);
    let b = SmallUInt32::constant(0x55555555);

    let result = a.xor(&mut cs, &b).unwrap();
    assert_eq!(result.get_value(), Some(0xFFFFFFFF));
    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_small_uint32_addmany() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::alloc(cs.namespace(|| "a"), Some(100)).unwrap();
    let b = SmallUInt32::alloc(cs.namespace(|| "b"), Some(200)).unwrap();
    let c = SmallUInt32::alloc(cs.namespace(|| "c"), Some(300)).unwrap();

    {
      let mut multi_eq = SmallMultiEq::<_, _, I32NoBatch<Fq>>::new(&mut cs);
      let result = SmallUInt32::addmany(multi_eq.namespace(|| "add"), &[a, b, c]).unwrap();
      assert_eq!(result.get_value(), Some(600));
    }

    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_small_uint32_addmany_overflow() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::alloc(cs.namespace(|| "a"), Some(0xFFFFFFFF)).unwrap();
    let b = SmallUInt32::alloc(cs.namespace(|| "b"), Some(1)).unwrap();

    {
      let mut multi_eq = SmallMultiEq::<_, _, I32NoBatch<Fq>>::new(&mut cs);
      let result = SmallUInt32::addmany(multi_eq.namespace(|| "add"), &[a, b]).unwrap();
      // Should wrap to 0
      assert_eq!(result.get_value(), Some(0));
    }

    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_small_uint32_addmany_small64() {
    use crate::small_field::I64Batch21;

    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::alloc(cs.namespace(|| "a"), Some(100)).unwrap();
    let b = SmallUInt32::alloc(cs.namespace(|| "b"), Some(200)).unwrap();
    let c = SmallUInt32::alloc(cs.namespace(|| "c"), Some(300)).unwrap();

    {
      // Use I64Batch21<Fq> config (full 35-bit addition path)
      let mut multi_eq = SmallMultiEq::<_, _, I64Batch21<Fq>>::new(&mut cs);
      let result = SmallUInt32::addmany(multi_eq.namespace(|| "add"), &[a, b, c]).unwrap();
      assert_eq!(result.get_value(), Some(600));
    }

    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_small_uint32_addmany_5_operands() {
    // SHA-256 uses 5-operand addition
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::alloc(cs.namespace(|| "a"), Some(0x12345678)).unwrap();
    let b = SmallUInt32::alloc(cs.namespace(|| "b"), Some(0x87654321)).unwrap();
    let c = SmallUInt32::alloc(cs.namespace(|| "c"), Some(0xDEADBEEF)).unwrap();
    let d = SmallUInt32::alloc(cs.namespace(|| "d"), Some(0xCAFEBABE)).unwrap();
    let e = SmallUInt32::alloc(cs.namespace(|| "e"), Some(0x01020304)).unwrap();

    let expected = 0x12345678u32
      .wrapping_add(0x87654321)
      .wrapping_add(0xDEADBEEF)
      .wrapping_add(0xCAFEBABE)
      .wrapping_add(0x01020304);

    {
      let mut multi_eq = SmallMultiEq::<_, _, I32NoBatch<Fq>>::new(&mut cs);
      let result = SmallUInt32::addmany(multi_eq.namespace(|| "add"), &[a, b, c, d, e]).unwrap();
      assert_eq!(result.get_value(), Some(expected));
    }

    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_small_uint32_sha256_ch() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // ch(a, b, c) = (a & b) ^ ((!a) & c)
    let a = SmallUInt32::constant(0xFF00FF00);
    let b = SmallUInt32::constant(0xF0F0F0F0);
    let c = SmallUInt32::constant(0x0F0F0F0F);

    let result = SmallUInt32::sha256_ch(&mut cs, &a, &b, &c).unwrap();

    // Expected: (0xFF00FF00 & 0xF0F0F0F0) ^ ((~0xFF00FF00) & 0x0F0F0F0F)
    //         = 0xF000F000 ^ (0x00FF00FF & 0x0F0F0F0F)
    //         = 0xF000F000 ^ 0x000F000F
    //         = 0xF00FF00F
    assert_eq!(result.get_value(), Some(0xF00FF00F));
    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_small_uint32_sha256_maj() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    // maj(a, b, c) = (a & b) ^ (a & c) ^ (b & c)
    let a = SmallUInt32::constant(0xFF00FF00);
    let b = SmallUInt32::constant(0xF0F0F0F0);
    let c = SmallUInt32::constant(0x0F0F0F0F);

    let result = SmallUInt32::sha256_maj(&mut cs, &a, &b, &c).unwrap();

    // Expected: (a & b) ^ (a & c) ^ (b & c)
    // = (0xFF00FF00 & 0xF0F0F0F0) ^ (0xFF00FF00 & 0x0F0F0F0F) ^ (0xF0F0F0F0 & 0x0F0F0F0F)
    // = 0xF000F000 ^ 0x0F000F00 ^ 0x00000000
    // = 0xFF00FF00
    let expected = 0xFF00FF00u32;
    assert_eq!(result.get_value(), Some(expected));
    assert!(cs.is_satisfied());
  }
}
