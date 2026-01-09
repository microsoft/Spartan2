// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallUInt32: 32-bit unsigned integer gadget for small-value sumcheck.
//!
//! This is a port of bellpepper's `UInt32` with bit operations for SHA-256.
//! Addition is handled externally via the `SmallMultiEq` trait's `addmany` method.
//!
//! # SHA256 Operations
//!
//! | Operation | Constraints |
//! |-----------|-------------|
//! | `xor()` | Delegates to Boolean |
//! | `rotr()` | No constraints - just reorders bits |
//! | `shr()` | No constraints - inserts zero bits |
//! | `sha256_ch/maj()` | Uses AND/XOR |

use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
};
use ff::PrimeField;

/// A 32-bit unsigned integer for circuits with small-value optimization.
#[derive(Clone, Debug)]
pub struct SmallUInt32 {
  /// Little-endian bit representation
  bits: [Boolean; 32],
  /// Cached value (if known)
  value: Option<u32>,
}

impl SmallUInt32 {
  /// Construct a `SmallUInt32` from a `Boolean` array.
  /// Bits are in little-endian order.
  pub fn from_bits_le(bits: &[Boolean; 32]) -> Self {
    let value = bits.iter().rev().try_fold(0u32, |acc, bit| {
      bit
        .get_value()
        .map(|b| if b { (acc << 1) | 1 } else { acc << 1 })
    });

    SmallUInt32 {
      bits: bits.clone(),
      value,
    }
  }

  /// Construct a `SmallUInt32` from a `Boolean` array in big-endian order.
  pub fn from_bits_be(bits: &[Boolean; 32]) -> Self {
    let mut bits_le = bits.clone();
    bits_le.reverse();
    Self::from_bits_le(&bits_le)
  }

  /// Get the bits in little-endian order.
  pub fn bits_le(&self) -> &[Boolean; 32] {
    &self.bits
  }

  /// Get the bits in big-endian order.
  pub fn into_bits_be(self) -> [Boolean; 32] {
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
    let bits: [Boolean; 32] = std::array::from_fn(|i| Boolean::constant((value >> i) & 1 == 1));

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
    let mut bits = [const { Boolean::Constant(false) }; 32];
    for (i, slot) in bits.iter_mut().enumerate() {
      *slot = Boolean::from(AllocatedBit::alloc(
        cs.namespace(|| format!("b{i}")),
        value.map(|v| (v >> i) & 1 == 1),
      )?);
    }
    Ok(SmallUInt32 { bits, value })
  }

  /// Right rotation.
  pub fn rotr(&self, by: usize) -> Self {
    let by = by % 32;
    let bits: [Boolean; 32] = std::array::from_fn(|i| self.bits[(i + by) % 32].clone());

    SmallUInt32 {
      bits,
      value: self.value.map(|v| v.rotate_right(by as u32)),
    }
  }

  /// Right shift.
  pub fn shr(&self, by: usize) -> Self {
    let bits: [Boolean; 32] = std::array::from_fn(|i| {
      if i + by < 32 {
        self.bits[i + by].clone()
      } else {
        Boolean::constant(false)
      }
    });

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
    let mut bits = [const { Boolean::Constant(false) }; 32];
    for (i, (slot, (a, b))) in bits
      .iter_mut()
      .zip(self.bits.iter().zip(other.bits.iter()))
      .enumerate()
    {
      *slot = Boolean::xor(cs.namespace(|| format!("b{i}")), a, b)?;
    }

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
    let mut bits = [const { Boolean::Constant(false) }; 32];
    for (i, (slot, ((a_bit, b_bit), c_bit))) in bits
      .iter_mut()
      .zip(a.bits.iter().zip(b.bits.iter()).zip(c.bits.iter()))
      .enumerate()
    {
      *slot = Boolean::sha256_ch(cs.namespace(|| format!("b{i}")), a_bit, b_bit, c_bit)?;
    }

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
    let mut bits = [const { Boolean::Constant(false) }; 32];
    for (i, (slot, ((a_bit, b_bit), c_bit))) in bits
      .iter_mut()
      .zip(a.bits.iter().zip(b.bits.iter()).zip(c.bits.iter()))
      .enumerate()
    {
      let mut bit_cs = cs.namespace(|| format!("b{i}"));
      // Optimized: Maj(a,b,c) = (a & b) ^ (c & (a ^ b))
      let t = Boolean::xor(bit_cs.namespace(|| "xor_ab"), a_bit, b_bit)?;
      let u = Boolean::and(bit_cs.namespace(|| "and_c_t"), c_bit, &t)?;
      let v = Boolean::and(bit_cs.namespace(|| "and_ab"), a_bit, b_bit)?;
      *slot = Boolean::xor(bit_cs.namespace(|| "xor_vu"), &v, &u)?;
    }

    Ok(SmallUInt32 {
      bits,
      value: a.value.and_then(|a| {
        b.value
          .and_then(|b| c.value.map(|c| (a & b) ^ (a & c) ^ (b & c)))
      }),
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::gadgets::{BatchingEq, NoBatchEq, SmallMultiEq};
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
  fn test_small_uint32_addmany_via_trait() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::alloc(cs.namespace(|| "a"), Some(100)).unwrap();
    let b = SmallUInt32::alloc(cs.namespace(|| "b"), Some(200)).unwrap();
    let c = SmallUInt32::alloc(cs.namespace(|| "c"), Some(300)).unwrap();

    {
      let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
      let result = eq.addmany(&[a, b, c]).unwrap();
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
      let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
      let result = eq.addmany(&[a, b]).unwrap();
      // Should wrap to 0
      assert_eq!(result.get_value(), Some(0));
    }

    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_small_uint32_addmany_batching() {
    let mut cs = TestConstraintSystem::<Fq>::new();

    let a = SmallUInt32::alloc(cs.namespace(|| "a"), Some(100)).unwrap();
    let b = SmallUInt32::alloc(cs.namespace(|| "b"), Some(200)).unwrap();
    let c = SmallUInt32::alloc(cs.namespace(|| "c"), Some(300)).unwrap();

    {
      // Use BatchingEq<21> (full 35-bit addition path)
      let mut eq = BatchingEq::<Fq, _, 21>::new(&mut cs);
      let result = eq.addmany(&[a, b, c]).unwrap();
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
      let mut eq = NoBatchEq::<Fq, _>::new(&mut cs);
      let result = eq.addmany(&[a, b, c, d, e]).unwrap();
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
