// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! A 32-bit unsigned integer gadget for bounded-coefficient SHA-256 circuits.

use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
};
use ff::PrimeField;

/// A 32-bit unsigned integer represented by little-endian Boolean bits.
#[derive(Clone, Debug)]
pub struct SmallUInt32 {
  bits: [Boolean; 32],
  value: Option<u32>,
}

impl SmallUInt32 {
  /// Construct from little-endian bits.
  pub fn from_bits_le(bits: &[Boolean; 32]) -> Self {
    let value = bits.iter().rev().try_fold(0u32, |acc, bit| {
      bit
        .get_value()
        .map(|b| if b { (acc << 1) | 1 } else { acc << 1 })
    });

    Self {
      bits: bits.clone(),
      value,
    }
  }

  /// Construct from big-endian bits.
  pub fn from_bits_be(bits: &[Boolean; 32]) -> Self {
    let mut bits_le = bits.clone();
    bits_le.reverse();
    Self::from_bits_le(&bits_le)
  }

  /// Return little-endian bits.
  pub fn bits_le(&self) -> &[Boolean; 32] {
    &self.bits
  }

  /// Return big-endian bits.
  pub fn into_bits_be(self) -> [Boolean; 32] {
    let mut bits = self.bits;
    bits.reverse();
    bits
  }

  /// Return the native value when all input bits were known.
  pub fn get_value(&self) -> Option<u32> {
    self.value
  }

  /// Create a constant word.
  pub fn constant(value: u32) -> Self {
    let bits = std::array::from_fn(|i| Boolean::constant((value >> i) & 1 == 1));
    Self {
      bits,
      value: Some(value),
    }
  }

  /// Allocate a word as 32 Boolean variables.
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
    Ok(Self { bits, value })
  }

  /// Rotate right by `by` bits.
  pub fn rotr(&self, by: usize) -> Self {
    let by = by % 32;
    let bits = std::array::from_fn(|i| self.bits[(i + by) % 32].clone());
    Self {
      bits,
      value: self.value.map(|v| v.rotate_right(by as u32)),
    }
  }

  /// Shift right by `by` bits.
  pub fn shr(&self, by: usize) -> Self {
    let bits = std::array::from_fn(|i| {
      if i + by < 32 {
        self.bits[i + by].clone()
      } else {
        Boolean::constant(false)
      }
    });
    Self {
      bits,
      value: self.value.map(|v| v >> by),
    }
  }

  /// Bitwise XOR.
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

    Ok(Self {
      bits,
      value: self.value.and_then(|a| other.value.map(|b| a ^ b)),
    })
  }

  /// SHA-256 choose: `(a & b) ^ (!a & c)`.
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

    Ok(Self {
      bits,
      value: a
        .value
        .and_then(|a| b.value.and_then(|b| c.value.map(|c| (a & b) ^ ((!a) & c)))),
    })
  }

  /// SHA-256 majority: `(a & b) ^ (a & c) ^ (b & c)`.
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
      let t = Boolean::xor(bit_cs.namespace(|| "xor_ab"), a_bit, b_bit)?;
      let u = Boolean::and(bit_cs.namespace(|| "and_c_t"), c_bit, &t)?;
      let v = Boolean::and(bit_cs.namespace(|| "and_ab"), a_bit, b_bit)?;
      *slot = Boolean::xor(bit_cs.namespace(|| "xor_vu"), &v, &u)?;
    }

    Ok(Self {
      bits,
      value: a.value.and_then(|a| {
        b.value
          .and_then(|b| c.value.map(|c| (a & b) ^ (a & c) ^ (b & c)))
      }),
    })
  }
}
