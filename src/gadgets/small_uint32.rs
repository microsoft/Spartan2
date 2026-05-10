// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallUInt32: 32-bit unsigned integer gadget for small-value sumcheck.
//!
//! Uses `SmallBoolean` and `SmallConstraintSystem<W, C>` instead of bellpepper types.
//! All bit witnesses are 0 or 1 (i8), all coefficients are i32.
//!
//! # SHA256 Operations
//!
//! | Operation | Constraints |
//! |-----------|-------------|
//! | `xor()` | Delegates to SmallBoolean |
//! | `rotr()` | No constraints - just reorders bits |
//! | `shr()` | No constraints - inserts zero bits |
//! | `sha256_ch/maj()` | Uses optimized SHA identities |

use bellpepper_core::SynthesisError;

use crate::{
  gadgets::small_boolean::{Double, NegOne, SmallBit, SmallBoolean},
  small_constraint_system::SmallConstraintSystem,
};

/// A 32-bit unsigned integer for circuits with small-value optimization.
#[derive(Clone, Debug)]
pub struct SmallUInt32 {
  /// Little-endian bit representation
  bits: [SmallBoolean; 32],
  /// Cached value (if known)
  value: Option<u32>,
}

impl SmallUInt32 {
  /// Construct a `SmallUInt32` from a `SmallBoolean` array.
  /// Bits are in little-endian order.
  pub fn from_bits_le(bits: &[SmallBoolean; 32]) -> Self {
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

  /// Construct a `SmallUInt32` from a `SmallBoolean` array in big-endian order.
  pub fn from_bits_be(bits: &[SmallBoolean; 32]) -> Self {
    let mut bits_le = bits.clone();
    bits_le.reverse();
    Self::from_bits_le(&bits_le)
  }

  /// Get the bits in little-endian order.
  pub fn bits_le(&self) -> &[SmallBoolean; 32] {
    &self.bits
  }

  /// Get the bits in big-endian order.
  pub fn into_bits_be(self) -> [SmallBoolean; 32] {
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
    let bits: [SmallBoolean; 32] =
      std::array::from_fn(|i| SmallBoolean::constant((value >> i) & 1 == 1));

    SmallUInt32 {
      bits,
      value: Some(value),
    }
  }

  /// Allocate a `SmallUInt32` in the constraint system.
  pub fn alloc<W, C, CS>(mut cs: CS, value: Option<u32>) -> Result<Self, SynthesisError>
  where
    W: Copy + From<bool>,
    C: Copy + From<bool> + NegOne,
    CS: SmallConstraintSystem<W, C>,
  {
    let mut bits = [const { SmallBoolean::Constant(false) }; 32];
    for (i, slot) in bits.iter_mut().enumerate() {
      *slot = SmallBoolean::Is(SmallBit::alloc(
        &mut cs.namespace(|| format!("b{i}")),
        value.map(|v| (v >> i) & 1 == 1),
      )?);
    }
    Ok(SmallUInt32 { bits, value })
  }

  /// Right rotation.
  pub fn rotr(&self, by: usize) -> Self {
    let by = by % 32;
    let bits: [SmallBoolean; 32] = std::array::from_fn(|i| self.bits[(i + by) % 32].clone());

    SmallUInt32 {
      bits,
      value: self.value.map(|v| v.rotate_right(by as u32)),
    }
  }

  /// Right shift.
  pub fn shr(&self, by: usize) -> Self {
    let bits: [SmallBoolean; 32] = std::array::from_fn(|i| {
      if i + by < 32 {
        self.bits[i + by].clone()
      } else {
        SmallBoolean::Constant(false)
      }
    });

    SmallUInt32 {
      bits,
      value: self.value.map(|v| v >> by),
    }
  }

  /// XOR with another `SmallUInt32`.
  pub fn xor<W, C, CS>(&self, mut cs: CS, other: &Self) -> Result<Self, SynthesisError>
  where
    W: Copy + From<bool>,
    C: Copy + From<bool> + NegOne + Double,
    CS: SmallConstraintSystem<W, C>,
  {
    let mut bits = [const { SmallBoolean::Constant(false) }; 32];
    for (i, (slot, (a, b))) in bits
      .iter_mut()
      .zip(self.bits.iter().zip(other.bits.iter()))
      .enumerate()
    {
      *slot = SmallBoolean::xor(cs.namespace(|| format!("b{i}")).inner, a, b)?;
    }

    Ok(SmallUInt32 {
      bits,
      value: self.value.and_then(|a| other.value.map(|b| a ^ b)),
    })
  }

  /// SHA-256 CH function: (a AND b) XOR ((NOT a) AND c)
  pub fn sha256_ch<W, C, CS>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    c: &Self,
  ) -> Result<Self, SynthesisError>
  where
    W: Copy + From<bool>,
    C: Copy + From<bool> + NegOne,
    CS: SmallConstraintSystem<W, C>,
  {
    let mut bits = [const { SmallBoolean::Constant(false) }; 32];
    for (i, (slot, ((a_bit, b_bit), c_bit))) in bits
      .iter_mut()
      .zip(a.bits.iter().zip(b.bits.iter()).zip(c.bits.iter()))
      .enumerate()
    {
      *slot = SmallBoolean::sha256_ch(cs.namespace(|| format!("b{i}")).inner, a_bit, b_bit, c_bit)?;
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
  /// Uses the optimized bellpepper-style identity from `SmallBoolean`.
  pub fn sha256_maj<W, C, CS>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    c: &Self,
  ) -> Result<Self, SynthesisError>
  where
    W: Copy + From<bool>,
    C: Copy + From<bool> + NegOne + Double,
    CS: SmallConstraintSystem<W, C>,
  {
    let mut bits = [const { SmallBoolean::Constant(false) }; 32];
    for (i, (slot, ((a_bit, b_bit), c_bit))) in bits
      .iter_mut()
      .zip(a.bits.iter().zip(b.bits.iter()).zip(c.bits.iter()))
      .enumerate()
    {
      *slot =
        SmallBoolean::sha256_maj(cs.namespace(|| format!("b{i}")).inner, a_bit, b_bit, c_bit)?;
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
