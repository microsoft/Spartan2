// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Wide integers and limb operations for delayed modular reduction.
//!
//! # Why We Built WideLimbs Instead of Using Existing Libraries
//!
//! ## The Problem
//!
//! When summing many field×field products `Σ (a_i × b_i)`, the standard approach
//! reduces each product immediately. Montgomery REDC is expensive (~100+ cycles),
//! so N products = N REDCs. Delayed reduction accumulates unreduced 512-bit products
//! in a 576-bit accumulator, reducing only once at the end: N products = 1 REDC.
//!
//! ## Why Not num-bigint?
//!
//! `num-bigint` uses heap-allocated `Vec<u64>` for limbs. In tight cryptographic
//! loops (sumcheck has millions of iterations), heap allocation adds ~30-45%
//! overhead vs stack allocation. We need stack-allocated fixed-size arrays.
//!
//! ## Why Not ff/PrimeField?
//!
//! The `ff` crate's `PrimeField` trait intentionally hides internal representation.
//! Every operation returns a **reduced** 4-limb result:
//!
//! ```text
//! trait Field {
//!     fn mul(self, rhs: Self) -> Self;  // Always returns reduced result
//!     // No: fn mul_wide(self, rhs: Self) -> [u64; 8];
//! }
//! ```
//!
//! There's no "unreduced product" type because it breaks the mathematical abstraction.
//! We need access to raw 8-limb products before reduction.
//!
//! ## Why Not crypto-bigint?
//!
//! `crypto-bigint` provides stack-allocated `Uint<N>`, but:
//!
//! 1. **Type system mismatch**: halo2curves fields are already in Montgomery form
//!    with their own constants. crypto-bigint's `MontyForm` manages its own
//!    Montgomery representation - we can't mix them without conversion overhead.
//!
//! 2. **No custom REDC**: We need `montgomery_reduce_9` that uses halo2curves-specific
//!    constants (MODULUS, MONT_INV, R512_MOD). crypto-bigint's REDC uses its own params.
//!
//! 3. **Overhead for simplicity**: Our `WideLimbs<N>` is ~30 lines. We'd add a full
//!    dependency, type conversions, and still need to write our own REDC.
//!
//! ## What WideLimbs Provides
//!
//! A minimal stack-allocated accumulator with public limbs for direct manipulation:
//!
//! ```text
//! pub struct WideLimbs<const N: usize>(pub [u64; N]);
//!                                      ^^^
//!                                      Direct limb access for carry chains
//! ```
//!
//! Used as `WideLimbs<9>` (576 bits) to accumulate up to 2^68 field×field products
//! before a single Montgomery reduction back to a 4-limb field element.

use num_traits::Zero;
use std::ops::{Add, AddAssign};

/// Stack-allocated wide integer with N 64-bit limbs.
///
/// Limbs are stored in little-endian order: `limbs[0]` is the least significant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideLimbs<const N: usize>(pub [u64; N]);

impl<const N: usize> Default for WideLimbs<N> {
  fn default() -> Self {
    Self([0u64; N])
  }
}

impl<const N: usize> AddAssign for WideLimbs<N> {
  #[inline]
  fn add_assign(&mut self, other: Self) {
    let mut carry = 0u64;
    for i in 0..N {
      let (sum, c1) = self.0[i].overflowing_add(other.0[i]);
      let (sum, c2) = sum.overflowing_add(carry);
      self.0[i] = sum;
      carry = (c1 as u64) + (c2 as u64);
    }
  }
}

impl<const N: usize> Add for WideLimbs<N> {
  type Output = Self;

  #[inline]
  fn add(mut self, other: Self) -> Self {
    self += other;
    self
  }
}

impl<const N: usize> Zero for WideLimbs<N> {
  #[inline]
  fn zero() -> Self {
    Self([0u64; N])
  }

  #[inline]
  fn is_zero(&self) -> bool {
    self.0.iter().all(|&x| x == 0)
  }
}

// =============================================================================
// Generic N-limb operations
// =============================================================================

/// Check if N-limb value a >= N-limb value b.
#[inline(always)]
pub(super) const fn gte<const N: usize>(a: &[u64; N], b: &[u64; N]) -> bool {
  let mut i = N;
  while i > 0 {
    i -= 1;
    if a[i] > b[i] {
      return true;
    }
    if a[i] < b[i] {
      return false;
    }
  }
  true // equal
}

/// Add two N-limb values, returning N limbs + carry (0 or 1).
#[inline(always)]
pub(super) const fn add<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], u64) {
  let mut result = [0u64; N];
  let mut carry = 0u128;
  let mut i = 0;
  while i < N {
    let sum = (a[i] as u128) + (b[i] as u128) + carry;
    result[i] = sum as u64;
    carry = sum >> 64;
    i += 1;
  }
  (result, carry as u64)
}

/// Subtract N-limb b from N-limb a: a - b (assumes a >= b).
#[inline(always)]
pub(super) const fn sub<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
  let mut result = [0u64; N];
  let mut borrow = 0u64;
  let mut i = 0;
  while i < N {
    let (diff, b1) = a[i].overflowing_sub(b[i]);
    let (diff2, b2) = diff.overflowing_sub(borrow);
    result[i] = diff2;
    borrow = (b1 as u64) + (b2 as u64);
    i += 1;
  }
  result
}

/// Shift an N-limb value left by one bit.
#[inline(always)]
pub(super) const fn shl<const N: usize>(a: &[u64; N]) -> [u64; N] {
  let mut result = [0u64; N];
  let mut carry = 0u64;
  let mut i = 0;
  while i < N {
    let new_carry = a[i] >> 63;
    result[i] = (a[i] << 1) | carry;
    carry = new_carry;
    i += 1;
  }
  result
}

/// Count leading zeros in N-limb value.
#[inline(always)]
pub(super) const fn clz<const N: usize>(a: &[u64; N]) -> u32 {
  let mut i = N;
  let mut count = 0u32;
  while i > 0 {
    i -= 1;
    if a[i] != 0 {
      return count + a[i].leading_zeros();
    }
    count += 64;
  }
  count
}

// =============================================================================
// 4-limb (256-bit) operations
// =============================================================================

/// Multiply two 4-limb values, producing an 8-limb result.
#[inline(always)]
pub const fn mul_4_by_4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
  let mut result = [0u64; 8];
  let mut i = 0;
  while i < 4 {
    let mut carry = 0u128;
    let mut j = 0;
    while j < 4 {
      let prod = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + carry;
      result[i + j] = prod as u64;
      carry = prod >> 64;
      j += 1;
    }
    result[i + 4] = carry as u64;
    i += 1;
  }
  result
}

// =============================================================================
// 5-limb (320-bit) operations
// =============================================================================

/// Check if 5-limb value a >= 4-limb value b.
#[inline(always)]
pub(super) const fn gte_5_4(a: &[u64; 5], b: &[u64; 4]) -> bool {
  // If a[4] > 0, then a >= b (since b fits in 256 bits)
  if a[4] > 0 {
    return true;
  }
  // Compare the lower 4 limbs from most significant to least
  let mut i = 4;
  while i > 0 {
    i -= 1;
    if a[i] > b[i] {
      return true;
    }
    if a[i] < b[i] {
      return false;
    }
  }
  true // equal
}

/// Subtract 4-limb from 5-limb: a - b (returns 5 limbs, assumes a >= b).
#[inline(always)]
pub(super) const fn sub_5_4(a: &[u64; 5], b: &[u64; 4]) -> [u64; 5] {
  let mut result = [0u64; 5];
  let mut borrow = 0u64;
  let mut i = 0;
  while i < 4 {
    let (diff, b1) = a[i].overflowing_sub(b[i]);
    let (diff2, b2) = diff.overflowing_sub(borrow);
    result[i] = diff2;
    borrow = (b1 as u64) + (b2 as u64);
    i += 1;
  }
  let (diff, _) = a[4].overflowing_sub(borrow);
  result[4] = diff;
  result
}

// =============================================================================
// 8-limb (512-bit) operations
// =============================================================================

/// Reduce an 8-limb value modulo a 4-limb prime using old-school binary long division.
///
/// Aligns p with x's MSB, then iteratively trial-subtracts at each bit position.
/// Slow (O(n²) in bits) but works in `const fn` context where Montgomery reduction
/// isn't available. Result fits in 4 limbs because `x mod p < p`.
#[inline(always)]
pub(super) const fn reduce_8_mod_4(x: &[u64; 8], p: &[u64; 4]) -> [u64; 4] {
  // Extend p to 8 limbs for comparison
  let p8: [u64; 8] = [p[0], p[1], p[2], p[3], 0, 0, 0, 0];

  // If x < p, we're done
  if !gte::<8>(x, &p8) {
    return [x[0], x[1], x[2], x[3]];
  }

  // Binary long division: shift p left until it's just <= x, then subtract
  let x_clz = clz::<8>(x);
  let p_clz = clz::<8>(&p8);

  // p needs to shift left by (p_clz - x_clz) bits to align with x
  if p_clz <= x_clz {
    // p is already larger than x in leading position, but we know x >= p
    // so just do final subtraction
    let result = sub::<8>(x, &p8);
    return [result[0], result[1], result[2], result[3]];
  }

  let shift_bits = p_clz - x_clz;
  let mut remainder = *x;

  // Shift p left by shift_bits
  let mut shifted_p = p8;
  let mut bits_to_shift = shift_bits;
  while bits_to_shift > 0 {
    shifted_p = shl::<8>(&shifted_p);
    bits_to_shift -= 1;
  }

  // Do shift_bits + 1 iterations of trial subtraction
  let mut iterations = shift_bits + 1;
  while iterations > 0 {
    if gte::<8>(&remainder, &shifted_p) {
      remainder = sub::<8>(&remainder, &shifted_p);
    }
    // Shift p right by 1 (for next iteration)
    if iterations > 1 {
      let mut i = 0;
      while i < 7 {
        shifted_p[i] = (shifted_p[i] >> 1) | (shifted_p[i + 1] << 63);
        i += 1;
      }
      shifted_p[7] >>= 1;
    }
    iterations -= 1;
  }

  [remainder[0], remainder[1], remainder[2], remainder[3]]
}
