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

/// Multiply two 4-limb values, producing an 8-limb result.
#[inline(always)]
pub fn mul_4_by_4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
  let mut result = [0u64; 8];
  for i in 0..4 {
    let mut carry = 0u128;
    for j in 0..4 {
      let prod = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + carry;
      result[i + j] = prod as u64;
      carry = prod >> 64;
    }
    result[i + 4] = carry as u64;
  }
  result
}

/// Subtract 4-limb from 5-limb: a - b (returns 5 limbs).
#[inline(always)]
pub(super) fn sub_5_4(a: &[u64; 5], b: &[u64; 4]) -> [u64; 5] {
  let mut result = [0u64; 5];
  let mut borrow = 0u64;
  for i in 0..4 {
    let (diff, b1) = a[i].overflowing_sub(b[i]);
    let (diff2, b2) = diff.overflowing_sub(borrow);
    result[i] = diff2;
    borrow = (b1 as u64) + (b2 as u64);
  }
  let (diff, _) = a[4].overflowing_sub(borrow);
  result[4] = diff;
  result
}

/// Check if 4-limb value a >= 4-limb value b.
#[inline(always)]
pub(super) fn gte_4_4(a: &[u64; 4], b: &[u64; 4]) -> bool {
  for i in (0..4).rev() {
    if a[i] > b[i] {
      return true;
    }
    if a[i] < b[i] {
      return false;
    }
  }
  true // equal
}

/// Add two 4-limb values, returning 4 limbs + carry (0 or 1).
#[inline(always)]
pub(super) fn add_4_4(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], u64) {
  let mut result = [0u64; 4];
  let mut carry = 0u128;
  for i in 0..4 {
    let sum = (a[i] as u128) + (b[i] as u128) + carry;
    result[i] = sum as u64;
    carry = sum >> 64;
  }
  (result, carry as u64)
}

/// Subtract 4-limb from 4-limb: a - b (assumes a >= b).
#[inline(always)]
pub(super) fn sub_4_4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
  let mut result = [0u64; 4];
  let mut borrow = 0u64;
  for i in 0..4 {
    let (diff, b1) = a[i].overflowing_sub(b[i]);
    let (diff2, b2) = diff.overflowing_sub(borrow);
    result[i] = diff2;
    borrow = (b1 as u64) + (b2 as u64);
  }
  debug_assert!(borrow == 0, "sub_4_4: a < b (borrow out)");
  result
}
