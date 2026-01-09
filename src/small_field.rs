// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value field operations for optimized sumcheck.
//!
//! This module provides types and operations for the small-value optimization
//! described in "Speeding Up Sum-Check Proving". The key insight is distinguishing
//! between three multiplication types:
//!
//! - **ss** (small × small): Native i32/i64 multiplication
//! - **sl** (small × large): Barrett-optimized multiplication (~3× faster)
//! - **ll** (large × large): Standard field multiplication
//!
//! For polynomial evaluations on the boolean hypercube (typically i32 values),
//! we can perform many operations in native integers before converting to field.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    SmallValueField Trait                     │
//! │  (ss_mul, sl_mul, isl_mul, small_to_field, etc.)            │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Barrett Reduction (internal)                    │
//! │  mul_fp_by_i64, mul_fq_by_i64 - ~9 base muls vs ~32 naive   │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::wide_limbs::{SignedWideLimbs, WideLimbs, sub_mag};
use ff::PrimeField;
use std::{
  fmt::Debug,
  marker::PhantomData,
  ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

// ============================================================================
// SmallValueField Trait
// ============================================================================

/// Trait for fields that support small-value optimization.
///
/// This trait defines operations for efficient arithmetic when polynomial
/// evaluations fit in native integers. The key optimization is avoiding
/// expensive field operations until absolutely necessary.
///
/// # Type Parameters
/// - `SmallValue`: Native type for witness values (i32 or i64)
///
/// # Implementations
/// - `Fp: SmallValueField<i32>` with `IntermediateSmallValue = i64`
/// - `Fp: SmallValueField<i64>` with `IntermediateSmallValue = i128`
///
/// # Overflow Bounds (for D=2 Spartan with typical witness values)
///
/// | Step | Bound | Bits | Container |
/// |------|-------|------|-----------|
/// | Original witness values | 2²⁰ | 20 | i32 |
/// | After 3 extensions (D=2) | 2²³ | 23 | i32 ✓ |
/// | Product of two extended | 2⁴⁶ | 46 | i64 ✓ |
pub trait SmallValueField<SmallValue>: PrimeField
where
  SmallValue: Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + Eq
    + Add<Output = SmallValue>
    + Sub<Output = SmallValue>
    + Neg<Output = SmallValue>
    + AddAssign
    + SubAssign
    + Send
    + Sync,
{
  /// Intermediate type for products (i64 for i32 inputs, i128 for i64 inputs).
  /// Used when multiplying two SmallValues together.
  type IntermediateSmallValue: Copy + Clone + Default + Debug + PartialEq + Eq + Send + Sync;

  // ===== Core Multiplications =====

  /// ss: small × small → intermediate (i32 × i32 → i64, or i64 × i64 → i128)
  fn ss_mul(a: SmallValue, b: SmallValue) -> Self::IntermediateSmallValue;

  /// sl: small × large → large (small × field → field)
  fn sl_mul(small: SmallValue, large: &Self) -> Self;

  /// isl: intermediate × large → large (intermediate × field → field)
  /// This is the key operation for accumulator building.
  fn isl_mul(small: Self::IntermediateSmallValue, large: &Self) -> Self;

  // ===== Conversions =====

  /// Convert SmallValue to field element.
  fn small_to_field(val: SmallValue) -> Self;

  /// Convert IntermediateSmallValue to field element.
  fn intermediate_to_field(val: Self::IntermediateSmallValue) -> Self;

  /// Try to convert a field element to SmallValue.
  /// Returns None if the value doesn't fit.
  fn try_field_to_small(val: &Self) -> Option<SmallValue>;
}

/// Extension trait for delayed modular reduction operations.
///
/// This trait extends `SmallValueField` with operations that accumulate
/// unreduced products in wide integers, reducing only at the end.
/// Used in hot paths like matrix-vector multiplication where many products
/// are summed together.
///
/// # Performance
///
/// Delaying reduction saves ~1 field multiplication per accumulation:
/// - Without delayed reduction: N additions + N reductions
/// - With delayed reduction: N additions + 1 reduction
pub trait DelayedReduction<SmallValue>: SmallValueField<SmallValue>
where
  SmallValue: Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + Eq
    + Add<Output = SmallValue>
    + Sub<Output = SmallValue>
    + Neg<Output = SmallValue>
    + AddAssign
    + SubAssign
    + Send
    + Sync,
{
  /// Unreduced accumulator for field × integer products.
  /// - For i32/i64: SignedWideLimbs<6> (384 bits)
  /// - For i64/i128: SignedWideLimbs<8> (512 bits)
  /// Sized to safely sum 2^(l/2) terms without overflow, assuming:
  ///   field_bits + product_bits + (l/2) < 64*N
  /// (N = limb count for this accumulator, 64 bits per limb).
  type UnreducedFieldInt: Copy + Clone + Default + Debug + AddAssign + Send + Sync + num_traits::Zero;

  /// Unreduced accumulator for field × field products (9 limbs, 576 bits).
  /// Used to delay modular reduction when summing many F × F products.
  /// The value is in 2R-scaled Montgomery form, reduced via Montgomery REDC.
  type UnreducedFieldField: Copy + Clone + Default + Debug + AddAssign + Send + Sync;

  /// Multiply field element by signed integer and add to unreduced accumulator.
  /// acc += field × intermediate (keeps result in unreduced form, handles sign internally)
  fn unreduced_field_int_mul_add(
    acc: &mut Self::UnreducedFieldInt,
    field: &Self,
    small: Self::IntermediateSmallValue,
  );

  /// Multiply two field elements and add to unreduced accumulator.
  /// acc += field_a × field_b (keeps result in 2R-scaled unreduced form)
  fn unreduced_field_field_mul_add(acc: &mut Self::UnreducedFieldField, field_a: &Self, field_b: &Self);

  /// Reduce an unreduced field×integer accumulator to a field element.
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self;

  /// Reduce an unreduced field×field accumulator to a field element.
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self;
}

/// Convert i64 to field element (handles negative values correctly).
#[inline]
pub fn i64_to_field<F: PrimeField>(val: i64) -> F {
  if val >= 0 {
    F::from(val as u64)
  } else {
    // Use wrapping_neg to handle i64::MIN correctly
    -F::from(val.wrapping_neg() as u64)
  }
}

/// Convert i128 to field element (handles negative values correctly).
#[inline]
pub fn i128_to_field<F: PrimeField>(val: i128) -> F {
  if val >= 0 {
    // Split into high and low u64 parts
    let low = val as u64;
    let high = (val >> 64) as u64;
    if high == 0 {
      F::from(low)
    } else {
      // result = low + high * 2^64
      F::from(low) + F::from(high) * two_pow_64::<F>()
    }
  } else {
    // Use wrapping_neg to handle i128::MIN correctly
    let pos = val.wrapping_neg() as u128;
    let low = pos as u64;
    let high = (pos >> 64) as u64;
    if high == 0 {
      -F::from(low)
    } else {
      -(F::from(low) + F::from(high) * two_pow_64::<F>())
    }
  }
}

/// Try to convert a field element to i64.
/// Returns None if the value doesn't fit in the i64 range.
#[inline]
pub fn try_field_to_i64<F: PrimeField>(val: &F) -> Option<i64> {
  let repr = val.to_repr();
  let bytes = repr.as_ref();

  // Check if value fits in positive i64 (high bytes all zero)
  let high_zero = bytes[8..].iter().all(|&b| b == 0);
  if high_zero {
    let val_u64 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    if val_u64 <= i64::MAX as u64 {
      return Some(val_u64 as i64);
    }
  }

  // Check if negation fits in i64 (value is negative)
  let neg_val = val.neg();
  let neg_repr = neg_val.to_repr();
  let neg_bytes = neg_repr.as_ref();
  let neg_high_zero = neg_bytes[8..].iter().all(|&b| b == 0);
  if neg_high_zero {
    let neg_u64 = u64::from_le_bytes(neg_bytes[..8].try_into().unwrap());
    if neg_u64 > 0 && neg_u64 <= (i64::MAX as u64) + 1 {
      return Some(-(neg_u64 as i128) as i64);
    }
  }

  None
}

/// Returns 2^64 as a field element (cached via lazy computation).
#[inline]
fn two_pow_64<F: PrimeField>() -> F {
  // 2^64 = (2^32)^2
  let two_32 = F::from(1u64 << 32);
  two_32 * two_32
}

// ============================================================================
// SmallValueConfig - Type-Level Configuration for Small-Value Optimization
// ============================================================================

/// Type-level marker: batching not available for this config.
///
/// Used when constraint coefficients would overflow if batched.
/// Each equality constraint is enforced directly.
pub struct NoBatching;

/// Type-level marker: batching with K constraints per flush.
///
/// Equality constraints are accumulated with coefficients 2^0, 2^1, ..., 2^(K-1)
/// and flushed as a single batched constraint.
pub struct Batching<const K: usize>;

/// Trait to extract batching information at compile time.
pub trait BatchingMode: 'static + Send + Sync {
  /// Maximum bits for batching coefficients.
  /// - `None` = batching not supported, use direct constraints
  /// - `Some(k)` = batch up to k constraints before flushing
  const MAX_COEFF_BITS: Option<usize>;
}

impl BatchingMode for NoBatching {
  const MAX_COEFF_BITS: Option<usize> = None;
}

impl<const K: usize> BatchingMode for Batching<K> {
  const MAX_COEFF_BITS: Option<usize> = Some(K);
}

/// Configuration trait for SmallMultiEq constraint system.
///
/// Bundles a scalar field type implementing `SmallValueField<SmallValue>` with
/// a batching mode. This allows `SmallMultiEq` to use the appropriate small-value
/// operations and batching behavior.
///
/// # Associated Types
/// - `Scalar`: Field type implementing `SmallValueField<SmallValue>`
/// - `SmallValue`: Native type for witness values (i32 or i64)
/// - `Batching`: Batching mode (NoBatching or Batching<K>)
pub trait SmallMultiEqConfig: 'static + Send + Sync {
  /// The scalar field type.
  type Scalar: SmallValueField<Self::SmallValue>;

  /// Small value type for witness coefficients.
  type SmallValue: Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + Eq
    + Add<Output = Self::SmallValue>
    + Sub<Output = Self::SmallValue>
    + Neg<Output = Self::SmallValue>
    + AddAssign
    + SubAssign
    + Send
    + Sync;

  /// Batching mode for constraint accumulation.
  type Batching: BatchingMode;
}

/// i32 configuration with no batching.
///
/// Use this for circuits where Az values would overflow i32 if batched.
/// Each equality constraint is enforced directly.
///
/// - SmallValue: i32
/// - Batching: None (direct constraints)
pub struct I32NoBatch<S>(PhantomData<S>);

impl<S> Default for I32NoBatch<S> {
  fn default() -> Self {
    I32NoBatch(PhantomData)
  }
}

impl<S: SmallValueField<i32>> SmallMultiEqConfig for I32NoBatch<S> {
  type Scalar = S;
  type SmallValue = i32;
  type Batching = NoBatching;
}

/// i64 configuration with batching (21 constraints per flush).
///
/// Use this for circuits with large positional coefficients (up to 2^34),
/// like SHA-256. The larger intermediate type (i128) allows batching while
/// keeping Az values within i64 bounds.
///
/// - SmallValue: i64
/// - Batching: 21 constraints per flush
///
/// MAX_COEFF_BITS = 21 because:
/// - Az ≤ 200 terms × 2^34 (positional) × 2^20 (batching) × 1 (witness) = 2^62 < 2^63
pub struct I64Batch21<S>(PhantomData<S>);

impl<S> Default for I64Batch21<S> {
  fn default() -> Self {
    I64Batch21(PhantomData)
  }
}

impl<S: SmallValueField<i64>> SmallMultiEqConfig for I64Batch21<S> {
  type Scalar = S;
  type SmallValue = i64;
  type Batching = Batching<21>;
}

// ============================================================================
// Barrett Reduction - Internal Implementation
// ============================================================================

/// Barrett reduction for optimized sl (small × large) multiplication.
///
/// Provides ~3× speedup for multiplying a field element by a small integer
/// compared to the naive approach of converting to field then multiplying.
///
/// # Cost Analysis
///
/// | Operation | Base Multiplications |
/// |-----------|---------------------|
/// | Naive (convert + multiply) | ~32 |
/// | Barrett reduction | ~9-13 |
///
/// # How It Works
///
/// Field elements are stored in Montgomery form: `a_mont = a × R mod p`
///
/// When multiplying by a small integer:
/// ```text
/// small × a_mont = small × (a × R) = (small × a) × R mod p
/// ```
///
/// The result is already in Montgomery form! We just need to:
/// 1. Multiply the Montgomery limbs by the small integer (4 base muls)
/// 2. Barrett-reduce the result back to 4 limbs (~5-9 base muls)
mod barrett {
  use halo2curves::pasta::{Fp, Fq};
  use std::ops::Neg;

  // ==========================================================================
  // FieldReductionConstants - Trait for field-specific reduction constants
  // ==========================================================================

  /// Trait providing precomputed constants for efficient modular reduction.
  ///
  /// # Overview
  ///
  /// When reducing a wide integer (more than 4 limbs = 256 bits) modulo a prime p,
  /// we need to handle "overflow limbs" that represent values ≥ 2^256. Each overflow
  /// limb at position i represents the value `limb[i] × 2^(64×i)`.
  ///
  /// # The R Constants
  ///
  /// For each bit position beyond 256 bits, we precompute `2^k mod p`:
  ///
  /// | Constant | Value | Used When |
  /// |----------|-------|-----------|
  /// | `R256_MOD` | 2^256 mod p | Reducing 5th limb (bits 256-319) |
  /// | `R320_MOD` | 2^320 mod p | Reducing 6th limb (bits 320-383) |
  /// | `R384_MOD` | 2^384 mod p | Reducing 7th limb (bits 384-447) |
  /// | `R448_MOD` | 2^448 mod p | Reducing 8th limb (bits 448-511) |
  /// | `R512_MOD` | 2^512 mod p | Reducing 9th limb (bits 512-575) |
  ///
  /// # Example: 6-limb Reduction
  ///
  /// For a 6-limb value `c = [c0, c1, c2, c3, c4, c5]` representing:
  /// ```text
  /// c = c0 + c1·2^64 + c2·2^128 + c3·2^192 + c4·2^256 + c5·2^320
  /// ```
  ///
  /// We reduce by computing:
  /// ```text
  /// c mod p = (c0 + c1·2^64 + c2·2^128 + c3·2^192)
  ///         + c4·(2^256 mod p)
  ///         + c5·(2^320 mod p)
  /// ```
  ///
  /// Since `R256_MOD` and `R320_MOD` are 4-limb values (< 2^256), multiplying
  /// by a single limb produces at most a 5-limb result, which can then be
  /// reduced further if needed.
  ///
  /// # Why This Works
  ///
  /// By the properties of modular arithmetic:
  /// `a ≡ b (mod p) ⟹ c·a ≡ c·b (mod p)`
  ///
  /// So `c5·2^320 ≡ c5·R320_MOD (mod p)`, and the right side is much smaller.
  ///
  /// # Performance
  ///
  /// This approach is ~3× faster than naive division because:
  /// 1. We avoid actual division entirely
  /// 2. Multiplying by precomputed R values is just 4-5 64-bit multiplications
  /// 3. The reduction converges quickly (usually 1-2 iterations)
  pub trait FieldReductionConstants {
    /// The 4-limb prime modulus p (little-endian, 256 bits)
    const MODULUS: [u64; 4];

    /// 2×p as a 5-limb value (for Barrett reduction comparisons)
    const MODULUS_2P: [u64; 5];

    /// Barrett approximation constant μ = floor(2^128 / (p >> 191))
    /// Used to estimate the quotient in Barrett reduction
    const MU: u64;

    /// 2^256 mod p - reduces the 5th limb (index 4) of a wide integer
    const R256_MOD: [u64; 4];

    /// 2^320 mod p - reduces the 6th limb (index 5) of a wide integer
    const R320_MOD: [u64; 4];

    /// 2^384 mod p - reduces the 7th limb (index 6) of a wide integer
    const R384_MOD: [u64; 4];

    /// 2^448 mod p - reduces the 8th limb (index 7) of a wide integer
    const R448_MOD: [u64; 4];

    /// 2^512 mod p - reduces the 9th limb (index 8) of a wide integer
    const R512_MOD: [u64; 4];

    /// Montgomery inverse: -p^(-1) mod 2^64
    /// Used in Montgomery REDC to eliminate low limbs
    const MONT_INV: u64;
  }

  // ==========================================================================
  // FieldReductionConstants implementation for Fp (Pallas base field)
  // ==========================================================================

  impl FieldReductionConstants for Fp {
    // p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
    const MODULUS: [u64; 4] = [
      0x992d30ed00000001,
      0x224698fc094cf91b,
      0x0000000000000000,
      0x4000000000000000,
    ];

    const MODULUS_2P: [u64; 5] = double_limbs(Self::MODULUS);

    const MU: u64 = 0xffffffffffffffff;

    // 2^256 mod p = 0x3fffffffffffffff992c350be41914ad34786d38fffffffd
    const R256_MOD: [u64; 4] = [
      0x34786d38fffffffd,
      0x992c350be41914ad,
      0xffffffffffffffff,
      0x3fffffffffffffff,
    ];

    // 2^320 mod p = 0x3fffffffffffffff76e59c0fdacc1b91bd91d548094cf917992d30ed00000001
    const R320_MOD: [u64; 4] = [
      0x992d30ed00000001,
      0xbd91d548094cf917,
      0x76e59c0fdacc1b91,
      0x3fffffffffffffff,
    ];

    // 2^384 mod p
    const R384_MOD: [u64; 4] = [
      0xcb8792c700000003,
      0x66d3caf41be6eb52,
      0x9b4b3c4bfffffffc,
      0x36e59c0fdacc1b91,
    ];

    // 2^448 mod p
    const R448_MOD: [u64; 4] = [
      0x9b9858f294cf91ba,
      0x8635bd2c4252b065,
      0x496d41af7b9cb714,
      0x1b4b3c4bfffffffc,
    ];

    // 2^512 mod p
    const R512_MOD: [u64; 4] = [
      0x8c78ecb30000000f,
      0xd7d30dbd8b0de0e7,
      0x7797a99bc3c95d18,
      0x096d41af7b9cb714,
    ];

    // -p^(-1) mod 2^64
    const MONT_INV: u64 = 0x992d30ecffffffff;
  }

  // ==========================================================================
  // FieldReductionConstants implementation for Fq (Pallas scalar field)
  // ==========================================================================

  impl FieldReductionConstants for Fq {
    // q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
    const MODULUS: [u64; 4] = [
      0x8c46eb2100000001,
      0x224698fc0994a8dd,
      0x0000000000000000,
      0x4000000000000000,
    ];

    const MODULUS_2P: [u64; 5] = double_limbs(Self::MODULUS);

    const MU: u64 = 0xffffffffffffffff;

    // 2^256 mod q
    const R256_MOD: [u64; 4] = [
      0x5b2b3e9cfffffffd,
      0x992c350be3420567,
      0xffffffffffffffff,
      0x3fffffffffffffff,
    ];

    // 2^320 mod q
    const R320_MOD: [u64; 4] = [
      0x8c46eb2100000001,
      0xf12aec780994a8d9,
      0x76e59c0fd9ad5c89,
      0x3fffffffffffffff,
    ];

    // 2^384 mod q
    const R384_MOD: [u64; 4] = [
      0xa4d4c16300000003,
      0x66d3caf41cbdfa98,
      0xcee4537bfffffffc,
      0x36e59c0fd9ad5c89,
    ];

    // 2^448 mod q
    const R448_MOD: [u64; 4] = [
      0xcc920bb9994a8dd9,
      0x87a7dcbe1ff6e0d7,
      0x496d41af7ccfdaa9,
      0x0ee4537bfffffffc,
    ];

    // 2^512 mod q
    const R512_MOD: [u64; 4] = [
      0xfc9678ff0000000f,
      0x67bb433d891a16e3,
      0x7fae231004ccf590,
      0x096d41af7ccfdaa9,
    ];

    // -q^(-1) mod 2^64
    const MONT_INV: u64 = 0x8c46eb20ffffffff;
  }

  // ==========================================================================
  // Const helper to double a 4-limb value to 5-limb value
  // ==========================================================================

  const fn double_limbs(p: [u64; 4]) -> [u64; 5] {
    let (r0, c0) = p[0].overflowing_add(p[0]);
    let (r1, c1) = {
      let (sum, c1a) = p[1].overflowing_add(p[1]);
      let (sum, c1b) = sum.overflowing_add(c0 as u64);
      (sum, c1a || c1b)
    };
    let (r2, c2) = {
      let (sum, c2a) = p[2].overflowing_add(p[2]);
      let (sum, c2b) = sum.overflowing_add(c1 as u64);
      (sum, c2a || c2b)
    };
    let (r3, c3) = {
      let (sum, c3a) = p[3].overflowing_add(p[3]);
      let (sum, c3b) = sum.overflowing_add(c2 as u64);
      (sum, c3a || c3b)
    };
    let r4 = c3 as u64;
    [r0, r1, r2, r3, r4]
  }

  // ==========================================================================
  // Precomputed 2^64 constants for i128 multiplication
  // ==========================================================================

  // 2^64 as a field element (little-endian limbs: [0, 1, 0, 0])
  // Since Pallas/Vesta primes are ~2^254, 2^64 < p and 2^64 mod p = 2^64
  #[allow(dead_code)] // Used by mul_fp_by_i128, will be wired up for Small64
  const TWO_POW_64_FP: Fp = Fp::from_raw([0, 1, 0, 0]);

  #[allow(dead_code)] // Used by mul_fq_by_i128, will be wired up for Small64
  const TWO_POW_64_FQ: Fq = Fq::from_raw([0, 1, 0, 0]);

  // ==========================================================================
  // Public API - i64 and i128 for SmallValueField/SmallValueConfig
  // ==========================================================================

  /// Multiply Pallas base field element by i64 (signed).
  #[inline]
  pub(super) fn mul_fp_by_i64(large: &Fp, small: i64) -> Fp {
    if small >= 0 {
      mul_fp_by_u64(large, small as u64)
    } else {
      mul_fp_by_u64(large, small.wrapping_neg() as u64).neg()
    }
  }

  /// Multiply Pallas scalar field element by i64 (signed).
  #[inline]
  pub(super) fn mul_fq_by_i64(large: &Fq, small: i64) -> Fq {
    if small >= 0 {
      mul_fq_by_u64(large, small as u64)
    } else {
      mul_fq_by_u64(large, small.wrapping_neg() as u64).neg()
    }
  }

  /// Multiply Pallas base field element by i128 (signed).
  /// Uses i64 Barrett reduction with 2^64 constant for efficiency.
  #[inline]
  #[allow(dead_code)] // Will be used for Small64 sumcheck
  pub(super) fn mul_fp_by_i128(large: &Fp, small: i128) -> Fp {
    if small >= 0 {
      mul_fp_by_u128(large, small as u128)
    } else {
      mul_fp_by_u128(large, small.wrapping_neg() as u128).neg()
    }
  }

  /// Multiply Pallas scalar field element by i128 (signed).
  /// Uses i64 Barrett reduction with 2^64 constant for efficiency.
  #[inline]
  #[allow(dead_code)] // Will be used for Small64 sumcheck
  pub(super) fn mul_fq_by_i128(large: &Fq, small: i128) -> Fq {
    if small >= 0 {
      mul_fq_by_u128(large, small as u128)
    } else {
      mul_fq_by_u128(large, small.wrapping_neg() as u128).neg()
    }
  }

  // ==========================================================================
  // Internal u128 multiplication (reuses u64 Barrett)
  // ==========================================================================

  #[inline]
  #[allow(dead_code)]
  fn mul_fp_by_u128(large: &Fp, small: u128) -> Fp {
    let low = small as u64;
    let high = (small >> 64) as u64;

    if high == 0 {
      mul_fp_by_u64(large, low)
    } else {
      // result = large * low + large * high * 2^64
      let low_part = mul_fp_by_u64(large, low);
      let high_part = mul_fp_by_u64(large, high);
      low_part + high_part * TWO_POW_64_FP
    }
  }

  #[inline]
  #[allow(dead_code)]
  fn mul_fq_by_u128(large: &Fq, small: u128) -> Fq {
    let low = small as u64;
    let high = (small >> 64) as u64;

    if high == 0 {
      mul_fq_by_u64(large, low)
    } else {
      // result = large * low + large * high * 2^64
      let low_part = mul_fq_by_u64(large, low);
      let high_part = mul_fq_by_u64(large, high);
      low_part + high_part * TWO_POW_64_FQ
    }
  }

  // ==========================================================================
  // Internal u64 multiplication
  // ==========================================================================

  #[inline]
  fn mul_fp_by_u64(large: &Fp, small: u64) -> Fp {
    if small == 0 {
      return Fp::zero();
    }
    if small == 1 {
      return *large;
    }
    let c = mul_4_by_1(&large.0, small);
    Fp(barrett_reduce_5_fp(&c))
  }

  #[inline]
  fn mul_fq_by_u64(large: &Fq, small: u64) -> Fq {
    if small == 0 {
      return Fq::zero();
    }
    if small == 1 {
      return *large;
    }
    let c = mul_4_by_1(&large.0, small);
    Fq(barrett_reduce_5_fq(&c))
  }

  // ==========================================================================
  // Core bignum operations
  // ==========================================================================

  #[inline(always)]
  fn mul_4_by_1(a: &[u64; 4], b: u64) -> [u64; 5] {
    let mut result = [0u64; 5];
    let mut carry = 0u128;
    for i in 0..4 {
      let prod = (a[i] as u128) * (b as u128) + carry;
      result[i] = prod as u64;
      carry = prod >> 64;
    }
    result[4] = carry as u64;
    result
  }

  /// Multiply-accumulate: acc + a * b + carry → (low, high)
  ///
  /// Fused operation that computes one limb of a multiply-accumulate in a single step,
  /// avoiding materialization of intermediate arrays.
  #[inline(always)]
  pub(super) fn mac(acc: u64, a: u64, b: u64, carry: u64) -> (u64, u64) {
    let prod = (a as u128) * (b as u128) + (acc as u128) + (carry as u128);
    (prod as u64, (prod >> 64) as u64)
  }

  /// Multiply two 4-limb values, producing an 8-limb result.
  #[inline(always)]
  pub(super) fn mul_4_by_4_ext(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
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

  /// Multiply 4-limb field by 2-limb integer (u128), producing a 6-limb result.
  /// Used for i64/i128 small-value optimization where IntermediateSmallValue is i128.
  #[inline(always)]
  pub(super) fn mul_4_by_2_ext(a: &[u64; 4], b: u128) -> [u64; 6] {
    let b_lo = b as u64;
    let b_hi = (b >> 64) as u64;

    // Multiply a by b_lo (4x1 -> 5 limbs)
    let mut result = [0u64; 6];
    let mut carry = 0u128;
    for i in 0..4 {
      let prod = (a[i] as u128) * (b_lo as u128) + carry;
      result[i] = prod as u64;
      carry = prod >> 64;
    }
    result[4] = carry as u64;

    // Multiply a by b_hi and add at offset 1 (4x1 -> 5 limbs, shifted)
    carry = 0u128;
    for i in 0..4 {
      let prod = (a[i] as u128) * (b_hi as u128) + (result[i + 1] as u128) + carry;
      result[i + 1] = prod as u64;
      carry = prod >> 64;
    }
    result[5] = carry as u64;

    result
  }

  /// Generic 5-limb Barrett reduction using trait constants.
  /// Reduces a 5-limb value (up to 320 bits) modulo p.
  #[inline(always)]
  fn barrett_reduce_5<F: FieldReductionConstants>(c: &[u64; 5]) -> [u64; 4] {
    let c_tilde = (c[3] >> 63) | (c[4] << 1);
    let m = {
      let product = (c_tilde as u128) * (F::MU as u128);
      (product >> 64) as u64
    };
    let m_times_2p = mul_5_by_1(&F::MODULUS_2P, m);
    let mut r = sub_5_5(c, &m_times_2p);
    // At most 2 iterations needed: after Barrett approximation, 0 ≤ r < 2p
    while gte_5_4(&r, &F::MODULUS) {
      r = sub_5_4(&r, &F::MODULUS);
    }
    [r[0], r[1], r[2], r[3]]
  }

  #[inline(always)]
  fn barrett_reduce_5_fp(c: &[u64; 5]) -> [u64; 4] {
    barrett_reduce_5::<Fp>(c)
  }

  #[inline(always)]
  fn barrett_reduce_5_fq(c: &[u64; 5]) -> [u64; 4] {
    barrett_reduce_5::<Fq>(c)
  }

  #[inline(always)]
  fn mul_5_by_1(a: &[u64; 5], b: u64) -> [u64; 5] {
    let mut result = [0u64; 5];
    let mut carry = 0u128;
    for i in 0..5 {
      let prod = (a[i] as u128) * (b as u128) + carry;
      result[i] = prod as u64;
      carry = prod >> 64;
    }
    result
  }

  #[inline(always)]
  fn sub_5_5(a: &[u64; 5], b: &[u64; 5]) -> [u64; 5] {
    let mut result = [0u64; 5];
    let mut borrow = 0u64;
    for i in 0..5 {
      let (diff, b1) = a[i].overflowing_sub(b[i]);
      let (diff2, b2) = diff.overflowing_sub(borrow);
      result[i] = diff2;
      borrow = (b1 as u64) + (b2 as u64);
    }
    result
  }

  #[inline(always)]
  fn sub_5_4(a: &[u64; 5], b: &[u64; 4]) -> [u64; 5] {
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

  #[inline(always)]
  fn gte_5_4(a: &[u64; 5], b: &[u64; 4]) -> bool {
    if a[4] > 0 {
      return true;
    }
    for i in (0..4).rev() {
      if a[i] > b[i] {
        return true;
      }
      if a[i] < b[i] {
        return false;
      }
    }
    true
  }

  // ==========================================================================
  // 6-limb Barrett reduction (for UnreducedFieldInt accumulator)
  // ==========================================================================

  /// Generic 6-limb Barrett reduction using trait constants.
  ///
  /// Reduces a 6-limb value (up to 384 bits) modulo p using precomputed
  /// R256 = 2^256 mod p and R320 = 2^320 mod p constants.
  ///
  /// Input is already in Montgomery form (R-scaled). This function reduces
  /// the 6-limb value mod p while preserving the Montgomery scaling.
  #[inline]
  fn barrett_reduce_6<F: FieldReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
    // Reduce c[5] * 2^320 ≡ c[5] * R320 (mod p), then reduce c[4] * 2^256, etc.
    // R320 = 2^320 mod p, R256 = 2^256 mod p
    //
    // We do: result = c[0..4] + c[4] * R256 + c[5] * R320 (mod p)

    // c[4] * R256_MOD (4x1 -> 5 limbs)
    let c4_contrib = mul_4_by_1(&F::R256_MOD, c[4]);
    // c[5] * R320_MOD (4x1 -> 5 limbs)
    let c5_contrib = mul_4_by_1(&F::R320_MOD, c[5]);

    // Sum: c[0..4] + c4_contrib + c5_contrib (could be up to 6 limbs)
    let mut sum = [0u64; 6];
    let mut carry = 0u128;
    for i in 0..4 {
      let s = (c[i] as u128) + (c4_contrib[i] as u128) + (c5_contrib[i] as u128) + carry;
      sum[i] = s as u64;
      carry = s >> 64;
    }
    // Limb 4: c4_contrib[4] + c5_contrib[4] + carry
    let s = (c4_contrib[4] as u128) + (c5_contrib[4] as u128) + carry;
    sum[4] = s as u64;
    sum[5] = (s >> 64) as u64;

    // Now reduce the 6-limb sum. If sum[5] or sum[4] is non-zero, recurse (limited depth)
    if sum[5] == 0 && sum[4] == 0 {
      // Result fits in 4 limbs, just do final reduction
      let mut r = [sum[0], sum[1], sum[2], sum[3]];
      while gte_4_4(&r, &F::MODULUS) {
        r = sub_4_4(&r, &F::MODULUS);
      }
      return r;
    }

    // Recurse (this will terminate because sum < c in most cases)
    barrett_reduce_6::<F>(&sum)
  }

  #[inline(always)]
  fn sub_4_4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
      let (diff, b1) = a[i].overflowing_sub(b[i]);
      let (diff2, b2) = diff.overflowing_sub(borrow);
      result[i] = diff2;
      borrow = (b1 as u64) + (b2 as u64);
    }
    result
  }

  #[inline]
  #[allow(dead_code)]
  pub(crate) fn barrett_reduce_6_fp(c: &[u64; 6]) -> [u64; 4] {
    barrett_reduce_6::<Fp>(c)
  }

  #[inline]
  #[allow(dead_code)]
  pub(crate) fn barrett_reduce_6_fq(c: &[u64; 6]) -> [u64; 4] {
    barrett_reduce_6::<Fq>(c)
  }

  // ==========================================================================
  // 8-limb Barrett reduction (for i64/i128 UnreducedFieldInt accumulator)
  // ==========================================================================

  /// Generic 8-limb Barrett reduction using trait constants.
  ///
  /// Reduces an 8-limb value (up to 512 bits) modulo p using precomputed
  /// R384 = 2^384 mod p and R448 = 2^448 mod p constants, then delegates
  /// to 6-limb reduction.
  #[inline]
  fn barrett_reduce_8<F: FieldReductionConstants>(c: &[u64; 8]) -> [u64; 4] {
    // Reduce high limbs: c[6] * 2^384 + c[7] * 2^448
    let c6_contrib = mul_4_by_1(&F::R384_MOD, c[6]);
    let c7_contrib = mul_4_by_1(&F::R448_MOD, c[7]);

    // Sum: c[0..6] + c6_contrib + c7_contrib
    let mut sum = [0u64; 6];
    let mut carry = 0u128;
    for i in 0..4 {
      let s = (c[i] as u128) + (c6_contrib[i] as u128) + (c7_contrib[i] as u128) + carry;
      sum[i] = s as u64;
      carry = s >> 64;
    }
    // Limbs 4-5: add c[4], c[5], and carry from high contributions
    let s = (c[4] as u128) + (c6_contrib[4] as u128) + (c7_contrib[4] as u128) + carry;
    sum[4] = s as u64;
    carry = s >> 64;
    let s = (c[5] as u128) + carry;
    sum[5] = s as u64;

    // Now reduce the 6-limb result
    barrett_reduce_6::<F>(&sum)
  }

  #[inline]
  pub(crate) fn barrett_reduce_8_fp(c: &[u64; 8]) -> [u64; 4] {
    barrett_reduce_8::<Fp>(c)
  }

  #[inline]
  pub(crate) fn barrett_reduce_8_fq(c: &[u64; 8]) -> [u64; 4] {
    barrett_reduce_8::<Fq>(c)
  }

  // ==========================================================================
  // 9-limb Montgomery REDC (for UnreducedFieldField accumulator)
  // ==========================================================================

  /// Generic Montgomery REDC for 9-limb input using trait constants.
  ///
  /// Reduces a 2R-scaled value (sum of field×field products) to 1R-scaled.
  /// Input: T representing x·R² (up to 9 limbs)
  /// Output: x·R mod p (4 limbs, standard Montgomery form)
  #[inline]
  fn montgomery_reduce_9<F: FieldReductionConstants>(c: &[u64; 9]) -> [u64; 4] {
    // Step 1: Reduce 9 limbs to 8 limbs using precomputed 2^512 mod p
    let mut t = [0u64; 9];
    if c[8] == 0 {
      t[..8].copy_from_slice(&c[..8]);
    } else {
      // t = c[0..8] + c[8] * R512_MOD
      let high_contribution = mul_4_by_1(&F::R512_MOD, c[8]);
      let mut carry = 0u128;
      for i in 0..5 {
        let sum = (c[i] as u128) + (high_contribution[i] as u128) + carry;
        t[i] = sum as u64;
        carry = sum >> 64;
      }
      for i in 5..8 {
        let sum = (c[i] as u128) + carry;
        t[i] = sum as u64;
        carry = sum >> 64;
      }
      t[8] = carry as u64;

      // Recurse if still > 8 limbs
      if t[8] > 0 {
        return montgomery_reduce_9::<F>(&t);
      }
    }

    // Step 2: Montgomery REDC on 8-limb value
    montgomery_reduce_8::<F>(&[t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]])
  }

  /// Generic Montgomery REDC for 8-limb input.
  /// Standard Montgomery reduction: T × R⁻¹ mod p
  #[inline]
  fn montgomery_reduce_8<F: FieldReductionConstants>(t: &[u64; 8]) -> [u64; 4] {
    // Use 9 limbs to track overflow
    let mut r = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], 0u64];

    // Montgomery reduction: for each of the low 4 limbs, eliminate it
    // by adding appropriate multiples of p
    for i in 0..4 {
      // q = r[i] * p' mod 2^64
      let q = r[i].wrapping_mul(F::MONT_INV);

      // r += q * p * 2^(64*i)
      // qp = q * p, which is 5 limbs (since p is 4 limbs and q is 1 limb)
      let qp = mul_4_by_1(&F::MODULUS, q);

      let mut carry = 0u128;
      for j in 0..5 {
        let sum = (r[i + j] as u128) + (qp[j] as u128) + carry;
        r[i + j] = sum as u64;
        carry = sum >> 64;
      }
      // Propagate remaining carry through the rest of the array
      for item in r[(i + 5)..9].iter_mut() {
        let sum = (*item as u128) + carry;
        *item = sum as u64;
        carry = sum >> 64;
        if carry == 0 {
          break;
        }
      }
    }

    // Now r[0..4] should be zero (by construction), result is in r[4..9]
    // We need to reduce this to [0, p)
    let mut result = [r[4], r[5], r[6], r[7], r[8]];

    // Reduce until result < p
    while result[4] > 0 || gte_4_4(&[result[0], result[1], result[2], result[3]], &F::MODULUS) {
      let sub = sub_5_4(&result, &F::MODULUS);
      result = sub;
    }

    [result[0], result[1], result[2], result[3]]
  }

  #[inline]
  #[allow(dead_code)]
  pub(crate) fn montgomery_reduce_9_fp(c: &[u64; 9]) -> [u64; 4] {
    montgomery_reduce_9::<Fp>(c)
  }

  #[inline]
  #[allow(dead_code)]
  pub(crate) fn montgomery_reduce_9_fq(c: &[u64; 9]) -> [u64; 4] {
    montgomery_reduce_9::<Fq>(c)
  }

  // Helper functions for 4-limb operations
  #[inline(always)]
  fn gte_4_4(a: &[u64; 4], b: &[u64; 4]) -> bool {
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

  // ==========================================================================
  // Tests
  // ==========================================================================

  #[cfg(test)]
  mod tests {
    use super::*;
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    #[test]
    fn test_barrett_fp_matches_naive() {
      for small in [0u64, 1, 2, 42, 1000, u32::MAX as u64, u64::MAX] {
        let large = Fp::random(&mut OsRng);
        let naive = Fp::from(small) * large;
        let barrett = mul_fp_by_u64(&large, small);
        assert_eq!(naive, barrett, "Fp mismatch for small = {}", small);
      }
    }

    #[test]
    fn test_barrett_fp_random() {
      let mut rng = OsRng;
      for _ in 0..1000 {
        let large = Fp::random(&mut rng);
        let small: u64 = rng.next_u64();
        assert_eq!(Fp::from(small) * large, mul_fp_by_u64(&large, small));
      }
    }

    #[test]
    fn test_barrett_fp_i64() {
      let large = Fp::from(42u64);
      assert_eq!(mul_fp_by_i64(&large, 100), Fp::from(100u64) * large);
      assert_eq!(mul_fp_by_i64(&large, -100), -Fp::from(100u64) * large);
    }

    #[test]
    fn test_barrett_fq_matches_naive() {
      for small in [0u64, 1, 2, 42, 1000, u32::MAX as u64, u64::MAX] {
        let large = Fq::random(&mut OsRng);
        let naive = Fq::from(small) * large;
        let barrett = mul_fq_by_u64(&large, small);
        assert_eq!(naive, barrett, "Fq mismatch for small = {}", small);
      }
    }

    #[test]
    fn test_barrett_fq_random() {
      let mut rng = OsRng;
      for _ in 0..1000 {
        let large = Fq::random(&mut rng);
        let small: u64 = rng.next_u64();
        assert_eq!(Fq::from(small) * large, mul_fq_by_u64(&large, small));
      }
    }

    #[test]
    fn test_barrett_fq_i64() {
      let large = Fq::from(42u64);
      assert_eq!(mul_fq_by_i64(&large, 100), Fq::from(100u64) * large);
      assert_eq!(mul_fq_by_i64(&large, -100), -Fq::from(100u64) * large);
    }

    #[test]
    fn test_barrett_fp_i128() {
      let large = Fp::random(&mut OsRng);

      // Test small i128 values (fits in i64)
      assert_eq!(mul_fp_by_i128(&large, 100), Fp::from(100u64) * large);
      assert_eq!(mul_fp_by_i128(&large, -100), -Fp::from(100u64) * large);

      // Test large i128 values (requires 2^64 decomposition)
      let big: i128 = (1i128 << 70) + 12345;
      let expected = crate::small_field::i128_to_field::<Fp>(big) * large;
      assert_eq!(mul_fp_by_i128(&large, big), expected);

      let neg_big: i128 = -((1i128 << 70) + 12345);
      let expected_neg = crate::small_field::i128_to_field::<Fp>(neg_big) * large;
      assert_eq!(mul_fp_by_i128(&large, neg_big), expected_neg);
    }

    #[test]
    fn test_barrett_fq_i128() {
      let large = Fq::random(&mut OsRng);

      // Test small i128 values (fits in i64)
      assert_eq!(mul_fq_by_i128(&large, 100), Fq::from(100u64) * large);
      assert_eq!(mul_fq_by_i128(&large, -100), -Fq::from(100u64) * large);

      // Test large i128 values (requires 2^64 decomposition)
      let big: i128 = (1i128 << 70) + 12345;
      let expected = crate::small_field::i128_to_field::<Fq>(big) * large;
      assert_eq!(mul_fq_by_i128(&large, big), expected);

      let neg_big: i128 = -((1i128 << 70) + 12345);
      let expected_neg = crate::small_field::i128_to_field::<Fq>(neg_big) * large;
      assert_eq!(mul_fq_by_i128(&large, neg_big), expected_neg);
    }

    #[test]
    fn test_barrett_i128_random() {
      let mut rng = OsRng;
      for _ in 0..100 {
        let large = Fq::random(&mut rng);
        // Generate random i128 in reasonable range
        let small: i128 = ((rng.next_u64() as i128) << 32) | (rng.next_u64() as i128);
        let small = if rng.next_u32().is_multiple_of(2) {
          small
        } else {
          -small
        };

        let result = mul_fq_by_i128(&large, small);
        let expected = crate::small_field::i128_to_field::<Fq>(small) * large;
        assert_eq!(result, expected);
      }
    }

    #[test]
    fn test_two_pow_64_constants() {
      // Verify TWO_POW_64_FP is correct
      let computed = Fp::from(1u64 << 32) * Fp::from(1u64 << 32);
      assert_eq!(TWO_POW_64_FP, computed);

      // Verify TWO_POW_64_FQ is correct
      let computed = Fq::from(1u64 << 32) * Fq::from(1u64 << 32);
      assert_eq!(TWO_POW_64_FQ, computed);
    }

    #[test]
    fn test_constants_match_halo2curves() {
      let p_minus_one = -Fp::ONE;
      let expected = Fp::from_raw([
        Fp::MODULUS[0].wrapping_sub(1),
        Fp::MODULUS[1],
        Fp::MODULUS[2],
        Fp::MODULUS[3],
      ]);
      assert_eq!(p_minus_one, expected);

      let q_minus_one = -Fq::ONE;
      let expected = Fq::from_raw([
        Fq::MODULUS[0].wrapping_sub(1),
        Fq::MODULUS[1],
        Fq::MODULUS[2],
        Fq::MODULUS[3],
      ]);
      assert_eq!(q_minus_one, expected);
    }

    // ========================================================================
    // 6-limb Barrett reduction tests
    // ========================================================================

    #[test]
    fn test_barrett_6_fp_zero() {
      let c = [0u64; 6];
      let result = Fp(barrett_reduce_6_fp(&c));
      assert_eq!(result, Fp::ZERO);
    }

    #[test]
    fn test_barrett_6_fp_from_product() {
      // Test with an actual field × integer product (the real use case)
      let field_elem = Fp::from(12345u64); // Creates Montgomery form
      let small = 9999u64;

      // Compute field × small as 5 limbs (Montgomery form)
      let product = mul_4_by_1(&field_elem.0, small);

      // Extend to 6 limbs
      let c = [
        product[0], product[1], product[2], product[3], product[4], 0,
      ];

      // Reduce
      let result = Fp(barrett_reduce_6_fp(&c));

      // Expected: field_elem * small
      let expected = field_elem * Fp::from(small);
      assert_eq!(result, expected);
    }

    #[test]
    fn test_barrett_6_fp_sum_of_products() {
      // Test summing multiple products (the accumulator use case)
      let mut rng = OsRng;
      let mut acc = [0u64; 6];
      let mut expected_sum = Fp::ZERO;

      // Sum 100 products
      for _ in 0..100 {
        let field_elem = Fp::random(&mut rng);
        let small = rng.next_u64() >> 32; // Keep small to avoid overflow in test

        // Accumulate expected result
        expected_sum += field_elem * Fp::from(small);

        // Compute field × small as 5 limbs
        let product = mul_4_by_1(&field_elem.0, small);

        // Add to accumulator with carry propagation
        let mut carry = 0u128;
        for i in 0..5 {
          let sum = (acc[i] as u128) + (product[i] as u128) + carry;
          acc[i] = sum as u64;
          carry = sum >> 64;
        }
        acc[5] = acc[5].wrapping_add(carry as u64);
      }

      // Reduce and compare
      let result = Fp(barrett_reduce_6_fp(&acc));
      assert_eq!(result, expected_sum);
    }

    #[test]
    fn test_barrett_6_fp_many_products() {
      // Stress test: sum many products to exercise 6-limb reduction
      // Note: In the real use case, we sum at most 2^(ℓ/2) products where ℓ ≤ 130.
      // For ℓ = 20 (a typical size), that's 2^10 = 1024 products.
      let mut rng = OsRng;
      let mut acc = [0u64; 6];
      let mut expected_sum = Fp::ZERO;

      // Sum 2000 products (realistic bound for medium-sized polynomials)
      for _ in 0..2000 {
        let field_elem = Fp::random(&mut rng);
        let small = rng.next_u64();

        expected_sum += field_elem * Fp::from(small);

        let product = mul_4_by_1(&field_elem.0, small);
        let mut carry = 0u128;
        for i in 0..5 {
          let sum = (acc[i] as u128) + (product[i] as u128) + carry;
          acc[i] = sum as u64;
          carry = sum >> 64;
        }
        acc[5] = acc[5].wrapping_add(carry as u64);
      }

      let result = Fp(barrett_reduce_6_fp(&acc));
      assert_eq!(result, expected_sum);
    }

    #[test]
    fn test_barrett_6_fq_sum_of_products() {
      let mut rng = OsRng;
      let mut acc = [0u64; 6];
      let mut expected_sum = Fq::ZERO;

      for _ in 0..100 {
        let field_elem = Fq::random(&mut rng);
        let small = rng.next_u64() >> 32;

        expected_sum += field_elem * Fq::from(small);

        let product = mul_4_by_1(&field_elem.0, small);
        let mut carry = 0u128;
        for i in 0..5 {
          let sum = (acc[i] as u128) + (product[i] as u128) + carry;
          acc[i] = sum as u64;
          carry = sum >> 64;
        }
        acc[5] = acc[5].wrapping_add(carry as u64);
      }

      let result = Fq(barrett_reduce_6_fq(&acc));
      assert_eq!(result, expected_sum);
    }

    // ========================================================================
    // 9-limb Montgomery REDC tests
    // ========================================================================

    /// Helper to multiply two 4-limb values, producing an 8-limb result
    fn mul_4_by_4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
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

    #[test]
    fn test_montgomery_9_fp_single_product() {
      // Test with a single field × field product
      let a = Fp::from(12345u64);
      let b = Fp::from(67890u64);

      // Compute a_mont × b_mont (8 limbs, representing (a*b)*R² in unreduced form)
      let product = mul_4_by_4(&a.0, &b.0);

      // Extend to 9 limbs
      let c = [
        product[0], product[1], product[2], product[3], product[4], product[5], product[6],
        product[7], 0,
      ];

      // Montgomery reduce: should give (a*b)*R mod p = (a*b) in Montgomery form
      let result = Fp(montgomery_reduce_9_fp(&c));

      // Expected: a * b in field
      let expected = a * b;
      assert_eq!(result, expected);
    }

    #[test]
    fn test_montgomery_9_fp_sum_of_products() {
      // Test summing multiple field × field products
      let mut rng = OsRng;
      let mut acc = [0u64; 9];
      let mut expected_sum = Fp::ZERO;

      // Sum 100 products
      for _ in 0..100 {
        let a = Fp::random(&mut rng);
        let b = Fp::random(&mut rng);

        // Accumulate expected result
        expected_sum += a * b;

        // Compute a_mont × b_mont as 8 limbs
        let product = mul_4_by_4(&a.0, &b.0);

        // Add to accumulator with carry propagation
        let mut carry = 0u128;
        for i in 0..8 {
          let sum = (acc[i] as u128) + (product[i] as u128) + carry;
          acc[i] = sum as u64;
          carry = sum >> 64;
        }
        acc[8] = acc[8].wrapping_add(carry as u64);
      }

      // Montgomery reduce and compare
      let result = Fp(montgomery_reduce_9_fp(&acc));
      assert_eq!(result, expected_sum);
    }

    #[test]
    fn test_montgomery_9_fp_many_products() {
      // Stress test with many products
      let mut rng = OsRng;
      let mut acc = [0u64; 9];
      let mut expected_sum = Fp::ZERO;

      // Sum 1000 products
      for _ in 0..1000 {
        let a = Fp::random(&mut rng);
        let b = Fp::random(&mut rng);

        expected_sum += a * b;

        let product = mul_4_by_4(&a.0, &b.0);
        let mut carry = 0u128;
        for i in 0..8 {
          let sum = (acc[i] as u128) + (product[i] as u128) + carry;
          acc[i] = sum as u64;
          carry = sum >> 64;
        }
        acc[8] = acc[8].wrapping_add(carry as u64);
      }

      let result = Fp(montgomery_reduce_9_fp(&acc));
      assert_eq!(result, expected_sum);
    }

    #[test]
    fn test_montgomery_9_fq_sum_of_products() {
      let mut rng = OsRng;
      let mut acc = [0u64; 9];
      let mut expected_sum = Fq::ZERO;

      for _ in 0..100 {
        let a = Fq::random(&mut rng);
        let b = Fq::random(&mut rng);

        expected_sum += a * b;

        let product = mul_4_by_4(&a.0, &b.0);
        let mut carry = 0u128;
        for i in 0..8 {
          let sum = (acc[i] as u128) + (product[i] as u128) + carry;
          acc[i] = sum as u64;
          carry = sum >> 64;
        }
        acc[8] = acc[8].wrapping_add(carry as u64);
      }

      let result = Fq(montgomery_reduce_9_fq(&acc));
      assert_eq!(result, expected_sum);
    }
  }
}

// ============================================================================
// SmallValueField Implementations
// ============================================================================

/// Helper for try_field_to_small: attempts to convert a field element to i32.
///
/// Returns Some(v) if the field element represents a small integer in [-2^31, 2^31-1].
fn try_field_to_small_impl<F: PrimeField>(val: &F) -> Option<i32> {
  let repr = val.to_repr();
  let bytes = repr.as_ref();

  // Check if value fits in positive i32
  let high_zero = bytes[4..].iter().all(|&b| b == 0);
  if high_zero {
    let val_u32 = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    if val_u32 <= i32::MAX as u32 {
      return Some(val_u32 as i32);
    }
  }

  // Check if negation fits in i32 (value is negative)
  let neg_val = val.neg();
  let neg_repr = neg_val.to_repr();
  let neg_bytes = neg_repr.as_ref();
  let neg_high_zero = neg_bytes[4..].iter().all(|&b| b == 0);
  if neg_high_zero {
    let neg_u32 = u32::from_le_bytes(neg_bytes[..4].try_into().unwrap());
    if neg_u32 > 0 && neg_u32 <= (i32::MAX as u32) + 1 {
      return Some(-(neg_u32 as i64) as i32);
    }
  }

  None
}

impl SmallValueField<i32> for halo2curves::pasta::Fp {
  type IntermediateSmallValue = i64;

  #[inline]
  fn ss_mul(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
  }

  #[inline]
  fn sl_mul(small: i32, large: &Self) -> Self {
    barrett::mul_fp_by_i64(large, small as i64)
  }

  #[inline]
  fn isl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_fp_by_i64(large, small)
  }

  #[inline]
  fn small_to_field(val: i32) -> Self {
    if val >= 0 {
      Self::from(val as u64)
    } else {
      -Self::from((-val) as u64)
    }
  }

  #[inline]
  fn intermediate_to_field(val: i64) -> Self {
    i64_to_field(val)
  }

  fn try_field_to_small(val: &Self) -> Option<i32> {
    try_field_to_small_impl(val)
  }
}

impl DelayedReduction<i32> for halo2curves::pasta::Fp {
  type UnreducedFieldInt = SignedWideLimbs<6>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_int_mul_add(acc: &mut Self::UnreducedFieldInt, field: &Self, small: i64) {
    // Handle sign: accumulate into pos or neg based on sign of small
    let (target, mag) = if small >= 0 {
      (&mut acc.pos, small as u64)
    } else {
      (&mut acc.neg, (-small) as u64)
    };
    // Fused multiply-accumulate: no intermediate array
    let a = &field.0;
    let (r0, c) = barrett::mac(target.0[0], a[0], mag, 0);
    let (r1, c) = barrett::mac(target.0[1], a[1], mag, c);
    let (r2, c) = barrett::mac(target.0[2], a[2], mag, c);
    let (r3, c) = barrett::mac(target.0[3], a[3], mag, c);
    // Propagate carry without multiply (just add)
    let (r4, of) = target.0[4].overflowing_add(c);
    target.0[0] = r0;
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = target.0[5].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  ) {
    // Compute field_a × field_b as 8 limbs and add to accumulator
    let product = barrett::mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    let (neg, mag) = sub_mag::<6>(&acc.pos.0, &acc.neg.0);
    let r = Self(barrett::barrett_reduce_6_fp(&mag));
    if neg { -r } else { r }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self(barrett::montgomery_reduce_9_fp(&acc.0))
  }
}

impl SmallValueField<i64> for halo2curves::pasta::Fp {
  type IntermediateSmallValue = i128;

  #[inline]
  fn ss_mul(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }

  #[inline]
  fn sl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_fp_by_i64(large, small)
  }

  #[inline]
  fn isl_mul(small: i128, large: &Self) -> Self {
    if small == 0 {
      return Self::zero();
    }
    let (is_neg, mag) = if small >= 0 {
      (false, small as u128)
    } else {
      (true, (-small) as u128)
    };
    // mul_4_by_2_ext produces 6 limbs, use barrett_reduce_6 directly (no padding)
    let product = barrett::mul_4_by_2_ext(&large.0, mag);
    let result = Self(barrett::barrett_reduce_6_fp(&product));
    if is_neg { -result } else { result }
  }

  #[inline]
  fn small_to_field(val: i64) -> Self {
    i64_to_field(val)
  }

  #[inline]
  fn intermediate_to_field(val: i128) -> Self {
    i128_to_field(val)
  }

  fn try_field_to_small(val: &Self) -> Option<i64> {
    let repr = val.to_repr();
    let bytes = repr.as_ref();

    // Check if value fits in positive i64
    let high_zero = bytes[8..].iter().all(|&b| b == 0);
    if high_zero {
      let val_u64 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
      if val_u64 <= i64::MAX as u64 {
        return Some(val_u64 as i64);
      }
    }

    // Check if negation fits in i64
    let neg_val = val.neg();
    let neg_repr = neg_val.to_repr();
    let neg_bytes = neg_repr.as_ref();
    let neg_high_zero = neg_bytes[8..].iter().all(|&b| b == 0);
    if neg_high_zero {
      let neg_u64 = u64::from_le_bytes(neg_bytes[..8].try_into().unwrap());
      if neg_u64 > 0 && neg_u64 <= (i64::MAX as u64) + 1 {
        return Some(-(neg_u64 as i128) as i64);
      }
    }

    None
  }
}

impl DelayedReduction<i64> for halo2curves::pasta::Fp {
  type UnreducedFieldInt = SignedWideLimbs<8>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_int_mul_add(acc: &mut Self::UnreducedFieldInt, field: &Self, small: i128) {
    let (target, mag) = if small >= 0 {
      (&mut acc.pos, small as u128)
    } else {
      (&mut acc.neg, (-small) as u128)
    };
    // Fused 4×2 multiply-accumulate: two passes at different offsets
    let a = &field.0;
    let b_lo = mag as u64;
    let b_hi = (mag >> 64) as u64;

    // Pass 1: multiply by b_lo at offset 0
    let (r0, c) = barrett::mac(target.0[0], a[0], b_lo, 0);
    let (r1, c) = barrett::mac(target.0[1], a[1], b_lo, c);
    let (r2, c) = barrett::mac(target.0[2], a[2], b_lo, c);
    let (r3, c) = barrett::mac(target.0[3], a[3], b_lo, c);
    // Propagate carry without multiply (just add)
    let (r4, of1) = target.0[4].overflowing_add(c);
    let c1 = of1 as u64;
    target.0[0] = r0;

    // Pass 2: multiply by b_hi at offset 1 (add to r1..r5)
    let (r1, c) = barrett::mac(r1, a[0], b_hi, 0);
    let (r2, c) = barrett::mac(r2, a[1], b_hi, c);
    let (r3, c) = barrett::mac(r3, a[2], b_hi, c);
    let (r4, c) = barrett::mac(r4, a[3], b_hi, c);
    // Add both carries (c from pass 2, c1 from pass 1) into position 5
    let (r5, c) = barrett::mac(target.0[5], c1, 1, c);
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = r5;
    // Propagate final carry through remaining limbs (just add)
    let (r6, of) = target.0[6].overflowing_add(c);
    target.0[6] = r6;
    target.0[7] = target.0[7].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  ) {
    let product = barrett::mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    let (neg, mag) = sub_mag::<8>(&acc.pos.0, &acc.neg.0);
    let r = Self(barrett::barrett_reduce_8_fp(&mag));
    if neg { -r } else { r }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self(barrett::montgomery_reduce_9_fp(&acc.0))
  }
}

impl SmallValueField<i32> for halo2curves::pasta::Fq {
  type IntermediateSmallValue = i64;

  #[inline]
  fn ss_mul(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
  }

  #[inline]
  fn sl_mul(small: i32, large: &Self) -> Self {
    barrett::mul_fq_by_i64(large, small as i64)
  }

  #[inline]
  fn isl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_fq_by_i64(large, small)
  }

  #[inline]
  fn small_to_field(val: i32) -> Self {
    if val >= 0 {
      Self::from(val as u64)
    } else {
      -Self::from((-val) as u64)
    }
  }

  #[inline]
  fn intermediate_to_field(val: i64) -> Self {
    i64_to_field(val)
  }

  fn try_field_to_small(val: &Self) -> Option<i32> {
    try_field_to_small_impl(val)
  }
}

impl DelayedReduction<i32> for halo2curves::pasta::Fq {
  type UnreducedFieldInt = SignedWideLimbs<6>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_int_mul_add(acc: &mut Self::UnreducedFieldInt, field: &Self, small: i64) {
    // Handle sign: accumulate into pos or neg based on sign of small
    let (target, mag) = if small >= 0 {
      (&mut acc.pos, small as u64)
    } else {
      (&mut acc.neg, (-small) as u64)
    };
    // Fused multiply-accumulate: no intermediate array
    let a = &field.0;
    let (r0, c) = barrett::mac(target.0[0], a[0], mag, 0);
    let (r1, c) = barrett::mac(target.0[1], a[1], mag, c);
    let (r2, c) = barrett::mac(target.0[2], a[2], mag, c);
    let (r3, c) = barrett::mac(target.0[3], a[3], mag, c);
    // Propagate carry without multiply (just add)
    let (r4, of) = target.0[4].overflowing_add(c);
    target.0[0] = r0;
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = target.0[5].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  ) {
    let product = barrett::mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    let (neg, mag) = sub_mag::<6>(&acc.pos.0, &acc.neg.0);
    let r = Self(barrett::barrett_reduce_6_fq(&mag));
    if neg { -r } else { r }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self(barrett::montgomery_reduce_9_fq(&acc.0))
  }
}

impl SmallValueField<i64> for halo2curves::pasta::Fq {
  type IntermediateSmallValue = i128;

  #[inline]
  fn ss_mul(a: i64, b: i64) -> i128 {
    (a as i128) * (b as i128)
  }

  #[inline]
  fn sl_mul(small: i64, large: &Self) -> Self {
    barrett::mul_fq_by_i64(large, small)
  }

  #[inline]
  fn isl_mul(small: i128, large: &Self) -> Self {
    if small == 0 {
      return Self::zero();
    }
    let (is_neg, mag) = if small >= 0 {
      (false, small as u128)
    } else {
      (true, (-small) as u128)
    };
    // mul_4_by_2_ext produces 6 limbs, use barrett_reduce_6 directly (no padding)
    let product = barrett::mul_4_by_2_ext(&large.0, mag);
    let result = Self(barrett::barrett_reduce_6_fq(&product));
    if is_neg { -result } else { result }
  }

  #[inline]
  fn small_to_field(val: i64) -> Self {
    i64_to_field(val)
  }

  #[inline]
  fn intermediate_to_field(val: i128) -> Self {
    i128_to_field(val)
  }

  fn try_field_to_small(val: &Self) -> Option<i64> {
    let repr = val.to_repr();
    let bytes = repr.as_ref();

    let high_zero = bytes[8..].iter().all(|&b| b == 0);
    if high_zero {
      let val_u64 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
      if val_u64 <= i64::MAX as u64 {
        return Some(val_u64 as i64);
      }
    }

    let neg_val = val.neg();
    let neg_repr = neg_val.to_repr();
    let neg_bytes = neg_repr.as_ref();
    let neg_high_zero = neg_bytes[8..].iter().all(|&b| b == 0);
    if neg_high_zero {
      let neg_u64 = u64::from_le_bytes(neg_bytes[..8].try_into().unwrap());
      if neg_u64 > 0 && neg_u64 <= (i64::MAX as u64) + 1 {
        return Some(-(neg_u64 as i128) as i64);
      }
    }

    None
  }
}

impl DelayedReduction<i64> for halo2curves::pasta::Fq {
  type UnreducedFieldInt = SignedWideLimbs<8>;
  type UnreducedFieldField = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_field_int_mul_add(acc: &mut Self::UnreducedFieldInt, field: &Self, small: i128) {
    let (target, mag) = if small >= 0 {
      (&mut acc.pos, small as u128)
    } else {
      (&mut acc.neg, (-small) as u128)
    };
    // Fused 4×2 multiply-accumulate: two passes at different offsets
    let a = &field.0;
    let b_lo = mag as u64;
    let b_hi = (mag >> 64) as u64;

    // Pass 1: multiply by b_lo at offset 0
    let (r0, c) = barrett::mac(target.0[0], a[0], b_lo, 0);
    let (r1, c) = barrett::mac(target.0[1], a[1], b_lo, c);
    let (r2, c) = barrett::mac(target.0[2], a[2], b_lo, c);
    let (r3, c) = barrett::mac(target.0[3], a[3], b_lo, c);
    // Propagate carry without multiply (just add)
    let (r4, of1) = target.0[4].overflowing_add(c);
    let c1 = of1 as u64;
    target.0[0] = r0;

    // Pass 2: multiply by b_hi at offset 1 (add to r1..r5)
    let (r1, c) = barrett::mac(r1, a[0], b_hi, 0);
    let (r2, c) = barrett::mac(r2, a[1], b_hi, c);
    let (r3, c) = barrett::mac(r3, a[2], b_hi, c);
    let (r4, c) = barrett::mac(r4, a[3], b_hi, c);
    // Add both carries (c1 from pass 1, c from pass 2) into position 5
    let (r5, c) = barrett::mac(target.0[5], c1, 1, c);
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = r5;
    // Propagate final carry through remaining limbs (just add)
    let (r6, of) = target.0[6].overflowing_add(c);
    target.0[6] = r6;
    target.0[7] = target.0[7].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn unreduced_field_field_mul_add(
    acc: &mut Self::UnreducedFieldField,
    field_a: &Self,
    field_b: &Self,
  ) {
    let product = barrett::mul_4_by_4_ext(&field_a.0, &field_b.0);
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
  }

  #[inline(always)]
  fn reduce_field_int(acc: &Self::UnreducedFieldInt) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    let (neg, mag) = sub_mag::<8>(&acc.pos.0, &acc.neg.0);
    let r = Self(barrett::barrett_reduce_8_fq(&mag));
    if neg { -r } else { r }
  }

  #[inline(always)]
  fn reduce_field_field(acc: &Self::UnreducedFieldField) -> Self {
    Self(barrett::montgomery_reduce_9_fq(&acc.0))
  }
}

// ============================================================================
// Tests
// ============================================================================

// FpI64/FqI64 wrapper types have been removed - use SmallValueField<i64> directly

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{polys::multilinear::MultilinearPolynomial, provider::pasta::pallas};

  type Scalar = pallas::Scalar;

  #[test]
  fn test_small_value_field_arithmetic() {
    let a: i32 = 10;
    let b: i32 = 3;

    assert_eq!(a + b, 13);
    assert_eq!(a - b, 7);
    assert_eq!(-a, -10);
    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(a, b), 30i64);
    assert_eq!(a * 5, 50); // ss_mul_const is just native multiplication
  }

  #[test]
  fn test_small_value_field_negative() {
    let a: i32 = -5;
    let b: i32 = 3;

    assert_eq!(a + b, -2);
    assert_eq!(a - b, -8);
    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(a, b), -15i64);

    let field_a = <Scalar as SmallValueField<i32>>::small_to_field(a);
    assert_eq!(field_a, -Scalar::from(5u64));
  }

  #[test]
  fn test_i64_to_field() {
    let pos: Scalar = i64_to_field(100);
    assert_eq!(pos, Scalar::from(100u64));

    let neg: Scalar = i64_to_field(-50);
    assert_eq!(neg, -Scalar::from(50u64));
  }

  #[test]
  fn test_try_field_to_i64_roundtrip() {
    // Test positive values
    for val in [0i64, 1, 100, 1_000_000, i64::MAX / 2, i64::MAX] {
      let field: Scalar = i64_to_field(val);
      let back = try_field_to_i64(&field).expect("should fit");
      assert_eq!(back, val, "roundtrip failed for {}", val);
    }

    // Test negative values
    for val in [-1i64, -100, -1_000_000, i64::MIN / 2, i64::MIN + 1] {
      let field: Scalar = i64_to_field(val);
      let back = try_field_to_i64(&field).expect("should fit");
      assert_eq!(back, val, "roundtrip failed for {}", val);
    }

    // Test i64::MIN separately (edge case)
    let field: Scalar = i64_to_field(i64::MIN);
    let back = try_field_to_i64(&field).expect("should fit");
    assert_eq!(back, i64::MIN, "roundtrip failed for i64::MIN");

    // Test values that don't fit in i64
    let too_large = Scalar::from(u64::MAX) + Scalar::from(1u64);
    assert!(try_field_to_i64(&too_large).is_none());
  }

  #[test]
  fn test_small_multilinear_polynomial() {
    let poly = MultilinearPolynomial::new(vec![1i32, 2, 3, 4]);

    assert_eq!(poly.num_vars(), 2);
    assert_eq!(poly.Z.len(), 4);
    assert_eq!(poly[0], 1);
    assert_eq!(poly[3], 4);
  }

  #[test]
  fn test_to_field_conversion() {
    let evals: Vec<i32> = vec![1, -2, 3, -4];
    let small_poly = MultilinearPolynomial::new(evals);
    let field_poly: MultilinearPolynomial<Scalar> = small_poly.to_field();

    assert_eq!(field_poly.Z[0], Scalar::from(1u64));
    assert_eq!(field_poly.Z[1], -Scalar::from(2u64));
    assert_eq!(field_poly.Z[2], Scalar::from(3u64));
    assert_eq!(field_poly.Z[3], -Scalar::from(4u64));
  }

  #[test]
  fn test_try_field_to_small_roundtrip() {
    assert_eq!(
      <Scalar as SmallValueField<i32>>::try_field_to_small(&Scalar::from(42u64)),
      Some(42)
    );
    assert_eq!(
      <Scalar as SmallValueField<i32>>::try_field_to_small(&-Scalar::from(100u64)),
      Some(-100)
    );
    assert_eq!(
      <Scalar as SmallValueField<i32>>::try_field_to_small(&Scalar::from(u64::MAX)),
      None
    );
  }

  #[test]
  fn test_isl_mul() {
    use ff::Field;
    use rand_core::OsRng;

    let large = Scalar::random(&mut OsRng);
    let small: i64 = 12345;

    let result = <Scalar as SmallValueField<i32>>::isl_mul(small, &large);
    let expected = i64_to_field::<Scalar>(small) * large;

    assert_eq!(result, expected);
  }

  #[test]
  fn test_sl_mul() {
    use ff::Field;
    use rand_core::OsRng;

    let large = Scalar::random(&mut OsRng);
    let small: i32 = -999;

    let result = <Scalar as SmallValueField<i32>>::sl_mul(small, &large);
    let expected = <Scalar as SmallValueField<i32>>::small_to_field(small) * large;

    assert_eq!(result, expected);
  }

  #[test]
  fn test_overflow_bounds() {
    let typical_witness = 1i32 << 20;
    let extension_factor = 27i32;
    let after_extension = typical_witness * extension_factor;

    let prod = <Scalar as SmallValueField<i32>>::ss_mul(after_extension, after_extension);
    assert!(prod > 0);
    assert!(prod < (1i64 << 55));
  }

  #[test]
  fn test_ss_sign_combinations() {
    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(100, 200), 20000i64);
    assert_eq!(
      <Scalar as SmallValueField<i32>>::ss_mul(-100, -200),
      20000i64
    );
    assert_eq!(
      <Scalar as SmallValueField<i32>>::ss_mul(100, -200),
      -20000i64
    );
    assert_eq!(
      <Scalar as SmallValueField<i32>>::ss_mul(-100, 200),
      -20000i64
    );
  }

  #[test]
  fn test_ss_zero_edge_cases() {
    let zero = 0i32;
    let val = 12345i32;

    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(zero, val), 0i64);
    assert_eq!(<Scalar as SmallValueField<i32>>::ss_mul(val, zero), 0i64);
  }

  #[test]
  fn test_isl_with_random() {
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    let mut rng = OsRng;
    for _ in 0..100 {
      let large = Scalar::random(&mut rng);
      let small = (rng.next_u64() % (i64::MAX as u64)) as i64;
      let small = if rng.next_u32().is_multiple_of(2) {
        small
      } else {
        -small
      };

      let result = <Scalar as SmallValueField<i32>>::isl_mul(small, &large);
      let expected = i64_to_field::<Scalar>(small) * large;

      assert_eq!(result, expected);
    }
  }

  #[test]
  fn test_fp_small_value_field() {
    use halo2curves::pasta::Fp;

    let a: i32 = 42;
    let b: i32 = -10;

    assert_eq!(a + b, 32);
    assert_eq!(<Fp as SmallValueField<i32>>::ss_mul(a, b), -420i64);
    assert_eq!(
      <Fp as SmallValueField<i32>>::small_to_field(a),
      Fp::from(42u64)
    );
  }

  #[test]
  fn test_unreduced_field_int_mul_add() {
    use crate::wide_limbs::SignedWideLimbs;
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    let mut rng = OsRng;
    let mut acc: SignedWideLimbs<6> = Default::default();
    let mut expected = Scalar::ZERO;

    // Sum 100 field × i64 products (mix of positive and negative)
    for i in 0..100 {
      let field = Scalar::random(&mut rng);
      let small_u = rng.next_u64() >> 32; // Keep smaller to avoid extreme overflow
      // Alternate signs for variety
      let small: i64 = if i % 2 == 0 {
        small_u as i64
      } else {
        -(small_u as i64)
      };

      <Scalar as DelayedReduction<i32>>::unreduced_field_int_mul_add(&mut acc, &field, small);
      expected += field * i64_to_field::<Scalar>(small);
    }

    let result = <Scalar as DelayedReduction<i32>>::reduce_field_int(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_unreduced_field_field_mul_add() {
    use crate::wide_limbs::WideLimbs;
    use ff::Field;
    use rand_core::OsRng;

    let mut rng = OsRng;
    let mut acc: WideLimbs<9> = Default::default();
    let mut expected = Scalar::ZERO;

    // Sum 100 field × field products
    for _ in 0..100 {
      let a = Scalar::random(&mut rng);
      let b = Scalar::random(&mut rng);

      <Scalar as DelayedReduction<i32>>::unreduced_field_field_mul_add(&mut acc, &a, &b);
      expected += a * b;
    }

    let result = <Scalar as DelayedReduction<i32>>::reduce_field_field(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_unreduced_field_int_many_products() {
    use crate::wide_limbs::SignedWideLimbs;
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    let mut rng = OsRng;
    let mut acc: SignedWideLimbs<6> = Default::default();
    let mut expected = Scalar::ZERO;

    // Stress test: sum 2000 products (mix of positive and negative)
    for i in 0..2000 {
      let field = Scalar::random(&mut rng);
      let small_u = rng.next_u64();
      // Alternate signs for variety
      let small: i64 = if i % 3 == 0 {
        (small_u >> 1) as i64 // positive, shifted to fit in i64
      } else if i % 3 == 1 {
        -((small_u >> 1) as i64) // negative
      } else {
        0 // occasionally zero
      };

      <Scalar as DelayedReduction<i32>>::unreduced_field_int_mul_add(&mut acc, &field, small);
      expected += field * i64_to_field::<Scalar>(small);
    }

    let result = <Scalar as DelayedReduction<i32>>::reduce_field_int(&acc);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_unreduced_field_field_many_products() {
    use crate::wide_limbs::WideLimbs;
    use ff::Field;
    use rand_core::OsRng;

    let mut rng = OsRng;
    let mut acc: WideLimbs<9> = Default::default();
    let mut expected = Scalar::ZERO;

    // Stress test: sum 1000 products
    for _ in 0..1000 {
      let a = Scalar::random(&mut rng);
      let b = Scalar::random(&mut rng);

      <Scalar as DelayedReduction<i32>>::unreduced_field_field_mul_add(&mut acc, &a, &b);
      expected += a * b;
    }

    let result = <Scalar as DelayedReduction<i32>>::reduce_field_field(&acc);
    assert_eq!(result, expected);
  }
}
