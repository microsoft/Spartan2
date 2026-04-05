// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Macros and const fn helpers for implementing field traits.
//!
//! This module provides macros for implementing various field-related traits:
//!
//! ## Reduction Constants
//! - [`impl_field_reduction_constants!`] - Montgomery REDC constants (all fields)
//! - [`impl_barrett_reduction_constants!`] - Generic Barrett constants (all fields)
//!
//! ## Limb Access
//! - [`impl_montgomery_limbs!`] - Montgomery limb access (all fields)
//!
//! ## Small Value Field
//! - [`impl_small_value_field!`] - SmallValueField<i32/i64/i128> (all fields)
//!
//! ## Delayed Reduction
//! - [`impl_delayed_reduction!`] - DelayedReduction for i32, i64, i128, and F×F
//!
//! All constants are computed at compile time from the field's `PrimeField::MODULUS`
//! and `Field::ONE` values.

use super::limbs::{clz, gte, gte_5_4, mul_4_by_4, reduce_8_mod_4, shl, shr, sub, sub_5_4};

// =============================================================================
// Implementation macros
// =============================================================================

/// Implement `FieldReductionConstants` for a field type.
///
/// This macro automatically computes all constants at compile time from the
/// field's `PrimeField::MODULUS` hex string and `Field::ONE` value.
///
/// # Example
/// ```ignore
/// crate::impl_field_reduction_constants!(Fp);
/// ```
#[macro_export]
macro_rules! impl_field_reduction_constants {
  ($field:ty) => {
    impl $crate::big_num::FieldReductionConstants for $field {
      const MODULUS: [u64; 4] =
        $crate::big_num::macros::parse_hex_to_limbs(<$field as ff::PrimeField>::MODULUS);
      const R_MOD: [u64; 4] = <$field as ff::Field>::ONE.0;
      const MONT_INV: u64 = $crate::big_num::macros::compute_mont_inv(Self::MODULUS[0]);
      const R512_MOD: [u64; 4] =
        $crate::big_num::macros::compute_r512_mod(Self::MODULUS, Self::R_MOD);
      const MAX_REDC_SUB_CORRECTIONS: usize =
        $crate::big_num::macros::compute_max_redc_sub_corrections(Self::MODULUS);
    }

    // Enforce small MAX_REDC_SUB_CORRECTIONS for performance.
    // The final reduction loop in montgomery_reduce_8 iterates this many times.
    // For 254-256 bit primes, this should be 1-4. Values > 8 indicate a problem.
    const _: () = assert!(
      <$field as $crate::big_num::FieldReductionConstants>::MAX_REDC_SUB_CORRECTIONS <= 8,
      "MAX_REDC_SUB_CORRECTIONS too large for efficient reduction"
    );
  };
}

/// Implement `MontgomeryLimbs` for a field type with `.0` containing `[u64; 4]`.
///
/// # Example
/// ```ignore
/// crate::impl_montgomery_limbs!(Fp);
/// ```
#[macro_export]
macro_rules! impl_montgomery_limbs {
  ($field:ty) => {
    impl $crate::big_num::montgomery::MontgomeryLimbs for $field {
      #[inline]
      fn from_limbs(limbs: [u64; 4]) -> Self {
        Self(limbs)
      }

      #[inline]
      fn to_limbs(&self) -> &[u64; 4] {
        &self.0
      }
    }
  };
}

/// Implement `BarrettReductionConstants` for a field type.
///
/// This macro computes generic μ-Barrett constants at compile time.
///
/// # Example
/// ```ignore
/// crate::impl_barrett_reduction_constants!(Bn254Fr);
/// ```
#[macro_export]
macro_rules! impl_barrett_reduction_constants {
  ($field:ty) => {
    impl $crate::big_num::BarrettReductionConstants for $field {
      const MODULUS: [u64; 4] =
        $crate::big_num::macros::parse_hex_to_limbs(<$field as ff::PrimeField>::MODULUS);
      const R384_MOD: [u64; 4] =
        $crate::big_num::macros::compute_r384_mod(Self::MODULUS, <$field as ff::Field>::ONE.0);
      const BARRETT_MU: [u64; 5] = $crate::big_num::macros::compute_barrett_mu(Self::MODULUS);
      const USE_4_LIMB_BARRETT: bool = $crate::big_num::macros::is_4_limb_barrett(Self::MODULUS);
    }
  };
}

// =============================================================================
// SmallValueField Macro
// =============================================================================

/// Implement `SmallValueField<i32>`, `SmallValueField<i64>`, `SmallValueField<i128>` for a field.
///
/// This macro generates implementations for converting between field elements and
/// small integer types (i32, i64, i128).
///
/// # Example
/// ```ignore
/// crate::impl_small_value_field!(pallas::Scalar);
/// ```
#[macro_export]
macro_rules! impl_small_value_field {
  ($field:ty) => {
    impl $crate::big_num::SmallValueField<i32> for $field {
      #[inline]
      fn small_to_field(val: i32) -> Self {
        $crate::big_num::small_value_field::i64_to_field(val as i64)
      }

      fn try_field_to_small(val: &Self) -> Option<i32> {
        $crate::big_num::small_value_field::try_field_to_small_i32(val)
      }
    }

    impl $crate::big_num::SmallValueField<i64> for $field {
      #[inline]
      fn small_to_field(val: i64) -> Self {
        $crate::big_num::small_value_field::i64_to_field(val)
      }

      fn try_field_to_small(val: &Self) -> Option<i64> {
        $crate::big_num::small_value_field::try_field_to_i64(val)
      }
    }

    impl $crate::big_num::SmallValueField<i128> for $field {
      #[inline]
      fn small_to_field(val: i128) -> Self {
        $crate::big_num::small_value_field::i128_to_field(val)
      }

      fn try_field_to_small(val: &Self) -> Option<i128> {
        $crate::big_num::small_value_field::try_field_to_i128(val)
      }
    }
  };
}

// =============================================================================
// Delayed Reduction Macro
// =============================================================================

/// Implement `DelayedReduction` for i32, i64, i128 using generic Barrett reduction.
///
/// This wires the accumulation functions to `barrett_reduce_6` and `barrett_reduce_7`
/// from the `barrett` module.
///
/// Note: `DelayedReduction<F> for F` (field × field) is provided by a blanket impl
/// in `delayed_reduction.rs` for all fields implementing `MontgomeryLimbs + PrimeField + Copy`.
///
/// # Example
/// ```ignore
/// crate::impl_delayed_reduction!(Bn254Fr);
/// ```
#[macro_export]
macro_rules! impl_delayed_reduction {
  ($field:ty) => {
    impl $crate::big_num::DelayedReduction<i32> for $field {
      type Accumulator = $crate::big_num::SignedWideLimbs<6>;

      #[inline(always)]
      fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &i32) {
        let value64 = *value as i64;
        let (target, mag) = if value64 >= 0 {
          (&mut acc.pos, value64 as u64)
        } else {
          (&mut acc.neg, value64.wrapping_neg() as u64)
        };
        $crate::big_num::delayed_reduction::accumulate_field_times_small(target, field, mag);
      }

      #[inline(always)]
      fn reduce(acc: &Self::Accumulator) -> Self {
        use $crate::big_num::montgomery::MontgomeryLimbs;
        match $crate::big_num::sub_mag::<6>(&acc.pos.0, &acc.neg.0) {
          $crate::big_num::SubMagResult::Positive(mag) => {
            Self::from_limbs($crate::big_num::barrett::barrett_reduce_6::<$field>(&mag))
          }
          $crate::big_num::SubMagResult::Negative(mag) => {
            -Self::from_limbs($crate::big_num::barrett::barrett_reduce_6::<$field>(&mag))
          }
        }
      }
    }

    impl $crate::big_num::DelayedReduction<i64> for $field {
      type Accumulator = $crate::big_num::SignedWideLimbs<6>;

      #[inline(always)]
      fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &i64) {
        let (target, mag) = if *value >= 0 {
          (&mut acc.pos, *value as u64)
        } else {
          (&mut acc.neg, (*value).wrapping_neg() as u64)
        };
        $crate::big_num::delayed_reduction::accumulate_field_times_small(target, field, mag);
      }

      #[inline(always)]
      fn reduce(acc: &Self::Accumulator) -> Self {
        use $crate::big_num::montgomery::MontgomeryLimbs;
        match $crate::big_num::sub_mag::<6>(&acc.pos.0, &acc.neg.0) {
          $crate::big_num::SubMagResult::Positive(mag) => {
            Self::from_limbs($crate::big_num::barrett::barrett_reduce_6::<$field>(&mag))
          }
          $crate::big_num::SubMagResult::Negative(mag) => {
            -Self::from_limbs($crate::big_num::barrett::barrett_reduce_6::<$field>(&mag))
          }
        }
      }
    }

    impl $crate::big_num::DelayedReduction<i128> for $field {
      type Accumulator = $crate::big_num::SignedWideLimbs<7>;

      #[inline(always)]
      fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &i128) {
        $crate::big_num::delayed_reduction::accumulate_field_times_i128(acc, field, value);
      }

      #[inline(always)]
      fn reduce(acc: &Self::Accumulator) -> Self {
        use $crate::big_num::montgomery::MontgomeryLimbs;
        match $crate::big_num::sub_mag::<7>(&acc.pos.0, &acc.neg.0) {
          $crate::big_num::SubMagResult::Positive(mag) => {
            Self::from_limbs($crate::big_num::barrett::barrett_reduce_7::<$field>(&mag))
          }
          $crate::big_num::SubMagResult::Negative(mag) => {
            -Self::from_limbs($crate::big_num::barrett::barrett_reduce_7::<$field>(&mag))
          }
        }
      }
    }
  };
}

// =============================================================================
// Const fn helpers for compile-time computation of field reduction constants
// =============================================================================

/// Parse a single hex character to its value (0-15).
const fn hex_char_to_nibble(c: u8) -> u8 {
  match c {
    b'0'..=b'9' => c - b'0',
    b'a'..=b'f' => c - b'a' + 10,
    b'A'..=b'F' => c - b'A' + 10,
    _ => panic!("invalid hex character"),
  }
}

/// Parse a 64-char or 66-char (with "0x" prefix) hex string to 4 little-endian u64 limbs.
///
/// The input string should be exactly 64 hex characters (or 66 with "0x" prefix)
/// representing a 256-bit number in big-endian order.
/// The output is 4 u64 limbs in little-endian order.
pub const fn parse_hex_to_limbs(s: &str) -> [u64; 4] {
  let bytes = s.as_bytes();

  // Handle optional "0x" prefix
  let (hex_start, hex_len) =
    if bytes.len() >= 2 && bytes[0] == b'0' && (bytes[1] == b'x' || bytes[1] == b'X') {
      (2, bytes.len() - 2)
    } else {
      (0, bytes.len())
    };

  assert!(
    hex_len == 64,
    "hex string must be 64 characters (excluding optional 0x prefix)"
  );

  let mut limbs = [0u64; 4];
  let mut i = 0;
  while i < 4 {
    // Each limb is 16 hex chars, big-endian in string
    // limbs[3] is most significant, comes first in string
    let limb_idx = 3 - i;
    let str_offset = hex_start + i * 16;

    let mut limb = 0u64;
    let mut j = 0;
    while j < 16 {
      let nibble = hex_char_to_nibble(bytes[str_offset + j]);
      limb = (limb << 4) | (nibble as u64);
      j += 1;
    }
    limbs[limb_idx] = limb;
    i += 1;
  }
  limbs
}

/// Compute -p[0]^(-1) mod 2^64 using a bit-doubling iteration.
///
/// **Why a 64-bit inverse, not 256-bit?**
/// We use limb-by-limb Montgomery reduction, zeroing one 64-bit limb per iteration.
/// Each step computes `m = t_low * mont_inv mod 2^64`, which only depends on
/// `p mod 2^64` (i.e., `p[0]`). This approach maps efficiently to 64-bit CPU
/// registers and avoids full 256×256-bit multiplications that a single-step
/// reduction with `-p^(-1) mod 2^256` would require.
///
/// **The iteration:** `inv' = inv * (2 - p0 * inv)` doubles correct bits.
///
/// Why? If `p0 * inv = 1 + e·2^k` (correct mod 2^k), then:
///   p0 * inv' = (1 + e·2^k)(1 - e·2^k) = 1 - e²·2^(2k) ≡ 1 (mod 2^(2k))
///
/// **Convergence:**
/// ```text
/// Start: p0 is odd, so p0·1 ≡ 1 (mod 2) →  1 bit
/// ×2 each iteration: 1 → 2 → 4 → 8 → 16 → 32 → 64 bits (6 iterations)
/// ```
///
/// Returns `-inv` for use in REDC.
pub const fn compute_mont_inv(p0: u64) -> u64 {
  // Bit-doubling iteration: inv = inv * (2 - p0 * inv)
  // Each iteration doubles correct bits: 1 → 2 → 4 → 8 → 16 → 32 → 64
  let mut inv = 1u64;
  inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
  inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
  inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
  inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
  inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
  inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
  inv.wrapping_neg()
}

/// Compute ⌊R/p⌋ where R = 2^256, by counting subtractions.
///
/// This is the maximum number of final subtractions needed in Montgomery REDC
/// to reduce a value from [0, R) into the canonical range [0, p).
pub const fn compute_max_redc_sub_corrections(p: [u64; 4]) -> usize {
  // Start with a = 2^256 represented as a 5-limb number
  // 2^256 = [0, 0, 0, 0, 1] in little-endian 5-limb representation
  let mut a = [0u64, 0, 0, 0, 1u64];
  let mut count = 0usize;

  // Count how many times we can subtract p from a
  while gte_5_4(&a, &p) {
    a = sub_5_4(&a, &p);
    count += 1;
  }

  count
}

/// Compute 2^512 mod p = R_MOD² mod p.
///
/// This computes R512_MOD = 2^512 mod p by squaring R_MOD (which is 2^256 mod p)
/// and then reducing the result modulo p.
pub const fn compute_r512_mod(p: [u64; 4], r_mod: [u64; 4]) -> [u64; 4] {
  // R512_MOD = R_MOD * R_MOD mod p = (2^256 mod p)² mod p = 2^512 mod p
  let r_squared = mul_4_by_4(&r_mod, &r_mod);
  reduce_8_mod_4(&r_squared, &p)
}

// =============================================================================
// Barrett reduction const fn helpers
// =============================================================================

/// Compute 2^384 mod p.
///
/// This computes R384_MOD = 2^384 mod p by multiplying R_MOD (2^256 mod p) by
/// 2^128 and then reducing.
pub const fn compute_r384_mod(p: [u64; 4], r_mod: [u64; 4]) -> [u64; 4] {
  // 2^384 = 2^256 * 2^128
  // R384_MOD = (R_MOD * 2^128) mod p
  // 2^128 as 4 limbs: [0, 0, 1, 0]
  let two_128 = [0u64, 0, 1, 0];
  let product = mul_4_by_4(&r_mod, &two_128);
  reduce_8_mod_4(&product, &p)
}

/// Compute Barrett reciprocal μ = ⌊2^512 / p⌋ at compile time.
///
/// For a 256-bit prime p, μ fits in 5 limbs (up to 320 bits).
/// Uses binary long division - O(n²) in bits but only runs at compile time.
pub const fn compute_barrett_mu(p: [u64; 4]) -> [u64; 5] {
  // 2^512 as 9 limbs: limb[8] = 1, rest = 0
  let mut dividend: [u64; 9] = [0, 0, 0, 0, 0, 0, 0, 0, 1];

  // Extend p to 9 limbs
  let divisor: [u64; 9] = [p[0], p[1], p[2], p[3], 0, 0, 0, 0, 0];

  // Quotient accumulator (5 limbs = 320 bits, enough for 2^512 / 2^254 ≈ 2^258)
  let mut quotient: [u64; 5] = [0; 5];

  // Find alignment: how many bits to shift divisor left
  let dividend_clz = clz::<9>(&dividend);
  let divisor_clz = clz::<9>(&divisor);

  if divisor_clz <= dividend_clz {
    // Divisor already larger than dividend (shouldn't happen for valid primes)
    return quotient;
  }

  let shift_bits = divisor_clz - dividend_clz;

  // Shift divisor left to align with dividend.
  // Batch by whole limbs (64 bits each), then remaining bits.
  let mut shifted_divisor = divisor;
  let whole_limbs = (shift_bits / 64) as usize;
  let rem_bits = shift_bits % 64;

  // Shift by whole limbs (move elements up)
  if whole_limbs > 0 {
    let mut i = 8;
    while i >= whole_limbs {
      shifted_divisor[i] = shifted_divisor[i - whole_limbs];
      i -= 1;
    }
    // Zero out the vacated low limbs (i is now whole_limbs - 1)
    let mut j = 0;
    while j < whole_limbs {
      shifted_divisor[j] = 0;
      j += 1;
    }
  }

  // Shift remaining bits one at a time
  let mut i = 0;
  while i < rem_bits {
    shifted_divisor = shl::<9>(&shifted_divisor);
    i += 1;
  }

  // Binary long division: shift_bits + 1 iterations
  let mut bit_pos = shift_bits;
  loop {
    if gte::<9>(&dividend, &shifted_divisor) {
      dividend = sub::<9>(&dividend, &shifted_divisor);
      // Set bit `bit_pos` in quotient
      let limb_idx = (bit_pos / 64) as usize;
      let bit_idx = bit_pos % 64;
      if limb_idx < 5 {
        quotient[limb_idx] |= 1u64 << bit_idx;
      }
    }

    if bit_pos == 0 {
      break;
    }
    bit_pos -= 1;

    // Shift divisor right by 1
    shifted_divisor = shr::<9>(&shifted_divisor);
  }

  quotient
}

/// Check if 2p < 2^256, enabling 4-limb Barrett fast path.
pub const fn is_4_limb_barrett(p: [u64; 4]) -> bool {
  // 2p < 2^256 iff p < 2^255 iff the MSB of p is 0
  p[3] < 0x8000_0000_0000_0000
}
