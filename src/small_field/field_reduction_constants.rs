// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Field-specific constants for modular reduction (Barrett and Montgomery).

use halo2curves::{
  bn256::Fr as Bn254Fr,
  pasta::{Fp, Fq},
  t256::Fq as T256Fq,
};

// ==========================================================================
// FieldReductionConstants - Trait for field-specific reduction constants
// ==========================================================================

/// Trait providing precomputed constants for efficient modular reduction.
///
/// # Overview
///
/// When reducing a wide integer (more than 4 limbs = 256 bits) modulo a prime p,
/// we need to handle "overflow limbs" that represent values >= 2^256. Each overflow
/// limb at position i represents the value `limb[i] * 2^(64*i)`.
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
/// | `R512_MOD` | 2^512 mod p | Reducing 9th limb (bits 512-575) |
///
/// # Example: 6-limb Reduction
///
/// For a 6-limb value `c = [c0, c1, c2, c3, c4, c5]` representing:
/// ```text
/// c = c0 + c1*2^64 + c2*2^128 + c3*2^192 + c4*2^256 + c5*2^320
/// ```
///
/// We reduce by computing:
/// ```text
/// c mod p = (c0 + c1*2^64 + c2*2^128 + c3*2^192)
///         + c4*(2^256 mod p)
///         + c5*(2^320 mod p)
/// ```
///
/// Since `R256_MOD` and `R320_MOD` are 4-limb values (< 2^256), multiplying
/// by a single limb produces at most a 5-limb result, which can then be
/// reduced further if needed.
///
/// # Why This Works
///
/// By the properties of modular arithmetic:
/// `a = b (mod p) => c*a = c*b (mod p)`
///
/// So `c5*2^320 = c5*R320_MOD (mod p)`, and the right side is much smaller.
///
/// # Performance
///
/// Much faster than naive division: avoids division entirely, uses only
/// 4-5 64-bit multiplications with precomputed constants.
pub trait FieldReductionConstants {
  /// The 4-limb prime modulus p (little-endian, 256 bits)
  const MODULUS: [u64; 4];

  /// 2^256 mod p - reduces the 5th limb (index 4) of a wide integer
  const R256_MOD: [u64; 4];

  /// 2^320 mod p - reduces the 6th limb (index 5) of a wide integer
  const R320_MOD: [u64; 4];

  /// 2^384 mod p - reduces the 7th limb (index 6) of a wide integer
  const R384_MOD: [u64; 4];

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
// FieldReductionConstants implementation for Bn254Fr (BN254 scalar field)
// ==========================================================================

impl FieldReductionConstants for Bn254Fr {
  // r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
  const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
  ];

  // 2^256 mod r
  const R256_MOD: [u64; 4] = [
    0xac96341c4ffffffb,
    0x36fc76959f60cd29,
    0x666ea36f7879462e,
    0x0e0a77c19a07df2f,
  ];

  // 2^320 mod r
  const R320_MOD: [u64; 4] = [
    0xb4c6edf97c5fb586,
    0x708c8d50bfeb93be,
    0x9ffd1de404f7e0ef,
    0x215b02ac9a392866,
  ];

  // 2^384 mod r
  const R384_MOD: [u64; 4] = [
    0xb075da81ef8cfeb9,
    0xa7f12acca5b6cd8c,
    0x32c475047957bf7b,
    0x03d581d748ffa25e,
  ];

  // 2^512 mod r
  const R512_MOD: [u64; 4] = [
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
  ];

  // -r^(-1) mod 2^64
  const MONT_INV: u64 = 0xc2e1f593efffffff;
}

// ==========================================================================
// FieldReductionConstants implementation for T256Fq (T256 scalar field = secp256r1 base field)
// ==========================================================================

impl FieldReductionConstants for T256Fq {
  // p = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
  const MODULUS: [u64; 4] = [
    0xffffffffffffffff,
    0x00000000ffffffff,
    0x0000000000000000,
    0xffffffff00000001,
  ];

  // 2^256 mod p
  const R256_MOD: [u64; 4] = [
    0x0000000000000001,
    0xffffffff00000000,
    0xffffffffffffffff,
    0x00000000fffffffe,
  ];

  // 2^320 mod p
  const R320_MOD: [u64; 4] = [
    0x00000000ffffffff,
    0x0000000100000001,
    0xfffffffeffffffff,
    0xfffffffe00000000,
  ];

  // 2^384 mod p
  const R384_MOD: [u64; 4] = [
    0xfffffffefffffffe,
    0x00000002ffffffff,
    0x0000000000000002,
    0xfffffffe00000001,
  ];

  // 2^512 mod p
  const R512_MOD: [u64; 4] = [
    0x0000000000000003,
    0xfffffffbffffffff,
    0xfffffffffffffffe,
    0x00000004fffffffd,
  ];

  // -p^(-1) mod 2^64
  const MONT_INV: u64 = 0x0000000000000001;
}
