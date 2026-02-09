// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Field-specific constants for Montgomery reduction.

use halo2curves::{
  bn256::Fr as Bn254Fr,
  pasta::{Fp, Fq},
  t256::Fq as T256Fq,
};

/// Trait providing precomputed constants for efficient modular reduction.
///
/// When reducing a wide integer (more than 4 limbs = 256 bits) modulo a prime p,
/// we need to handle "overflow limbs" that represent values >= 2^256.
pub trait FieldReductionConstants {
  /// The 4-limb prime modulus p (little-endian, 256 bits)
  const MODULUS: [u64; 4];

  /// R² mod p where R = 2^256 (equivalently, 2^512 mod p).
  ///
  /// Used in `montgomery_reduce_9` to fold a 9-limb accumulator to 8 limbs:
  /// the 9th limb h represents h·2^512, which we replace with h·R512_MOD ≡ h·2^512 (mod p).
  const R512_MOD: [u64; 4];

  /// Montgomery inverse: -p^(-1) mod 2^64
  const MONT_INV: u64;

  /// R mod p = 2^256 mod p (Montgomery representation of 1)
  /// Used for carry correction after folding: if carry c=1, add R_MOD.
  const R_MOD: [u64; 4];

  /// Q = ⌊R/p⌋, the number of conditional subtracts needed to canonicalize
  /// a value in [0, R) to [0, p).
  ///
  /// Used in `montgomery_reduce_8` after the 5th-limb check brings the value below R.
  const MAX_CANONICALIZE_SUBS: usize;
}

impl FieldReductionConstants for Fp {
  const MODULUS: [u64; 4] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
  ];

  const R512_MOD: [u64; 4] = [
    0x8c78ecb30000000f,
    0xd7d30dbd8b0de0e7,
    0x7797a99bc3c95d18,
    0x096d41af7b9cb714,
  ];

  const MONT_INV: u64 = 0x992d30ecffffffff;

  // R mod p = 2^256 mod p (extracted from Fp::ONE.0)
  const R_MOD: [u64; 4] = [
    0x34786d38fffffffd,
    0x992c350be41914ad,
    0xffffffffffffffff,
    0x3fffffffffffffff,
  ];

  // Q = ⌊R/p⌋ = 3 (p ≈ 2^254, so R/p ≈ 4, floor is 3)
  const MAX_CANONICALIZE_SUBS: usize = 3;
}

impl FieldReductionConstants for Fq {
  const MODULUS: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
  ];

  const R512_MOD: [u64; 4] = [
    0xfc9678ff0000000f,
    0x67bb433d891a16e3,
    0x7fae231004ccf590,
    0x096d41af7ccfdaa9,
  ];

  const MONT_INV: u64 = 0x8c46eb20ffffffff;

  // R mod p = 2^256 mod p (extracted from Fq::ONE.0)
  const R_MOD: [u64; 4] = [
    0x5b2b3e9cfffffffd,
    0x992c350be3420567,
    0xffffffffffffffff,
    0x3fffffffffffffff,
  ];

  // Q = ⌊R/p⌋ = 3 (p ≈ 2^254, so R/p ≈ 4, floor is 3)
  const MAX_CANONICALIZE_SUBS: usize = 3;
}

impl FieldReductionConstants for Bn254Fr {
  const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
  ];

  const R512_MOD: [u64; 4] = [
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
  ];

  const MONT_INV: u64 = 0xc2e1f593efffffff;

  // R mod p = 2^256 mod p (extracted from Bn254Fr::ONE.0)
  const R_MOD: [u64; 4] = [
    0xac96341c4ffffffb,
    0x36fc76959f60cd29,
    0x666ea36f7879462e,
    0x0e0a77c19a07df2f,
  ];

  // Q = ⌊R/p⌋ = 5 (p ≈ 0.76 * 2^254, so R/p ≈ 5.26)
  const MAX_CANONICALIZE_SUBS: usize = 5;
}

impl FieldReductionConstants for T256Fq {
  const MODULUS: [u64; 4] = [
    0xffffffffffffffff,
    0x00000000ffffffff,
    0x0000000000000000,
    0xffffffff00000001,
  ];

  const R512_MOD: [u64; 4] = [
    0x0000000000000003,
    0xfffffffbffffffff,
    0xfffffffffffffffe,
    0x00000004fffffffd,
  ];

  const MONT_INV: u64 = 0x0000000000000001;

  // R mod p = 2^256 mod p (extracted from T256Fq::ONE.0)
  const R_MOD: [u64; 4] = [
    0x0000000000000001,
    0xffffffff00000000,
    0xffffffffffffffff,
    0x00000000fffffffe,
  ];

  // Q = ⌊R/p⌋ = 1 (p ≈ 2^256 - 2^224, so R/p ≈ 1.000...)
  const MAX_CANONICALIZE_SUBS: usize = 1;
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::big_num::montgomery::MontgomeryLimbs;
  use ff::PrimeField;
  use num_bigint::BigUint;

  /// Convert 4 little-endian u64 limbs to a BigUint.
  fn limbs_to_biguint(limbs: &[u64; 4]) -> BigUint {
    let mut bytes = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
      bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes)
  }


  /// Verify R512_MOD ≡ 2^512 (mod p) using big integer arithmetic.
  /// This catches any typos in the hardcoded constant.
  fn test_r512_mod_direct<F: FieldReductionConstants>() {
    let p = limbs_to_biguint(&F::MODULUS);
    let two_pow_512 = BigUint::from(1u64) << 512;
    let expected = &two_pow_512 % &p;
    let actual = limbs_to_biguint(&F::R512_MOD);

    assert_eq!(
      actual, expected,
      "R512_MOD must equal 2^512 mod p"
    );
  }

  /// Verify MONT_INV: p * MONT_INV ≡ -1 (mod 2^64)
  /// This is the Montgomery inverse used in REDC.
  fn test_mont_inv<F: FieldReductionConstants>() {
    let product = F::MODULUS[0].wrapping_mul(F::MONT_INV);
    assert_eq!(
      product,
      u64::MAX,
      "MONT_INV verification failed: p[0] * MONT_INV should equal -1 (mod 2^64)"
    );
  }

  /// Verify MODULUS matches the field's actual modulus.
  /// We check by verifying that MODULUS - 1 equals the field's max element (-1).
  fn test_modulus<F: FieldReductionConstants + PrimeField>() {
    let neg_one = -F::ONE;
    let neg_one_limbs = neg_one.to_repr();

    // Convert our MODULUS-1 to bytes (little-endian)
    let mut modulus_minus_one = <F as FieldReductionConstants>::MODULUS;
    let (new_val, borrow) = modulus_minus_one[0].overflowing_sub(1);
    modulus_minus_one[0] = new_val;
    assert!(!borrow, "MODULUS should be > 1");

    let mut modulus_bytes = [0u8; 32];
    for (i, limb) in modulus_minus_one.iter().enumerate() {
      modulus_bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }

    assert_eq!(
      neg_one_limbs.as_ref(),
      &modulus_bytes[..],
      "MODULUS verification failed: does not match field's actual modulus"
    );
  }

  /// Verify R_MOD equals the internal limbs of F::ONE (Montgomery representation of 1).
  /// This ensures our hardcoded constant matches the field's actual R mod p value.
  fn test_r_mod<F: FieldReductionConstants + MontgomeryLimbs + PrimeField>() {
    let one = F::ONE;
    let one_limbs = one.to_limbs();
    assert_eq!(
      <F as FieldReductionConstants>::R_MOD,
      *one_limbs,
      "R_MOD must equal F::ONE's internal limbs (Montgomery representation of 1)"
    );
  }

  /// Verify MODULUS bit length matches expected field size (254-256 bits).
  fn test_modulus_bit_length<F: FieldReductionConstants>() {
    let p = limbs_to_biguint(&F::MODULUS);
    assert!(
      p.bits() >= 254 && p.bits() <= 256,
      "MODULUS should be 254-256 bits, got {} bits",
      p.bits()
    );
  }

  /// Verify Q = ⌊R/p⌋ using big integer division.
  fn test_q_value_direct<F: FieldReductionConstants>() {
    let p = limbs_to_biguint(&F::MODULUS);
    let r = BigUint::from(1u64) << 256;
    let expected_q = &r / &p;
    assert_eq!(
      BigUint::from(F::MAX_CANONICALIZE_SUBS as u64),
      expected_q,
      "MAX_CANONICALIZE_SUBS should equal floor(R/p)"
    );
  }

  // =========================================================================
  // MODULUS: Verify hardcoded modulus matches field's actual prime
  // =========================================================================

  #[test]
  fn test_fp_modulus() {
    test_modulus::<Fp>();
  }

  #[test]
  fn test_fq_modulus() {
    test_modulus::<Fq>();
  }

  #[test]
  fn test_bn254fr_modulus() {
    test_modulus::<Bn254Fr>();
  }

  #[test]
  fn test_t256fq_modulus() {
    test_modulus::<T256Fq>();
  }

  #[test]
  fn test_fp_modulus_bit_length() {
    test_modulus_bit_length::<Fp>();
  }

  #[test]
  fn test_fq_modulus_bit_length() {
    test_modulus_bit_length::<Fq>();
  }

  #[test]
  fn test_bn254fr_modulus_bit_length() {
    test_modulus_bit_length::<Bn254Fr>();
  }

  #[test]
  fn test_t256fq_modulus_bit_length() {
    test_modulus_bit_length::<T256Fq>();
  }

  // =========================================================================
  // MONT_INV: Verify p[0] * MONT_INV ≡ -1 (mod 2^64)
  // =========================================================================

  #[test]
  fn test_fp_mont_inv() {
    test_mont_inv::<Fp>();
  }

  #[test]
  fn test_fq_mont_inv() {
    test_mont_inv::<Fq>();
  }

  #[test]
  fn test_bn254fr_mont_inv() {
    test_mont_inv::<Bn254Fr>();
  }

  #[test]
  fn test_t256fq_mont_inv() {
    test_mont_inv::<T256Fq>();
  }

  // =========================================================================
  // R_MOD: Verify R_MOD = 2^256 mod p (Montgomery form of 1)
  // =========================================================================

  #[test]
  fn test_fp_r_mod() {
    test_r_mod::<Fp>();
  }

  #[test]
  fn test_fq_r_mod() {
    test_r_mod::<Fq>();
  }

  #[test]
  fn test_bn254fr_r_mod() {
    test_r_mod::<Bn254Fr>();
  }

  #[test]
  fn test_t256fq_r_mod() {
    test_r_mod::<T256Fq>();
  }

  // =========================================================================
  // R512_MOD: Verify R512_MOD = 2^512 mod p (direct BigUint computation)
  // =========================================================================

  #[test]
  fn test_fp_r512_mod_direct() {
    test_r512_mod_direct::<Fp>();
  }

  #[test]
  fn test_fq_r512_mod_direct() {
    test_r512_mod_direct::<Fq>();
  }

  #[test]
  fn test_bn254fr_r512_mod_direct() {
    test_r512_mod_direct::<Bn254Fr>();
  }

  #[test]
  fn test_t256fq_r512_mod_direct() {
    test_r512_mod_direct::<T256Fq>();
  }

  // =========================================================================
  // MAX_CANONICALIZE_SUBS (Q): Verify Q = ⌊R/p⌋ and is tight
  // =========================================================================

  #[test]
  fn test_fp_q_value_direct() {
    test_q_value_direct::<Fp>();
  }

  #[test]
  fn test_fq_q_value_direct() {
    test_q_value_direct::<Fq>();
  }

  #[test]
  fn test_bn254fr_q_value_direct() {
    test_q_value_direct::<Bn254Fr>();
  }

  #[test]
  fn test_t256fq_q_value_direct() {
    test_q_value_direct::<T256Fq>();
  }

}
