// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Field-specific constants for Montgomery reduction.

use halo2curves::{
  bn256::Fr as Bn254Fr,
  pasta::{Fp, Fq},
  secp256r1::Fq as P256Fq,
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

  /// Maximum correction subtractions needed in Montgomery REDC: ⌊R/p⌋.
  ///
  /// Standard REDC produces results in [0, R) where R = 2^256. Since R > p
  /// for 256-bit primes, we need up to ⌊R/p⌋ final subtractions to reduce
  /// the result into the canonical range [0, p).
  const MAX_REDC_SUB_CORRECTIONS: usize;
}

// Use macro to implement FieldReductionConstants for all supported fields.
// All constants are computed at compile time from the field's MODULUS and ONE.
crate::impl_field_reduction_constants!(Fp);
crate::impl_field_reduction_constants!(Fq);
crate::impl_field_reduction_constants!(Bn254Fr);
crate::impl_field_reduction_constants!(T256Fq);
crate::impl_field_reduction_constants!(P256Fq);

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
  fn test_r512_mod_direct_impl<F: FieldReductionConstants>() {
    let p = limbs_to_biguint(&F::MODULUS);
    let two_pow_512 = BigUint::from(1u64) << 512;
    let expected = &two_pow_512 % &p;
    let actual = limbs_to_biguint(&F::R512_MOD);

    assert_eq!(actual, expected, "R512_MOD must equal 2^512 mod p");
  }

  /// Verify MONT_INV: p * MONT_INV ≡ -1 (mod 2^64)
  fn test_mont_inv_impl<F: FieldReductionConstants>() {
    let product = F::MODULUS[0].wrapping_mul(F::MONT_INV);
    assert_eq!(
      product,
      u64::MAX,
      "MONT_INV verification failed: p[0] * MONT_INV should equal -1 (mod 2^64)"
    );
  }

  /// Verify MODULUS matches the field's actual modulus.
  fn test_modulus_impl<F: FieldReductionConstants + PrimeField>() {
    let neg_one = -F::ONE;
    let neg_one_limbs = neg_one.to_repr();

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

  /// Verify R_MOD equals the internal limbs of F::ONE.
  fn test_r_mod_impl<F: FieldReductionConstants + MontgomeryLimbs + PrimeField>() {
    let one = F::ONE;
    let one_limbs = one.to_limbs();
    assert_eq!(
      <F as FieldReductionConstants>::R_MOD,
      *one_limbs,
      "R_MOD must equal F::ONE's internal limbs (Montgomery representation of 1)"
    );
  }

  /// Verify MODULUS bit length matches expected field size (254-256 bits).
  fn test_modulus_bit_length_impl<F: FieldReductionConstants>() {
    let p = limbs_to_biguint(&F::MODULUS);
    assert!(
      p.bits() >= 254 && p.bits() <= 256,
      "MODULUS should be 254-256 bits, got {} bits",
      p.bits()
    );
  }

  /// Verify MAX_REDC_SUB_CORRECTIONS = ⌊R/p⌋ using big integer division.
  fn test_max_redc_sub_corrections_impl<F: FieldReductionConstants>() {
    let p = limbs_to_biguint(&F::MODULUS);
    let r = BigUint::from(1u64) << 256;
    let expected = &r / &p;
    assert_eq!(
      BigUint::from(F::MAX_REDC_SUB_CORRECTIONS as u64),
      expected,
      "MAX_REDC_SUB_CORRECTIONS should equal floor(R/p)"
    );
  }

  /// Generate a test module for a field type's FieldReductionConstants.
  macro_rules! test_field_reduction_constants {
    ($mod_name:ident, $field:ty) => {
      mod $mod_name {
        use super::*;

        #[test]
        fn modulus() {
          test_modulus_impl::<$field>();
        }

        #[test]
        fn modulus_bit_length() {
          test_modulus_bit_length_impl::<$field>();
        }

        #[test]
        fn mont_inv() {
          test_mont_inv_impl::<$field>();
        }

        #[test]
        fn r_mod() {
          test_r_mod_impl::<$field>();
        }

        #[test]
        fn r512_mod_direct() {
          test_r512_mod_direct_impl::<$field>();
        }

        #[test]
        fn max_redc_sub_corrections() {
          test_max_redc_sub_corrections_impl::<$field>();
        }
      }
    };
  }

  test_field_reduction_constants!(fp_tests, Fp);
  test_field_reduction_constants!(fq_tests, Fq);
  test_field_reduction_constants!(bn254fr_tests, Bn254Fr);
  test_field_reduction_constants!(t256fq_tests, T256Fq);
  test_field_reduction_constants!(p256fq_tests, P256Fq);
}
