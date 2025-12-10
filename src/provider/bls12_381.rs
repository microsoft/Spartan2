// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module implements the Spartan traits for BLS12-381 G1 curve.
//!
//! BLS12-381 is a pairing-friendly elliptic curve offering ~128 bits of security.
//! This implementation provides compatibility with Dory-PC polynomial commitments.

use crate::{
  impl_traits,
  provider::{
    msm::{msm, msm_small},
    traits::{DlogGroup, DlogGroupExt},
  },
  traits::{Group, PrimeFieldExt, transcript::TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::FromUniformBytes;
use halo2curves::{
  CurveAffine, CurveExt,
  bls12381::{G1, G1Affine},
  group::{Curve, Group as AnotherGroup, cofactor::CofactorCurveAffine},
};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Num, ToPrimitive};
use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;

/// Re-exports that give access to the standard aliases used in the code base, for BLS12-381 G1
pub mod g1 {
  pub use halo2curves::bls12381::{Fq as Base, Fr as Scalar, G1 as Point, G1Affine as Affine};
}

impl_traits!(
  g1,
  G1,
  G1Affine,
  // BLS12-381 scalar field order (Fr)
  "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
  // BLS12-381 base field order (Fq)  
  "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
);

// Implement TranscriptReprTrait for Fq (base field) - required by Engine trait
impl<G: Group> TranscriptReprTrait<G> for halo2curves::bls12381::Fq {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_bytes().into_iter().rev().collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::traits::DlogGroup;
  use halo2curves::group::Curve;
  use ff::Field;
  use rand_core::OsRng;

  // ==================== BASIC FUNCTIONALITY ====================

  #[test]
  fn test_bls12_381_from_label() {
    let label = b"test_bls12_381_from_label";
    for n in [1, 2, 4, 8, 16, 32, 64, 128] {
      let ck = g1::Point::from_label(label, n);
      assert_eq!(ck.len(), n);
      
      for point in &ck {
        assert!(bool::from(point.is_on_curve()));
      }
    }
  }

  #[test]
  fn test_bls12_381_generator() {
    let generator = g1::Point::generator();
    assert!(!bool::from(generator.is_identity()));
    assert!(bool::from(generator.to_affine().is_on_curve()));
  }

  #[test]
  fn test_bls12_381_zero() {
    let zero = g1::Point::zero();
    assert!(bool::from(zero.is_identity()));
  }

  // ==================== DETERMINISM & CONSISTENCY ====================

  #[test]
  fn test_from_label_deterministic() {
    // Same label must produce same generators
    let label = b"determinism_test";
    let n = 32;
    
    let ck1 = g1::Point::from_label(label, n);
    let ck2 = g1::Point::from_label(label, n);
    
    assert_eq!(ck1, ck2, "from_label must be deterministic");
  }

  #[test]
  fn test_from_label_different_labels_differ() {
    // Different labels must produce different generators
    let ck1 = g1::Point::from_label(b"label_a", 8);
    let ck2 = g1::Point::from_label(b"label_b", 8);
    
    assert_ne!(ck1, ck2, "different labels must produce different generators");
  }

  #[test]
  fn test_from_label_prefix_independence() {
    // Requesting more generators should extend, not change existing
    let label = b"prefix_test";
    let ck_small = g1::Point::from_label(label, 16);
    let ck_large = g1::Point::from_label(label, 32);
    
    assert_eq!(&ck_small[..], &ck_large[..16], "prefix must be consistent");
  }

  // ==================== GROUP LAW PROPERTIES ====================

  #[test]
  fn test_group_identity() {
    let g = g1::Point::generator();
    let zero = g1::Point::zero();
    
    // g + 0 = g
    assert_eq!(g + zero, g, "identity element property");
    // 0 + g = g
    assert_eq!(zero + g, g, "identity element property (commutative)");
  }

  #[test]
  fn test_group_inverse() {
    let g = g1::Point::generator();
    let neg_g = -g;
    
    // g + (-g) = 0
    assert!(bool::from((g + neg_g).is_identity()), "inverse property");
  }

  #[test]
  fn test_group_associativity() {
    let a = g1::Point::generator();
    let b = a + a;
    let c = b + a;
    
    // (a + b) + c = a + (b + c)
    let lhs = (a + b) + c;
    let rhs = a + (b + c);
    assert_eq!(lhs, rhs, "associativity");
  }

  #[test]
  fn test_scalar_multiplication() {
    use halo2curves::group::Group;
    
    let g = g1::Point::generator();
    let two = g1::Scalar::from(2u64);
    let three = g1::Scalar::from(3u64);
    
    // 2*g = g + g
    assert_eq!(g * two, g + g, "scalar mul by 2");
    
    // 3*g = g + g + g
    assert_eq!(g * three, g + g + g, "scalar mul by 3");
    
    // 0*g = identity
    assert!(bool::from((g * g1::Scalar::ZERO).is_identity()), "scalar mul by 0");
    
    // 1*g = g
    assert_eq!(g * g1::Scalar::ONE, g, "scalar mul by 1");
  }

  // ==================== MSM CORRECTNESS ====================

  #[test]
  fn test_msm_correctness_against_naive() {
    let n = 16;
    let label = b"msm_correctness";
    let bases = g1::Point::from_label(label, n);
    
    let scalars: Vec<g1::Scalar> = (0..n)
      .map(|i| g1::Scalar::from((i + 1) as u64))
      .collect();

    // MSM result
    let msm_result = g1::Point::vartime_multiscalar_mul(&scalars, &bases, true).unwrap();
    
    // Naive computation: sum of scalar_i * base_i
    let mut naive_result = g1::Point::zero();
    for (scalar, base) in scalars.iter().zip(bases.iter()) {
      naive_result += g1::Point::from(*base) * scalar;
    }
    
    assert_eq!(msm_result, naive_result, "MSM must match naive computation");
  }

  #[test]
  fn test_msm_empty() {
    let bases: Vec<g1::Affine> = vec![];
    let scalars: Vec<g1::Scalar> = vec![];
    
    let result = g1::Point::vartime_multiscalar_mul(&scalars, &bases, true);
    // Empty MSM should return identity or error - both are valid
    if let Ok(point) = result {
      assert!(bool::from(point.is_identity()), "empty MSM should be identity");
    }
  }

  #[test]
  fn test_msm_single_element() {
    let bases = g1::Point::from_label(b"single", 1);
    let scalar = g1::Scalar::from(42u64);
    
    let msm_result = g1::Point::vartime_multiscalar_mul(&[scalar], &bases, true).unwrap();
    let expected = g1::Point::from(bases[0]) * scalar;
    
    assert_eq!(msm_result, expected, "single-element MSM");
  }

  #[test]
  fn test_msm_all_zeros() {
    let n = 8;
    let bases = g1::Point::from_label(b"zeros", n);
    let scalars: Vec<g1::Scalar> = vec![g1::Scalar::ZERO; n];
    
    let result = g1::Point::vartime_multiscalar_mul(&scalars, &bases, true).unwrap();
    assert!(bool::from(result.is_identity()), "MSM with all-zero scalars");
  }

  // ==================== FIELD ARITHMETIC ====================

  #[test]
  fn test_scalar_field_properties() {
    let a = g1::Scalar::random(OsRng);
    let b = g1::Scalar::random(OsRng);
    let c = g1::Scalar::random(OsRng);
    
    // Commutativity
    assert_eq!(a + b, b + a, "addition commutativity");
    assert_eq!(a * b, b * a, "multiplication commutativity");
    
    // Associativity
    assert_eq!((a + b) + c, a + (b + c), "addition associativity");
    assert_eq!((a * b) * c, a * (b * c), "multiplication associativity");
    
    // Distributivity
    assert_eq!(a * (b + c), a * b + a * c, "distributivity");
    
    // Identity
    assert_eq!(a + g1::Scalar::ZERO, a, "additive identity");
    assert_eq!(a * g1::Scalar::ONE, a, "multiplicative identity");
    
    // Inverse
    assert_eq!(a + (-a), g1::Scalar::ZERO, "additive inverse");
    if !bool::from(a.is_zero()) {
      assert_eq!(a * a.invert().unwrap(), g1::Scalar::ONE, "multiplicative inverse");
    }
  }

  #[test]
  fn test_scalar_zero_inverse_fails() {
    let zero = g1::Scalar::ZERO;
    assert!(bool::from(zero.invert().is_none()), "zero has no inverse");
  }

  // ==================== TRANSCRIPT REPR ====================

  #[test]
  fn test_transcript_repr_deterministic() {
    use crate::traits::transcript::TranscriptReprTrait;
    
    let g = g1::Point::generator();
    let bytes1: Vec<u8> = <g1::Point as TranscriptReprTrait<g1::Point>>::to_transcript_bytes(&g);
    let bytes2: Vec<u8> = <g1::Point as TranscriptReprTrait<g1::Point>>::to_transcript_bytes(&g);
    
    assert_eq!(bytes1, bytes2, "transcript repr must be deterministic");
    assert!(!bytes1.is_empty());
  }

  #[test]
  fn test_transcript_repr_different_points_differ() {
    use crate::traits::transcript::TranscriptReprTrait;
    
    let g = g1::Point::generator();
    let two_g = g + g;
    
    let bytes1: Vec<u8> = <g1::Point as TranscriptReprTrait<g1::Point>>::to_transcript_bytes(&g);
    let bytes2: Vec<u8> = <g1::Point as TranscriptReprTrait<g1::Point>>::to_transcript_bytes(&two_g);
    
    assert_ne!(bytes1, bytes2, "different points must have different repr");
  }

  // ==================== COORDINATES ====================

  #[test]
  fn test_to_coordinates() {
    let g = g1::Point::generator();
    let (x, y, is_inf) = g.to_coordinates();
    
    assert!(!is_inf, "generator is not at infinity");
    
    // Verify point is on curve: y^2 = x^3 + 4 (BLS12-381 G1)
    let y_sq = y.square();
    let x_cubed_plus_b = x.square() * x + halo2curves::bls12381::Fq::from(4u64);
    assert_eq!(y_sq, x_cubed_plus_b, "generator must satisfy curve equation");
  }

  #[test]
  fn test_identity_coordinates() {
    let zero = g1::Point::zero();
    let (_, _, is_inf) = zero.to_coordinates();
    
    assert!(is_inf, "identity point is at infinity");
  }
}

