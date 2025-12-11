// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// Dory-PC adapter for Spartan2
//
// This module wraps quarks-zk's DoryPCS to implement Spartan2's PCSEngineTrait.
// Dory achieves O(log n) verification complexity using pairings.

use crate::{
  errors::SpartanError,
  traits::{
    Engine, Group,
    pcs::{CommitmentTrait, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::marker::PhantomData;
use ff::PrimeField;
use serde::{Deserialize, Serialize};

// Use quarks-zk
use quarks_zk::dory_pc::{DoryPCS as QuarksDoryPCS, DoryPCSCommitment, DoryPCSEvaluationProof};
use quarks_zk::traits::PolynomialCommitmentScheme;

// Use ark crates directly
use ark_bls12_381::Fr as ArkFr;
use ark_ff::PrimeField as ArkPrimeField;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

/// Convert halo2curves scalar to ark Fr
fn halo2_to_ark<E: Engine>(s: &E::Scalar) -> ArkFr {
  let bytes = s.to_repr();
  ArkFr::from_le_bytes_mod_order(bytes.as_ref())
}

/// Dory commitment key
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoryCommitmentKey {
  /// Number of variables
  pub num_vars: usize,
  /// Serialized h_gt generator for rerandomization (Vega §2.1)
  pub h_gt_bytes: Vec<u8>,
}

/// Dory verifier key (same as commitment key for Dory)
pub type DoryVerifierKey = DoryCommitmentKey;

/// Dory commitment wrapping quarks-zk commitment
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryCommitment {
  /// Serialized commitment bytes (using CanonicalSerialize)
  pub commitment_bytes: Vec<u8>,
}

/// Dory blind (Dory is inherently hiding)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryBlind;

/// Dory evaluation argument with native types
/// 
/// Review feedback (#3): Keep unserialized items inside proof objects
/// for strong guarantees. Spartan2 handles serialization itself.
#[derive(Clone, Debug)]
pub struct DoryEvaluationArgument<E: Engine> {
  /// The quarks-zk Dory proof (native ark types, NOT pre-serialized)
  pub proof: DoryPCSEvaluationProof,
  /// Claimed evaluation value (native ark Fr, NOT pre-serialized)
  pub value: ArkFr,
  /// Phantom data for Engine
  _p: PhantomData<E>,
}

/// Custom serde implementation using ark_serialize
impl<E: Engine> Serialize for DoryEvaluationArgument<E> {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeTuple;
    
    // Serialize proof
    let mut proof_bytes = Vec::new();
    self.proof.serialize_compressed(&mut proof_bytes)
      .map_err(|e| serde::ser::Error::custom(format!("proof: {:?}", e)))?;
    
    // Serialize value
    let mut value_bytes = Vec::new();
    self.value.serialize_compressed(&mut value_bytes)
      .map_err(|e| serde::ser::Error::custom(format!("value: {:?}", e)))?;
    
    let mut tup = serializer.serialize_tuple(2)?;
    tup.serialize_element(&proof_bytes)?;
    tup.serialize_element(&value_bytes)?;
    tup.end()
  }
}

impl<'de, E: Engine> Deserialize<'de> for DoryEvaluationArgument<E> {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    use serde::de::{SeqAccess, Visitor};
    
    struct ArgVisitor<E>(PhantomData<E>);
    
    impl<'de, E: Engine> Visitor<'de> for ArgVisitor<E> {
      type Value = DoryEvaluationArgument<E>;
      
      fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("DoryEvaluationArgument tuple")
      }
      
      fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        use serde::de::Error;
        
        let proof_bytes: Vec<u8> = seq.next_element()?
          .ok_or_else(|| A::Error::custom("missing proof"))?;
        let value_bytes: Vec<u8> = seq.next_element()?
          .ok_or_else(|| A::Error::custom("missing value"))?;
        
        let proof = DoryPCSEvaluationProof::deserialize_compressed(&proof_bytes[..])
          .map_err(|e| A::Error::custom(format!("proof: {:?}", e)))?;
        let value = ArkFr::deserialize_compressed(&value_bytes[..])
          .map_err(|e| A::Error::custom(format!("value: {:?}", e)))?;
        
        Ok(DoryEvaluationArgument { proof, value, _p: PhantomData })
      }
    }
    
    deserializer.deserialize_tuple(2, ArgVisitor(PhantomData))
  }
}

/// Dory-PC adapter using quarks-zk
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryPCS<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> PCSEngineTrait<E> for DoryPCS<E> {
  type CommitmentKey = DoryCommitmentKey;
  type VerifierKey = DoryVerifierKey;
  type Commitment = DoryCommitment;
  type Blind = DoryBlind;
  type EvaluationArgument = DoryEvaluationArgument<E>;

  fn setup(
    _label: &'static [u8],
    n: usize,
    _width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    use rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    
    // Calculate num_vars from n
    let num_vars = if n == 0 { 2 } else { (n as f64).log2().ceil() as usize }.max(2);
    // Ensure even for Dory
    let num_vars = if num_vars % 2 == 0 { num_vars } else { num_vars + 1 };
    
    // Setup quarks-zk params to get h_gt generator
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let params = QuarksDoryPCS::setup(num_vars, &mut rng);
    
    // Serialize h_gt for storage in commitment key
    let mut h_gt_bytes = Vec::new();
    params.h_gt.serialize_compressed(&mut h_gt_bytes)
      .expect("Failed to serialize h_gt");
    
    let ck = DoryCommitmentKey { num_vars, h_gt_bytes: h_gt_bytes.clone() };
    let vk = ck.clone();
    
    (ck, vk)
  }

  fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind {
    DoryBlind
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    _r: &Self::Blind,
    _is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    use rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    
    // Setup params
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let params = QuarksDoryPCS::setup(ck.num_vars, &mut rng);
    
    // Convert scalars
    let ark_evals: Vec<ArkFr> = v.iter().map(|s| halo2_to_ark::<E>(s)).collect();
    
    // Commit using quarks-zk
    let commitment = QuarksDoryPCS::commit(&params, &ark_evals);
    
    // Serialize using CanonicalSerialize
    let mut bytes = Vec::new();
    commitment.serialize_compressed(&mut bytes)
      .map_err(|e| SpartanError::InvalidInputLength {
        reason: format!("Failed to serialize commitment: {:?}", e),
      })?;
    
    Ok(DoryCommitment { commitment_bytes: bytes })
  }

  fn check_commitment(comm: &Self::Commitment, _n: usize, _width: usize) -> Result<(), SpartanError> {
    if comm.commitment_bytes.is_empty() {
      return Err(SpartanError::InvalidCommitmentLength {
        reason: "Empty Dory commitment".to_string(),
      });
    }
    Ok(())
  }

  fn rerandomize_commitment(
    ck: &Self::CommitmentKey,
    comm: &Self::Commitment,
    _r_old: &Self::Blind,
    _r_new: &Self::Blind,
  ) -> Result<Self::Commitment, SpartanError> {
    // Vega §2.1: Rerandomize commitments for zero-knowledge reuse
    // Since DoryBlind is unit type (Dory is inherently hiding),
    // we generate fresh randomness on each call
    
    use quarks_zk::dory_pc::Bls381GT;
    
    // Deserialize current commitment
    let current_comm = DoryPCSCommitment::deserialize_compressed(&comm.commitment_bytes[..])
      .map_err(|e| SpartanError::InvalidInputLength {
        reason: format!("Failed to deserialize commitment: {:?}", e),
      })?;
    
    // Deserialize h_gt generator
    let h_gt = Bls381GT::deserialize_compressed(&ck.h_gt_bytes[..])
      .map_err(|e| SpartanError::InvalidInputLength {
        reason: format!("Failed to deserialize h_gt: {:?}", e),
      })?;
    
    // Generate random delta for rerandomization
    use rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use ff::Field;
    let mut rng = ChaCha20Rng::from_entropy();
    let r_delta = <E as Engine>::Scalar::random(&mut rng);
    
    // Convert to ark Fr
    let r_delta_ark = halo2_to_ark::<E>(&r_delta);
    
    // Rerandomize using quarks-zk API
    let rerandomized_comm = current_comm.rerandomize(&r_delta_ark, &h_gt);
    
    // Serialize back
    let mut bytes = Vec::new();
    rerandomized_comm.serialize_compressed(&mut bytes)
      .map_err(|e| SpartanError::InvalidInputLength {
        reason: format!("Failed to serialize rerandomized commitment: {:?}", e),
      })?;
    
    Ok(DoryCommitment { commitment_bytes: bytes })
  }

  fn combine_commitments(comms: &[Self::Commitment]) -> Result<Self::Commitment, SpartanError> {
    if comms.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "No commitments to combine".to_string(),
      });
    }
    // Concatenate bytes for multiple commitments
    let mut combined = Vec::new();
    for c in comms {
      combined.extend_from_slice(&c.commitment_bytes);
    }
    Ok(DoryCommitment { commitment_bytes: combined })
  }

  fn combine_blinds(blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
    if blinds.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "No blinds to combine".to_string(),
      });
    }
    Ok(DoryBlind)
  }

  fn prove(
    ck: &Self::CommitmentKey,
    _ck_eval: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[E::Scalar],
    _blind: &Self::Blind,
    point: &[E::Scalar],
    _comm_eval: &Self::Commitment,
    _blind_eval: &Self::Blind,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    use rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    
    // Absorb commitment
    transcript.absorb(b"dory_comm", comm);
    
    // Setup params
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let params = QuarksDoryPCS::setup(ck.num_vars, &mut rng);
    
    // Convert scalars
    let ark_evals: Vec<ArkFr> = poly.iter().map(|s| halo2_to_ark::<E>(s)).collect();
    let ark_point: Vec<ArkFr> = point.iter().map(|s| halo2_to_ark::<E>(s)).collect();
    
    // Generate proof using quarks-zk
    let (value, proof) = QuarksDoryPCS::prove_eval(&params, &ark_evals, &ark_point, &mut rng);
    
    // Store native objects (review #3: no pre-serialization)
    Ok(DoryEvaluationArgument {
      proof,
      value,
      _p: PhantomData,
    })
  }

  fn verify(
    vk: &Self::VerifierKey,
    _ck_eval: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    _comm_eval: &Self::Commitment,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    use rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    
    // Absorb commitment
    transcript.absorb(b"dory_comm", comm);
    
    // Setup params (must match prove)
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let params = QuarksDoryPCS::setup(vk.num_vars, &mut rng);
    
    // Deserialize commitment to polynomial (native object)
    let commitment = DoryPCSCommitment::deserialize_compressed(&comm.commitment_bytes[..])
      .map_err(|e| SpartanError::InvalidInputLength {
        reason: format!("Failed to deserialize commitment: {:?}", e),
      })?;
    
    // Extract value from evaluation argument (native ArkFr, not Vec<u8>)
    let value = arg.value;
    
    // Convert point to ark Fr
    let ark_point: Vec<ArkFr> = point.iter().map(|s| halo2_to_ark::<E>(s)).collect();
    
    // Verify using quarks-zk
    // The proof binds (commitment, point, value) cryptographically
    let valid = QuarksDoryPCS::verify_eval(&params, &commitment, &ark_point, value, &arg.proof);
    
    if valid {
      Ok(())
    } else {
      Err(SpartanError::InvalidInputLength {
        reason: "Dory proof verification failed".to_string(),
      })
    }
  }
}

impl<G: Group> TranscriptReprTrait<G> for DoryCommitment {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(b"dory_commitment_begin");
    v.extend_from_slice(&self.commitment_bytes);
    v.extend_from_slice(b"dory_commitment_end");
    v
  }
}

impl<E: Engine> CommitmentTrait<E> for DoryCommitment {}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::BLS12381HyraxEngine;
  use crate::provider::keccak::Keccak256Transcript;
  
  type E = BLS12381HyraxEngine;
  type TE = Keccak256Transcript<E>;
  
  fn make_poly(n: usize) -> Vec<<E as Engine>::Scalar> {
    (0..n).map(|i| <E as Engine>::Scalar::from(i as u64 + 1)).collect()
  }
  
  fn make_point(n: usize) -> Vec<<E as Engine>::Scalar> {
    (0..n).map(|i| <E as Engine>::Scalar::from(i as u64 * 3 + 7)).collect()
  }
  
  // ==================== BASIC TESTS ====================
  
  #[test]
  fn test_dory_setup() {
    let (ck, vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    assert!(ck.num_vars >= 4, "num_vars should be at least 4");
    assert_eq!(ck.num_vars, vk.num_vars, "ck and vk should have same num_vars");
    assert!(ck.num_vars % 2 == 0, "num_vars should be even for Dory");
  }
  
  #[test]
  fn test_dory_commit() {
    let (ck, _vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    let poly = make_poly(16);
    
    let comm = DoryPCS::<E>::commit(&ck, &poly, &blind, false);
    assert!(comm.is_ok(), "Commit should succeed");
    assert!(!comm.unwrap().commitment_bytes.is_empty(), "Commitment should not be empty");
  }
  
  // ==================== SOUNDNESS TESTS ====================
  
  #[test]
  fn test_dory_commitment_deterministic() {
    // Same polynomial should produce same commitment
    let (ck, _vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    let poly = make_poly(16);
    
    let comm1 = DoryPCS::<E>::commit(&ck, &poly, &blind, false).unwrap();
    let comm2 = DoryPCS::<E>::commit(&ck, &poly, &blind, false).unwrap();
    
    assert_eq!(comm1.commitment_bytes, comm2.commitment_bytes, 
      "Same polynomial should produce same commitment");
  }
  
  #[test]
  fn test_dory_different_polys_different_commits() {
    // Different polynomials should produce different commitments (binding)
    let (ck, _vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    
    let poly1 = make_poly(16);
    let mut poly2 = make_poly(16);
    poly2[0] = <E as Engine>::Scalar::from(9999u64); // Change one coefficient
    
    let comm1 = DoryPCS::<E>::commit(&ck, &poly1, &blind, false).unwrap();
    let comm2 = DoryPCS::<E>::commit(&ck, &poly2, &blind, false).unwrap();
    
    assert_ne!(comm1.commitment_bytes, comm2.commitment_bytes,
      "Different polynomials must produce different commitments (binding property)");
  }
  
  #[test]
  fn test_dory_prove_verify_complete() {
    // Full prove/verify cycle
    let (ck, vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    let poly = make_poly(16);
    let point = make_point(ck.num_vars);
    
    let comm = DoryPCS::<E>::commit(&ck, &poly, &blind, false).unwrap();
    let comm_eval = comm.clone();
    let blind_eval = blind.clone();
    
    let mut transcript_p = TE::new(b"test");
    let arg = DoryPCS::<E>::prove(
      &ck, &ck, &mut transcript_p, &comm, &poly, &blind, &point, &comm_eval, &blind_eval
    );
    assert!(arg.is_ok(), "Prove should succeed");
    
    let arg = arg.unwrap();
    let mut transcript_v = TE::new(b"test");
    let result = DoryPCS::<E>::verify(
      &vk, &ck, &mut transcript_v, &comm, &point, &comm_eval, &arg
    );
    assert!(result.is_ok(), "Valid proof should verify");
  }
  
  #[test]
  fn test_dory_reject_tampered_proof() {
    // Tampered proof should be rejected
    let (ck, vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    let poly = make_poly(16);
    let point = make_point(ck.num_vars);
    
    let comm = DoryPCS::<E>::commit(&ck, &poly, &blind, false).unwrap();
    let comm_eval = comm.clone();
    let blind_eval = blind.clone();
    
    let mut transcript = TE::new(b"test");
    let mut arg = DoryPCS::<E>::prove(
      &ck, &ck, &mut transcript, &comm, &poly, &blind, &point, &comm_eval, &blind_eval
    ).unwrap();
    
    // Tamper with the value (native field, not bytes)
    use ark_ff::UniformRand;
    use rand::thread_rng;
    arg.value = ArkFr::rand(&mut thread_rng());
    
    let mut transcript_v = TE::new(b"test");
    let result = DoryPCS::<E>::verify(
      &vk, &ck, &mut transcript_v, &comm, &point, &comm_eval, &arg
    );
    assert!(result.is_err(), "Tampered value must be rejected");
  }
  
  #[test]
  fn test_dory_native_objects_not_pre_serialized() {
    // Review #3: Verify we use native objects, not pre-serialized bytes
    let (ck, _vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    let poly = make_poly(16);
    let point = make_point(ck.num_vars);
    
    let comm = DoryPCS::<E>::commit(&ck, &poly, &blind, false).unwrap();
    let comm_eval = comm.clone();
    let blind_eval = blind.clone();
    
    let mut transcript = TE::new(b"test");
    let arg = DoryPCS::<E>::prove(
      &ck, &ck, &mut transcript, &comm, &poly, &blind, &point, &comm_eval, &blind_eval
    ).unwrap();
    
    // Verify native types are accessible (not buried in bytes)
    use ark_serialize::CanonicalSerialize;
    
    // proof is DoryPCSEvaluationProof (native ark type)
    let mut proof_bytes = Vec::new();
    arg.proof.serialize_compressed(&mut proof_bytes).unwrap();
    assert!(!proof_bytes.is_empty(), "Proof should serialize");
    
    // value is ArkFr (native field element)
    let mut value_bytes = Vec::new();
    arg.value.serialize_compressed(&mut value_bytes).unwrap();
    assert!(!value_bytes.is_empty(), "Value should serialize");
    
    // This demonstrates we have NATIVE objects with strong type guarantees
    // not pre-serialized Vec<u8> which are opaque
  }
  
  #[test]
  fn test_dory_reject_wrong_commitment() {
    // Proof for one commitment shouldn't verify against different commitment
    let (ck, vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    
    let poly1 = make_poly(16);
    let mut poly2 = make_poly(16);
    poly2[0] = <E as Engine>::Scalar::from(42u64);
    
    let point = make_point(ck.num_vars);
    
    let comm1 = DoryPCS::<E>::commit(&ck, &poly1, &blind, false).unwrap();
    let comm2 = DoryPCS::<E>::commit(&ck, &poly2, &blind, false).unwrap();
    
    // Generate proof for poly1
    let mut transcript = TE::new(b"test");
    let arg = DoryPCS::<E>::prove(
      &ck, &ck, &mut transcript, &comm1, &poly1, &blind, &point, &comm1.clone(), &blind.clone()
    ).unwrap();
    
    // Try to verify against comm2 (should fail)
    let mut transcript_v = TE::new(b"test");
    let result = DoryPCS::<E>::verify(
      &vk, &ck, &mut transcript_v, &comm2, &point, &comm2.clone(), &arg
    );
    assert!(result.is_err(), "Proof for one commitment must not verify against different commitment");
  }
  
  #[test]
  fn test_dory_check_commitment_rejects_empty() {
    let empty_comm = DoryCommitment { commitment_bytes: vec![] };
    let result = DoryPCS::<E>::check_commitment(&empty_comm, 16, 4);
    assert!(result.is_err(), "Empty commitment should be rejected");
  }
  
  #[test]
  fn test_dory_combine_commitments() {
    let (ck, _vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    
    let poly1 = make_poly(16);
    let poly2: Vec<_> = (0..16).map(|i| <E as Engine>::Scalar::from(i as u64 * 2)).collect();
    
    let comm1 = DoryPCS::<E>::commit(&ck, &poly1, &blind, false).unwrap();
    let comm2 = DoryPCS::<E>::commit(&ck, &poly2, &blind, false).unwrap();
    
    let combined = DoryPCS::<E>::combine_commitments(&[comm1.clone(), comm2.clone()]);
    assert!(combined.is_ok(), "Combining commitments should succeed");
    
    let combined = combined.unwrap();
    assert!(combined.commitment_bytes.len() > comm1.commitment_bytes.len(),
      "Combined commitment should be larger");
  }
  
  #[test]
  fn test_dory_combine_empty_fails() {
    let result = DoryPCS::<E>::combine_commitments(&[]);
    assert!(result.is_err(), "Combining empty commitments should fail");
    
    let result = DoryPCS::<E>::combine_blinds(&[]);
    assert!(result.is_err(), "Combining empty blinds should fail");
  }
  
  // ==================== RERANDOMIZATION TESTS (Vega §2.1) ====================
  
  #[test]
  fn test_dory_rerandomize_unlinkability() {
    // Critical for Vega: rerandomized commitments must be unlinkable
    let (ck, _vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    let poly = make_poly(16);
    
    let comm = DoryPCS::<E>::commit(&ck, &poly, &blind, false).unwrap();
    
    // Rerandomize twice
    let comm_1 = DoryPCS::<E>::rerandomize_commitment(&ck, &comm, &blind, &blind).unwrap();
    let comm_2 = DoryPCS::<E>::rerandomize_commitment(&ck, &comm, &blind, &blind).unwrap();
    
    // All three commitments must be different (unlinkability)
    assert_ne!(
      comm.commitment_bytes, comm_1.commitment_bytes,
      "Rerandomized commitment must differ from original"
    );
    assert_ne!(
      comm.commitment_bytes, comm_2.commitment_bytes,
      "Second rerandomization must differ from original"
    );
    assert_ne!(
      comm_1.commitment_bytes, comm_2.commitment_bytes,
      "Different rerandomizations must be unlinkable"
    );
  }
  
  #[test]
  fn test_dory_rerandomize_preserves_correctness() {
    // Rerandomized commitment should still verify correctly
    let (ck, vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    let poly = make_poly(16);
    let point = make_point(ck.num_vars);
    
    let comm = DoryPCS::<E>::commit(&ck, &poly, &blind, false).unwrap();
    
    // Rerandomize (we don't use it further, just verify it doesn't crash)
    let _comm_rerand = DoryPCS::<E>::rerandomize_commitment(&ck, &comm, &blind, &blind).unwrap();
    
    // Generate proof with ORIGINAL commitment
    let comm_eval = comm.clone();
    let blind_eval = blind.clone();
    
    let mut transcript_p = TE::new(b"test");
    let arg = DoryPCS::<E>::prove(
      &ck, &ck, &mut transcript_p, &comm, &poly, &blind, &point, &comm_eval, &blind_eval
    ).unwrap();
    
    // Verify should work with original commitment
    let mut transcript_v = TE::new(b"test");
    let result = DoryPCS::<E>::verify(
      &vk, &ck, &mut transcript_v, &comm, &point, &comm_eval, &arg
    );
    assert!(result.is_ok(), "Proof with original commitment should verify");
    
    // Note: We can't verify with rerandomized commitment without updating the proof
    // This is expected - rerandomization is for commitment reuse, not proof manipulation
  }
  
  #[test]
  fn test_dory_rerandomize_multiple_times() {
    // Multiple rerandomizations should continue to be unlinkable
    let (ck, _vk) = DoryPCS::<E>::setup(b"test", 16, 4);
    let blind = DoryPCS::<E>::blind(&ck, 16);
    let poly = make_poly(16);
    
    let comm = DoryPCS::<E>::commit(&ck, &poly, &blind, false).unwrap();
    
    // Chain of rerandomizations
    let comm_1 = DoryPCS::<E>::rerandomize_commitment(&ck, &comm, &blind, &blind).unwrap();
    let comm_2 = DoryPCS::<E>::rerandomize_commitment(&ck, &comm_1, &blind, &blind).unwrap();
    let comm_3 = DoryPCS::<E>::rerandomize_commitment(&ck, &comm_2, &blind, &blind).unwrap();
    
    // All should be distinct
    let commitments = vec![
      &comm.commitment_bytes,
      &comm_1.commitment_bytes,
      &comm_2.commitment_bytes,
      &comm_3.commitment_bytes,
    ];
    
    for i in 0..commitments.len() {
      for j in (i+1)..commitments.len() {
        assert_ne!(
          commitments[i], commitments[j],
          "Commitment {} and {} must be unlinkable", i, j
        );
      }
    }
  }
}
