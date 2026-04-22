// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
//
// Hybrid Inner Product Argument (IPA): k rounds of Bulletproofs-style halving
// followed by a Schnorr vector tail proof. Balances proof size and prover speed.
//
// For n-element vectors with k bullet rounds, proof size is 2k group elements
// + n/2^k scalars + O(1), vs O(n) scalars for the linear IPA.
//
// Protocol:
// 1. Equality proof connecting h_c-based evaluation commitment to h-based one
// 2. Combine witness and evaluation commitments via random challenge r
// 3. k rounds of Bulletproofs halving reduction
// 4. Schnorr vector proof on the reduced witness

#![allow(non_snake_case)]

use crate::{
  errors::SpartanError,
  provider::traits::{DlogGroup, DlogGroupExt},
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use core::fmt::Debug;
use ff::Field;
use serde::{Deserialize, Serialize};

use super::ipa::{InnerProductInstance, InnerProductWitness, inner_product};

/// Bulletproofs-style halving proof for inner product arguments.
///
/// In each of log(n) rounds, the prover sends L and R commitments that allow
/// the verifier to fold the generators, witness, and public vectors in half.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BulletReductionProof<E: Engine>
where
  E::GE: DlogGroupExt,
{
  L_vec: Vec<E::GE>,
  R_vec: Vec<E::GE>,
}

impl<E: Engine> BulletReductionProof<E>
where
  E::GE: DlogGroupExt,
{
  /// Prove a bullet reduction (num_rounds out of lg_n rounds) using the no-generator-fold
  /// technique. Returns (proof, a_hat_vec, b_hat_vec, c_coeffs, blind_hat).
  #[allow(clippy::too_many_arguments)]
  pub fn prove(
    transcript: &mut E::TE,
    Q: &E::GE,
    G_affine: &[<E::GE as DlogGroup>::AffineGroupElement],
    H: &E::GE,
    a_vec: &[E::Scalar],
    b_vec: &[E::Scalar],
    blind: &E::Scalar,
    blinds_vec: &[(E::Scalar, E::Scalar)],
    num_rounds: usize,
  ) -> Result<
    (
      Self,
      Vec<E::Scalar>,
      Vec<E::Scalar>,
      Vec<E::Scalar>,
      E::Scalar,
    ),
    SpartanError,
  > {
    let n_orig = a_vec.len();
    if !n_orig.is_power_of_two() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("bullet prove: n={n_orig} is not a power of two"),
      });
    }
    let lg_n = n_orig.trailing_zeros() as usize;
    if num_rounds > lg_n {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("bullet prove: num_rounds={num_rounds} exceeds lg_n={lg_n}"),
      });
    }
    if G_affine.len() != n_orig {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "bullet prove: G_affine has {} elements but need {n_orig}",
          G_affine.len()
        ),
      });
    }
    if b_vec.len() != n_orig {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "bullet prove: b_vec has {} elements but need {n_orig}",
          b_vec.len()
        ),
      });
    }
    if blinds_vec.len() < num_rounds {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "bullet prove: blinds_vec has {} elements but need at least {num_rounds}",
          blinds_vec.len()
        ),
      });
    }

    let mut a = a_vec.to_vec();
    let mut b = b_vec.to_vec();
    let mut c = vec![E::Scalar::ONE; n_orig];

    let mut L_vec = Vec::with_capacity(num_rounds);
    let mut R_vec = Vec::with_capacity(num_rounds);
    let mut blind_final = *blind;

    let mut l_scalars = vec![E::Scalar::ZERO; n_orig];
    let mut r_scalars = vec![E::Scalar::ZERO; n_orig];

    for (round, (blind_L, blind_R)) in blinds_vec.iter().enumerate().take(num_rounds) {
      let n_curr = n_orig >> round;
      let half = n_curr / 2;
      let (a_L, a_R) = a.split_at(half);
      let (b_L, b_R) = b.split_at(half);

      let c_L = inner_product(a_L, b_R);
      let c_R = inner_product(a_R, b_L);

      let bit_mask = 1usize << (lg_n - 1 - round);

      for j in 0..n_orig {
        if j & bit_mask != 0 {
          let folded_right_idx = (j % n_curr) - half;
          l_scalars[j] = a_L[folded_right_idx] * c[j];
          r_scalars[j] = E::Scalar::ZERO;
        } else {
          let folded_left_idx = j % n_curr;
          r_scalars[j] = a_R[folded_left_idx] * c[j];
          l_scalars[j] = E::Scalar::ZERO;
        }
      }

      let L =
        E::GE::vartime_multiscalar_mul(&l_scalars, G_affine, true)? + *Q * c_L + *H * *blind_L;
      let R =
        E::GE::vartime_multiscalar_mul(&r_scalars, G_affine, true)? + *Q * c_R + *H * *blind_R;

      transcript.absorb(b"L", &L);
      transcript.absorb(b"R", &R);
      let u: E::Scalar = transcript.squeeze(b"u")?;
      let u_inv: Option<E::Scalar> = u.invert().into();
      let u_inv = u_inv.ok_or(SpartanError::DivisionByZero)?;

      for (j, c_j) in c.iter_mut().enumerate().take(n_orig) {
        if j & bit_mask == 0 {
          *c_j *= u_inv;
        } else {
          *c_j *= u;
        }
      }

      let a_new: Vec<E::Scalar> = (0..half).map(|i| a_L[i] * u + a_R[i] * u_inv).collect();
      let b_new: Vec<E::Scalar> = (0..half).map(|i| b_L[i] * u_inv + b_R[i] * u).collect();
      blind_final = blind_final + *blind_L * u.square() + *blind_R * u_inv.square();

      L_vec.push(L);
      R_vec.push(R);
      a = a_new;
      b = b_new;
    }

    Ok((
      BulletReductionProof { L_vec, R_vec },
      a, // reduced witness vector (length n/2^k)
      b, // reduced public vector (length n/2^k)
      c, // accumulated generator coefficients (length n)
      blind_final,
    ))
  }

  /// Verify the bullet reduction and reconstruct the reduced state.
  ///
  /// Returns (s_values, Gamma_hat, b_hat_vec) where
  /// s_values are the verification scalars for reconstructing reduced generators.
  pub fn verify(
    &self,
    n: usize,
    b: &[E::Scalar],
    transcript: &mut E::TE,
    Gamma: &E::GE,
  ) -> Result<(Vec<E::Scalar>, E::GE, Vec<E::Scalar>), SpartanError> {
    let k = self.L_vec.len();
    if self.R_vec.len() != k {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "bullet verify: L_vec has {k} elements but R_vec has {}",
          self.R_vec.len()
        ),
      });
    }
    let lg_n = n.trailing_zeros() as usize;
    if k > lg_n || !n.is_power_of_two() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "bullet verify: k={} exceeds lg_n={} or n={} is not a power of two",
          k, lg_n, n
        ),
      });
    }
    if b.len() != n {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("bullet verify: b has {} elements but need {n}", b.len()),
      });
    }
    let n_reduced = n >> k;

    // Recompute challenges from transcript
    let mut challenges = Vec::with_capacity(k);
    for i in 0..k {
      transcript.absorb(b"L", &self.L_vec[i]);
      transcript.absorb(b"R", &self.R_vec[i]);
      challenges.push(transcript.squeeze(b"u")?);
    }

    let challenges_inv: Vec<E::Scalar> = challenges
      .iter()
      .map(|u| {
        let inv: Option<E::Scalar> = u.invert().into();
        inv.ok_or(SpartanError::DivisionByZero)
      })
      .collect::<Result<Vec<_>, _>>()?;
    let challenges_sq: Vec<E::Scalar> = challenges.iter().map(|u| u.square()).collect();
    let challenges_inv_sq: Vec<E::Scalar> = challenges_inv.iter().map(|u| u.square()).collect();

    // Compute s values for the k rounds (length n, grouped into n_reduced groups of 2^k)
    // s[j] = product of u/u_inv based on bits of j corresponding to the k rounds
    let mut s = vec![E::Scalar::ONE; n];
    for round in 0..k {
      let bit_mask = 1usize << (lg_n - 1 - round);
      let u = challenges[round];
      let u_inv = challenges_inv[round];
      for (j, s_j) in s.iter_mut().enumerate().take(n) {
        if j & bit_mask == 0 {
          *s_j *= u_inv;
        } else {
          *s_j *= u;
        }
      }
    }

    // Gamma_hat = Gamma + sum(u_sq[i] * L[i] + u_inv_sq[i] * R[i])
    let mut Gamma_hat = *Gamma;
    for i in 0..k {
      Gamma_hat =
        Gamma_hat + self.L_vec[i] * challenges_sq[i] + self.R_vec[i] * challenges_inv_sq[i];
    }

    // Fold b vector through k rounds
    let mut b_folded = b.to_vec();
    for round in 0..k {
      let half = b_folded.len() / 2;
      let u = challenges[round];
      let u_inv = challenges_inv[round];
      let (b_L, b_R) = b_folded.split_at(half);
      b_folded = (0..half).map(|i| b_L[i] * u_inv + b_R[i] * u).collect();
    }
    if b_folded.len() != n_reduced {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "bullet verify: b_folded has {} elements, expected {n_reduced}",
          b_folded.len()
        ),
      });
    }

    Ok((s, Gamma_hat, b_folded))
  }
}

/// Hybrid IPA: k rounds of bullet reduction followed by a Schnorr vector proof.
///
/// Balances proof size and prover speed: k bullet rounds compress the proof
/// while the Schnorr tail keeps prover time manageable.
///
/// Proof size: 2k group elements (L/R) + n/2^k scalars (z_vec) + 2 points + 3 scalars.
/// Prover/verifier time: k rounds of MSM + O(n) tail MSM (expanded back to original generators).
///
/// The number of bullet rounds is controlled by `BULLET_ROUNDS`.
///
/// Protocol:
/// 1. Equality proof connecting h_c-based evaluation commitment to h-based one
/// 2. Combine commitments: Gamma = comm_a + r * comm_v
/// 3. k rounds of bullet reduction -> Gamma_hat, reduced generators g_hat, reduced b_hat
/// 4. Schnorr proof: prove knowledge of (a_hat, blind_hat) s.t.
///    Gamma_hat = MSM(a_hat, G_combined) + blind_hat * h
///    where G_combined[i] = g_hat[i] + b_hat[i] * Q
///    This simultaneously proves the commitment and inner product claims.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InnerProductArgumentHybrid<E: Engine, const BULLET_ROUNDS: usize>
where
  E::GE: DlogGroupExt,
{
  bullet: BulletReductionProof<E>,
  comm_v: E::GE,
  T_eq: E::GE,
  s1_eq: E::Scalar,
  s2_eq: E::Scalar,
  // Schnorr tail
  D: E::GE,
  z_vec: Vec<E::Scalar>,
  z_r: E::Scalar,
}

impl<E: Engine, const BULLET_ROUNDS: usize> InnerProductArgumentHybrid<E, BULLET_ROUNDS>
where
  E::GE: DlogGroupExt,
{
  fn protocol_name() -> &'static [u8] {
    b"inner product argument (hybrid)"
  }

  /// Proves the hybrid inner product argument.
  pub fn prove(
    ck: &[<E::GE as DlogGroup>::AffineGroupElement],
    h: &E::GE,
    ck_c: &<E::GE as DlogGroup>::AffineGroupElement,
    h_c: &E::GE,
    U: &InnerProductInstance<E>,
    W: &InnerProductWitness<E>,
    transcript: &mut E::TE,
  ) -> Result<Self, SpartanError> {
    transcript.dom_sep(Self::protocol_name());
    transcript.absorb(b"U", U);

    let n = W.a_vec.len();
    if !n.is_power_of_two() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("hybrid IPA prove: n={n} is not a power of two"),
      });
    }
    if ck.len() < n {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "hybrid IPA prove: ck has {} elements but need at least {n}",
          ck.len()
        ),
      });
    }
    let ck = &ck[..n];
    let lg_n = n.trailing_zeros() as usize;
    let k = BULLET_ROUNDS.min(lg_n);

    let mut rng = rand::thread_rng();

    if U.b_vec.len() != n {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "hybrid IPA prove: b_vec has {} elements but need {n}",
          U.b_vec.len()
        ),
      });
    }

    // Step 1: Re-commit evaluation under h
    let v = inner_product(&W.a_vec, &U.b_vec);
    let r_v = E::Scalar::random(&mut rng);
    let comm_v = E::GE::group(ck_c) * v + *h * r_v;
    transcript.absorb(b"comm_v", &comm_v);

    // Step 2: Equality proof (proves comm_v and comm_c commit to same value)
    let t1 = E::Scalar::random(&mut rng);
    let t2 = E::Scalar::random(&mut rng);
    let T_eq = *h * t1 - *h_c * t2;
    transcript.absorb(b"T_eq", &T_eq);
    let e: E::Scalar = transcript.squeeze(b"e_eq")?;
    let s1_eq = t1 + e * r_v;
    let s2_eq = t2 + e * W.r_c;

    // Step 3: Combine commitments: Gamma = comm_a + r * comm_v
    let r: E::Scalar = transcript.squeeze(b"r")?;
    let Q = E::GE::group(ck_c) * r;
    let blind_Gamma = W.r_a + r * r_v;

    // Step 4: k rounds of bullet reduction
    let blinds_vec: Vec<(E::Scalar, E::Scalar)> = (0..k)
      .map(|_| (E::Scalar::random(&mut rng), E::Scalar::random(&mut rng)))
      .collect();

    let (bullet, a_hat, b_hat, c_coeffs, blind_hat) = BulletReductionProof::prove(
      transcript,
      &Q,
      ck,
      h,
      &W.a_vec,
      &U.b_vec,
      &blind_Gamma,
      &blinds_vec,
      k,
    )?;

    // Step 5: Schnorr tail without materializing g_hat.
    // Instead of computing g_hat[i] = sum_m c_coeffs[j]*G[j] and G_combined, we expand
    // the Schnorr nonce vector back to original generators:
    //   MSM(d_vec, G_combined) = MSM(expanded_d, G) + <d_vec, b_hat> * Q
    // where expanded_d[j] = d_vec[j % n_reduced] * c_coeffs[j]
    let n_reduced = n >> k;
    let d_vec: Vec<E::Scalar> = (0..n_reduced)
      .map(|_| E::Scalar::random(&mut rng))
      .collect();
    let r_d = E::Scalar::random(&mut rng);

    let expanded_d: Vec<E::Scalar> = (0..n).map(|j| d_vec[j % n_reduced] * c_coeffs[j]).collect();
    let ip_d_b = inner_product(&d_vec, &b_hat);

    let D = E::GE::vartime_multiscalar_mul(&expanded_d, ck, true)? + Q * ip_d_b + *h * r_d;
    transcript.absorb(b"D", &D);

    let c_chal: E::Scalar = transcript.squeeze(b"c_schnorr")?;

    let z_vec: Vec<E::Scalar> = d_vec
      .iter()
      .zip(a_hat.iter())
      .map(|(d, a)| *d + c_chal * *a)
      .collect();
    let z_r = r_d + c_chal * blind_hat;

    Ok(Self {
      bullet,
      comm_v,
      T_eq,
      s1_eq,
      s2_eq,
      D,
      z_vec,
      z_r,
    })
  }

  /// Verifies the hybrid inner product argument.
  pub fn verify(
    &self,
    ck: &[<E::GE as DlogGroup>::AffineGroupElement],
    h: &E::GE,
    ck_c: &<E::GE as DlogGroup>::AffineGroupElement,
    h_c: &E::GE,
    n: usize,
    U: &InnerProductInstance<E>,
    transcript: &mut E::TE,
  ) -> Result<(), SpartanError> {
    transcript.dom_sep(Self::protocol_name());
    transcript.absorb(b"U", U);

    let k = self.bullet.L_vec.len();

    if n == 0 || !n.is_power_of_two() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("Hybrid IPA verify: n must be a non-zero power of two, got {n}"),
      });
    }
    let lg_n = n.ilog2() as usize;
    let expected_k = BULLET_ROUNDS.min(lg_n);
    if k != expected_k {
      return Err(SpartanError::InvalidPCS {
        reason: format!("Hybrid IPA verify: proof has {k} bullet rounds, expected {expected_k}"),
      });
    }

    if ck.len() < n {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "Hybrid IPA verify: ck has {} elements but need {}",
          ck.len(),
          n
        ),
      });
    }
    let ck = &ck[..n];

    // Step 1: Verify equality proof
    transcript.absorb(b"comm_v", &self.comm_v);
    transcript.absorb(b"T_eq", &self.T_eq);
    let e: E::Scalar = transcript.squeeze(b"e_eq")?;

    let lhs_eq = *h * self.s1_eq - *h_c * self.s2_eq;
    let rhs_eq = self.T_eq + (self.comm_v - U.comm_c) * e;
    if lhs_eq != rhs_eq {
      return Err(SpartanError::InvalidPCS {
        reason: "Hybrid IPA: equality proof failed".to_string(),
      });
    }

    // Step 2: Combine commitments
    let r: E::Scalar = transcript.squeeze(b"r")?;
    let Gamma = U.comm_a_vec + self.comm_v * r;
    let Q = E::GE::group(ck_c) * r;

    // Step 3: Verify bullet reduction -> (s_values, Gamma_hat, b_hat)
    // bullet.verify validates k <= lg_n, so n_reduced is safe to compute after it succeeds.
    let (s_values, Gamma_hat, b_hat) = self.bullet.verify(n, &U.b_vec, transcript, &Gamma)?;
    let n_reduced = n >> k;

    // Step 4: Schnorr verification without materializing g_hat.
    // Expand z_vec back to original generators:
    //   MSM(z_vec, G_combined) = MSM(expanded_z, G) + <z_vec, b_hat> * Q
    if self.z_vec.len() != n_reduced {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "Hybrid IPA: z_vec has {} elements, expected {}",
          self.z_vec.len(),
          n_reduced
        ),
      });
    }
    let expanded_z: Vec<E::Scalar> = (0..n)
      .map(|j| self.z_vec[j % n_reduced] * s_values[j])
      .collect();
    let ip_z_b = inner_product(&self.z_vec, &b_hat);

    // Step 5: Verify Schnorr tail
    // Check: MSM(z_vec, G_combined) + z_r * h == D + c * Gamma_hat
    transcript.absorb(b"D", &self.D);
    let c_chal: E::Scalar = transcript.squeeze(b"c_schnorr")?;

    let lhs = E::GE::vartime_multiscalar_mul(&expanded_z, ck, true)? + Q * ip_z_b + *h * self.z_r;
    let rhs = self.D + Gamma_hat * c_chal;

    if lhs != rhs {
      return Err(SpartanError::InvalidPCS {
        reason: "Hybrid IPA: Schnorr tail verification failed".to_string(),
      });
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::traits::DlogGroup;
  use ff::Field;
  use rand::thread_rng;

  use crate::provider::T256HyraxEngine;
  type E = T256HyraxEngine;
  type Scalar = <E as Engine>::Scalar;
  type GE = <E as Engine>::GE;
  type TE = <E as Engine>::TE;

  fn test_hybrid_ipa_with_engine<E: Engine, const K: usize>()
  where
    E::GE: DlogGroupExt,
  {
    let mut rng = thread_rng();

    for log_n in [4, 8, 11] {
      if K > log_n {
        continue;
      }
      let n = 1 << log_n;

      let gens = E::GE::from_label(b"test_ipa_hybrid", n + 1);
      let ck = &gens[..n];
      let h = E::GE::group(&gens[n]);

      let gens_eval = E::GE::from_label(b"test_ipa_hybrid_eval", 2);
      let ck_c = &gens_eval[0];
      let h_c = E::GE::group(&gens_eval[1]);

      let a_vec: Vec<E::Scalar> = (0..n).map(|_| E::Scalar::random(&mut rng)).collect();
      let b_vec: Vec<E::Scalar> = (0..n).map(|_| E::Scalar::random(&mut rng)).collect();
      let v = inner_product(&a_vec, &b_vec);

      let r_a = E::Scalar::random(&mut rng);
      let comm_a = E::GE::vartime_multiscalar_mul(&a_vec, ck, true).unwrap() + h * r_a;
      let r_c = E::Scalar::random(&mut rng);
      let comm_c = E::GE::group(ck_c) * v + h_c * r_c;

      let instance = InnerProductInstance::<E>::new(&comm_a, &b_vec, &comm_c);
      let witness = InnerProductWitness::<E>::new(&a_vec, &r_a, &r_c);

      let mut pt = E::TE::new(b"test_hybrid_ipa");
      let proof =
        InnerProductArgumentHybrid::<E, K>::prove(ck, &h, ck_c, &h_c, &instance, &witness, &mut pt)
          .unwrap();

      let proof_bytes = bincode::serialize(&proof).unwrap();
      let n_reduced = n >> K;
      tracing::debug!(
        "Hybrid IPA k={} n={}: proof={} bytes, z_vec_len={}, expected_reduced={}",
        K,
        n,
        proof_bytes.len(),
        proof.z_vec.len(),
        n_reduced,
      );

      let mut vt = E::TE::new(b"test_hybrid_ipa");
      let result = proof.verify(ck, &h, ck_c, &h_c, n, &instance, &mut vt);
      assert!(
        result.is_ok(),
        "Hybrid IPA k={} verify failed for n={}: {:?}",
        K,
        n,
        result
      );
    }
  }

  #[test]
  fn test_hybrid_ipa_k2_t256() {
    test_hybrid_ipa_with_engine::<E, 2>();
  }

  #[test]
  fn test_hybrid_ipa_k3_t256() {
    test_hybrid_ipa_with_engine::<E, 3>();
  }

  /// Verify that prove/verify both work when ck is longer than n (Hyrax scenario).
  #[test]
  fn test_hybrid_ipa_oversized_ck() {
    let mut rng = thread_rng();
    const K: usize = 2;
    let n = 16;
    let extra = 8;

    let gens = GE::from_label(b"test_ipa_hybrid", n + extra + 1);
    let ck = &gens[..n + extra]; // deliberately longer than n
    let h = GE::group(&gens[n + extra]);

    let gens_eval = GE::from_label(b"test_ipa_hybrid_eval", 2);
    let ck_c = &gens_eval[0];
    let h_c = GE::group(&gens_eval[1]);

    let a_vec: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let b_vec: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let v = inner_product(&a_vec, &b_vec);

    let r_a = Scalar::random(&mut rng);
    let comm_a = GE::vartime_multiscalar_mul(&a_vec, &ck[..n], true).unwrap() + h * r_a;
    let r_c = Scalar::random(&mut rng);
    let comm_c = GE::group(ck_c) * v + h_c * r_c;

    let instance = InnerProductInstance::<E>::new(&comm_a, &b_vec, &comm_c);
    let witness = InnerProductWitness::<E>::new(&a_vec, &r_a, &r_c);

    let mut pt = TE::new(b"test_hybrid_ipa");
    let proof =
      InnerProductArgumentHybrid::<E, K>::prove(ck, &h, ck_c, &h_c, &instance, &witness, &mut pt)
        .expect("prove with oversized ck should succeed");

    let mut vt = TE::new(b"test_hybrid_ipa");
    proof
      .verify(ck, &h, ck_c, &h_c, n, &instance, &mut vt)
      .expect("verify with oversized ck should succeed");
  }

  /// Malformed proofs should return errors, not panic.
  #[test]
  fn test_hybrid_ipa_malformed_proof_errors() {
    let mut rng = thread_rng();
    const K: usize = 2;
    let n = 16;

    let gens = GE::from_label(b"test_ipa_hybrid", n + 1);
    let ck = &gens[..n];
    let h = GE::group(&gens[n]);

    let gens_eval = GE::from_label(b"test_ipa_hybrid_eval", 2);
    let ck_c = &gens_eval[0];
    let h_c = GE::group(&gens_eval[1]);

    let a_vec: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let b_vec: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let v = inner_product(&a_vec, &b_vec);

    let r_a = Scalar::random(&mut rng);
    let comm_a = GE::vartime_multiscalar_mul(&a_vec, ck, true).unwrap() + h * r_a;
    let r_c = Scalar::random(&mut rng);
    let comm_c = GE::group(ck_c) * v + h_c * r_c;

    let instance = InnerProductInstance::<E>::new(&comm_a, &b_vec, &comm_c);
    let witness = InnerProductWitness::<E>::new(&a_vec, &r_a, &r_c);

    let mut pt = TE::new(b"test_hybrid_ipa");
    let proof =
      InnerProductArgumentHybrid::<E, K>::prove(ck, &h, ck_c, &h_c, &instance, &witness, &mut pt)
        .unwrap();

    // Tamper: truncate z_vec to wrong length
    let mut bad_proof = proof.clone();
    bad_proof.z_vec.pop();
    let mut vt = TE::new(b"test_hybrid_ipa");
    assert!(
      bad_proof
        .verify(ck, &h, ck_c, &h_c, n, &instance, &mut vt)
        .is_err(),
      "truncated z_vec should fail verification"
    );

    // Tamper: corrupt a z_vec element
    let mut bad_proof = proof.clone();
    bad_proof.z_vec[0] += Scalar::ONE;
    let mut vt = TE::new(b"test_hybrid_ipa");
    assert!(
      bad_proof
        .verify(ck, &h, ck_c, &h_c, n, &instance, &mut vt)
        .is_err(),
      "corrupted z_vec should fail verification"
    );

    // Tamper: mismatched L/R lengths
    let mut bad_proof = proof.clone();
    bad_proof.bullet.R_vec.pop();
    let mut vt = TE::new(b"test_hybrid_ipa");
    assert!(
      bad_proof
        .verify(ck, &h, ck_c, &h_c, n, &instance, &mut vt)
        .is_err(),
      "mismatched L/R should fail verification"
    );
  }

  /// Mismatched witness/instance lengths should return errors from prove.
  #[test]
  fn test_hybrid_ipa_prove_mismatched_lengths() {
    let mut rng = thread_rng();
    const K: usize = 2;
    let n = 16;

    let gens = GE::from_label(b"test_ipa_hybrid", n + 1);
    let ck = &gens[..n];
    let h = GE::group(&gens[n]);

    let gens_eval = GE::from_label(b"test_ipa_hybrid_eval", 2);
    let ck_c = &gens_eval[0];
    let h_c = GE::group(&gens_eval[1]);

    let a_vec: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    // b_vec with wrong length
    let b_vec_bad: Vec<Scalar> = (0..n + 1).map(|_| Scalar::random(&mut rng)).collect();

    let r_a = Scalar::random(&mut rng);
    let comm_a = GE::vartime_multiscalar_mul(&a_vec, ck, true).unwrap() + h * r_a;
    let r_c = Scalar::random(&mut rng);
    let v = Scalar::random(&mut rng);
    let comm_c = GE::group(ck_c) * v + h_c * r_c;

    let instance = InnerProductInstance::<E>::new(&comm_a, &b_vec_bad, &comm_c);
    let witness = InnerProductWitness::<E>::new(&a_vec, &r_a, &r_c);

    let mut pt = TE::new(b"test_hybrid_ipa");
    let result =
      InnerProductArgumentHybrid::<E, K>::prove(ck, &h, ck_c, &h_c, &instance, &witness, &mut pt);
    assert!(
      result.is_err(),
      "mismatched a_vec/b_vec lengths should return error"
    );
  }
}
