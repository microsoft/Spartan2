// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Extended folding utilities and cross-term commitment helpers for NIFS.
#![allow(non_snake_case)]
use crate::{
  Blind, Commitment, CommitmentKey, PCS,
  errors::SpartanError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  start_span,
  traits::{
    Engine,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    transcript::TranscriptReprTrait,
  },
};
use ff::{Field, PrimeField};
use rayon::prelude::*;
use tracing::info;

// ------------------------------------------------------------------------------------------------
// Cross-term commitment helpers on the R1CS shape
// ------------------------------------------------------------------------------------------------
impl<E: Engine> R1CSShape<E> {
  /// Compute the cross-term `T` and its commitment when folding a relaxed instance `(U1, W1)`
  /// with a regular instance `(U2, W2)`.
  pub fn commit_T(
    &self,
    ck: &CommitmentKey<E>,
    U1: &RelaxedR1CSInstance<E>,
    W1: &RelaxedR1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    r_T: &Blind<E>,
  ) -> Result<(Vec<E::Scalar>, Commitment<E>), SpartanError> {
    // Form Z = (W1+W2, u1+1, X1+X2) without intermediate allocations.
    let n_w = W1.W.len();
    if W2.W.len() != n_w {
      return Err(SpartanError::InvalidWitnessLength);
    }
    let n_x = U1.X.len();
    if U2.X.len() != n_x {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "commit_T: U1.X.len() ({}) != U2.X.len() ({})",
          n_x,
          U2.X.len()
        ),
      });
    }
    let total = n_w + 1 + n_x;
    let mut Z = Vec::with_capacity(total);

    for i in 0..n_w {
      Z.push(W1.W[i] + W2.W[i]);
    }
    Z.push(U1.u + E::Scalar::ONE);
    for i in 0..n_x {
      Z.push(U1.X[i] + U2.X[i]);
    }

    // Effective relaxation parameter.
    let u = U1.u + E::Scalar::ONE;

    let (_mv_span, mv_t) = start_span!("commit_T_multiply_vec");
    let (AZ, BZ, CZ) = self.multiply_vec(&Z)?;
    info!(elapsed_ms = %mv_t.elapsed().as_millis(), "commit_T_multiply_vec");

    let (_cross_span, cross_t) = start_span!("commit_T_cross_term");
    let T: Vec<E::Scalar> = if rayon::current_num_threads() <= 1 {
      (0..AZ.len())
        .map(|i| AZ[i] * BZ[i] - u * CZ[i] - W1.E[i])
        .collect()
    } else {
      AZ.par_iter()
        .zip(BZ.par_iter())
        .zip(CZ.par_iter())
        .zip(W1.E.par_iter())
        .map(|(((az, bz), cz), e)| *az * *bz - u * *cz - *e)
        .collect()
    };
    info!(elapsed_ms = %cross_t.elapsed().as_millis(), size = %T.len(), "commit_T_cross_term");

    let (_comm_span, comm_t) = start_span!("commit_T_commit");
    // Auto-detect if T has small values for faster MSM
    let t_is_small = T.iter().all(|s| {
      let bytes = s.to_repr();
      bytes.as_ref()[8..].iter().all(|&b| b == 0)
    });
    let comm_T = PCS::<E>::commit(ck, &T, r_T, t_is_small)?;
    info!(elapsed_ms = %comm_t.elapsed().as_millis(), "commit_T_commit");
    Ok((T, comm_T))
  }
}

// ------------------------------------------------------------------------------------------------
// Folding operations on relaxed witnesses
// ------------------------------------------------------------------------------------------------
impl<E: Engine> RelaxedR1CSWitness<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Folds a relaxed R1CS witness with a regular R1CS witness using the challenge `r`.
  ///
  /// This computes the folded witness as:
  /// - `W_new = W1 + r * W2`
  /// - `E_new = E1 + r * T`
  ///
  /// # Arguments
  /// * `W2` - The regular R1CS witness to fold with
  /// * `T` - The cross-term vector
  /// * `r_T` - The blinding factor for the cross-term
  /// * `r` - The folding challenge
  ///
  /// # Returns
  /// The folded relaxed R1CS witness, or an error if the dimensions don't match.
  pub fn fold(
    &self,
    W2: &R1CSWitness<E>,
    T: &[E::Scalar],
    r_T: &Blind<E>,
    r: &E::Scalar,
  ) -> Result<RelaxedR1CSWitness<E>, SpartanError> {
    if self.W.len() != W2.W.len() || self.E.len() != T.len() {
      return Err(SpartanError::InvalidWitnessLength);
    }

    let W = self
      .W
      .iter()
      .zip(&W2.W)
      .map(|(w1, w2)| *w1 + *r * *w2)
      .collect::<Vec<_>>();

    let E_vec = self
      .E
      .iter()
      .zip(T)
      .map(|(e1, t)| *e1 + *r * *t)
      .collect::<Vec<_>>();

    let r_W = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
      &[self.r_W.clone(), W2.r_W.clone()],
      &[<E::Scalar as Field>::ONE, *r],
    )?;

    let r_E = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
      &[self.r_E.clone(), r_T.clone()],
      &[<E::Scalar as Field>::ONE, *r],
    )?;

    Ok(RelaxedR1CSWitness {
      W,
      r_W,
      E: E_vec,
      r_E,
    })
  }
}

// ------------------------------------------------------------------------------------------------
// Folding operations on relaxed instances
// ------------------------------------------------------------------------------------------------
impl<E: Engine> RelaxedR1CSInstance<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Folds a relaxed R1CS instance with a regular R1CS instance using the challenge `r`.
  ///
  /// This computes the folded instance as:
  /// - `comm_W_new = comm_W1 + r * comm_W2`
  /// - `comm_E_new = comm_E1 + r * comm_T`
  /// - `X_new = X1 + r * X2`
  /// - `u_new = u1 + r`
  ///
  /// # Arguments
  /// * `U2` - The regular R1CS instance to fold with
  /// * `comm_T` - The commitment to the cross-term
  /// * `r` - The folding challenge
  ///
  /// # Returns
  /// The folded relaxed R1CS instance.
  pub fn fold(
    &self,
    U2: &R1CSInstance<E>,
    comm_T: &Commitment<E>,
    r: &E::Scalar,
  ) -> RelaxedR1CSInstance<E> {
    let X = self
      .X
      .par_iter()
      .zip(&U2.X)
      .map(|(a, b)| *a + *r * *b)
      .collect::<Vec<_>>();

    let comm_W = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &[self.comm_W.clone(), U2.comm_W.clone()],
      &[<E::Scalar as Field>::ONE, *r],
    )
    .expect("fold commitments");

    let comm_E = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &[self.comm_E.clone(), comm_T.clone()],
      &[<E::Scalar as Field>::ONE, *r],
    )
    .expect("fold commitments");

    RelaxedR1CSInstance {
      comm_W,
      comm_E,
      X,
      u: self.u + *r,
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Byte encoding for relaxed instances (needed by transcripts)
// ------------------------------------------------------------------------------------------------
impl<E: Engine> TranscriptReprTrait<E::GE> for RelaxedR1CSInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_W.to_transcript_bytes(),
      self.comm_E.to_transcript_bytes(),
      self.u.to_transcript_bytes(),
      self.X.as_slice().to_transcript_bytes(),
    ]
    .concat()
  }
}

// ------------------------------------------------------------------------------------------------
// Folding operations on regular witnesses
// ------------------------------------------------------------------------------------------------
impl<E: Engine> R1CSWitness<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Fold this witness with another witness using barycentric weight `r_b`.
  ///
  /// The resulting witness corresponds to `(1 - r_b) * self + r_b * W2`.
  pub fn fold(&self, w2: &R1CSWitness<E>, r_b: &E::Scalar) -> Result<Self, SpartanError> {
    if self.W.len() != w2.W.len() {
      return Err(SpartanError::InvalidWitnessLength);
    }

    // Combine the witness vectors.
    let new_w = self
      .W
      .par_iter()
      .zip(&w2.W)
      .map(|(w1, w2)| *w1 + *r_b * (*w2 - *w1))
      .collect::<Vec<_>>();

    // Combine their blinds.
    let r_w = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
      &[self.r_W.clone(), w2.r_W.clone()],
      &[<E::Scalar as Field>::ONE - *r_b, *r_b],
    )?;

    Ok(Self {
      W: new_w,
      r_W: r_w,
      is_small: false, // after folding, witnesses are not small
    })
  }
}
