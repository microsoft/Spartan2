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
  traits::{
    Engine,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    transcript::TranscriptReprTrait,
  },
};
use ff::Field;
use rayon::prelude::*;

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
    // Form the concatenated vectors z1 = (W1, u1, X1) and z2 = (W2, 1, X2).
    let Z1 = [W1.W.clone(), vec![U1.u], U1.X.clone()].concat();
    let Z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();

    if Z1.len() != Z2.len() {
      return Err(SpartanError::InvalidWitnessLength);
    }

    // Compute Z = Z1 + Z2 element-wise.
    let Z: Vec<E::Scalar> = Z1
      .into_par_iter()
      .zip(Z2.into_par_iter())
      .map(|(z1, z2)| z1 + z2)
      .collect();

    // Effective relaxation parameter.
    let u = U1.u + E::Scalar::ONE;

    let (AZ, BZ, CZ) = self.multiply_vec(&Z)?;

    let T: Vec<E::Scalar> = AZ
      .par_iter()
      .zip(BZ.par_iter())
      .zip(CZ.par_iter())
      .zip(W1.E.par_iter())
      .map(|(((az, bz), cz), e)| *az * *bz - u * *cz - *e)
      .collect();

    let comm_T = PCS::<E>::commit(ck, &T, r_T, false)?;
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
      .par_iter()
      .zip(&W2.W)
      .map(|(w1, w2)| *w1 + *r * *w2)
      .collect::<Vec<_>>();

    let E_vec = self
      .E
      .par_iter()
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
