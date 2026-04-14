// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Nova's non-Interactive Folding Scheme (NIFS)
use crate::{
  Blind, Commitment, CommitmentKey, PCS,
  errors::SpartanError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    Engine,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    transcript::TranscriptEngineTrait,
  },
};
use serde::{Deserialize, Serialize};
use tracing::info;
use crate::start_span;

/// Nova NIFS proof containing the commitment to the cross-term `T`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NovaNIFS<E: Engine> {
  pub(crate) comm_T: Commitment<E>,
}

impl<E: Engine> NovaNIFS<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Fold a relaxed instance/witness `(U1, W1)` with a regular instance/witness `(U2, W2)`.
  /// Returns the folded witness plus the folded scalar fields `(u, X)` -- no commitment
  /// arithmetic, keeping the prover fast.
  pub fn prove(
    ck: &CommitmentKey<E>,
    S: &R1CSShape<E>,
    U1: &RelaxedR1CSInstance<E>,
    W1: &RelaxedR1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    transcript: &mut E::TE,
  ) -> Result<(Self, RelaxedR1CSWitness<E>, E::Scalar, Vec<E::Scalar>), SpartanError> {
    // Use the caller-provided transcript and absorb both instances.
    transcript.absorb(b"U1", U1);
    transcript.absorb(b"U2", U2);

    // Compute the cross-term commitment.
    let (_ct_span, ct_t) = start_span!("nifs_commit_T");
    let r_T: Blind<E> = PCS::<E>::blind(ck, S.num_cons);
    let (T, comm_T) = S.commit_T(ck, U1, W1, U2, W2, &r_T)?;
    info!(elapsed_ms = %ct_t.elapsed().as_millis(), num_cons = %S.num_cons, "nifs_commit_T");

    // Continue transcript with the cross-term commitment and derive the challenge `r`.
    transcript.absorb(b"comm_T", &comm_T);
    let r = transcript.squeeze(b"r")?;

    let (_fold_span, fold_t) = start_span!("nifs_fold");
    let W = W1.fold(W2, &T, &r_T, &r)?;
    // Fold scalar fields only (no commitment arithmetic)
    let u_folded = U1.u + r;
    let X_folded: Vec<E::Scalar> = U1
      .X
      .iter()
      .zip(&U2.X)
      .map(|(a, b)| *a + r * *b)
      .collect();
    info!(elapsed_ms = %fold_t.elapsed().as_millis(), "nifs_fold");

    Ok((Self { comm_T }, W, u_folded, X_folded))
  }

  /// Verify folding given a regular instance `U2` that corresponds to the
  /// prover's split multi-round instance after conversion.
  pub fn verify(
    &self,
    transcript: &mut E::TE,
    U1: &RelaxedR1CSInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> Result<RelaxedR1CSInstance<E>, SpartanError> {
    transcript.absorb(b"U1", U1);
    transcript.absorb(b"U2", U2);
    transcript.absorb(b"comm_T", &self.comm_T);
    let r = transcript.squeeze(b"r")?;

    Ok(U1.fold(U2, &self.comm_T, &r))
  }
}
