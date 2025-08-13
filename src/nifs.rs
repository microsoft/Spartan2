//! Non-Interactive Folding Scheme (NIFS) for Spartan2.
#![allow(non_snake_case)]

use crate::{
  Blind, Commitment, CommitmentKey,
  errors::SpartanError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{Engine, folding::FoldingEngineTrait, transcript::TranscriptEngineTrait},
};
use serde::{Deserialize, Serialize};

/// NIFS proof containing the commitment to the cross-term `T`.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<E: Engine> {
  pub(crate) comm_T: Commitment<E>,
}

impl<E: Engine> NIFS<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Fold a relaxed instance/witness `(U1, W1)` with a regular instance/witness `(U2, W2)`.
  pub fn prove(
    ck: &CommitmentKey<E>,
    S: &R1CSShape<E>,
    U1: &RelaxedR1CSInstance<E>,
    W1: &RelaxedR1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    transcript: &mut E::TE,
  ) -> Result<(Self, (RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>)), SpartanError> {
    // Use the caller-provided transcript and absorb both instances.
    transcript.absorb(b"U1", U1);
    transcript.absorb(b"U2", U2);

    // Compute the cross-term commitment.
    let r_T: Blind<E> = <E::PCS as crate::traits::pcs::PCSEngineTrait<E>>::blind(ck);
    let (T, comm_T) = S.commit_T(ck, U1, W1, U2, W2, &r_T)?;

    // Continue transcript with the cross-term commitment and derive the challenge `r`.
    transcript.absorb(b"comm_T", &comm_T);
    let r = transcript.squeeze(b"r")?;

    let U = U1.fold(U2, &comm_T, &r);
    let W = W1.fold(W2, &T, &r_T, &r)?;

    Ok((Self { comm_T }, (U, W)))
  }

  /// Verify folding given a regular instance `U2` that corresponds to the
  /// prover's split multi-round instance after conversion.
  pub fn verify(
    &self,
    transcript: &mut E::TE,
    U1: &RelaxedR1CSInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> Result<RelaxedR1CSInstance<E>, SpartanError> {
    // Re-run the prover’s transcript schedule to obtain `r` using caller transcript.
    transcript.absorb(b"U1", U1);
    transcript.absorb(b"U2", U2);
    transcript.absorb(b"comm_T", &self.comm_T);
    let r = transcript.squeeze(b"r")?;

    Ok(U1.fold(U2, &self.comm_T, &r))
  }
}

// ------------------------------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    CommitmentKey,
    bellpepper::tests::TwoRoundPermutationCircuit,
    provider::P256HyraxEngine,
    r1cs::{R1CSWitness, SparseMatrix, SplitMultiRoundR1CSInstance, SplitMultiRoundR1CSShape},
    traits::{Engine, pcs::PCSEngineTrait},
  };
  use ff::Field;

  type E = P256HyraxEngine;

  fn multiround_fold_with() {
    // Build a trivial 1-constraint, 1-variable, 1-round shape.
    let width = 16usize;
    let num_cons = 1usize;
    let num_vars_per_round = vec![1usize];
    let num_challenges_per_round = vec![0usize];
    let num_public = 0usize;

    let make_sparse =
      |rows: usize, cols: usize| SparseMatrix::<<E as Engine>::Scalar>::new(&[], rows, cols);
    let total_vars = num_vars_per_round.iter().sum::<usize>();
    let total_challenges = num_challenges_per_round.iter().sum::<usize>();
    let cols = total_vars + 1 + num_public + total_challenges;

    let shape_mr = SplitMultiRoundR1CSShape::<E>::new(
      width,
      num_cons,
      num_vars_per_round.clone(),
      num_challenges_per_round.clone(),
      num_public,
      make_sparse(num_cons, cols),
      make_sparse(num_cons, cols),
      make_sparse(num_cons, cols),
    )
    .expect("shape construction");

    let S_reg = shape_mr.to_regular_shape();
    let (ck, _vk): (CommitmentKey<E>, _) = shape_mr.commitment_key();

    // Build witness & commitments for the (single-round) multi-round instance.
    let total_vars: usize = shape_mr.num_vars_per_round.iter().sum();
    let mut W_vec = vec![<E as Engine>::Scalar::ZERO; total_vars];
    W_vec[0] = <E as Engine>::Scalar::ONE; // Set first variable to 1.
    let r_W = <<E as Engine>::PCS as PCSEngineTrait<E>>::blind(&ck);
    let (comm_round, _r_remaining) = <<E as Engine>::PCS as PCSEngineTrait<E>>::commit_partial(
      &ck,
      &W_vec[0..shape_mr.num_vars_per_round[0]],
      &r_W,
      false,
    )
    .unwrap();

    let inst_mr =
      SplitMultiRoundR1CSInstance::<E>::new(&shape_mr, vec![comm_round], vec![], vec![vec![]])
        .expect("instance construction");

    let inst_reg = inst_mr.to_regular_instance().unwrap();
    let wit_reg = R1CSWitness::<E>::new_unchecked(W_vec.clone(), r_W.clone(), false).unwrap();

    // Create a random relaxed instance/witness compatible with the same shape.
    let (running_U, running_W) = S_reg.sample_random_instance_witness(&ck).unwrap();

    // Prove & verify
    let mut transcript = <E as Engine>::TE::new(b"nifs");
    let (proof, (folded_U, folded_W)) = NIFS::<E>::prove(
      &ck,
      &S_reg,
      &running_U,
      &running_W,
      &inst_reg,
      &wit_reg,
      &mut transcript,
    )
    .unwrap();

    // Validation now happens at callsite; here we provide a regular U2 and transcript
    let mut transcript = <E as Engine>::TE::new(b"nifs");
    let U2_reg = inst_mr.to_regular_instance().unwrap();
    let verified_U = proof.verify(&mut transcript, &running_U, &U2_reg).unwrap();
    assert_eq!(verified_U, folded_U);
    assert!(S_reg.is_sat_relaxed(&ck, &folded_U, &folded_W).is_ok());
  }

  #[test]
  fn test_multiround_fold() {
    multiround_fold_with();
  }

  // ----------------------------------------------------------------------------------
  // Permutation folding test
  // ----------------------------------------------------------------------------------
  fn multiround_permutation_fold_with() {
    use crate::bellpepper::{
      r1cs::{MultiRoundSpartanShape, MultiRoundSpartanWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
    };

    let is_small = false;
    let circuit = TwoRoundPermutationCircuit;

    // Generate multi-round shape via Bellpepper
    let (shape_mr, ck, _vk) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&circuit).unwrap();

    // Witness generation across rounds
    let mut state =
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::initialize_multiround_witness(
        &shape_mr, &ck, &circuit, is_small,
      )
      .unwrap();
    let mut transcript = <E as Engine>::TE::new(b"nifs");

    let num_rounds =
      <TwoRoundPermutationCircuit as crate::traits::circuit::MultiRoundCircuit<E>>::num_rounds(
        &circuit,
      );
    for r in 0..num_rounds {
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::process_round(
        &mut state,
        &shape_mr,
        &ck,
        &circuit,
        r,
        is_small,
        &mut transcript,
      )
      .unwrap();
    }

    let (inst_split, wit_reg) =
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::finalize_multiround_witness(
        &mut state, &shape_mr, &ck, &circuit, is_small,
      )
      .unwrap();

    let inst_reg = inst_split.to_regular_instance().unwrap();
    let S_reg = shape_mr.to_regular_shape();

    // Random relaxed instance
    let (running_U, running_W) = S_reg.sample_random_instance_witness(&ck).unwrap();

    // NIFS prove + verify
    let mut transcript_nifs = <E as Engine>::TE::new(b"nifs");
    let (proof, (folded_U, folded_W)) = NIFS::<E>::prove(
      &ck,
      &S_reg,
      &running_U,
      &running_W,
      &inst_reg,
      &wit_reg,
      &mut transcript_nifs,
    )
    .unwrap();

    let mut transcript = <E as Engine>::TE::new(b"nifs");
    let U2_reg = inst_split.to_regular_instance().unwrap();
    let verified_U = proof.verify(&mut transcript, &running_U, &U2_reg).unwrap();
    assert_eq!(verified_U, folded_U);
    assert!(S_reg.is_sat_relaxed(&ck, &folded_U, &folded_W).is_ok());
  }

  #[test]
  fn test_multiround_permutation_fold() {
    multiround_permutation_fold_with();
  }
}
