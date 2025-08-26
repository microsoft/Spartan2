//! This module implements the spartan SNARK protocol.
//! It provides the prover and verifier keys, as well as the SNARK itself.
#![allow(non_snake_case)]
use crate::{
  CommitmentKey,
  bellpepper::{
    r1cs::{
      MultiRoundSpartanShape, MultiRoundSpartanWitness, PrecommittedState, SpartanShape,
      SpartanWitness,
    },
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  digest::{DigestComputer, SimpleDigestible},
  errors::SpartanError,
  nifs::NIFS,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
  },
  r1cs::{
    R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness, SparseMatrix, SplitMultiRoundR1CSInstance,
    SplitMultiRoundR1CSShape, SplitR1CSInstance, SplitR1CSShape,
  },
  start_span,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::SpartanCircuit,
    folding::FoldingEngineTrait,
    pcs::PCSEngineTrait,
    snark::{DigestHelperTrait, R1CSSNARKTrait, SpartanDigest},
    transcript::TranscriptEngineTrait,
  },
  zk::SpartanVerifierCircuit,
};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanProverKey<E: Engine> {
  ck: CommitmentKey<E>,
  S: SplitR1CSShape<E>,
  // Verifier-circuit (multi-round) parameters
  verifier_shape_mr: SplitMultiRoundR1CSShape<E>,
  // Precomputed regular (single-round) verifier shape
  verifier_shape_reg: R1CSShape<E>,
  verifier_ck_mr: CommitmentKey<E>,
  vk_digest: SpartanDigest, // digest of the verifier's key
}

impl<E: Engine> SpartanProverKey<E> {
  /// Returns sizes associated with the SplitR1CSShape.
  /// It returns an array of 10 elements containing:
  /// [num_cons_unpadded, num_shared_unpadded, num_precommitted_unpadded, num_rest_unpadded,
  ///  num_cons, num_shared, num_precommitted, num_rest,
  ///  num_public, num_challenges]
  pub fn sizes(&self) -> [usize; 10] {
    self.S.sizes()
  }
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanVerifierKey<E: Engine> {
  vk_ee: <E::PCS as PCSEngineTrait<E>>::VerifierKey,
  S: SplitR1CSShape<E>,
  // Verifier-circuit (multi-round) shape
  verifier_shape_mr: SplitMultiRoundR1CSShape<E>,
  // Precomputed regular (single-round) verifier shape
  verifier_shape_reg: R1CSShape<E>,
  // Commitment key for the verifier-circuit (multi-round) shape; shared with prover
  verifier_ck_mr: CommitmentKey<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<SpartanDigest>,
}

impl<E: Engine> SimpleDigestible for SpartanVerifierKey<E> {}

impl<E: Engine> DigestHelperTrait<E> for SpartanVerifierKey<E> {
  /// Returns the digest of the verifier's key.
  fn digest(&self) -> Result<SpartanDigest, SpartanError> {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::<_>::new(self);
        dc.digest()
      })
      .cloned()
      .map_err(|_| SpartanError::DigestError {
        reason: "Unable to compute digest for SpartanVerifierKey".to_string(),
      })
  }
}

/// Binds "row" variables of (A, B, C) matrices viewed as 2d multilinear polynomials
pub(crate) fn compute_eval_table_sparse<E: Engine>(
  S: &SplitR1CSShape<E>,
  rx: &[E::Scalar],
) -> (Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>) {
  assert_eq!(rx.len(), S.num_cons);

  let inner = |M: &SparseMatrix<E::Scalar>, M_evals: &mut Vec<E::Scalar>| {
    for (row_idx, ptrs) in M.indptr.windows(2).enumerate() {
      for (val, col_idx) in M.get_row_unchecked(ptrs.try_into().unwrap()) {
        M_evals[*col_idx] += rx[row_idx] * val;
      }
    }
  };

  let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
  let (A_evals, (B_evals, C_evals)) = rayon::join(
    || {
      let mut A_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * num_vars];
      inner(&S.A, &mut A_evals);
      A_evals
    },
    || {
      rayon::join(
        || {
          let mut B_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * num_vars];
          inner(&S.B, &mut B_evals);
          B_evals
        },
        || {
          let mut C_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * num_vars];
          inner(&S.C, &mut C_evals);
          C_evals
        },
      )
    },
  );

  (A_evals, B_evals, C_evals)
}

/// A type that holds the pre-processed state for proving
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PrepSNARK<E: Engine> {
  ps: PrecommittedState<E>,
}

/// A succinct non-interactive argument of knowledge (SNARK) for a relaxed R1CS instance,
/// produced using Spartan's combination of sum-check protocols and polynomial commitments.
/// This proof attests to knowledge of a witness satisfying the given R1CS constraints.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSSNARK<E: Engine> {
  // Multi-round verifier instance capturing the non-ZK verification trace
  U_verifier: SplitMultiRoundR1CSInstance<E>,
  // Original single-round R1CS instance
  U: SplitR1CSInstance<E>,
  // NIFS proof for folding a random relaxed instance with the verifier instance
  nifs_proof: NIFS<E>,
  // The random relaxed instance used for folding
  random_U: RelaxedR1CSInstance<E>,
  // Folded relaxed witness produced during NIFS proving
  folded_W: RelaxedR1CSWitness<E>,
  // PCS opening for the original witness commitment
  eval_W: E::Scalar,
  // PCS evaluation argument
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
}

impl<E: Engine> R1CSSNARKTrait<E> for R1CSSNARK<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  type ProverKey = SpartanProverKey<E>;
  type VerifierKey = SpartanVerifierKey<E>;
  type PrepSNARK = PrepSNARK<E>;

  fn setup<C: SpartanCircuit<E>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError> {
    let S = ShapeCS::r1cs_shape(&circuit)?;
    let (ck, vk_ee) = SplitR1CSShape::commitment_key(&[&S])?;

    // Derive verifier-circuit (multi-round) shape based on outer/inner rounds
    let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
    let num_rounds_x = usize::try_from(S.num_cons.ilog2()).unwrap();
    let num_rounds_y = usize::try_from(num_vars.ilog2()).unwrap() + 1;
    let zero = E::Scalar::ZERO;
    let spartan_verifier_circuit = SpartanVerifierCircuit::<E> {
      outer_polys: vec![[zero; 4]; num_rounds_x],
      claim_Az: zero,
      claim_Bz: zero,
      claim_Cz: zero,
      tau_at_rx: zero,
      inner_polys: vec![[zero; 3]; num_rounds_y],
      eval_A: zero,
      eval_B: zero,
      eval_C: zero,
      eval_W: zero,
      eval_X: zero,
    };
    let (verifier_shape_mr, verifier_ck_mr, _vk_mr) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&spartan_verifier_circuit)?;
    let verifier_shape_reg = verifier_shape_mr.to_regular_shape();

    // Derive verifier-circuit (multi-round) shape based on outer/inner rounds
    let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
    let num_rounds_x = usize::try_from(S.num_cons.ilog2()).unwrap();
    let num_rounds_y = usize::try_from(num_vars.ilog2()).unwrap() + 1;
    let zero = E::Scalar::ZERO;
    let spartan_verifier_circuit = SpartanVerifierCircuit::<E> {
      outer_polys: vec![[zero; 4]; num_rounds_x],
      claim_Az: zero,
      claim_Bz: zero,
      claim_Cz: zero,
      tau_at_rx: zero,
      inner_polys: vec![[zero; 3]; num_rounds_y],
      eval_A: zero,
      eval_B: zero,
      eval_C: zero,
      eval_W: zero,
      eval_X: zero,
    };
    let (verifier_shape_mr, verifier_ck_mr, _vk_mr) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&spartan_verifier_circuit)?;
    let verifier_shape_reg = verifier_shape_mr.to_regular_shape();

    let vk: SpartanVerifierKey<E> = SpartanVerifierKey {
      S: S.clone(),
      vk_ee,
      verifier_shape_mr: verifier_shape_mr.clone(),
      verifier_shape_reg: verifier_shape_reg.clone(),
      verifier_ck_mr: verifier_ck_mr.clone(),
      digest: OnceCell::new(),
    };
    let pk = Self::ProverKey {
      ck,
      S,
      verifier_shape_mr,
      verifier_shape_reg,
      verifier_ck_mr,
      vk_digest: vk.digest()?,
    };

    Ok((pk, vk))
  }

  /// Prepares the SNARK for proving
  fn prep_prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<Self::PrepSNARK, SpartanError> {
    let mut ps = SatisfyingAssignment::shared_witness(&pk.S, &pk.ck, &circuit, is_small)?;
    SatisfyingAssignment::precommitted_witness(&mut ps, &pk.S, &pk.ck, &circuit, is_small)?;

    Ok(PrepSNARK { ps })
  }

  /// produces a succinct proof of satisfiability of an R1CS instance
  fn prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    prep_snark: &Self::PrepSNARK,
    is_small: bool,
  ) -> Result<Self, SpartanError> {
    let mut prep_snark = prep_snark.clone(); // make a copy so we can modify it

    let mut transcript = E::TE::new(b"R1CSSNARK");
    transcript.absorb(b"vk", &pk.vk_digest);
    // absorb public IO before
    let public_values = circuit
      .public_values()
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Circuit does not provide public IO: {e}"),
      })?;
    transcript.absorb(b"public_values", &public_values.as_slice());

    // Original R1CS instance and witness (used for PCS evaluation only)
    let (U, W) = SatisfyingAssignment::r1cs_instance_and_witness(
      &mut prep_snark.ps,
      &pk.S,
      &pk.ck,
      &circuit,
      is_small,
      &mut transcript,
    )?;

    // Multi-round witness may contain random field elements that are NOT small, so always commit full elements.
    let mr_is_small = false;

    // Prepare vectors and polynomials for building the verifier-circuit trace
    let mut z = W.W.clone();
    z.push(E::Scalar::ONE);
    z.extend_from_slice(&U.public_values);
    z.extend_from_slice(&U.challenges);

    let num_vars = pk.S.num_shared + pk.S.num_precommitted + pk.S.num_rest;
    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(pk.S.num_cons.ilog2()).unwrap(),
      usize::try_from(num_vars.ilog2()).unwrap() + 1,
    );

    // Build tau and Az/Bz/Cz polynomials
    let tau = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;
    let mut poly_tau = MultilinearPolynomial::new(tau.evals());

    let (Az, Bz, Cz) = pk.S.multiply_vec(&z)?;
    let mut poly_Az = MultilinearPolynomial::new(Az);
    let mut poly_Bz = MultilinearPolynomial::new(Bz);
    let mut poly_Cz = MultilinearPolynomial::new(Cz);

    // Helpers now live in `sumcheck.rs` and are called below via `SumcheckProof::*`.

    // Initialize multi-round verifier circuit (will be filled as we go)
    let zero = E::Scalar::ZERO;
    let mut verifier_circuit = SpartanVerifierCircuit::<E> {
      outer_polys: vec![[zero; 4]; num_rounds_x],
      claim_Az: zero,
      claim_Bz: zero,
      claim_Cz: zero,
      tau_at_rx: zero,
      inner_polys: vec![[zero; 3]; num_rounds_y],
      eval_A: zero,
      eval_B: zero,
      eval_C: zero,
      eval_W: zero,
      eval_X: zero,
    };

    // Build the multi-round instance by interleaving with challenge generation
    let mut state =
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::initialize_multiround_witness(
        &pk.verifier_shape_mr,
        &pk.verifier_ck_mr,
        &verifier_circuit,
        mr_is_small,
      )?;

    // Outer sum-check
    let r_x = SumcheckProof::<E>::prove_cubic_with_additive_term_zk(
      num_rounds_x,
      &mut poly_tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_Cz,
      &mut verifier_circuit,
      &mut state,
      &pk.verifier_shape_mr,
      &pk.verifier_ck_mr,
      mr_is_small,
      &mut transcript,
    )?;

    // Outer final round data
    verifier_circuit.claim_Az = poly_Az[0];
    verifier_circuit.claim_Bz = poly_Bz[0];
    verifier_circuit.claim_Cz = poly_Cz[0];
    verifier_circuit.tau_at_rx = poly_tau[0];
    // Process the synthetic "outer final" round in the circuit and capture challenges (unused)
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut state,
      &pk.verifier_shape_mr,
      &pk.verifier_ck_mr,
      &verifier_circuit,
      num_rounds_x,
      mr_is_small,
      &mut transcript,
    )?;

    // Inner setup: derive r from transcript by processing that round
    let inner_setup_round = num_rounds_x + 1;
    let chals = SatisfyingAssignment::<E>::process_round(
      &mut state,
      &pk.verifier_shape_mr,
      &pk.verifier_ck_mr,
      &verifier_circuit,
      inner_setup_round,
      mr_is_small,
      &mut transcript,
    )?;
    let r = chals[0];

    // Prepare inner polynomials
    let evals_rx = EqPolynomial::evals_from_points(&r_x);
    let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&pk.S, &evals_rx);
    let poly_ABC: Vec<E::Scalar> = (0..evals_A.len())
      .into_par_iter()
      .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
      .collect();
    z.resize(num_vars * 2, E::Scalar::ZERO);
    let mut poly_ABC = MultilinearPolynomial::new(poly_ABC);
    let mut poly_z = MultilinearPolynomial::new(z);

    // Inner sum-check
    let claim_inner_joint =
      verifier_circuit.claim_Az + r * verifier_circuit.claim_Bz + r * r * verifier_circuit.claim_Cz;

    let r_y = SumcheckProof::<E>::prove_quad_zk(
      &claim_inner_joint,
      num_rounds_y,
      &mut poly_ABC,
      &mut poly_z,
      &mut verifier_circuit,
      &mut state,
      &pk.verifier_shape_mr,
      &pk.verifier_ck_mr,
      mr_is_small,
      &mut transcript,
      num_rounds_x + 2,
    )?;

    // Final evaluations for the circuit's last round
    // Compute PCS evaluation for original witness at point r_y[1..]

    let U_regular = U.to_regular_instance()?;
    let (eval_W, eval_arg) = E::PCS::prove(
      &pk.ck,
      &mut transcript,
      &U_regular.comm_W,
      &W.W,
      &W.r_W,
      &r_y[1..],
    )?;

    // Compute final evaluations needed for the inner-final round
    let eval_X = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(U_regular.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars = pk.S.num_shared + pk.S.num_precommitted + pk.S.num_rest;
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).unwrap();
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };

    // Evaluate A, B, C at (r_x, r_y)
    let (eval_A, eval_B, eval_C) = {
      let T_y = EqPolynomial::evals_from_points(&r_y);
      multi_inner_product(&T_y, &evals_A, &evals_B, &evals_C)
    };

    // Set verifier circuit public values before processing inner-final round
    verifier_circuit.eval_W = eval_W;
    verifier_circuit.eval_X = eval_X;
    verifier_circuit.eval_A = eval_A;
    verifier_circuit.eval_B = eval_B;
    verifier_circuit.eval_C = eval_C;

    // Process the circuit's inner-final round and capture challenges (unused)
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut state,
      &pk.verifier_shape_mr,
      &pk.verifier_ck_mr,
      &verifier_circuit,
      (num_rounds_x + 2) + num_rounds_y,
      mr_is_small,
      &mut transcript,
    )?;

    // Finalize multi-round witness and construct NIFS proof
    let (U_verifier, W_verifier) =
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::finalize_multiround_witness(
        &mut state,
        &pk.verifier_shape_mr,
        &pk.verifier_ck_mr,
        &verifier_circuit,
        mr_is_small,
      )?;
    // Use the instance as produced by witness finalization; its public values
    // are exactly those absorbed during round 0 by the prover.
    let U_verifier_regular = U_verifier.to_regular_instance()?;
    let S_verifier = &pk.verifier_shape_reg;
    let (random_U, random_W) = S_verifier.sample_random_instance_witness(&pk.verifier_ck_mr)?;
    let (nifs_proof, folded_W) = NIFS::<E>::prove(
      &pk.verifier_ck_mr,
      S_verifier,
      &random_U,
      &random_W,
      &U_verifier_regular,
      &W_verifier,
      &mut transcript,
    )?;

    Ok(R1CSSNARK {
      U_verifier,
      nifs_proof,
      random_U,
      folded_W,
      eval_W,
      eval_arg,
      U,
    })
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey) -> Result<Vec<E::Scalar>, SpartanError> {
    // Verify by checking the multi-round verifier instance via NIFS folding
    let (_verify_span, _verify_t) = start_span!("r1cs_snark_verify");
    let ck_verifier = &vk.verifier_ck_mr;
    let mut transcript = E::TE::new(b"R1CSSNARK");
    transcript.absorb(b"vk", &vk.digest()?);
    transcript.absorb(b"public_values", &self.U.public_values.as_slice());

    // Validate the provided split R1CS instance and advance the transcript
    self.U.validate(&vk.S, &mut transcript)?;

    // Recreate tau polynomial coefficients via Fiat-Shamir and advance transcript
    let num_rounds_x = usize::try_from(vk.S.num_cons.ilog2()).unwrap();
    let tau = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;

    // Reproduce the multi-round transcript schedule up to (but excluding) the final round
    // so that PCS occurs at the same transcript state as in the prover.
    let S_verifier = &vk.verifier_shape_mr;
    let U_verifier = &self.U_verifier;
    if S_verifier.num_rounds == 0 {
      return Err(SpartanError::ProofVerifyError {
        reason: "Verifier shape has zero rounds".to_string(),
      });
    }

    // Validate rounds [0 .. num_rounds-1)
    for round in 0..(S_verifier.num_rounds - 1) {
      if round > 0 {
        // absorb commitment of previous round and check size
        <E::PCS as PCSEngineTrait<E>>::check_partial(
          &U_verifier.comm_w_per_round[round - 1],
          S_verifier.num_vars_per_round[round - 1],
        )?;
        transcript.absorb(b"comm_w_round", &U_verifier.comm_w_per_round[round - 1]);
      }
      // derive and validate challenges for this round
      let expected = (0..S_verifier.num_challenges_per_round[round])
        .map(|_| transcript.squeeze(b"challenge"))
        .collect::<Result<Vec<E::Scalar>, SpartanError>>()?;
      if expected != U_verifier.challenges_per_round[round] {
        return Err(SpartanError::ProofVerifyError {
          reason: format!("Challenges for round {round} do not match"),
        });
      }
    }

    let U_verifier_regular = self.U_verifier.to_regular_instance()?;

    // Verify PCS opening at r_y[1..] using challenges extracted from the multi-round instance
    // Derive expected challenge counts from the original shape sizes
    let num_vars = vk.S.num_shared + vk.S.num_precommitted + vk.S.num_rest;
    let num_rounds_x = usize::try_from(vk.S.num_cons.ilog2()).unwrap();
    let num_rounds_y = usize::try_from(num_vars.ilog2()).unwrap() + 1;

    // The regular instance packs [challenges..., public_values...], where public_values are 5 items (A,B,C,tau_at_rx,X)
    let num_public_values = 5usize;
    if U_verifier_regular.X.len() < num_public_values {
      return Err(SpartanError::ProofVerifyError {
        reason: "Verifier instance missing public values".to_string(),
      });
    }
    let total_challenges = U_verifier_regular.X.len() - num_public_values;
    if total_challenges != num_rounds_x + 1 + num_rounds_y {
      return Err(SpartanError::ProofVerifyError {
        reason: "Unexpected number of challenges in verifier instance".to_string(),
      });
    }
    let all_challenges = &U_verifier_regular.X[0..total_challenges];

    let start_inner = num_rounds_x + 1; // index of r_y[0]
    let r_x = &all_challenges[0..num_rounds_x];
    let r_y = &all_challenges[start_inner..start_inner + num_rounds_y];

    // Recompute eval_A, eval_B, eval_C at (r_x, r_y)
    let (eval_A, eval_B, eval_C) = {
      let T_x = EqPolynomial::evals_from_points(r_x);
      let T_y = EqPolynomial::evals_from_points(r_y);
      let multi_eval = |M: &SparseMatrix<E::Scalar>| -> E::Scalar {
        M.indptr
          .par_windows(2)
          .enumerate()
          .map(|(row_idx, ptrs)| {
            M.get_row_unchecked(ptrs.try_into().unwrap())
              .map(|(val, col_idx)| {
                let prod = T_x[row_idx] * T_y[*col_idx];
                if *val == E::Scalar::ONE {
                  prod
                } else if *val == -E::Scalar::ONE {
                  -prod
                } else {
                  prod * val
                }
              })
              .sum::<E::Scalar>()
          })
          .sum()
      };
      (
        multi_eval(&vk.S.A),
        multi_eval(&vk.S.B),
        multi_eval(&vk.S.C),
      )
    };

    // Recompute eval_X from original circuit public IO at r_y[1..]
    let U_regular = self.U.to_regular_instance()?;

    let eval_X = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(U_regular.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars = vk.S.num_shared + vk.S.num_precommitted + vk.S.num_rest;
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).unwrap();
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };

    // Recompute tau(r_x) using the same tau polynomial challenges
    let tau_at_rx = tau.evaluate(r_x);

    // Compare against the instance's public inputs [eval_A, eval_B, eval_C, tau_at_rx, eval_X]
    let pub_start = total_challenges;
    let pub_vals = &U_verifier_regular.X[pub_start..pub_start + num_public_values];
    if pub_vals[0] != eval_A
      || pub_vals[1] != eval_B
      || pub_vals[2] != eval_C
      || pub_vals[3] != tau_at_rx
      || pub_vals[4] != eval_X
    {
      return Err(SpartanError::ProofVerifyError {
        reason:
          "Verifier instance public values do not match recomputed evaluations (A,B,C,tau_at_rx,X)"
            .to_string(),
      });
    }

    // Continue with PCS verification on the same transcript
    E::PCS::verify(
      &vk.vk_ee,
      &mut transcript,
      &U_regular.comm_W,
      &r_y[1..],
      &self.eval_W,
      &self.eval_arg,
    )?;

    // Finish the last (final) multi-round step to catch up with the prover's schedule
    let last_round = S_verifier.num_rounds - 1;
    if last_round > 0 {
      <E::PCS as PCSEngineTrait<E>>::check_partial(
        &U_verifier.comm_w_per_round[last_round - 1],
        S_verifier.num_vars_per_round[last_round - 1],
      )?;
      transcript.absorb(
        b"comm_w_round",
        &U_verifier.comm_w_per_round[last_round - 1],
      );
    }
    let expected_last = (0..S_verifier.num_challenges_per_round[last_round])
      .map(|_| transcript.squeeze(b"challenge"))
      .collect::<Result<Vec<E::Scalar>, SpartanError>>()?;
    if expected_last != U_verifier.challenges_per_round[last_round] {
      return Err(SpartanError::ProofVerifyError {
        reason: format!("Challenges for round {last_round} do not match"),
      });
    }

    // Finally, run NIFS verification using the same transcript
    let folded_U = self
      .nifs_proof
      .verify(&mut transcript, &self.random_U, &U_verifier_regular)?;

    // Check satisfiability of the folded relaxed instance with the folded witness
    let S_verifier = &vk.verifier_shape_reg;
    S_verifier
      .is_sat_relaxed(ck_verifier, &folded_U, &self.folded_W)
      .map_err(|e| SpartanError::ProofVerifyError {
        reason: format!("Folded instance not satisfiable: {e}"),
      })?;

    // Return original circuit public IO carried in the proof
    Ok(self.U.public_values.clone())
  }
}

/// computes an inner products <t, a>, <t, b>, and <t,c>
fn multi_inner_product<T: Field + Send + Sync>(t: &[T], a: &[T], b: &[T], c: &[T]) -> (T, T, T) {
  assert_eq!(t.len(), a.len());
  assert_eq!(a.len(), b.len());
  assert_eq!(b.len(), c.len());

  (0..t.len())
    .into_par_iter()
    .map(|i| {
      let ti = t[i]; // read t[i] once
      (ti * a[i], ti * b[i], ti * c[i])
    })
    .reduce(
      || (T::ZERO, T::ZERO, T::ZERO),
      |(sa, sb, sc), (xa, xb, xc)| (sa + xa, sb + xb, sc + xc),
    )
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
  use tracing_subscriber::EnvFilter;

  #[cfg(feature = "jem")]
  use tikv_jemallocator::Jemalloc;

  #[cfg(feature = "jem")]
  #[global_allocator]
  static GLOBAL: Jemalloc = Jemalloc;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit {}

  impl<E: Engine> SpartanCircuit<E> for CubicCircuit {
    fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
      Ok(vec![E::Scalar::from(15u64)])
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      // In this example, we do not have shared variables.
      Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<<E as Engine>::Scalar>>(
      &self,
      _: &mut CS,
      _: &[AllocatedNum<E::Scalar>], // shared variables, if any
    ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
      // In this example, we do not have precommitted variables.
      Ok(vec![])
    }

    fn num_challenges(&self) -> usize {
      // In this example, we do not use challenges.
      0
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
      _: &[AllocatedNum<E::Scalar>],
      _: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::ONE + E::Scalar::ONE))?;
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + E::Scalar::from(5u64))
      })?;

      cs.enforce(
        || "y = x^3 + x + 5",
        |lc| {
          lc + x_cu.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + y.get_variable(),
      );

      let _ = y.inputize(cs.namespace(|| "output"));

      Ok(())
    }
  }

  #[test]
  fn test_snark() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true) // no bold colour codes
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();

    type E = crate::provider::PallasHyraxEngine;
    type S = R1CSSNARK<E>;
    test_snark_with::<E, S>();

    type E2 = crate::provider::T256HyraxEngine;
    type S2 = R1CSSNARK<E2>;
    test_snark_with::<E2, S2>();
  }

  fn test_snark_with<E: Engine, S: R1CSSNARKTrait<E>>() {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) = S::setup(circuit.clone()).unwrap();

    // generate pre-processed state for proving
    let prep_snark = S::prep_prove(&pk, circuit.clone(), false).unwrap();

    // generate a witness and proof
    let res = S::prove(&pk, circuit, &mut prep_snark, false);
    assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk);
    assert!(res.is_ok());
    assert_eq!(res.unwrap(), [<E as Engine>::Scalar::from(15u64)])
  }
}
