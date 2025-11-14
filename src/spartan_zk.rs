//! This module implements the Spartan zkSNARK protocol. We provide zero-knowledge via Nova's folding scheme
//! It provides the prover and verifier keys, as well as the zkSNARK itself.
use crate::{
  CommitmentKey,
  bellpepper::{
    r1cs::{
      MultiRoundSpartanShape, MultiRoundSpartanWitness, PrecommittedState, RerandomizationTrait,
      SpartanShape, SpartanWitness,
    },
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  digest::{DigestComputer, SimpleDigestible},
  errors::SpartanError,
  math::Math,
  nifs::NovaNIFS,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
  },
  r1cs::{
    R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness, SplitMultiRoundR1CSInstance,
    SplitMultiRoundR1CSShape, SplitR1CSInstance, SplitR1CSShape,
  },
  start_span,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::SpartanCircuit,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    snark::{DigestHelperTrait, R1CSSNARKTrait, SpartanDigest},
    transcript::TranscriptEngineTrait,
  },
  zk::SpartanVerifierCircuit,
};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanProverKey<E: Engine> {
  ck: CommitmentKey<E>,
  S: SplitR1CSShape<E>,
  // Verifier-circuit (multi-round) parameters
  vc_shape: SplitMultiRoundR1CSShape<E>,
  // Precomputed regular (single-round) verifier shape
  vc_shape_regular: R1CSShape<E>,
  vc_ck: CommitmentKey<E>,
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
  vc_shape: SplitMultiRoundR1CSShape<E>,
  // Precomputed regular (single-round) verifier shape
  vc_shape_regular: R1CSShape<E>,
  // Commitment key for the verifier-circuit (multi-round) shape; shared with prover
  vc_ck: CommitmentKey<E>,
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

/// A type that holds the pre-processed state for proving
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanPrepZkSNARK<E: Engine> {
  ps: PrecommittedState<E>,
}

/// A succinct non-interactive argument of knowledge (SNARK) for a relaxed R1CS instance,
/// produced using Spartan's combination of sum-check protocols and polynomial commitments.
/// This proof attests to knowledge of a witness satisfying the given R1CS constraints.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanZkSNARK<E: Engine> {
  // Original R1CS instance
  U: SplitR1CSInstance<E>,

  // Multi-round verifier instance capturing the non-ZK verification trace
  U_verifier: SplitMultiRoundR1CSInstance<E>,
  // The random relaxed instance used for folding
  random_U: RelaxedR1CSInstance<E>,
  // NIFS proof for folding a random relaxed instance with the verifier instance
  nifs: NovaNIFS<E>,
  // Folded relaxed witness produced during NIFS proving
  folded_W: RelaxedR1CSWitness<E>,

  // PCS evaluation argument
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
}

impl<E: Engine> R1CSSNARKTrait<E> for SpartanZkSNARK<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  type ProverKey = SpartanProverKey<E>;
  type VerifierKey = SpartanVerifierKey<E>;
  type PrepSNARK = SpartanPrepZkSNARK<E>;

  fn setup<C: SpartanCircuit<E>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError> {
    let S = ShapeCS::r1cs_shape(&circuit)?;
    let (ck, vk_ee) = SplitR1CSShape::commitment_key(&[&S])?;

    // Derive verifier-circuit (multi-round) shape based on outer/inner rounds
    let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
    let num_rounds_x = S.num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;
    let zero = E::Scalar::ZERO;
    let vc = SpartanVerifierCircuit::<E> {
      outer_polys: vec![[zero; 4]; num_rounds_x],
      claim_Az: zero,
      claim_Bz: zero,
      claim_Cz: zero,
      tau_at_rx: zero,
      inner_polys: vec![[zero; 3]; num_rounds_y],
      eval_W: zero,
      eval_X: zero,
    };
    let (vc_shape, vc_ck, _vk_mr) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&vc)?;
    let vc_shape_regular = vc_shape.to_regular_shape();

    let vk = Self::VerifierKey {
      S: S.clone(),
      vk_ee,
      vc_shape: vc_shape.clone(),
      vc_shape_regular: vc_shape_regular.clone(),
      vc_ck: vc_ck.clone(),
      digest: OnceCell::new(),
    };
    let pk = Self::ProverKey {
      ck,
      S,
      vc_shape,
      vc_shape_regular,
      vc_ck,
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

    Ok(SpartanPrepZkSNARK { ps })
  }

  /// produces a succinct proof of satisfiability of an R1CS instance
  fn prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    prep_snark: &Self::PrepSNARK,
    is_small: bool,
  ) -> Result<Self, SpartanError> {
    let (_prove_span, prove_t) = start_span!("spartan_zk_prove");

    // rerandomize the prep state
    let (_rerandomize_span, rerandomize_t) = start_span!("rerandomize_prep_state");
    let mut ps = prep_snark.ps.rerandomize(&pk.ck, &pk.S)?;
    info!(elapsed_ms = %rerandomize_t.elapsed().as_millis(), "rerandomize_prep_state");

    let mut transcript = E::TE::new(b"SpartanZkSNARK");
    transcript.absorb(b"vk", &pk.vk_digest);
    // absorb public IO before
    let public_values = circuit
      .public_values()
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Circuit does not provide public IO: {e}"),
      })?;
    transcript.absorb(b"public_values", &public_values.as_slice());

    // Original R1CS instance and witness (used for PCS evaluation only)
    let (_instance_span, instance_t) = start_span!("r1cs_instance_and_witness");
    let (U, W) = SatisfyingAssignment::r1cs_instance_and_witness(
      &mut ps,
      &pk.S,
      &pk.ck,
      &circuit,
      is_small,
      &mut transcript,
    )?;
    info!(elapsed_ms = %instance_t.elapsed().as_millis(), "r1cs_instance_and_witness");

    // Prepare vectors and polynomials for building the verifier-circuit trace
    let mut z = W.W.clone();
    z.push(E::Scalar::ONE);
    z.extend_from_slice(&U.public_values);
    z.extend_from_slice(&U.challenges);

    let num_vars = pk.S.num_shared + pk.S.num_precommitted + pk.S.num_rest;
    let (num_rounds_x, num_rounds_y) = (pk.S.num_cons.log_2(), num_vars.log_2() + 1);

    // Sample tau challenges used for the outer equality polynomial
    let (_taus_span, taus_t) = start_span!("sample_taus");
    let taus = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"t"))
      .collect::<Result<Vec<_>, SpartanError>>()?;
    info!(elapsed_ms = %taus_t.elapsed().as_millis(), "sample_taus");

    let (_mv_span, mv_t) = start_span!("matrix_vector_multiply");
    let (Az, Bz, Cz) = pk.S.multiply_vec(&z)?;
    info!(
      elapsed_ms = %mv_t.elapsed().as_millis(),
      constraints = %pk.S.num_cons,
      vars = %num_vars,
      "matrix_vector_multiply"
    );

    let (_mp_span, mp_t) = start_span!("prepare_multilinear_polys");
    let mut poly_Az = MultilinearPolynomial::new(Az);
    let mut poly_Bz = MultilinearPolynomial::new(Bz);
    let mut poly_Cz = MultilinearPolynomial::new(Cz);
    info!(elapsed_ms = %mp_t.elapsed().as_millis(), "prepare_multilinear_polys");

    // Initialize multi-round verifier circuit (will be filled as we go)
    let mut verifier_circuit = SpartanVerifierCircuit::<E>::default(num_rounds_x, num_rounds_y);
    let mut state = SatisfyingAssignment::<E>::initialize_multiround_witness(&pk.vc_shape)?;

    // Outer sum-check
    let (_sc_span, sc_t) = start_span!("outer_sumcheck");
    let r_x = SumcheckProof::<E>::prove_cubic_with_additive_term_zk(
      num_rounds_x,
      &taus,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_Cz,
      &mut verifier_circuit,
      &mut state,
      &pk.vc_shape,
      &pk.vc_ck,
      &mut transcript,
    )?;
    info!(elapsed_ms = %sc_t.elapsed().as_millis(), "outer_sumcheck");

    // Outer final round data
    verifier_circuit.claim_Az = poly_Az[0];
    verifier_circuit.claim_Bz = poly_Bz[0];
    verifier_circuit.claim_Cz = poly_Cz[0];
    verifier_circuit.tau_at_rx = EqPolynomial::new(taus).evaluate(&r_x);

    // Process the "outer final" round in the circuit and capture challenge
    let chals = SatisfyingAssignment::<E>::process_round(
      &mut state,
      &pk.vc_shape,
      &pk.vc_ck,
      &verifier_circuit,
      num_rounds_x,
      &mut transcript,
    )?;
    let r = chals[0];

    // Prepare inner polynomials
    let (_eval_rx_span, eval_rx_t) = start_span!("compute_eval_rx");
    let evals_rx = EqPolynomial::evals_from_points(&r_x);
    info!(elapsed_ms = %eval_rx_t.elapsed().as_millis(), "compute_eval_rx");

    let (_sparse_span, sparse_t) = start_span!("compute_eval_table_sparse");
    let (evals_A, evals_B, evals_C) = pk.S.bind_row_vars(&evals_rx);
    info!(elapsed_ms = %sparse_t.elapsed().as_millis(), "compute_eval_table_sparse");

    let (_abc_span, abc_t) = start_span!("prepare_poly_ABC");
    let poly_ABC: Vec<E::Scalar> = (0..evals_A.len())
      .into_par_iter()
      .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
      .collect();
    info!(elapsed_ms = %abc_t.elapsed().as_millis(), "prepare_poly_ABC");

    let (_z_span, z_t) = start_span!("prepare_poly_z");
    z.resize(num_vars * 2, E::Scalar::ZERO);
    let mut poly_ABC = MultilinearPolynomial::new(poly_ABC);
    let mut poly_z = MultilinearPolynomial::new(z);
    info!(elapsed_ms = %z_t.elapsed().as_millis(), "prepare_poly_z");

    // Inner sum-check
    let (_sc2_span, sc2_t) = start_span!("inner_sumcheck");
    let claim_inner_joint =
      verifier_circuit.claim_Az + r * verifier_circuit.claim_Bz + r * r * verifier_circuit.claim_Cz;

    let (r_y, evals) = SumcheckProof::<E>::prove_quad_zk(
      &claim_inner_joint,
      num_rounds_y,
      &mut poly_ABC,
      &mut poly_z,
      &mut verifier_circuit,
      &mut state,
      &pk.vc_shape,
      &pk.vc_ck,
      &mut transcript,
      num_rounds_x + 1,
    )?;
    let eval_Z = evals[1]; // evaluation of Z at r_y
    info!(elapsed_ms = %sc2_t.elapsed().as_millis(), "inner_sumcheck");

    // Compute final evaluations needed for the inner-final round
    let U_regular = U.to_regular_instance()?;
    let eval_X = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(U_regular.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      SparsePolynomial::new(num_rounds_y - 1, X).evaluate(&r_y[1..])
    };

    // compute eval_W = (eval_Z - r_y[0] * eval_X) / (1 - r_y[0]) because Z = (W, 1, X)
    let eval_W = (eval_Z - r_y[0] * eval_X) * (E::Scalar::ONE - r_y[0]).invert().unwrap();

    // Process the inner-final equality round
    // Set verifier circuit public values before processing inner-final round
    verifier_circuit.eval_W = eval_W;
    verifier_circuit.eval_X = eval_X;
    _ = SatisfyingAssignment::<E>::process_round(
      &mut state,
      &pk.vc_shape,
      &pk.vc_ck,
      &verifier_circuit,
      (num_rounds_x + 1) + num_rounds_y,
      &mut transcript,
    )?;

    // Process the dedicated commit-only round for eval_W
    let eval_w_commit_round = num_rounds_x + 1 + num_rounds_y + 1;
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut state,
      &pk.vc_shape,
      &pk.vc_ck,
      &verifier_circuit,
      eval_w_commit_round,
      &mut transcript,
    )?;

    // Finalize multi-round witness and construct NIFS proof
    let (_nifs_span, nifs_t) = start_span!("finalize_and_nifs");
    let (U_verifier, W_verifier) =
      SatisfyingAssignment::<E>::finalize_multiround_witness(&mut state, &pk.vc_shape)?;

    // Use the instance as produced by witness finalization; its public values
    // are exactly those absorbed during round 0 by the prover.
    let U_verifier_regular = U_verifier.to_regular_instance()?;
    let S_verifier = &pk.vc_shape_regular;
    let (random_U, random_W) = S_verifier.sample_random_instance_witness(&pk.vc_ck)?;
    let (nifs, folded_W) = NovaNIFS::<E>::prove(
      &pk.vc_ck,
      S_verifier,
      &random_U,
      &random_W,
      &U_verifier_regular,
      &W_verifier,
      &mut transcript,
    )?;
    info!(elapsed_ms = %nifs_t.elapsed().as_millis(), "finalize_and_nifs");

    // prove the claimed polynomial evaluation at point r_y[1..]
    let (_pcs_span, pcs_t) = start_span!("pcs_prove");
    let eval_arg = E::PCS::prove(
      &pk.ck,
      &pk.vc_ck,
      &mut transcript,
      &U_regular.comm_W,
      &W.W,
      &W.r_W,
      &r_y[1..],
      &U_verifier.comm_w_per_round[eval_w_commit_round],
      &state.r_w_per_round[eval_w_commit_round],
    )?;
    info!(elapsed_ms = %pcs_t.elapsed().as_millis(), "pcs_prove");

    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "spartan_zk_prove");
    Ok(SpartanZkSNARK {
      U_verifier,
      nifs,
      random_U,
      folded_W,
      eval_arg,
      U,
    })
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey) -> Result<Vec<E::Scalar>, SpartanError> {
    // Verify by checking the multi-round verifier instance via NIFS folding
    let (_verify_span, verify_t) = start_span!("spartan_zk_verify");
    let ck_verifier = &vk.vc_ck;
    let mut transcript = E::TE::new(b"SpartanZkSNARK");
    transcript.absorb(b"vk", &vk.digest()?);
    transcript.absorb(b"public_values", &self.U.public_values.as_slice());

    // Validate the provided split R1CS instance and advance the transcript
    self.U.validate(&vk.S, &mut transcript)?;

    // Recreate tau polynomial coefficients via Fiat-Shamir and advance transcript
    let (_tau_span, tau_t) = start_span!("compute_tau_verify");
    let num_rounds_x = vk.S.num_cons.log_2();
    let tau = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;
    info!(elapsed_ms = %tau_t.elapsed().as_millis(), "compute_tau_verify");

    // validate the provided multi-round verifier instance and advance transcript
    self.U_verifier.validate(&vk.vc_shape, &mut transcript)?;

    // Derive expected challenge counts from the original shape sizes
    let num_vars = vk.S.num_shared + vk.S.num_precommitted + vk.S.num_rest;
    let num_rounds_x = vk.S.num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let U_verifier_regular = self.U_verifier.to_regular_instance()?;

    let num_public_values = 3usize;
    let num_challenges = num_rounds_x + 1 + num_rounds_y;

    if U_verifier_regular.X.len() != num_challenges + num_public_values {
      return Err(SpartanError::ProofVerifyError {
        reason: format!(
          "Verifier instance has incorrect number of public IO: expected {}, got {}",
          num_challenges + num_public_values,
          U_verifier_regular.X.len()
        ),
      });
    }
    let challenges = &U_verifier_regular.X[0..num_challenges];
    let public_values = &U_verifier_regular.X[num_challenges..num_challenges + 3];

    let r_x = challenges[0..num_rounds_x].to_vec();
    let r = challenges[num_rounds_x]; // r for combining inner claims
    let r_y = challenges[num_rounds_x + 1..].to_vec();

    // compute eval_A, eval_B, eval_C at (r_x, r_y)
    let (_matrix_eval_span, matrix_eval_t) = start_span!("matrix_evaluations");
    let T_x = EqPolynomial::evals_from_points(&r_x);
    let T_y = EqPolynomial::evals_from_points(&r_y);
    let (eval_A, eval_B, eval_C) = vk.S.evaluate_with_tables(&T_x, &T_y);
    let quotient = eval_A + r * eval_B + r * r * eval_C;
    info!(elapsed_ms = %matrix_eval_t.elapsed().as_millis(), "matrix_evaluations");

    // Recompute eval_X from original circuit public IO at r_y[1..]
    let U_regular = self.U.to_regular_instance()?;

    let eval_X = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(U_regular.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars = vk.S.num_shared + vk.S.num_precommitted + vk.S.num_rest;
      SparsePolynomial::new(num_vars.log_2(), X).evaluate(&r_y[1..])
    };

    // Recompute tau(r_x) using the same tau polynomial challenges
    let tau_at_rx = tau.evaluate(&r_x);

    // Compare against the instance's public inputs [tau_at_rx, eval_X, quotient]
    if public_values[0] != tau_at_rx || public_values[1] != eval_X || public_values[2] != quotient {
      return Err(SpartanError::ProofVerifyError {
        reason:
          "Verifier instance public values do not match recomputed evaluations (tau_at_rx, eval_X, quotient)"
            .to_string(),
      });
    }

    // Finally, run NIFS verification using the same transcript
    let (_nifs_verify_span, nifs_verify_t) = start_span!("nifs_verify");
    let folded_U = self
      .nifs
      .verify(&mut transcript, &self.random_U, &U_verifier_regular)?;

    // Check satisfiability of the folded relaxed instance with the folded witness
    vk.vc_shape_regular
      .is_sat_relaxed(ck_verifier, &folded_U, &self.folded_W)
      .map_err(|e| SpartanError::ProofVerifyError {
        reason: format!("Folded instance not satisfiable: {e}"),
      })?;
    info!(elapsed_ms = %nifs_verify_t.elapsed().as_millis(), "nifs_verify");

    // Continue with PCS verification on the same transcript
    // Use the commitment from the dedicated eval_W commit-only last round
    let (_pcs_verify_span, pcs_verify_t) = start_span!("pcs_verify");
    let eval_w_commit_round = num_rounds_x + 1 + num_rounds_y + 1;
    E::PCS::verify(
      &vk.vk_ee,
      &vk.vc_ck,
      &mut transcript,
      &U_regular.comm_W,
      &r_y[1..],
      &self.U_verifier.comm_w_per_round[eval_w_commit_round],
      &self.eval_arg,
    )?;
    info!(elapsed_ms = %pcs_verify_t.elapsed().as_millis(), "pcs_verify");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "spartan_zk_verify");
    // Return original circuit public IO carried in the proof
    Ok(self.U.public_values.clone())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
  use tracing_subscriber::EnvFilter;

  #[cfg(feature = "jem")]
  #[global_allocator]
  static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

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
  fn test_zksnark() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true) // no bold colour codes
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();

    type E = crate::provider::PallasHyraxEngine;
    type S = SpartanZkSNARK<E>;
    test_zksnark_with::<E, S>();

    type E2 = crate::provider::T256HyraxEngine;
    type S2 = SpartanZkSNARK<E2>;
    test_zksnark_with::<E2, S2>();
  }

  fn test_zksnark_with<E: Engine, S: R1CSSNARKTrait<E>>() {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) = S::setup(circuit.clone()).unwrap();

    // generate pre-processed state for proving
    let prep_snark = S::prep_prove(&pk, circuit.clone(), false).unwrap();

    // generate a witness and proof
    let res = S::prove(&pk, circuit.clone(), &prep_snark, false);
    assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk);
    assert!(res.is_ok());
    assert_eq!(res.unwrap(), [<E as Engine>::Scalar::from(15u64)])
  }
}
