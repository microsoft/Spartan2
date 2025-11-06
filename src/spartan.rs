//! This module implements the Spartan SNARK protocol.
//! It provides the prover and verifier keys, as well as the SNARK itself.
use crate::{
  Blind, CommitmentKey, MULTIROUND_COMMITMENT_WIDTH,
  bellpepper::{
    r1cs::{PrecommittedState, SpartanShape, SpartanWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  digest::{DigestComputer, SimpleDigestible},
  errors::SpartanError,
  math::Math,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
  },
  r1cs::{SplitR1CSInstance, SplitR1CSShape},
  start_span,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::SpartanCircuit,
    pcs::PCSEngineTrait,
    snark::{DigestHelperTrait, R1CSSNARKTrait, SpartanDigest},
    transcript::TranscriptEngineTrait,
  },
};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanProverKey<E: Engine> {
  ck: CommitmentKey<E>,
  ck_s: CommitmentKey<E>,
  S: SplitR1CSShape<E>,
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
  ck_s: CommitmentKey<E>,
  S: SplitR1CSShape<E>,
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
pub struct SpartanPrepSNARK<E: Engine> {
  ps: PrecommittedState<E>,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanSNARK<E: Engine> {
  U: SplitR1CSInstance<E>,
  sc_proof_outer: SumcheckProof<E>,
  claims_outer: (E::Scalar, E::Scalar, E::Scalar),
  sc_proof_inner: SumcheckProof<E>,
  eval_W: E::Scalar,
  blind_eval_W: Blind<E>, // it is okay to send the blind since we are targeting non-zk
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
}

impl<E: Engine> R1CSSNARKTrait<E> for SpartanSNARK<E> {
  type ProverKey = SpartanProverKey<E>;
  type VerifierKey = SpartanVerifierKey<E>;
  type PrepSNARK = SpartanPrepSNARK<E>;

  fn setup<C: SpartanCircuit<E>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError> {
    let S = ShapeCS::r1cs_shape(&circuit)?;
    let (ck, vk_ee) = SplitR1CSShape::commitment_key(&[&S])?;
    let (ck_s, _) = E::PCS::setup(b"ck_s", 1, MULTIROUND_COMMITMENT_WIDTH); // for committing to a scalar

    let vk: SpartanVerifierKey<E> = SpartanVerifierKey {
      S: S.clone(),
      vk_ee,
      ck_s: ck_s.clone(),
      digest: OnceCell::new(),
    };
    let pk = Self::ProverKey {
      ck,
      ck_s,
      S,
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

    Ok(SpartanPrepSNARK { ps })
  }

  /// produces a succinct proof of satisfiability of an R1CS instance
  fn prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    prep_snark: &Self::PrepSNARK,
    is_small: bool,
  ) -> Result<Self, SpartanError> {
    let (_prove_span, prove_t) = start_span!("spartan_snark_prove");
    let mut prep_snark = prep_snark.clone(); // make a copy so we can modify it

    let mut transcript = E::TE::new(b"SpartanSNARK");
    transcript.absorb(b"vk", &pk.vk_digest);

    let public_values = circuit
      .public_values()
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Circuit does not provide public IO: {e}"),
      })?;

    // absorb the public values into the transcript
    transcript.absorb(b"public_values", &public_values.as_slice());

    let (U, W) = SatisfyingAssignment::r1cs_instance_and_witness(
      &mut prep_snark.ps,
      &pk.S,
      &pk.ck,
      &circuit,
      is_small,
      &mut transcript,
    )?;

    // compute the full satisfying assignment by concatenating W.W, 1, and U.X
    let mut z = [
      W.W.clone(),
      vec![E::Scalar::ONE],
      U.public_values.clone(),
      U.challenges.clone(),
    ]
    .concat();

    let num_vars = pk.S.num_shared + pk.S.num_precommitted + pk.S.num_rest;
    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(pk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check preparation
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let (_mv_span, mv_t) = start_span!("matrix_vector_multiply");
    let (Az, Bz, Cz) = pk.S.multiply_vec(&z)?;
    info!(
      elapsed_ms = %mv_t.elapsed().as_millis(),
      constraints = %pk.S.num_cons,
      vars = %num_vars,
      "matrix_vector_multiply"
    );

    let (_mp_span, mp_t) = start_span!("prepare_multilinear_polys");
    let (mut poly_Az, mut poly_Bz, mut poly_Cz) = (
      MultilinearPolynomial::new(Az),
      MultilinearPolynomial::new(Bz),
      MultilinearPolynomial::new(Cz),
    );
    info!(elapsed_ms = %mp_t.elapsed().as_millis(), "prepare_multilinear_polys");

    // outer sum-check
    let (_sc_span, sc_t) = start_span!("outer_sumcheck");

    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_three_inputs(
      &E::Scalar::ZERO, // claim is zero
      tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_Cz,
      &mut transcript,
    )?;

    // claims from the end of sum-check
    let (claim_Az, claim_Bz, claim_Cz): (E::Scalar, E::Scalar, E::Scalar) =
      (claims_outer[0], claims_outer[1], claims_outer[2]);
    transcript.absorb(b"claims_outer", &[claim_Az, claim_Bz, claim_Cz].as_slice());
    info!(elapsed_ms = %sc_t.elapsed().as_millis(), "outer_sumcheck");

    // inner sum-check preparation
    let (_r_span, r_t) = start_span!("prepare_inner_claims");
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;
    info!(elapsed_ms = %r_t.elapsed().as_millis(), "prepare_inner_claims");

    let (_eval_rx_span, eval_rx_t) = start_span!("compute_eval_rx");
    let evals_rx = EqPolynomial::evals_from_points(&r_x.clone());
    info!(elapsed_ms = %eval_rx_t.elapsed().as_millis(), "compute_eval_rx");

    let (_sparse_span, sparse_t) = start_span!("compute_eval_table_sparse");
    let (evals_A, evals_B, evals_C) = pk.S.bind_row_vars(&evals_rx);
    info!(elapsed_ms = %sparse_t.elapsed().as_millis(), "compute_eval_table_sparse");

    let (_abc_span, abc_t) = start_span!("prepare_poly_ABC");
    assert_eq!(evals_A.len(), evals_B.len());
    assert_eq!(evals_A.len(), evals_C.len());
    let poly_ABC = (0..evals_A.len())
      .into_par_iter()
      .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
      .collect::<Vec<E::Scalar>>();
    info!(elapsed_ms = %abc_t.elapsed().as_millis(), "prepare_poly_ABC");

    let (_z_span, z_t) = start_span!("prepare_poly_z");
    let poly_z = {
      z.resize(num_vars * 2, E::Scalar::ZERO);
      z
    };
    info!(elapsed_ms = %z_t.elapsed().as_millis(), "prepare_poly_z");

    // inner sum-check
    let (_sc2_span, sc2_t) = start_span!("inner_sumcheck");

    debug!("Proving inner sum-check with {} rounds", num_rounds_y);
    debug!(
      "Inner sum-check sizes - poly_ABC: {}, poly_z: {}",
      poly_ABC.len(),
      poly_z.len()
    );
    let comb_func = |poly_A_comp: &E::Scalar, poly_B_comp: &E::Scalar| -> E::Scalar {
      *poly_A_comp * *poly_B_comp
    };
    let (sc_proof_inner, r_y, claims_inner) = SumcheckProof::prove_quad(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      comb_func,
      &mut transcript,
    )?;
    let eval_Z = claims_inner[1]; // evaluation of Z at r_y
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

    let (_pcs_span, pcs_t) = start_span!("pcs_prove");
    let blind_eval_W = E::PCS::blind(&pk.ck_s, 1); // blind for committing to eval_W
    let comm_eval_W = E::PCS::commit(&pk.ck_s, &[eval_W], &blind_eval_W, false)?; // commitment to eval_W
    let U_regular = U.to_regular_instance()?;
    let eval_arg = E::PCS::prove(
      &pk.ck,
      &pk.ck_s,
      &mut transcript,
      &U_regular.comm_W,
      &W.W,
      &W.r_W,
      &r_y[1..],
      &comm_eval_W,
      &blind_eval_W,
    )?;
    info!(elapsed_ms = %pcs_t.elapsed().as_millis(), "pcs_prove");

    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "spartan_snark_prove");
    Ok(SpartanSNARK {
      U,
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      sc_proof_inner,
      eval_W,
      blind_eval_W,
      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey) -> Result<Vec<E::Scalar>, SpartanError> {
    let (_verify_span, verify_t) = start_span!("spartan_snark_verify");
    let mut transcript = E::TE::new(b"SpartanSNARK");

    // append the digest of R1CS matrices
    transcript.absorb(b"vk", &vk.digest()?);
    transcript.absorb(b"public_values", &self.U.public_values.as_slice());

    // validate the provided split R1CS instance and convert to regular instance
    self.U.validate(&vk.S, &mut transcript)?;
    let U_regular = self.U.to_regular_instance()?;

    let num_vars = vk.S.num_shared + vk.S.num_precommitted + vk.S.num_rest;

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(vk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(num_vars.ilog2()).unwrap() + 1),
    );

    info!(
      "Verifying R1CS SNARK with {} rounds for outer sum-check and {} rounds for inner sum-check",
      num_rounds_x, num_rounds_y
    );

    // outer sum-check
    let (_tau_span, tau_t) = start_span!("compute_tau_verify");
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;
    info!(elapsed_ms = %tau_t.elapsed().as_millis(), "compute_tau_verify");

    let (_outer_sumcheck_span, outer_sumcheck_t) = start_span!("outer_sumcheck_verify");
    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(E::Scalar::ZERO, num_rounds_x, 3, &mut transcript)?;

    // verify claim_outer_final
    let (claim_Az, claim_Bz, claim_Cz) = self.claims_outer;
    let taus_bound_rx = tau.evaluate(&r_x);
    let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
    if claim_outer_final != claim_outer_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    info!(elapsed_ms = %outer_sumcheck_t.elapsed().as_millis(), "outer_sumcheck_verify");

    transcript.absorb(
      b"claims_outer",
      &[
        self.claims_outer.0,
        self.claims_outer.1,
        self.claims_outer.2,
      ]
      .as_slice(),
    );

    // inner sum-check
    let (_inner_sumcheck_span, inner_sumcheck_t) = start_span!("inner_sumcheck_verify");
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint =
      self.claims_outer.0 + r * self.claims_outer.1 + r * r * self.claims_outer.2;

    let (claim_inner_final, r_y) =
      self
        .sc_proof_inner
        .verify(claim_inner_joint, num_rounds_y, 2, &mut transcript)?;

    // verify claim_inner_final
    let eval_Z = {
      let eval_X = {
        // public IO is (1, X)
        let X = vec![E::Scalar::ONE]
          .into_iter()
          .chain(U_regular.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        SparsePolynomial::new(num_vars.log_2(), X).evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    // compute evaluations of R1CS matrices
    let (_matrix_eval_span, matrix_eval_t) = start_span!("matrix_evaluations");
    let T_x = EqPolynomial::evals_from_points(&r_x);
    let T_y = EqPolynomial::evals_from_points(&r_y);
    let (eval_A, eval_B, eval_C) = vk.S.evaluate_with_tables(&T_x, &T_y);

    let claim_inner_final_expected = (eval_A + r * eval_B + r * r * eval_C) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    info!(elapsed_ms = %matrix_eval_t.elapsed().as_millis(), "matrix_evaluations");
    info!(elapsed_ms = %inner_sumcheck_t.elapsed().as_millis(), "inner_sumcheck_verify");

    // verify
    let (_pcs_verify_span, pcs_verify_t) = start_span!("pcs_verify");
    let comm_eval_W = E::PCS::commit(&vk.ck_s, &[self.eval_W], &self.blind_eval_W, false)?; // commitment to eval_W
    E::PCS::verify(
      &vk.vk_ee,
      &vk.ck_s,
      &mut transcript,
      &U_regular.comm_W,
      &r_y[1..],
      &comm_eval_W,
      &self.eval_arg,
    )?;
    info!(elapsed_ms = %pcs_verify_t.elapsed().as_millis(), "pcs_verify");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "spartan_snark_verify");
    Ok(self.U.public_values.clone())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
  use tracing_subscriber::EnvFilter;

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
    type S = SpartanSNARK<E>;
    test_snark_with::<E, S>();

    type E2 = crate::provider::T256HyraxEngine;
    type S2 = SpartanSNARK<E2>;
    test_snark_with::<E2, S2>();
  }

  fn test_snark_with<E: Engine, S: R1CSSNARKTrait<E>>() {
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
