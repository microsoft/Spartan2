//! This module implements the spartan SNARK protocol.
//! It provides the prover and verifier keys, as well as the SNARK itself.
use crate::{
  CommitmentKey,
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
  r1cs::{SparseMatrix, SplitR1CSInstance, SplitR1CSShape},
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
use std::time::Instant;
use tracing::{debug, info, info_span};

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanProverKey<E: Engine> {
  ck: CommitmentKey<E>,
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

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSSNARK<E: Engine> {
  U: SplitR1CSInstance<E>,
  sc_proof_outer: SumcheckProof<E>,
  claims_outer: (E::Scalar, E::Scalar, E::Scalar),
  sc_proof_inner: SumcheckProof<E>,
  eval_W: E::Scalar,
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
}

impl<E: Engine> R1CSSNARKTrait<E> for R1CSSNARK<E> {
  type ProverKey = SpartanProverKey<E>;
  type VerifierKey = SpartanVerifierKey<E>;
  type PrepSNARK = PrepSNARK<E>;

  fn setup<C: SpartanCircuit<E>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError> {
    let S = ShapeCS::r1cs_shape(&circuit)?;
    let (ck, vk_ee) = SplitR1CSShape::commitment_key(&[&S])?;

    let vk: SpartanVerifierKey<E> = SpartanVerifierKey {
      S: S.clone(),
      vk_ee,
      digest: OnceCell::new(),
    };
    let pk = Self::ProverKey {
      ck,
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
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;

    let (_poly_tau_span, poly_tau_t) = start_span!("prepare_poly_tau");
    let mut poly_tau = MultilinearPolynomial::new(tau.evals());
    info!(elapsed_ms = %poly_tau_t.elapsed().as_millis(), "prepare_poly_tau");

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

    let comb_func_outer =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };
    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_additive_term(
      &E::Scalar::ZERO, // claim is zero
      num_rounds_x,
      &mut poly_tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_Cz,
      comb_func_outer,
      &mut transcript,
    )?;

    // claims from the end of sum-check
    let (claim_Az, claim_Bz, claim_Cz): (E::Scalar, E::Scalar, E::Scalar) =
      (claims_outer[1], claims_outer[2], claims_outer[3]);
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
    let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&pk.S, &evals_rx);
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
    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      comb_func,
      &mut transcript,
    )?;
    info!(elapsed_ms = %sc2_t.elapsed().as_millis(), "inner_sumcheck");

    let (_pcs_span, pcs_t) = start_span!("pcs_prove");
    let U_regular = U.to_regular_instance()?;
    let (eval_W, eval_arg) = E::PCS::prove(
      &pk.ck,
      &mut transcript,
      &U_regular.comm_W,
      &W.W,
      &W.r_W,
      &r_y[1..],
    )?;
    info!(elapsed_ms = %pcs_t.elapsed().as_millis(), "pcs_prove");

    Ok(R1CSSNARK {
      U,
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      sc_proof_inner,
      eval_W,
      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey) -> Result<Vec<E::Scalar>, SpartanError> {
    let (_verify_span, verify_t) = start_span!("r1cs_snark_verify");
    let mut transcript = E::TE::new(b"R1CSSNARK");

    // append the digest of R1CS matrices
    transcript.absorb(b"vk", &vk.digest()?);

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
    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          r_x: &[E::Scalar],
                          r_y: &[E::Scalar]|
     -> Vec<E::Scalar> {
      let evaluate_with_table =
        |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
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

      let (T_x, T_y) = rayon::join(
        || EqPolynomial::evals_from_points(r_x),
        || EqPolynomial::evals_from_points(r_y),
      );

      (0..M_vec.len())
        .into_par_iter()
        .map(|i| evaluate_with_table(M_vec[i], &T_x, &T_y))
        .collect()
    };

    let evals = multi_evaluate(&[&vk.S.A, &vk.S.B, &vk.S.C], &r_x, &r_y);

    let claim_inner_final_expected = (evals[0] + r * evals[1] + r * r * evals[2]) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    info!(elapsed_ms = %matrix_eval_t.elapsed().as_millis(), "matrix_evaluations");
    info!(elapsed_ms = %inner_sumcheck_t.elapsed().as_millis(), "inner_sumcheck_verify");

    // verify
    let (_pcs_verify_span, pcs_verify_t) = start_span!("pcs_verify");
    E::PCS::verify(
      &vk.vk_ee,
      &mut transcript,
      &U_regular.comm_W,
      &r_y[1..],
      &self.eval_W,
      &self.eval_arg,
    )?;
    info!(elapsed_ms = %pcs_verify_t.elapsed().as_millis(), "pcs_verify");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "r1cs_snark_verify");
    Ok(self.U.public_values.clone())
  }
}

// Test-only utilities to extract prover artifacts needed to instantiate the
// SpartanVerifierCircuit with concrete univariate polynomials and challenges.
#[cfg(test)]
pub(crate) mod test_utils {
  use crate::traits::circuit::{MultiRoundCircuit, SpartanCircuit};
  use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
  use std::marker::PhantomData;

  #[derive(Clone)]
  pub(crate) struct MultiRoundAdapter<E: crate::traits::Engine, C: MultiRoundCircuit<E>> {
    pub circuit: C,
    pub _e: PhantomData<E>,
  }

  impl<E: crate::traits::Engine, C: MultiRoundCircuit<E>> SpartanCircuit<E>
    for MultiRoundAdapter<E, C>
  {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
      self.circuit.public_values()
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _cs: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _cs: &mut CS,
      _shared: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![])
    }

    fn num_challenges(&self) -> usize {
      (0..self.circuit.num_rounds())
        .map(|r| self.circuit.num_challenges(r).unwrap())
        .sum()
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _shared: &[AllocatedNum<E::Scalar>],
      _precommitted: &[AllocatedNum<E::Scalar>],
      challenges: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
      let mut prior_vars: Vec<Vec<AllocatedNum<E::Scalar>>> = Vec::new();
      let mut prev_chals: Vec<Vec<AllocatedNum<E::Scalar>>> = Vec::new();
      let mut offset = 0usize;

      for round_index in 0..self.circuit.num_rounds() {
        let n = self.circuit.num_challenges(round_index)?;
        let chal_slice = match (challenges, n) {
          (Some(all), m) if m > 0 => Some(&all[offset..offset + m]),
          _ => None,
        };
        let (vars, cals) =
          self
            .circuit
            .rounds(cs, round_index, &prior_vars, &prev_chals, chal_slice)?;
        prior_vars.push(vars);
        prev_chals.push(cals);
        offset += n;
      }
      Ok(())
    }
  }

  use super::*;

  // ----------------------------------------------------------------------------------------------
  // Interactive, round-by-round builder (test-only)
  // ----------------------------------------------------------------------------------------------
  use crate::polys::eq::EqPolynomial;
  use crate::polys::multilinear::{MultilinearPolynomial, SparsePolynomial};
  use crate::polys::univariate::UniPoly;

  fn eval_points_cubic_with_additive_term<E: Engine, F>(
    poly_a: &MultilinearPolynomial<E::Scalar>,
    poly_b: &MultilinearPolynomial<E::Scalar>,
    poly_c: &MultilinearPolynomial<E::Scalar>,
    poly_d: &MultilinearPolynomial<E::Scalar>,
    comb: &F,
  ) -> (E::Scalar, E::Scalar, E::Scalar)
  where
    F: Fn(&E::Scalar, &E::Scalar, &E::Scalar, &E::Scalar) -> E::Scalar + Sync,
  {
    let len = poly_a.Z.len() / 2;
    (0..len)
      .into_par_iter()
      .map(|i| {
        let a_low = poly_a[i];
        let a_high = poly_a[i + len];
        let b_low = poly_b[i];
        let b_high = poly_b[i + len];
        let c_low = poly_c[i];
        let c_high = poly_c[i + len];
        let d_low = poly_d[i];
        let d_high = poly_d[i + len];

        let eval0 = comb(&a_low, &b_low, &c_low, &d_low);
        let a_bnd = a_high + a_high - a_low;
        let b_bnd = b_high + b_high - b_low;
        let c_bnd = c_high + c_high - c_low;
        let d_bnd = d_high + d_high - d_low;
        let eval2 = comb(&a_bnd, &b_bnd, &c_bnd, &d_bnd);

        let a_bnd2 = a_bnd + a_high - a_low;
        let b_bnd2 = b_bnd + b_high - b_low;
        let c_bnd2 = c_bnd + c_high - c_low;
        let d_bnd2 = d_bnd + d_high - d_low;
        let eval3 = comb(&a_bnd2, &b_bnd2, &c_bnd2, &d_bnd2);
        (eval0, eval2, eval3)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
        |mut acc, v| {
          acc.0 += v.0;
          acc.1 += v.1;
          acc.2 += v.2;
          acc
        },
      )
  }

  fn eval_points_quad<E: Engine, F>(
    poly_a: &MultilinearPolynomial<E::Scalar>,
    poly_b: &MultilinearPolynomial<E::Scalar>,
    comb: &F,
  ) -> (E::Scalar, E::Scalar)
  where
    F: Fn(&E::Scalar, &E::Scalar) -> E::Scalar + Sync,
  {
    let len = poly_a.Z.len() / 2;
    (0..len)
      .into_par_iter()
      .map(|i| {
        let a_low = poly_a[i];
        let a_high = poly_a[i + len];
        let b_low = poly_b[i];
        let b_high = poly_b[i + len];
        let eval0 = comb(&a_low, &b_low);
        let a_bnd = a_high + a_high - a_low;
        let b_bnd = b_high + b_high - b_low;
        let eval2 = comb(&a_bnd, &b_bnd);
        (eval0, eval2)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO),
        |mut acc, v| {
          acc.0 += v.0;
          acc.1 += v.1;
          acc
        },
      )
  }

  pub(crate) struct InteractiveSession<E: Engine> {
    pk: SpartanProverKey<E>,
    u: SplitR1CSInstance<E>,
    w: crate::r1cs::R1CSWitness<E>,
    transcript: E::TE,
    // common
    pub num_rounds_x: usize,
    pub num_rounds_y: usize,
    // outer state
    poly_tau: MultilinearPolynomial<E::Scalar>,
    poly_az: MultilinearPolynomial<E::Scalar>,
    poly_bz: MultilinearPolynomial<E::Scalar>,
    poly_cz: MultilinearPolynomial<E::Scalar>,
    pub outer_round: usize,
    claim_outer_round: E::Scalar,
    pub r_x: Vec<E::Scalar>,
    // inner state
    r_inner: Option<E::Scalar>,
    claim_inner_joint: Option<E::Scalar>,
    poly_abc: Option<MultilinearPolynomial<E::Scalar>>, // constructed after outer
    poly_z: Option<MultilinearPolynomial<E::Scalar>>,   // constructed after outer
    pub inner_round: usize,
    pub r_y: Vec<E::Scalar>,
    // Split-phase helpers for interleaving with external challenge source
    last_outer_coeffs: Option<[E::Scalar; 4]>,
    last_inner_coeffs: Option<[E::Scalar; 3]>,
  }

  impl<E: Engine> InteractiveSession<E> {
    pub fn begin<C: crate::traits::circuit::MultiRoundCircuit<E> + Clone>(
      circuit: C,
    ) -> Result<Self, SpartanError> {
      let adapter = MultiRoundAdapter::<E, _> {
        circuit: circuit.clone(),
        _e: core::marker::PhantomData,
      };
      let (pk, _vk) = <R1CSSNARK<E> as R1CSSNARKTrait<E>>::setup(adapter.clone())?;
      let mut ps = <R1CSSNARK<E> as R1CSSNARKTrait<E>>::prep_prove(&pk, adapter.clone(), false)?;

      let mut transcript = E::TE::new(b"R1CSSNARK");
      transcript.absorb(b"vk", &pk.vk_digest);
      let public_values = circuit
        .public_values()
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Circuit does not provide public IO: {e}"),
        })?;
      transcript.absorb(b"public_values", &public_values.as_slice());

      let (u, w) = SatisfyingAssignment::r1cs_instance_and_witness(
        &mut ps.ps,
        &pk.S,
        &pk.ck,
        &adapter,
        false,
        &mut transcript,
      )?;

      let z = [
        w.W.clone(),
        vec![E::Scalar::ONE],
        u.public_values.clone(),
        u.challenges.clone(),
      ]
      .concat();

      let num_vars = pk.S.num_shared + pk.S.num_precommitted + pk.S.num_rest;
      let (num_rounds_x, num_rounds_y) = (
        usize::try_from(pk.S.num_cons.ilog2()).unwrap(),
        usize::try_from(num_vars.ilog2()).unwrap() + 1,
      );

      let tau = (0..num_rounds_x)
        .map(|_| transcript.squeeze(b"t"))
        .collect::<Result<EqPolynomial<_>, SpartanError>>()?;

      let (az, bz, cz) = pk.S.multiply_vec(&z)?;
      let poly_tau = MultilinearPolynomial::new(tau.evals());
      let poly_az = MultilinearPolynomial::new(az);
      let poly_bz = MultilinearPolynomial::new(bz);
      let poly_cz = MultilinearPolynomial::new(cz);

      Ok(Self {
        pk,
        u,
        w,
        transcript,
        num_rounds_x,
        num_rounds_y,
        poly_tau,
        poly_az,
        poly_bz,
        poly_cz,
        outer_round: 0,
        claim_outer_round: E::Scalar::ZERO,
        r_x: Vec::with_capacity(num_rounds_x),
        r_inner: None,
        claim_inner_joint: None,
        poly_abc: None,
        poly_z: None,
        inner_round: 0,
        r_y: Vec::with_capacity(num_rounds_y),
        last_outer_coeffs: None,
        last_inner_coeffs: None,
      })
    }

    /// Produces the outer-round univariate coefficients for the current round
    /// without consuming a challenge. Must be followed by `outer_bind` with
    /// the driver-provided challenge to advance state.
    pub fn outer_produce_poly(&mut self) -> Result<[E::Scalar; 4], SpartanError> {
      assert!(self.outer_round < self.num_rounds_x);
      let comb = |a: &E::Scalar, b: &E::Scalar, c: &E::Scalar, d: &E::Scalar| -> E::Scalar {
        *a * (*b * *c - *d)
      };
      let (eval0, eval2, eval3) = eval_points_cubic_with_additive_term::<E, _>(
        &self.poly_tau,
        &self.poly_az,
        &self.poly_bz,
        &self.poly_cz,
        &comb,
      );
      // Interpolate polynomial coefficients from evals at t in {0,1,2,3}
      let evals = vec![eval0, self.claim_outer_round - eval0, eval2, eval3];
      let poly = UniPoly::from_evals(&evals)?;
      let coeffs = [
        poly.coeffs[0],
        poly.coeffs[1],
        poly.coeffs[2],
        poly.coeffs[3],
      ];
      self.last_outer_coeffs = Some(coeffs);
      Ok(coeffs)
    }

    /// Consumes an outer-round challenge to advance the state to the next round.
    pub fn outer_bind(&mut self, r_i: E::Scalar) {
      assert!(self.outer_round < self.num_rounds_x);
      let [c0, c1, c2, c3] = self
        .last_outer_coeffs
        .expect("call outer_produce_poly first");
      self.r_x.push(r_i);
      let r2 = r_i * r_i;
      let r3 = r2 * r_i;
      self.claim_outer_round = c0 + c1 * r_i + c2 * r2 + c3 * r3;
      rayon::join(
        || self.poly_tau.bind_poly_var_top(&r_i),
        || {
          rayon::join(
            || self.poly_az.bind_poly_var_top(&r_i),
            || self.poly_bz.bind_poly_var_top(&r_i),
          );
          self.poly_cz.bind_poly_var_top(&r_i)
        },
      );
      self.outer_round += 1;
      // clear cached coeffs for safety
      self.last_outer_coeffs = None;
    }

    pub fn outer_finalize(&self) -> (E::Scalar, E::Scalar, E::Scalar) {
      (self.poly_az[0], self.poly_bz[0], self.poly_cz[0])
    }

    /// Returns tau evaluated at the verifier's outer challenges r_x, i.e., tau(r_x).
    /// Must be called after all outer rounds have been completed.
    pub fn outer_tau_eval(&self) -> E::Scalar {
      assert!(self.outer_round == self.num_rounds_x);
      self.poly_tau[0]
    }

    /// Initializes inner round state with an externally provided r.
    pub fn inner_setup_given_r(&mut self, r_inner: E::Scalar) -> Result<E::Scalar, SpartanError> {
      assert!(self.outer_round == self.num_rounds_x);
      let (claim_az, claim_bz, claim_cz) = self.outer_finalize();
      let claim_inner_joint = claim_az + r_inner * claim_bz + r_inner * r_inner * claim_cz;
      self.r_inner = Some(r_inner);
      self.claim_inner_joint = Some(claim_inner_joint);

      // Build inner polynomials
      let evals_rx = EqPolynomial::evals_from_points(&self.r_x);
      let (evals_a, evals_b, evals_c) = compute_eval_table_sparse(&self.pk.S, &evals_rx);
      let poly_abc: Vec<E::Scalar> = (0..evals_a.len())
        .into_par_iter()
        .map(|i| evals_a[i] + r_inner * evals_b[i] + r_inner * r_inner * evals_c[i])
        .collect();

      // z vector
      let mut z = [
        self.w.W.clone(),
        vec![E::Scalar::ONE],
        self.u.public_values.clone(),
        self.u.challenges.clone(),
      ]
      .concat();
      let num_vars = self.pk.S.num_shared + self.pk.S.num_precommitted + self.pk.S.num_rest;
      z.resize(num_vars * 2, E::Scalar::ZERO);

      self.poly_abc = Some(MultilinearPolynomial::new(poly_abc));
      self.poly_z = Some(MultilinearPolynomial::new(z));
      Ok(claim_inner_joint)
    }

    /// Produces inner-round univariate coefficients for current inner round
    /// without consuming a challenge. Must be followed by `inner_bind`.
    pub fn inner_produce_poly(&mut self) -> Result<[E::Scalar; 3], SpartanError> {
      assert!(self.inner_round < self.num_rounds_y);
      let poly_abc = self.poly_abc.as_ref().expect("inner_setup first");
      let poly_z = self.poly_z.as_ref().expect("inner_setup first");
      let claim = *self.claim_inner_joint.as_ref().expect("inner_setup first");
      let comb = |a: &E::Scalar, b: &E::Scalar| -> E::Scalar { *a * *b };
      let (eval0, eval2) = eval_points_quad::<E, _>(poly_abc, poly_z, &comb);
      let evals = vec![eval0, claim - eval0, eval2];
      let poly = UniPoly::from_evals(&evals)?;
      let coeffs = [poly.coeffs[0], poly.coeffs[1], poly.coeffs[2]];
      self.last_inner_coeffs = Some(coeffs);
      Ok(coeffs)
    }

    /// Consumes an inner-round challenge to advance the state to the next round.
    pub fn inner_bind(&mut self, r_i: E::Scalar) {
      assert!(self.inner_round < self.num_rounds_y);
      let [c0, c1, c2] = self
        .last_inner_coeffs
        .expect("call inner_produce_poly first");
      self.r_y.push(r_i);
      let r2 = r_i * r_i;
      let claim = c0 + c1 * r_i + c2 * r2;
      self.claim_inner_joint = Some(claim);
      let poly_abc = self.poly_abc.as_mut().expect("inner_setup first");
      let poly_z = self.poly_z.as_mut().expect("inner_setup first");
      rayon::join(
        || poly_abc.bind_poly_var_top(&r_i),
        || poly_z.bind_poly_var_top(&r_i),
      );
      self.inner_round += 1;
      self.last_inner_coeffs = None;
    }

    pub fn inner_finalize(
      &mut self,
    ) -> Result<(E::Scalar, E::Scalar, (E::Scalar, E::Scalar, E::Scalar)), SpartanError> {
      let u_regular = self.u.to_regular_instance()?;
      let (eval_w, _eval_arg) = E::PCS::prove(
        &self.pk.ck,
        &mut self.transcript,
        &u_regular.comm_W,
        &self.w.W,
        &self.w.r_W,
        &self.r_y[1..],
      )?;

      let eval_x = {
        let x_vec = vec![E::Scalar::ONE]
          .into_iter()
          .chain(u_regular.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        let num_vars = self.pk.S.num_shared + self.pk.S.num_precommitted + self.pk.S.num_rest;
        SparsePolynomial::new(num_vars.log_2(), x_vec).evaluate(&self.r_y[1..])
      };

      // A/B/C evaluations at (r_x, r_y)
      let evals = {
        let multi = |mats: &[&SparseMatrix<E::Scalar>], r_x: &[E::Scalar], r_y: &[E::Scalar]| {
          let (t_x, t_y) = rayon::join(
            || EqPolynomial::evals_from_points(r_x),
            || EqPolynomial::evals_from_points(r_y),
          );
          mats
            .into_par_iter()
            .map(|m| {
              m.indptr
                .par_windows(2)
                .enumerate()
                .map(|(row_idx, ptrs)| {
                  m.get_row_unchecked(ptrs.try_into().unwrap())
                    .map(|(val, col_idx)| {
                      let prod = t_x[row_idx] * t_y[*col_idx];
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
            })
            .collect::<Vec<E::Scalar>>()
        };
        multi(
          &[&self.pk.S.A, &self.pk.S.B, &self.pk.S.C],
          &self.r_x,
          &self.r_y,
        )
      };

      Ok((eval_w, eval_x, (evals[0], evals[1], evals[2])))
    }
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
    let res = S::prove(&pk, circuit.clone(), &prep_snark, false);
    assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk);
    assert!(res.is_ok());
    assert_eq!(res.unwrap(), [<E as Engine>::Scalar::from(15u64)])
  }
}
