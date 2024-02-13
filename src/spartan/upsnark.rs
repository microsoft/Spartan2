//! This module implements `RelaxedR1CSSNARKTrait` using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
//! This version of Spartan does not use preprocessing so the verifier keeps the entire
//! description of R1CS matrices. This is essentially optimal for the verifier when using
//! an IPA-based polynomial commitment scheme.
//! The difference between this file and snark.rs is that it trims out the "Relaxed" parts 
//! and only works with (normal) R1CS, making it more efficient.
//! This basic R1CSStruct also implements "uniform" and "precommitted" traits. 

use crate::{
  bellpepper::{
    r1cs::{SpartanShape, SpartanWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  digest::{DigestComputer, SimpleDigestible},
  errors::SpartanError,
  r1cs::{R1CSShape, R1CSInstance},
  spartan::{
    polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial, multilinear::SparsePolynomial},
    sumcheck::SumcheckProof,
    // PolyEvalInstance, PolyEvalWitness,
  },
  traits::{
    commitment::CommitmentTrait, evaluation::EvaluationEngineTrait, snark::RelaxedR1CSSNARKTrait, 
    upsnark::{UniformSNARKTrait, PrecommittedSNARKTrait}, 
    Group, TranscriptEngineTrait,
  },
  Commitment, CommitmentKey, CompressedCommitment,
};
use bellpepper_core::{Circuit, ConstraintSystem};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<G: Group, EE: EvaluationEngineTrait<G>> {
  ck: CommitmentKey<G>,
  pk_ee: EE::ProverKey,
  S: R1CSShape<G>,
  vk_digest: G::Scalar, // digest of the verifier's key
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<G: Group, EE: EvaluationEngineTrait<G>> {
  vk_ee: EE::VerifierKey,
  S: R1CSShape<G>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<G::Scalar>,
}

impl<G: Group, EE: EvaluationEngineTrait<G>> SimpleDigestible for VerifierKey<G, EE> {}

impl<G: Group, EE: EvaluationEngineTrait<G>> VerifierKey<G, EE> {
  fn new(shape: R1CSShape<G>, vk_ee: EE::VerifierKey) -> Self {
    VerifierKey {
      vk_ee,
      S: shape,
      digest: OnceCell::new(),
    }
  }

  /// Returns the digest of the verifier's key.
  pub fn digest(&self) -> G::Scalar {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::<G::Scalar, _>::new(self);
        dc.digest()
      })
      .cloned()
      .expect("Failure to retrieve digest!")
  }
}

/// A uniform version fo the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct UniformProverKey<G: Group, EE: EvaluationEngineTrait<G>> {
  ck: CommitmentKey<G>,
  pk_ee: EE::ProverKey,
  S: R1CSShape<G>,
  S_single: R1CSShape<G>, 
  num_steps: usize, // Number of steps
  vk_digest: G::Scalar, // digest of the verifier's key
}


/// A uniform version of the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct UniformVerifierKey<G: Group, EE: EvaluationEngineTrait<G>> {
  vk_ee: EE::VerifierKey,
  S: R1CSShape<G>, // The full shape
  S_single: R1CSShape<G>, // A single step's shape
  num_steps: usize, // Number of steps
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<G::Scalar>,
}

impl<G: Group, EE: EvaluationEngineTrait<G>> SimpleDigestible for UniformVerifierKey<G, EE> {}

impl<G: Group, EE: EvaluationEngineTrait<G>> UniformVerifierKey<G, EE> {
  fn new(shape: R1CSShape<G>, vk_ee: EE::VerifierKey, shape_single: R1CSShape<G>, num_steps: usize) -> Self {
    UniformVerifierKey {
      vk_ee,
      S: shape,
      S_single: shape_single,
      num_steps: num_steps,
      digest: OnceCell::new(),
    }
  }

  /// Returns the digest of the verifier's key.
  pub fn digest(&self) -> G::Scalar {
    self
      .digest
      .get_or_try_init(|| {
        let vk = VerifierKey::<G, EE>::new(self.S_single.clone(), self.vk_ee.clone()); 
        let dc = DigestComputer::<G::Scalar, _>::new(&vk);
        dc.digest()
      })
      .cloned()
      .expect("Failure to retrieve digest!")
  }
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSSNARK<G: Group, EE: EvaluationEngineTrait<G>> {
  comm_W: CompressedCommitment<G>,
  sc_proof_outer: SumcheckProof<G>,
  claims_outer: (G::Scalar, G::Scalar, G::Scalar),
  sc_proof_inner: SumcheckProof<G>,
  eval_W: G::Scalar,
  eval_arg: EE::EvaluationArgument,
}

impl<G: Group, EE: EvaluationEngineTrait<G>> RelaxedR1CSSNARKTrait<G> for R1CSSNARK<G, EE> {
  type ProverKey = UniformProverKey<G, EE>;
  type VerifierKey = UniformVerifierKey<G, EE>;

  fn setup<C: Circuit<G::Scalar>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError> {
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (S, ck) = cs.r1cs_shape();

    let (pk_ee, vk_ee) = EE::setup(&ck);

    let vk: UniformVerifierKey<G, EE> = UniformVerifierKey::new(S.clone(), vk_ee, S.clone(), 1);

    let pk = UniformProverKey {
      ck,
      pk_ee,
      S: S.clone(),
      S_single: S,
      num_steps: 0,
      vk_digest: vk.digest(),
    };

    Ok((pk, vk))
  }

  /// produces a succinct proof of satisfiability of a `RelaxedR1CS` instance
  #[tracing::instrument(skip_all, name = "Spartan2::UPSnark::prove")]
  fn prove<C: Circuit<G::Scalar>>(pk: &Self::ProverKey, circuit: C) -> Result<Self, SpartanError> {
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let _ = circuit.synthesize(&mut cs);

    // Commits to witness (expensive)
    let (u, w) = cs
      .r1cs_instance_and_witness(&pk.S, &pk.ck)
      .map_err(|_e| SpartanError::UnSat)?;

    let non_commitment_span = tracing::span!(tracing::Level::INFO, "PostCommitProve");
    let _guard = non_commitment_span.enter();

    let W = w.pad(&pk.S); // pad the witness
    let mut transcript = G::TE::new(b"R1CSSNARK");

    // sanity check that R1CSShape has certain size characteristics
    pk.S.check_regular_shape();

    // append the digest of vk (which includes R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", &u);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let mut z = [w.W.clone(), vec![1.into()], u.X.clone()].concat();

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(pk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(pk.S.num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<Vec<G::Scalar>, SpartanError>>()?;

    let mut poly_tau = MultilinearPolynomial::new(EqPolynomial::new(tau).evals());
    // poly_Az is the polynomial extended from the vector Az 
    let (mut poly_Az, mut poly_Bz, mut poly_Cz) = {
      let (poly_Az, poly_Bz, poly_Cz) = pk.S.multiply_vec(&z)?;
      (
        MultilinearPolynomial::new(poly_Az),
        MultilinearPolynomial::new(poly_Bz),
        MultilinearPolynomial::new(poly_Cz),
      )
    };

    let comb_func_outer =
      |poly_A_comp: &G::Scalar,
       poly_B_comp: &G::Scalar,
       poly_C_comp: &G::Scalar,
       poly_D_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };
    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_additive_term(
      &G::Scalar::ZERO, // claim is zero
      num_rounds_x,
      &mut poly_tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_Cz,
      comb_func_outer,
      &mut transcript,
    )?;

    // claims from the end of sum-check
    // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness 
    let (claim_Az, claim_Bz): (G::Scalar, G::Scalar) = (claims_outer[1], claims_outer[2]);
    let claim_Cz = claims_outer[3];
    transcript.absorb(
      b"claims_outer",
      &[claim_Az, claim_Bz, claim_Cz].as_slice(),
    );

    // inner sum-check
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;

    let span = tracing::span!(tracing::Level::TRACE, "poly_ABC");
    let _enter = span.enter();
    // this is the polynomial extended from the vector r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
    let poly_ABC = {
      let NUM_STEPS_BITS = pk.num_steps.trailing_zeros();
      let (rx_con, rx_ts) = r_x.split_at(r_x.len() - NUM_STEPS_BITS as usize);
      let eq_rx_con = EqPolynomial::new(rx_con.to_vec()).evals();
      let eq_rx_ts = EqPolynomial::new(rx_ts.to_vec()).evals();

      // Bounds "row" variables of full (A, B, C) matrices 
      // viewed as 2d multilinear polynomials
      // built using the single_step (A, B, C) matrices 
      let compute_eval_table_sparse_uniform =
        |S_single: &R1CSShape<G>, N_STEPS: usize, eq_rx_con: &[G::Scalar], eq_rx_ts: &[G::Scalar]| -> (Vec<G::Scalar>, Vec<G::Scalar>, Vec<G::Scalar>) {
          assert_eq!(eq_rx_con.len().ilog2() + eq_rx_ts.len().ilog2(), pk.S.num_cons.ilog2());

          let inner = |small_M: &Vec<(usize, usize, G::Scalar)>, M_evals: &mut Vec<G::Scalar>| {
            // Evaluate \tilde smallM(r_x, y) for all y 
            let mut small_M_evals = vec![G::Scalar::ZERO; pk.S_single.num_vars + 1];
            for (row, col, val) in small_M.iter() {
              small_M_evals[*col] += eq_rx_con[*row] * val;
            }

            // Handles all entries but the last one with the constant 1 variable
            M_evals.par_iter_mut().take(pk.S.num_vars).enumerate().for_each(|(col, m_eval)| {
              *m_eval = eq_rx_ts[col % N_STEPS] * small_M_evals[col / N_STEPS];
            });

            // Handles the constant 1 variable 
            small_M.iter()
              .filter(|(_, col, _)| *col == pk.S_single.num_vars) 
              .for_each(|(row, _, val)| {
                  (0..N_STEPS).for_each(|t| {
                      let col_t = pk.S.num_vars;
                      M_evals[col_t] += eq_rx_con[*row] * eq_rx_ts[t] * val;
                  });
              });
          };

          let (mut A_evals, mut B_evals, mut C_evals) = (
            vec![G::Scalar::ZERO; 2 * S_single.num_vars * N_STEPS],
            vec![G::Scalar::ZERO; 2 * S_single.num_vars * N_STEPS],
            vec![G::Scalar::ZERO; 2 * S_single.num_vars * N_STEPS]
          );
          rayon::join(
            || inner(&S_single.A, &mut A_evals),
            || rayon::join(
              || inner(&S_single.B, &mut B_evals),
              || inner(&S_single.C, &mut C_evals),
            ),
          );

          (A_evals, B_evals, C_evals)
        };

      // evals_A is the vector of evaluations of A(r_x, y) for all y
      // The summation of this should be claims_A right? 
      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse_uniform(&pk.S_single, pk.num_steps, &eq_rx_con, &eq_rx_ts);

      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());

      let span_e = tracing::span!(tracing::Level::TRACE, "eval_combo_old");
      let _enter_e = span_e.enter();
      let r_sq = r * r;
      let thing = (0..evals_A.len())
        .into_par_iter()
        .map(|i| {
          evals_A[i] + evals_B[i] * r + evals_C[i] * r_sq
        })
        .collect::<Vec<G::Scalar>>();
      drop(_enter_e);
      drop(span_e);

      thing
    };
    drop(_enter);
    drop(span);

    let poly_z = {
      z.resize(pk.S.num_vars * 2, G::Scalar::ZERO);
      z
    };

    let comb_func = |poly_A_comp: &G::Scalar, poly_B_comp: &G::Scalar| -> G::Scalar {
      *poly_A_comp * *poly_B_comp
    };
    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad(
      &claim_inner_joint, // r_A * v_A + r_B * v_B + r_C * v_C
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC), // r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
      &mut MultilinearPolynomial::new(poly_z), // z(y) for all y
      comb_func,
      &mut transcript,
    )?;

    let span = tracing::span!(tracing::Level::TRACE, "MultilinearPolynomial::evaluate_with");
    let _enter = span.enter();
    let eval_W = MultilinearPolynomial::evaluate_with(&W.W, &r_y[1..]);
    drop(_enter);
    drop(span);

    let eval_arg = EE::prove(
      &pk.ck,
      &pk.pk_ee,
      &mut transcript,
      &u.comm_W,
      &W.W.clone(),
      &r_y[1..].to_vec(),
      &eval_W,
    )?;

    Ok(R1CSSNARK {
      comm_W: u.comm_W.compress(),
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      sc_proof_inner,
      eval_W,
      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  #[tracing::instrument(skip_all, name = "SNARK::verify")]
  fn verify(&self, vk: &Self::VerifierKey, io: &[G::Scalar]) -> Result<(), SpartanError> {
    // construct an instance using the provided commitment to the witness and IO
    let comm_W = Commitment::<G>::decompress(&self.comm_W)?;
    let u = R1CSInstance::new(&vk.S, &comm_W, io)?;

    let mut transcript = G::TE::new(b"R1CSSNARK");

    // append the digest of R1CS matrices and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &vk.digest());
    transcript.absorb(b"U", &u);

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(vk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(vk.S.num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<Vec<G::Scalar>, SpartanError>>()?;

    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(G::Scalar::ZERO, num_rounds_x, 3, &mut transcript)?;

    // verify claim_outer_final
    let (claim_Az, claim_Bz, claim_Cz) = self.claims_outer;
    let taus_bound_rx = EqPolynomial::new(tau).evaluate(&r_x);
    let claim_outer_final_expected =
      taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
    if claim_outer_final != claim_outer_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }

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
        // constant term
        let mut poly_X = vec![(0, 1.into())];
        //remaining inputs
        poly_X.extend(
          (0..u.X.len())
            .map(|i| (i + 1, u.X[i]))
            .collect::<Vec<(usize, G::Scalar)>>(),
        );
        SparsePolynomial::new(usize::try_from(vk.S.num_vars.ilog2()).unwrap(), poly_X)
          .evaluate(&r_y[1..])
      };
      (G::Scalar::ONE - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    // compute evaluations of R1CS matrices
    let multi_evaluate = |M_vec: &[&[(usize, usize, G::Scalar)]],
                          r_x: &[G::Scalar],
                          r_y: &[G::Scalar]|
     -> Vec<G::Scalar> {
      let evaluate_with_table =
        |M: &[(usize, usize, G::Scalar)], T_x: &[G::Scalar], T_y: &[G::Scalar]| -> G::Scalar {
          (0..M.len())
            .into_par_iter()
            .map(|i| {
              let (row, col, val) = M[i];
              T_x[row] * T_y[col] * val
            })
            .sum()
        };

      let (T_x, T_y) = rayon::join(
        || EqPolynomial::new(r_x.to_vec()).evals(),
        || EqPolynomial::new(r_y.to_vec()).evals(),
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

    // verify
    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &u.comm_W.clone(),
      &r_y[1..].to_vec(),
      &self.eval_W,
      &self.eval_arg,
    )?;

    Ok(())
  }
}

impl<G: Group, EE: EvaluationEngineTrait<G>> UniformSNARKTrait<G> for R1CSSNARK<G, EE> {
  #[tracing::instrument(skip_all, name = "SNARK::setup_uniform")]
  fn setup_uniform<C: Circuit<G::Scalar>>(
    circuit: C,
    num_steps: usize, 
  ) -> Result<(UniformProverKey<G, EE>, UniformVerifierKey<G, EE>), SpartanError> {
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    // let (S, S_single, ck) = cs.r1cs_shape_uniform(num_steps);
    let (S, S_single, ck) = cs.r1cs_shape_uniform(num_steps);

    let (pk_ee, vk_ee) = EE::setup(&ck);

    let vk: UniformVerifierKey<G, EE> = UniformVerifierKey::new(S.clone(), vk_ee, S_single.clone(), num_steps);

    let pk = UniformProverKey {
      ck,
      pk_ee,
      S,
      S_single, 
      num_steps, 
      vk_digest: vk.digest(),
    };

    Ok((pk, vk))
  }
}


impl<G: Group, EE: EvaluationEngineTrait<G>> PrecommittedSNARKTrait<G> for R1CSSNARK<G, EE> {
  #[tracing::instrument(skip_all, name = "SNARK::setup_uniform")]
  fn setup_precommitted<C: Circuit<G::Scalar>>(
    circuit: C,
    num_steps: usize, 
  ) -> Result<(UniformProverKey<G, EE>, UniformVerifierKey<G, EE>), SpartanError> {
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (S, S_single, ck) = cs.r1cs_shape_uniform_variablewise(num_steps); // TODO(arasuarun): replace with precommitted version

    let (pk_ee, vk_ee) = EE::setup(&ck);

    let vk: UniformVerifierKey<G, EE> = UniformVerifierKey::new(S.clone(), vk_ee, S_single.clone(), num_steps);

    let pk = UniformProverKey {
      ck,
      pk_ee,
      S: S.clone(),
      S_single: S_single,
      num_steps, 
      vk_digest: vk.digest(),
    };

    Ok((pk, vk))
  }
}