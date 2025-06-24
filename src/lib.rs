//! This library implements Spartan, a high-speed SNARK.
//! We currently implement a non-preprocessing version of Spartan
//! that is generic over the polynomial commitment and evaluation argument (i.e., a PCS).
#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::type_complexity)]
#![forbid(unsafe_code)]

use bellpepper_core::{Circuit, ConstraintSystem};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{info, info_span};

// private modules
mod math;
mod r1cs;

#[macro_use]
mod macros;

// public modules
pub mod bellpepper;
pub mod digest;
pub mod errors;
pub mod polys;
pub mod provider;
pub mod sumcheck;
pub mod traits;

use bellpepper::{
  r1cs::{SpartanShape, SpartanWitness},
  shape_cs::ShapeCS,
  solver::SatisfyingAssignment,
};
use digest::{DigestComputer, SimpleDigestible};
use errors::SpartanError;
use math::Math;
use polys::{
  eq::EqPolynomial,
  multilinear::{MultilinearPolynomial, SparsePolynomial},
};
use r1cs::{R1CSInstance, R1CSShape, R1CSWitness, SparseMatrix};
use sumcheck::SumcheckProof;
use traits::{
  Engine,
  pcs::PCSEngineTrait,
  snark::{DigestHelperTrait, R1CSSNARKTrait, SpartanDigest},
  transcript::TranscriptEngineTrait,
};

/// Start a span + timer, return `(Span, Instant)`.
macro_rules! start_span {
    ($name:expr $(, $($fmt:tt)+)?) => {{
        let span       = info_span!($name $(, $($fmt)+)?);
        let span_clone = span.clone();    // lives as long as the guard
        let _guard      = span_clone.enter();
        (span, Instant::now())
    }};
}
pub(crate) use start_span;

type CommitmentKey<E> = <<E as traits::Engine>::PCS as PCSEngineTrait<E>>::CommitmentKey;
type VerifierKey<E> = <<E as traits::Engine>::PCS as PCSEngineTrait<E>>::VerifierKey;
type Commitment<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::Commitment;
type PCS<E> = <E as Engine>::PCS;
type DerandKey<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::DerandKey;
type Blind<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::Blind;

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanProverKey<E: Engine> {
  ck: CommitmentKey<E>,
  S: R1CSShape<E>,
  vk_digest: SpartanDigest, // digest of the verifier's key
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanVerifierKey<E: Engine> {
  vk_ee: <E::PCS as PCSEngineTrait<E>>::VerifierKey,
  S: R1CSShape<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<SpartanDigest>,
}

impl<E: Engine> SimpleDigestible for SpartanVerifierKey<E> {}

impl<E: Engine> SpartanVerifierKey<E> {
  fn new(shape: R1CSShape<E>, vk_ee: <E::PCS as PCSEngineTrait<E>>::VerifierKey) -> Self {
    SpartanVerifierKey {
      vk_ee,
      S: shape,
      digest: OnceCell::new(),
    }
  }
}

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

/// Bounds "row" variables of (A, B, C) matrices viewed as 2d multilinear polynomials
fn compute_eval_table_sparse<E: Engine>(
  S: &R1CSShape<E>,
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

  let (A_evals, (B_evals, C_evals)) = rayon::join(
    || {
      let mut A_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars];
      inner(&S.A, &mut A_evals);
      A_evals
    },
    || {
      rayon::join(
        || {
          let mut B_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars];
          inner(&S.B, &mut B_evals);
          B_evals
        },
        || {
          let mut C_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars];
          inner(&S.C, &mut C_evals);
          C_evals
        },
      )
    },
  );

  (A_evals, B_evals, C_evals)
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSSNARK<E: Engine> {
  U: R1CSInstance<E>,
  sc_proof_outer: SumcheckProof<E>,
  claims_outer: (E::Scalar, E::Scalar, E::Scalar),
  sc_proof_inner: SumcheckProof<E>,
  eval_W: E::Scalar,
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
}

impl<E: Engine> R1CSSNARKTrait<E> for R1CSSNARK<E> {
  type ProverKey = SpartanProverKey<E>;
  type VerifierKey = SpartanVerifierKey<E>;

  fn setup<C: Circuit<E::Scalar>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError> {
    let mut cs: ShapeCS<E> = ShapeCS::new();
    circuit
      .synthesize(&mut cs)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize circuit: {e}"),
      })?;

    // Padding the ShapeCS: constraints (rows) and variables (columns)
    let num_constraints = cs.num_constraints();

    (num_constraints..num_constraints.next_power_of_two()).for_each(|i| {
      cs.enforce(
        || format!("padding_constraint_{i}"),
        |lc| lc,
        |lc| lc,
        |lc| lc,
      )
    });

    let num_vars = cs.num_aux();
    let num_io = cs.num_inputs();

    let num_vars_padded = num_vars.next_power_of_two();
    (num_vars..num_vars_padded).for_each(|i| {
      cs.alloc(|| format!("padding_var_{i}"), || Ok(E::Scalar::ZERO))
        .unwrap();
    });

    // ensure num_io < num_vars
    if num_io >= num_vars_padded {
      (num_vars_padded..num_io).for_each(|i| {
        cs.alloc(|| format!("padding_var_for_io_{i}"), || Ok(E::Scalar::ZERO))
          .unwrap();
      });
    }

    let (S, ck, vk) = cs.r1cs_shape();
    let vk: SpartanVerifierKey<E> = SpartanVerifierKey::new(S.clone(), vk);
    let pk = Self::ProverKey {
      ck,
      S,
      vk_digest: vk.digest()?,
    };

    Ok((pk, vk))
  }

  fn gen_witness<C: Circuit<<E as Engine>::Scalar>>(
    pk: &Self::ProverKey,
    circuit: C,
    is_small: bool,
  ) -> Result<(R1CSInstance<E>, r1cs::R1CSWitness<E>), SpartanError> {
    let mut cs: SatisfyingAssignment<E> = SatisfyingAssignment::new();
    circuit
      .synthesize(&mut cs)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize witness: {e}"),
      })?;

    let (U, W) = cs
      .r1cs_instance_and_witness(&pk.S, &pk.ck, is_small)
      .map_err(|_e| SpartanError::UnSat {
        reason: "Unable to synthesize witness".to_string(),
      })?;

    // derandomize instance
    let (W, r_W) = W.derandomize();
    let U = U.derandomize(&E::PCS::derand_key(&pk.ck), &r_W);

    Ok((U, W))
  }

  /// produces a succinct proof of satisfiability of an R1CS instance
  fn prove(
    pk: &Self::ProverKey,
    U: &R1CSInstance<E>,
    W: &R1CSWitness<E>,
  ) -> Result<Self, SpartanError> {
    let mut transcript = E::TE::new(b"R1CSSNARK");

    // append the digest of vk (which includes R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U);

    // compute the full satisfying assignment by concatenating W.W, 1, and U.X
    let mut z = [W.W.clone(), vec![E::Scalar::ONE], U.X.clone()].concat();

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(pk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(pk.S.num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check
    let (_sc_span, sc_t) = start_span!("outer_sumcheck");

    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;

    let mut poly_tau = MultilinearPolynomial::new(tau.evals());
    let (mut poly_Az, mut poly_Bz, mut poly_Cz) = {
      let (Az, Bz, Cz) = pk.S.multiply_vec(&z)?;
      (
        MultilinearPolynomial::new(Az),
        MultilinearPolynomial::new(Bz),
        MultilinearPolynomial::new(Cz),
      )
    };

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

    // inner sum-check
    let (_sc2_span, sc2_t) = start_span!("inner_sumcheck");
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;

    let poly_ABC = {
      // compute the initial evaluation table for R(\tau, x)
      let evals_rx = EqPolynomial::evals_from_points(&r_x.clone());

      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&pk.S, &evals_rx);

      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());
      (0..evals_A.len())
        .into_par_iter()
        .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
        .collect::<Vec<E::Scalar>>()
    };

    let poly_z = {
      z.resize(pk.S.num_vars * 2, E::Scalar::ZERO);
      z
    };

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

    let (_we_span, we_t) = start_span!("witness_polyeval");
    let eval_W = MultilinearPolynomial::evaluate_with(&W.W, &r_y[1..]);
    info!(elapsed_ms = %we_t.elapsed().as_millis(), "witness_polyeval");

    let (_pcs_span, pcs_t) = start_span!("pcs_prove");
    let eval_arg = E::PCS::prove(&pk.ck, &mut transcript, &U.comm_W, &W.W, &r_y[1..], &eval_W)?;
    info!(elapsed_ms = %pcs_t.elapsed().as_millis(), "pcs_prove");

    Ok(R1CSSNARK {
      U: U.clone(),
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      sc_proof_inner,
      eval_W,
      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey) -> Result<Vec<E::Scalar>, SpartanError> {
    let mut transcript = E::TE::new(b"R1CSSNARK");

    // append the digest of R1CS matrices and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &vk.digest()?);
    transcript.absorb(b"U", &self.U);

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(vk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(vk.S.num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;

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
        // public IO is (1, X)
        let X = vec![E::Scalar::ONE]
          .into_iter()
          .chain(self.U.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        SparsePolynomial::new(vk.S.num_vars.log_2(), X).evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    // compute evaluations of R1CS matrices
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
                .map(|(val, col_idx)| T_x[row_idx] * T_y[*col_idx] * val)
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

    // verify
    E::PCS::verify(
      &vk.vk_ee,
      &mut transcript,
      &self.U.comm_W,
      &r_y[1..],
      &self.eval_W,
      &self.eval_arg,
    )?;

    Ok(self.U.X.clone())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
  use ff::PrimeField;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit {}

  impl<F: PrimeField> Circuit<F> for CubicCircuit {
    fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(F::ONE + F::ONE))?;
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
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
    type E = crate::provider::PallasIPAEngine;
    type S = R1CSSNARK<E>;
    test_snark_with::<E, S>();

    type E2 = crate::provider::T256IPAEngine;
    type S2 = crate::R1CSSNARK<E2>;
    test_snark_with::<E2, S2>();

    type E3 = crate::provider::PallasHyraxEngine;
    type S3 = R1CSSNARK<E3>;
    test_snark_with::<E3, S3>();

    type E4 = crate::provider::T256HyraxEngine;
    type S4 = crate::R1CSSNARK<E4>;
    test_snark_with::<E4, S4>();
  }

  fn test_snark_with<E: Engine, S: R1CSSNARKTrait<E>>() {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) = S::setup(circuit.clone()).unwrap();

    // generate a witness
    let (U, W) = S::gen_witness(&pk, circuit.clone(), false).unwrap();

    // produce a SNARK
    let res = S::prove(&pk, &U, &W);
    assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk);
    assert!(res.is_ok());
    assert_eq!(res.unwrap(), [<E as Engine>::Scalar::from(15u64)])
  }
}
