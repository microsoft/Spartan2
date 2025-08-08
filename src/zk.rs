#![allow(non_snake_case)]
//! Zero-knowledge circuit implementing the algebraic checks of the Spartan verifier logic.
//!
//! [`SpartanVerifierCircuit`] constrains the complete Spartan verification across
//! `outer_len + inner_len + 3` rounds: outer sum-check, inner sum-check, and bridging rounds.
//!
//! Note: This circuit only encodes the algebraic checks of the verifier. It does **not**
//! encode the Fiat-Shamir challenge generation, so no proof composition is performed here.

use crate::traits::{Engine, circuit::MultiRoundCircuit};
use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::Field;
use tracing::debug;

/// Evaluates a polynomial using Horner's method within R1CS constraints.
fn eval_poly_horner<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  cs: &mut CS,
  coeffs: &[AllocatedNum<E::Scalar>],
  x: &AllocatedNum<E::Scalar>,
) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
  // Start from highest coefficient.
  let mut acc = coeffs.last().unwrap().clone(); // degree ≥ 1 in practice
  // We iterate from degree-1 down to 0 (excluding last which we already used).
  for (i, c_i) in coeffs.iter().rev().skip(1).enumerate() {
    // acc = acc * x
    let acc_times_x = AllocatedNum::alloc(cs.namespace(|| format!("horner_mul_{i}")), || {
      let a = acc.get_value().unwrap_or(E::Scalar::ZERO);
      let xv = x.get_value().unwrap_or(E::Scalar::ZERO);
      Ok(a * xv)
    })?;
    cs.enforce(
      || format!("horner_mul_enf_{i}"),
      |lc| lc + acc.get_variable(),
      |lc| lc + x.get_variable(),
      |lc| lc + acc_times_x.get_variable(),
    );

    // acc = acc * x + c_i
    let new_acc = AllocatedNum::alloc(cs.namespace(|| format!("horner_add_{i}")), || {
      let prod = acc_times_x.get_value().unwrap_or(E::Scalar::ZERO);
      let cv = c_i.get_value().unwrap_or(E::Scalar::ZERO);
      Ok(prod + cv)
    })?;
    // Enforce new_acc = acc_times_x + c_i
    cs.enforce(
      || format!("horner_add_enf_{i}"),
      |lc| lc + acc_times_x.get_variable() + c_i.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + new_acc.get_variable(),
    );
    acc = new_acc;
  }
  Ok(acc)
}

/// Circuit constraining Spartan verifier computation across multiple rounds.
#[derive(Clone, Debug, Default)]
pub struct SpartanVerifierCircuit<E: Engine> {
  outer_polys: Vec<[E::Scalar; 4]>,
  claim_Az: E::Scalar,
  claim_Bz: E::Scalar,
  claim_Cz: E::Scalar,
  tau_eval: E::Scalar,
  inner_polys: Vec<[E::Scalar; 3]>,
  eval_A: E::Scalar,
  eval_B: E::Scalar,
  eval_C: E::Scalar,
  eval_W: E::Scalar,
  eval_X: E::Scalar,
}

impl<E: Engine> SpartanVerifierCircuit<E> {
  fn outer_rounds(&self) -> usize {
    self.outer_polys.len()
  }
  fn inner_rounds(&self) -> usize {
    self.inner_polys.len()
  }
  fn idx_outer_final(&self) -> usize {
    self.outer_rounds()
  }
  fn idx_inner_setup(&self) -> usize {
    self.idx_outer_final() + 1
  }
  fn idx_inner_start(&self) -> usize {
    self.idx_inner_setup() + 1
  }
  fn idx_inner_final(&self) -> usize {
    self.idx_inner_start() + self.inner_rounds()
  }
}

impl<E: Engine> MultiRoundCircuit<E> for SpartanVerifierCircuit<E> {
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    // Expose only non-challenge public inputs
    Ok(vec![self.eval_A, self.eval_B, self.eval_C, self.eval_X])
  }

  fn num_challenges(&self, round_index: usize) -> Result<usize, SynthesisError> {
    if round_index < self.outer_rounds() {
      Ok(1)
    } else if round_index == self.idx_outer_final() || round_index == self.idx_inner_final() {
      Ok(0)
    } else if round_index == self.idx_inner_setup()
      || (round_index >= self.idx_inner_start() && round_index < self.idx_inner_final())
    {
      Ok(1)
    } else {
      Err(SynthesisError::Unsatisfiable)
    }
  }

  fn rounds<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    round_index: usize,
    prior_round_vars: &[Vec<AllocatedNum<E::Scalar>>],
    prev_challenges: &[Vec<AllocatedNum<E::Scalar>>],
    challenges: Option<&[E::Scalar]>,
  ) -> Result<(Vec<AllocatedNum<E::Scalar>>, Vec<AllocatedNum<E::Scalar>>), SynthesisError> {
    // Helper macros
    macro_rules! alloc_coeffs {
      ($ns:expr, $coeffs:expr) => {{
        let mut v = Vec::new();
        for (i, c) in $coeffs.iter().enumerate() {
          v.push(AllocatedNum::alloc(
            $ns.namespace(|| format!("coef_{i}")),
            || Ok(*c),
          )?);
        }
        v
      }};
    }

    // Routing
    if round_index < self.outer_rounds() {
      // Outer cubic sum-check per-round consistency
      let i = round_index;

      let mut ns_round = cs.namespace(|| format!("outer_round_{i}"));
      let coeffs_raw = self.outer_polys[i];
      let coeffs = alloc_coeffs!(ns_round, &coeffs_raw);

      let r_val = challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO);
      let r = AllocatedNum::alloc_input(ns_round.namespace(|| "r_x"), || Ok(r_val))?;

      // Sum-check consistency: g(0)+g(1)=prev_e
      let sum_p01 = AllocatedNum::alloc(ns_round.namespace(|| "sum_p01"), || {
        Ok(coeffs_raw[0] + coeffs_raw[0] + coeffs_raw[1] + coeffs_raw[2] + coeffs_raw[3])
      })?;

      ns_round.enforce(
        || "sum_p01_def",
        |lc| lc + sum_p01.get_variable(),
        |lc| lc + CS::one(),
        |lc| {
          lc + coeffs[0].get_variable()
            + coeffs[0].get_variable()
            + coeffs[1].get_variable()
            + coeffs[2].get_variable()
            + coeffs[3].get_variable()
        },
      );

      if i == 0 {
        ns_round.enforce(
          || "outer_prev_claim_zero",
          |lc| lc + sum_p01.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc, // constant 0
        );
      } else {
        let prev_e = &prior_round_vars.last().expect("has previous round")[0];
        ns_round.enforce(
          || format!("outer_consistency_{i}"),
          |lc| lc + sum_p01.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + prev_e.get_variable(),
        );
      }

      // Compute p(r)
      let p_r = {
        let mut ns_eval = ns_round.namespace(|| "poly_eval");
        eval_poly_horner::<E, _>(&mut ns_eval, &coeffs, &r)?
      };

      debug!(outer_round = i, "p_r = {:?}", p_r.get_value());
      Ok((vec![p_r], vec![r]))
    } else if round_index == self.idx_outer_final() {
      // Final outer equality: p_final == tau * (Az*Bz - Cz)
      let prev_e = &prior_round_vars.last().expect("has previous round")[0];

      let claim_Az = AllocatedNum::alloc(cs.namespace(|| "Az_outer"), || Ok(self.claim_Az))?;
      let claim_Bz = AllocatedNum::alloc(cs.namespace(|| "Bz_outer"), || Ok(self.claim_Bz))?;
      let claim_Cz = AllocatedNum::alloc(cs.namespace(|| "Cz_outer"), || Ok(self.claim_Cz))?;
      // Use provided tau evaluation as a private witness
      let tau = AllocatedNum::alloc(cs.namespace(|| "tau_eval"), || Ok(self.tau_eval))?;

      // diff = Az*Bz - Cz
      let prod = AllocatedNum::alloc(cs.namespace(|| "AzBz"), || {
        let a = claim_Az
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let b = claim_Bz
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        Ok(a * b)
      })?;
      cs.enforce(
        || "prod = Az*Bz",
        |lc| lc + claim_Az.get_variable(),
        |lc| lc + claim_Bz.get_variable(),
        |lc| lc + prod.get_variable(),
      );

      let diff = AllocatedNum::alloc(cs.namespace(|| "diff"), || {
        let p = prod.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let c = claim_Cz
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        Ok(p - c)
      })?;
      cs.enforce(
        || "diff = prod - Cz",
        |lc| lc + prod.get_variable() - claim_Cz.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + diff.get_variable(),
      );

      // expected = tau * diff
      let expected = AllocatedNum::alloc(cs.namespace(|| "expected"), || {
        let t = tau.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let d = diff.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(t * d)
      })?;
      cs.enforce(
        || "expected = tau*diff",
        |lc| lc + tau.get_variable(),
        |lc| lc + diff.get_variable(),
        |lc| lc + expected.get_variable(),
      );

      // prev_e == expected
      cs.enforce(
        || "outer_final_eq",
        |lc| lc + prev_e.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + expected.get_variable(),
      );

      Ok((
        vec![
          prev_e.clone(),
          claim_Az.clone(),
          claim_Bz.clone(),
          claim_Cz.clone(),
        ],
        vec![],
      ))
    } else if round_index == self.idx_inner_setup() {
      // Inner setup: introduce global r and compute claim_inner_joint
      let r_val = challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO);
      let r = AllocatedNum::alloc_input(cs.namespace(|| "r"), || Ok(r_val))?;

      let prev_outer_vars = &prior_round_vars[self.idx_outer_final()];
      let claim_Az = prev_outer_vars[1].clone();
      let claim_Bz = prev_outer_vars[2].clone();
      let claim_Cz = prev_outer_vars[3].clone();

      // r^2
      let r_sq = AllocatedNum::alloc(cs.namespace(|| "r_sq"), || {
        let rv = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(rv * rv)
      })?;
      cs.enforce(
        || "r_sq = r*r",
        |lc| lc + r.get_variable(),
        |lc| lc + r.get_variable(),
        |lc| lc + r_sq.get_variable(),
      );

      // tmp1 = Bz * r
      let tmp1 = AllocatedNum::alloc(cs.namespace(|| "tmp1"), || {
        let b = claim_Bz
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let rv = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(b * rv)
      })?;
      cs.enforce(
        || "tmp1 = Bz*r",
        |lc| lc + claim_Bz.get_variable(),
        |lc| lc + r.get_variable(),
        |lc| lc + tmp1.get_variable(),
      );

      // tmp2 = Cz * r^2
      let tmp2 = AllocatedNum::alloc(cs.namespace(|| "tmp2"), || {
        let c = claim_Cz
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let rsq = r_sq.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(c * rsq)
      })?;
      cs.enforce(
        || "tmp2 = Cz*r_sq",
        |lc| lc + claim_Cz.get_variable(),
        |lc| lc + r_sq.get_variable(),
        |lc| lc + tmp2.get_variable(),
      );

      // claim_inner_joint = Az + tmp1 + tmp2
      let inner_joint = AllocatedNum::alloc(cs.namespace(|| "inner_joint"), || {
        let a = claim_Az
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let t1 = tmp1.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let t2 = tmp2.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(a + t1 + t2)
      })?;
      cs.enforce(
        || "inner_joint_eq",
        |lc| lc + claim_Az.get_variable() + tmp1.get_variable() + tmp2.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + inner_joint.get_variable(),
      );

      Ok((vec![inner_joint], vec![r]))
    } else if round_index >= self.idx_inner_start() && round_index < self.idx_inner_final() {
      // Inner quadratic sum-check per-round consistency
      let idx = round_index - self.idx_inner_start();

      let mut ns_round = cs.namespace(|| format!("inner_round_{idx}"));

      let coeffs_raw = self.inner_polys[idx];
      let coeffs = alloc_coeffs!(ns_round, &coeffs_raw);

      let r_y_val = challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO);
      let r_y = AllocatedNum::alloc_input(ns_round.namespace(|| "r_y"), || Ok(r_y_val))?;

      // Sum-check consistency: h(0)+h(1)=prev_e
      let sum_p01 = AllocatedNum::alloc(ns_round.namespace(|| "sum_p01"), || {
        Ok(coeffs_raw[0] + coeffs_raw[0] + coeffs_raw[1] + coeffs_raw[2])
      })?;
      ns_round.enforce(
        || "sum_p01_def",
        |lc| lc + sum_p01.get_variable(),
        |lc| lc + CS::one(),
        |lc| {
          lc + coeffs[0].get_variable()
            + coeffs[0].get_variable()
            + coeffs[1].get_variable()
            + coeffs[2].get_variable()
        },
      );

      let prev_e = &prior_round_vars.last().expect("prev inner round")[0];
      ns_round.enforce(
        || format!("inner_consistency_{idx}"),
        |lc| lc + sum_p01.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + prev_e.get_variable(),
      );

      let p_ry = {
        let mut ns_eval = ns_round.namespace(|| "poly_eval");
        eval_poly_horner::<E, _>(&mut ns_eval, &coeffs, &r_y)?
      };
      debug!(inner_round = idx, "p_ry = {:?}", p_ry.get_value());

      Ok((vec![p_ry], vec![r_y]))
    } else if round_index == self.idx_inner_final() {
      // Final inner equality
      let prev_e = &prior_round_vars.last().expect("prev inner round")[0];

      let r = prev_challenges[self.idx_inner_setup()][0].clone();

      // Public inputs for eval_A, eval_B, eval_C, and eval_X are inputized here when known
      let eval_A = AllocatedNum::alloc(cs.namespace(|| "eval_A"), || Ok(self.eval_A))?;
      eval_A.inputize(cs.namespace(|| "eval_A_input"))?;
      let eval_B = AllocatedNum::alloc(cs.namespace(|| "eval_B"), || Ok(self.eval_B))?;
      eval_B.inputize(cs.namespace(|| "eval_B_input"))?;
      let eval_C = AllocatedNum::alloc(cs.namespace(|| "eval_C"), || Ok(self.eval_C))?;
      eval_C.inputize(cs.namespace(|| "eval_C_input"))?;
      let eval_W = AllocatedNum::alloc(cs.namespace(|| "eval_W"), || Ok(self.eval_W))?;
      let eval_X = AllocatedNum::alloc(cs.namespace(|| "eval_X"), || Ok(self.eval_X))?;
      eval_X.inputize(cs.namespace(|| "eval_X_input"))?;

      let r_y0 = prev_challenges[self.idx_inner_start()][0].clone();
      let one_minus_ry0 = AllocatedNum::alloc(cs.namespace(|| "one_minus_ry0"), || {
        let rv = r_y0.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(E::Scalar::ONE - rv)
      })?;
      cs.enforce(
        || "one_minus_ry0_def",
        |lc| lc + one_minus_ry0.get_variable() + r_y0.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
      );

      // tmp_w = eval_W * one_minus_ry0
      let tmp_w = AllocatedNum::alloc(cs.namespace(|| "tmp_w"), || {
        let ew = eval_W.get_value().unwrap_or(E::Scalar::ZERO);
        let om = one_minus_ry0.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(ew * om)
      })?;
      cs.enforce(
        || "tmp_w_def",
        |lc| lc + eval_W.get_variable(),
        |lc| lc + one_minus_ry0.get_variable(),
        |lc| lc + tmp_w.get_variable(),
      );

      // tmp_x = eval_X * r_y0
      let tmp_x = AllocatedNum::alloc(cs.namespace(|| "tmp_x"), || {
        let ex = eval_X.get_value().unwrap_or(E::Scalar::ZERO);
        let ry = r_y0.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(ex * ry)
      })?;
      cs.enforce(
        || "tmp_x_def",
        |lc| lc + eval_X.get_variable(),
        |lc| lc + r_y0.get_variable(),
        |lc| lc + tmp_x.get_variable(),
      );

      // sum_z_expected = tmp_w + tmp_x
      let sum_z_expected = AllocatedNum::alloc(cs.namespace(|| "sum_z_expected"), || {
        let tw = tmp_w.get_value().unwrap_or(E::Scalar::ZERO);
        let tx = tmp_x.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(tw + tx)
      })?;
      cs.enforce(
        || "sum_z_expected_def",
        |lc| lc + tmp_w.get_variable() + tmp_x.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + sum_z_expected.get_variable(),
      );

      // r^2
      let r_sq = AllocatedNum::alloc(cs.namespace(|| "r_sq_final"), || {
        let rv = r.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(rv * rv)
      })?;
      cs.enforce(
        || "r_sq_final = r*r",
        |lc| lc + r.get_variable(),
        |lc| lc + r.get_variable(),
        |lc| lc + r_sq.get_variable(),
      );

      // tmp1 = eval_B * r
      let tmp1 = AllocatedNum::alloc(cs.namespace(|| "tmp1_final"), || {
        let eb = eval_B
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let rv = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(eb * rv)
      })?;
      cs.enforce(
        || "tmp1_final = B*r",
        |lc| lc + eval_B.get_variable(),
        |lc| lc + r.get_variable(),
        |lc| lc + tmp1.get_variable(),
      );

      // tmp2 = eval_C * r_sq
      let tmp2 = AllocatedNum::alloc(cs.namespace(|| "tmp2_final"), || {
        let ec = eval_C
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let rsq = r_sq.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(ec * rsq)
      })?;
      cs.enforce(
        || "tmp2_final = C*r_sq",
        |lc| lc + eval_C.get_variable(),
        |lc| lc + r_sq.get_variable(),
        |lc| lc + tmp2.get_variable(),
      );

      // sum = A + tmp1 + tmp2
      let sum = AllocatedNum::alloc(cs.namespace(|| "sum_final"), || {
        let a = eval_A
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let t1 = tmp1.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let t2 = tmp2.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(a + t1 + t2)
      })?;
      cs.enforce(
        || "sum_final_def",
        |lc| lc + eval_A.get_variable() + tmp1.get_variable() + tmp2.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + sum.get_variable(),
      );

      // expected = sum * sum_z_expected
      let expected = AllocatedNum::alloc(cs.namespace(|| "expected_final"), || {
        let s = sum.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let z = sum_z_expected
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        Ok(s * z)
      })?;
      cs.enforce(
        || "expected_final = sum*sum_z_expected",
        |lc| lc + sum.get_variable(),
        |lc| lc + sum_z_expected.get_variable(),
        |lc| lc + expected.get_variable(),
      );

      // prev_e == expected
      cs.enforce(
        || "inner_final_eq",
        |lc| lc + prev_e.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + expected.get_variable(),
      );

      Ok((vec![prev_e.clone()], vec![]))
    } else {
      Err(SynthesisError::Unsatisfiable)
    }
  }

  fn num_rounds(&self) -> usize {
    self.idx_inner_final() + 1
  }
}

// ------------------------------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
  use super::*;
  use crate::bellpepper::{
    r1cs::{MultiRoundSpartanShape, MultiRoundSpartanWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  };
  use crate::nifs::NIFS;
  use crate::spartan::test_utils::InteractiveSession;
  use crate::traits::{Engine, transcript::TranscriptEngineTrait};
  use tracing_subscriber::EnvFilter;

  type E = crate::provider::P256HyraxEngine;

  #[test]
  fn test_verifier_circuit_on_repeated_squaring() {
    type E = crate::provider::P256HyraxEngine;
    // Keep this modest to keep test fast
    const NUM_SQUARINGS: usize = 8;
    let init_value = <E as Engine>::Scalar::from(3u64);
    let mr_circuit = RepeatedSquaringCircuit::<E> {
      num_squarings: NUM_SQUARINGS,
      init_val: init_value,
    };

    // Initialize session (prover emulator)
    let mut session =
      InteractiveSession::<E>::begin(mr_circuit.clone()).expect("initialize interactive session");

    // Pre-size circuit with zeroed placeholders
    let outer_len = session.num_rounds_x;
    let inner_len = session.num_rounds_y;
    let zero = <E as Engine>::Scalar::ZERO;
    let mut circuit = SpartanVerifierCircuit::<E> {
      outer_polys: vec![[zero; 4]; outer_len],
      claim_Az: zero,
      claim_Bz: zero,
      claim_Cz: zero,
      tau_eval: zero,
      inner_polys: vec![[zero; 3]; inner_len],
      eval_A: zero,
      eval_B: zero,
      eval_C: zero,
      eval_W: zero,
      eval_X: zero,
    };

    // Build shape from sized circuit
    let (shape_mr, ck, _vk) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&circuit).unwrap();
    let mut state =
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::initialize_multiround_witness(
        &shape_mr, &ck, &circuit, false,
      )
      .unwrap();
    let mut transcript = <E as Engine>::TE::new(b"nifs");

    // Interleave outer rounds
    for i in 0..outer_len {
      let coeffs = session.outer_produce_poly().expect("outer produce");
      circuit.outer_polys[i] = coeffs;
      let chals = SatisfyingAssignment::<E>::process_round_returning_challenges(
        &mut state,
        &shape_mr,
        &ck,
        &circuit,
        i,
        false,
        &mut transcript,
      )
      .expect("process outer round");
      let r_i = chals[0];
      session.outer_bind(r_i);
    }

    // Outer final round: set claims and tau_eval
    let (claim_Az, claim_Bz, claim_Cz) = session.outer_finalize();
    circuit.claim_Az = claim_Az;
    circuit.claim_Bz = claim_Bz;
    circuit.claim_Cz = claim_Cz;
    circuit.tau_eval = session.outer_tau_eval();

    SatisfyingAssignment::<E>::process_round_returning_challenges(
      &mut state,
      &shape_mr,
      &ck,
      &circuit,
      circuit.idx_outer_final(),
      false,
      &mut transcript,
    )
    .expect("process outer final");

    // Inner setup: driver generates r, we pass to session
    let inner_setup_round = circuit.idx_inner_setup();
    let chals = SatisfyingAssignment::<E>::process_round_returning_challenges(
      &mut state,
      &shape_mr,
      &ck,
      &circuit,
      inner_setup_round,
      false,
      &mut transcript,
    )
    .expect("process inner setup");
    let r_inner = chals[0];
    let _claim_inner_joint = session
      .inner_setup_given_r(r_inner)
      .expect("inner setup given r");

    // Inner rounds
    for j in 0..inner_len {
      let coeffs = session.inner_produce_poly().expect("inner produce");
      circuit.inner_polys[j] = coeffs;
      let chals = SatisfyingAssignment::<E>::process_round_returning_challenges(
        &mut state,
        &shape_mr,
        &ck,
        &circuit,
        circuit.idx_inner_start() + j,
        false,
        &mut transcript,
      )
      .expect("process inner round");
      let r_y_j = chals[0];
      session.inner_bind(r_y_j);
    }

    // Final evaluations for inner final round
    let (eval_W, eval_X, (eval_A, eval_B, eval_C)) = session.inner_finalize().expect("inner fin");
    circuit.eval_W = eval_W;
    circuit.eval_X = eval_X;
    circuit.eval_A = eval_A;
    circuit.eval_B = eval_B;
    circuit.eval_C = eval_C;

    SatisfyingAssignment::<E>::process_round_returning_challenges(
      &mut state,
      &shape_mr,
      &ck,
      &circuit,
      circuit.idx_inner_final(),
      false,
      &mut transcript,
    )
    .expect("process inner final");

    // Finalize and assert satisfiable
    let (inst_split, wit_reg) =
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::finalize_multiround_witness(
        &mut state, &shape_mr, &ck, &circuit, false,
      )
      .unwrap();
    let inst_reg = inst_split.to_regular_instance().unwrap();
    let S_reg = shape_mr.to_regular_shape();
    assert!(S_reg.is_sat(&ck, &inst_reg, &wit_reg).is_ok());
  }

  #[test]
  fn test_full_verifier_nifs_fold() {
    // Initialize logger for easier debugging (optional)
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();

    // Build a minimal circuit with 1 outer round and 1 inner round (all zeros)
    let circuit = SpartanVerifierCircuit::<E> {
      outer_polys: vec![[<E as Engine>::Scalar::ZERO; 4]],
      claim_Az: <E as Engine>::Scalar::ZERO,
      claim_Bz: <E as Engine>::Scalar::ZERO,
      claim_Cz: <E as Engine>::Scalar::ZERO,
      tau_eval: <E as Engine>::Scalar::ZERO,
      inner_polys: vec![[<E as Engine>::Scalar::ZERO; 3]],
      eval_A: <E as Engine>::Scalar::ZERO,
      eval_B: <E as Engine>::Scalar::ZERO,
      eval_C: <E as Engine>::Scalar::ZERO,
      eval_W: <E as Engine>::Scalar::ZERO,
      eval_X: <E as Engine>::Scalar::ZERO,
    };

    // Generate multi-round shape
    let (shape_mr, ck, _vk) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&circuit).unwrap();
    // Check number of R1CS constraints (unpadded)
    assert_eq!(shape_mr.num_cons_unpadded, 36);

    // Witness generation across rounds
    let mut state =
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::initialize_multiround_witness(
        &shape_mr, &ck, &circuit, false,
      )
      .unwrap();
    let mut transcript = <E as Engine>::TE::new(b"nifs");

    let num_rounds = circuit.num_rounds();
    for r in 0..num_rounds {
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::process_round(
        &mut state,
        &shape_mr,
        &ck,
        &circuit,
        r,
        false,
        &mut transcript,
      )
      .unwrap();
    }

    let (inst_split, wit_reg) =
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::finalize_multiround_witness(
        &mut state, &shape_mr, &ck, &circuit, false,
      )
      .unwrap();

    let inst_reg = inst_split.to_regular_instance().unwrap();
    let S_reg = shape_mr.to_regular_shape();

    // Random relaxed instance / witness to fold with
    let (running_U, running_W) = S_reg.sample_random_instance_witness(&ck).unwrap();

    // NIFS prove & verify
    let (proof, (folded_U, folded_W)) =
      NIFS::<E>::prove(&ck, &S_reg, &running_U, &running_W, &inst_reg, &wit_reg).unwrap();

    let verified_U = proof.verify(&running_U, &inst_split, &shape_mr).unwrap();
    assert_eq!(verified_U, folded_U);
    assert!(S_reg.is_sat_relaxed(&ck, &folded_U, &folded_W).is_ok());
  }

  #[derive(Clone)]
  struct RepeatedSquaringCircuit<E: Engine> {
    num_squarings: usize,
    init_val: E::Scalar,
  }

  impl<E: Engine> MultiRoundCircuit<E> for RepeatedSquaringCircuit<E> {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
      // Expose initial and final values as public inputs
      let mut v = self.init_val;
      for _ in 0..self.num_squarings {
        v = v * v;
      }
      Ok(vec![self.init_val, v])
    }

    fn num_challenges(&self, _round_index: usize) -> Result<usize, SynthesisError> {
      Ok(0)
    }

    fn rounds<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      round_index: usize,
      prior_round_vars: &[Vec<AllocatedNum<E::Scalar>>],
      _prev_challenges: &[Vec<AllocatedNum<E::Scalar>>],
      _challenges: Option<&[E::Scalar]>,
    ) -> Result<(Vec<AllocatedNum<E::Scalar>>, Vec<AllocatedNum<E::Scalar>>), SynthesisError> {
      if round_index == 0 {
        let x0 = AllocatedNum::alloc(cs.namespace(|| "x0"), || Ok(self.init_val))?;
        x0.inputize(cs.namespace(|| "x0_input"))?;
        Ok((vec![x0], vec![]))
      } else if round_index <= self.num_squarings {
        let prev_x = &prior_round_vars.last().unwrap()[0];
        let y = AllocatedNum::alloc(cs.namespace(|| format!("x_{round_index}")), || {
          let v = prev_x
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          Ok(v * v)
        })?;
        cs.enforce(
          || format!("square_eq_{round_index}"),
          |lc| lc + prev_x.get_variable(),
          |lc| lc + prev_x.get_variable(),
          |lc| lc + y.get_variable(),
        );
        if round_index == self.num_squarings {
          y.inputize(cs.namespace(|| "y_final_input"))?;
        }
        Ok((vec![y], vec![]))
      } else {
        Err(SynthesisError::Unsatisfiable)
      }
    }

    fn num_rounds(&self) -> usize {
      self.num_squarings + 1
    }
  }
}
