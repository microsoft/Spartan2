#![allow(non_snake_case)]
//! Zero-knowledge circuit implementing the algebraic checks of the Spartan verifier logic.
//!
//! [`SpartanVerifierCircuit`] constrains the complete Spartan verification across
//! `outer_len + inner_len + 3` rounds: outer sum-check, inner sum-check, and bridging rounds.
//!
//! Note: This circuit only encodes the algebraic checks of the verifier. It does **not**
//! encode the Fiat-Shamir challenge generation, so no proof composition is performed here.

use crate::{
  MULTIROUND_COMMITMENT_WIDTH,
  traits::{Engine, circuit::MultiRoundCircuit},
};
use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::{Field, PrimeField};

/// Evaluates a polynomial using Horner's method within R1CS constraints.
fn eval_poly_horner<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  mut cs: CS,
  coeffs: &[AllocatedNum<E::Scalar>],
  x: &AllocatedNum<E::Scalar>,
) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
  // Start from highest coefficient.
  let mut acc = coeffs.last().unwrap().clone(); // degree ≥ 1 in practice
  // We iterate from degree-1 down to 0 (excluding last which we already used).
  for (i, c_i) in coeffs.iter().rev().skip(1).enumerate() {
    // new_acc = acc * x + c_i
    let new_acc = AllocatedNum::alloc(cs.namespace(|| format!("new_acc_{i}")), || {
      let a = acc.get_value().unwrap_or(E::Scalar::ZERO);
      let xv = x.get_value().unwrap_or(E::Scalar::ZERO);
      let c = c_i.get_value().unwrap_or(E::Scalar::ZERO);
      Ok(a * xv + c)
    })?;
    cs.enforce(
      || format!("enforce_new_acc_{i}"),
      |lc| lc + acc.get_variable(),
      |lc| lc + x.get_variable(),
      |lc| lc + new_acc.get_variable() - c_i.get_variable(),
    );

    acc = new_acc;
  }
  Ok(acc)
}

/// Allocates a new variable fixed to zero and enforces the constraint `z = 0`.
fn alloc_zero<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  mut cs: CS,
) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
  // Allocate with value 0
  let z = AllocatedNum::alloc(cs.namespace(|| "zero"), || Ok(E::Scalar::ZERO))?;
  // Enforce z * 1 = 0
  cs.enforce(
    || "z_is_zero",
    |lc| lc + z.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc, // constant 0
  );
  Ok(z)
}

/// Circuit constraining Spartan verifier computation across multiple rounds.
#[derive(Clone, Debug, Default)]
pub struct SpartanVerifierCircuit<E: Engine> {
  pub(crate) outer_polys: Vec<[E::Scalar; 4]>,
  pub(crate) claim_Az: E::Scalar,
  pub(crate) claim_Bz: E::Scalar,
  pub(crate) claim_Cz: E::Scalar,
  pub(crate) tau_at_rx: E::Scalar,
  pub(crate) inner_polys: Vec<[E::Scalar; 3]>,
  pub(crate) eval_A: E::Scalar,
  pub(crate) eval_B: E::Scalar,
  pub(crate) eval_C: E::Scalar,
  pub(crate) eval_W: E::Scalar,
  pub(crate) eval_X: E::Scalar,
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
  fn idx_inner_commit_w(&self) -> usize {
    self.idx_inner_start() + self.inner_rounds()
  }
  fn idx_inner_final(&self) -> usize {
    self.idx_inner_commit_w() + 1
  }
}

impl<E: Engine> MultiRoundCircuit<E> for SpartanVerifierCircuit<E> {
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    // Public non-challenge inputs: [eval_A, eval_B, eval_C, tau_at_rx, eval_X]
    Ok(vec![
      self.eval_A,
      self.eval_B,
      self.eval_C,
      self.tau_at_rx,
      self.eval_X,
    ])
  }

  fn num_challenges(&self, round_index: usize) -> Result<usize, SynthesisError> {
    if round_index < self.outer_rounds() {
      Ok(1)
    } else if round_index == self.idx_outer_final() {
      Ok(0)
    } else if round_index == self.idx_inner_setup()
      || (round_index >= self.idx_inner_start() && round_index < self.idx_inner_commit_w())
    {
      Ok(1)
    } else if round_index == self.idx_inner_commit_w() || round_index == self.idx_inner_final() {
      Ok(0)
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
    // Helper method
    fn alloc_coeffs<E: Engine, CS: ConstraintSystem<E::Scalar>>(
      mut cs: CS,
      coeffs: &[E::Scalar],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      coeffs
        .iter()
        .enumerate()
        .map(|(i, c)| AllocatedNum::alloc(cs.namespace(|| format!("coef_{i}")), || Ok(*c)))
        .collect::<Result<Vec<_>, _>>()
    }

    // Routing
    if round_index < self.outer_rounds() {
      // Outer cubic sum-check per-round consistency
      let coeffs = alloc_coeffs::<E, _>(
        cs.namespace(|| format!("outer_sc_coeffs_round_{}", round_index)),
        &self.outer_polys[round_index],
      )?;
      let r = AllocatedNum::alloc_input(cs.namespace(|| format!("r_x_{round_index}")), || {
        Ok(challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO))
      })?;

      // Sum-check consistency: g(0)+g(1)=prev_e; prev_e is 0 for first round
      cs.enforce(
        || format!("outer_prev_claim_{round_index}"),
        |lc| {
          lc + coeffs[0].get_variable()
            + coeffs[0].get_variable()
            + coeffs[1].get_variable()
            + coeffs[2].get_variable()
            + coeffs[3].get_variable()
        },
        |lc| lc + CS::one(),
        |lc| {
          if round_index == 0 {
            lc
          } else {
            let prev_e = &prior_round_vars.last().expect("has previous round")[0];
            lc + prev_e.get_variable()
          }
        },
      );

      // Compute p(r)
      let p_r = eval_poly_horner::<E, _>(
        &mut cs.namespace(|| format!("poly_eval_{round_index}")),
        &coeffs,
        &r,
      )?;

      Ok((vec![p_r], vec![r]))
    } else if round_index == self.idx_outer_final() {
      // Final outer equality: p_final == tau * (Az*Bz - Cz)
      let prev_e = &prior_round_vars.last().expect("has previous round")[0];

      let claim_Az = AllocatedNum::alloc(cs.namespace(|| "Az_outer"), || Ok(self.claim_Az))?;
      let claim_Bz = AllocatedNum::alloc(cs.namespace(|| "Bz_outer"), || Ok(self.claim_Bz))?;
      let claim_Cz = AllocatedNum::alloc(cs.namespace(|| "Cz_outer"), || Ok(self.claim_Cz))?;
      let tau_at_rx =
        AllocatedNum::alloc(cs.namespace(|| "tau_at_rx_outer"), || Ok(self.tau_at_rx))?;

      // prod_Az_Bz = Az*Bz
      let prod_Az_Bz = AllocatedNum::alloc(cs.namespace(|| "AzBz"), || {
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
        |lc| lc + prod_Az_Bz.get_variable(),
      );

      // prev_e = tau_at_rx * (prod_AzBz - Cz)
      cs.enforce(
        || "prev_e = tau_at_rx*(prod_AzBz - Cz)",
        |lc| lc + tau_at_rx.get_variable(),
        |lc| lc + prod_Az_Bz.get_variable() - claim_Cz.get_variable(),
        |lc| lc + prev_e.get_variable(),
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
      let claim_Az = &prev_outer_vars[1];
      let claim_Bz = &prev_outer_vars[2];
      let claim_Cz = &prev_outer_vars[3];

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

      // r_times_Bz = Bz * r
      let r_times_Bz = AllocatedNum::alloc(cs.namespace(|| "r_times_Bz"), || {
        let b = claim_Bz
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let rv = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(b * rv)
      })?;
      cs.enforce(
        || "r_times_Bz = Bz*r",
        |lc| lc + claim_Bz.get_variable(),
        |lc| lc + r.get_variable(),
        |lc| lc + r_times_Bz.get_variable(),
      );

      // claim_inner_joint = Az + r_times_claim_Bz + r_sq * claim_Cz
      // We instead check: claim_inner_joint - Az - r_times_claim_Bz = r_sq * claim_Cz
      let claim_inner_joint = AllocatedNum::alloc(cs.namespace(|| "inner_joint"), || {
        let claim_Az_v = claim_Az
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let r_times_Bz_v = r_times_Bz
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let r_sq_v = r_sq.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let claim_Cz_v = claim_Cz
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        Ok(claim_Az_v + r_times_Bz_v + r_sq_v * claim_Cz_v)
      })?;
      cs.enforce(
        || "inner_joint_eq",
        |lc| lc + claim_Cz.get_variable(),
        |lc| lc + r_sq.get_variable(),
        |lc| {
          lc + claim_inner_joint.get_variable()
            - claim_Az.get_variable()
            - r_times_Bz.get_variable()
        },
      );

      Ok((vec![claim_inner_joint], vec![r]))
    } else if round_index >= self.idx_inner_start() && round_index < self.idx_inner_commit_w() {
      // Inner quadratic sum-check per-round consistency
      let idx = round_index - self.idx_inner_start();

      let coeffs = alloc_coeffs::<E, _>(
        cs.namespace(|| format!("inner_round_{idx}")),
        &self.inner_polys[idx],
      )?;
      let r_y =
        AllocatedNum::alloc_input(cs.namespace(|| format!("inner_round_{idx}_ry")), || {
          Ok(challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO))
        })?;

      // Sum-check consistency: h(0)+h(1)=prev_e
      let prev_e = &prior_round_vars.last().expect("prev inner round")[0];
      cs.enforce(
        || format!("inner_consistency_{idx}"),
        |lc| {
          lc + coeffs[0].get_variable()
            + coeffs[0].get_variable()
            + coeffs[1].get_variable()
            + coeffs[2].get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc + prev_e.get_variable(),
      );

      let p_ry = eval_poly_horner::<E, _>(
        cs.namespace(|| format!("inner_round_{idx}_poly_eval")),
        &coeffs,
        &r_y,
      )?;
      Ok((vec![p_ry], vec![r_y]))
    } else if round_index == self.idx_inner_commit_w() {
      // Dedicated commit round for eval_W only (padded to commitment width)
      let eval_W = AllocatedNum::alloc(cs.namespace(|| "eval_W"), || Ok(self.eval_W))?;

      // Pad to width
      let pad_width = MULTIROUND_COMMITMENT_WIDTH - 1;
      for j in 0..pad_width {
        alloc_zero::<E, _>(cs.namespace(|| format!("pad_eval_W_{j}")))?;
      }
      Ok((vec![eval_W], vec![]))
    } else if round_index == self.idx_inner_final() {
      // Final inner equality
      // The previous logical round whose claim must be matched is the last inner sum-check round
      let prev_e = &prior_round_vars[self.idx_inner_final() - 2][0];
      let r = &prev_challenges[self.idx_inner_setup()][0];
      let eval_W = &prior_round_vars[self.idx_inner_commit_w()][0];

      // Public inputs for eval_A, eval_B, eval_C, tau_at_rx, and eval_X are inputized here when known
      let eval_A = AllocatedNum::alloc(cs.namespace(|| "eval_A"), || Ok(self.eval_A))?;
      eval_A.inputize(cs.namespace(|| "eval_A_input"))?;
      let eval_B = AllocatedNum::alloc(cs.namespace(|| "eval_B"), || Ok(self.eval_B))?;
      eval_B.inputize(cs.namespace(|| "eval_B_input"))?;
      let eval_C = AllocatedNum::alloc(cs.namespace(|| "eval_C"), || Ok(self.eval_C))?;
      eval_C.inputize(cs.namespace(|| "eval_C_input"))?;
      let tau_at_rx = AllocatedNum::alloc(cs.namespace(|| "tau_at_rx"), || Ok(self.tau_at_rx))?;
      tau_at_rx.inputize(cs.namespace(|| "tau_at_rx_input"))?;

      let eval_X = AllocatedNum::alloc(cs.namespace(|| "eval_X"), || Ok(self.eval_X))?;
      eval_X.inputize(cs.namespace(|| "eval_X_input"))?;

      let r_y0 = &prev_challenges[self.idx_inner_start()][0];

      // tmp_w = eval_W * (1 - ry0)
      let tmp_w = AllocatedNum::alloc(cs.namespace(|| "tmp_w"), || {
        let ew = eval_W.get_value().unwrap_or(E::Scalar::ZERO);
        let om = E::Scalar::ONE - r_y0.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(ew * om)
      })?;
      cs.enforce(
        || "tmp_w_def",
        |lc| lc + eval_W.get_variable(),
        |lc| lc + CS::one() - r_y0.get_variable(),
        |lc| lc + tmp_w.get_variable(),
      );

      // sum_z_expected = tmp_w + eval_X * r_y0
      // sum_z_expected - tmp_w = eval_X * r_y0
      let sum_z_expected = AllocatedNum::alloc(cs.namespace(|| "sum_z_expected"), || {
        let tw = tmp_w.get_value().unwrap_or(E::Scalar::ZERO);
        let tx = eval_X.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(tw + tx * r_y0.get_value().unwrap_or(E::Scalar::ZERO))
      })?;
      cs.enforce(
        || "sum_z_expected_def",
        |lc| lc + eval_X.get_variable(),
        |lc| lc + r_y0.get_variable(),
        |lc| lc + sum_z_expected.get_variable() - tmp_w.get_variable(),
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

      // prev_e = (eval_A + tmp1 + tmp2) * sum_z_expected
      cs.enforce(
        || "prev_e = (eval_A + tmp1 + tmp2)*sum_z_expected",
        |lc| lc + eval_A.get_variable() + tmp1.get_variable() + tmp2.get_variable(),
        |lc| lc + sum_z_expected.get_variable(),
        |lc| lc + prev_e.get_variable(),
      );

      Ok((vec![], vec![]))
    } else {
      Err(SynthesisError::Unsatisfiable)
    }
  }

  fn num_rounds(&self) -> usize {
    self.idx_inner_final() + 1
  }
}

/// NeutronNova verifier circuit constraining computation across multiple rounds.
#[derive(Clone, Debug, Default)]
pub struct NeutronNovaVerifierCircuit<E: Engine> {
  pub(crate) outer_polys_step: Vec<[E::Scalar; 4]>,
  pub(crate) outer_polys_core: Vec<[E::Scalar; 4]>,

  pub(crate) claim_Az_step: E::Scalar,
  pub(crate) claim_Bz_step: E::Scalar,
  pub(crate) claim_Cz_step: E::Scalar,

  pub(crate) claim_Az_core: E::Scalar,
  pub(crate) claim_Bz_core: E::Scalar,
  pub(crate) claim_Cz_core: E::Scalar,

  // Fiat–Shamir coefficients mixing step & core branches are derived as round challenges
  pub(crate) tau_at_rx: E::Scalar,

  pub(crate) inner_polys_step: Vec<[E::Scalar; 3]>,
  pub(crate) inner_polys_core: Vec<[E::Scalar; 3]>,

  pub(crate) eval_A_step: E::Scalar,
  pub(crate) eval_B_step: E::Scalar,
  pub(crate) eval_C_step: E::Scalar,
  pub(crate) eval_W_step: E::Scalar,

  pub(crate) eval_A_core: E::Scalar,
  pub(crate) eval_B_core: E::Scalar,
  pub(crate) eval_C_core: E::Scalar,
  pub(crate) eval_W_core: E::Scalar,

  pub(crate) eval_X_step: E::Scalar,
  pub(crate) eval_X_core: E::Scalar,

  pub(crate) t_out_step: E::Scalar,
  // NIFS cubic sum-check polynomials (4 coeffs per round)
  pub(crate) nifs_polys: Vec<[E::Scalar; 4]>,
  // Accumulated equality value acc_eq = \prod_t eq(r_b_t; rho_t), provided as a public input
  pub(crate) rho_acc_at_rb: E::Scalar,
}

impl<E: Engine> NeutronNovaVerifierCircuit<E> {
  fn nifs_rounds(&self) -> usize {
    self.nifs_polys.len()
  }

  fn outer_rounds(&self) -> usize {
    self.outer_polys_step.len() // same as outer_polys_core.len()
  }
  fn inner_rounds(&self) -> usize {
    self.inner_polys_step.len() // same as inner_polys_core.len()
  }
  fn idx_nifs_final(&self) -> usize {
    self.nifs_rounds()
  }

  fn idx_outer_start(&self) -> usize {
    self.idx_nifs_final() + 1
  }

  fn idx_outer_final(&self) -> usize {
    self.idx_outer_start() + self.outer_rounds()
  }

  fn idx_inner_setup(&self) -> usize {
    self.idx_outer_final() + 1
  }

  fn idx_inner_start(&self) -> usize {
    self.idx_inner_setup() + 1
  }

  /// Returns the round index at which the circuit commits only to `eval_W` for the step circuit.
  fn idx_inner_commit_w_step(&self) -> usize {
    self.idx_inner_start() + self.inner_rounds()
  }

  /// Returns the round index at which the circuit commits only to `eval_W` for the core circuit.
  fn idx_inner_commit_w_core(&self) -> usize {
    self.idx_inner_commit_w_step() + 1
  }

  fn idx_inner_final(&self) -> usize {
    self.idx_inner_commit_w_core() + 1
  }
}

impl<E: Engine> NeutronNovaVerifierCircuit<E> {
  // Enforces outer branch equality: prev_e = tau * (Az*Bz - Cz)
  fn build_outer_branch<CSType: ConstraintSystem<E::Scalar>>(
    cs_branch: &mut CSType,
    Az: &AllocatedNum<E::Scalar>,
    Bz: &AllocatedNum<E::Scalar>,
    Cz: &AllocatedNum<E::Scalar>,
    prev_e: &AllocatedNum<E::Scalar>,
    tau: &AllocatedNum<E::Scalar>,
    label: &str,
  ) -> Result<(), SynthesisError> {
    let prod = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_prod")), || {
      let a = Az.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let b = Bz.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(a * b)
    })?;
    cs_branch.enforce(
      || format!("{label}_prod_enf"),
      |lc| lc + Az.get_variable(),
      |lc| lc + Bz.get_variable(),
      |lc| lc + prod.get_variable(),
    );

    let diff = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_diff")), || {
      let p = prod.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let c = Cz.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(p - c)
    })?;
    cs_branch.enforce(
      || format!("{label}_diff_enf"),
      |lc| lc + prod.get_variable() - Cz.get_variable(),
      |lc| lc + CSType::one(),
      |lc| lc + diff.get_variable(),
    );
    let expected =
      AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_expected")), || {
        let t = tau.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let d = diff.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(t * d)
      })?;
    cs_branch.enforce(
      || format!("{label}_expected_enf"),
      |lc| lc + tau.get_variable(),
      |lc| lc + diff.get_variable(),
      |lc| lc + expected.get_variable(),
    );

    // prev_e == expected
    cs_branch.enforce(
      || format!("{label}_equality"),
      |lc| lc + prev_e.get_variable(),
      |lc| lc + CSType::one(),
      |lc| lc + expected.get_variable(),
    );
    Ok(())
  }

  // Computes joint claim: Az + r*Bz + r^2*Cz
  fn compute_joint_claim<CSType: ConstraintSystem<E::Scalar>>(
    cs_branch: &mut CSType,
    Az: &AllocatedNum<E::Scalar>,
    Bz: &AllocatedNum<E::Scalar>,
    Cz: &AllocatedNum<E::Scalar>,
    r: &AllocatedNum<E::Scalar>,
    r_sq: &AllocatedNum<E::Scalar>,
    label: &str,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
    let tmp1 = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_tmp1")), || {
      let b = Bz.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let rv = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(b * rv)
    })?;
    cs_branch.enforce(
      || format!("{label}_tmp1_enf"),
      |lc| lc + Bz.get_variable(),
      |lc| lc + r.get_variable(),
      |lc| lc + tmp1.get_variable(),
    );

    let tmp2 = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_tmp2")), || {
      let c = Cz.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let rsq = r_sq.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(c * rsq)
    })?;
    cs_branch.enforce(
      || format!("{label}_tmp2_enf"),
      |lc| lc + Cz.get_variable(),
      |lc| lc + r_sq.get_variable(),
      |lc| lc + tmp2.get_variable(),
    );

    let joint = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_joint")), || {
      let a = Az.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let t1 = tmp1.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let t2 = tmp2.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(a + t1 + t2)
    })?;
    cs_branch.enforce(
      || format!("{label}_joint_enf"),
      |lc| lc + Az.get_variable() + tmp1.get_variable() + tmp2.get_variable(),
      |lc| lc + CSType::one(),
      |lc| lc + joint.get_variable(),
    );
    Ok(joint)
  }

  // Enforces inner branch equality with eval_Z computation
  fn build_inner_branch<CSType: ConstraintSystem<E::Scalar>>(
    cs_branch: &mut CSType,
    params: &InnerBranchParams<'_, E::Scalar>,
  ) -> Result<(), SynthesisError> {
    let InnerBranchParams {
      eval_A,
      eval_B,
      eval_C,
      eval_W,
      prev_e,
      r,
      eval_X,
      one_minus_ry0,
      r_y0,
      label,
    } = params;
    // r^2
    let r_sq = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_r_sq")), || {
      let rv = r.get_value().unwrap_or(E::Scalar::ZERO);
      Ok(rv * rv)
    })?;
    cs_branch.enforce(
      || format!("{label}_r_sq_enf"),
      |lc| lc + r.get_variable(),
      |lc| lc + r.get_variable(),
      |lc| lc + r_sq.get_variable(),
    );

    // tmp1 = eval_B * r
    let tmp1 = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_tmp1")), || {
      let b = eval_B
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let rv = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(b * rv)
    })?;
    cs_branch.enforce(
      || format!("{label}_tmp1_enf"),
      |lc| lc + eval_B.get_variable(),
      |lc| lc + r.get_variable(),
      |lc| lc + tmp1.get_variable(),
    );

    // tmp2 = eval_C * r_sq
    let tmp2 = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_tmp2")), || {
      let c = eval_C
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let rsq = r_sq.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(c * rsq)
    })?;
    cs_branch.enforce(
      || format!("{label}_tmp2_enf"),
      |lc| lc + eval_C.get_variable(),
      |lc| lc + r_sq.get_variable(),
      |lc| lc + tmp2.get_variable(),
    );

    // sum = eval_A + tmp1 + tmp2
    let sum = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_sum")), || {
      let a = eval_A
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let t1 = tmp1.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let t2 = tmp2.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(a + t1 + t2)
    })?;
    cs_branch.enforce(
      || format!("{label}_sum_enf"),
      |lc| lc + eval_A.get_variable() + tmp1.get_variable() + tmp2.get_variable(),
      |lc| lc + CSType::one(),
      |lc| lc + sum.get_variable(),
    );

    // tmp_w = eval_W * one_minus_ry0
    let tmp_w = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_tmp_w")), || {
      let ew = eval_W.get_value().unwrap_or(E::Scalar::ZERO);
      let om = one_minus_ry0.get_value().unwrap_or(E::Scalar::ZERO);
      Ok(ew * om)
    })?;
    cs_branch.enforce(
      || format!("{label}_tmp_w_enf"),
      |lc| lc + eval_W.get_variable(),
      |lc| lc + one_minus_ry0.get_variable(),
      |lc| lc + tmp_w.get_variable(),
    );

    // tmp_x = eval_X * r_y0
    let tmp_x = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_tmp_x")), || {
      let ex = eval_X.get_value().unwrap_or(E::Scalar::ZERO);
      let ry = r_y0.get_value().unwrap_or(E::Scalar::ZERO);
      Ok(ex * ry)
    })?;
    cs_branch.enforce(
      || format!("{label}_tmp_x_enf"),
      |lc| lc + eval_X.get_variable(),
      |lc| lc + r_y0.get_variable(),
      |lc| lc + tmp_x.get_variable(),
    );

    // eval_Z = tmp_w + tmp_x
    let eval_Z = AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_eval_Z")), || {
      let tw = tmp_w.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let tx = tmp_x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(tw + tx)
    })?;
    cs_branch.enforce(
      || format!("{label}_eval_Z_enf"),
      |lc| lc + tmp_w.get_variable() + tmp_x.get_variable(),
      |lc| lc + CSType::one(),
      |lc| lc + eval_Z.get_variable(),
    );

    // expected = sum * eval_Z
    let expected =
      AllocatedNum::alloc(cs_branch.namespace(|| format!("{label}_expected")), || {
        let s = sum.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let z = eval_Z
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        Ok(s * z)
      })?;
    cs_branch.enforce(
      || format!("{label}_expected_enf"),
      |lc| lc + sum.get_variable(),
      |lc| lc + eval_Z.get_variable(),
      |lc| lc + expected.get_variable(),
    );

    // prev_e == expected
    cs_branch.enforce(
      || format!("{label}_equality"),
      |lc| lc + prev_e.get_variable(),
      |lc| lc + CSType::one(),
      |lc| lc + expected.get_variable(),
    );
    Ok(())
  }
}

// Helper struct to group parameters for build_inner_branch
struct InnerBranchParams<'a, F: PrimeField> {
  eval_A: &'a AllocatedNum<F>,
  eval_B: &'a AllocatedNum<F>,
  eval_C: &'a AllocatedNum<F>,
  eval_W: &'a AllocatedNum<F>,
  prev_e: &'a AllocatedNum<F>,
  r: &'a AllocatedNum<F>,
  eval_X: &'a AllocatedNum<F>,
  one_minus_ry0: &'a AllocatedNum<F>,
  r_y0: &'a AllocatedNum<F>,
  label: &'a str,
}

impl<E: Engine> MultiRoundCircuit<E> for NeutronNovaVerifierCircuit<E> {
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    // Public inputs used by NeutronNovaSNARK verifier instance:
    // [eval_A_step, eval_B_step, eval_C_step,
    //  eval_A_core, eval_B_core, eval_C_core,
    //  tau_at_rx, eval_X_step, eval_X_core, rho_acc_at_rb]
    Ok(vec![
      self.eval_A_step,
      self.eval_B_step,
      self.eval_C_step,
      self.eval_A_core,
      self.eval_B_core,
      self.eval_C_core,
      self.tau_at_rx,
      self.eval_X_step,
      self.eval_X_core,
      self.rho_acc_at_rb,
    ])
  }

  fn num_challenges(&self, round_index: usize) -> Result<usize, SynthesisError> {
    if round_index < self.nifs_rounds() {
      return Ok(1); // r_b only; rho_t provided as witness
    }
    if round_index == self.idx_nifs_final() {
      return Ok(0);
    }
    if round_index >= self.idx_outer_start() && round_index < self.idx_outer_final() {
      return Ok(1); // r_x
    }
    if round_index == self.idx_outer_final() {
      return Ok(1); // c_outer challenge
    }
    if round_index == self.idx_inner_setup() {
      return Ok(1); // r
    }
    if round_index >= self.idx_inner_start() && round_index < self.idx_inner_commit_w_step() {
      return Ok(1); // r_y[i]
    }
    if round_index == self.idx_inner_commit_w_step() {
      return Ok(0); // commit eval_W_step
    }
    if round_index == self.idx_inner_commit_w_core() {
      return Ok(0); // commit eval_W_core
    }
    if round_index == self.idx_inner_final() {
      return Ok(1); // c_inner challenge
    }
    Err(SynthesisError::Unsatisfiable)
  }

  fn rounds<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    round_index: usize,
    prior_round_vars: &[Vec<AllocatedNum<E::Scalar>>],
    prev_challenges: &[Vec<AllocatedNum<E::Scalar>>],
    challenges: Option<&[E::Scalar]>,
  ) -> Result<(Vec<AllocatedNum<E::Scalar>>, Vec<AllocatedNum<E::Scalar>>), SynthesisError> {
    // NIFS cubic sum-check rounds
    if round_index < self.nifs_rounds() {
      let i = round_index;
      let mut ns_round = cs.namespace(|| format!("nifs_round_{i}"));

      let coeffs_raw = self.nifs_polys[i];
      let coeffs: Vec<_> = coeffs_raw
        .iter()
        .enumerate()
        .map(|(j, c)| {
          AllocatedNum::alloc(ns_round.namespace(|| format!("coef_{i}_{j}")), || Ok(*c))
        })
        .collect::<Result<_, _>>()?;

      // Enforce g_i(0)+g_i(1)=prev_claim
      // sum_p01 = 2*c0 + c1 + c2 + c3
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
        // Initial NIFS claim is zero
        ns_round.enforce(
          || "nifs_init_claim_zero",
          |lc| lc + sum_p01.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc, // 0
        );
      } else {
        // Carry-over relation: g_i(0) + g_i(1) must equal the previous round's claim e_{i-1} = p_{i-1}(r_{b,i-1}).
        // The previous round exposes p_{i-1}(r_b) as its first output variable (index 0).
        let prev_p_rb = &prior_round_vars[i - 1][0];
        ns_round.enforce(
          || format!("nifs_consistency_{i}"),
          |lc| lc + sum_p01.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + prev_p_rb.get_variable(),
        );
      }

      let r_val = challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO);
      let r_b = AllocatedNum::alloc_input(ns_round.namespace(|| "r_b"), || Ok(r_val))?;

      let p_rb = {
        let mut ns_eval = ns_round.namespace(|| "poly_eval");
        eval_poly_horner::<E, _>(&mut ns_eval, &coeffs, &r_b)?
      };
      // Expose both p_rb and the current round's claim sum so the next round can
      // consume the claim as its expected sum.
      return Ok((vec![p_rb, sum_p01], vec![r_b]));
    } else if round_index == self.idx_nifs_final() {
      // Final NIFS equality (mirrors prover): p_last(r_b_last) == rho_acc_at_rb * t_out_step
      // The previous NIFS round exposes [p_rb, carry]; bind p_rb to the equality.
      let prev_vars = &prior_round_vars[self.idx_nifs_final() - 1];
      let p_rb_last = &prev_vars[0];

      let t_out_step = AllocatedNum::alloc(cs.namespace(|| "t_out_step"), || Ok(self.t_out_step))?;
      let rho_acc = AllocatedNum::alloc(cs.namespace(|| "rho_acc_at_rb_nifs"), || {
        Ok(self.rho_acc_at_rb)
      })?;

      let rhs = AllocatedNum::alloc(cs.namespace(|| "nifs_rhs"), || {
        let a = rho_acc.get_value().unwrap_or(E::Scalar::ZERO);
        let b = t_out_step.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(a * b)
      })?;
      cs.enforce(
        || "nifs_rhs_enf",
        |lc| lc + rho_acc.get_variable(),
        |lc| lc + t_out_step.get_variable(),
        |lc| lc + rhs.get_variable(),
      );

      cs.enforce(
        || "nifs_final_eq",
        |lc| lc + p_rb_last.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + rhs.get_variable(),
      );

      // Expose rho_acc so later rounds (inner-final) can inputize it as a public value without re-allocation
      return Ok((vec![p_rb_last.clone(), rho_acc.clone()], vec![]));
    }
    if round_index >= self.idx_outer_start() && round_index < self.idx_outer_final() {
      let i = round_index - self.idx_outer_start();
      let mut ns_round = cs.namespace(|| format!("outer_round_{i}"));

      let coeffs_step_raw = self.outer_polys_step[i];
      let coeffs_core_raw = self.outer_polys_core[i];
      let mut coeffs_step = Vec::new();
      for (j, c) in coeffs_step_raw.iter().enumerate() {
        coeffs_step.push(AllocatedNum::alloc(
          ns_round.namespace(|| format!("coef_step_{i}_{j}")),
          || Ok(*c),
        )?);
      }

      let mut coeffs_core = Vec::new();
      for (j, c) in coeffs_core_raw.iter().enumerate() {
        coeffs_core.push(AllocatedNum::alloc(
          ns_round.namespace(|| format!("coef_core_{i}_{j}")),
          || Ok(*c),
        )?);
      }

      let r_val = challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO);
      let r = AllocatedNum::alloc_input(ns_round.namespace(|| "r_x"), || Ok(r_val))?;
      let sum_p01_step = AllocatedNum::alloc(ns_round.namespace(|| "sum_p01_step"), || {
        Ok(
          coeffs_step_raw[0]
            + coeffs_step_raw[0]
            + coeffs_step_raw[1]
            + coeffs_step_raw[2]
            + coeffs_step_raw[3],
        )
      })?;
      let sum_p01_core = AllocatedNum::alloc(ns_round.namespace(|| "sum_p01_core"), || {
        Ok(
          coeffs_core_raw[0]
            + coeffs_core_raw[0]
            + coeffs_core_raw[1]
            + coeffs_core_raw[2]
            + coeffs_core_raw[3],
        )
      })?;

      // Enforce definitions of the sums
      ns_round.enforce(
        || "sum_p01_step_def",
        |lc| lc + sum_p01_step.get_variable(),
        |lc| lc + CS::one(),
        |lc| {
          lc + coeffs_step[0].get_variable()
            + coeffs_step[0].get_variable()
            + coeffs_step[1].get_variable()
            + coeffs_step[2].get_variable()
            + coeffs_step[3].get_variable()
        },
      );
      ns_round.enforce(
        || "sum_p01_core_def",
        |lc| lc + sum_p01_core.get_variable(),
        |lc| lc + CS::one(),
        |lc| {
          lc + coeffs_core[0].get_variable()
            + coeffs_core[0].get_variable()
            + coeffs_core[1].get_variable()
            + coeffs_core[2].get_variable()
            + coeffs_core[3].get_variable()
        },
      );

      // Consistency with previous round
      if i == 0 {
        // The step branch starts from the folded instance target T_out, whereas the core branch
        // starts from 0 (it is mixed in later via c_outer).

        // Allocate witness T_out for the step branch
        let t_out_step =
          AllocatedNum::alloc(ns_round.namespace(|| "t_out_step"), || Ok(self.t_out_step))?;

        // Enforce sum_p01_step == t_out_step
        ns_round.enforce(
          || "outer_init_step",
          |lc| lc + sum_p01_step.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + t_out_step.get_variable(),
        );

        // Core branch still starts from zero
        ns_round.enforce(
          || "outer_init_zero_core",
          |lc| lc + sum_p01_core.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc, // constant 0
        );
      } else {
        // Reference the immediately previous OUTER round explicitly.
        let prev_vars = &prior_round_vars[self.idx_outer_start() + i - 1];
        let prev_e_step = &prev_vars[0];
        let prev_e_core = &prev_vars[1];
        ns_round.enforce(
          || format!("outer_consistency_step_{i}"),
          |lc| lc + sum_p01_step.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + prev_e_step.get_variable(),
        );
        ns_round.enforce(
          || format!("outer_consistency_core_{i}"),
          |lc| lc + sum_p01_core.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + prev_e_core.get_variable(),
        );
      }

      // Evaluate polynomials at r
      let p_r_step = {
        let mut ns_eval = ns_round.namespace(|| "poly_eval_step");
        eval_poly_horner::<E, _>(&mut ns_eval, &coeffs_step, &r)?
      };
      let p_r_core = {
        let mut ns_eval = ns_round.namespace(|| "poly_eval_core");
        eval_poly_horner::<E, _>(&mut ns_eval, &coeffs_core, &r)?
      };

      Ok((vec![p_r_step, p_r_core], vec![r]))
    } else if round_index == self.idx_outer_final() {
      // Outer final equality round
      // Previous variables come from the previous OUTER round explicitly
      let prev_vars = &prior_round_vars[self.idx_outer_final() - 1];
      let prev_e_step = &prev_vars[0];
      let prev_e_core = &prev_vars[1];

      let claim_Az_step =
        AllocatedNum::alloc(cs.namespace(|| "Az_step"), || Ok(self.claim_Az_step))?;
      let claim_Bz_step =
        AllocatedNum::alloc(cs.namespace(|| "Bz_step"), || Ok(self.claim_Bz_step))?;
      let claim_Cz_step =
        AllocatedNum::alloc(cs.namespace(|| "Cz_step"), || Ok(self.claim_Cz_step))?;

      let claim_Az_core =
        AllocatedNum::alloc(cs.namespace(|| "Az_core"), || Ok(self.claim_Az_core))?;
      let claim_Bz_core =
        AllocatedNum::alloc(cs.namespace(|| "Bz_core"), || Ok(self.claim_Bz_core))?;
      let claim_Cz_core =
        AllocatedNum::alloc(cs.namespace(|| "Cz_core"), || Ok(self.claim_Cz_core))?;

      let c_outer_val = challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO);
      let c_outer = AllocatedNum::alloc_input(cs.namespace(|| "c_outer"), || Ok(c_outer_val))?;
      let tau = AllocatedNum::alloc(cs.namespace(|| "tau_at_rx_outer"), || Ok(self.tau_at_rx))?;

      // Step branch constraints
      {
        let mut ns_step = cs.namespace(|| "outer_final_step");
        Self::build_outer_branch(
          &mut ns_step,
          &claim_Az_step,
          &claim_Bz_step,
          &claim_Cz_step,
          prev_e_step,
          &tau,
          "step",
        )?;
      }
      // Core branch constraints
      {
        let mut ns_core = cs.namespace(|| "outer_final_core");
        Self::build_outer_branch(
          &mut ns_core,
          &claim_Az_core,
          &claim_Bz_core,
          &claim_Cz_core,
          prev_e_core,
          &tau,
          "core",
        )?;
      }
      // Combined outer equality with coefficient c_outer
      let prod_c_prev_e_core = AllocatedNum::alloc(cs.namespace(|| "c_prev_prod"), || {
        let c = c_outer.get_value().unwrap_or(E::Scalar::ZERO);
        let pec = prev_e_core.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(c * pec)
      })?;
      cs.enforce(
        || "c_prev_prod_enf",
        |lc| lc + c_outer.get_variable(),
        |lc| lc + prev_e_core.get_variable(),
        |lc| lc + prod_c_prev_e_core.get_variable(),
      );

      let lhs = AllocatedNum::alloc(cs.namespace(|| "lhs_comb"), || {
        let pes = prev_e_step.get_value().unwrap_or(E::Scalar::ZERO);
        let add = prod_c_prev_e_core.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(pes + add)
      })?;
      cs.enforce(
        || "lhs_comb_enf",
        |lc| lc + prev_e_step.get_variable() + prod_c_prev_e_core.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + lhs.get_variable(),
      );

      // diff_step and diff_core values derived earlier (we recompute compactly)
      let diff_step = {
        let mut ns = cs.namespace(|| "diff_step_inline");
        let prod = AllocatedNum::alloc(ns.namespace(|| "prod"), || {
          Ok(self.claim_Az_step * self.claim_Bz_step)
        })?;
        ns.enforce(
          || "prod_enf",
          |lc| lc + claim_Az_step.get_variable(),
          |lc| lc + claim_Bz_step.get_variable(),
          |lc| lc + prod.get_variable(),
        );
        let diff = AllocatedNum::alloc(ns.namespace(|| "diff"), || {
          Ok(self.claim_Az_step * self.claim_Bz_step - self.claim_Cz_step)
        })?;
        ns.enforce(
          || "diff_enf",
          |lc| lc + prod.get_variable() - claim_Cz_step.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + diff.get_variable(),
        );
        diff
      };

      let diff_core = {
        let mut ns = cs.namespace(|| "diff_core_inline");
        let prod = AllocatedNum::alloc(ns.namespace(|| "prod"), || {
          Ok(self.claim_Az_core * self.claim_Bz_core)
        })?;
        ns.enforce(
          || "prod_enf",
          |lc| lc + claim_Az_core.get_variable(),
          |lc| lc + claim_Bz_core.get_variable(),
          |lc| lc + prod.get_variable(),
        );
        let diff = AllocatedNum::alloc(ns.namespace(|| "diff"), || {
          Ok(self.claim_Az_core * self.claim_Bz_core - self.claim_Cz_core)
        })?;
        ns.enforce(
          || "diff_enf",
          |lc| lc + prod.get_variable() - claim_Cz_core.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + diff.get_variable(),
        );
        diff
      };

      let prod_c_diff_core = AllocatedNum::alloc(cs.namespace(|| "c_diff_prod"), || {
        let c = c_outer.get_value().unwrap_or(E::Scalar::ZERO);
        let dc = diff_core.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(c * dc)
      })?;
      cs.enforce(
        || "c_diff_prod_enf",
        |lc| lc + c_outer.get_variable(),
        |lc| lc + diff_core.get_variable(),
        |lc| lc + prod_c_diff_core.get_variable(),
      );

      let rhs_inner = AllocatedNum::alloc(cs.namespace(|| "rhs_inner"), || {
        let ds = diff_step.get_value().unwrap_or(E::Scalar::ZERO);
        let add = prod_c_diff_core.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(ds + add)
      })?;
      cs.enforce(
        || "rhs_inner_enf",
        |lc| lc + diff_step.get_variable() + prod_c_diff_core.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + rhs_inner.get_variable(),
      );

      let rhs = AllocatedNum::alloc(cs.namespace(|| "rhs_comb"), || {
        let tauv = tau.get_value().unwrap_or(E::Scalar::ZERO);
        let inner = rhs_inner.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(tauv * inner)
      })?;
      cs.enforce(
        || "rhs_comb_enf",
        |lc| lc + tau.get_variable(),
        |lc| lc + rhs_inner.get_variable(),
        |lc| lc + rhs.get_variable(),
      );

      cs.enforce(
        || "outer_combined_equality",
        |lc| lc + lhs.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + rhs.get_variable(),
      );

      Ok((
        vec![
          prev_e_step.clone(),
          prev_e_core.clone(),
          claim_Az_step,
          claim_Bz_step,
          claim_Cz_step,
          claim_Az_core,
          claim_Bz_core,
          claim_Cz_core,
        ],
        vec![c_outer],
      ))
    } else if round_index == self.idx_inner_setup() {
      // Inner setup round
      let mut ns_setup = cs.namespace(|| "inner_setup");

      let r_val = challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO);
      let r = AllocatedNum::alloc_input(ns_setup.namespace(|| "r"), || Ok(r_val))?;
      let r_sq = AllocatedNum::alloc(ns_setup.namespace(|| "r_sq"), || {
        let rv = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(rv * rv)
      })?;
      ns_setup.enforce(
        || "r_sq = r*r",
        |lc| lc + r.get_variable(),
        |lc| lc + r.get_variable(),
        |lc| lc + r_sq.get_variable(),
      );

      // Fetch Az, Bz, Cz values from outer-final outputs
      let outer_vars = &prior_round_vars[self.idx_outer_final()];
      let Az_step = &outer_vars[2];
      let Bz_step = &outer_vars[3];
      let Cz_step = &outer_vars[4];
      let Az_core = &outer_vars[5];
      let Bz_core = &outer_vars[6];
      let Cz_core = &outer_vars[7];

      // Compute for step and core
      let joint_step = {
        let mut ns_step = ns_setup.namespace(|| "step");
        Self::compute_joint_claim(&mut ns_step, Az_step, Bz_step, Cz_step, &r, &r_sq, "step")?
      };
      let joint_core = {
        let mut ns_core = ns_setup.namespace(|| "core");
        Self::compute_joint_claim(&mut ns_core, Az_core, Bz_core, Cz_core, &r, &r_sq, "core")?
      };

      Ok((vec![joint_step, joint_core], vec![r]))
    } else if round_index >= self.idx_inner_start() && round_index < self.idx_inner_commit_w_step()
    {
      // Inner quadratic sum-check rounds
      let idx = round_index - self.idx_inner_start();
      let mut ns_round = cs.namespace(|| format!("inner_round_{idx}"));

      let coeffs_step_raw = self.inner_polys_step[idx];
      let coeffs_core_raw = self.inner_polys_core[idx];
      let mut coeffs_step = Vec::new();
      for (j, c) in coeffs_step_raw.iter().enumerate() {
        coeffs_step.push(AllocatedNum::alloc(
          ns_round.namespace(|| format!("coef_inner_step_{idx}_{j}")),
          || Ok(*c),
        )?);
      }

      let mut coeffs_core = Vec::new();
      for (j, c) in coeffs_core_raw.iter().enumerate() {
        coeffs_core.push(AllocatedNum::alloc(
          ns_round.namespace(|| format!("coef_inner_core_{idx}_{j}")),
          || Ok(*c),
        )?);
      }

      let r_val = challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO);
      let r = AllocatedNum::alloc_input(ns_round.namespace(|| "r_y"), || Ok(r_val))?;
      let sum_p01_step = AllocatedNum::alloc(ns_round.namespace(|| "sum_p01_step"), || {
        Ok(coeffs_step_raw[0] + coeffs_step_raw[0] + coeffs_step_raw[1] + coeffs_step_raw[2])
      })?;
      let sum_p01_core = AllocatedNum::alloc(ns_round.namespace(|| "sum_p01_core"), || {
        Ok(coeffs_core_raw[0] + coeffs_core_raw[0] + coeffs_core_raw[1] + coeffs_core_raw[2])
      })?;

      // Enforce definitions
      ns_round.enforce(
        || "sum_p01_step_def",
        |lc| lc + sum_p01_step.get_variable(),
        |lc| lc + CS::one(),
        |lc| {
          lc + coeffs_step[0].get_variable()
            + coeffs_step[0].get_variable()
            + coeffs_step[1].get_variable()
            + coeffs_step[2].get_variable()
        },
      );
      ns_round.enforce(
        || "sum_p01_core_def",
        |lc| lc + sum_p01_core.get_variable(),
        |lc| lc + CS::one(),
        |lc| {
          lc + coeffs_core[0].get_variable()
            + coeffs_core[0].get_variable()
            + coeffs_core[1].get_variable()
            + coeffs_core[2].get_variable()
        },
      );

      // Consistency with previous claims
      let prev_vars = if idx == 0 {
        // The first inner round should match the values from inner_setup
        &prior_round_vars[self.idx_inner_setup()]
      } else {
        // Otherwise, use the immediately previous inner round
        &prior_round_vars[self.idx_inner_start() + idx - 1]
      };
      let prev_e_step = &prev_vars[0];
      let prev_e_core = &prev_vars[1];
      ns_round.enforce(
        || format!("inner_consistency_step_{idx}"),
        |lc| lc + sum_p01_step.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + prev_e_step.get_variable(),
      );
      ns_round.enforce(
        || format!("inner_consistency_core_{idx}"),
        |lc| lc + sum_p01_core.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + prev_e_core.get_variable(),
      );

      // Evaluate polynomials at r
      let p_r_step = {
        let mut ns_eval = ns_round.namespace(|| "poly_eval_step");
        eval_poly_horner::<E, _>(&mut ns_eval, &coeffs_step, &r)?
      };
      let p_r_core = {
        let mut ns_eval = ns_round.namespace(|| "poly_eval_core");
        eval_poly_horner::<E, _>(&mut ns_eval, &coeffs_core, &r)?
      };

      Ok((vec![p_r_step, p_r_core], vec![r]))
    } else if round_index == self.idx_inner_commit_w_step() {
      // Commit round for eval_W_step
      let eval_W_step = AllocatedNum::alloc(cs.namespace(|| "eval_W_step_commit"), || {
        Ok(self.eval_W_step)
      })?;
      // Pad to per-round commitment width
      let pad_width = MULTIROUND_COMMITMENT_WIDTH - 1;
      for j in 0..pad_width {
        alloc_zero::<E, _>(cs.namespace(|| format!("pad_eval_W_step_{j}")))?;
      }
      Ok((vec![eval_W_step], vec![]))
    } else if round_index == self.idx_inner_commit_w_core() {
      // Commit round for eval_W_core
      let eval_W_core = AllocatedNum::alloc(cs.namespace(|| "eval_W_core_commit"), || {
        Ok(self.eval_W_core)
      })?;
      // Pad to per-round commitment width
      let pad_width = MULTIROUND_COMMITMENT_WIDTH - 1;
      for j in 0..pad_width {
        alloc_zero::<E, _>(cs.namespace(|| format!("pad_eval_W_core_{j}")))?;
      }
      Ok((vec![eval_W_core], vec![]))
    } else if round_index == self.idx_inner_final() {
      // Inner final equality round
      // Use outputs from the last inner sum-check round (before commit rounds)
      let prev_vars = &prior_round_vars[self.idx_inner_commit_w_step() - 1];
      let prev_e_step = &prev_vars[0];
      let prev_e_core = &prev_vars[1];

      let r = &prev_challenges[self.idx_inner_setup()][0];
      let r_y0 = &prev_challenges[self.idx_inner_start()][0];

      // Challenge must be inputized before any public inputs; mirror r_b style
      let c_inner_val = challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO);
      let c_inner = AllocatedNum::alloc_input(cs.namespace(|| "c_inner"), || Ok(c_inner_val))?;

      let eval_A_step =
        AllocatedNum::alloc(cs.namespace(|| "eval_A_step"), || Ok(self.eval_A_step))?;
      eval_A_step.inputize(cs.namespace(|| "eval_A_step_inp"))?;
      let eval_B_step =
        AllocatedNum::alloc(cs.namespace(|| "eval_B_step"), || Ok(self.eval_B_step))?;
      eval_B_step.inputize(cs.namespace(|| "eval_B_step_inp"))?;
      let eval_C_step =
        AllocatedNum::alloc(cs.namespace(|| "eval_C_step"), || Ok(self.eval_C_step))?;
      eval_C_step.inputize(cs.namespace(|| "eval_C_step_inp"))?;

      let eval_A_core =
        AllocatedNum::alloc(cs.namespace(|| "eval_A_core"), || Ok(self.eval_A_core))?;
      eval_A_core.inputize(cs.namespace(|| "eval_A_core_inp"))?;
      let eval_B_core =
        AllocatedNum::alloc(cs.namespace(|| "eval_B_core"), || Ok(self.eval_B_core))?;
      eval_B_core.inputize(cs.namespace(|| "eval_B_core_inp"))?;
      let eval_C_core =
        AllocatedNum::alloc(cs.namespace(|| "eval_C_core"), || Ok(self.eval_C_core))?;
      eval_C_core.inputize(cs.namespace(|| "eval_C_core_inp"))?;

      // tau_at_rx is a public input for binding to the transcript-derived value
      let tau_at_rx = AllocatedNum::alloc(cs.namespace(|| "tau_at_rx"), || Ok(self.tau_at_rx))?;
      tau_at_rx.inputize(cs.namespace(|| "tau_at_rx_inp"))?;

      let eval_X_step =
        AllocatedNum::alloc(cs.namespace(|| "eval_X_step"), || Ok(self.eval_X_step))?;
      eval_X_step.inputize(cs.namespace(|| "eval_X_step_inp"))?;
      let eval_X_core =
        AllocatedNum::alloc(cs.namespace(|| "eval_X_core"), || Ok(self.eval_X_core))?;
      eval_X_core.inputize(cs.namespace(|| "eval_X_core_inp"))?;

      let rho_acc_from_nifs = &prior_round_vars[self.idx_nifs_final()][1];
      rho_acc_from_nifs.inputize(cs.namespace(|| "rho_acc_at_rb_inp"))?;
      // Fetch committed eval_W values from the dedicated commit rounds
      let eval_W_step = &prior_round_vars[self.idx_inner_commit_w_step()][0];
      let eval_W_core = &prior_round_vars[self.idx_inner_commit_w_core()][0];

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

      // Combined inner equality with coefficient c_inner
      let r_sq_var = AllocatedNum::alloc(cs.namespace(|| "r_sq_inner_final"), || {
        let rv = r.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(rv * rv)
      })?;
      cs.enforce(
        || "r_sq_inner_final = r*r",
        |lc| lc + r.get_variable(),
        |lc| lc + r.get_variable(),
        |lc| lc + r_sq_var.get_variable(),
      );

      // Compute step branch: sum = eval_A + r*eval_B + r^2*eval_C and eval_Z
      let (sum_step_comb, evalZ_step) = {
        let mut ns = cs.namespace(|| "branch_step_comb");

        // tmp1 = eval_B * r
        let tmp1 = AllocatedNum::alloc(ns.namespace(|| "stepc_tmp1"), || {
          let b = eval_B_step
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          let rv = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok(b * rv)
        })?;
        ns.enforce(
          || "stepc_tmp1_enf",
          |lc| lc + eval_B_step.get_variable(),
          |lc| lc + r.get_variable(),
          |lc| lc + tmp1.get_variable(),
        );

        // tmp2 = eval_C * r_sq
        let tmp2 = AllocatedNum::alloc(ns.namespace(|| "stepc_tmp2"), || {
          let c = eval_C_step
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          let rsq = r_sq_var
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          Ok(c * rsq)
        })?;
        ns.enforce(
          || "stepc_tmp2_enf",
          |lc| lc + eval_C_step.get_variable(),
          |lc| lc + r_sq_var.get_variable(),
          |lc| lc + tmp2.get_variable(),
        );

        // sum = eval_A + tmp1 + tmp2
        let sum = AllocatedNum::alloc(ns.namespace(|| "stepc_sum"), || {
          let a = eval_A_step
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          let t1 = tmp1.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          let t2 = tmp2.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok(a + t1 + t2)
        })?;
        ns.enforce(
          || "stepc_sum_enf",
          |lc| lc + eval_A_step.get_variable() + tmp1.get_variable() + tmp2.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + sum.get_variable(),
        );

        // eval_Z = (1 - r_y0)*eval_W + r_y0*eval_X
        let tmpw = AllocatedNum::alloc(ns.namespace(|| "stepc_tmpw"), || {
          let ew = eval_W_step.get_value().unwrap_or(E::Scalar::ZERO);
          let om = one_minus_ry0.get_value().unwrap_or(E::Scalar::ZERO);
          Ok(ew * om)
        })?;
        ns.enforce(
          || "stepc_tmpw_enf",
          |lc| lc + eval_W_step.get_variable(),
          |lc| lc + one_minus_ry0.get_variable(),
          |lc| lc + tmpw.get_variable(),
        );
        let tmpx = AllocatedNum::alloc(ns.namespace(|| "stepc_tmpx"), || {
          let ex = eval_X_step.get_value().unwrap_or(E::Scalar::ZERO);
          let ry = r_y0.get_value().unwrap_or(E::Scalar::ZERO);
          Ok(ex * ry)
        })?;
        ns.enforce(
          || "stepc_tmpx_enf",
          |lc| lc + eval_X_step.get_variable(),
          |lc| lc + r_y0.get_variable(),
          |lc| lc + tmpx.get_variable(),
        );
        let eval_Z = AllocatedNum::alloc(ns.namespace(|| "stepc_evalZ"), || {
          let wv = tmpw.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          let xv = tmpx.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok(wv + xv)
        })?;
        ns.enforce(
          || "stepc_evalZ_enf",
          |lc| lc + tmpw.get_variable() + tmpx.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + eval_Z.get_variable(),
        );

        (sum, eval_Z)
      };

      // Compute core branch: sum = eval_A + r*eval_B + r^2*eval_C and eval_Z
      let (sum_core_comb, evalZ_core) = {
        let mut ns = cs.namespace(|| "branch_core_comb");

        // tmp1 = eval_B * r
        let tmp1 = AllocatedNum::alloc(ns.namespace(|| "corec_tmp1"), || {
          let b = eval_B_core
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          let rv = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok(b * rv)
        })?;
        ns.enforce(
          || "corec_tmp1_enf",
          |lc| lc + eval_B_core.get_variable(),
          |lc| lc + r.get_variable(),
          |lc| lc + tmp1.get_variable(),
        );

        // tmp2 = eval_C * r_sq
        let tmp2 = AllocatedNum::alloc(ns.namespace(|| "corec_tmp2"), || {
          let c = eval_C_core
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          let rsq = r_sq_var
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          Ok(c * rsq)
        })?;
        ns.enforce(
          || "corec_tmp2_enf",
          |lc| lc + eval_C_core.get_variable(),
          |lc| lc + r_sq_var.get_variable(),
          |lc| lc + tmp2.get_variable(),
        );

        // sum = eval_A + tmp1 + tmp2
        let sum = AllocatedNum::alloc(ns.namespace(|| "corec_sum"), || {
          let a = eval_A_core
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          let t1 = tmp1.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          let t2 = tmp2.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok(a + t1 + t2)
        })?;
        ns.enforce(
          || "corec_sum_enf",
          |lc| lc + eval_A_core.get_variable() + tmp1.get_variable() + tmp2.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + sum.get_variable(),
        );

        // eval_Z = (1 - r_y0)*eval_W + r_y0*eval_X
        let tmpw = AllocatedNum::alloc(ns.namespace(|| "corec_tmpw"), || {
          let ew = eval_W_core.get_value().unwrap_or(E::Scalar::ZERO);
          let om = one_minus_ry0.get_value().unwrap_or(E::Scalar::ZERO);
          Ok(ew * om)
        })?;
        ns.enforce(
          || "corec_tmpw_enf",
          |lc| lc + eval_W_core.get_variable(),
          |lc| lc + one_minus_ry0.get_variable(),
          |lc| lc + tmpw.get_variable(),
        );
        let tmpx = AllocatedNum::alloc(ns.namespace(|| "corec_tmpx"), || {
          let ex = eval_X_core.get_value().unwrap_or(E::Scalar::ZERO);
          let ry = r_y0.get_value().unwrap_or(E::Scalar::ZERO);
          Ok(ex * ry)
        })?;
        ns.enforce(
          || "corec_tmpx_enf",
          |lc| lc + eval_X_core.get_variable(),
          |lc| lc + r_y0.get_variable(),
          |lc| lc + tmpx.get_variable(),
        );
        let eval_Z = AllocatedNum::alloc(ns.namespace(|| "corec_evalZ"), || {
          let wv = tmpw.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          let xv = tmpx.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok(wv + xv)
        })?;
        ns.enforce(
          || "corec_evalZ_enf",
          |lc| lc + tmpw.get_variable() + tmpx.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + eval_Z.get_variable(),
        );

        (sum, eval_Z)
      };

      // term_step = sum_step * evalZ_step
      let term_step = AllocatedNum::alloc(cs.namespace(|| "term_step"), || {
        let s = sum_step_comb
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let z = evalZ_step
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        Ok(s * z)
      })?;
      cs.enforce(
        || "term_step_enf",
        |lc| lc + sum_step_comb.get_variable(),
        |lc| lc + evalZ_step.get_variable(),
        |lc| lc + term_step.get_variable(),
      );

      // term_core = sum_core * evalZ_core
      let term_core = AllocatedNum::alloc(cs.namespace(|| "term_core"), || {
        let s = sum_core_comb
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let z = evalZ_core
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        Ok(s * z)
      })?;
      cs.enforce(
        || "term_core_enf",
        |lc| lc + sum_core_comb.get_variable(),
        |lc| lc + evalZ_core.get_variable(),
        |lc| lc + term_core.get_variable(),
      );

      // prod_c_term_core = c_inner * term_core
      let prod_c_term_core = AllocatedNum::alloc(cs.namespace(|| "c_term_prod"), || {
        let c = c_inner.get_value().unwrap_or(E::Scalar::ZERO);
        let tc = term_core.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(c * tc)
      })?;
      cs.enforce(
        || "c_term_prod_enf",
        |lc| lc + c_inner.get_variable(),
        |lc| lc + term_core.get_variable(),
        |lc| lc + prod_c_term_core.get_variable(),
      );

      // rhs_comb = term_step + c_inner*term_core
      let rhs_comb = AllocatedNum::alloc(cs.namespace(|| "rhs_comb_inner"), || {
        let ts = term_step.get_value().unwrap_or(E::Scalar::ZERO);
        let add = prod_c_term_core.get_value().unwrap_or(E::Scalar::ZERO);
        Ok(ts + add)
      })?;
      cs.enforce(
        || "rhs_comb_inner_enf",
        |lc| lc + term_step.get_variable() + prod_c_term_core.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + rhs_comb.get_variable(),
      );

      // lhs_comb already prev_e_step + c_inner*prev_e_core
      let prod_c_prev_e_core_inner =
        AllocatedNum::alloc(cs.namespace(|| "c_prev_e_core2"), || {
          let c = c_inner.get_value().unwrap_or(E::Scalar::ZERO);
          let pec = prev_e_core.get_value().unwrap_or(E::Scalar::ZERO);
          Ok(c * pec)
        })?;
      cs.enforce(
        || "c_prev_e_core2_enf",
        |lc| lc + c_inner.get_variable(),
        |lc| lc + prev_e_core.get_variable(),
        |lc| lc + prod_c_prev_e_core_inner.get_variable(),
      );
      let lhs_comb_inner = AllocatedNum::alloc(cs.namespace(|| "lhs_comb_inner"), || {
        let pes = prev_e_step.get_value().unwrap_or(E::Scalar::ZERO);
        let add = prod_c_prev_e_core_inner
          .get_value()
          .unwrap_or(E::Scalar::ZERO);
        Ok(pes + add)
      })?;
      cs.enforce(
        || "lhs_comb_inner_enf",
        |lc| lc + prev_e_step.get_variable() + prod_c_prev_e_core_inner.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + lhs_comb_inner.get_variable(),
      );

      // Final equality
      cs.enforce(
        || "inner_combined_equality",
        |lc| lc + lhs_comb_inner.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + rhs_comb.get_variable(),
      );

      {
        let mut ns_step = cs.namespace(|| "inner_final_step");
        Self::build_inner_branch(
          &mut ns_step,
          &InnerBranchParams {
            eval_A: &eval_A_step,
            eval_B: &eval_B_step,
            eval_C: &eval_C_step,
            eval_W: eval_W_step,
            prev_e: prev_e_step,
            r,
            eval_X: &eval_X_step,
            one_minus_ry0: &one_minus_ry0,
            r_y0,
            label: "step",
          },
        )?;
      }
      {
        let mut ns_core = cs.namespace(|| "inner_final_core");
        Self::build_inner_branch(
          &mut ns_core,
          &InnerBranchParams {
            eval_A: &eval_A_core,
            eval_B: &eval_B_core,
            eval_C: &eval_C_core,
            eval_W: eval_W_core,
            prev_e: prev_e_core,
            r,
            eval_X: &eval_X_core,
            one_minus_ry0: &one_minus_ry0,
            r_y0,
            label: "core",
          },
        )?;
      }

      Ok((
        vec![prev_e_step.clone(), prev_e_core.clone()],
        vec![c_inner],
      ))
    } else {
      Err(SynthesisError::Unsatisfiable)
    }
  }

  fn num_rounds(&self) -> usize {
    self.idx_inner_final() + 1
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    bellpepper::{
      r1cs::{MultiRoundSpartanShape, MultiRoundSpartanWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
    },
    nifs::NIFS,
    provider::P256HyraxEngine,
    traits::{self, Engine, transcript::TranscriptEngineTrait},
  };
  use tracing_subscriber::EnvFilter;

  type E = crate::provider::P256HyraxEngine;

  #[test]
  fn test_full_spartan_verifier_nifs_fold() {
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
      tau_at_rx: <E as Engine>::Scalar::ZERO,
      inner_polys: vec![[<E as Engine>::Scalar::ZERO; 3]],
      eval_A: <E as Engine>::Scalar::ZERO,
      eval_B: <E as Engine>::Scalar::ZERO,
      eval_C: <E as Engine>::Scalar::ZERO,
      eval_W: <E as Engine>::Scalar::ZERO,
      eval_X: <E as Engine>::Scalar::ZERO,
    };

    let (shape_mr, ck, _vk) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&circuit).unwrap();
    assert_eq!(shape_mr.num_cons_unpadded, 26);
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

    let (running_U, running_W) = S_reg.sample_random_instance_witness(&ck).unwrap();

    let mut transcript_nifs = <E as Engine>::TE::new(b"nifs");
    let (proof, folded_W) = NIFS::<E>::prove(
      &ck,
      &S_reg,
      &running_U,
      &running_W,
      &inst_reg,
      &wit_reg,
      &mut transcript_nifs,
    )
    .unwrap();

    let mut transcript_verify = <E as Engine>::TE::new(b"nifs");
    let U2_reg = inst_split.to_regular_instance().unwrap();
    let verified_U = proof
      .verify(&mut transcript_verify, &running_U, &U2_reg)
      .unwrap();
    assert!(S_reg.is_sat_relaxed(&ck, &verified_U, &folded_W).is_ok());
  }

  #[test]
  fn test_full_neutron_verifier_nifs_fold() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();

    let mut circuit = NeutronNovaVerifierCircuit::<E> {
      outer_polys_step: vec![[<E as Engine>::Scalar::ZERO; 4]],
      outer_polys_core: vec![[<E as Engine>::Scalar::ZERO; 4]],

      claim_Az_step: <E as Engine>::Scalar::ZERO,
      claim_Bz_step: <E as Engine>::Scalar::ZERO,
      claim_Cz_step: <E as Engine>::Scalar::ZERO,

      claim_Az_core: <E as Engine>::Scalar::ZERO,
      claim_Bz_core: <E as Engine>::Scalar::ZERO,
      claim_Cz_core: <E as Engine>::Scalar::ZERO,

      tau_at_rx: <E as Engine>::Scalar::ZERO,

      inner_polys_step: vec![[<E as Engine>::Scalar::ZERO; 3]],
      inner_polys_core: vec![[<E as Engine>::Scalar::ZERO; 3]],

      eval_A_step: <E as Engine>::Scalar::ZERO,
      eval_B_step: <E as Engine>::Scalar::ZERO,
      eval_C_step: <E as Engine>::Scalar::ZERO,
      eval_W_step: <E as Engine>::Scalar::ZERO,

      eval_A_core: <E as Engine>::Scalar::ZERO,
      eval_B_core: <E as Engine>::Scalar::ZERO,
      eval_C_core: <E as Engine>::Scalar::ZERO,
      eval_W_core: <E as Engine>::Scalar::ZERO,

      eval_X_step: <E as Engine>::Scalar::ZERO,
      eval_X_core: <E as Engine>::Scalar::ZERO,

      t_out_step: <E as Engine>::Scalar::ZERO,
      nifs_polys: vec![[<E as Engine>::Scalar::ZERO; 4]],
      rho_acc_at_rb: <E as Engine>::Scalar::ONE,
    };

    let (shape_mr, ck, _vk) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&circuit).unwrap();

    assert!(shape_mr.num_cons_unpadded > 0);
    let mut state =
      <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::initialize_multiround_witness(
        &shape_mr, &ck, &circuit, false,
      )
      .unwrap();
    let mut transcript = <E as Engine>::TE::new(b"nifs");

    let num_rounds = circuit.num_rounds();
    // First, run NIFS challenge rounds to capture r_b values
    let nifs_rounds = circuit.nifs_polys.len();
    for i in 0..nifs_rounds {
      let chals = <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::process_round(
        &mut state,
        &shape_mr,
        &ck,
        &circuit,
        i,
        false,
        &mut transcript,
      )
      .unwrap();
      let _r_b = chals[0];
    }
    // Set rho_acc_at_rb for the NIFS final equality (trivial in this mock)
    circuit.rho_acc_at_rb = <P256HyraxEngine as traits::Engine>::Scalar::ONE;

    // Continue with remaining rounds starting at NIFS final
    for r in nifs_rounds..num_rounds {
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

    let (running_U, running_W) = S_reg.sample_random_instance_witness(&ck).unwrap();

    let mut transcript_nifs = <E as Engine>::TE::new(b"nifs");
    let (proof, folded_W) = NIFS::<E>::prove(
      &ck,
      &S_reg,
      &running_U,
      &running_W,
      &inst_reg,
      &wit_reg,
      &mut transcript_nifs,
    )
    .unwrap();

    let mut transcript_verify = <E as Engine>::TE::new(b"nifs");
    let U2_reg = inst_split.to_regular_instance().unwrap();
    let verified_U = proof
      .verify(&mut transcript_verify, &running_U, &U2_reg)
      .unwrap();
    assert!(S_reg.is_sat_relaxed(&ck, &verified_U, &folded_W).is_ok());
  }
}
