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
use ff::Field;

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

/// Helper function to allocate polynomial coefficients in R1CS constraints.
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

/// Enforces the final check of the outer sum-check of Spartan:
/// prev_claim = tau_at_rx * (claim_Az*claim_Bz - claim_Cz)
fn enforce_outer_sc_final_check<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  mut cs: CS,
  claim_Az: &AllocatedNum<E::Scalar>,
  claim_Bz: &AllocatedNum<E::Scalar>,
  claim_Cz: &AllocatedNum<E::Scalar>,
  tau_at_rx: &AllocatedNum<E::Scalar>,
  prev_claim: &AllocatedNum<E::Scalar>,
) -> Result<(), SynthesisError> {
  // prod_Az_Bz = claim_Az * claim_Bz
  let prod_Az_Bz = claim_Az.mul(cs.namespace(|| "AzBz"), claim_Bz)?;

  // prev_claim = tau_at_rx * (prod_AzBz - Cz)
  cs.enforce(
    || "prev_claim = tau_at_rx*(prod_AzBz - Cz)",
    |lc| lc + tau_at_rx.get_variable(),
    |lc| lc + prod_Az_Bz.get_variable() - claim_Cz.get_variable(),
    |lc| lc + prev_claim.get_variable(),
  );

  Ok(())
}

/// Computes joint_claim = Az + r*Bz + r^2*Cz and returns joint_claim
fn compute_joint_claim<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  mut cs: CS,
  Az: &AllocatedNum<E::Scalar>,
  Bz: &AllocatedNum<E::Scalar>,
  Cz: &AllocatedNum<E::Scalar>,
  r: &AllocatedNum<E::Scalar>,
  r_sq: &AllocatedNum<E::Scalar>,
) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
  // r_times_Bz = Bz * r
  let r_times_Bz = r.mul(cs.namespace(|| "r_times_Bz"), Bz)?;

  // joint = Az + r_times_claim_Bz + r_sq * claim_Cz
  // We instead check: joint - Az - r_times_claim_Bz = r_sq * claim_Cz
  let joint = AllocatedNum::alloc(cs.namespace(|| "joint"), || {
    let a = Az.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    let t1 = r_times_Bz
      .get_value()
      .ok_or(SynthesisError::AssignmentMissing)?;
    let rsq = r_sq.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    let c = Cz.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    Ok(a + t1 + rsq * c)
  })?;
  cs.enforce(
    || "joint_enf",
    |lc| lc + Cz.get_variable(),
    |lc| lc + r_sq.get_variable(),
    |lc| lc + joint.get_variable() - Az.get_variable() - r_times_Bz.get_variable(),
  );
  Ok(joint)
}

/// Enforces the final check of the inner sum-check of Spartan:
/// eval_z \gets (1-r_y0)*eval_W + r_y0*eval_X
/// quotient = prev_claim / eval_z
/// inputize(quotient)
fn enforce_inner_sc_final_check<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  mut cs: CS,
  r_y0: &AllocatedNum<E::Scalar>,
  eval_W: &AllocatedNum<E::Scalar>,
  eval_X: &AllocatedNum<E::Scalar>,
  prev_claim: &AllocatedNum<E::Scalar>,
) -> Result<(), SynthesisError> {
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

  // prev_e = (eval_A + r * eval_B + r^2 * eval_C) * sum_z_expected
  // we compute prev_e / sum_z_expected and inputize that
  // The verifier can compute eval_A + r * eval_B + r^2 * eval_C and check equality
  let quotient = AllocatedNum::alloc_input(cs.namespace(|| "quotient"), || {
    let prev_claim_v = prev_claim
      .get_value()
      .ok_or(SynthesisError::AssignmentMissing)?;
    let se_v = sum_z_expected
      .get_value()
      .ok_or(SynthesisError::AssignmentMissing)?;
    Ok(if se_v.is_zero().into() {
      E::Scalar::ZERO
    } else {
      prev_claim_v * se_v.invert().unwrap()
    })
  })?;

  // check that quotient * sum_z_expected = prev_e
  cs.enforce(
    || "check_quotient",
    |lc| lc + quotient.get_variable(),
    |lc| lc + sum_z_expected.get_variable(),
    |lc| lc + prev_claim.get_variable(),
  );

  Ok(())
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
    // Routing
    if round_index < self.outer_rounds() {
      // Outer cubic sum-check per-round consistency
      let coeffs = alloc_coeffs::<E, _>(
        cs.namespace(|| format!("outer_sc_coeffs_round_{round_index}")),
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
      // Final outer equality: prev_claim == tau_at_rx * (claim_Az*claim_Bz - claim_Cz)
      let prev_claim = &prior_round_vars.last().expect("has previous round")[0];

      let claim_Az = AllocatedNum::alloc(cs.namespace(|| "Az_outer"), || Ok(self.claim_Az))?;
      let claim_Bz = AllocatedNum::alloc(cs.namespace(|| "Bz_outer"), || Ok(self.claim_Bz))?;
      let claim_Cz = AllocatedNum::alloc(cs.namespace(|| "Cz_outer"), || Ok(self.claim_Cz))?;
      let tau_at_rx =
        AllocatedNum::alloc(cs.namespace(|| "tau_at_rx_outer"), || Ok(self.tau_at_rx))?;

      enforce_outer_sc_final_check::<E, _>(
        cs.namespace(|| "enforce_outer_final_check"),
        &claim_Az,
        &claim_Bz,
        &claim_Cz,
        &tau_at_rx,
        prev_claim,
      )?;

      Ok((
        vec![
          prev_claim.clone(),
          claim_Az.clone(),
          claim_Bz.clone(),
          claim_Cz.clone(),
          tau_at_rx.clone(),
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
      let r_sq = r.square(cs.namespace(|| "r_sq"))?;

      let claim_inner_joint = compute_joint_claim::<E, _>(
        cs.namespace(|| "compute_inner_joint_claim"),
        claim_Az,
        claim_Bz,
        claim_Cz,
        &r,
        &r_sq,
      )?;

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
      let eval_W = &prior_round_vars[self.idx_inner_commit_w()][0];

      let tau_at_rx = prior_round_vars[self.idx_outer_final()][4].clone();
      tau_at_rx.inputize(cs.namespace(|| "inputize_tau_at_rx"))?;

      let eval_X = AllocatedNum::alloc_input(cs.namespace(|| "eval_X"), || Ok(self.eval_X))?;
      let r_y0 = &prev_challenges[self.idx_inner_start()][0];

      enforce_inner_sc_final_check::<E, _>(
        cs.namespace(|| "enforce_inner_final_check"),
        r_y0,
        eval_W,
        &eval_X,
        prev_e,
      )?;

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
  // NeutronNova folding scheme verifier state across multiple rounds
  // NIFS cubic sum-check polynomials (4 coeffs per round)
  pub(crate) nifs_polys: Vec<[E::Scalar; 4]>,
  // Accumulated equality value acc_eq = \prod_t eq(r_b_t; rho_t), provided as a public input
  pub(crate) rho_acc_at_rb: E::Scalar,
  pub(crate) t_out_step: E::Scalar,

  // Spartan sum-check proof for step and core circuits
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

  pub(crate) eval_W_step: E::Scalar,
  pub(crate) eval_W_core: E::Scalar,

  pub(crate) eval_X_step: E::Scalar,
  pub(crate) eval_X_core: E::Scalar,
}

impl<E: Engine> NeutronNovaVerifierCircuit<E> {
  /// Creates a default instance of the NeutronNova verifier circuit with zeroed fields.
  pub fn default(num_rounds_z: usize, num_rounds_x: usize, num_rounds_y: usize) -> Self {
    Self {
      outer_polys_step: vec![[E::Scalar::ZERO; 4]; num_rounds_x],
      outer_polys_core: vec![[E::Scalar::ZERO; 4]; num_rounds_x],
      claim_Az_step: E::Scalar::ZERO,
      claim_Bz_step: E::Scalar::ZERO,
      claim_Cz_step: E::Scalar::ZERO,
      claim_Az_core: E::Scalar::ZERO,
      claim_Bz_core: E::Scalar::ZERO,
      claim_Cz_core: E::Scalar::ZERO,
      tau_at_rx: E::Scalar::ZERO,
      inner_polys_step: vec![[E::Scalar::ZERO; 3]; num_rounds_y],
      inner_polys_core: vec![[E::Scalar::ZERO; 3]; num_rounds_y],
      eval_W_step: E::Scalar::ZERO,
      eval_W_core: E::Scalar::ZERO,
      eval_X_step: E::Scalar::ZERO,
      eval_X_core: E::Scalar::ZERO,
      t_out_step: E::Scalar::ZERO,
      nifs_polys: vec![[E::Scalar::ZERO; 4]; num_rounds_z],
      rho_acc_at_rb: E::Scalar::ZERO,
    }
  }

  // Number of NIFS rounds
  fn nifs_rounds(&self) -> usize {
    self.nifs_polys.len()
  }
  // Number of outer sum-check rounds (same for step and core)
  fn outer_rounds(&self) -> usize {
    self.outer_polys_step.len() // same as outer_polys_core.len()
  }
  // Number of inner sum-check rounds (same for step and core)
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

impl<E: Engine> MultiRoundCircuit<E> for NeutronNovaVerifierCircuit<E> {
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
      return Ok(0);
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
      return Ok(0);
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
      let coeffs = alloc_coeffs::<E, _>(
        cs.namespace(|| format!("nifs_round_{round_index}")),
        &self.nifs_polys[round_index],
      )?;

      // Enforce g_i(0)+g_i(1)=prev_claim directly
      // When round index is zero, previous claim is zero
      cs.enforce(
        || format!("nifs_init_claim_zero {round_index}"),
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
            let prev_p_rb = &prior_round_vars[round_index - 1][0];
            lc + prev_p_rb.get_variable()
          }
        }, // 0
      );

      let r_b = AllocatedNum::alloc_input(cs.namespace(|| format!("r_b_{round_index}")), || {
        Ok(challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO))
      })?;
      let p_rb = eval_poly_horner::<E, _>(
        cs.namespace(|| format!("poly_eval_{round_index}")),
        &coeffs,
        &r_b,
      )?;

      // Expose p_rb only; next round consumes it as previous claim
      Ok((vec![p_rb], vec![r_b]))
    } else if round_index == self.idx_nifs_final() {
      // Final NIFS equality (mirrors prover): p_last(r_b_last) == rho_acc_at_rb * t_out_step
      // The previous NIFS round exposes [p_rb, carry]; bind p_rb to the equality.
      let prev_vars = &prior_round_vars[self.idx_nifs_final() - 1];
      let p_rb_last = &prev_vars[0];

      let t_out_step = AllocatedNum::alloc(cs.namespace(|| "t_out_step"), || Ok(self.t_out_step))?;
      let rho_acc = AllocatedNum::alloc(cs.namespace(|| "rho_acc_at_rb_nifs"), || {
        Ok(self.rho_acc_at_rb)
      })?;

      // rho_acc * t_out_step = p_rb_last
      cs.enforce(
        || "nifs_final_eq",
        |lc| lc + rho_acc.get_variable(),
        |lc| lc + t_out_step.get_variable(),
        |lc| lc + p_rb_last.get_variable(),
      );

      // Expose rho_acc so later rounds (inner-final) can inputize it as a public value without re-allocation
      Ok((
        vec![p_rb_last.clone(), rho_acc.clone(), t_out_step.clone()],
        vec![],
      ))
    } else if round_index >= self.idx_outer_start() && round_index < self.idx_outer_final() {
      let i = round_index - self.idx_outer_start();

      let coeffs_step = alloc_coeffs::<E, _>(
        cs.namespace(|| format!("outer_step_{i}")),
        &self.outer_polys_step[i],
      )?;
      let coeffs_core = alloc_coeffs::<E, _>(
        cs.namespace(|| format!("outer_core_{i}")),
        &self.outer_polys_core[i],
      )?;

      let r = AllocatedNum::alloc_input(cs.namespace(|| format!("r_x_{i}")), || {
        Ok(challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO))
      })?;

      // Consistency with previous round
      // The step branch starts from the folded instance target T_out, whereas the core branch
      // starts from 0
      // Enforce sum(coeffs_step) == if i ==0 {t_out_step} else {prev_e_step}
      cs.enforce(
        || format!("outer_init_step zero_{i}"),
        |lc| {
          lc + coeffs_step[0].get_variable()
            + coeffs_step[0].get_variable()
            + coeffs_step[1].get_variable()
            + coeffs_step[2].get_variable()
            + coeffs_step[3].get_variable()
        },
        |lc| lc + CS::one(),
        |lc| {
          lc + if i == 0 {
            let t_out_step = prior_round_vars[self.idx_nifs_final()][2].clone();
            t_out_step.get_variable()
          } else {
            let prev_e_step = &prior_round_vars[self.idx_outer_start() + i - 1][0];
            prev_e_step.get_variable()
          }
        },
      );

      // Core branch still starts from zero: sum(coeffs_core) == 0
      cs.enforce(
        || format!("outer_init_zero_core_{i}"),
        |lc| {
          lc + coeffs_core[0].get_variable()
            + coeffs_core[0].get_variable()
            + coeffs_core[1].get_variable()
            + coeffs_core[2].get_variable()
            + coeffs_core[3].get_variable()
        },
        |lc| lc + CS::one(),
        |lc| {
          if i == 0 {
            lc
          } else {
            let prev_e_core = &prior_round_vars[self.idx_outer_start() + i - 1][1];
            lc + prev_e_core.get_variable()
          }
        }, // constant 0
      );

      // Evaluate polynomials at r
      let p_r_step = eval_poly_horner::<E, _>(
        cs.namespace(|| format!("poly_eval_step_{i}")),
        &coeffs_step,
        &r,
      )?;
      let p_r_core = eval_poly_horner::<E, _>(
        cs.namespace(|| format!("poly_eval_core_{i}")),
        &coeffs_core,
        &r,
      )?;

      Ok((vec![p_r_step, p_r_core], vec![r]))
    } else if round_index == self.idx_outer_final() {
      // Outer final equality round
      // Previous variables come from the previous OUTER round explicitly
      let prev_claim_step = &prior_round_vars[self.idx_outer_final() - 1][0];
      let prev_claim_core = &prior_round_vars[self.idx_outer_final() - 1][1];

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

      let tau_at_rx =
        AllocatedNum::alloc(cs.namespace(|| "tau_at_rx_outer"), || Ok(self.tau_at_rx))?;

      enforce_outer_sc_final_check::<E, _>(
        cs.namespace(|| "enforce_outer_final_check"),
        &claim_Az_step,
        &claim_Bz_step,
        &claim_Cz_step,
        &tau_at_rx,
        prev_claim_step,
      )?;

      enforce_outer_sc_final_check::<E, _>(
        cs.namespace(|| "enforce_outer_final_check_core"),
        &claim_Az_core,
        &claim_Bz_core,
        &claim_Cz_core,
        &tau_at_rx,
        prev_claim_core,
      )?;

      Ok((
        vec![
          prev_claim_step.clone(),
          prev_claim_core.clone(),
          claim_Az_step,
          claim_Bz_step,
          claim_Cz_step,
          claim_Az_core,
          claim_Bz_core,
          claim_Cz_core,
          tau_at_rx.clone(),
        ],
        vec![],
      ))
    } else if round_index == self.idx_inner_setup() {
      // Inner setup round
      let r = AllocatedNum::alloc_input(cs.namespace(|| "r"), || {
        Ok(challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO))
      })?;
      let r_sq = r.square(cs.namespace(|| "r_sq"))?;

      // Fetch Az, Bz, Cz values from outer-final outputs
      let outer_vars = &prior_round_vars[self.idx_outer_final()];
      let Az_step = &outer_vars[2];
      let Bz_step = &outer_vars[3];
      let Cz_step = &outer_vars[4];
      let Az_core = &outer_vars[5];
      let Bz_core = &outer_vars[6];
      let Cz_core = &outer_vars[7];

      // Compute for step and core
      let joint_step = compute_joint_claim::<E, _>(
        cs.namespace(|| "step"),
        Az_step,
        Bz_step,
        Cz_step,
        &r,
        &r_sq,
      )?;
      let joint_core = compute_joint_claim::<E, _>(
        cs.namespace(|| "core"),
        Az_core,
        Bz_core,
        Cz_core,
        &r,
        &r_sq,
      )?;

      Ok((vec![joint_step, joint_core], vec![r]))
    } else if round_index >= self.idx_inner_start() && round_index < self.idx_inner_commit_w_step()
    {
      // Inner quadratic sum-check rounds
      let idx = round_index - self.idx_inner_start();

      let coeffs_step = alloc_coeffs::<E, _>(
        cs.namespace(|| format!("inner_step_{idx}")),
        &self.inner_polys_step[idx],
      )?;
      let coeffs_core = alloc_coeffs::<E, _>(
        cs.namespace(|| format!("inner_core_{idx}")),
        &self.inner_polys_core[idx],
      )?;

      let r = AllocatedNum::alloc_input(cs.namespace(|| format!("r_y[{idx}]")), || {
        Ok(challenges.map(|c| c[0]).unwrap_or(E::Scalar::ZERO))
      })?;
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

      // sum(coeffs_step) == prev_e_step
      cs.enforce(
        || format!("inner_consistency_step_{idx}"),
        |lc| {
          lc + coeffs_step[0].get_variable()
            + coeffs_step[0].get_variable()
            + coeffs_step[1].get_variable()
            + coeffs_step[2].get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc + prev_e_step.get_variable(),
      );

      // sum(coeffs_core) == prev_e_core
      cs.enforce(
        || format!("inner_consistency_core_{idx}"),
        |lc| {
          lc + coeffs_core[0].get_variable()
            + coeffs_core[0].get_variable()
            + coeffs_core[1].get_variable()
            + coeffs_core[2].get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc + prev_e_core.get_variable(),
      );

      // Evaluate polynomials at r
      let p_r_step = eval_poly_horner::<E, _>(
        cs.namespace(|| format!("inner_poly_eval_step_idx_{idx}")),
        &coeffs_step,
        &r,
      )?;
      let p_r_core = eval_poly_horner::<E, _>(
        cs.namespace(|| format!("inner_poly_eval_core_idx_{idx}")),
        &coeffs_core,
        &r,
      )?;

      Ok((vec![p_r_step, p_r_core], vec![r]))
    } else if round_index == self.idx_inner_commit_w_step() {
      // Commit round for eval_W_step
      let eval_W_step =
        AllocatedNum::alloc(cs.namespace(|| "eval_W_step"), || Ok(self.eval_W_step))?;

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

      let r_y0 = &prev_challenges[self.idx_inner_start()][0];

      // tau_at_rx is a public input
      let tau_at_rx = prior_round_vars[self.idx_outer_final()][8].clone();
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

      enforce_inner_sc_final_check::<E, _>(
        cs.namespace(|| "enforce_inner_final_check_step"),
        r_y0,
        eval_W_step,
        &eval_X_step,
        prev_e_step,
      )?;

      enforce_inner_sc_final_check::<E, _>(
        cs.namespace(|| "enforce_inner_final_check_core"),
        r_y0,
        eval_W_core,
        &eval_X_core,
        prev_e_core,
      )?;

      Ok((vec![prev_e_step.clone(), prev_e_core.clone()], vec![]))
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
      eval_W: <E as Engine>::Scalar::ZERO,
      eval_X: <E as Engine>::Scalar::ZERO,
    };

    let (shape_mr, ck, _vk) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&circuit).unwrap();
    assert_eq!(shape_mr.num_cons_unpadded, 19);
    let mut state = SatisfyingAssignment::<E>::initialize_multiround_witness(&shape_mr).unwrap();
    let mut transcript = <E as Engine>::TE::new(b"nifs");

    let num_rounds = circuit.num_rounds();
    for r in 0..num_rounds {
      SatisfyingAssignment::<E>::process_round(
        &mut state,
        &shape_mr,
        &ck,
        &circuit,
        r,
        &mut transcript,
      )
      .unwrap();
    }

    let (inst_split, wit_reg) =
      SatisfyingAssignment::<E>::finalize_multiround_witness(&mut state, &shape_mr).unwrap();

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
      eval_W_step: <E as Engine>::Scalar::ZERO,
      eval_W_core: <E as Engine>::Scalar::ZERO,

      eval_X_step: <E as Engine>::Scalar::ZERO,
      eval_X_core: <E as Engine>::Scalar::ZERO,

      t_out_step: <E as Engine>::Scalar::ZERO,
      nifs_polys: vec![[<E as Engine>::Scalar::ZERO; 4]],
      rho_acc_at_rb: <E as Engine>::Scalar::ONE,
    };

    let (shape_mr, ck, _vk) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&circuit).unwrap();

    assert_eq!(shape_mr.num_cons_unpadded, 44);
    let mut state = SatisfyingAssignment::<E>::initialize_multiround_witness(&shape_mr).unwrap();
    let mut transcript = <E as Engine>::TE::new(b"nifs");

    let num_rounds = circuit.num_rounds();
    // First, run NIFS challenge rounds to capture r_b values
    let nifs_rounds = circuit.nifs_polys.len();
    for i in 0..nifs_rounds {
      let chals = SatisfyingAssignment::<E>::process_round(
        &mut state,
        &shape_mr,
        &ck,
        &circuit,
        i,
        &mut transcript,
      )
      .unwrap();
      let _r_b = chals[0];
    }
    // Set rho_acc_at_rb for the NIFS final equality (trivial in this mock)
    circuit.rho_acc_at_rb = <P256HyraxEngine as traits::Engine>::Scalar::ONE;

    // Continue with remaining rounds starting at NIFS final
    for r in nifs_rounds..num_rounds {
      SatisfyingAssignment::<E>::process_round(
        &mut state,
        &shape_mr,
        &ck,
        &circuit,
        r,
        &mut transcript,
      )
      .unwrap();
    }

    let (inst_split, wit_reg) =
      SatisfyingAssignment::<E>::finalize_multiround_witness(&mut state, &shape_mr).unwrap();

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
