//! This module implements NeutronNova's folding scheme for folding together a batch of R1CS instances
//! This implementation focuses on a non-recursive version of NeutronNova and tagets the case where the batch size is moderately large.
//! Since we are in the non-recursive setting, we simply fold a batch of instances into one (all at once, via multi-folding)
//! and then use spartan to prove that folded instance.

//! This module implements a non-interactive folding scheme from NeutronNova
#![allow(non_snake_case)]
use crate::{
  errors::SpartanError,
  math::Math,
  polys::{eq::EqPolynomial, univariate::UniPoly},
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness},
  traits::{Engine, pcs::FoldingEngineTrait, transcript::TranscriptEngineTrait},
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Holds the proof produced by the NeutronNova folding scheme followed by Spartan SNARK
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronSNARK<E: Engine> {
  poly: UniPoly<E::Scalar>,
}

impl<E: Engine> NeutronSNARK<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Computes the evaluations of the sum-check polynomial at 0, 2, 3, and 4
  #[inline]
  #[allow(clippy::too_many_arguments)]
  fn prove_helper(
    rho: &E::Scalar,
    (left, right): (usize, usize),
    e1: &[E::Scalar],
    Az1: &[E::Scalar],
    Bz1: &[E::Scalar],
    Cz1: &[E::Scalar],
    e2: &[E::Scalar],
    Az2: &[E::Scalar],
    Bz2: &[E::Scalar],
    Cz2: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) {
    // sanity check sizes
    assert_eq!(e1.len(), left + right);
    assert_eq!(Az1.len(), left * right);
    assert_eq!(Bz1.len(), left * right);
    assert_eq!(Cz1.len(), left * right);
    assert_eq!(e2.len(), left + right);
    assert_eq!(Az2.len(), left * right);
    assert_eq!(Bz2.len(), left * right);
    assert_eq!(Cz2.len(), left * right);

    let comb_func = |c1: &E::Scalar, c2: &E::Scalar, c3: &E::Scalar, c4: &E::Scalar| -> E::Scalar {
      *c1 * (*c2 * *c3 - *c4)
    };
    let (eval_at_0, eval_at_2, eval_at_3, eval_at_4, eval_at_5) = (0..right)
      .into_par_iter()
      .map(|i| {
        let (i_eval_at_0, i_eval_at_2, i_eval_at_3, i_eval_at_4, i_eval_at_5) = (0..left)
          .into_par_iter()
          .map(|j| {
            // Turn the two dimensional (i, j) into a single dimension index
            let k = i * left + j;

            // eval 0: bound_func is A(low)
            let eval_point_0 = comb_func(&e1[j], &Az1[k], &Bz1[k], &Cz1[k]);

            // eval 2: bound_func is -A(low) + 2*A(high)
            let poly_e_bound_point = e2[j] + e2[j] - e1[j];
            let poly_Az_bound_point = Az2[k] + Az2[k] - Az1[k];
            let poly_Bz_bound_point = Bz2[k] + Bz2[k] - Bz1[k];
            let poly_Cz_bound_point = Cz2[k] + Cz2[k] - Cz1[k];
            let eval_point_2 = comb_func(
              &poly_e_bound_point,
              &poly_Az_bound_point,
              &poly_Bz_bound_point,
              &poly_Cz_bound_point,
            );

            // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
            let poly_e_bound_point = poly_e_bound_point + e2[j] - e1[j];
            let poly_Az_bound_point = poly_Az_bound_point + Az2[k] - Az1[k];
            let poly_Bz_bound_point = poly_Bz_bound_point + Bz2[k] - Bz1[k];
            let poly_Cz_bound_point = poly_Cz_bound_point + Cz2[k] - Cz1[k];
            let eval_point_3 = comb_func(
              &poly_e_bound_point,
              &poly_Az_bound_point,
              &poly_Bz_bound_point,
              &poly_Cz_bound_point,
            );

            // eval 4: bound_func is -3A(low) + 4A(high); computed incrementally with bound_func applied to eval(3)
            let poly_e_bound_point = poly_e_bound_point + e2[j] - e1[j];
            let poly_Az_bound_point = poly_Az_bound_point + Az2[k] - Az1[k];
            let poly_Bz_bound_point = poly_Bz_bound_point + Bz2[k] - Bz1[k];
            let poly_Cz_bound_point = poly_Cz_bound_point + Cz2[k] - Cz1[k];
            let eval_point_4 = comb_func(
              &poly_e_bound_point,
              &poly_Az_bound_point,
              &poly_Bz_bound_point,
              &poly_Cz_bound_point,
            );

            // eval 5: bound_func is -4A(low) + 5A(high); computed incrementally with bound_func applied to eval(4)
            let poly_e_bound_point = poly_e_bound_point + e2[j] - e1[j];
            let poly_Az_bound_point = poly_Az_bound_point + Az2[k] - Az1[k];
            let poly_Bz_bound_point = poly_Bz_bound_point + Bz2[k] - Bz1[k];
            let poly_Cz_bound_point = poly_Cz_bound_point + Cz2[k] - Cz1[k];
            let eval_point_5 = comb_func(
              &poly_e_bound_point,
              &poly_Az_bound_point,
              &poly_Bz_bound_point,
              &poly_Cz_bound_point,
            );

            (
              eval_point_0,
              eval_point_2,
              eval_point_3,
              eval_point_4,
              eval_point_5,
            )
          })
          .reduce(
            || {
              (
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
              )
            },
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
          );

        let f1 = &e1[left..];
        let f2 = &e2[left..];

        // eval 0: bound_func is A(low)
        let eval_at_0 = f1[i] * i_eval_at_0;

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_f_bound_point = f2[i] + f2[i] - f1[i];
        let eval_at_2 = poly_f_bound_point * i_eval_at_2;

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let poly_f_bound_point = poly_f_bound_point + f2[i] - f1[i];
        let eval_at_3 = poly_f_bound_point * i_eval_at_3;

        // eval 4: bound_func is -3A(low) + 4A(high); computed incrementally with bound_func applied to eval(3)
        let poly_f_bound_point = poly_f_bound_point + f2[i] - f1[i];
        let eval_at_4 = poly_f_bound_point * i_eval_at_4;

        // eval 5: bound_func is -4A(low) + 5A(high); computed incrementally with bound_func applied to eval(4)
        let poly_f_bound_point = poly_f_bound_point + f2[i] - f1[i];
        let eval_at_5 = poly_f_bound_point * i_eval_at_5;

        (eval_at_0, eval_at_2, eval_at_3, eval_at_4, eval_at_5)
      })
      .reduce(
        || {
          (
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
          )
        },
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
      );

    // multiply by the common factors
    let one_minus_rho = E::Scalar::ONE - rho;
    let three_rho_minus_one = E::Scalar::from(3) * rho - E::Scalar::ONE;
    let five_rho_minus_two = E::Scalar::from(5) * rho - E::Scalar::from(2);
    let seven_rho_minus_three = E::Scalar::from(7) * rho - E::Scalar::from(3);
    let nine_rho_minus_four = E::Scalar::from(9) * rho - E::Scalar::from(4);

    (
      eval_at_0 * one_minus_rho,
      eval_at_2 * three_rho_minus_one,
      eval_at_3 * five_rho_minus_two,
      eval_at_4 * seven_rho_minus_three,
      eval_at_5 * nine_rho_minus_four,
    )
  }

  /// Prove the folding of a batch of R1CS instances
  pub fn prove(
    pp_digest: &E::Scalar,
    S: &R1CSShape<E>,
    instances: &[R1CSInstance<E>],
    witnesses: &[R1CSWitness<E>],
    transcript: &mut E::TE,
  ) -> Result<Self, SpartanError> {
    let ell = S.num_cons.next_power_of_two().log_2();
    // we split ell into ell1 and ell2 such that ell1 + ell2 = ell and ell1 >= ell2
    let ell1 = ell.div_ceil(2); // This ensures ell1 >= ell2
    let ell2 = ell / 2;
    let left = 1 << ell1;
    let right = 1 << ell2;

    let U1 = &instances[0];
    let W1 = &witnesses[0];
    let U2 = &instances[1];
    let W2 = &witnesses[1];

    // append the digest of pp to the transcript
    transcript.absorb(b"pp_digest", pp_digest);

    // append U1 and U2 to transcript
    transcript.absorb(b"U1", U1);
    transcript.absorb(b"U2", U2);

    // generate a challenge for the eq polynomial
    let tau = (0..ell)
      .map(|_| transcript.squeeze(b"tau"))
      .collect::<Result<Vec<_>, _>>()?;

    let E1 = EqPolynomial::new(tau[..ell1].to_vec()).evals();
    let E2 = EqPolynomial::new(tau[ell1..].to_vec()).evals();
    let E = [E1, E2].concat();

    let rho = transcript.squeeze(b"rho")?;

    let T = E::Scalar::ZERO; // we need all instances to be satisfying, so T is zero

    let (res1, res2) = rayon::join(
      || {
        let z1 = [W1.W.clone(), vec![E::Scalar::ONE], U1.X.clone()].concat();
        S.multiply_vec(&z1)
      },
      || {
        let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
        S.multiply_vec(&z2)
      },
    );

    let (Az1, Bz1, Cz1) = res1?;
    let (Az2, Bz2, Cz2) = res2?;

    // compute the sum-check polynomial's evaluations at 0, 2, 3
    let (eval_point_0, eval_point_2, eval_point_3, eval_point_4, eval_point_5) = Self::prove_helper(
      &rho,
      (left, right),
      &E,
      &Az1,
      &Bz1,
      &Cz1,
      &E,
      &Az2,
      &Bz2,
      &Cz2,
    );

    let evals = vec![
      eval_point_0,
      T - eval_point_0,
      eval_point_2,
      eval_point_3,
      eval_point_4,
      eval_point_5,
    ];
    let poly = UniPoly::<E::Scalar>::from_evals(&evals)?;

    // absorb poly in the RO
    transcript.absorb(b"poly", &poly);

    // squeeze a challenge
    let r_b = transcript.squeeze(b"r_b")?;

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let _T_out = poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap(); // TODO: remove unwrap

    let _U = U1.fold(U2, &r_b)?;
    let _W = W1.fold(W2, &r_b)?;

    // TODO prove U using W as witness and T_out as target

    Ok(Self { poly })
  }

  /*pub fn verify(
    &self,
    vk: &VerifierKey<E>,
    instances: &[R1CSInstance<E>],
    transcript: &mut E::TE,
  ) -> Result<(), SpartanError> {
    OK(())
  }*/
}
