//! This module implements NeutronNova's folding scheme for folding together a batch of R1CS instances
//! This implementation focuses on a non-recursive version of NeutronNova and targets the case where the batch size is moderately large.
//! Since we are in the non-recursive setting, we simply fold a batch of instances into one (all at once, via multi-folding)
//! and then use spartan to prove that folded instance.
#![allow(non_snake_case)]
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
    power::PowPolynomial,
    univariate::UniPoly,
  },
  r1cs::{R1CSInstance, R1CSWitness, SparseMatrix, SplitR1CSInstance, SplitR1CSShape},
  spartan::compute_eval_table_sparse,
  start_span,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::SpartanCircuit,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    snark::{DigestHelperTrait, SpartanDigest},
    transcript::TranscriptEngineTrait,
  },
};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info, info_span};

fn compute_tensor_decomp(n: usize) -> (usize, usize, usize) {
  let ell = n.next_power_of_two().log_2();
  // we split ell into ell1 and ell2 such that ell1 + ell2 = ell and ell1 >= ell2
  let ell1 = ell.div_ceil(2); // This ensures ell1 >= ell2
  let ell2 = ell / 2;
  let left = 1 << ell1;
  let right = 1 << ell2;

  (ell, left, right)
}

/// A type that holds the folded instance produced by NeutronNova NIFS
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedR1CSInstance<E: Engine> {
  U: R1CSInstance<E>,
  tau: E::Scalar, // the challenge for the equality polynomial
  T: E::Scalar,   // the target value for the folded instance
}

/// A type that holds the NeutronNova NIFS (Non-Interactive Folding Scheme)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaNIFS<E: Engine> {
  polys: Vec<UniPoly<E::Scalar>>,
}

#[inline]
#[allow(clippy::needless_range_loop)]
fn suffix_weight_full<F: Field>(t: usize, ell_b: usize, pair_idx: usize, rhos: &[F]) -> F {
  let mut w = F::ONE;
  let mut k = pair_idx;
  for s in (t + 1)..ell_b {
    let bit = (k & 1) as u8; // LSB-first
    w *= if bit == 0 { F::ONE - rhos[s] } else { rhos[s] };
    k >>= 1;
  }
  w
}

impl<E: Engine> NeutronNovaNIFS<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Computes the evaluations of the sum-check polynomial at 0, 2, and 3
  #[inline]
  #[allow(clippy::too_many_arguments)]
  fn prove_helper(
    rho: &E::Scalar,
    (left, right): (usize, usize),
    e: &[E::Scalar],
    Az1: &[E::Scalar],
    Bz1: &[E::Scalar],
    Cz1: &[E::Scalar],
    Az2: &[E::Scalar],
    Bz2: &[E::Scalar],
    Cz2: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar, E::Scalar) {
    // sanity check sizes
    assert_eq!(e.len(), left + right);
    assert_eq!(Az1.len(), left * right);
    assert_eq!(Bz1.len(), left * right);
    assert_eq!(Cz1.len(), left * right);
    assert_eq!(Az2.len(), left * right);
    assert_eq!(Bz2.len(), left * right);
    assert_eq!(Cz2.len(), left * right);

    let comb_func = |c1: &E::Scalar, c2: &E::Scalar, c3: &E::Scalar, c4: &E::Scalar| -> E::Scalar {
      *c1 * (*c2 * *c3 - *c4)
    };
    let (eval_at_0, eval_at_2, eval_at_3) = (0..right)
      .into_par_iter()
      .map(|i| {
        let (i_eval_at_0, i_eval_at_2, i_eval_at_3) = (0..left)
          .into_par_iter()
          .map(|j| {
            // Turn the two dimensional (i, j) into a single dimension index
            let k = i * left + j;
            let poly_e_bound_point = e[j];

            // eval 0: bound_func is A(low)
            let eval_point_0 = comb_func(&poly_e_bound_point, &Az1[k], &Bz1[k], &Cz1[k]);

            // eval 2: bound_func is -A(low) + 2*A(high)
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
            let poly_Az_bound_point = poly_Az_bound_point + Az2[k] - Az1[k];
            let poly_Bz_bound_point = poly_Bz_bound_point + Bz2[k] - Bz1[k];
            let poly_Cz_bound_point = poly_Cz_bound_point + Cz2[k] - Cz1[k];
            let eval_point_3 = comb_func(
              &poly_e_bound_point,
              &poly_Az_bound_point,
              &poly_Bz_bound_point,
              &poly_Cz_bound_point,
            );

            (eval_point_0, eval_point_2, eval_point_3)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
          );

        let f = &e[left..];

        let poly_f_bound_point = f[i];

        // eval 0: bound_func is A(low)
        let eval_at_0 = poly_f_bound_point * i_eval_at_0;

        // eval 2: bound_func is -A(low) + 2*A(high)
        let eval_at_2 = poly_f_bound_point * i_eval_at_2;

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let eval_at_3 = poly_f_bound_point * i_eval_at_3;

        (eval_at_0, eval_at_2, eval_at_3)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
      );

    // multiply by the common factors
    let one_minus_rho = E::Scalar::ONE - rho;
    let three_rho_minus_one = E::Scalar::from(3) * rho - E::Scalar::ONE;
    let five_rho_minus_two = E::Scalar::from(5) * rho - E::Scalar::from(2);

    (
      eval_at_0 * one_minus_rho,
      eval_at_2 * three_rho_minus_one,
      eval_at_3 * five_rho_minus_two,
    )
  }

  /// NeutronNova NIFS prove for a batch of R1CS instances
  pub fn prove(
    S: &SplitR1CSShape<E>,
    Us: &[R1CSInstance<E>],
    Ws: &[R1CSWitness<E>],
    transcript: &mut E::TE,
  ) -> Result<
    (
      Self,
      R1CSInstance<E>,
      R1CSWitness<E>,
      Vec<E::Scalar>,
      Vec<E::Scalar>,
      Vec<E::Scalar>,
      Vec<E::Scalar>,
      E::Scalar,
    ),
    SpartanError,
  > {
    let n = Us.len().next_power_of_two();
    let ell_b = n.log_2();

    if Us.is_empty() || Us.len() != Ws.len() {
      return Err(SpartanError::IncorrectWitness {
        reason: "Us and Ws must have the same length".to_string(),
      });
    }

    let mut Us = Us.to_vec();
    let mut Ws = Ws.to_vec();
    if Us.len() < n {
      Us.extend(vec![Us[0].clone(); n - Us.len()]);
      Ws.extend(vec![Ws[0].clone(); n - Ws.len()]);
    }

    for U in Us.iter() {
      transcript.absorb(b"U", U);
    }
    let T = E::Scalar::ZERO;
    transcript.absorb(b"T", &T);

    let (ell_cons, left, right) = compute_tensor_decomp(S.num_cons);
    let tau = transcript.squeeze(b"tau")?;
    let E_eq = PowPolynomial::new(&tau, ell_cons).split_evals(left, right);

    // length of chunk
    let chunk_len = left * right;

    let mut rhos = Vec::with_capacity(ell_b);
    for _ in 0..ell_b {
      rhos.push(transcript.squeeze(b"rho")?);
    }

    // Compute (A z, B z, C z) for each instance in parallel, minimizing clones
    let triples = (0..n)
      .into_par_iter()
      .map(|i| {
        // Build z = [W || 1 || X] without intermediate temporary concat vectors
        let w = &Ws[i].W;
        let x = &Us[i].X;
        let mut z = Vec::with_capacity(w.len() + 1 + x.len());
        z.extend_from_slice(w);
        z.push(E::Scalar::ONE);
        z.extend_from_slice(x);
        S.multiply_vec(&z)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Split the triples without cloning inner vectors
    let mut A_layers: Vec<Vec<E::Scalar>> = Vec::with_capacity(n);
    let mut B_layers: Vec<Vec<E::Scalar>> = Vec::with_capacity(n);
    let mut C_layers: Vec<Vec<E::Scalar>> = Vec::with_capacity(n);
    for (a, b, c) in triples {
      A_layers.push(a);
      B_layers.push(b);
      C_layers.push(c);
    }

    let mut polys: Vec<UniPoly<E::Scalar>> = Vec::with_capacity(ell_b);
    let mut r_bs: Vec<E::Scalar> = Vec::with_capacity(ell_b);
    let mut T_cur = E::Scalar::ZERO; // the current target value, starts at 0
    let mut acc_eq = E::Scalar::ONE;
    let mut m = n;
    for t in 0..ell_b {
      let rho_t = rhos[t];

      // Round polynomial: use rho_t inside prove_helper (this multiplies by eq(b_t; rho_t))
      let pairs = m / 2;

      let (e0, e2, e3) = (0..pairs)
        .into_par_iter()
        .map(|pair_idx| {
          let lo = 2 * pair_idx;
          let hi = lo + 1;
          let (a0, a2, a3) = Self::prove_helper(
            &rho_t,
            (left, right),
            &E_eq,
            &A_layers[lo],
            &B_layers[lo],
            &C_layers[lo],
            &A_layers[hi],
            &B_layers[hi],
            &C_layers[hi],
          );

          let w = suffix_weight_full::<E::Scalar>(t, ell_b, pair_idx, &rhos);

          (a0 * w, a2 * w, a3 * w)
        })
        .reduce(
          || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
          |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

      let se0 = acc_eq * e0;
      let se2 = acc_eq * e2;
      let se3 = acc_eq * e3;
      let poly_t = UniPoly::<E::Scalar>::from_evals(&[se0, T_cur - se0, se2, se3])?;
      polys.push(poly_t.clone());

      // Commit poly_t, then draw r_t (Fiat–Shamir per round)
      transcript.absorb(b"poly", &poly_t);
      let r_b = transcript.squeeze(b"r_b")?;

      acc_eq *= (E::Scalar::ONE - r_b) * (E::Scalar::ONE - rho_t) + r_b * rho_t; // update the accumulated equality polynomial
      T_cur = poly_t.evaluate(&r_b);
      r_bs.push(r_b);

      // Fold A/B/C for next round (weights 1-r_b, r_b)
      let mut next_A: Vec<Vec<E::Scalar>> = Vec::with_capacity(pairs);
      let mut next_B: Vec<Vec<E::Scalar>> = Vec::with_capacity(pairs);
      let mut next_C: Vec<Vec<E::Scalar>> = Vec::with_capacity(pairs);

      next_A.par_extend((0..pairs).into_par_iter().map(|i| {
        let lo = 2 * i;
        let hi = lo + 1;
        let mut v = vec![E::Scalar::ZERO; chunk_len];
        v.iter_mut().enumerate().for_each(|(k, val)| {
          *val = A_layers[lo][k] + (A_layers[hi][k] - A_layers[lo][k]) * r_b;
        });
        v
      }));
      next_B.par_extend((0..pairs).into_par_iter().map(|i| {
        let lo = 2 * i;
        let hi = lo + 1;
        let mut v = vec![E::Scalar::ZERO; chunk_len];
        v.iter_mut().enumerate().for_each(|(k, val)| {
          *val = B_layers[lo][k] + (B_layers[hi][k] - B_layers[lo][k]) * r_b;
        });
        v
      }));
      next_C.par_extend((0..pairs).into_par_iter().map(|i| {
        let lo = 2 * i;
        let hi = lo + 1;
        let mut v = vec![E::Scalar::ZERO; chunk_len];
        v.iter_mut().enumerate().for_each(|(k, val)| {
          *val = C_layers[lo][k] + (C_layers[hi][k] - C_layers[lo][k]) * r_b;
        });
        v
      }));

      A_layers = next_A;
      B_layers = next_B;
      C_layers = next_C;

      // m becomes ceil(m/2)
      m = pairs;
    }

    // T_out = poly_last(r_last) / eq(r_b, rho)
    let T_out = T_cur * acc_eq.invert().unwrap();

    let folded_W = R1CSWitness::fold_multiple(&r_bs, &Ws)?;
    let folded_U = R1CSInstance::fold_multiple(&r_bs, &Us);

    Ok((
      Self { polys },
      folded_U,
      folded_W,
      E_eq,
      A_layers[0].clone(),
      B_layers[0].clone(),
      C_layers[0].clone(),
      T_out,
    ))
  }

  /// NeutronNova NIFS verify for a batch of R1CS instances
  pub fn verify(
    &self,
    Us: &[R1CSInstance<E>],
    transcript: &mut E::TE,
  ) -> Result<FoldedR1CSInstance<E>, SpartanError> {
    let n = Us.len().next_power_of_two();
    let ell_b = n.log_2();

    let mut Us = Us.to_vec();
    if Us.len() < n {
      Us.extend(vec![Us[0].clone(); n - Us.len()]);
    }

    for U in Us.iter() {
      transcript.absorb(b"U", U);
    }
    let T = E::Scalar::ZERO;
    transcript.absorb(b"T", &T);

    let tau = transcript.squeeze(b"tau")?;

    let mut rhos = Vec::with_capacity(ell_b);
    for _ in 0..ell_b {
      rhos.push(transcript.squeeze(b"rho")?);
    }

    if self.polys.len() != ell_b {
      return Err(SpartanError::ProofVerifyError {
        reason: format!("Expected {} polys, got {}", ell_b, self.polys.len()),
      });
    }

    // Then, per round: absorb poly_t, squeeze r_b
    let mut r_bs = Vec::with_capacity(ell_b);
    let mut T_cur = E::Scalar::ZERO; // current target value, starts at 0
    let mut acc_eq = E::Scalar::ONE; // accumulated equality polynomial
    for (t, poly_t) in self.polys.iter().enumerate() {
      if poly_t.degree() != 3 || poly_t.eval_at_zero() + poly_t.eval_at_one() != T_cur {
        return Err(SpartanError::ProofVerifyError {
          reason: format!("poly {t} is not valid"),
        });
      }
      transcript.absorb(b"poly", poly_t);

      let r_b = transcript.squeeze(b"r_b")?;
      T_cur = poly_t.evaluate(&r_b);
      acc_eq *= (E::Scalar::ONE - r_b) * (E::Scalar::ONE - rhos[t]) + r_b * rhos[t]; // update the accumulated equality polynomial

      r_bs.push(r_b);
    }

    // Fold public instances with the same r_b sequence
    let folded_U = R1CSInstance::fold_multiple(&r_bs, &Us);

    // T_out = poly_last(r_last) / eq(r_b, rho)
    let T_out = T_cur * acc_eq.invert().unwrap();

    Ok(FoldedR1CSInstance {
      U: folded_U,
      tau,
      T: T_out,
    })
  }
}

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaProverKey<E: Engine> {
  ck: CommitmentKey<E>,
  S_step: SplitR1CSShape<E>,
  S_core: SplitR1CSShape<E>,
  vk_digest: SpartanDigest, // digest of the verifier's key
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaVerifierKey<E: Engine> {
  ck: CommitmentKey<E>,
  vk_ee: <E::PCS as PCSEngineTrait<E>>::VerifierKey,
  S_step: SplitR1CSShape<E>,
  S_core: SplitR1CSShape<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<SpartanDigest>,
}

impl<E: Engine> SimpleDigestible for NeutronNovaVerifierKey<E> {}

impl<E: Engine> DigestHelperTrait<E> for NeutronNovaVerifierKey<E> {
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
pub struct NeutronNovaPrepSNARK<E: Engine> {
  ps_step: Vec<PrecommittedState<E>>,
  ps_core: PrecommittedState<E>,
}

/// Holds the proof produced by the NeutronNova folding scheme followed by NeutronNova SNARK
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaSNARK<E: Engine> {
  step_instances: Vec<SplitR1CSInstance<E>>,
  core_instance: SplitR1CSInstance<E>,
  nifs: NeutronNovaNIFS<E>,
  sc_proof_outer: SumcheckProof<E>,
  claims_outer: (E::Scalar, E::Scalar, E::Scalar),
  sc_proof_inner: SumcheckProof<E>,
  eval_W: E::Scalar,
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
}

impl<E: Engine> NeutronNovaSNARK<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Sets up the NeutronNova SNARK for a batch of circuits of type `C1` and a single circuit of type `C2`
  pub fn setup<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    step_circuit: &C1,
    core_circuit: &C2,
  ) -> Result<(NeutronNovaProverKey<E>, NeutronNovaVerifierKey<E>), SpartanError> {
    let S_step = ShapeCS::r1cs_shape(step_circuit)?;
    let S_core = ShapeCS::r1cs_shape(core_circuit)?;

    let (ck, vk_ee) = SplitR1CSShape::commitment_key(&[&S_step, &S_core])?;

    let vk: NeutronNovaVerifierKey<E> = NeutronNovaVerifierKey {
      ck: ck.clone(),
      S_step: S_step.clone(),
      S_core: S_core.clone(),
      vk_ee,
      digest: OnceCell::new(),
    };
    let pk = NeutronNovaProverKey {
      ck,
      S_step,
      S_core,
      vk_digest: vk.digest()?,
    };

    Ok((pk, vk))
  }

  /// Prepares the pre-processed state for proving
  pub fn prep_prove<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<NeutronNovaPrepSNARK<E>, SpartanError> {
    // we synthesize shared witness for the first circuit; every other circuit including the core circuit shares this witness
    let mut ps =
      SatisfyingAssignment::shared_witness(&pk.S_step, &pk.ck, &step_circuits[0], is_small)?;

    let ps_step = (0..step_circuits.len())
      .into_par_iter()
      .map(|i| {
        // copy ps to avoid mutating the original shared witness
        let mut ps_i = ps.clone();
        SatisfyingAssignment::precommitted_witness(
          &mut ps_i,
          &pk.S_step,
          &pk.ck,
          &step_circuits[i],
          is_small,
        )?;
        Ok(ps_i)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // we don't need to make a copy of ps for the core circuit, as it will be used only once
    SatisfyingAssignment::precommitted_witness(
      &mut ps,
      &pk.S_core,
      &pk.ck,
      core_circuit,
      is_small,
    )?;

    Ok(NeutronNovaPrepSNARK {
      ps_step,
      ps_core: ps,
    })
  }

  /// Prove the folding of a batch of R1CS instances and a core circuit that connects them together
  pub fn prove<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    prep_snark: &NeutronNovaPrepSNARK<E>,
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<Self, SpartanError> {
    let mut prep_snark = prep_snark.clone(); // make a copy so we can modify it

    // Parallel generation of instances and witnesses
    // Build instances and witnesses in one parallel pass
    let (step_instances, step_witnesses) = prep_snark
      .ps_step
      .par_iter_mut()
      .zip(step_circuits.par_iter().enumerate())
      .map(|(pre_state, (i, circuit))| {
        let mut transcript = E::TE::new(b"neutronnova_prove");
        transcript.absorb(b"vk", &pk.vk_digest);
        transcript.absorb(
          b"num_circuits",
          &E::Scalar::from(step_circuits.len() as u64),
        );
        transcript.absorb(b"circuit_index", &E::Scalar::from(i as u64));

        let public_values = circuit
          .public_values()
          .map_err(|e| SpartanError::SynthesisError {
            reason: format!("Circuit does not provide public IO: {e}"),
          })?;
        transcript.absorb(b"public_values", &public_values.as_slice());

        SatisfyingAssignment::r1cs_instance_and_witness(
          pre_state,
          &pk.S_step,
          &pk.ck,
          circuit,
          is_small,
          &mut transcript,
        )
      })
      .try_fold(
        || (Vec::new(), Vec::new()),
        |mut acc, res: Result<_, SpartanError>| {
          let (u, w) = res?;
          acc.0.push(u);
          acc.1.push(w);
          Ok(acc)
        },
      )
      .try_reduce(
        || (Vec::new(), Vec::new()),
        |mut a, mut b| {
          a.0.append(&mut b.0);
          a.1.append(&mut b.1);
          Ok(a)
        },
      )?;

    let step_instances_regular = step_instances
      .iter()
      .map(|u| u.to_regular_instance())
      .collect::<Result<Vec<_>, _>>()?;

    // synthesize the core instance
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &pk.vk_digest);
    let (core_instance, _core_witness) = SatisfyingAssignment::r1cs_instance_and_witness(
      &mut prep_snark.ps_core,
      &pk.S_core,
      &pk.ck,
      core_circuit,
      is_small,
      &mut transcript,
    )?;
    let core_instance_regular = core_instance.to_regular_instance()?;

    // We start a new transcript for the NeutronNova NIFS proof
    // All instances will be absorbed into the transcript
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &pk.vk_digest);

    // absorb the core instance
    transcript.absorb(b"core_instance", &core_instance_regular);

    // NIFS absorbs and folds the step instances
    let (nifs, folded_U, folded_W, E, Az, Bz, Cz, T_out) = NeutronNovaNIFS::prove(
      &pk.S_step,
      &step_instances_regular,
      &step_witnesses,
      &mut transcript,
    )?;

    // we now prove the validity of folded witness
    let (_ell, left, right) = compute_tensor_decomp(pk.S_step.num_cons);
    let (E1, E2) = E.split_at(left);
    let mut full_E = vec![E::Scalar::ONE; left * right];
    full_E
      .par_chunks_mut(left)
      .enumerate()
      .for_each(|(i, row)| {
        let e2 = E2[i];
        row.iter_mut().zip(E1.iter()).for_each(|(val, e1)| {
          *val = e2 * *e1;
        });
      });
    let mut poly_tau = MultilinearPolynomial::new(full_E);

    let num_vars = pk.S_step.num_shared + pk.S_step.num_precommitted + pk.S_step.num_rest;
    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(pk.S_step.num_cons.ilog2()).unwrap(),
      (usize::try_from(num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check preparation
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
    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::<E>::prove_cubic_with_additive_term(
      &T_out,
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
    let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&pk.S_step, &evals_rx);
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
      // z = [W || 1 || X], then already zero-padded to length 2 * num_vars
      let mut v = vec![E::Scalar::ZERO; num_vars * 2];

      let w_len = folded_W.W.len();
      v[..w_len].copy_from_slice(&folded_W.W);

      v[w_len] = E::Scalar::ONE;

      let x_len = folded_U.X.len();
      v[w_len + 1..w_len + 1 + x_len].copy_from_slice(&folded_U.X);
      v
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
    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::<E>::prove_quad(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      comb_func,
      &mut transcript,
    )?;
    info!(elapsed_ms = %sc2_t.elapsed().as_millis(), "inner_sumcheck");

    let (_pcs_span, pcs_t) = start_span!("pcs_prove");
    let (eval_W, eval_arg) = E::PCS::prove(
      &pk.ck,
      &mut transcript,
      &folded_U.comm_W,
      &folded_W.W,
      &folded_W.r_W,
      &r_y[1..],
    )?;
    info!(elapsed_ms = %pcs_t.elapsed().as_millis(), "pcs_prove");

    Ok(Self {
      step_instances,
      core_instance,
      nifs,
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      sc_proof_inner,
      eval_W,
      eval_arg,
    })
  }

  /// Verifies the NeutronNovaSNARK and returns the public IO from the instances
  pub fn verify(
    &self,
    vk: &NeutronNovaVerifierKey<E>,
    num_instances: usize,
  ) -> Result<(Vec<Vec<E::Scalar>>, Vec<E::Scalar>), SpartanError> {
    let (_verify_span, verify_t) = start_span!("neutronnova_verify");
    if num_instances != self.step_instances.len() {
      return Err(SpartanError::ProofVerifyError {
        reason: format!(
          "Expected {} instances, got {}",
          num_instances,
          self.step_instances.len()
        ),
      });
    }

    // validate the step instances
    for (i, u) in self.step_instances.iter().enumerate() {
      let mut transcript = E::TE::new(b"neutronnova_prove");
      transcript.absorb(b"vk", &vk.digest()?);
      transcript.absorb(
        b"num_circuits",
        &E::Scalar::from(self.step_instances.len() as u64),
      );
      transcript.absorb(b"circuit_index", &E::Scalar::from(i as u64));
      u.validate(&vk.S_step, &mut transcript)?;
    }

    // validate the core instance
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &vk.digest()?);
    self.core_instance.validate(&vk.S_core, &mut transcript)?;

    // we require all step instances to have the same shared commitment and match the shared commitment of the core instance
    for u in &self.step_instances {
      if u.comm_W_shared != self.core_instance.comm_W_shared {
        return Err(SpartanError::ProofVerifyError {
          reason: "All instances must have the same shared commitment".to_string(),
        });
      }
    }

    let step_instances_regular = self
      .step_instances
      .par_iter()
      .map(|u| u.to_regular_instance())
      .collect::<Result<Vec<_>, _>>()?;

    let core_instance_regular = self.core_instance.to_regular_instance()?;

    // We start a new transcript for the NeutronNova NIFS proof
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &vk.digest()?);
    transcript.absorb(b"core_instance", &core_instance_regular);

    // absorb the step instances and fold them
    let folded_U = self.nifs.verify(&step_instances_regular, &mut transcript)?;

    let num_vars = vk.S_step.num_shared + vk.S_step.num_precommitted + vk.S_step.num_rest;

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(vk.S_step.num_cons.ilog2()).unwrap(),
      (usize::try_from(num_vars.ilog2()).unwrap() + 1),
    );

    info!(
      "Verifying R1CS SNARK with {} rounds for outer sum-check and {} rounds for inner sum-check",
      num_rounds_x, num_rounds_y
    );

    // outer sum-check
    let (_outer_sumcheck_span, outer_sumcheck_t) = start_span!("outer_sumcheck_verify");
    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(folded_U.T, num_rounds_x, 3, &mut transcript)?;

    // verify claim_outer_final
    let (claim_Az, claim_Bz, claim_Cz) = self.claims_outer;
    let e_bound_rx = PowPolynomial::new(&folded_U.tau, r_x.len()).evaluate(&r_x)?;
    let claim_outer_final_expected = e_bound_rx * (claim_Az * claim_Bz - claim_Cz);
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
          .chain(folded_U.U.X.iter().cloned())
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

    let evals = multi_evaluate(&[&vk.S_step.A, &vk.S_step.B, &vk.S_step.C], &r_x, &r_y);

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
      &folded_U.U.comm_W,
      &r_y[1..],
      &self.eval_W,
      &self.eval_arg,
    )?;
    info!(elapsed_ms = %pcs_verify_t.elapsed().as_millis(), "pcs_verify");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "neutronnova_verify");

    let public_values_step = self
      .step_instances
      .iter()
      .take(num_instances)
      .map(|u| u.public_values.clone())
      .collect::<Vec<Vec<_>>>();

    let public_values_core = self.core_instance.public_values.clone();

    // return a vector of public values
    Ok((public_values_step, public_values_core))
  }
}

#[cfg(test)]
mod benchmarks {
  use super::*;
  use crate::provider::T256HyraxEngine;
  use bellpepper::gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    sha256::sha256,
  };
  use bellpepper_core::{ConstraintSystem, SynthesisError};
  use core::marker::PhantomData;

  #[derive(Clone, Debug)]
  struct Sha256Circuit<E: Engine> {
    preimage: Vec<u8>,
    _p: PhantomData<E>,
  }

  impl<E: Engine> SpartanCircuit<E> for Sha256Circuit<E> {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
      Ok(vec![E::Scalar::ZERO]) // Placeholder, we don't use public values in this example
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![]) // Placeholder, we don't use shared variables in this example
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![]) // Placeholder, we don't use precommitted variables in this example
    }

    fn num_challenges(&self) -> usize {
      0 // Placeholder, we don't use challenges in this example
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _shared: &[AllocatedNum<E::Scalar>],
      _precommitted: &[AllocatedNum<E::Scalar>],
      _challenges: Option<&[E::Scalar]>, // challenges from the verifier
    ) -> Result<(), SynthesisError> {
      // we write a circuit that checks if the input is a SHA256 preimage
      let bit_values: Vec<_> = self
        .preimage
        .clone()
        .into_iter()
        .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
        .map(Some)
        .collect();
      assert_eq!(bit_values.len(), self.preimage.len() * 8);

      let preimage_bits = bit_values
        .into_iter()
        .enumerate()
        .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
        .map(|b| b.map(Boolean::from))
        .collect::<Result<Vec<_>, _>>()?;

      let _ = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::ZERO))?;
      x.inputize(cs.namespace(|| "inputize x"))?;

      Ok(())
    }
  }

  fn generate_sha_r1cs<E: Engine>(
    num_circuits: usize,
    len: usize,
  ) -> (
    NeutronNovaProverKey<E>,
    NeutronNovaVerifierKey<E>,
    Vec<Sha256Circuit<E>>,
  )
  where
    E::PCS: FoldingEngineTrait<E>, // Ensure that the PCS supports folding
  {
    let circuit = Sha256Circuit::<E> {
      preimage: vec![0u8; len],
      _p: Default::default(),
    };

    let (pk, vk) = NeutronNovaSNARK::<E>::setup(&circuit, &circuit).unwrap();

    let circuits = (0..num_circuits)
      .map(|i| Sha256Circuit::<E> {
        preimage: vec![i as u8; len],
        _p: Default::default(),
      })
      .collect::<Vec<_>>();

    (pk, vk, circuits)
  }

  fn test_neutron_inner<E: Engine, C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    name: &str,
    pk: &NeutronNovaProverKey<E>,
    vk: &NeutronNovaVerifierKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
  ) where
    E::PCS: FoldingEngineTrait<E>,
  {
    println!(
      "[bench_neutron_inner] name: {name}, num_circuits: {}",
      step_circuits.len()
    );
    // sanity check: prove and verify before benching
    let ps = NeutronNovaSNARK::<E>::prep_prove(pk, step_circuits, core_circuit, true).unwrap();

    let res = NeutronNovaSNARK::prove(pk, step_circuits, core_circuit, &ps, true);
    assert!(res.is_ok());

    let snark = res.unwrap();
    let res = snark.verify(vk, step_circuits.len());
    assert!(res.is_ok());

    let (public_values_step, _public_values_core) = res.unwrap();
    assert_eq!(public_values_step.len(), step_circuits.len());
  }

  #[test]
  fn test_neutron_sha256() {
    type E = T256HyraxEngine;

    for num_circuits in [2, 7, 32, 64] {
      for len in [32, 64].iter() {
        let (pk, vk, circuits) = generate_sha_r1cs::<E>(num_circuits, *len);
        test_neutron_inner(
          &format!("sha256_num_circuits={num_circuits}_len={len}"),
          &pk,
          &vk,
          &circuits,
          &circuits[0], // core circuit is the first one, for test purposes
        );
      }
    }
  }
}
