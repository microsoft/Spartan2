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
  polys::{power::PowPolynomial, univariate::UniPoly},
  r1cs::{R1CSInstance, R1CSWitness, SplitR1CSInstance, SplitR1CSShape},
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
  /// Computes the evaluations of the sum-check polynomial at 0, 2, 3, and 4
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
  ) -> (E::Scalar, E::Scalar, E::Scalar, E::Scalar) {
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
    let (eval_at_0, eval_at_2, eval_at_3, eval_at_4) = (0..right)
      .into_par_iter()
      .map(|i| {
        let (i_eval_at_0, i_eval_at_2, i_eval_at_3, i_eval_at_4) = (0..left)
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

            // eval 4: bound_func is -3A(low) + 4A(high); computed incrementally with bound_func applied to eval(3)
            let poly_Az_bound_point = poly_Az_bound_point + Az2[k] - Az1[k];
            let poly_Bz_bound_point = poly_Bz_bound_point + Bz2[k] - Bz1[k];
            let poly_Cz_bound_point = poly_Cz_bound_point + Cz2[k] - Cz1[k];
            let eval_point_4 = comb_func(
              &poly_e_bound_point,
              &poly_Az_bound_point,
              &poly_Bz_bound_point,
              &poly_Cz_bound_point,
            );

            (eval_point_0, eval_point_2, eval_point_3, eval_point_4)
          })
          .reduce(
            || {
              (
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
              )
            },
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
          );

        let f = &e[left..];

        let poly_f_bound_point = f[i];

        // eval 0: bound_func is A(low)
        let eval_at_0 = poly_f_bound_point * i_eval_at_0;

        // eval 2: bound_func is -A(low) + 2*A(high)
        let eval_at_2 = poly_f_bound_point * i_eval_at_2;

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let eval_at_3 = poly_f_bound_point * i_eval_at_3;

        // eval 4: bound_func is -3A(low) + 4A(high); computed incrementally with bound_func applied to eval(3)
        let eval_at_4 = poly_f_bound_point * i_eval_at_4;

        (eval_at_0, eval_at_2, eval_at_3, eval_at_4)
      })
      .reduce(
        || {
          (
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
          )
        },
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
      );

    // multiply by the common factors
    let one_minus_rho = E::Scalar::ONE - rho;
    let three_rho_minus_one = E::Scalar::from(3) * rho - E::Scalar::ONE;
    let five_rho_minus_two = E::Scalar::from(5) * rho - E::Scalar::from(2);
    let seven_rho_minus_three = E::Scalar::from(7) * rho - E::Scalar::from(3);

    (
      eval_at_0 * one_minus_rho,
      eval_at_2 * three_rho_minus_one,
      eval_at_3 * five_rho_minus_two,
      eval_at_4 * seven_rho_minus_three,
    )
  }

  /// NeutronNova NIFS prove for a batch of R1CS instances
  pub fn prove(
    S: &SplitR1CSShape<E>,
    Us: &[R1CSInstance<E>],
    Ws: &[R1CSWitness<E>],
    transcript: &mut E::TE,
  ) -> Result<(Self, R1CSWitness<E>), SpartanError> {
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

      let (e0, e2, e3, e4) = (0..pairs)
        .into_par_iter()
        .map(|pair_idx| {
          let lo = 2 * pair_idx;
          let hi = lo + 1;
          let (a0, a2, a3, a4) = Self::prove_helper(
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

          (a0 * w, a2 * w, a3 * w, a4 * w)
        })
        .reduce(
          || {
            (
              E::Scalar::ZERO,
              E::Scalar::ZERO,
              E::Scalar::ZERO,
              E::Scalar::ZERO,
            )
          },
          |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
        );

      let se0 = acc_eq * e0;
      let se2 = acc_eq * e2;
      let se3 = acc_eq * e3;
      let se4 = acc_eq * e4;
      let poly_t = UniPoly::<E::Scalar>::from_evals(&[se0, T_cur - se0, se2, se3, se4])?;
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

    // Fold witnesses with the same r_t sequence
    let folded_W = R1CSWitness::fold_multiple(&r_bs, &Ws)?;

    Ok((Self { polys }, folded_W))
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
      if poly_t.degree() != 4 || poly_t.eval_at_zero() + poly_t.eval_at_one() != T_cur {
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
  S: SplitR1CSShape<E>,
  vk_digest: SpartanDigest, // digest of the verifier's key
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaVerifierKey<E: Engine> {
  ck: CommitmentKey<E>,
  vk_ee: <E::PCS as PCSEngineTrait<E>>::VerifierKey,
  S: SplitR1CSShape<E>,
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
  ps: Vec<PrecommittedState<E>>,
}

/// Holds the proof produced by the NeutronNova folding scheme followed by NeutronNova SNARK
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaSNARK<E: Engine> {
  instances: Vec<SplitR1CSInstance<E>>,
  nifs: NeutronNovaNIFS<E>,
  folded_W: R1CSWitness<E>,
}

impl<E: Engine> NeutronNovaSNARK<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Sets up the NeutronNova SNARK for a batch of circuits
  pub fn setup<C: SpartanCircuit<E>>(
    circuit: &[C], // uniform circuit
  ) -> Result<(NeutronNovaProverKey<E>, NeutronNovaVerifierKey<E>), SpartanError> {
    let (S, ck, vk_ee) = ShapeCS::r1cs_shape(&circuit[0])?;

    let vk: NeutronNovaVerifierKey<E> = NeutronNovaVerifierKey {
      ck: ck.clone(),
      S: S.clone(),
      vk_ee,
      digest: OnceCell::new(),
    };
    let pk = NeutronNovaProverKey {
      ck,
      S,
      vk_digest: vk.digest()?,
    };

    Ok((pk, vk))
  }

  /// Prepares the pre-processed state for proving
  pub fn prep_prove<C: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    circuits: &[C],
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<NeutronNovaPrepSNARK<E>, SpartanError> {
    let ps = (0..circuits.len())
      .into_par_iter()
      .map(|i| SatisfyingAssignment::precommitted_witness(&pk.S, &pk.ck, &circuits[i], is_small))
      .collect::<Result<Vec<_>, _>>()?;

    Ok(NeutronNovaPrepSNARK { ps })
  }

  /// Prove the folding of a batch of R1CS instances
  pub fn prove<C: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    circuits: &[C],
    prep_snark: &mut NeutronNovaPrepSNARK<E>,
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<Self, SpartanError> {
    // Parallel generation of instances and witnesses
    // Build instances and witnesses in one parallel pass (avoid intermediate Vec + unzip)
    let (instances, witnesses) = prep_snark
      .ps
      .par_iter_mut()
      .zip(circuits.par_iter().enumerate())
      .map(|(pre_state, (i, circuit))| {
        let mut transcript = E::TE::new(b"neutronnova_prove");
        transcript.absorb(b"vk", &pk.vk_digest);
        transcript.absorb(b"num_circuits", &E::Scalar::from(circuits.len() as u64));
        transcript.absorb(b"circuit_index", &E::Scalar::from(i as u64));

        let public_values = circuit
          .public_values()
          .map_err(|e| SpartanError::SynthesisError {
            reason: format!("Circuit does not provide public IO: {e}"),
          })?;
        transcript.absorb(b"public_values", &public_values.as_slice());

        SatisfyingAssignment::r1cs_instance_and_witness(
          pre_state,
          &pk.S,
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

    let instances_regular = instances
      .iter()
      .map(|u| u.to_regular_instance())
      .collect::<Result<Vec<_>, _>>()?;

    // We start a new transcript for the NeutronNova NIFS proof
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &pk.vk_digest);

    let (nifs, folded_W) =
      NeutronNovaNIFS::prove(&pk.S, &instances_regular, &witnesses, &mut transcript)?;

    Ok(Self {
      instances,
      nifs,
      folded_W,
    })
  }

  /// Verifies the NeutronNovaSNARK
  pub fn verify(&self, vk: &NeutronNovaVerifierKey<E>) -> Result<(), SpartanError> {
    for instance in &self.instances {
      let mut transcript = E::TE::new(b"neutronnova_prove");
      transcript.absorb(b"vk", &vk.digest()?);
      transcript.absorb(
        b"num_circuits",
        &E::Scalar::from(self.instances.len() as u64),
      );

      instance.validate(&vk.S, &mut transcript)?;
    }

    let instances = self
      .instances
      .par_iter()
      .map(|u| u.to_regular_instance())
      .collect::<Result<Vec<_>, _>>()?;

    // We start a new transcript for the NeutronNova NIFS proof
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &vk.digest()?);

    let folded_U = self.nifs.verify(&instances, &mut transcript)?;

    // check the satisfiability of folded instance using the provided witness
    is_sat_with_target(&vk.ck, &vk.S, &folded_U, &self.folded_W)?;

    Ok(())
  }
}

/// Check if the folded witness satisfies the folded instance
fn is_sat_with_target<E: Engine>(
  ck: &CommitmentKey<E>,
  S: &SplitR1CSShape<E>,
  U: &FoldedR1CSInstance<E>,
  W: &R1CSWitness<E>,
) -> Result<(), SpartanError> {
  let (ell, left, right) = compute_tensor_decomp(S.num_cons);
  let E = PowPolynomial::new(&U.tau, ell).split_evals(left, right);

  let z = [W.W.clone(), vec![E::Scalar::ONE], U.U.X.clone()].concat();
  let (Az, Bz, Cz) = S.multiply_vec(&z)?;

  // full_E is the outer outer product of E1 and E2
  // E1 and E2 are splits of E
  let (E1, E2) = E.split_at(left);
  let mut full_E = vec![E::Scalar::ONE; left * right];
  for i in 0..right {
    for j in 0..left {
      full_E[i * left + j] = E2[i] * E1[j];
    }
  }

  let sum = full_E
    .par_iter()
    .zip(Az.par_iter())
    .zip(Bz.par_iter())
    .zip(Cz.par_iter())
    .map(|(((e, a), b), c)| *e * ((*a) * (*b) - *c))
    .reduce(|| E::Scalar::ZERO, |acc, x| acc + x);

  if sum != U.T {
    println!("sum: {sum:?}");
    println!("U.T: {:?}", U.T);
    return Err(SpartanError::UnSat {
      reason: "sum != U.T".to_string(),
    });
  }

  // check the validity of the commitments
  let comm_W = E::PCS::commit(ck, &W.W, &W.r_W, W.is_small)?;

  if comm_W != U.U.comm_W {
    return Err(SpartanError::UnSat {
      reason: "comm_W != U.comm_W".to_string(),
    });
  }

  Ok(())
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
  use criterion::Criterion;

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

    let (pk, vk) = NeutronNovaSNARK::<E>::setup(std::slice::from_ref(&circuit)).unwrap();

    let circuits = (0..num_circuits)
      .map(|i| Sha256Circuit::<E> {
        preimage: vec![i as u8; len],
        _p: Default::default(),
      })
      .collect::<Vec<_>>();

    (pk, vk, circuits)
  }

  fn bench_neutron_inner<E: Engine, C: SpartanCircuit<E>>(
    c: &mut Criterion,
    name: &str,
    pk: &NeutronNovaProverKey<E>,
    vk: &NeutronNovaVerifierKey<E>,
    circuits: &[C],
  ) where
    E::PCS: FoldingEngineTrait<E>,
  {
    println!(
      "[bench_neutron_inner] name: {name}, num_circuits: {}",
      circuits.len()
    );
    // sanity check: prove and verify before benching
    let mut ps = NeutronNovaSNARK::<E>::prep_prove(pk, circuits, true).unwrap();

    let res = NeutronNovaSNARK::prove(pk, circuits, &mut ps, true);
    assert!(res.is_ok());

    let snark = res.unwrap();
    let res = snark.verify(vk);
    assert!(res.is_ok());

    c.bench_function(&format!("neutron_snark_{name}"), |b| {
      b.iter(|| {
        let mut ps = NeutronNovaSNARK::<E>::prep_prove(pk, circuits, true).unwrap();

        let res = NeutronNovaSNARK::prove(pk, circuits, &mut ps, true);
        assert!(res.is_ok());
      })
    });
  }

  #[test]
  fn bench_neutron_sha256() {
    type E = T256HyraxEngine;

    let mut criterion = Criterion::default();
    for num_circuits in [7, 32, 64] {
      for len in [64].iter() {
        let (pk, vk, circuits) = generate_sha_r1cs::<E>(num_circuits, *len);
        bench_neutron_inner(
          &mut criterion,
          &format!("sha256_num_circuits={num_circuits}_len={len}"),
          &pk,
          &vk,
          &circuits,
        );
      }
    }
  }
}
