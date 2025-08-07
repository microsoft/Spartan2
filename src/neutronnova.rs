//! This module implements NeutronNova's folding scheme for folding together a batch of R1CS instances
//! This implementation focuses on a non-recursive version of NeutronNova and tagets the case where the batch size is moderately large.
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
  r1cs::{R1CSInstance, R1CSWitness, SplitR1CSShape},
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

fn compute_tensor_decomp(n: usize) -> (usize, usize, usize) {
  let ell = n.next_power_of_two().log_2();
  // we split ell into ell1 and ell2 such that ell1 + ell2 = ell and ell1 >= ell2
  let ell1 = ell.div_ceil(2); // This ensures ell1 >= ell2
  let ell2 = ell / 2;
  let left = 1 << ell1;
  let right = 1 << ell2;

  (ell, left, right)
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
  poly: UniPoly<E::Scalar>,
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

  /// Prepares the SNARK for proving
  pub fn prep_prove<C: SpartanCircuit<E>>(
    pk: &NeutronNovaProverKey<E>,
    circuit: &[C],
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<NeutronNovaPrepSNARK<E>, SpartanError> {
    let ps = (0..circuit.len())
      .map(|i| SatisfyingAssignment::precommitted_witness(&pk.S, &pk.ck, &circuit[i], is_small))
      .collect::<Result<Vec<_>, _>>()?;

    Ok(NeutronNovaPrepSNARK { ps })
  }

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

  /// Prove the folding of a batch of R1CS instances
  pub fn prove(
    pk: &NeutronNovaProverKey<E>,
    instances: &[R1CSInstance<E>],
    witnesses: &[R1CSWitness<E>],
  ) -> Result<Self, SpartanError> {
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &pk.vk_digest);

    let U1 = &instances[0];
    let W1 = &witnesses[0];
    let U2 = &instances[1];
    let W2 = &witnesses[1];

    // append U1 and U2 to transcript
    transcript.absorb(b"U1", U1);
    transcript.absorb(b"U2", U2);

    let T = E::Scalar::ZERO; // we need all instances to be satisfying, so T is zero
    transcript.absorb(b"T", &T);

    // generate a challenge for the eq polynomial
    let tau = transcript.squeeze(b"tau")?;
    let (ell, left, right) = compute_tensor_decomp(pk.S.num_cons);
    let E = PowPolynomial::new(&tau, ell).split_evals(left, right);

    let rho = transcript.squeeze(b"rho")?;

    let (res1, res2) = rayon::join(
      || {
        let z1 = [W1.W.clone(), vec![E::Scalar::ONE], U1.X.clone()].concat();
        pk.S.multiply_vec(&z1)
      },
      || {
        let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
        pk.S.multiply_vec(&z2)
      },
    );

    let (Az1, Bz1, Cz1) = res1?;
    let (Az2, Bz2, Cz2) = res2?;

    // compute the sum-check polynomial's evaluations at 0, 2, 3
    let (eval_point_0, eval_point_2, eval_point_3, eval_point_4) =
      Self::prove_helper(&rho, (left, right), &E, &Az1, &Bz1, &Cz1, &Az2, &Bz2, &Cz2);

    let evals = vec![
      eval_point_0,
      T - eval_point_0,
      eval_point_2,
      eval_point_3,
      eval_point_4,
    ];
    let poly = UniPoly::<E::Scalar>::from_evals(&evals)?;

    // absorb poly in the RO
    transcript.absorb(b"poly", &poly);

    // squeeze a challenge
    let r_b = transcript.squeeze(b"r_b")?;

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let _T_out = poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap(); // TODO: remove unwrap

    let _folded_U = U1.fold(U2, &r_b)?;
    let folded_W = W1.fold(W2, &r_b)?;

    Ok(Self { poly, folded_W })
  }

  /// Verifies the NeutronNovaSNARK
  pub fn verify(
    &self,
    vk: &NeutronNovaVerifierKey<E>,
    instances: &[R1CSInstance<E>],
  ) -> Result<(), SpartanError> {
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &vk.digest()?);

    let U1 = &instances[0];
    let U2 = &instances[1];

    // append U1 and U2 to transcript
    transcript.absorb(b"U1", U1);
    transcript.absorb(b"U2", U2);

    let T = E::Scalar::ZERO; // we need all instances to be satisfying, so T is zero
    transcript.absorb(b"T", &T);

    // generate a challenge for the eq polynomial
    let tau = transcript.squeeze(b"tau")?;
    let (ell, left, right) = compute_tensor_decomp(vk.S.num_cons);
    let E = PowPolynomial::new(&tau, ell).split_evals(left, right);

    let rho = transcript.squeeze(b"rho")?;

    if self.poly.degree() != 4 {
      return Err(SpartanError::ProofVerifyError {
        reason: "NeutronNovaSNARK poly must be of degree 4".to_string(),
      });
    }

    // absorb poly in the RO
    transcript.absorb(b"poly", &self.poly);

    // squeeze a challenge
    let r_b = transcript.squeeze(b"r_b")?;

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap(); // TODO: remove unwrap

    let folded_U = U1.fold(U2, &r_b)?;

    // check the satisfiability of folded instance using the provided witness
    is_sat_with_target(&vk.ck, &vk.S, &folded_U, &self.folded_W, &E, T_out)?;

    Ok(())
  }
}

/// Check if the folded witness satisfies the folded instance
fn is_sat_with_target<E: Engine>(
  ck: &CommitmentKey<E>,
  S: &SplitR1CSShape<E>,
  U: &R1CSInstance<E>,
  W: &R1CSWitness<E>,
  E: &[E::Scalar],
  T: E::Scalar,
) -> Result<(), SpartanError> {
  let (_ell, left, right) = compute_tensor_decomp(S.num_cons);

  let z = [W.W.clone(), vec![E::Scalar::ONE], U.X.clone()].concat();
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

  if sum != T {
    println!("sum: {sum:?}");
    println!("U.T: {T:?}");
    return Err(SpartanError::UnSat {
      reason: "sum != U.T".to_string(),
    });
  }

  // check the validity of the commitments
  let comm_W = E::PCS::commit(ck, &W.W, &W.r_W, W.is_small)?;

  if comm_W != U.comm_W {
    return Err(SpartanError::UnSat {
      reason: "comm_W != U.comm_W".to_string(),
    });
  }

  Ok(())
}

#[cfg(test)]
mod benchmarks {
  use super::*;
  use crate::{
    bellpepper::{
      solver::SatisfyingAssignment,
      test_r1cs::{TestSpartanShape, TestSpartanWitness},
      test_shape_cs::TestShapeCS,
    },
    provider::T256HyraxEngine,
    r1cs::R1CSShape,
  };
  use bellpepper::gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    sha256::sha256,
  };
  use bellpepper_core::{ConstraintSystem, SynthesisError};
  use core::marker::PhantomData;
  use criterion::Criterion;
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

    pub fn num_challenges(&self) -> usize {
      0 // Placeholder, we don't use challenges in this example
    }

    pub fn synthesize<CS: ConstraintSystem<E::Scalar>>(
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

  fn generarate_sha_r1cs<E: Engine>(
    len: usize,
  ) -> (
    NeutronNovaProverKey<E>,
    NeutronNovaVerifierKey<E>,
    Vec<R1CSInstance<E>>,
    Vec<R1CSWitness<E>>,
  ) {
    let circuit = Sha256Circuit::<E> {
      preimage: vec![0u8; len],
      _p: Default::default(),
    };

    let (pk, vk) = NeutronNovaSNARK::<E>::setup(&[circuit.clone()]).unwrap();

    let mut cs: TestShapeCS<E> = TestShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (S, ck, _vk) = cs.r1cs_shape().unwrap();
    let S = S.pad();

    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U1, W1) = cs.r1cs_instance_and_witness(&S, &ck, true).unwrap();

    let circuit2 = Sha256Circuit::<E> {
      preimage: vec![1u8; len],
      _p: Default::default(),
    };
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit2.synthesize(&mut cs);
    let (U2, W2) = cs.r1cs_instance_and_witness(&S, &ck, true).unwrap();

    let instances = vec![U1, U2];
    let witnesses = vec![W1, W2];

    (ck, S, instances, witnesses)
  }

  fn bench_neutron_inner<E: Engine>(
    c: &mut Criterion,
    name: &str,
    pk: &NeutronNovaProverKey<E>,
    vk: &NeutronNovaVerifierKey<E>,
    instances: &[R1CSInstance<E>],
    witnesses: &[R1CSWitness<E>],
  ) where
    E::PCS: FoldingEngineTrait<E>,
  {
    // sanity check: prove and verify before benching
    let res = NeutronNovaSNARK::prove(pk, instances, witnesses);
    assert!(res.is_ok());

    let snark = res.unwrap();
    let res = snark.verify(vk, instances);
    assert!(res.is_ok());

    c.bench_function(&format!("neutron_snark_{name}"), |b| {
      b.iter(|| {
        let res = NeutronNovaSNARK::prove(pk, instances, witnesses);
        assert!(res.is_ok());
      })
    });
  }

  #[test]
  fn bench_neutron_sha256() {
    type E = T256HyraxEngine;

    let mut criterion = Criterion::default();
    for len in [64, 128].iter() {
      let (ck, S, instances, witnesses) = generarate_sha_r1cs::<E>(*len);
      bench_neutron_inner(
        &mut criterion,
        format!("sha256_{len}", &ck, &S, &instances, &witnesses),
      );
    }
  }
}
