//! This module implements the Hyrax polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
use crate::{
  errors::SpartanError,
  provider::ipa_pc::{InnerProductArgument, InnerProductInstance, InnerProductWitness},
  provider::pedersen::{CommitmentKeyExtTrait,CompressedCommitment},
  spartan::polynomial::{EqPolynomial, MultilinearPolynomial},
  traits::{
    commitment::{
      CommitmentEngineTrait, CommitmentTrait
    },
    Group, TranscriptEngineTrait, TranscriptReprTrait,
  },
  CommitmentKey
};
use rayon::prelude::*;

/// Structure that holds Poly Commits
#[derive(Debug)]
pub struct PolyCommit<G: Group> {
  /// Commitment
  pub comm: Vec<CompressedCommitment<G>>,
}

/// Hyrax PC generators and functions to commit and prove evaluation
pub struct HyraxPC<G: Group> {
  gens_v: CommitmentKey<G>,  // generator for vectors
  gens_s: CommitmentKey<G>, // generator for scalars (eval)
}

impl<G: Group> TranscriptReprTrait<G> for PolyCommit<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut v = Vec::new();
    v.append(&mut b"poly_commitment_begin".to_vec());

    for c in &self.comm {
      v.append(&mut c.to_transcript_bytes());
    }

    v.append(&mut b"poly_commitment_end".to_vec());
    v
  }
}


impl<G: Group> HyraxPC<G>
where
  G: Group,
  CommitmentKey<G>: CommitmentKeyExtTrait<G, CE = G::CE>,
{
  /// Derives generators for Hyrax PC, where num_vars is the number of variables in multilinear poly
  pub fn new(num_vars: usize, label: &'static [u8]) -> Self {
    let (_left, right) = EqPolynomial::<G::Scalar>::compute_factored_lens(num_vars);
    let gens_v = G::CE::setup(label, (2usize).pow(right as u32));
    let gens_s = G::CE::setup(b"gens_s", 1);
    HyraxPC { gens_v, gens_s }
  }

  fn commit_inner(
    &self,
    poly: &MultilinearPolynomial<G::Scalar>,
    L_size: usize,
  ) -> PolyCommit<G> {
    let R_size = poly.len() / L_size;

    assert_eq!(L_size * R_size, poly.len());

    let comm = (0..L_size)
      .into_par_iter()
      .map(|i| {
        G::CE::commit(
          &self.gens_v,
          &poly.get_Z()[R_size * i..R_size * (i + 1)],
        )
        .compress()
      })
      .collect();

    PolyCommit { comm }
  }

  /// Commits to a multilinear polynomial and returns commitment and blind
  pub fn commit(
    &self,
    poly: &MultilinearPolynomial<G::Scalar>,
  ) -> PolyCommit<G> {
    let n = poly.len();
    let ell = poly.get_num_vars();
    assert_eq!(n, (2usize).pow(ell as u32));

    let (left_num_vars, right_num_vars) = EqPolynomial::<G::Scalar>::compute_factored_lens(ell);
    let L_size = (2usize).pow(left_num_vars as u32);
    let R_size = (2usize).pow(right_num_vars as u32);
    assert_eq!(L_size * R_size, n);

    self.commit_inner(poly, L_size)
  }

  /// Proves the evaluation of polynomial at a random point r
  pub fn prove_eval(
    &self,
    poly: &MultilinearPolynomial<G::Scalar>, // defined as vector Z
    poly_com: &PolyCommit<G>,
    r: &[G::Scalar],      // point at which the polynomial is evaluated
    Zr: &G::Scalar,       // evaluation of poly(r)
    transcript: &mut G::TE,
  ) -> Result<
    (
      InnerProductArgument<G>,
      InnerProductWitness<G>,
    ),
    SpartanError,
  > {
    transcript.absorb(b"poly_com", poly_com);

    // assert vectors are of the right size
    assert_eq!(poly.get_num_vars(), r.len());

    let (left_num_vars, right_num_vars) = EqPolynomial::<G::Scalar>::compute_factored_lens(r.len());
    let L_size = (2usize).pow(left_num_vars as u32);
    let R_size = (2usize).pow(right_num_vars as u32);

    // compute the L and R vectors (these depend only on the public challenge r so they are public)
    let eq = EqPolynomial::new(r.to_vec());
    let (L, R) = eq.compute_factored_evals();
    assert_eq!(L.len(), L_size);
    assert_eq!(R.len(), R_size);

    // compute the vector underneath L*Z
    // compute vector-matrix product between L and Z viewed as a matrix
    let LZ = poly.bound(&L);

    // Translation between this stuff and IPA
    // LZ = x_vec
    // LZ_blind = r_x
    // Zr = y
    // blind_Zr = r_y
    // R = a_vec

    // Commit to LZ and Zr
    let com_LZ = G::CE::commit(&self.gens_v, &LZ); 
    //let com_Zr = G::CE::commit(&self.gens_s, &[*Zr]);

    // a dot product argument (IPA) of size R_size
    let ipa_instance = InnerProductInstance::<G>::new(&com_LZ, &R, Zr);
    let ipa_witness = InnerProductWitness::<G>::new(&LZ);
    let ipa = InnerProductArgument::<G>::prove(
      &self.gens_v,
      &self.gens_s,
      &ipa_instance,
      &ipa_witness,
      transcript,
    )?;

    Ok((ipa, ipa_witness))
  }

  /// Verifies the proof showing the evaluation of a committed polynomial at a random point
  pub fn verify_eval(
    &self,
    r: &[G::Scalar], // point at which the polynomial was evaluated
    poly_com: &PolyCommit<G>,
    Zr: &G::Scalar, 
    ipa: &InnerProductArgument<G>,
    transcript: &mut G::TE,
  ) -> Result<(), SpartanError> {
    transcript.absorb(b"poly_com", poly_com);

    // compute L and R
    let eq = EqPolynomial::new(r.to_vec());
    let (L, R) = eq.compute_factored_evals();

    // compute a weighted sum of commitments and L
    let gens: CommitmentKey<G> = 
      CommitmentKey::<G>::reinterpret_commitments_as_ck(&poly_com.comm)?;

    let com_LZ = G::CE::commit(&gens, &L); // computes MSM of commitment and L

    let ipa_instance = InnerProductInstance::<G>::new(&com_LZ, &R, Zr);

    ipa.verify(
      &self.gens_v,
      &self.gens_s,
      L.len(),
      &ipa_instance,
      transcript,
    )
  }
}

#[cfg(test)]
mod tests {
  use ff::Field;
  use super::*;
  type G = pasta_curves::pallas::Point;
  use crate::traits::TranscriptEngineTrait;
  use rand::rngs::OsRng;

  fn inner_product<T>(a: &[T], b: &[T]) -> T
  where
    T: Field + Send + Sync,
  {
    assert_eq!(a.len(), b.len());
    (0..a.len())
      .into_par_iter()
      .map(|i| a[i] * b[i])
      .reduce(|| T::ZERO, |x, y| x + y)
  }

  fn evaluate_with_LR(
    Z: &[<G as Group>::Scalar],
    r: &[<G as Group>::Scalar],
  ) -> <G as Group>::Scalar {
    let eq = EqPolynomial::new(r.to_vec());
    let (L, R) = eq.compute_factored_evals();

    let ell = r.len();
    // ensure ell is even
    assert!(ell % 2 == 0);

    // compute n = 2^\ell
    let n = (2usize).pow(ell as u32);

    // compute m = sqrt(n) = 2^{\ell/2}
    let m = (n as f64).sqrt() as usize;

    // compute vector-matrix product between L and Z viewed as a matrix
    let LZ = (0..m)
      .map(|i| {
        (0..m)
          .map(|j| L[j] * Z[j * m + i])
          .fold(<G as Group>::Scalar::ZERO, |acc, item| acc + item)
      })
      .collect::<Vec<<G as Group>::Scalar>>();

    // compute dot product between LZ and R
    inner_product(&LZ, &R)
  }

  fn to_scalar(x: usize) -> <G as Group>::Scalar {
    (0..x)
      .map(|_i| <G as Group>::Scalar::ONE)
      .fold(<G as Group>::Scalar::ZERO, |acc, item| acc + item)
  }

  #[test]
  fn check_polynomial_evaluation() {
    // Z = [1, 2, 1, 4]
    let Z = vec![to_scalar(1), to_scalar(2), to_scalar(1), to_scalar(4)];

    // r = [4,3]
    let r = vec![to_scalar(4), to_scalar(3)];

    let eval_with_LR = evaluate_with_LR(&Z, &r);
    let poly = MultilinearPolynomial::new(Z);

    let eval = poly.evaluate(&r);
    assert_eq!(eval, to_scalar(28));
    assert_eq!(eval_with_LR, eval);
  }

  #[test]
  fn check_hyrax_pc_commit() {
    let Z = vec![to_scalar(1), to_scalar(2), to_scalar(1), to_scalar(4)];

    let poly = MultilinearPolynomial::new(Z);

    // Public stuff
    let num_vars = 2;
    assert_eq!(num_vars, poly.get_num_vars());
    let r = vec![to_scalar(4), to_scalar(3)]; // r = [4,3]

    // Prover actions
    let eval = poly.evaluate(&r);
    assert_eq!(eval, to_scalar(28));

    let prover_gens = HyraxPC::new(num_vars, b"poly_test");
    let poly_comm = prover_gens.commit(&poly);

    let mut prover_transcript = G::TE::new(b"example");

    let blind_eval = <G as Group>::Scalar::random(&mut OsRng);

    let (ipa_proof, _ipa_witness, comm_eval): (InnerProductArgument<G>, InnerProductWitness<G>, _) =
      prover_gens
        .prove_eval(
          &poly,
          &poly_comm,
          &r,
          &eval,
          &blind_eval,
          &mut prover_transcript,
        )
        .unwrap();

    // Verifier actions

    let verifier_gens = HyraxPC::new(num_vars, b"poly_test");
    let mut verifier_transcript = G::TE::new(b"example");

    let res = verifier_gens.verify_eval(
      &r,
      &poly_comm,
      &comm_eval,
      &ipa_proof,
      &mut verifier_transcript,
    );
    assert!(res.is_ok());
  }
}
