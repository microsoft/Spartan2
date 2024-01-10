//! This module implements `EvaluationEngine` using an IPA-based polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
use crate::{
  errors::SpartanError,
  provider::pedersen::{
    Commitment as PedersenCommitment, CommitmentEngine as PedersenCommitmentEngine,
    CommitmentEngineExtTrait, CommitmentKey as PedersenCommitmentKey,
    CompressedCommitment as PedersenCompressedCommitment,
  },
  spartan::polys::eq::EqPolynomial,
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    evaluation::EvaluationEngineTrait,
    Group, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey,
};
use core::iter;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Provides an implementation of the prover key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<G: Group> {
  ck_s: CommitmentKey<G>,
}

/// Provides an implementation of the verifier key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<G: Group> {
  ck_v: CommitmentKey<G>,
  ck_s: CommitmentKey<G>,
}

/// Provides an implementation of a polynomial evaluation engine using IPA
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<G: Group> {
  _p: PhantomData<G>,
}

impl<G> EvaluationEngineTrait<G> for EvaluationEngine<G>
where
  G: Group<CE = PedersenCommitmentEngine<G>>,
{
  type CE = G::CE;
  type ProverKey = ProverKey<G>;
  type VerifierKey = VerifierKey<G>;
  type EvaluationArgument = InnerProductArgument<G>;

  fn setup(
    ck: &<<G as Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey) {
    let ck_c = G::CE::setup(b"ipa", 1);

    let pk = ProverKey { ck_s: ck_c.clone() };
    let vk = VerifierKey {
      ck_v: ck.clone(),
      ck_s: ck_c,
    };

    (pk, vk)
  }

  fn prove(
    ck: &CommitmentKey<G>,
    pk: &Self::ProverKey,
    transcript: &mut G::TE,
    comm: &Commitment<G>,
    poly: &[G::Scalar],
    point: &[G::Scalar],
    eval: &G::Scalar,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    let u = InnerProductInstance::new(comm, &EqPolynomial::new(point.to_vec()).evals(), eval);
    let w = InnerProductWitness::new(poly);

    InnerProductArgument::prove(ck, &pk.ck_s, &u, &w, transcript)
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut G::TE,
    comm: &Commitment<G>,
    point: &[G::Scalar],
    eval: &G::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    let u = InnerProductInstance::new(comm, &EqPolynomial::new(point.to_vec()).evals(), eval);

    arg.verify(
      &vk.ck_v,
      &vk.ck_s,
      (2_usize).pow(point.len() as u32),
      &u,
      transcript,
    )?;

    Ok(())
  }
}

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

/// An inner product instance consists of a commitment to a vector `a` and another vector `b`
/// and the claim that c = <a, b>.
pub struct InnerProductInstance<G: Group> {
  comm_a_vec: PedersenCommitment<G>,
  b_vec: Vec<G::Scalar>,
  c: G::Scalar,
}

impl<G: Group> InnerProductInstance<G> {
  /// Creates a new inner product instance
  pub fn new(comm_a_vec: &PedersenCommitment<G>, b_vec: &[G::Scalar], c: &G::Scalar) -> Self {
    InnerProductInstance {
      comm_a_vec: *comm_a_vec,
      b_vec: b_vec.to_vec(),
      c: *c,
    }
  }
}

impl<G: Group> TranscriptReprTrait<G> for InnerProductInstance<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // we do not need to include self.b_vec as in our context it is produced from the transcript
    [
      self.comm_a_vec.to_transcript_bytes(),
      self.c.to_transcript_bytes(),
    ]
    .concat()
  }
}

/// An inner product witness consists the vector `a`.
pub struct InnerProductWitness<G: Group> {
  a_vec: Vec<G::Scalar>,
}

impl<G: Group> InnerProductWitness<G> {
  /// Creates a new inner product witness
  pub fn new(a_vec: &[G::Scalar]) -> Self {
    InnerProductWitness {
      a_vec: a_vec.to_vec(),
    }
  }
}

/// An inner product argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InnerProductArgument<G: Group> {
  L_vec: Vec<PedersenCompressedCommitment<G>>,
  R_vec: Vec<PedersenCompressedCommitment<G>>,
  a_hat: G::Scalar,
}

impl<G: Group> InnerProductArgument<G> {
  const fn protocol_name() -> &'static [u8] {
    b"IPA"
  }

  /// Proves an inner product relationship
  pub fn prove(
    ck: &PedersenCommitmentKey<G>,
    ck_c: &PedersenCommitmentKey<G>,
    U: &InnerProductInstance<G>,
    W: &InnerProductWitness<G>,
    transcript: &mut G::TE,
  ) -> Result<Self, SpartanError> {
    transcript.dom_sep(Self::protocol_name());

    let (ck, _) = PedersenCommitmentEngine::<G>::split_at(ck, U.b_vec.len());

    if U.b_vec.len() != W.a_vec.len() {
      return Err(SpartanError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // sample a random base for commiting to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = PedersenCommitmentEngine::<G>::scale(ck_c, &r);

    // a closure that executes a step of the recursive inner product argument
    let prove_inner = |a_vec: &[G::Scalar],
                       b_vec: &[G::Scalar],
                       ck: &PedersenCommitmentKey<G>,
                       transcript: &mut G::TE|
     -> Result<
      (
        PedersenCompressedCommitment<G>,
        PedersenCompressedCommitment<G>,
        Vec<G::Scalar>,
        Vec<G::Scalar>,
        PedersenCommitmentKey<G>,
      ),
      SpartanError,
    > {
      let n = a_vec.len();
      let (ck_L, ck_R) = PedersenCommitmentEngine::split_at(ck, n / 2);

      let c_L = inner_product(&a_vec[0..n / 2], &b_vec[n / 2..n]);
      let c_R = inner_product(&a_vec[n / 2..n], &b_vec[0..n / 2]);

      let L = PedersenCommitmentEngine::commit(
        &PedersenCommitmentEngine::combine(&ck_R, &ck_c),
        &a_vec[0..n / 2]
          .iter()
          .chain(iter::once(&c_L))
          .copied()
          .collect::<Vec<G::Scalar>>(),
      )
      .compress();
      let R = PedersenCommitmentEngine::commit(
        &PedersenCommitmentEngine::combine(&ck_L, &ck_c),
        &a_vec[n / 2..n]
          .iter()
          .chain(iter::once(&c_R))
          .copied()
          .collect::<Vec<G::Scalar>>(),
      )
      .compress();

      transcript.absorb(b"L", &L);
      transcript.absorb(b"R", &R);

      let r = transcript.squeeze(b"r")?;
      let r_inverse = r.invert().unwrap();

      // fold the left half and the right half
      let a_vec_folded = a_vec[0..n / 2]
        .par_iter()
        .zip(a_vec[n / 2..n].par_iter())
        .map(|(a_L, a_R)| *a_L * r + r_inverse * *a_R)
        .collect::<Vec<G::Scalar>>();

      let b_vec_folded = b_vec[0..n / 2]
        .par_iter()
        .zip(b_vec[n / 2..n].par_iter())
        .map(|(b_L, b_R)| *b_L * r_inverse + r * *b_R)
        .collect::<Vec<G::Scalar>>();

      let ck_folded = PedersenCommitmentEngine::fold(ck, &r_inverse, &r);

      Ok((L, R, a_vec_folded, b_vec_folded, ck_folded))
    };

    // two vectors to hold the logarithmic number of group elements
    let mut L_vec: Vec<PedersenCompressedCommitment<G>> = Vec::new();
    let mut R_vec: Vec<PedersenCompressedCommitment<G>> = Vec::new();

    // we create mutable copies of vectors and generators
    let mut a_vec = W.a_vec.to_vec();
    let mut b_vec = U.b_vec.to_vec();
    let mut ck = ck;
    for _i in 0..usize::try_from(U.b_vec.len().ilog2()).unwrap() {
      let (L, R, a_vec_folded, b_vec_folded, ck_folded) =
        prove_inner(&a_vec, &b_vec, &ck, transcript)?;
      L_vec.push(L);
      R_vec.push(R);

      a_vec = a_vec_folded;
      b_vec = b_vec_folded;
      ck = ck_folded;
    }

    Ok(InnerProductArgument {
      L_vec,
      R_vec,
      a_hat: a_vec[0],
    })
  }

  /// Verifies an inner product relationship
  pub fn verify(
    &self,
    ck: &PedersenCommitmentKey<G>,
    ck_c: &PedersenCommitmentKey<G>,
    n: usize,
    U: &InnerProductInstance<G>,
    transcript: &mut G::TE,
  ) -> Result<(), SpartanError> {
    let (ck, _) = PedersenCommitmentEngine::split_at(ck, U.b_vec.len());

    transcript.dom_sep(Self::protocol_name());
    if U.b_vec.len() != n
      || n != (1 << self.L_vec.len())
      || self.L_vec.len() != self.R_vec.len()
      || self.L_vec.len() >= 32
    {
      return Err(SpartanError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // sample a random base for commiting to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = PedersenCommitmentEngine::scale(ck_c, &r);

    let P = U.comm_a_vec + PedersenCommitmentEngine::<G>::commit(&ck_c, &[U.c]);

    let batch_invert = |v: &[G::Scalar]| -> Result<Vec<G::Scalar>, SpartanError> {
      let mut products = vec![G::Scalar::ZERO; v.len()];
      let mut acc = G::Scalar::ONE;

      for i in 0..v.len() {
        products[i] = acc;
        acc *= v[i];
      }

      // we can compute an inversion only if acc is non-zero
      if acc == G::Scalar::ZERO {
        return Err(SpartanError::InternalError);
      }

      // compute the inverse once for all entries
      acc = acc.invert().unwrap();

      let mut inv = vec![G::Scalar::ZERO; v.len()];
      for i in 0..v.len() {
        let tmp = acc * v[v.len() - 1 - i];
        inv[v.len() - 1 - i] = products[v.len() - 1 - i] * acc;
        acc = tmp;
      }

      Ok(inv)
    };

    // compute a vector of public coins using self.L_vec and self.R_vec
    let r = (0..self.L_vec.len())
      .map(|i| {
        transcript.absorb(b"L", &self.L_vec[i]);
        transcript.absorb(b"R", &self.R_vec[i]);
        transcript.squeeze(b"r")
      })
      .collect::<Result<Vec<G::Scalar>, SpartanError>>()?;

    // precompute scalars necessary for verification
    let r_square: Vec<G::Scalar> = (0..self.L_vec.len())
      .into_par_iter()
      .map(|i| r[i] * r[i])
      .collect();
    let r_inverse = batch_invert(&r)?;
    let r_inverse_square: Vec<G::Scalar> = (0..self.L_vec.len())
      .into_par_iter()
      .map(|i| r_inverse[i] * r_inverse[i])
      .collect();

    // compute the vector with the tensor structure
    let s = {
      let mut s = vec![G::Scalar::ZERO; n];
      s[0] = {
        let mut v = G::Scalar::ONE;
        for r_inverse_i in r_inverse {
          v *= r_inverse_i;
        }
        v
      };
      for i in 1..n {
        let pos_in_r = (31 - (i as u32).leading_zeros()) as usize;
        s[i] = s[i - (1 << pos_in_r)] * r_square[(self.L_vec.len() - 1) - pos_in_r];
      }
      s
    };

    let ck_hat = {
      let c = PedersenCommitmentEngine::<G>::commit(&ck, &s);
      PedersenCommitmentEngine::<G>::reinterpret_commitments_as_ck(&[c])
    };

    let b_hat = inner_product(&U.b_vec, &s);

    let P_hat = {
      let ck_folded = {
        let L_vec_decomp = self
          .L_vec
          .iter()
          .map(|L| PedersenCommitment::<G>::decompress(L))
          .collect::<Result<Vec<_>, _>>()?;
        let R_vec_decomp = self
          .R_vec
          .iter()
          .map(|R| PedersenCommitment::<G>::decompress(R))
          .collect::<Result<Vec<_>, _>>()?;

        let ck_L = PedersenCommitmentEngine::<G>::reinterpret_commitments_as_ck(&L_vec_decomp);
        let ck_R = PedersenCommitmentEngine::<G>::reinterpret_commitments_as_ck(&R_vec_decomp);
        let ck_P = PedersenCommitmentEngine::<G>::reinterpret_commitments_as_ck(&[P]);
        PedersenCommitmentEngine::combine(&PedersenCommitmentEngine::combine(&ck_L, &ck_R), &ck_P)
      };

      PedersenCommitmentEngine::<G>::commit(
        &ck_folded,
        &r_square
          .iter()
          .chain(r_inverse_square.iter())
          .chain(iter::once(&G::Scalar::ONE))
          .copied()
          .collect::<Vec<G::Scalar>>(),
      )
    };

    if P_hat
      == PedersenCommitmentEngine::<G>::commit(
        &PedersenCommitmentEngine::combine(&ck_hat, &ck_c),
        &[self.a_hat, self.a_hat * b_hat],
      )
    {
      Ok(())
    } else {
      Err(SpartanError::InvalidIPA)
    }
  }
}
