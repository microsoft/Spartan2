//! This module implements the Hyrax polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
#![allow(unused_imports)]
use crate::{
  Commitment, CommitmentKey,
  errors::SpartanError,
  math::Math,
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  provider::{
    ipa_pc::{InnerProductArgument, InnerProductInstance, InnerProductWitness},
    pedersen::{
      Commitment as PedersenCommitment, CommitmentEngine as PedersenCommitmentEngine,
      CommitmentKey as PedersenCommitmentKey,
    },
    traits::DlogGroup,
  },
  traits::{
    Engine, Group, TranscriptEngineTrait, TranscriptReprTrait,
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    evaluation::EvaluationEngineTrait,
  },
};
use core::{
  marker::PhantomData,
  ops::{Add, AddAssign, Mul, MulAssign},
};
use itertools::{
  EitherOrBoth::{Both, Left, Right},
  Itertools,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that holds commitment generators for Hyrax commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCommitmentKey<E: Engine>
where
  E::GE: DlogGroup,
{
  ck: PedersenCommitmentKey<E>,
}

/// Structure that holds commitments
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCommitment<E: Engine> {
  comm: Vec<PedersenCommitment<E>>,
}

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyraxCommitmentEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> Len for HyraxCommitmentKey<E>
where
  E::GE: DlogGroup,
{
  fn length(&self) -> usize {
    self.ck.len() * self.ck.len() // we can commit to these many elements
  }
}

impl<E: Engine> CommitmentEngineTrait<E> for HyraxCommitmentEngine<E>
where
  E::GE: DlogGroup,
{
  type CommitmentKey = HyraxCommitmentKey<E>;
  type Commitment = HyraxCommitment<E>;

  /// Derives generators for Hyrax PC, where num_vars is the number of variables in multilinear poly
  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    let num_vars = n.next_power_of_two().log_2();
    let (_left, right) = EqPolynomial::<E::Scalar>::compute_factored_lens(num_vars);
    let ck = PedersenCommitmentEngine::setup(label, (2usize).pow(right as u32));
    HyraxCommitmentKey { ck }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar]) -> Self::Commitment {
    let poly = MultilinearPolynomial::new(v.to_vec());
    let n = poly.len();
    let ell = poly.get_num_vars();
    assert_eq!(n, (2usize).pow(ell as u32));

    let (left_num_vars, right_num_vars) = EqPolynomial::<E::Scalar>::compute_factored_lens(ell);
    let L_size = (2usize).pow(left_num_vars as u32);
    let R_size = (2usize).pow(right_num_vars as u32);
    assert_eq!(L_size * R_size, n);

    let comm = (0..L_size)
      .collect::<Vec<usize>>()
      .into_par_iter()
      .map(|i| {
        PedersenCommitmentEngine::commit(&ck.ck, &poly.get_Z()[R_size * i..R_size * (i + 1)])
      })
      .collect();

    HyraxCommitment {
      comm,
      is_default: false,
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for HyraxCommitment<E> 
  where E::GE: DlogGroup
{
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

impl<E: Engine> CommitmentTrait<E> for HyraxCommitment<E>
where
  E::GE: DlogGroup,
{
}


/*
/// Provides an implementation of the hyrax key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxProverKey<E: Engine> {
  ck_s: CommitmentKey<E>,
}

/// Provides an implementation of the hyrax key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxVerifierKey<E: Engine> {
  ck_v: CommitmentKey<E>,
  ck_s: CommitmentKey<E>,
}

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxEvaluationArgument<E: Engine> {
  ipa: InnerProductArgument<E>,
}

/// Provides an implementation of a polynomial evaluation engine using Hyrax PC
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxEvaluationEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E> EvaluationEngineTrait<E> for HyraxEvaluationEngine<E>
where
  E: Engine<CE = HyraxCommitmentEngine<E>>,
{
  type CE = G::CE;
  type ProverKey = HyraxProverKey<E>;
  type VerifierKey = HyraxVerifierKey<E>;
  type EvaluationArgument = HyraxEvaluationArgument<E>;

  fn setup(
    ck: &<<G as Group>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey) {
    let pk = HyraxProverKey::<E> {
      ck_s: G::CE::setup(b"hyrax", 1),
    };

    let vk = HyraxVerifierKey::<E> {
      ck_v: ck.clone(),
      ck_s: G::CE::setup(b"hyrax", 1),
    };

    (pk, vk)
  }

  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    transcript: &mut G::TE,
    comm: &Commitment<E>,
    poly: &[G::Scalar],
    point: &[G::Scalar],
    eval: &G::Scalar,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    transcript.absorb(b"poly_com", comm);

    let poly_m = MultilinearPolynomial::<G::Scalar>::new(poly.to_vec());

    // assert vectors are of the right size
    assert_eq!(poly_m.get_num_vars(), point.len());

    let (left_num_vars, right_num_vars) =
      EqPolynomial::<G::Scalar>::compute_factored_lens(point.len());
    let L_size = (2usize).pow(left_num_vars as u32);
    let R_size = (2usize).pow(right_num_vars as u32);

    // compute the L and R vectors (these depend only on the public challenge point so they are public)
    let eq = EqPolynomial::new(point.to_vec());
    let (L, R) = eq.compute_factored_evals();

    assert_eq!(L.len(), L_size);
    assert_eq!(R.len(), R_size);

    // compute the vector underneath L*Z
    // compute vector-matrix product between L and Z viewed as a matrix
    let LZ = poly_m.bound(&L);

    // Commit to LZ
    let com_LZ = PedersenCommitmentEngine::commit(&ck.ck, &LZ);

    // a dot product argument (IPA) of size R_size
    let ipa_instance = InnerProductInstance::<E>::new(&com_LZ, &R, eval);
    let ipa_witness = InnerProductWitness::<E>::new(&LZ);
    let ipa = InnerProductArgument::<E>::prove(
      &ck.ck,
      &pk.ck_s.ck,
      &ipa_instance,
      &ipa_witness,
      transcript,
    )?;

    Ok(HyraxEvaluationArgument { ipa })
  }

  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut G::TE,
    comm: &Commitment<E>,
    point: &[G::Scalar],
    eval: &G::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    transcript.absorb(b"poly_com", comm);

    // compute L and R
    let eq = EqPolynomial::new(point.to_vec());
    let (L, R) = eq.compute_factored_evals();

    // compute a weighted sum of commitments and L
    let ck = PedersenCommitmentEngine::reinterpret_commitments_as_ck(&comm.comm);

    let com_LZ = PedersenCommitmentEngine::commit(&ck, &L); // computes MSM of commitment and L

    let ipa_instance = InnerProductInstance::<E>::new(&com_LZ, &R, eval);

    arg
      .ipa
      .verify(&vk.ck_v.ck, &vk.ck_s.ck, R.len(), &ipa_instance, transcript)
  }
}
*/
