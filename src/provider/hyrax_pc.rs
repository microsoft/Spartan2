//! This module implements the Hyrax polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
#![allow(unused_imports)]
use crate::{
  Blind, Commitment, CommitmentKey,
  errors::SpartanError,
  math::Math,
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  provider::{
    ipa_pc::{InnerProductArgument, InnerProductInstance, InnerProductWitness},
    pedersen::{
      Commitment as PedersenCommitment, CommitmentEngine as PedersenCommitmentEngine,
      CommitmentKey as PedersenCommitmentKey, DerandKey as PedersenDerandKey,
    },
    traits::{DlogGroup, DlogGroupExt},
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
use ff::Field;
use itertools::{
  EitherOrBoth::{Both, Left, Right},
  Itertools,
};
use num_integer::Integer;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that holds commitment generators for Hyrax commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCommitmentKey<E: Engine>
where
  E::GE: DlogGroup,
{
  num_rows: usize,
  num_cols: usize,
  ck: PedersenCommitmentKey<E>,
}

/// Implements derandomization key for Hyrax commitment key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxDerandKey<E: Engine>
where
  E::GE: DlogGroup,
{
  dk: PedersenDerandKey<E>,
}

/// Structure that holds commitments
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCommitment<E: Engine> {
  comm: Vec<PedersenCommitment<E>>,
}

/// Structure that holds blinds
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxBlind<E: Engine> {
  blind: Option<Vec<E::Scalar>>,
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
    self.num_rows * self.num_cols
  }
}

impl<E: Engine> Default for HyraxBlind<E> {
  fn default() -> Self {
    HyraxBlind { blind: None }
  }
}

fn compute_factored_lens(n: usize) -> (usize, usize) {
  let ell = n.log_2();
  // we split ell into ell1 and ell2 such that ell1 + ell2 = ell and ell1 >= ell2
  let ell1 = (ell + 1) / 2; // This ensures ell1 >= ell2
  let ell2 = ell / 2;
  (ell1, ell2)
}

impl<E: Engine> CommitmentEngineTrait<E> for HyraxCommitmentEngine<E>
where
  E::GE: DlogGroupExt,
{
  type CommitmentKey = HyraxCommitmentKey<E>;
  type DerandKey = HyraxDerandKey<E>;
  type Commitment = HyraxCommitment<E>;
  type Blind = HyraxBlind<E>;

  /// Derives generators for Hyrax PC, where num_vars is the number of variables in multilinear poly
  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    let n = n.next_power_of_two();
    let (num_rows, num_cols) = compute_factored_lens(n);
    let ck = PedersenCommitmentEngine::setup(label, num_cols);
    HyraxCommitmentKey {
      num_rows,
      num_cols,
      ck,
    }
  }

  fn derand_key(ck: &Self::CommitmentKey) -> Self::DerandKey {
    HyraxDerandKey {
      dk: PedersenCommitmentEngine::derand_key(&ck.ck),
    }
  }

  fn blind(ck: &Self::CommitmentKey) -> Self::Blind {
    // TODO: make Pedersen blind also an option type
    HyraxBlind {
      blind: Some(
        (0..ck.num_rows)
          .map(|_| PedersenCommitmentEngine::blind(&ck.ck))
          .collect::<Vec<E::Scalar>>(),
      ),
    }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar], r: &Self::Blind) -> Self::Commitment {
    let mut v = v.to_vec();
    // pad with zeros
    if v.len() != ck.num_rows * ck.num_cols {
      v.extend(vec![E::Scalar::ZERO; ck.num_rows * ck.num_cols - v.len()]);
    }

    let r = if r.blind.is_none() {
      vec![E::Scalar::ZERO; ck.num_rows]
    } else {
      r.blind.clone().unwrap()
    };

    let comm = (0..ck.num_rows)
      .collect::<Vec<usize>>()
      .into_par_iter()
      .map(|i| {
        PedersenCommitmentEngine::commit(&ck.ck, &v[ck.num_cols * i..ck.num_cols * (i + 1)], &r[i])
      })
      .collect();

    HyraxCommitment { comm }
  }

  fn commit_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    ck: &Self::CommitmentKey,
    v: &[T],
    r: &Self::Blind,
  ) -> Self::Commitment {
    let mut v = v.to_vec();
    // pad with zeros
    if v.len() != ck.num_rows * ck.num_cols {
      v.extend(vec![T::zero(); ck.num_rows * ck.num_cols - v.len()]);
    }

    let r = if r.blind.is_none() {
      vec![E::Scalar::ZERO; ck.num_rows]
    } else {
      r.blind.clone().unwrap()
    };

    let comm = (0..ck.num_rows)
      .collect::<Vec<usize>>()
      .into_par_iter()
      .map(|i| {
        PedersenCommitmentEngine::commit_small(
          &ck.ck,
          &v[ck.num_cols * i..ck.num_cols * (i + 1)],
          &r[i],
        )
      })
      .collect();

    HyraxCommitment { comm }
  }

  fn derandomize(
    dk: &Self::DerandKey,
    comm: &Self::Commitment,
    r: &Self::Blind,
  ) -> Self::Commitment {
    if r.blind.is_none() {
      comm.clone()
    } else {
      let r = r.blind.clone().unwrap();
      HyraxCommitment {
        comm: (0..comm.comm.len())
          .map(|i| PedersenCommitmentEngine::derandomize(&dk.dk, &comm.comm[i], &r[i]))
          .collect(),
      }
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for HyraxCommitment<E>
where
  E::GE: DlogGroup,
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

impl<E: Engine> CommitmentTrait<E> for HyraxCommitment<E> where E::GE: DlogGroup {}

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
