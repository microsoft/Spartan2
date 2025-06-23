//! This module provides an implementation of a `CommitmentEngine` and an `EvaluationEngine` using an IPA-based polynomial commitment scheme
use crate::{
  errors::SpartanError,
  polys::eq::EqPolynomial,
  provider::{
    pcs::ipa::{InnerProductArgument, InnerProductInstance, InnerProductWitness},
    traits::{DlogGroup, DlogGroupExt},
  },
  traits::{
    Engine, TranscriptReprTrait,
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    evaluation::EvaluationEngineTrait,
  },
};
use core::{fmt::Debug, marker::PhantomData};
use ff::Field;
use num_integer::Integer;
use num_traits::ToPrimitive;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

/// A type that holds commitment generators
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentKey<E: Engine>
where
  E::GE: DlogGroup,
{
  pub(crate) ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  h: <E::GE as DlogGroup>::AffineGroupElement,
}

impl<E: Engine> Len for CommitmentKey<E>
where
  E::GE: DlogGroup,
{
  fn length(&self) -> usize {
    self.ck.len()
  }
}

/// A type that holds blinding generator
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerandKey<E: Engine>
where
  E::GE: DlogGroup,
{
  h: <E::GE as DlogGroup>::AffineGroupElement,
}

/// A type that holds a commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Commitment<E: Engine> {
  pub(crate) comm: E::GE,
}

impl<E: Engine> CommitmentTrait<E> for Commitment<E> where E::GE: DlogGroup {}

impl<E: Engine> Default for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn default() -> Self {
    Commitment {
      comm: E::GE::zero(),
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.comm.to_transcript_bytes()
  }
}

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> CommitmentEngineTrait<E> for CommitmentEngine<E>
where
  E::GE: DlogGroupExt,
{
  type CommitmentKey = CommitmentKey<E>;
  type Commitment = Commitment<E>;
  type Blind = E::Scalar;
  type DerandKey = DerandKey<E>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    let gens = E::GE::from_label(label, n.next_power_of_two() + 1);

    let (h, ck) = gens.split_first().unwrap();

    Self::CommitmentKey {
      ck: ck.to_vec(),
      h: *h,
    }
  }

  fn derand_key(ck: &Self::CommitmentKey) -> Self::DerandKey {
    Self::DerandKey { h: ck.h }
  }

  fn blind(_: &Self::CommitmentKey) -> Self::Blind {
    E::Scalar::random(&mut OsRng)
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar], r: &Self::Blind) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());

    Commitment {
      comm: E::GE::vartime_multiscalar_mul(v, &ck.ck[..v.len()])
        + <E::GE as DlogGroup>::group(&ck.h) * r,
    }
  }

  fn commit_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    ck: &Self::CommitmentKey,
    v: &[T],
    r: &Self::Blind,
  ) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());

    Commitment {
      comm: E::GE::vartime_multiscalar_mul_small(v, &ck.ck[..v.len()])
        + <E::GE as DlogGroup>::group(&ck.h) * r,
    }
  }

  fn derandomize(
    dk: &Self::DerandKey,
    commit: &Self::Commitment,
    r: &Self::Blind,
  ) -> Self::Commitment {
    Commitment {
      comm: commit.comm - <E::GE as DlogGroup>::group(&dk.h) * r,
    }
  }
}

/// Provides an implementation of the prover key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine>
where
  E::GE: DlogGroup,
{
  ck_s: CommitmentKey<E>,
}

/// Provides an implementation of the verifier key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine>
where
  E::GE: DlogGroup,
{
  ck_v: CommitmentKey<E>,
  ck_s: CommitmentKey<E>,
}

/// Provides an implementation of a polynomial evaluation engine using IPA
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E> EvaluationEngineTrait<E> for EvaluationEngine<E>
where
  E: Engine<CE = CommitmentEngine<E>>,
  E::GE: DlogGroupExt,
{
  type ProverKey = ProverKey<E>;
  type VerifierKey = VerifierKey<E>;
  type EvaluationArgument = InnerProductArgument<E>;

  fn setup(
    ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey) {
    let ck_s = CommitmentEngine::setup(b"ipa", 1);

    let pk = ProverKey { ck_s: ck_s.clone() };
    let vk = VerifierKey {
      ck_v: ck.clone(),
      ck_s,
    };

    (pk, vk)
  }

  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    transcript: &mut E::TE,
    comm: &Commitment<E>,
    poly: &[E::Scalar],
    point: &[E::Scalar],
    eval: &E::Scalar,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    let u = InnerProductInstance::new(&comm.comm, &EqPolynomial::new(point.to_vec()).evals(), eval);
    let w = InnerProductWitness::new(poly);

    InnerProductArgument::prove(&ck.ck, &pk.ck_s.ck[0], &u, &w, transcript)
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &Commitment<E>,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    let u = InnerProductInstance::new(&comm.comm, &EqPolynomial::new(point.to_vec()).evals(), eval);

    arg.verify(
      &vk.ck_v.ck,
      &vk.ck_s.ck[0],
      (2_usize).pow(point.len() as u32),
      &u,
      transcript,
    )?;

    Ok(())
  }
}
