//! This module provides an implementation of a `PCS` using an IPA-based polynomial commitment scheme
use crate::{
  errors::SpartanError,
  polys::eq::EqPolynomial,
  provider::{
    pcs::ipa::{InnerProductArgument, InnerProductInstance, InnerProductWitness},
    traits::{DlogGroup, DlogGroupExt},
  },
  traits::{
    Engine,
    pcs::{CommitmentTrait, Len, PCSEngineTrait},
    transcript::TranscriptReprTrait,
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
  /// generator for the polynomial
  ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,

  /// generator for the blind term
  h: <E::GE as DlogGroup>::AffineGroupElement,

  /// generator for committing to evaluation
  ck_s: <E::GE as DlogGroup>::AffineGroupElement,
}

/// Provides an implementation of the verifier key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine>
where
  E::GE: DlogGroup,
{
  /// generator for the polynomial
  ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,

  /// generator for the blind term
  h: <E::GE as DlogGroup>::AffineGroupElement,

  /// generator for committing to evaluation
  ck_s: <E::GE as DlogGroup>::AffineGroupElement,
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
  comm: E::GE,
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
pub struct IPAPCS<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> PCSEngineTrait<E> for IPAPCS<E>
where
  E::GE: DlogGroupExt,
{
  type CommitmentKey = CommitmentKey<E>;
  type VerifierKey = VerifierKey<E>;
  type Commitment = Commitment<E>;
  type Blind = E::Scalar;
  type DerandKey = DerandKey<E>;
  type EvaluationArgument = InnerProductArgument<E>;

  fn setup(label: &'static [u8], n: usize) -> (Self::CommitmentKey, Self::VerifierKey) {
    let padded_n = n.next_power_of_two();
    let gens = E::GE::from_label(label, padded_n + 2);

    let ck = gens[..padded_n].to_vec();
    let h = gens[padded_n];
    let ck_s = gens[padded_n + 1];

    let vk = VerifierKey {
      ck: ck.clone(),
      h,
      ck_s,
    };

    let ck = CommitmentKey { ck, h, ck_s };

    (ck, vk)
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

  fn prove(
    ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[E::Scalar],
    point: &[E::Scalar],
    eval: &E::Scalar,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    let u = InnerProductInstance::new(&comm.comm, &EqPolynomial::new(point.to_vec()).evals(), eval);
    let w = InnerProductWitness::new(poly);

    InnerProductArgument::prove(&ck.ck, &ck.ck_s, &u, &w, transcript)
  }

  /// A method to verify purported evaluations of a committed polynomial
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    let u = InnerProductInstance::new(&comm.comm, &EqPolynomial::new(point.to_vec()).evals(), eval);

    arg.verify(
      &vk.ck,
      &vk.ck_s,
      (2_usize).pow(point.len() as u32),
      &u,
      transcript,
    )?;

    Ok(())
  }
}
