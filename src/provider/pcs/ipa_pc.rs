//! This module provides an implementation of a `PCS` using an IPA-based polynomial commitment scheme
use crate::{
  errors::SpartanError,
  polys::eq::EqPolynomial,
  provider::{
    pcs::ipa::{InnerProductArgument, InnerProductInstance, InnerProductWitness, inner_product},
    traits::{DlogGroup, DlogGroupExt},
  },
  start_span,
  traits::{
    Engine,
    pcs::{CommitmentTrait, PCSEngineTrait},
    transcript::TranscriptReprTrait,
  },
};
use core::{fmt::Debug, marker::PhantomData};
use ff::{Field, PrimeField};
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{info, info_span};

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

  fn blind(_: &Self::CommitmentKey) -> Self::Blind {
    E::Scalar::random(&mut OsRng)
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    r: &Self::Blind,
    is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    if ck.ck.len() < v.len() {
      return Err(SpartanError::InvalidCommitmentKeyLength);
    }

    let comm = if !is_small {
      E::GE::vartime_multiscalar_mul(v, &ck.ck[..v.len()], true)?
        + <E::GE as DlogGroup>::group(&ck.h) * r
    } else {
      let scalars_small = v
        .par_iter()
        .map(|s| s.to_repr().as_ref()[0] as u64)
        .collect::<Vec<_>>();
      E::GE::vartime_multiscalar_mul_small(&scalars_small, &ck.ck[..scalars_small.len()], false)?
        + <E::GE as DlogGroup>::group(&ck.h) * r
    };

    Ok(Commitment { comm })
  }

  fn prove(
    ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[E::Scalar],
    blind: &E::Scalar,
    point: &[E::Scalar],
  ) -> Result<(E::Scalar, Self::EvaluationArgument), SpartanError> {
    let (_prep_span, prep_t) = start_span!("ipa_prove_prepare");
    let b_vec = EqPolynomial::new(point.to_vec()).evals();
    let eval = inner_product(poly, &b_vec);

    let u = InnerProductInstance::new(&comm.comm, &b_vec, &eval);
    let w = InnerProductWitness::new(poly, blind);
    info!(elapsed_ms = %prep_t.elapsed().as_millis(), "ipa_prove_prepare");

    let (_prove_span, prove_t) = start_span!("ipa_prove_argument");
    let result = InnerProductArgument::prove(&ck.ck, &ck.ck_s, &u, &w, transcript)?;
    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "ipa_prove_argument");

    Ok((eval, result))
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
    let (_verify_span, verify_t) = start_span!("ipa_pcs_verify");
    let u = InnerProductInstance::new(&comm.comm, &EqPolynomial::new(point.to_vec()).evals(), eval);

    arg.verify(
      &vk.ck,
      &vk.h,
      &vk.ck_s,
      (2_usize).pow(point.len() as u32),
      &u,
      transcript,
    )?;

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "ipa_pcs_verify");
    Ok(())
  }
}
