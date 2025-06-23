//! This module implements the Hyrax polynomial commitment scheme
#[allow(unused)]
use crate::{
  Blind, Commitment, CommitmentKey,
  errors::SpartanError,
  math::Math,
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  provider::{
    pcs::ipa::{InnerProductArgument, InnerProductInstance, InnerProductWitness},
    traits::{DlogGroup, DlogGroupExt},
  },
  traits::{
    Engine,
    pcs::{CommitmentTrait, Len, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::marker::PhantomData;
use ff::Field;
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
  ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  h: <E::GE as DlogGroup>::AffineGroupElement,
  ck_s: <E::GE as DlogGroup>::AffineGroupElement,
}

/// A type that holds the verifier key for Hyrax commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxVerifierKey<E: Engine>
where
  E::GE: DlogGroup,
{
  num_rows: usize,
  num_cols: usize,
  ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  h: <E::GE as DlogGroup>::AffineGroupElement,
  ck_s: <E::GE as DlogGroup>::AffineGroupElement,
}

impl<E: Engine> Len for HyraxCommitmentKey<E>
where
  E::GE: DlogGroup,
{
  fn length(&self) -> usize {
    self.num_rows * self.num_cols
  }
}

/// Implements derandomization key for Hyrax commitment key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxDerandKey<E: Engine>
where
  E::GE: DlogGroupExt,
{
  h: <E::GE as DlogGroup>::AffineGroupElement,
}

/// Structure that holds commitments
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCommitment<E: Engine> {
  comm: Vec<E::GE>,
}

/// Structure that holds blinds
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxBlind<E: Engine> {
  blind: Option<Vec<E::Scalar>>,
}

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyraxPCS<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> Default for HyraxBlind<E> {
  fn default() -> Self {
    HyraxBlind { blind: None }
  }
}

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxEvaluationArgument<E: Engine>
where
  E::GE: DlogGroup,
{
  ipa: InnerProductArgument<E>,
}

fn compute_factored_lens(n: usize) -> (usize, usize) {
  let ell = n.log_2();
  // we split ell into ell1 and ell2 such that ell1 + ell2 = ell and ell1 >= ell2
  let ell1 = ell.div_ceil(2); // This ensures ell1 >= ell2
  let ell2 = ell / 2;

  (1 << ell1, 1 << ell2)
}

impl<E: Engine> PCSEngineTrait<E> for HyraxPCS<E>
where
  E::GE: DlogGroupExt,
{
  type CommitmentKey = HyraxCommitmentKey<E>;
  type VerifierKey = HyraxVerifierKey<E>;
  type DerandKey = HyraxDerandKey<E>;
  type Commitment = HyraxCommitment<E>;
  type Blind = HyraxBlind<E>;
  type EvaluationArgument = HyraxEvaluationArgument<E>;

  /// Derives generators for Hyrax PC, where num_vars is the number of variables in multilinear poly
  fn setup(label: &'static [u8], n: usize) -> (Self::CommitmentKey, Self::VerifierKey) {
    let n = n.next_power_of_two();
    let (num_rows, num_cols) = compute_factored_lens(n);

    let gens = E::GE::from_label(label, num_cols + 2);
    let ck = gens[..num_cols].to_vec();
    let h = gens[num_cols];
    let ck_s = gens[num_cols + 1];

    let ck = Self::CommitmentKey {
      num_rows,
      num_cols,
      ck: ck.clone(),
      h: h.clone(),
      ck_s: ck_s.clone(),
    };

    let vk = Self::VerifierKey {
      num_rows,
      num_cols,
      ck: ck.ck.clone(),
      h: h.clone(),
      ck_s: ck_s.clone(),
    };

    (ck, vk)
  }

  fn derand_key(ck: &Self::CommitmentKey) -> Self::DerandKey {
    HyraxDerandKey { h: ck.h }
  }

  fn blind(ck: &Self::CommitmentKey) -> Self::Blind {
    HyraxBlind {
      blind: Some(
        (0..ck.num_rows)
          .map(|_| E::Scalar::ZERO)
          .collect::<Vec<E::Scalar>>(),
      ),
    }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar], r: &Self::Blind) -> Self::Commitment {
    if v.len() > ck.num_rows * ck.num_cols {
      panic!(
        "Input vector is too large {} and ck.num_rows = {}, ck.num_cols= {}",
        v.len(),
        ck.num_rows,
        ck.num_cols
      );
    }

    let mut v = v.to_vec();
    // pad with zeros
    if v.len() < ck.num_rows * ck.num_cols {
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
        E::GE::vartime_multiscalar_mul(
          &v[ck.num_cols * i..ck.num_cols * (i + 1)],
          &ck.ck[..ck.num_cols],
        ) + <E::GE as DlogGroup>::group(&ck.h) * r[i]
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
        E::GE::vartime_multiscalar_mul_small(
          &v[ck.num_cols * i..ck.num_cols * (i + 1)],
          &ck.ck[..ck.num_cols],
        ) + <E::GE as DlogGroup>::group(&ck.h) * r[i]
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
          .map(|i| comm.comm[i] - <E::GE as DlogGroup>::group(&dk.h) * r[i])
          .collect(),
      }
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
    if poly.len() != (2usize).pow(point.len() as u32) {
      return Err(SpartanError::InvalidInputLength);
    }

    transcript.absorb(b"poly_com", comm);

    let (num_rows, num_cols) = compute_factored_lens(poly.len());

    let (num_vars_rows, _) = (num_rows.log_2(), num_cols.log_2());

    let L = EqPolynomial::new(point[..num_vars_rows].to_vec()).evals();
    let R = EqPolynomial::new(point[num_vars_rows..].to_vec()).evals();

    let poly_m = MultilinearPolynomial::<E::Scalar>::new(poly.to_vec());

    // compute the vector underneath L*Z
    // compute vector-matrix product between L and Z viewed as a matrix
    let LZ = poly_m.bind(&L, &R);

    // Commit to LZ with a blind of zero
    let comm_LZ = E::GE::vartime_multiscalar_mul(&LZ, &ck.ck[..LZ.len()]);

    // a dot product argument (IPA) of size R_size
    let ipa_instance = InnerProductInstance::<E>::new(&comm_LZ, &R, eval);
    let ipa_witness = InnerProductWitness::<E>::new(&LZ);
    let ipa =
      InnerProductArgument::<E>::prove(&ck.ck, &ck.ck_s, &ipa_instance, &ipa_witness, transcript)?;

    Ok(HyraxEvaluationArgument { ipa })
  }

  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    transcript.absorb(b"poly_com", comm);

    // compute L and R
    // n = 2^point.len()
    let n = (2_usize).pow(point.len() as u32);
    let (num_rows, num_cols) = compute_factored_lens(n);

    let (num_vars_rows, _num_vars_cols) = (num_rows.log_2(), num_cols.log_2());

    let L = EqPolynomial::new(point[..num_vars_rows].to_vec()).evals();
    let R = EqPolynomial::new(point[num_vars_rows..].to_vec()).evals();

    // compute a weighted sum of commitments and L
    // convert the commitments to affine form so we can do a multi-scalar multiplication
    let ck: Vec<_> = comm.comm.iter().map(|c| c.affine()).collect();
    let comm_LZ = E::GE::vartime_multiscalar_mul(&L, &ck[..L.len()]);

    let ipa_instance = InnerProductInstance::<E>::new(&comm_LZ, &R, eval);

    arg
      .ipa
      .verify(&vk.ck, &vk.ck_s, R.len(), &ipa_instance, transcript)
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for HyraxCommitment<E>
where
  E::GE: DlogGroupExt,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut v = Vec::new();
    v.append(&mut b"poly_commitment_begin".to_vec());

    for c in &self.comm {
      v.extend(c.to_transcript_bytes());
    }

    v.append(&mut b"poly_commitment_end".to_vec());
    v
  }
}

impl<E: Engine> CommitmentTrait<E> for HyraxCommitment<E> where E::GE: DlogGroupExt {}
