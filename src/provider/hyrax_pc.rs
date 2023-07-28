//! This module implements the Hyrax polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
use crate::{
  errors::SpartanError,
  provider::ipa_pc::{InnerProductArgument, InnerProductInstance, InnerProductWitness},
  provider::pedersen::{
    Commitment as PedersenCommitment, CommitmentEngine as PedersenCommitmentEngine,
    CommitmentEngineExtTrait, CommitmentKey as PedersenCommitmentKey,
    CompressedCommitment as PedersenCompressedCommitment,
  },
  spartan::{
    math::Math,
    polynomial::{EqPolynomial, MultilinearPolynomial},
  },
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    evaluation::EvaluationEngineTrait,
    Group, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey,
};
use core::ops::{Add, AddAssign, Mul, MulAssign};
use itertools::{
  EitherOrBoth::{Both, Left, Right},
  Itertools,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// A type that holds commitment generators for Hyrax commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCommitmentKey<G: Group> {
  ck: PedersenCommitmentKey<G>,
  _p: PhantomData<G>,
}

/// Structure that holds commitments
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCommitment<G: Group> {
  comm: Vec<PedersenCommitment<G>>,
  is_default: bool,
}

/// Structure that holds compressed commitments
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCompressedCommitment<G: Group> {
  comm: Vec<PedersenCompressedCommitment<G>>,
  is_default: bool,
}

impl<G: Group> Default for HyraxCommitment<G> {
  fn default() -> Self {
    HyraxCommitment {
      comm: vec![],
      is_default: true,
    }
  }
}

impl<G: Group> CommitmentTrait<G> for HyraxCommitment<G> {
  type CompressedCommitment = HyraxCompressedCommitment<G>;

  fn compress(&self) -> Self::CompressedCommitment {
    HyraxCompressedCommitment {
      comm: self.comm.iter().map(|c| c.compress()).collect::<Vec<_>>(),
      is_default: self.is_default,
    }
  }

  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, SpartanError> {
    let comm = c
      .comm
      .iter()
      .map(|c| <PedersenCommitment<G> as CommitmentTrait<G>>::decompress(c))
      .collect::<Result<Vec<_>, _>>()?;
    Ok(HyraxCommitment {
      comm,
      is_default: c.is_default,
    })
  }
}

impl<G: Group> MulAssign<G::Scalar> for HyraxCommitment<G> {
  fn mul_assign(&mut self, scalar: G::Scalar) {
    let result = (self as &HyraxCommitment<G>)
      .comm
      .iter()
      .map(|c| c * &scalar)
      .collect();
    *self = HyraxCommitment {
      comm: result,
      is_default: self.is_default,
    };
  }
}

impl<'a, 'b, G: Group> Mul<&'b G::Scalar> for &'a HyraxCommitment<G> {
  type Output = HyraxCommitment<G>;
  fn mul(self, scalar: &'b G::Scalar) -> HyraxCommitment<G> {
    let result = self.comm.iter().map(|c| c * scalar).collect();
    HyraxCommitment {
      comm: result,
      is_default: self.is_default,
    }
  }
}

impl<G: Group> Mul<G::Scalar> for HyraxCommitment<G> {
  type Output = HyraxCommitment<G>;

  fn mul(self, scalar: G::Scalar) -> HyraxCommitment<G> {
    let result = self.comm.iter().map(|c| c * &scalar).collect();
    HyraxCommitment {
      comm: result,
      is_default: self.is_default,
    }
  }
}

impl<'b, G: Group> AddAssign<&'b HyraxCommitment<G>> for HyraxCommitment<G> {
  fn add_assign(&mut self, other: &'b HyraxCommitment<G>) {
    if self.is_default {
      *self = other.clone();
    } else if other.is_default {
      return;
    } else {
      let result = (self as &HyraxCommitment<G>)
        .comm
        .iter()
        .zip_longest(other.comm.iter())
        .map(|x| match x {
          Both(a, b) => a + b,
          Left(a) => *a,
          Right(b) => *b,
        })
        .collect();
      *self = HyraxCommitment {
        comm: result,
        is_default: self.is_default,
      };
    }
  }
}

impl<'a, 'b, G: Group> Add<&'b HyraxCommitment<G>> for &'a HyraxCommitment<G> {
  type Output = HyraxCommitment<G>;
  fn add(self, other: &'b HyraxCommitment<G>) -> HyraxCommitment<G> {
    if self.is_default {
      other.clone()
    } else if other.is_default {
      self.clone()
    } else {
      let result = self
        .comm
        .iter()
        .zip_longest(other.comm.iter())
        .map(|x| match x {
          Both(a, b) => a + b,
          Left(a) => *a,
          Right(b) => *b,
        })
        .collect();
      HyraxCommitment {
        comm: result,
        is_default: self.is_default,
      }
    }
  }
}

macro_rules! define_add_variants {
  (G = $g:path, LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
    impl<'b, G: $g> Add<&'b $rhs> for $lhs {
      type Output = $out;
      fn add(self, rhs: &'b $rhs) -> $out {
        &self + rhs
      }
    }

    impl<'a, G: $g> Add<$rhs> for &'a $lhs {
      type Output = $out;
      fn add(self, rhs: $rhs) -> $out {
        self + &rhs
      }
    }

    impl<G: $g> Add<$rhs> for $lhs {
      type Output = $out;
      fn add(self, rhs: $rhs) -> $out {
        &self + &rhs
      }
    }
  };
}

macro_rules! define_add_assign_variants {
  (G = $g:path, LHS = $lhs:ty, RHS = $rhs:ty) => {
    impl<G: $g> AddAssign<$rhs> for $lhs {
      fn add_assign(&mut self, rhs: $rhs) {
        *self += &rhs;
      }
    }
  };
}

define_add_assign_variants!(G = Group, LHS = HyraxCommitment<G>, RHS = HyraxCommitment<G>);
define_add_variants!(G = Group, LHS = HyraxCommitment<G>, RHS = HyraxCommitment<G>, Output = HyraxCommitment<G>);

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyraxCommitmentEngine<G: Group> {
  _p: PhantomData<G>,
}

impl<G: Group> CommitmentEngineTrait<G> for HyraxCommitmentEngine<G> {
  type CommitmentKey = HyraxCommitmentKey<G>;
  type Commitment = HyraxCommitment<G>;

  /// Derives generators for Hyrax PC, where num_vars is the number of variables in multilinear poly
  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    let num_vars = n.next_power_of_two().log_2();
    let (_left, right) = EqPolynomial::<G::Scalar>::compute_factored_lens(num_vars);
    let ck = PedersenCommitmentEngine::setup(label, (2usize).pow(right as u32));
    HyraxCommitmentKey {
      ck,
      _p: Default::default(),
    }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[G::Scalar]) -> Self::Commitment {
    let poly = MultilinearPolynomial::new(v.to_vec());
    let n = poly.len();
    let ell = poly.get_num_vars();
    assert_eq!(n, (2usize).pow(ell as u32));

    let (left_num_vars, right_num_vars) = EqPolynomial::<G::Scalar>::compute_factored_lens(ell);
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

impl<G: Group> TranscriptReprTrait<G> for HyraxCommitment<G> {
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

impl<G: Group> TranscriptReprTrait<G> for HyraxCompressedCommitment<G> {
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

/// Provides an implementation of the hyrax key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxProverKey<G: Group> {
  ck_s: CommitmentKey<G>,
}

/// Provides an implementation of the hyrax key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxVerifierKey<G: Group> {
  ck_v: CommitmentKey<G>,
  ck_s: CommitmentKey<G>,
}

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxEvaluationArgument<G: Group> {
  ipa: InnerProductArgument<G>,
}

/// Provides an implementation of a polynomial evaluation engine using Hyrax PC
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxEvaluationEngine<G: Group> {
  _p: PhantomData<G>,
}

impl<G> EvaluationEngineTrait<G> for HyraxEvaluationEngine<G>
where
  G: Group<CE = HyraxCommitmentEngine<G>>,
{
  type CE = G::CE;
  type ProverKey = HyraxProverKey<G>;
  type VerifierKey = HyraxVerifierKey<G>;
  type EvaluationArgument = HyraxEvaluationArgument<G>;

  fn setup(
    ck: &<Self::CE as CommitmentEngineTrait<G>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey) {
    let pk = HyraxProverKey::<G> {
      ck_s: G::CE::setup(b"hyrax", 1),
    };

    let vk = HyraxVerifierKey::<G> {
      ck_v: ck.clone(),
      ck_s: G::CE::setup(b"hyrax", 1),
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
    let ipa_instance = InnerProductInstance::<G>::new(&com_LZ, &R, eval);
    let ipa_witness = InnerProductWitness::<G>::new(&LZ);
    let ipa = InnerProductArgument::<G>::prove(
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
    comm: &Commitment<G>,
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

    let ipa_instance = InnerProductInstance::<G>::new(&com_LZ, &R, eval);

    arg
      .ipa
      .verify(&vk.ck_v.ck, &vk.ck_s.ck, L.len(), &ipa_instance, transcript)
  }
}
