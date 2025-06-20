//! This module provides an implementation of a commitment engine
use crate::{
  errors::SpartanError,
  provider::traits::{DlogGroup, DlogGroupExt, HomomorphicCommitmentTrait},
  traits::{
    Engine, TranscriptReprTrait,
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
  },
};
use core::{
  fmt::Debug,
  marker::PhantomData,
  ops::{Add, Mul, MulAssign},
};
use ff::Field;
use num_integer::Integer;
use num_traits::ToPrimitive;
use rand_core::OsRng;
use rayon::prelude::*;
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
    let (x, y, is_infinity) = self.comm.to_coordinates();
    let is_infinity_byte = (!is_infinity).into();
    [
      x.to_transcript_bytes(),
      y.to_transcript_bytes(),
      [is_infinity_byte].to_vec(),
    ]
    .concat()
  }
}

impl<E: Engine> MulAssign<E::Scalar> for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn mul_assign(&mut self, scalar: E::Scalar) {
    *self = Commitment {
      comm: self.comm * scalar,
    };
  }
}

impl<'b, E: Engine> Mul<&'b E::Scalar> for &'_ Commitment<E>
where
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;
  fn mul(self, scalar: &'b E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E: Engine> Mul<E::Scalar> for Commitment<E>
where
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;

  fn mul(self, scalar: E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E: Engine> Add for Commitment<E>
where
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;

  fn add(self, other: Commitment<E>) -> Commitment<E> {
    Commitment {
      comm: self.comm + other.comm,
    }
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

/// A trait listing properties of a commitment key that can be managed in a divide-and-conquer fashion
pub trait CommitmentKeyExtTrait<E: Engine>
where
  E::GE: DlogGroup,
{
  /// Splits the commitment key into two pieces at a specified point
  fn split_at(&self, n: usize) -> (Self, Self)
  where
    Self: Sized;

  /// Combines two commitment keys into one
  fn combine(&self, other: &Self) -> Self;

  /// Folds the two commitment keys into one using the provided weights
  fn fold(&self, w1: &E::Scalar, w2: &E::Scalar) -> Self;

  /// Scales the commitment key using the provided scalar
  fn scale(&self, r: &E::Scalar) -> Self;

  /// Reinterprets commitments as commitment keys
  fn reinterpret_commitments_as_ck(
    c: &[<E::CE as CommitmentEngineTrait<E>>::Commitment],
  ) -> Result<Self, SpartanError>
  where
    Self: Sized;
}

impl<E: Engine<CE = CommitmentEngine<E>>> CommitmentKeyExtTrait<E> for CommitmentKey<E>
where
  E::GE: DlogGroupExt,
{
  fn split_at(&self, n: usize) -> (CommitmentKey<E>, CommitmentKey<E>) {
    (
      CommitmentKey {
        ck: self.ck[0..n].to_vec(),
        h: self.h,
      },
      CommitmentKey {
        ck: self.ck[n..].to_vec(),
        h: self.h,
      },
    )
  }

  fn combine(&self, other: &CommitmentKey<E>) -> CommitmentKey<E> {
    let ck = {
      let mut c = self.ck.clone();
      c.extend(other.ck.clone());
      c
    };
    CommitmentKey { ck, h: self.h }
  }

  // combines the left and right halves of `self` using `w1` and `w2` as the weights
  fn fold(&self, w1: &E::Scalar, w2: &E::Scalar) -> CommitmentKey<E> {
    let w = vec![*w1, *w2];
    let (L, R) = self.split_at(self.ck.len() / 2);

    let ck = (0..self.ck.len() / 2)
      .into_par_iter()
      .map(|i| {
        let bases = [L.ck[i], R.ck[i]].to_vec();
        E::GE::vartime_multiscalar_mul(&w, &bases).affine()
      })
      .collect();

    CommitmentKey { ck, h: self.h }
  }

  /// Scales each element in `self` by `r`
  fn scale(&self, r: &E::Scalar) -> Self {
    let ck_scaled = self
      .ck
      .clone()
      .into_par_iter()
      .map(|g| E::GE::vartime_multiscalar_mul(&[*r], &[g]).affine())
      .collect();

    CommitmentKey {
      ck: ck_scaled,
      h: self.h,
    }
  }

  /// reinterprets a vector of commitments as a set of generators
  fn reinterpret_commitments_as_ck(c: &[Commitment<E>]) -> Result<Self, SpartanError> {
    let ck = (0..c.len())
      .into_par_iter()
      .map(|i| c[i].comm.affine())
      .collect();

    // cmt is derandomized by the point that this is called
    Ok(CommitmentKey {
      ck,
      h: E::GE::zero().affine(), // this is okay, since this method is used in IPA only,
                                 // and we only use non-blinding commits afterwards
                                 // bc we don't use ZK IPA
    })
  }
}

impl<E: Engine> HomomorphicCommitmentTrait<E> for Commitment<E> where E::GE: DlogGroupExt {}
