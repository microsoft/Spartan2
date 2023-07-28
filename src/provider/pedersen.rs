//! This module provides an implementation of a commitment engine
use crate::{
  errors::SpartanError,
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    CompressedGroup, Group, TranscriptReprTrait,
  },
};
use core::{
  fmt::Debug,
  marker::PhantomData,
  ops::{Add, AddAssign, Mul, MulAssign},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that holds commitment generators
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentKey<G: Group> {
  ck: Vec<G::PreprocessedGroupElement>,
}

/// A type that holds a commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Commitment<G: Group> {
  comm: G,
}

/// A type that holds a compressed commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedCommitment<G: Group> {
  comm: G::CompressedGroupElement,
}

impl<G: Group> CommitmentTrait<G> for Commitment<G> {
  type CompressedCommitment = CompressedCommitment<G>;

  fn compress(&self) -> Self::CompressedCommitment {
    CompressedCommitment {
      comm: self.comm.compress(),
    }
  }

  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, SpartanError> {
    let comm = c.comm.decompress();
    if comm.is_none() {
      return Err(SpartanError::DecompressionError);
    }
    Ok(Commitment {
      comm: comm.unwrap(),
    })
  }
}

impl<G: Group> Default for Commitment<G> {
  fn default() -> Self {
    Commitment { comm: G::zero() }
  }
}

impl<G: Group> TranscriptReprTrait<G> for Commitment<G> {
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

impl<G: Group> TranscriptReprTrait<G> for CompressedCommitment<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.comm.to_transcript_bytes()
  }
}

impl<G: Group> MulAssign<G::Scalar> for Commitment<G> {
  fn mul_assign(&mut self, scalar: G::Scalar) {
    let result = (self as &Commitment<G>).comm * scalar;
    *self = Commitment { comm: result };
  }
}

impl<'a, 'b, G: Group> Mul<&'b G::Scalar> for &'a Commitment<G> {
  type Output = Commitment<G>;
  fn mul(self, scalar: &'b G::Scalar) -> Commitment<G> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<G: Group> Mul<G::Scalar> for Commitment<G> {
  type Output = Commitment<G>;

  fn mul(self, scalar: G::Scalar) -> Commitment<G> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<'b, G: Group> AddAssign<&'b Commitment<G>> for Commitment<G> {
  fn add_assign(&mut self, other: &'b Commitment<G>) {
    let result = (self as &Commitment<G>).comm + other.comm;
    *self = Commitment { comm: result };
  }
}

impl<'a, 'b, G: Group> Add<&'b Commitment<G>> for &'a Commitment<G> {
  type Output = Commitment<G>;
  fn add(self, other: &'b Commitment<G>) -> Commitment<G> {
    Commitment {
      comm: self.comm + other.comm,
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

define_add_assign_variants!(G = Group, LHS = Commitment<G>, RHS = Commitment<G>);
define_add_variants!(G = Group, LHS = Commitment<G>, RHS = Commitment<G>, Output = Commitment<G>);

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentEngine<G: Group> {
  _p: PhantomData<G>,
}

impl<G: Group> CommitmentEngineTrait<G> for CommitmentEngine<G> {
  type CommitmentKey = CommitmentKey<G>;
  type Commitment = Commitment<G>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    Self::CommitmentKey {
      ck: G::from_label(label, n.next_power_of_two()),
    }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[G::Scalar]) -> Self::Commitment {
    if ck.ck.len() < v.len() {
      println!("commitment key length: {}", ck.ck.len());
      println!("v length: {}", v.len());
    }
    assert!(ck.ck.len() >= v.len());
    Commitment {
      comm: G::vartime_multiscalar_mul(v, &ck.ck[..v.len()]),
    }
  }
}

/// A trait listing properties of a commitment key that can be managed in a divide-and-conquer fashion
pub trait CommitmentKeyExtTrait<G: Group> {
  /// Splits the commitment key into two pieces at a specified point
  fn split_at(ck: &Self::CommitmentKey, n: usize) -> (Self::CommitmentKey, Self::CommitmentKey);

  /// Combines two commitment keys into one
  fn combine(ck: &Self::CommitmentKey, other: &Self::CommitmentKey) -> Self::CommitmentKey;

  /// Folds the two commitment keys into one using the provided weights
  fn fold(ck: &Self::CommitmentKey, w1: &G::Scalar, w2: &G::Scalar) -> Self::CommitmentKey;

  /// Scales the commitment key using the provided scalar
<<<<<<< HEAD
  fn scale(&self, r: &G::Scalar) -> Self;

  /// Reinterprets commitments as commitment keys
  fn reinterpret_commitments_as_ck(
    c: &[<<<G as Group>::CE as CommitmentEngineTrait<G>>::Commitment as CommitmentTrait<G>>::CompressedCommitment],
  ) -> Result<Self, SpartanError>
  where
    Self: Sized;
}

impl<G: Group<CE = CommitmentEngine<G>>> CommitmentKeyExtTrait<G> for CommitmentKey<G> {
  fn split_at(&self, n: usize) -> (CommitmentKey<G>, CommitmentKey<G>) {
    (
      CommitmentKey {
        ck: self.ck[0..n].to_vec(),
      },
      CommitmentKey {
        ck: self.ck[n..].to_vec(),
=======
  fn scale(ck: &Self::CommitmentKey, r: &G::Scalar) -> Self::CommitmentKey;

  /// Reinterprets the commitments as a commitment key
  fn reinterpret_commitments_as_ck(commitments: &[Self::Commitment]) -> Self::CommitmentKey;
}

impl<G: Group> CommitmentEngineExtTrait<G> for CommitmentEngine<G> {
  fn split_at(ck: &Self::CommitmentKey, n: usize) -> (Self::CommitmentKey, Self::CommitmentKey) {
    (
      CommitmentKey {
        ck: ck.ck[0..n].to_vec(),
        _p: Default::default(),
      },
      CommitmentKey {
        ck: ck.ck[n..].to_vec(),
        _p: Default::default(),
>>>>>>> checkpoint
      },
    )
  }

  fn combine(ck: &Self::CommitmentKey, other: &Self::CommitmentKey) -> Self::CommitmentKey {
    let ck = {
      let mut c = ck.ck.clone();
      c.extend(other.ck.clone());
      c
    };
<<<<<<< HEAD
    CommitmentKey { ck }
=======
    Self::CommitmentKey {
      ck,
      _p: Default::default(),
    }
>>>>>>> checkpoint
  }

  fn fold(ck: &Self::CommitmentKey, w1: &G::Scalar, w2: &G::Scalar) -> Self::CommitmentKey {
    let w = vec![*w1, *w2];
    let (L, R) = Self::split_at(ck, ck.ck.len() / 2);

    let ck = (0..ck.ck.len() / 2)
      .into_par_iter()
      .map(|i| {
        let bases = [L.ck[i].clone(), R.ck[i].clone()].to_vec();
        G::vartime_multiscalar_mul(&w, &bases).preprocessed()
      })
      .collect();

<<<<<<< HEAD
    CommitmentKey { ck }
=======
    Self::CommitmentKey {
      ck,
      _p: Default::default(),
    }
>>>>>>> checkpoint
  }

  fn scale(ck: &Self::CommitmentKey, r: &G::Scalar) -> Self::CommitmentKey {
    let ck_scaled = ck
      .ck
      .clone()
      .into_par_iter()
      .map(|g| G::vartime_multiscalar_mul(&[*r], &[g]).preprocessed())
      .collect();

<<<<<<< HEAD
    CommitmentKey { ck: ck_scaled }
  }

  /// reinterprets a vector of commitments as a set of generators
  fn reinterpret_commitments_as_ck(c: &[CompressedCommitment<G>]) -> Result<Self, SpartanError> {
    let d = (0..c.len())
      .into_par_iter()
      .map(|i| Commitment::<G>::decompress(&c[i]))
      .collect::<Result<Vec<Commitment<G>>, SpartanError>>()?;
    let ck = (0..d.len())
      .into_par_iter()
      .map(|i| d[i].comm.preprocessed())
      .collect();
    Ok(CommitmentKey { ck })
=======
    Self::CommitmentKey {
      ck: ck_scaled,
      _p: Default::default(),
    }
  }

  fn reinterpret_commitments_as_ck(commitments: &[Self::Commitment]) -> Self::CommitmentKey {
    Self::CommitmentKey {
      ck: commitments.iter().map(|c| c.comm.preprocessed()).collect(),
      _p: Default::default(),
    }
>>>>>>> checkpoint
  }
}
