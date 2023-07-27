//! This module implements the Hyrax polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
use crate::{
  errors::SpartanError,
  provider::ipa_pc::{InnerProductArgument, InnerProductInstance, InnerProductWitness},
  provider::pedersen::{
    Commitment as PedersenCommitment, CommitmentEngine as PedersenCommitmentEngine,
    CommitmentKey as PedersenCommitmentKey, CommitmentKeyExtTrait,
  },
  spartan::polynomial::{EqPolynomial, MultilinearPolynomial},
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    evaluation::EvaluationEngineTrait,
    Group, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey,
};
use core::ops::{Add, AddAssign, Mul, MulAssign};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Structure that holds Poly Commits
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PolyCommit<G: Group> {
  /// Commitment
  comm: Vec<PedersenCommitment<G>>,
  is_default: bool,
}

impl<G: Group> Default for PolyCommit<G> {
  fn default() -> Self {
    PolyCommit {
      comm: vec![],
      is_default: true,
    }
  }
}

impl<G: Group> CommitmentTrait<G> for PolyCommit<G> {
  type CompressedCommitment = PolyCommit<G>;

  fn compress(&self) -> Self::CompressedCommitment {
    *self
  }

  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, SpartanError> {
    Ok(*c)
  }
}

impl<G: Group> MulAssign<G::Scalar> for PolyCommit<G> {
  fn mul_assign(&mut self, scalar: G::Scalar) {
    let result = (self as &PolyCommit<G>)
      .comm
      .iter()
      .map(|c| c * &scalar)
      .collect();
    *self = PolyCommit {
      comm: result,
      is_default: self.is_default,
    };
  }
}

impl<'a, 'b, G: Group> Mul<&'b G::Scalar> for &'a PolyCommit<G> {
  type Output = PolyCommit<G>;
  fn mul(self, scalar: &'b G::Scalar) -> PolyCommit<G> {
    let result = self.comm.iter().map(|c| c * &scalar).collect();
    PolyCommit {
      comm: result,
      is_default: self.is_default,
    }
  }
}

impl<G: Group> Mul<G::Scalar> for PolyCommit<G> {
  type Output = PolyCommit<G>;

  fn mul(self, scalar: G::Scalar) -> PolyCommit<G> {
    let result = self.comm.iter().map(|c| c * &scalar).collect();
    PolyCommit {
      comm: result,
      is_default: self.is_default,
    }
  }
}

impl<'b, G: Group> AddAssign<&'b PolyCommit<G>> for PolyCommit<G> {
  fn add_assign(&mut self, other: &'b PolyCommit<G>) {
    if self.is_default {
      *self = other.clone();
    } else if other.is_default {
      return;
    } else {
      let result = (self as &PolyCommit<G>)
        .comm
        .iter()
        .zip(other.comm.iter())
        .map(|(a, b)| a + b)
        .collect();
      *self = PolyCommit {
        comm: result,
        is_default: self.is_default,
      };
    }
  }
}

impl<'a, 'b, G: Group> Add<&'b PolyCommit<G>> for &'a PolyCommit<G> {
  type Output = PolyCommit<G>;
  fn add(self, other: &'b PolyCommit<G>) -> PolyCommit<G> {
    if self.is_default {
      return other.clone();
    } else if other.is_default {
      return self.clone();
    } else {
      let result = self
        .comm
        .iter()
        .zip(other.comm.iter())
        .map(|(a, b)| a + b)
        .collect();
      PolyCommit {
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

define_add_assign_variants!(G = Group, LHS = PolyCommit<G>, RHS = PolyCommit<G>);
define_add_variants!(G = Group, LHS = PolyCommit<G>, RHS = PolyCommit<G>, Output = PolyCommit<G>);

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyraxCommitmentEngine<G: Group> {
  _p: PhantomData<G>,
}

impl<G: Group> CommitmentEngineTrait<G> for HyraxCommitmentEngine<G> {
  type CommitmentKey = PedersenCommitmentKey<G>;
  type Commitment = PolyCommit<G>;

  /// Derives generators for Hyrax PC, where num_vars is the number of variables in multilinear poly
  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    let (_left, right) = EqPolynomial::<G::Scalar>::compute_factored_lens(n);
    PedersenCommitmentEngine::setup(label, (2usize).pow(right as u32))
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
      .map(|i| PedersenCommitmentEngine::commit(ck, &poly.get_Z()[R_size * i..R_size * (i + 1)]))
      .collect();

    PolyCommit {
      comm,
      is_default: false,
    }
  }
}

impl<G: Group> TranscriptReprTrait<G> for PolyCommit<G> {
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
  G: Group,
  CommitmentKey<G>: CommitmentKeyExtTrait<G, CE = G::CE>,
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
    let com_LZ = G::CE::commit(ck, &LZ);

    // a dot product argument (IPA) of size R_size
    let ipa_instance = InnerProductInstance::<G>::new(&com_LZ, &R, eval);
    let ipa_witness = InnerProductWitness::<G>::new(&LZ);
    let ipa =
      InnerProductArgument::<G>::prove(ck, &pk.ck_s, &ipa_instance, &ipa_witness, transcript)?;

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
    let gens: PedersenCommitmentKey<G> =
      PedersenCommitmentKey::<G>::reinterpret_commitments_as_ck(&comm.comm)?;

    let com_LZ = PedersenCommitmentEngine::commit(&gens, &L); // computes MSM of commitment and L

    let ipa_instance = InnerProductInstance::<G>::new(&com_LZ, &R, eval);

    arg
      .ipa
      .verify(&vk.ck_v, &vk.ck_s, L.len(), &ipa_instance, transcript)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  type G = pasta_curves::pallas::Point;
  use crate::traits::TranscriptEngineTrait;

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

  fn evaluate_with_LR(
    Z: &[<G as Group>::Scalar],
    r: &[<G as Group>::Scalar],
  ) -> <G as Group>::Scalar {
    let eq = EqPolynomial::new(r.to_vec());
    let (L, R) = eq.compute_factored_evals();

    let ell = r.len();
    // ensure ell is even
    assert!(ell % 2 == 0);

    // compute n = 2^\ell
    let n = (2usize).pow(ell as u32);

    // compute m = sqrt(n) = 2^{\ell/2}
    let m = (n as f64).sqrt() as usize;

    // compute vector-matrix product between L and Z viewed as a matrix
    let LZ = (0..m)
      .map(|i| {
        (0..m)
          .map(|j| L[j] * Z[j * m + i])
          .fold(<G as Group>::Scalar::ZERO, |acc, item| acc + item)
      })
      .collect::<Vec<<G as Group>::Scalar>>();

    // compute dot product between LZ and R
    inner_product(&LZ, &R)
  }

  fn to_scalar(x: usize) -> <G as Group>::Scalar {
    (0..x)
      .map(|_i| <G as Group>::Scalar::ONE)
      .fold(<G as Group>::Scalar::ZERO, |acc, item| acc + item)
  }

  #[test]
  fn check_polynomial_evaluation() {
    // Z = [1, 2, 1, 4]
    let Z = vec![to_scalar(1), to_scalar(2), to_scalar(1), to_scalar(4)];

    // r = [4,3]
    let r = vec![to_scalar(4), to_scalar(3)];

    let eval_with_LR = evaluate_with_LR(&Z, &r);
    let poly = MultilinearPolynomial::new(Z);

    let eval = poly.evaluate(&r);
    assert_eq!(eval, to_scalar(28));
    assert_eq!(eval_with_LR, eval);
  }

  #[test]
  fn check_hyrax_pc_commit() {
    let Z = vec![to_scalar(1), to_scalar(2), to_scalar(1), to_scalar(4)];

    let poly = MultilinearPolynomial::new(Z);

    // Public stuff
    let num_vars = 2;
    assert_eq!(num_vars, poly.get_num_vars());
    let r = vec![to_scalar(4), to_scalar(3)]; // r = [4,3]

    // Prover actions
    let eval = poly.evaluate(&r);
    assert_eq!(eval, to_scalar(28));

    let prover_gens = HyraxPC::new(num_vars, b"poly_test");
    let poly_comm = prover_gens.commit(&poly);

    let mut prover_transcript = <pasta_curves::Ep as Group>::TE::new(b"example");

    let (ipa_proof, _ipa_witness): (InnerProductArgument<G>, InnerProductWitness<G>) = prover_gens
      .prove_eval(&poly, &poly_comm, &r, &eval, &mut prover_transcript)
      .unwrap();

    // Verifier actions

    let verifier_gens = HyraxPC::new(num_vars, b"poly_test");
    let mut verifier_transcript = <pasta_curves::Ep as Group>::TE::new(b"example");

    let res =
      verifier_gens.verify_eval(&r, &poly_comm, &eval, &ipa_proof, &mut verifier_transcript);
    assert!(res.is_ok());
  }
}
