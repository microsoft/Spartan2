//! This module implements the Hyrax polynomial commitment scheme
#[allow(unused)]
use crate::{
  Blind, Commitment, CommitmentKey,
  errors::SpartanError,
  math::Math,
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  provider::traits::{DlogGroup, DlogGroupExt},
  traits::{
    Engine, TranscriptEngineTrait, TranscriptReprTrait,
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    evaluation::EvaluationEngineTrait,
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
pub struct HyraxCommitmentEngine<E: Engine> {
  _p: PhantomData<E>,
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

    let gens = E::GE::from_label(label, n.next_power_of_two() + 1);
    let (h, ck) = gens.split_first().unwrap();

    Self::CommitmentKey {
      num_rows,
      num_cols,
      ck: ck.to_vec(),
      h: *h,
    }
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
        E::GE::vartime_multiscalar_mul(
          &v[ck.num_cols * i..ck.num_cols * (i + 1)],
          &ck.ck[..v.len()],
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
          &ck.ck[..v.len()],
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
}

impl<E: Engine> TranscriptReprTrait<E::GE> for HyraxCommitment<E>
where
  E::GE: DlogGroupExt,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut v = Vec::new();
    v.append(&mut b"poly_commitment_begin".to_vec());

    for c in &self.comm {
      let (x, y, is_infinity) = c.to_coordinates();
      let is_infinity_byte = (!is_infinity).into();
      let bytes = [
        x.to_transcript_bytes(),
        y.to_transcript_bytes(),
        [is_infinity_byte].to_vec(),
      ]
      .concat();
      v.extend(bytes);
    }

    v.append(&mut b"poly_commitment_end".to_vec());
    v
  }
}

impl<E: Engine> CommitmentTrait<E> for HyraxCommitment<E> where E::GE: DlogGroupExt {}

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
  E::GE: DlogGroupExt,
{
  type ProverKey = HyraxProverKey<E>;
  type VerifierKey = HyraxVerifierKey<E>;
  type EvaluationArgument = HyraxEvaluationArgument<E>;

  fn setup(ck: &CommitmentKey<E>) -> (Self::ProverKey, Self::VerifierKey) {
    let pk = HyraxProverKey::<E> {
      ck_s: E::CE::setup(b"hyrax", 1),
    };

    let vk = HyraxVerifierKey::<E> {
      ck_v: ck.clone(),
      ck_s: E::CE::setup(b"hyrax", 1),
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
    transcript: &mut E::TE,
    comm: &Commitment<E>,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    transcript.absorb(b"poly_com", comm);

    // compute L and R
    // n = 2^point.len()
    let n = (2_usize).pow(point.len() as u32);
    let (num_rows, num_cols) = compute_factored_lens(n);

    let (num_vars_rows, num_vars_cols) = (num_rows.log_2(), num_cols.log_2());

    let L = EqPolynomial::new(point[..num_vars_rows].to_vec()).evals();
    let R = EqPolynomial::new(point[num_vars_rows..].to_vec()).evals();

    // compute a weighted sum of commitments and L
    // convert the commitments to affine form so we can do a multi-scalar multiplication
    let ck = comm.comm.map(|c| c.to_affine()).collect();
    let comm_LZ = E::GE::vartime_multiscalar_mul(&L, &ck[..L.len()]);

    let ipa_instance = InnerProductInstance::<E>::new(&comm_LZ, &R, eval);

    arg
      .ipa
      .verify(&vk.ck_v.ck, &vk.ck_s.ck, R.len(), &ipa_instance, transcript)
  }
}

fn inner_product<T: Field + Send + Sync>(a: &[T], b: &[T]) -> T {
  assert_eq!(a.len(), b.len());
  (0..a.len())
    .into_par_iter()
    .map(|i| a[i] * b[i])
    .reduce(|| T::ZERO, |x, y| x + y)
}

/// An inner product instance consists of a commitment to a vector `a` and another vector `b`
/// and the claim that c = <a, b>.
pub struct InnerProductInstance<E: Engine> {
  comm_a_vec: E::GE,
  b_vec: Vec<E::Scalar>,
  c: E::Scalar,
}

impl<E> InnerProductInstance<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn new(comm_a_vec: &E::GE, b_vec: &[E::Scalar], c: &E::Scalar) -> Self {
    InnerProductInstance {
      comm_a_vec: comm_a_vec.clone(),
      b_vec: b_vec.to_vec(),
      c: *c,
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for InnerProductInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // we do not need to include self.b_vec as in our context it is produced from the transcript
    [
      self.comm_a_vec.to_transcript_bytes(),
      self.c.to_transcript_bytes(),
    ]
    .concat()
  }
}

pub(crate) struct InnerProductWitness<E: Engine> {
  a_vec: Vec<E::Scalar>,
}

impl<E: Engine> InnerProductWitness<E> {
  fn new(a_vec: &[E::Scalar]) -> Self {
    InnerProductWitness {
      a_vec: a_vec.to_vec(),
    }
  }
}

/// An inner product argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InnerProductArgument<E: Engine> {
  L_vec: Vec<Commitment<E>>,
  R_vec: Vec<Commitment<E>>,
  a_hat: E::Scalar,
}

impl<E> InnerProductArgument<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  const fn protocol_name() -> &'static [u8] {
    b"IPA"
  }

  fn prove(
    ck: &Vec<<E::GE as DlogGroup>::AffineGroupElement>,
    ck_c: &<E::GE as DlogGroup>::AffineGroupElement,
    U: &InnerProductInstance<E>,
    W: &InnerProductWitness<E>,
    transcript: &mut E::TE,
  ) -> Result<Self, SpartanError> {
    transcript.dom_sep(Self::protocol_name());

    let (ck, _) = ck.split_at(U.b_vec.len());

    if U.b_vec.len() != W.a_vec.len() {
      return Err(SpartanError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // sample a random base for committing to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = ck_c.scale(&r);

    // a closure that executes a step of the recursive inner product argument
    let prove_inner = |a_vec: &[E::Scalar],
                       b_vec: &[E::Scalar],
                       ck: &CommitmentKey<E>,
                       transcript: &mut E::TE|
     -> Result<
      (
        Commitment<E>,
        Commitment<E>,
        Vec<E::Scalar>,
        Vec<E::Scalar>,
        CommitmentKey<E>,
      ),
      SpartanError,
    > {
      let n = a_vec.len();
      let (ck_L, ck_R) = ck.split_at(n / 2);

      let c_L = inner_product(&a_vec[0..n / 2], &b_vec[n / 2..n]);
      let c_R = inner_product(&a_vec[n / 2..n], &b_vec[0..n / 2]);

      let L = E::GE::vartime_multiscalar_mul(
        &a_vec[0..n / 2]
          .iter()
          .chain(iter::once(&c_L))
          .copied()
          .collect::<Vec<E::Scalar>>(),
        &ck_R.combine(&ck_c),
      );
      let R = E::GE::vartime_multiscalar_mul(
        &a_vec[n / 2..n]
          .iter()
          .chain(iter::once(&c_R))
          .copied()
          .collect::<Vec<E::Scalar>>(),
        &ck_L.combine(&ck_c),
      );

      transcript.absorb(b"L", &L);
      transcript.absorb(b"R", &R);

      let r = transcript.squeeze(b"r")?;
      let r_inverse = r.invert().unwrap();

      // fold the left half and the right half
      let a_vec_folded = a_vec[0..n / 2]
        .par_iter()
        .zip(a_vec[n / 2..n].par_iter())
        .map(|(a_L, a_R)| *a_L * r + r_inverse * *a_R)
        .collect::<Vec<E::Scalar>>();

      let b_vec_folded = b_vec[0..n / 2]
        .par_iter()
        .zip(b_vec[n / 2..n].par_iter())
        .map(|(b_L, b_R)| *b_L * r_inverse + r * *b_R)
        .collect::<Vec<E::Scalar>>();

      let ck_folded = ck.fold(&r_inverse, &r);

      Ok((L, R, a_vec_folded, b_vec_folded, ck_folded))
    };

    // two vectors to hold the logarithmic number of group elements
    let mut L_vec: Vec<Commitment<E>> = Vec::new();
    let mut R_vec: Vec<Commitment<E>> = Vec::new();

    // we create mutable copies of vectors and generators
    let mut a_vec = W.a_vec.to_vec();
    let mut b_vec = U.b_vec.to_vec();
    let mut ck = ck;
    for _i in 0..usize::try_from(U.b_vec.len().ilog2()).unwrap() {
      let (L, R, a_vec_folded, b_vec_folded, ck_folded) =
        prove_inner(&a_vec, &b_vec, &ck, transcript)?;
      L_vec.push(L);
      R_vec.push(R);

      a_vec = a_vec_folded;
      b_vec = b_vec_folded;
      ck = ck_folded;
    }

    Ok(InnerProductArgument {
      L_vec,
      R_vec,
      a_hat: a_vec[0],
    })
  }

  fn verify(
    &self,
    ck: &CommitmentKey<E>,
    ck_c: &CommitmentKey<E>,
    n: usize,
    U: &InnerProductInstance<E>,
    transcript: &mut E::TE,
  ) -> Result<(), SpartanError> {
    let (ck, _) = ck.split_at(U.b_vec.len());

    transcript.dom_sep(Self::protocol_name());
    if U.b_vec.len() != n
      || n != (1 << self.L_vec.len())
      || self.L_vec.len() != self.R_vec.len()
      || self.L_vec.len() >= 32
    {
      return Err(SpartanError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // sample a random base for committing to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = ck_c * r;

    let P = U.comm_a_vec.clone() + ck_c * U.c;

    let batch_invert = |v: &[E::Scalar]| -> Result<Vec<E::Scalar>, SpartanError> {
      let mut products = vec![E::Scalar::ZERO; v.len()];
      let mut acc = E::Scalar::ONE;

      for i in 0..v.len() {
        products[i] = acc;
        acc *= v[i];
      }

      // return error if acc is zero
      acc = match Option::from(acc.invert()) {
        Some(inv) => inv,
        None => return Err(SpartanError::InternalError),
      };

      // compute the inverse once for all entries
      let mut inv = vec![E::Scalar::ZERO; v.len()];
      for i in (0..v.len()).rev() {
        let tmp = acc * v[i];
        inv[i] = products[i] * acc;
        acc = tmp;
      }

      Ok(inv)
    };

    // compute a vector of public coins using self.L_vec and self.R_vec
    let r = (0..self.L_vec.len())
      .map(|i| {
        transcript.absorb(b"L", &self.L_vec[i]);
        transcript.absorb(b"R", &self.R_vec[i]);
        transcript.squeeze(b"r")
      })
      .collect::<Result<Vec<E::Scalar>, SpartanError>>()?;

    // precompute scalars necessary for verification
    let r_square: Vec<E::Scalar> = (0..self.L_vec.len())
      .into_par_iter()
      .map(|i| r[i] * r[i])
      .collect();
    let r_inverse = batch_invert(&r)?;
    let r_inverse_square: Vec<E::Scalar> = (0..self.L_vec.len())
      .into_par_iter()
      .map(|i| r_inverse[i] * r_inverse[i])
      .collect();

    // compute the vector with the tensor structure
    let s = {
      let mut s = vec![E::Scalar::ZERO; n];
      s[0] = {
        let mut v = E::Scalar::ONE;
        for r_inverse_i in r_inverse {
          v *= r_inverse_i;
        }
        v
      };
      for i in 1..n {
        let pos_in_r = (31 - (i as u32).leading_zeros()) as usize;
        s[i] = s[i - (1 << pos_in_r)] * r_square[(self.L_vec.len() - 1) - pos_in_r];
      }
      s
    };

    let ck_hat = ck * s;

    let b_hat = inner_product(&U.b_vec, &s);

    let P_hat = {
      let ck_folded = {
        let ck_L = CommitmentKey::<E>::reinterpret_commitments_as_ck(&self.L_vec)?;
        let ck_R = CommitmentKey::<E>::reinterpret_commitments_as_ck(&self.R_vec)?;
        let ck_P = CommitmentKey::<E>::reinterpret_commitments_as_ck(&[P])?;
        ck_L.combine(&ck_R).combine(&ck_P)
      };

      CE::<E>::commit(
        &ck_folded,
        &r_square
          .iter()
          .chain(r_inverse_square.iter())
          .chain(iter::once(&E::Scalar::ONE))
          .copied()
          .collect::<Vec<E::Scalar>>(),
        &Blind::<E>::default(),
      )
    };

    if P_hat
      == CE::<E>::commit(
        &ck_hat.combine(&ck_c),
        &[self.a_hat, self.a_hat * b_hat],
        &Blind::<E>::default(),
      )
    {
      Ok(())
    } else {
      Err(SpartanError::InvalidPCS)
    }
  }
}*/
