//! Inner Product Argument (IPA) implementation
use crate::{
  errors::SpartanError,
  provider::traits::{DlogGroup, DlogGroupExt},
  traits::{
    Engine,
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::{fmt::Debug, iter};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// computes the inner product of two vectors in parallel.
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
  /// Creates a new inner product instance
  pub fn new(comm_a_vec: &E::GE, b_vec: &[E::Scalar], c: &E::Scalar) -> Self {
    InnerProductInstance {
      comm_a_vec: *comm_a_vec,
      b_vec: b_vec.to_vec(),
      c: *c,
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for InnerProductInstance<E>
where
  E::GE: DlogGroup,
{
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
  pub fn new(a_vec: &[E::Scalar]) -> Self {
    InnerProductWitness {
      a_vec: a_vec.to_vec(),
    }
  }
}

/// An inner product argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InnerProductArgument<E: Engine>
where
  E::GE: DlogGroup,
{
  L_vec: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  R_vec: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  a_hat: E::Scalar,
}

impl<E> InnerProductArgument<E>
where
  E: Engine,
  E::GE: DlogGroupExt,
{
  const fn protocol_name() -> &'static [u8] {
    b"IPA"
  }

  /// Proves the inner product argument
  pub(crate) fn prove(
    ck: &[<E::GE as DlogGroup>::AffineGroupElement],
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
    let ck_c = (E::GE::group(ck_c) * r).affine();

    // a closure that executes a step of the recursive inner product argument
    let prove_inner = |a_vec: &[E::Scalar],
                       b_vec: &[E::Scalar],
                       ck: &[<E::GE as DlogGroup>::AffineGroupElement],
                       transcript: &mut E::TE|
     -> Result<
      (
        <E::GE as DlogGroup>::AffineGroupElement,
        <E::GE as DlogGroup>::AffineGroupElement,
        Vec<E::Scalar>,
        Vec<E::Scalar>,
        Vec<<E::GE as DlogGroup>::AffineGroupElement>,
      ),
      SpartanError,
    > {
      let n = a_vec.len();
      let (ck_L, ck_R) = ck.split_at(n / 2);

      let (c_L, c_R) = rayon::join(
        || inner_product(&a_vec[0..n / 2], &b_vec[n / 2..n]),
        || inner_product(&a_vec[n / 2..n], &b_vec[0..n / 2]),
      );

      let (L, R) = rayon::join(
        || {
          E::GE::vartime_multiscalar_mul(
            &a_vec[0..n / 2]
              .iter()
              .chain(iter::once(&c_L))
              .copied()
              .collect::<Vec<E::Scalar>>(),
            &[ck_R, &[ck_c]].concat(),
            true,
          )
          .affine()
        },
        || {
          E::GE::vartime_multiscalar_mul(
            &a_vec[n / 2..n]
              .iter()
              .chain(iter::once(&c_R))
              .copied()
              .collect::<Vec<E::Scalar>>(),
            &[ck_L, &[ck_c]].concat(),
            true,
          )
          .affine()
        },
      );

      transcript.absorb(b"L", &L);
      transcript.absorb(b"R", &R);

      let r = transcript.squeeze(b"r")?;
      let r_inverse = r.invert().unwrap();

      // fold the left half and the right half
      let ((a_vec_folded, b_vec_folded), ck_folded) = rayon::join(
        || {
          rayon::join(
            || {
              a_vec[0..n / 2]
                .par_iter()
                .zip(a_vec[n / 2..n].par_iter())
                .map(|(a_L, a_R)| *a_L * r + r_inverse * *a_R)
                .collect::<Vec<E::Scalar>>()
            },
            || {
              b_vec[0..n / 2]
                .par_iter()
                .zip(b_vec[n / 2..n].par_iter())
                .map(|(b_L, b_R)| *b_L * r_inverse + r * *b_R)
                .collect::<Vec<E::Scalar>>()
            },
          )
        },
        || {
          let (left, right) = ck.split_at(ck.len() / 2);
          left
            .par_iter()
            .zip(right.par_iter())
            .map(|(l_i, r_i)| (E::GE::group(l_i) * r_inverse + E::GE::group(r_i) * r).affine())
            .collect::<Vec<_>>()
        },
      );

      Ok((L, R, a_vec_folded, b_vec_folded, ck_folded))
    };

    // two vectors to hold the logarithmic number of group elements
    let mut L_vec: Vec<<E::GE as DlogGroup>::AffineGroupElement> = Vec::new();
    let mut R_vec: Vec<<E::GE as DlogGroup>::AffineGroupElement> = Vec::new();

    // we create mutable copies of vectors and generators
    let mut a_vec = W.a_vec.to_vec();
    let mut b_vec = U.b_vec.to_vec();
    let mut ck = ck.to_vec();
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

  /// Verifies the inner product argument
  pub fn verify(
    &self,
    ck: &[<E::GE as DlogGroup>::AffineGroupElement],
    ck_c: &<E::GE as DlogGroup>::AffineGroupElement,
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
    let ck_c = (E::GE::group(ck_c) * r).affine();

    let P = (U.comm_a_vec + E::GE::group(&ck_c) * U.c).affine();

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

    let ck_hat = E::GE::vartime_multiscalar_mul(&s, ck, true);

    let b_hat = inner_product(&U.b_vec, &s);

    let P_hat = {
      let ck_folded = [self.L_vec.clone(), self.R_vec.clone(), vec![P]].concat();

      E::GE::vartime_multiscalar_mul(
        &r_square
          .iter()
          .chain(r_inverse_square.iter())
          .chain(iter::once(&E::Scalar::ONE))
          .copied()
          .collect::<Vec<E::Scalar>>(),
        &ck_folded,
        true,
      )
    };

    let rhs = ck_hat * self.a_hat + <E::GE as DlogGroup>::group(&ck_c) * (self.a_hat * b_hat);

    if P_hat == rhs {
      Ok(())
    } else {
      Err(SpartanError::InvalidPCS)
    }
  }
}
