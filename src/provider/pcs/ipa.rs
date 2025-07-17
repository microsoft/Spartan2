//! Inner Product Argument (IPA) implementation
use crate::{
  errors::SpartanError,
  provider::traits::{DlogGroup, DlogGroupExt},
  start_span,
  traits::{
    Engine,
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::{fmt::Debug, iter};
use ff::Field;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{info, info_span};

/// computes the inner product of two vectors in parallel.
pub(crate) fn inner_product<T: Field + Send + Sync>(a: &[T], b: &[T]) -> T {
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

/// Holds witness for the inner product instance.
pub struct InnerProductWitness<E: Engine> {
  a_vec: Vec<E::Scalar>,
  r_a: E::Scalar, // blind for the commitment to a_vec
}

impl<E: Engine> InnerProductInstance<E>
where
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

impl<E: Engine> InnerProductWitness<E> {
  /// Creates a new inner product witness
  pub fn new(a_vec: &[E::Scalar], r_a: &E::Scalar) -> Self {
    InnerProductWitness {
      a_vec: a_vec.to_vec(),
      r_a: *r_a,
    }
  }
}

/// An inner product argument using Bulletproofs
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InnerProductArgument<E: Engine>
where
  E::GE: DlogGroup,
{
  r_a: E::Scalar, // blind for the commitment to a_vec
  L_vec: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  R_vec: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  a_hat: E::Scalar,
}

impl<E: Engine> InnerProductArgument<E>
where
  E::GE: DlogGroupExt,
{
  const fn protocol_name() -> &'static [u8] {
    b"IPA"
  }

  /// Proves the inner product argument
  pub fn prove(
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

      let (L_result, R_result) = rayon::join(
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
          .map(|point| point.affine())
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
          .map(|point| point.affine())
        },
      );
      let L = L_result?;
      let R = R_result?;

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

    let (_recursion_span, recursion_t) = start_span!("ipa_recursion");
    // we create mutable copies of vectors and generators
    let mut a_vec = W.a_vec.to_vec();
    let mut b_vec = U.b_vec.to_vec();
    let mut ck = ck.to_vec();
    let num_rounds = usize::try_from(U.b_vec.len().ilog2()).unwrap();
    for _i in 0..num_rounds {
      let (L, R, a_vec_folded, b_vec_folded, ck_folded) =
        prove_inner(&a_vec, &b_vec, &ck, transcript)?;
      L_vec.push(L);
      R_vec.push(R);

      a_vec = a_vec_folded;
      b_vec = b_vec_folded;
      ck = ck_folded;
    }
    info!(
      elapsed_ms = %recursion_t.elapsed().as_millis(),
      rounds = %num_rounds,
      "ipa_recursion"
    );

    Ok(InnerProductArgument {
      r_a: W.r_a,
      L_vec,
      R_vec,
      a_hat: a_vec[0],
    })
  }

  /// Verifies the inner product argument
  pub fn verify(
    &self,
    ck: &[<E::GE as DlogGroup>::AffineGroupElement],
    h: &<E::GE as DlogGroup>::AffineGroupElement,
    ck_c: &<E::GE as DlogGroup>::AffineGroupElement,
    n: usize,
    U: &InnerProductInstance<E>,
    transcript: &mut E::TE,
  ) -> Result<(), SpartanError> {
    let (_verify_span, verify_t) = start_span!("ipa_verify");
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

    // remove the blind
    let comm_a_vec = U.comm_a_vec - E::GE::group(h) * self.r_a;

    // sample a random base for committing to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = (E::GE::group(ck_c) * r).affine();

    let P = (comm_a_vec + E::GE::group(&ck_c) * U.c).affine();

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
    let (_challenges_span, challenges_t) = start_span!("ipa_compute_challenges");
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
    info!(elapsed_ms = %challenges_t.elapsed().as_millis(), "ipa_compute_challenges");

    // compute the vector with the tensor structure
    let (_tensor_span, tensor_t) = start_span!("ipa_compute_tensor");
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

    let ck_hat = E::GE::vartime_multiscalar_mul(&s, ck, true)?;

    let b_hat = inner_product(&U.b_vec, &s);
    info!(elapsed_ms = %tensor_t.elapsed().as_millis(), "ipa_compute_tensor");

    let (_final_check_span, final_check_t) = start_span!("ipa_final_check");
    let ck_folded = [self.L_vec.clone(), self.R_vec.clone(), vec![P]].concat();
    let P_hat = E::GE::vartime_multiscalar_mul(
      &r_square
        .iter()
        .chain(r_inverse_square.iter())
        .chain(iter::once(&E::Scalar::ONE))
        .copied()
        .collect::<Vec<E::Scalar>>(),
      &ck_folded,
      true,
    )?;

    let rhs = ck_hat * self.a_hat + <E::GE as DlogGroup>::group(&ck_c) * (self.a_hat * b_hat);
    info!(elapsed_ms = %final_check_t.elapsed().as_millis(), "ipa_final_check");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "ipa_verify");
    if P_hat == rhs {
      Ok(())
    } else {
      Err(SpartanError::InvalidPCS)
    }
  }
}

// Instance: C_a, C_c, b_vec
// Witness: a_vec, r_a, c, r_c
// Sat if: C_x = Com(x, r_x), C_c = Com(c, r_c), and y = <a_vec, b_vec>
//
// P: samples d_vec, r_\beta, r_\delta, and sends:
// \delta \gets Com(d_vec, r_delta)
// \beta \gets Com(<b_vec, d_vec>, r_beta)
//
// V: sends a challenge r
//
// P: sends
// z_vec \gets r * a_vec + d_vec
// z_\delta \gets r * r_a + r_\delta
// z_\beta \gets r * r_c + r_\beta
//
// V: checks
// r * Comm_a + delta =? Com(z_vec, z_\delta)
// r * Comm_c + beta =? Com(<z_vec, b_vec>, z_\beta)
//
/// An inner product argument using a linear-sized argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InnerProductArgumentLinear<E: Engine>
where
  E::GE: DlogGroupExt,
{
  delta: E::GE,
  beta: E::GE,
  z_vec: Vec<E::Scalar>,
  z_delta: E::Scalar,
  z_beta: E::Scalar,
}

impl<E: Engine> InnerProductArgumentLinear<E>
where
  E::GE: DlogGroupExt,
{
  fn protocol_name() -> &'static [u8] {
    b"inner product argument (linear)"
  }

  /// Proves the inner product argument
  pub fn prove(
    ck: &[<E::GE as DlogGroup>::AffineGroupElement],
    h: &<E::GE as DlogGroup>::AffineGroupElement,
    ck_c: &<E::GE as DlogGroup>::AffineGroupElement,
    U: &InnerProductInstance<E>,
    W: &InnerProductWitness<E>,
    transcript: &mut E::TE,
  ) -> Result<Self, SpartanError> {
    transcript.dom_sep(Self::protocol_name());

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // produce randomness for the proofs
    let d_vec = (0..U.b_vec.len())
      .map(|_| E::Scalar::random(&mut OsRng))
      .collect::<Vec<E::Scalar>>();
    let r_delta = E::Scalar::random(&mut OsRng);
    let r_beta = E::Scalar::random(&mut OsRng);

    let delta = E::GE::vartime_multiscalar_mul(&d_vec, &ck[0..d_vec.len()], true)?
      + E::GE::group(h) * r_delta;
    let beta = E::GE::group(ck_c) * inner_product(&U.b_vec, &d_vec) + E::GE::group(h) * r_beta;

    transcript.absorb(b"delta", &delta);
    transcript.absorb(b"beta", &beta);

    let r = transcript.squeeze(b"r")?;

    let z_vec = (0..d_vec.len())
      .map(|i| r * W.a_vec[i] + d_vec[i])
      .collect::<Vec<E::Scalar>>();

    let z_delta = r * W.r_a + r_delta;
    let z_beta = r_beta; // since r_c = 0 

    Ok(Self {
      delta,
      z_vec,
      z_delta,
      beta,
      z_beta,
    })
  }

  /// Verifies the inner product argument
  pub fn verify(
    &self,
    ck: &[<E::GE as DlogGroup>::AffineGroupElement],
    h: &<E::GE as DlogGroup>::AffineGroupElement,
    ck_c: &<E::GE as DlogGroup>::AffineGroupElement,
    n: usize,
    U: &InnerProductInstance<E>,
    transcript: &mut E::TE,
  ) -> Result<(), SpartanError> {
    transcript.dom_sep(Self::protocol_name());

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    transcript.absorb(b"delta", &self.delta);
    transcript.absorb(b"beta", &self.beta);

    let r = transcript.squeeze(b"r")?;

    if self.z_vec.len() != n || ck.len() < self.z_vec.len() {
      return Err(SpartanError::InvalidInputLength);
    }

    if U.comm_a_vec * r + self.delta
      != E::GE::vartime_multiscalar_mul(&self.z_vec, &ck[0..self.z_vec.len()], true)?
        + E::GE::group(h) * self.z_delta
    {
      return Err(SpartanError::InvalidPCS);
    }

    if E::GE::group(ck_c) * (U.c * r) + self.beta
      != E::GE::group(ck_c) * inner_product(&self.z_vec, &U.b_vec) + E::GE::group(h) * self.z_beta
    {
      return Err(SpartanError::InvalidPCS);
    }

    Ok(())
  }
}
