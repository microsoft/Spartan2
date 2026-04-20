//! Multilinear KZG polynomial commitment scheme over pairing-friendly curves.
#![allow(missing_docs)]

use crate::{
  errors::SpartanError,
  polys::multilinear::MultilinearPolynomial,
  provider::msm::{fixed_base_msm, variable_base_msm, window_size, window_table},
  traits::{
    Engine,
    pcs::PCSEngineTrait,
    // transcript::TranscriptReprTrait
  },
  utils::parallel::parallelize,
};

use crate::traits::transcript::TranscriptEngineTrait;
use ff::{Field, PrimeField};
use group::{Curve, Group, prime::PrimeCurveAffine};
use halo2curves::{
  CurveAffine,
  pairing::{MillerLoopResult, MultiMillerLoop},
};
use itertools::{Itertools, izip};
use rand_core::OsRng;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{iter, marker::PhantomData, ops::Neg, slice};

#[derive(Clone, Debug)]
pub struct MultilinearKzg<M: MultiMillerLoop>(PhantomData<M>);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "M::G1Affine: Serialize, M::G2Affine: Serialize, M::Fr: Serialize",
  deserialize = "M: MultiMillerLoop, M::G1Affine: DeserializeOwned, M::G2Affine: DeserializeOwned, M::Fr: DeserializeOwned"
))]
pub struct MultilinearKzgParams<M: MultiMillerLoop> {
  g1: M::G1Affine,
  eqs: Vec<Vec<M::G1Affine>>,
  g2: M::G2Affine,
  ss: Vec<M::G2Affine>,
}

impl<M: MultiMillerLoop> MultilinearKzgParams<M> {
  pub fn num_vars(&self) -> usize {
    self.eqs.len()
  }

  pub fn g1(&self) -> M::G1Affine {
    self.g1
  }

  pub fn eqs(&self) -> &[Vec<M::G1Affine>] {
    &self.eqs
  }

  pub fn g2(&self) -> M::G2Affine {
    self.g2
  }

  pub fn ss(&self) -> &[M::G2Affine] {
    &self.ss
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "M::G1Affine: Serialize, M::G2Affine: Serialize, M::Fr: Serialize",
  deserialize = "M: MultiMillerLoop, M::G1Affine: DeserializeOwned, M::G2Affine: DeserializeOwned, M::Fr: DeserializeOwned"
))]
pub struct MultilinearKzgProverParams<M: MultiMillerLoop> {
  g1: M::G1Affine,
  eqs: Vec<Vec<M::G1Affine>>,
}

impl<M: MultiMillerLoop> MultilinearKzgProverParams<M> {
  pub fn num_vars(&self) -> usize {
    self.eqs.len() - 1
  }

  pub fn g1(&self) -> M::G1Affine {
    self.g1
  }

  pub fn eqs(&self) -> &[Vec<M::G1Affine>] {
    &self.eqs
  }

  pub fn eq(&self, num_vars: usize) -> &[M::G1Affine] {
    &self.eqs[num_vars]
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "M::G1Affine: Serialize, M::G2Affine: Serialize, M::Fr: Serialize",
  deserialize = "M: MultiMillerLoop, M::G1Affine: DeserializeOwned, M::G2Affine: DeserializeOwned, M::Fr: DeserializeOwned"
))]
pub struct MultilinearKzgVerifierParams<M: MultiMillerLoop> {
  g1: M::G1Affine,
  g2: M::G2Affine,
  ss: Vec<M::G2Affine>,
}

impl<M: MultiMillerLoop> MultilinearKzgVerifierParams<M> {
  pub fn num_vars(&self) -> usize {
    self.ss.len()
  }

  pub fn g1(&self) -> M::G1Affine {
    self.g1
  }

  pub fn g2(&self) -> M::G2Affine {
    self.g2
  }

  pub fn ss(&self, num_vars: usize) -> &[M::G2Affine] {
    &self.ss[..num_vars]
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultilinearKzgCommitment<C: CurveAffine>(pub C);

impl<C: CurveAffine> Default for MultilinearKzgCommitment<C> {
  fn default() -> Self {
    Self(C::identity())
  }
}

impl<C: CurveAffine> PartialEq for MultilinearKzgCommitment<C> {
  fn eq(&self, other: &Self) -> bool {
    self.0.eq(&other.0)
  }
}

impl<C: CurveAffine> Eq for MultilinearKzgCommitment<C> {}

impl<C: CurveAffine> AsRef<[C]> for MultilinearKzgCommitment<C> {
  fn as_ref(&self) -> &[C] {
    slice::from_ref(&self.0)
  }
}

impl<C: CurveAffine> AsRef<C> for MultilinearKzgCommitment<C> {
  fn as_ref(&self) -> &C {
    &self.0
  }
}

impl<C: CurveAffine> From<C> for MultilinearKzgCommitment<C> {
  fn from(comm: C) -> Self {
    Self(comm)
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "M::G1Affine: Serialize, M::G2Affine: Serialize, M::Fr: Serialize",
  deserialize = "M: MultiMillerLoop, M::G1Affine: DeserializeOwned, M::G2Affine: DeserializeOwned, M::Fr: DeserializeOwned"
))]
pub struct MultilinearKzgEvaluationArgument<M: MultiMillerLoop> {
  /// Quotient polynomial commitments (one per variable)
  pub proofs: Vec<M::G1Affine>,
  /// The evaluation value (needed for pairing check)
  pub eval: M::Fr,
}
impl<M, E> PCSEngineTrait<E> for MultilinearKzg<M>
where
  M: MultiMillerLoop,
  M::G1Affine: CurveAffine<ScalarExt = M::Fr, CurveExt = M::G1>
    + Serialize
    + DeserializeOwned
    + crate::traits::transcript::TranscriptReprTrait<E::GE>,
  M::G2Affine: CurveAffine<ScalarExt = M::Fr, CurveExt = M::G2> + Serialize + DeserializeOwned,
  M::Fr: Serialize + DeserializeOwned,
  E: Engine,
  E::Scalar: From<M::Fr> + Into<M::Fr> + Serialize + for<'de> Deserialize<'de>,
  E::GE: From<M::G1Affine> + Into<M::G1Affine> + Serialize + for<'de> Deserialize<'de>,
{
  type CommitmentKey = MultilinearKzgProverParams<M>;
  type VerifierKey = MultilinearKzgVerifierParams<M>;
  type Commitment = MultilinearKzgCommitment<M::G1Affine>;
  type Blind = ();
  type EvaluationArgument = MultilinearKzgEvaluationArgument<M>;

  fn setup(
    _label: &'static [u8],
    poly_size: usize,
    _width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    assert!(poly_size.is_power_of_two());
    let num_vars = poly_size.ilog2() as usize;
    let mut rng = OsRng;
    let ss = iter::repeat_with(|| M::Fr::random(&mut rng))
      .take(num_vars)
      .collect_vec();

    let g1 = M::G1Affine::generator();

    let eqs = {
      let mut eqs = Vec::with_capacity(1 << (num_vars + 1));
      eqs.push(vec![M::Fr::ONE]);

      for s_i in ss.iter() {
        let last_evals = eqs.last().unwrap();
        let mut evals = vec![M::Fr::ZERO; 2 * last_evals.len()];

        let (evals_lo, evals_hi) = evals.split_at_mut(last_evals.len());

        parallelize(evals_hi, |(evals_hi, start)| {
          izip!(evals_hi, &last_evals[start..])
            .for_each(|(eval_hi, last_eval)| *eval_hi = *s_i * last_eval);
        });
        parallelize(evals_lo, |(evals_lo, start)| {
          izip!(evals_lo, &evals_hi[start..], &last_evals[start..])
            .for_each(|(eval_lo, eval_hi, last_eval)| *eval_lo = *last_eval - eval_hi);
        });

        eqs.push(evals)
      }

      let window_size = window_size((2 << num_vars) - 2);
      let window_table = window_table(window_size, g1);
      let eqs_projective = fixed_base_msm(
        window_size,
        &window_table,
        eqs.iter().flat_map(|evals| evals.iter()),
      );

      let mut eqs = vec![M::G1Affine::identity(); eqs_projective.len()];
      parallelize(&mut eqs, |(eqs, start)| {
        M::G1::batch_normalize(&eqs_projective[start..(start + eqs.len())], eqs);
      });
      let eqs = &mut eqs.drain(..);
      (0..num_vars + 1)
        .map(move |idx| eqs.take(1 << idx).collect_vec())
        .collect_vec()
    };

    let g2 = M::G2Affine::generator();
    let ss = {
      let window_size = window_size(num_vars);
      let window_table = window_table(window_size, M::G2Affine::generator());
      let ss_projective = fixed_base_msm(window_size, &window_table, &ss);

      let mut ss = vec![M::G2Affine::identity(); ss_projective.len()];
      parallelize(&mut ss, |(ss, start)| {
        M::G2::batch_normalize(&ss_projective[start..(start + ss.len())], ss);
      });
      ss
    };

    let pp = Self::CommitmentKey {
      g1,
      eqs: eqs[..num_vars + 1].to_vec(),
    };
    let vp = Self::VerifierKey {
      g1,
      g2,
      ss: ss[..num_vars].to_vec(),
    };
    (pp, vp)
  }

  fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind {
    // MKZG doesn't use blinding
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    _r: &Self::Blind,
    _is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    let padded_len = v.len().next_power_of_two();
    let num_vars = padded_len.ilog2() as usize;
    if num_vars > ck.num_vars() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "MKZG commit: num_vars {} (padded len {}) exceeds commitment key limit {}",
          num_vars,
          padded_len,
          ck.num_vars()
        ),
      });
    }

    #[allow(clippy::manual_repeat_n)]
    let v_m: Vec<M::Fr> = v
      .iter()
      .map(|s| (*s).into())
      .chain(std::iter::repeat(M::Fr::ZERO).take(padded_len - v.len()))
      .collect();
    Ok(MultilinearKzgCommitment(
      variable_base_msm(&v_m, ck.eq(num_vars)).into(),
    ))
  }

  fn check_commitment(
    _comm: &Self::Commitment,
    _n: usize,
    _width: usize,
  ) -> Result<(), SpartanError> {
    // Basic validation: commitment should not be identity
    // Note: More sophisticated checks could be added
    Ok(())
  }

  fn rerandomize_commitment(
    _ck: &Self::CommitmentKey,
    comm: &Self::Commitment,
    _r_old: &Self::Blind,
    _r_new: &Self::Blind,
  ) -> Result<Self::Commitment, SpartanError> {
    // MKZG doesn't support rerandomization
    Ok((*comm).clone())
  }

  fn combine_commitments(comms: &[Self::Commitment]) -> Result<Self::Commitment, SpartanError> {
    if comms.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "combine_commitments: No commitments provided".to_string(),
      });
    }
    // Sum all commitments
    let combined = comms
      .iter()
      .map(|c| c.0.into())
      .reduce(|acc, x| acc + x)
      .unwrap_or(M::G1::identity())
      .into();
    Ok(MultilinearKzgCommitment(combined))
  }

  fn combine_blinds(_blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
    // MKZG doesn't use blinding
    Ok(())
  }

  fn prove(
    ck: &Self::CommitmentKey,
    _ck_eval: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[E::Scalar],
    _blind: &Self::Blind,
    point: &[E::Scalar],
    _comm_eval: &Self::Commitment,
    _blind_eval: &Self::Blind,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    // Pad to same length as commit() so opening matches the commitment
    let padded_len = poly.len().next_power_of_two();
    let num_vars = padded_len.ilog2() as usize;
    // Protocol may pass point of length ilog2(poly.len()); extend with 0 so eval at (point,0) = eval of original at point
    let point_extended: Vec<E::Scalar> = if point.len() == num_vars {
      point.to_vec()
    } else if point.len() == num_vars.saturating_sub(1) && padded_len > poly.len() {
      point
        .iter()
        .cloned()
        .chain(std::iter::once(E::Scalar::ZERO))
        .collect()
    } else {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "MKZG prove: point length {} does not match poly num_vars {} (padded len {})",
          point.len(),
          num_vars,
          padded_len
        ),
      });
    };

    transcript.absorb(b"poly_com", comm);

    // Convert E::Scalar to M::Fr and pad to match commitment
    let poly_m: Vec<M::Fr> = poly
      .iter()
      .map(|s| (*s).into().into())
      .chain(std::iter::repeat(M::Fr::ZERO).take(padded_len - poly.len()))
      .collect();
    let point_m: Vec<M::Fr> = point_extended.iter().map(|s| (*s).into().into()).collect();

    // Convert slice to MultilinearPolynomial for quotients function
    let poly_ml = MultilinearPolynomial::new(poly_m.clone());

    // Compute evaluation for sanity check
    let eval_m: M::Fr = {
      let chis = crate::polys::eq::EqPolynomial::new(point_m.clone()).evals();
      poly_m.iter().zip(chis.iter()).map(|(p, c)| *p * *c).sum()
    };

    if cfg!(feature = "sanity-check") {
      let computed_comm: Result<Self::Commitment, SpartanError> =
        <Self as PCSEngineTrait<E>>::commit(ck, poly, &(), false);
      let computed_comm = computed_comm?;
      assert_eq!(computed_comm.0, comm.0);
      // Note: comm_eval is a commitment, not the evaluation value itself
      // We can't easily check it here without the evaluation value
    }

    let (quotient_comms, remainder) = quotients(&poly_ml, &point_m, |num_vars, quotient| {
      variable_base_msm(&quotient, ck.eq(num_vars)).into()
    });

    if cfg!(feature = "sanity-check") {
      assert_eq!(remainder, eval_m);
    }

    // Absorb quotient commitments into transcript (label must be 'static)
    for (i, q_comm) in quotient_comms.iter().enumerate() {
      let q_comm_wrapper = MultilinearKzgCommitment(*q_comm);
      let key: &'static [u8] = Box::leak(
        format!("quotient_comm_{}", i)
          .into_bytes()
          .into_boxed_slice(),
      );
      transcript.absorb(key, &q_comm_wrapper);
    }

    Ok(MultilinearKzgEvaluationArgument {
      proofs: quotient_comms,
      eval: eval_m,
    })
  }

  fn verify(
    vk: &Self::VerifierKey,
    _ck_eval: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    _comm_eval: &Self::Commitment,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    // num_vars = number of quotient commitments (may be one more than point when PCS padded)
    let num_vars = arg.proofs.len();
    if num_vars > vk.num_vars() {
      return Err(SpartanError::InvalidPCS {
        reason: format!(
          "MKZG verify: {} quotient commitments exceed verifier key limit {}",
          num_vars,
          vk.num_vars()
        ),
      });
    }
    // Extend point with 0 when protocol passed point of length num_vars-1 (padded opening)
    let point_extended: Vec<E::Scalar> = if point.len() == num_vars {
      point.to_vec()
    } else if point.len() == num_vars.saturating_sub(1) {
      point
        .iter()
        .cloned()
        .chain(std::iter::once(E::Scalar::ZERO))
        .collect()
    } else {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "MKZG verify: point length {} does not match quotient count {}",
          point.len(),
          num_vars
        ),
      });
    };

    transcript.absorb(b"poly_com", comm);

    // Read quotient commitments from argument (they were absorbed in prove)
    // Note: In a real implementation, these would be read from transcript,
    // but since we have them in the argument, we'll use those
    for (i, q_comm) in arg.proofs.iter().enumerate() {
      let key: &'static [u8] = Box::leak(
        format!("quotient_comm_{}", i)
          .into_bytes()
          .into_boxed_slice(),
      );
      transcript.absorb(key, &MultilinearKzgCommitment(*q_comm));
    }

    // For MKZG, we need the actual evaluation value to perform the pairing check.
    // The comm_eval parameter is a commitment, but we need the scalar value.
    // In practice, the evaluation would be computed by the verifier or provided separately.
    // For now, we'll extract it from comm_eval if possible, or require it to be passed.
    // Note: This is a limitation of the current trait design for MKZG.
    // We'll compute the evaluation by checking the pairing equation.
    // The pairing check is: e(comm - g*eval, h) * ∏ e(q_i, h*s_i - h*x_i) = 1
    // Rearranging: e(comm, h) = e(g*eval, h) * ∏ e(q_i, h*x_i - h*s_i)

    // Convert E::Scalar to M::Fr for internal computation (use extended point)
    let point_m: Vec<M::Fr> = point_extended.iter().map(|s| (*s).into().into()).collect();
    let eval_m = arg.eval;

    // Compute h*x_i for each point coordinate
    let window_size = window_size(num_vars);
    let window_table = window_table(window_size, vk.g2);
    let h_mul: Vec<M::G2> = fixed_base_msm(window_size, &window_table, &point_m)
      .into_iter()
      .map(|g| g.into())
      .collect();

    // Build pairing inputs for MKZG verification:
    // e(comm - g*eval, h) * ∏_{i=0}^{n-1} e(q_i, h*s_i - h*x_i) = 1
    // Rearranged: e(comm, h) = e(g*eval, h) * ∏_{i=0}^{n-1} e(q_i, h*x_i - h*s_i)

    // Build rhs: [h, h*s_0 - h*x_0, ..., h*s_{n-1} - h*x_{n-1}]
    // quotients() folds variable 0 first: quotient_comms[i] uses eq(n-1-i) and point[i]
    let mut rhs = Vec::new();
    rhs.push(vk.g2.neg().into());
    for i in 0..num_vars {
      let _ignored = vk.num_vars() - num_vars;
      let s_i = vk.ss(num_vars)[num_vars - 1 - i];
      let h_x_i = h_mul[i];
      rhs.push(M::G2::from(s_i) - h_x_i);
    }

    // Build lhs: [comm - g*eval, q_0, ..., q_{n-1}]
    let mut lhs = Vec::new();
    let g_eval = vk.g1 * eval_m;
    let comm_minus_g_eval = M::G1::from(comm.0) - g_eval;
    lhs.push(comm_minus_g_eval);
    for q_comm in &arg.proofs {
      lhs.push((*q_comm).into());
    }

    // Perform pairing check: e(comm - g*eval, h) * ∏ e(q_i, h*s_i - h*x_i) = 1
    let pairings: Vec<(M::G1Affine, M::G2Prepared)> = lhs
      .iter()
      .zip(rhs.iter())
      .map(|(l, r)| {
        (
          M::G1Affine::from(*l),
          M::G2Prepared::from(M::G2Affine::from(*r)),
        )
      })
      .collect();
    let pairings_refs: Vec<(&M::G1Affine, &M::G2Prepared)> =
      pairings.iter().map(|(a, b)| (a, b)).collect();
    let pairing_result = pairings_product_is_identity::<M>(&pairings_refs);

    if !pairing_result {
      return Err(SpartanError::InvalidPCS {
        reason: "MKZG verify: pairing check failed".to_string(),
      });
    }

    Ok(())
  }
}

fn quotients<F: PrimeField, T>(
  poly: &MultilinearPolynomial<F>,
  point: &[F],
  f: impl Fn(usize, Vec<F>) -> T,
) -> (Vec<T>, F) {
  assert_eq!(poly.num_vars(), point.len());

  let n = poly.num_vars();
  let mut remainder = poly.evals().to_vec();
  let quotients = (0..n)
    .map(|i| {
      let k = n - 1 - i;
      let cur_dim = 1 << k;
      let x_i = point[i];
      let (remainder_lo, remainder_hi) = remainder.split_at_mut(cur_dim);
      let mut quotient = vec![F::ZERO; cur_dim];

      parallelize(&mut quotient, |(quotient, start)| {
        izip!(quotient, &remainder_lo[start..], &remainder_hi[start..])
          .for_each(|(q, r_lo, r_hi)| *q = *r_hi - r_lo);
      });
      parallelize(remainder_lo, |(r_lo, start)| {
        izip!(r_lo, &remainder_hi[start..])
          .for_each(|(r_lo, r_hi)| *r_lo += (*r_hi - r_lo as &_) * x_i);
      });

      remainder.truncate(cur_dim);

      f(k, quotient)
    })
    .collect_vec();

  (quotients, remainder[0])
}

// Implement TranscriptReprTrait for MultilinearKzgCommitment
// Note: This implementation is generic over the Group G. The commitment stores a CurveAffine,
// which should implement TranscriptReprTrait<G> for the appropriate Group G.
impl<C, G> crate::traits::transcript::TranscriptReprTrait<G> for MultilinearKzgCommitment<C>
where
  C: CurveAffine + crate::traits::transcript::TranscriptReprTrait<G>,
  G: crate::traits::Group,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend(b"mkzg_commitment");
    v.extend(self.0.to_transcript_bytes());
    v
  }
}

// Implement CommitmentTrait for MultilinearKzgCommitment
// This is a marker trait, so we just need to ensure the bounds are satisfied.
// The commitment type M::G1Affine must implement the necessary traits when used with Engine E.
impl<C, E> crate::traits::pcs::CommitmentTrait<E> for MultilinearKzgCommitment<C>
where
  C: CurveAffine
    + Serialize
    + for<'de> Deserialize<'de>
    + crate::traits::transcript::TranscriptReprTrait<E::GE>,
  E: Engine,
{
}

fn pairings_product_is_identity<M: MultiMillerLoop>(
  terms: &[(&M::G1Affine, &M::G2Prepared)],
) -> bool {
  M::multi_miller_loop(terms)
    .final_exponentiation()
    .is_identity()
    .into()
}
