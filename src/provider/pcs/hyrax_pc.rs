//! This module implements the Hyrax polynomial commitment scheme
#[allow(unused)]
use crate::{
  Blind, Commitment, CommitmentKey,
  errors::SpartanError,
  math::Math,
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  provider::{
    pcs::ipa::{
      InnerProductArgumentLinear, InnerProductInstance, InnerProductWitness, inner_product,
    },
    traits::{DlogGroup, DlogGroupExt},
  },
  start_span,
  traits::{
    Engine,
    pcs::{CommitmentTrait, FoldingEngineTrait, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::marker::PhantomData;
use ff::{Field, PrimeField};
use num_integer::div_ceil;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{info, info_span};

type AffineGroupElement<E> = <<E as Engine>::GE as DlogGroup>::AffineGroupElement;

/// A type that holds commitment generators for Hyrax commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCommitmentKey<E: Engine>
where
  E::GE: DlogGroup,
{
  num_cols: usize,
  ck: Vec<AffineGroupElement<E>>,
  h: AffineGroupElement<E>,
  ck_s: AffineGroupElement<E>,
}

/// A type that holds the verifier key for Hyrax commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxVerifierKey<E: Engine>
where
  E::GE: DlogGroup,
{
  num_cols: usize,
  ck: Vec<AffineGroupElement<E>>,
  h: AffineGroupElement<E>,
  ck_s: AffineGroupElement<E>,
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
  blind: Vec<E::Scalar>,
}

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyraxPCS<E: Engine> {
  _p: PhantomData<E>,
}

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxEvaluationArgument<E: Engine>
where
  E::GE: DlogGroupExt,
{
  ipa: InnerProductArgumentLinear<E>,
}

impl<E: Engine> PCSEngineTrait<E> for HyraxPCS<E>
where
  E::GE: DlogGroupExt,
{
  type CommitmentKey = HyraxCommitmentKey<E>;
  type VerifierKey = HyraxVerifierKey<E>;
  type Commitment = HyraxCommitment<E>;
  type PartialCommitment = HyraxCommitment<E>;
  type Blind = HyraxBlind<E>;
  type EvaluationArgument = HyraxEvaluationArgument<E>;

  fn width() -> usize {
    1024 // Hyrax PC is always 1024 columns wide
  }

  /// Derives generators for Hyrax PC, where num_vars is the number of variables in multilinear poly
  fn setup(label: &'static [u8], _n: usize) -> (Self::CommitmentKey, Self::VerifierKey) {
    let num_cols = Self::width();
    let gens = E::GE::from_label(label, num_cols + 2);
    let ck = gens[..num_cols].to_vec();
    let h = gens[num_cols];
    let ck_s = gens[num_cols + 1];

    let vk = Self::VerifierKey {
      num_cols,
      ck: ck.clone(),
      h,
      ck_s,
    };

    let ck = Self::CommitmentKey {
      num_cols,
      ck,
      h,
      ck_s,
    };

    (ck, vk)
  }

  fn blind(ck: &Self::CommitmentKey, n: usize) -> Self::Blind {
    let num_rows = div_ceil(n, ck.num_cols);

    HyraxBlind {
      blind: (0..num_rows)
        .map(|_| E::Scalar::random(&mut OsRng))
        .collect::<Vec<E::Scalar>>(),
    }
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    r: &Self::Blind,
    is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    let n = v.len();

    // compute the expected number of columns
    let num_cols = ck.num_cols;
    let num_rows = div_ceil(n, num_cols);

    let comm = (0..num_rows)
      .into_par_iter()
      .map(|i| {
        let upper = i.saturating_mul(num_cols).saturating_add(num_cols);
        let lower = i.saturating_mul(num_cols);
        let scalars = if upper > n {
          &v[lower..]
        } else {
          &v[lower..upper]
        };

        let msm_result = if !is_small {
          E::GE::vartime_multiscalar_mul(scalars, &ck.ck[..scalars.len()], false)?
        } else {
          let scalars_small = scalars
            .par_iter()
            .map(|s| s.to_repr().as_ref()[0] as u64)
            .collect::<Vec<_>>();
          E::GE::vartime_multiscalar_mul_small(
            &scalars_small,
            &ck.ck[..scalars_small.len()],
            false,
          )?
        };
        Ok(msm_result + <E::GE as DlogGroup>::group(&ck.h) * r.blind[i])
      })
      .collect::<Result<Vec<_>, _>>()?;

    Ok(HyraxCommitment { comm })
  }

  fn commit_partial(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    r: &Self::Blind,
    is_small: bool,
  ) -> Result<(Self::PartialCommitment, Self::Blind), SpartanError> {
    // commit to the vector using the provided blinds
    let partial = Self::commit(ck, v, r, is_small)?;

    // check how many blinds were used
    let r_remaining = HyraxBlind {
      blind: r.blind[partial.comm.len()..].to_vec(),
    };

    // return the partial commitment and the remaining blinds
    Ok((partial, r_remaining))
  }

  fn check_partial(comm: &Self::PartialCommitment, n: usize) -> Result<(), SpartanError> {
    let num_rows = div_ceil(n, Self::width());
    if comm.comm.len() != num_rows {
      return Err(SpartanError::InvalidCommitmentLength {
        reason: format!(
          "InvalidCommitmentLength: actual: {}, expected: {}",
          comm.comm.len(),
          num_rows
        ),
      });
    }
    Ok(())
  }

  fn combine_partial(
    partial_comms: &[Self::PartialCommitment],
  ) -> Result<Self::Commitment, SpartanError> {
    if partial_comms.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "combine_partial: No partial commitments provided".to_string(),
      });
    }
    // combine comm from each partial commitment
    let comm = partial_comms
      .iter()
      .flat_map(|pc| pc.comm.clone())
      .collect::<Vec<_>>();
    Ok(HyraxCommitment { comm })
  }

  fn combine_blinds(blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
    if blinds.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "combine_blinds: No blinds provided".to_string(),
      });
    }
    let mut blinds_comb = Vec::new();
    for b in blinds {
      blinds_comb.extend_from_slice(&b.blind);
    }
    Ok(HyraxBlind { blind: blinds_comb })
  }

  fn prove(
    ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[E::Scalar],
    blind: &Self::Blind,
    point: &[E::Scalar],
  ) -> Result<(E::Scalar, Self::EvaluationArgument), SpartanError> {
    let n = poly.len();
    let (_setup_span, setup_t) = start_span!("hyrax_prove_prep");
    if n != (2usize).pow(point.len() as u32) {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "Hyrax prove: Expected {} elements in poly, got {}",
          (2_usize).pow(point.len() as u32),
          n
        ),
      });
    }

    transcript.absorb(b"poly_com", comm);

    let num_cols = ck.num_cols;
    let num_rows = div_ceil(n, num_cols);

    let (num_vars_rows, _) = (num_rows.log_2(), num_cols.log_2());

    let (comm_LZ, R, LZ, r_LZ) = if num_vars_rows == 0 {
      let comm_LZ = comm.comm[0];
      let R = EqPolynomial::new(point.to_vec()).evals();
      let LZ = poly.to_vec();
      let r_LZ = blind.blind[0];

      (comm_LZ, R, LZ, r_LZ)
    } else {
      let (L, R) = rayon::join(
        || EqPolynomial::new(point[..num_vars_rows].to_vec()).evals(),
        || EqPolynomial::new(point[num_vars_rows..].to_vec()).evals(),
      );

      info!(elapsed_ms = %setup_t.elapsed().as_millis(), "hyrax_prove_prep");

      let (_bind_span, bind_t) = start_span!("hyrax_prove_bind");
      // compute the vector underneath L*Z
      // compute vector-matrix product between L and Z viewed as a matrix
      let LZ = MultilinearPolynomial::bind_with(poly, &L, R.len());
      info!(elapsed_ms = %bind_t.elapsed().as_millis(), "hyrax_prove_bind");

      let (_commit_span, commit_t) = start_span!("hyrax_prove_commit");

      let r_LZ = (0..L.len())
        .into_par_iter()
        .map(|i| L[i] * blind.blind[i])
        .reduce(|| E::Scalar::ZERO, |acc, x| acc + x);
      let comm_LZ = E::GE::vartime_multiscalar_mul(&LZ, &ck.ck[..LZ.len()], true)?
        + <E::GE as DlogGroup>::group(&ck.h) * r_LZ;

      info!(elapsed_ms = %commit_t.elapsed().as_millis(), "hyrax_prove_commit");

      (comm_LZ, R, LZ, r_LZ)
    };

    let (_ipa_span, ipa_t) = start_span!("hyrax_prove_ipa");
    // compute the evaluation of the multilinear polynomial at the point
    let eval = inner_product(&LZ, &R);

    // a dot product argument (IPA) of size R_size
    let ipa_instance = InnerProductInstance::<E>::new(&comm_LZ, &R, &eval);
    let ipa_witness = InnerProductWitness::<E>::new(&LZ, &r_LZ);
    let ipa = InnerProductArgumentLinear::<E>::prove(
      &ck.ck,
      &ck.h,
      &ck.ck_s,
      &ipa_instance,
      &ipa_witness,
      transcript,
    )?;
    info!(elapsed_ms = %ipa_t.elapsed().as_millis(), "hyrax_prove_ipa");

    Ok((eval, HyraxEvaluationArgument { ipa }))
  }

  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    let (_verify_span, verify_t) = start_span!("hyrax_pcs_verify");
    transcript.absorb(b"poly_com", comm);

    // compute L and R
    let (_lr_span, lr_t) = start_span!("hyrax_compute_lr");
    // n = 2^point.len()
    let n = (2_usize).pow(point.len() as u32);
    let num_cols = vk.num_cols;
    let num_rows = div_ceil(n, num_cols);

    let (num_vars_rows, _num_vars_cols) = (num_rows.log_2(), num_cols.log_2());

    let (comm_LZ, R) = if num_vars_rows == 0 {
      let R = EqPolynomial::new(point.to_vec()).evals();
      (comm.comm[0], R)
    } else {
      let L = EqPolynomial::new(point[..num_vars_rows].to_vec()).evals();
      let R = EqPolynomial::new(point[num_vars_rows..].to_vec()).evals();

      // compute a weighted sum of commitments and L
      // convert the commitments to affine form so we can do a multi-scalar multiplication
      let ck: Vec<_> = comm.comm.iter().map(|c| c.affine()).collect();
      let comm_LZ = E::GE::vartime_multiscalar_mul(&L, &ck[..L.len()], true)?;
      info!(elapsed_ms = %lr_t.elapsed().as_millis(), "hyrax_compute_lr");

      (comm_LZ, R)
    };

    let ipa_instance = InnerProductInstance::<E>::new(&comm_LZ, &R, eval);

    let result = arg
      .ipa
      .verify(&vk.ck, &vk.h, &vk.ck_s, R.len(), &ipa_instance, transcript);

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "hyrax_pcs_verify");
    result
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

impl<E: Engine> FoldingEngineTrait<E> for HyraxPCS<E>
where
  E::GE: DlogGroupExt,
{
  fn fold_commitments(
    comms: &[Self::Commitment],
    weights: &[E::Scalar],
  ) -> Result<Self::Commitment, SpartanError> {
    if comms.is_empty() || weights.is_empty() || comms.len() != weights.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: "fold_commitments: Commitments and weights must have the same length".to_string(),
      });
    }

    // scale ith commitment by the ith weight
    let folded_comm = comms
      .par_iter()
      .zip(weights.par_iter())
      .map(|(comm, weight)| {
        // scale each commitment by the corresponding weight
        comm
          .comm
          .par_iter()
          .map(|c| *c * weight)
          .collect::<Vec<_>>()
      })
      .reduce(
        || vec![E::GE::zero(); comms[0].comm.len()],
        |acc, x| {
          acc
            .par_iter()
            .zip(x.par_iter())
            .map(|(a, b)| *a + *b)
            .collect::<Vec<_>>()
        },
      );

    Ok(Self::Commitment { comm: folded_comm })
  }

  fn fold_blinds(
    blinds: &[Self::Blind],
    weights: &[<E as Engine>::Scalar],
  ) -> Result<Self::Blind, SpartanError> {
    if blinds.is_empty() || blinds.len() != weights.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: "fold_blinds: blinds and weights must be non-empty and same length".into(),
      });
    }
    let n = blinds[0].blind.len();
    if !blinds.iter().all(|b| b.blind.len() == n) {
      return Err(SpartanError::InvalidInputLength {
        reason: "fold_blinds: all inner blind vectors must have the same length".into(),
      });
    }

    let acc = blinds
      .par_iter()
      .zip(weights.par_iter())
      .fold(
        || vec![<E as Engine>::Scalar::ZERO; n],
        |mut local, (b, w)| {
          for (a, &x) in local.iter_mut().zip(&b.blind) {
            *a += x * *w;
          }
          local
        },
      )
      .reduce(
        || vec![E::Scalar::ZERO; n],
        |mut a, b| {
          for (ai, bi) in a.iter_mut().zip(b) {
            *ai += bi;
          }
          a
        },
      );

    Ok(Self::Blind { blind: acc })
  }
}
