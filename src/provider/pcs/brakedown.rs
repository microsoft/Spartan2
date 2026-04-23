//! Most part are ported from https://github.com/conroi/lcpc and
//! https://github.com/han0110/plonkish.git.
//!
//! [GLSTW21]: https://eprint.iacr.org/2021/1043.pdf
use crate::{
  errors::SpartanError,
  hash::{Hash, Output},
  linear_code::{
    LinearCodes,
    brakedown::{BrakedownCode, BrakedownCodeSpec},
  },
  polys::eq::EqPolynomial,
  provider::{pcs::ipa::inner_product, traits::DlogGroupExt},
  traits::{
    Engine,
    pcs::{CommitmentTrait, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use ff::{Field, PrimeField};
use num_integer::div_ceil;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha3::Digest;
use std::{borrow::Cow, marker::PhantomData, mem::size_of};

/// Parameters for Brakedown polynomial commitment scheme
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "E::Scalar: Serialize",
  deserialize = "E::Scalar: DeserializeOwned"
))]
pub struct MultilinearBrakedownParam<E: Engine> {
  /// Number of variables
  pub num_vars: usize,
  /// Number of rows in the matrix
  pub num_rows: usize,
  /// Brakedown linear code specification
  pub code: BrakedownCode<E::Scalar>,
}

/// Commitment for Brakedown polynomial commitment scheme
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "E::Scalar: Serialize",
  deserialize = "E::Scalar: DeserializeOwned"
))]
pub struct MultilinearBrakedownCommitment<E: Engine, H: Hash> {
  /// Root of the Merkle tree
  pub root: Vec<u8>,
  /// Rows of the matrix (prover's side)
  #[serde(skip)]
  pub rows: Option<Vec<E::Scalar>>,
  /// Intermediate hashes of the Merkle tree (prover's side)
  #[serde(skip)]
  pub intermediate_hashes: Option<Vec<Vec<u8>>>,
  /// Phantom data for hash function
  #[serde(skip)]
  pub _p: PhantomData<fn() -> H>,
}

impl<E: Engine, H: Hash> Default for MultilinearBrakedownCommitment<E, H> {
  fn default() -> Self {
    Self {
      root: vec![],
      rows: None,
      intermediate_hashes: None,
      _p: PhantomData,
    }
  }
}

impl<E: Engine, H: Hash> PartialEq for MultilinearBrakedownCommitment<E, H> {
  fn eq(&self, other: &Self) -> bool {
    self.root == other.root
  }
}

impl<E: Engine, H: Hash> Eq for MultilinearBrakedownCommitment<E, H> {}

impl<E: Engine, H: Hash + Send + Sync> TranscriptReprTrait<E::GE>
  for MultilinearBrakedownCommitment<E, H>
where
  E::GE: DlogGroupExt,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend(b"brakedown_commitment_begin");
    v.extend(&self.root);
    v.extend(b"brakedown_commitment_end");
    v
  }
}

impl<E: Engine, H: Hash + Send + Sync> CommitmentTrait<E> for MultilinearBrakedownCommitment<E, H> where
  E::GE: DlogGroupExt
{
}

/// Blind for Brakedown polynomial commitment scheme
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
  serialize = "E::Scalar: Serialize",
  deserialize = "E::Scalar: DeserializeOwned"
))]
pub struct MultilinearBrakedownBlind<E: Engine> {
  /// Blind values
  pub blind: Vec<E::Scalar>,
}

/// Evaluation argument for Brakedown polynomial commitment scheme
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "E::Scalar: Serialize",
  deserialize = "E::Scalar: DeserializeOwned"
))]
pub struct MultilinearBrakedownEvaluationArgument<E: Engine, H: Hash> {
  /// Combined rows proof, eg: F1,F2
  pub combined_rows: Vec<Vec<E::Scalar>>,
  /// Column items proof, entries of the corresponding “column” of ˆu, namely ˆu1,j , . . . , uˆm,j
  pub column_items: Vec<Vec<E::Scalar>>,
  /// Merkle paths of ˆu1,j , . . . , uˆm,j
  pub merkle_paths: Vec<Vec<Vec<u8>>>,
  /// Columns idx Q,a random set of size ` = Θ(λ)
  pub columns_idx: Vec<usize>,
  /// Phantom data for hash function
  #[serde(skip)]
  pub _p: PhantomData<fn() -> H>,
}

/// Brakedown polynomial commitment scheme
#[derive(Debug)]
pub struct MultilinearBrakedown<E: Engine, H: Hash, S: BrakedownCodeSpec>(PhantomData<(E, H, S)>);

impl<E: Engine, H, S> Clone for MultilinearBrakedown<E, H, S>
where
  E::GE: DlogGroupExt,
  H: Hash + Send + Sync,
  S: BrakedownCodeSpec + Send + Sync,
{
  fn clone(&self) -> Self {
    Self(PhantomData)
  }
}

impl<E: Engine, H, S> PCSEngineTrait<E> for MultilinearBrakedown<E, H, S>
where
  E::GE: DlogGroupExt,
  E::Scalar: Serialize + DeserializeOwned,
  H: Hash + Send + Sync,
  S: BrakedownCodeSpec + Send + Sync,
{
  type CommitmentKey = MultilinearBrakedownParam<E>;
  type VerifierKey = MultilinearBrakedownParam<E>;
  type Commitment = MultilinearBrakedownCommitment<E, H>;
  type Blind = MultilinearBrakedownBlind<E>;
  type EvaluationArgument = MultilinearBrakedownEvaluationArgument<E, H>;

  fn setup(
    _label: &'static [u8],
    n: usize,
    _width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    assert!(n.is_power_of_two());
    let num_vars = n.ilog2() as usize;
    let mut rng = OsRng;
    let brakedown =
      BrakedownCode::new_multilinear::<S>(num_vars, 20.min((1 << num_vars) - 1), &mut rng);

    let param = MultilinearBrakedownParam::<E> {
      num_vars,
      num_rows: (1 << num_vars) / brakedown.row_len(),
      code: brakedown,
    };
    (param.clone(), param)
  }

  fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind {
    MultilinearBrakedownBlind { blind: vec![] }
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    _r: &Self::Blind,
    _is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    let n = v.len();
    let expected_n = 1 << ck.num_vars;
    if n > expected_n {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("Brakedown commit: Expected at most {expected_n} elements, got {n}"),
      });
    }

    let mut v_padded;
    let v = if n < expected_n {
      v_padded = v.to_vec();
      v_padded.resize(expected_n, E::Scalar::ZERO);
      &v_padded
    } else {
      v
    };

    let row_len = ck.code.row_len();
    let codeword_len = ck.code.codeword_len();
    // Define the encode metrics(expressed in vector)
    let mut rows = vec![E::Scalar::ZERO; ck.num_rows * codeword_len];

    // Encode rows using Brakedown code
    let num_threads = rayon::current_num_threads();
    let chunk_size = div_ceil(ck.num_rows, num_threads);
    // Convert the coeffs and encode one into metrics.
    rows
      .chunks_exact_mut(chunk_size * codeword_len)
      .zip(v.chunks_exact(chunk_size * row_len))
      .par_bridge()
      .for_each(|(rows_chunk, evals_chunk)| {
        // Encode each eval into row.
        for (row, evals) in rows_chunk
          .chunks_exact_mut(codeword_len)
          .zip(evals_chunk.chunks_exact(row_len))
        {
          row[..evals.len()].copy_from_slice(evals);
          ck.code.encode(row);
        }
      });

    // Hash columns
    let depth = codeword_len.next_power_of_two().ilog2() as usize;
    let mut hashes = vec![Output::<H>::default(); (2 << depth) - 1];
    hashes[..codeword_len]
      .par_iter_mut()
      .enumerate()
      .for_each(|(column, hash)| {
        let mut hasher = H::new();
        rows
          .iter()
          .skip(column)
          .step_by(codeword_len)
          .for_each(|item| hasher.update_field_element(item));
        *hash = hasher.finalize();
      });

    // Merklize column hashes
    let mut offset = 0;
    for width in (1..=depth).rev().map(|d| 1 << d) {
      let (input, output) = hashes[offset..].split_at_mut(width);
      let chunk_size = div_ceil(output.len(), num_threads);
      input
        .chunks(2 * chunk_size)
        .zip(output.chunks_mut(chunk_size))
        .par_bridge()
        .for_each(|(input_chunk, output_chunk)| {
          let mut hasher = H::new();
          for (input_pair, output) in input_chunk.chunks_exact(2).zip(output_chunk.iter_mut()) {
            hasher.update(&input_pair[0]);
            hasher.update(&input_pair[1]);
            *output = hasher.finalize();
            hasher = H::new();
          }
        });
      offset += width;
    }

    let (intermediate_hashes, root) = {
      let mut intermediate_hashes = hashes;
      let root = intermediate_hashes.pop().unwrap();
      (intermediate_hashes, root)
    };

    Ok(MultilinearBrakedownCommitment {
      root: root.to_vec(),
      rows: Some(rows),
      intermediate_hashes: Some(
        intermediate_hashes
          .into_iter()
          .map(|h| h.to_vec())
          .collect(),
      ),
      _p: PhantomData,
    })
  }

  fn check_commitment(
    _comm: &Self::Commitment,
    _n: usize,
    _width: usize,
  ) -> Result<(), SpartanError> {
    Ok(())
  }

  fn rerandomize_commitment(
    _ck: &Self::CommitmentKey,
    comm: &Self::Commitment,
    _r_old: &Self::Blind,
    _r_new: &Self::Blind,
  ) -> Result<Self::Commitment, SpartanError> {
    Ok(comm.clone())
  }

  fn combine_commitments(comms: &[Self::Commitment]) -> Result<Self::Commitment, SpartanError> {
    if comms.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "combine_commitments: No commitments provided".to_string(),
      });
    }
    Ok(comms[0].clone())
  }

  fn combine_blinds(blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
    if blinds.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "combine_blinds: No blinds provided".to_string(),
      });
    }
    Ok(MultilinearBrakedownBlind { blind: vec![] })
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
    let n = poly.len();
    let expected_n = 1 << ck.num_vars;
    if n > expected_n {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("Brakedown prove: Expected at most {expected_n} elements in poly, got {n}"),
      });
    }

    let mut poly_padded;
    let poly = if n < expected_n {
      poly_padded = poly.to_vec();
      poly_padded.resize(expected_n, E::Scalar::ZERO);
      &poly_padded
    } else {
      poly
    };

    if point.len() != ck.num_vars {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "Brakedown prove: Expected {} elements in point, got {}",
          ck.num_vars,
          point.len()
        ),
      });
    }

    transcript.absorb(b"poly_com", comm);

    let row_len = ck.code.row_len();
    let codeword_len = ck.code.codeword_len();

    // Compute evaluation
    let eval = {
      let chis = EqPolynomial::evals_from_points(point);
      inner_product(poly, &chis)
    };
    // if comm_eval. != eval {
    //   return Err(SpartanError::InternalError {
    //     reason: "Brakedown prove: comm_eval check failed".to_string(),
    //   });
    // }

    let (q_1, q_2) = point_to_tensor(ck.num_rows, point);
    // The F1 and F2
    let mut combined_rows_proof = Vec::new();

    // 1. Proximity Testing (Random linear combinations)
    //    compute u^0 = Σᵢ rᵢ·uᵢ
    let combine = |combined_row: &mut [E::Scalar], coeffs: &[E::Scalar]| {
      combined_row
        .par_iter_mut()
        .enumerate()
        .for_each(|(column, combined)| {
          *combined = E::Scalar::ZERO;
          coeffs
            .iter()
            .zip(poly.iter().skip(column).step_by(row_len))
            .for_each(|(coeff, eval)| {
              *combined += *coeff * *eval;
            });
        });
    };

    if ck.num_rows > 1 {
      for _ in 0..ck.code.num_proximity_testing() {
        // generate the random $r \in \mathbb{F}^m$
        let r: Vec<E::Scalar> = (0..ck.num_rows)
          .map(|_| transcript.squeeze(b"proximity_challenge"))
          .collect::<Result<Vec<_>, SpartanError>>()?;
        let mut combined_row = vec![E::Scalar::ZERO; row_len];
        // compute F1(u^0), where $u^0 = \sum r_i u_i$, to ensure $\text{Enc}(u^0)[j] = \sum r_i \hat{u}_i[j]$
        combine(&mut combined_row, &r);
        transcript.absorb(b"combined_row", &combined_row.as_slice());
        combined_rows_proof.push(combined_row);
      }
    }

    // 2. Evaluation Consistency (Specific linear combination using q_1)
    //    compute F2(u^00), where $u^{00} = \sum q_{1,i} u_i$, ensure $g(r) = \langle u^{00}, q_2 \rangle$
    let mut q_1_combined_row = vec![E::Scalar::ZERO; row_len];
    combine(&mut q_1_combined_row, &q_1);
    transcript.absorb(b"q_1_combined_row", &q_1_combined_row.as_slice());

    // Verify consistency locally
    let computed_eval = inner_product(&q_1_combined_row, &q_2);
    if computed_eval != eval {
      return Err(SpartanError::InternalError {
        reason: "Brakedown prove: Consistency check failed".to_string(),
      });
    }
    combined_rows_proof.push(q_1_combined_row);

    // Open merkle tree
    let depth = codeword_len.next_power_of_two().ilog2() as usize;
    let mut column_items_proof = Vec::new();
    let mut merkle_paths_proof = Vec::new();
    let mut columns_proof = Vec::new();

    // Re-calculate or use cached rows/hashes
    let (rows, intermediate_hashes) =
      if let (Some(rows), Some(hashes)) = (&comm.rows, &comm.intermediate_hashes) {
        (Cow::Borrowed(rows), Cow::Borrowed(hashes))
      } else {
        let comm_re = Self::commit(ck, poly, _blind, false)?;
        (
          Cow::Owned(comm_re.rows.unwrap()),
          Cow::Owned(comm_re.intermediate_hashes.unwrap()),
        )
      };

    for _ in 0..ck.code.num_column_opening() {
      // generate Q to be a random set of size \ell = Θ(λ) with Q ⊆ [N].
      let column_idx = squeeze_challenge_idx::<E>(transcript, codeword_len);
      columns_proof.push(column_idx);

      // select the idx column from encoded metrics
      let column: Vec<E::Scalar> = rows
        .iter()
        .skip(column_idx)
        .step_by(codeword_len)
        .copied()
        .collect();
      column_items_proof.push(column);

      // opening merkle commit with specific column
      let mut path = Vec::new();
      let mut offset = 0;
      for (idx, width) in (1..=depth).rev().map(|d| 1 << d).enumerate() {
        let neighbor_idx = (column_idx >> idx) ^ 1;
        path.push(intermediate_hashes[offset + neighbor_idx].clone());
        offset += width;
      }
      merkle_paths_proof.push(path);
    }

    Ok(MultilinearBrakedownEvaluationArgument {
      combined_rows: combined_rows_proof,
      column_items: column_items_proof,
      merkle_paths: merkle_paths_proof,
      columns_idx: columns_proof,
      _p: PhantomData,
    })
  }

  fn verify(
    vk: &Self::VerifierKey,
    _ck_eval: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    comm_eval: &Self::Commitment,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    if point.len() != vk.num_vars {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "Brakedown verify: Expected {} elements in point, got {}",
          vk.num_vars,
          point.len()
        ),
      });
    }

    transcript.absorb(b"poly_com", comm);

    let row_len = vk.code.row_len();
    let codeword_len = vk.code.codeword_len();

    // The evaluation should be in comm_eval.rows[0]
    let eval = if let Some(rows) = &comm_eval.rows {
      if !rows.is_empty() {
        rows[0]
      } else {
        return Err(SpartanError::InvalidPCS {
          reason: "Empty rows in comm_eval".to_string(),
        });
      }
    } else {
      return Err(SpartanError::InvalidPCS {
        reason: "Missing evaluation in comm_eval".to_string(),
      });
    };

    let (q_1, q_2) = point_to_tensor(vk.num_rows, point);
    let mut all_coeffs = Vec::new();
    let num_proximity = if vk.num_rows > 1 {
      vk.code.num_proximity_testing()
    } else {
      0
    };

    if arg.combined_rows.len() != num_proximity + 1 {
      return Err(SpartanError::InvalidPCS {
        reason: format!(
          "Brakedown verify: Expected {} combined_rows, got {}",
          num_proximity + 1,
          arg.combined_rows.len()
        ),
      });
    }

    if vk.num_rows > 1 {
      // Get proximity challenges from transcript
      for i in 0..num_proximity {
        let coeffs: Vec<E::Scalar> = (0..vk.num_rows)
          .map(|_| transcript.squeeze(b"proximity_challenge"))
          .collect::<Result<Vec<_>, SpartanError>>()?;
        transcript.absorb(b"combined_row", &arg.combined_rows[i].as_slice());
        all_coeffs.push(coeffs);
      }
    }

    // Get evaluation challenge (q_1)
    transcript.absorb(
      b"q_1_combined_row",
      &arg.combined_rows[num_proximity].as_slice(),
    );
    all_coeffs.push(q_1);

    // Encode all combined rows
    let mut encoded_combined_rows = Vec::new();
    for row in &arg.combined_rows {
      if row.len() < row_len {
        return Err(SpartanError::InvalidPCS {
          reason: "Invalid combined_row length".to_string(),
        });
      }
      let mut encoded = vec![E::Scalar::ZERO; codeword_len];
      encoded[..row_len].copy_from_slice(&row[..row_len]);
      vk.code.encode(&mut encoded);
      encoded_combined_rows.push(encoded);
    }

    // Verify merkle tree openings
    let depth = codeword_len.next_power_of_two().ilog2() as usize;
    if arg.columns_idx.len() != vk.code.num_column_opening() {
      return Err(SpartanError::InvalidPCS {
        reason: "Invalid number of column openings".to_string(),
      });
    }

    for (i, &column) in arg.columns_idx.iter().enumerate() {
      if column >= codeword_len {
        return Err(SpartanError::InvalidPCS {
          reason: "Invalid column index".to_string(),
        });
      }

      let items = &arg.column_items[i];
      if items.len() != vk.num_rows {
        return Err(SpartanError::InvalidPCS {
          reason: "Invalid column_items length".to_string(),
        });
      }

      let path = &arg.merkle_paths[i];
      if path.len() != depth {
        return Err(SpartanError::InvalidPCS {
          reason: "Invalid merkle path length".to_string(),
        });
      }

      // Verify proximity and consistency for all combined rows
      for (t, coeffs) in all_coeffs.iter().enumerate() {
        let expected_item = if vk.num_rows > 1 {
          inner_product(coeffs, items)
        } else {
          items[0]
        };
        if expected_item != encoded_combined_rows[t][column] {
          return Err(SpartanError::InvalidPCS {
            reason: format!("Consistency failure for row {t}"),
          });
        }
      }

      // Verify merkle tree opening
      let mut hasher = H::new();
      let mut output = {
        for item in items.iter() {
          hasher.update_field_element(item);
        }
        hasher.finalize().to_vec()
      };
      for (idx, neighbor) in path.iter().enumerate() {
        let mut hasher = H::new();
        if (column >> idx) & 1 == 0 {
          hasher.update(&output);
          hasher.update(neighbor);
        } else {
          hasher.update(neighbor);
          hasher.update(&output);
        }
        output = hasher.finalize().to_vec();
      }
      if output != comm.root {
        return Err(SpartanError::InvalidPCS {
          reason: "Invalid merkle tree opening".to_string(),
        });
      }
    }

    // Finally verify the evaluation itself
    let final_combined_row = &arg.combined_rows[num_proximity];
    if inner_product(final_combined_row, &q_2) != eval {
      return Err(SpartanError::InvalidPCS {
        reason: "Consistency failure for final evaluation".to_string(),
      });
    }

    Ok(())
  }
}

// Helper functions
fn point_to_tensor<F: PrimeField>(num_rows: usize, point: &[F]) -> (Vec<F>, Vec<F>) {
  assert!(num_rows.is_power_of_two());
  let num_vars_rows = num_rows.ilog2() as usize;
  let (rows_pt, cols_pt) = point.split_at(num_vars_rows);
  let q_1 = EqPolynomial::new(rows_pt.to_vec()).evals();
  let q_2 = EqPolynomial::new(cols_pt.to_vec()).evals();
  (q_1, q_2)
}

fn squeeze_challenge_idx<E: Engine>(transcript: &mut E::TE, cap: usize) -> usize {
  let challenge = transcript
    .squeeze(b"column_challenge")
    .unwrap_or(E::Scalar::ZERO);
  let mut bytes = [0u8; size_of::<u32>()];
  let repr = challenge.to_repr();
  let repr_bytes = repr.as_ref();
  let len = bytes.len().min(repr_bytes.len());
  bytes[..len].copy_from_slice(&repr_bytes[..len]);
  u32::from_le_bytes(bytes) as usize % cap
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    hash::Keccak256,
    linear_code::brakedown::BrakedownCodeSpec6,
    provider::{keccak::Keccak256Transcript, pasta::pallas},
  };

  #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
  pub struct PallasBrakedownEngine;

  impl Engine for PallasBrakedownEngine {
    type Base = pallas::Base;
    type Scalar = pallas::Scalar;
    type GE = pallas::Point;
    type TE = Keccak256Transcript<Self>;
    type PCS = MultilinearBrakedown<Self, Keccak256, BrakedownCodeSpec6>;
  }

  type E = PallasBrakedownEngine;
  type H = Keccak256;
  type S = BrakedownCodeSpec6;
  type PCS = MultilinearBrakedown<E, H, S>;

  #[test]
  fn test_brakedown_pcs() {
    let n: usize = 1024;
    let num_vars = (n as u32).ilog2() as usize;
    let (ck, vk) = PCS::setup(b"test", n, 0);

    let mut rng = OsRng;
    let poly: Vec<<E as Engine>::Scalar> = (0..n)
      .map(|_| <E as Engine>::Scalar::random(&mut rng))
      .collect();
    let point: Vec<<E as Engine>::Scalar> = (0..num_vars)
      .map(|_| <E as Engine>::Scalar::random(&mut rng))
      .collect();

    let eval = {
      let chis = EqPolynomial::new(point.clone()).evals();
      inner_product(&poly, &chis)
    };

    let blind = PCS::blind(&ck, n);
    let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

    let mut transcript_p = <E as Engine>::TE::new(b"test");

    // Create comm_eval with the evaluation value (mimicking Spartan's usage)
    let comm_eval = MultilinearBrakedownCommitment {
      root: vec![],
      rows: Some(vec![eval]),
      intermediate_hashes: None,
      _p: PhantomData,
    };
    let blind_eval = PCS::blind(&ck, 1);

    let proof = PCS::prove(
      &ck,
      &ck,
      &mut transcript_p,
      &comm,
      &poly,
      &blind,
      &point,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();

    let mut transcript_v = <E as Engine>::TE::new(b"test");
    PCS::verify(
      &vk,
      &ck,
      &mut transcript_v,
      &comm,
      &point,
      &comm_eval,
      &proof,
    )
    .unwrap();
  }
}
