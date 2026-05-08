// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! # Sparse Matrices
//!
//! This module defines a custom implementation of CSR/CSC sparse matrices.
//! Specifically, we implement sparse matrix / dense vector multiplication
//! to compute the `A z`, `B z`, and `C z` in Spartan.
use ff::PrimeField;
use num_traits::{Bounded, One, Signed, Zero};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{Div, Mul};

use crate::big_num::{
  DelayedReduction, ExtensionBound, ExtensionSmallValue, SmallValueField, WideMul,
};
use crate::errors::SpartanError;

/// Threshold below which we use sequential (non-rayon) iteration.
const PARALLEL_THRESHOLD: usize = 4096;

/// Precomputed SpMV accelerator for a fixed sparse matrix.
///
/// Classifies entries by coefficient magnitude to avoid expensive field
/// multiplications for the common cases in R1CS:
/// - +/-1: just add/subtract (no multiplication)
/// - small +/-k (|k| <= 7): repeated addition (cheaper than field mul)
/// - general: full field multiplication
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecomputedSparseMatrix<F: PrimeField> {
  pub(crate) num_rows: usize,
  pub(crate) num_cols: usize,
  // Cumulative offsets into entry arrays (length = num_rows + 1).
  // Row i's entries span offsets[i]..offsets[i+1].
  // Uses u32 to halve memory vs usize (max 4B entries).
  pub(crate) off_unit_pos: Vec<u32>,
  pub(crate) off_unit_neg: Vec<u32>,
  pub(crate) off_small: Vec<u32>,
  pub(crate) off_general: Vec<u32>,
  pub(crate) unit_pos_cols: Vec<u32>,
  pub(crate) unit_neg_cols: Vec<u32>,
  pub(crate) small_cols: Vec<u32>,
  pub(crate) small_coeffs: Vec<i8>,
  pub(crate) general_cols: Vec<u32>,
  pub(crate) general_vals: Vec<F>,
}

impl<F: PrimeField> PrecomputedSparseMatrix<F> {
  /// Build from a CSR SparseMatrix by classifying entries.
  pub fn from_sparse(m: &SparseMatrix<F>) -> Self {
    let num_rows = m.indptr.len() - 1;
    let one = F::ONE;
    let neg_one = -F::ONE;

    let small_pos: Vec<F> = (2u64..=7).map(F::from).collect();
    let small_neg: Vec<F> = (2u64..=7).map(|k| -F::from(k)).collect();

    let mut off_unit_pos = Vec::with_capacity(num_rows + 1);
    let mut off_unit_neg = Vec::with_capacity(num_rows + 1);
    let mut off_small = Vec::with_capacity(num_rows + 1);
    let mut off_general = Vec::with_capacity(num_rows + 1);
    let mut unit_pos_cols: Vec<u32> = Vec::new();
    let mut unit_neg_cols: Vec<u32> = Vec::new();
    let mut small_cols: Vec<u32> = Vec::new();
    let mut small_coeffs: Vec<i8> = Vec::new();
    let mut general_cols: Vec<u32> = Vec::new();
    let mut general_vals = Vec::new();

    for ptrs in m.indptr.windows(2) {
      off_unit_pos.push(unit_pos_cols.len() as u32);
      off_unit_neg.push(unit_neg_cols.len() as u32);
      off_small.push(small_cols.len() as u32);
      off_general.push(general_cols.len() as u32);

      for (&val, &col) in m.data[ptrs[0]..ptrs[1]]
        .iter()
        .zip(&m.indices[ptrs[0]..ptrs[1]])
      {
        if val == one {
          unit_pos_cols.push(col as u32);
        } else if val == neg_one {
          unit_neg_cols.push(col as u32);
        } else if let Some(k) = small_pos.iter().position(|&v| v == val) {
          small_cols.push(col as u32);
          small_coeffs.push((k as i8) + 2);
        } else if let Some(k) = small_neg.iter().position(|&v| v == val) {
          small_cols.push(col as u32);
          small_coeffs.push(-((k as i8) + 2));
        } else {
          general_cols.push(col as u32);
          general_vals.push(val);
        }
      }
    }
    // Sentinel for last row's end
    debug_assert!(
      unit_pos_cols.len() <= u32::MAX as usize,
      "unit_pos_cols exceeds u32 range"
    );
    debug_assert!(
      unit_neg_cols.len() <= u32::MAX as usize,
      "unit_neg_cols exceeds u32 range"
    );
    debug_assert!(
      small_cols.len() <= u32::MAX as usize,
      "small_cols exceeds u32 range"
    );
    debug_assert!(
      general_cols.len() <= u32::MAX as usize,
      "general_cols exceeds u32 range"
    );
    debug_assert!(
      m.cols <= u32::MAX as usize,
      "column count exceeds u32 range"
    );
    off_unit_pos.push(unit_pos_cols.len() as u32);
    off_unit_neg.push(unit_neg_cols.len() as u32);
    off_small.push(small_cols.len() as u32);
    off_general.push(general_cols.len() as u32);

    Self {
      num_rows,
      num_cols: m.cols,
      off_unit_pos,
      off_unit_neg,
      off_small,
      off_general,
      unit_pos_cols,
      unit_neg_cols,
      small_cols,
      small_coeffs,
      general_cols,
      general_vals,
    }
  }

  #[inline(always)]
  pub(crate) fn small_mul(coeff: i8, x: F) -> F {
    let abs = coeff.unsigned_abs();
    let result = match abs {
      2 => x.double(),
      3 => x.double() + x,
      4 => x.double().double(),
      5 => x.double().double() + x,
      6 => {
        let d = x.double();
        d.double() + d
      }
      7 => {
        let d = x.double();
        d.double() + d + x
      }
      _ => unreachable!(),
    };
    if coeff < 0 { -result } else { result }
  }

  /// Get the range of unit_pos entries for a given row.
  #[inline(always)]
  pub(crate) fn range_unit_pos(&self, row: usize) -> (usize, usize) {
    (
      self.off_unit_pos[row] as usize,
      self.off_unit_pos[row + 1] as usize,
    )
  }

  /// Get the range of unit_neg entries for a given row.
  #[inline(always)]
  pub(crate) fn range_unit_neg(&self, row: usize) -> (usize, usize) {
    (
      self.off_unit_neg[row] as usize,
      self.off_unit_neg[row + 1] as usize,
    )
  }

  /// Get the range of small entries for a given row.
  #[inline(always)]
  pub(crate) fn range_small(&self, row: usize) -> (usize, usize) {
    (
      self.off_small[row] as usize,
      self.off_small[row + 1] as usize,
    )
  }

  /// Get the range of general entries for a given row.
  #[inline(always)]
  pub(crate) fn range_general(&self, row: usize) -> (usize, usize) {
    (
      self.off_general[row] as usize,
      self.off_general[row + 1] as usize,
    )
  }

  #[inline(always)]
  fn compute_row_single(&self, row: usize, v: &[F]) -> F {
    let mut sum = F::ZERO;

    let (start, end) = self.range_unit_pos(row);
    for i in start..end {
      sum += v[self.unit_pos_cols[i] as usize];
    }

    let (start, end) = self.range_unit_neg(row);
    for i in start..end {
      sum -= v[self.unit_neg_cols[i] as usize];
    }

    let (start, end) = self.range_small(row);
    for i in start..end {
      sum += Self::small_mul(self.small_coeffs[i], v[self.small_cols[i] as usize]);
    }

    let (start, end) = self.range_general(row);
    for i in start..end {
      sum += self.general_vals[i] * v[self.general_cols[i] as usize];
    }

    sum
  }

  /// Fast SpMV using precomputed coefficient classification.
  pub fn multiply_vec(&self, vector: &[F]) -> Vec<F> {
    assert_eq!(self.num_cols, vector.len(), "invalid shape");
    if self.num_rows <= PARALLEL_THRESHOLD || rayon::current_num_threads() <= 1 {
      (0..self.num_rows)
        .map(|r| self.compute_row_single(r, vector))
        .collect()
    } else {
      (0..self.num_rows)
        .into_par_iter()
        .map(|r| self.compute_row_single(r, vector))
        .collect()
    }
  }

  /// Batched SpMV: multiply the same matrix by multiple vectors in a single pass.
  /// Uses sub-batches of BATCH_SIZE to keep working set in cache.
  pub fn multiply_vec_batched(&self, vectors: &[&[F]]) -> Vec<Vec<F>> {
    let n_vecs = vectors.len();
    if n_vecs == 0 {
      return vec![];
    }
    for v in vectors {
      assert_eq!(self.num_cols, v.len(), "invalid shape");
    }

    // For small numbers of vectors or small matrices, just call single-vector multiply
    const BATCH_SIZE: usize = 4;
    if n_vecs <= 1 || self.num_rows <= 256 {
      return vectors.iter().map(|v| self.multiply_vec(v)).collect();
    }

    let mut results: Vec<Vec<F>> = (0..n_vecs).map(|_| vec![F::ZERO; self.num_rows]).collect();

    // Process vectors in sub-batches to keep working set in L3 cache
    for batch_start in (0..n_vecs).step_by(BATCH_SIZE) {
      let batch_end = (batch_start + BATCH_SIZE).min(n_vecs);
      let batch_len = batch_end - batch_start;

      #[allow(clippy::needless_range_loop)]
      for row in 0..self.num_rows {
        // unit_pos: coefficient = +1
        let (start, end) = self.range_unit_pos(row);
        for idx in start..end {
          let col = self.unit_pos_cols[idx] as usize;
          for b in 0..batch_len {
            results[batch_start + b][row] += vectors[batch_start + b][col];
          }
        }

        // unit_neg: coefficient = -1
        let (start, end) = self.range_unit_neg(row);
        for idx in start..end {
          let col = self.unit_neg_cols[idx] as usize;
          for b in 0..batch_len {
            results[batch_start + b][row] -= vectors[batch_start + b][col];
          }
        }

        // small coefficients
        let (start, end) = self.range_small(row);
        for idx in start..end {
          let col = self.small_cols[idx] as usize;
          let coeff = self.small_coeffs[idx];
          for b in 0..batch_len {
            results[batch_start + b][row] += Self::small_mul(coeff, vectors[batch_start + b][col]);
          }
        }

        // general: full field multiply
        let (start, end) = self.range_general(row);
        for idx in start..end {
          let col = self.general_cols[idx] as usize;
          let val = self.general_vals[idx];
          for b in 0..batch_len {
            results[batch_start + b][row] += val * vectors[batch_start + b][col];
          }
        }
      }
    }

    results
  }

  /// Build a FilteredSpmv containing only entries with col >= col_min.
  pub fn build_filtered(&self, col_min: usize, num_rows_used: usize) -> FilteredSpmv<F> {
    let num_rows = num_rows_used.min(self.num_rows);
    let one = F::ONE;
    let neg_one = -F::ONE;

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for row in 0..num_rows {
      let (start, end) = self.range_unit_pos(row);
      for i in start..end {
        let col = self.unit_pos_cols[i] as usize;
        if col >= col_min {
          rows.push(row as u32);
          cols.push(col as u32);
          vals.push(one);
        }
      }

      let (start, end) = self.range_unit_neg(row);
      for i in start..end {
        let col = self.unit_neg_cols[i] as usize;
        if col >= col_min {
          rows.push(row as u32);
          cols.push(col as u32);
          vals.push(neg_one);
        }
      }

      let (start, end) = self.range_small(row);
      for i in start..end {
        let col = self.small_cols[i] as usize;
        if col >= col_min {
          let coeff = self.small_coeffs[i];
          let val = Self::small_mul(coeff, one);
          rows.push(row as u32);
          cols.push(col as u32);
          vals.push(val);
        }
      }

      let (start, end) = self.range_general(row);
      for i in start..end {
        let col = self.general_cols[i] as usize;
        if col >= col_min {
          rows.push(row as u32);
          cols.push(col as u32);
          vals.push(self.general_vals[i]);
        }
      }
    }

    FilteredSpmv { rows, cols, vals }
  }
}

/// Compact representation of matrix entries filtered by column threshold.
/// Stores only entries with col >= col_min, avoiding iteration over the
/// full matrix during incremental SpMV.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilteredSpmv<F: PrimeField> {
  pub(crate) rows: Vec<u32>,
  pub(crate) cols: Vec<u32>,
  pub(crate) vals: Vec<F>,
}

impl<F: PrimeField> FilteredSpmv<F> {
  /// Accumulate filtered entries into output: out[row] += val * vector[col]
  #[inline]
  pub fn multiply_vec_add(&self, vector: &[F], out: &mut [F]) {
    for i in 0..self.rows.len() {
      out[self.rows[i] as usize] += self.vals[i] * vector[self.cols[i] as usize];
    }
  }
}

/// CSR format sparse matrix, We follow the names used by scipy.
/// Detailed explanation here: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseMatrix<F: PrimeField> {
  /// all non-zero values in the matrix
  pub data: Vec<F>,
  /// column indices
  pub indices: Vec<usize>,
  /// row information
  pub indptr: Vec<usize>,
  /// number of columns
  pub cols: usize,
}

impl<F: PrimeField> SparseMatrix<F> {
  /// Write raw bytes for digest computation (much faster than bincode Serialize).
  pub fn write_digest_bytes<W: std::io::Write>(&self, w: &mut W) -> Result<(), std::io::Error> {
    // Write lengths and cols as fixed-size le bytes
    w.write_all(&(self.data.len() as u64).to_le_bytes())?;
    w.write_all(&(self.indices.len() as u64).to_le_bytes())?;
    w.write_all(&(self.indptr.len() as u64).to_le_bytes())?;
    w.write_all(&(self.cols as u64).to_le_bytes())?;
    // Write field elements as raw repr bytes
    for d in &self.data {
      w.write_all(d.to_repr().as_ref())?;
    }
    // Write indices as raw bytes
    for idx in &self.indices {
      w.write_all(&(*idx as u64).to_le_bytes())?;
    }
    // Write indptr as raw bytes
    for ptr in &self.indptr {
      w.write_all(&(*ptr as u64).to_le_bytes())?;
    }
    Ok(())
  }

  /// 0x0 empty matrix
  pub fn empty() -> Self {
    SparseMatrix {
      data: vec![],
      indices: vec![],
      indptr: vec![0],
      cols: 0,
    }
  }

  /// Construct from the COO representation; Vec<usize(row), usize(col), F>.
  /// We assume that the rows are sorted during construction.
  #[cfg(test)]
  pub fn new(matrix: &[(usize, usize, F)], rows: usize, cols: usize) -> Self {
    let mut new_matrix = vec![vec![]; rows];
    for (row, col, val) in matrix {
      new_matrix[*row].push((*col, *val));
    }

    for row in new_matrix.iter() {
      assert!(row.windows(2).all(|w| w[0].0 < w[1].0));
    }

    let mut indptr = vec![0; rows + 1];
    for (i, col) in new_matrix.iter().enumerate() {
      indptr[i + 1] = indptr[i] + col.len();
    }

    let mut indices = vec![];
    let mut data = vec![];
    for col in new_matrix {
      let (idx, val): (Vec<_>, Vec<_>) = col.into_iter().unzip();
      indices.extend(idx);
      data.extend(val);
    }

    SparseMatrix {
      data,
      indices,
      indptr,
      cols,
    }
  }

  /// Retrieves the data for row slice [i..j] from `ptrs`.
  /// We assume that `ptrs` is indexed from `indptrs` and do not check if the
  /// returned slice is actually a valid row.
  pub fn get_row_unchecked(&self, ptrs: &[usize; 2]) -> impl Iterator<Item = (&F, &usize)> {
    self.data[ptrs[0]..ptrs[1]]
      .iter()
      .zip(&self.indices[ptrs[0]..ptrs[1]])
  }

  /// Multiply by a dense vector; uses rayon/gpu.
  ///
  /// # Errors
  /// Returns `SpartanError::InvalidInputLength` if the vector length doesn't match the matrix dimensions.
  pub fn multiply_vec(&self, vector: &[F]) -> Result<Vec<F>, SpartanError> {
    if self.cols != vector.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SparseMatrix multiply_vec: Expected {} elements in vector, got {}",
          self.cols,
          vector.len()
        ),
      });
    }

    Ok(self.multiply_vec_unchecked(vector))
  }

  /// Multiply by a dense vector; uses rayon/gpu.
  /// This does not check that the shape of the matrix/vector are compatible.
  pub fn multiply_vec_unchecked(&self, vector: &[F]) -> Vec<F> {
    let num_rows = self.indptr.len() - 1;
    if num_rows <= PARALLEL_THRESHOLD {
      self
        .indptr
        .windows(2)
        .map(|ptrs| {
          let row_ptrs = [ptrs[0], ptrs[1]];
          self
            .get_row_unchecked(&row_ptrs)
            .map(|(val, col_idx)| *val * vector[*col_idx])
            .sum()
        })
        .collect()
    } else {
      self
        .indptr
        .par_windows(2)
        .map(|ptrs| {
          let row_ptrs = [ptrs[0], ptrs[1]];
          self
            .get_row_unchecked(&row_ptrs)
            .map(|(val, col_idx)| *val * vector[*col_idx])
            .sum()
        })
        .collect()
    }
  }

  /// Multiply by a dense small-value vector and return extension-safe small values.
  ///
  /// This is used by the full-small NeutronNova accumulator path. It accumulates
  /// `field coefficient × small vector entry` with delayed reduction, reduces the
  /// row result to a field element, then enforces the Lagrange-extension bound.
  pub fn multiply_vec_small<const D: usize, SV>(
    &self,
    z: &[SV],
    lb: usize,
  ) -> Result<Vec<SV>, SpartanError>
  where
    F: SmallValueField<SV> + DelayedReduction<SV> + Sync,
    SV: WideMul + Bounded + Copy + Send + Sync + Into<SV::Product>,
    SV::Product:
      Copy + Ord + Signed + Div<Output = SV::Product> + Mul<Output = SV::Product> + One + From<i32>,
  {
    if self.cols != z.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SparseMatrix multiply_vec_small: Expected {} elements in vector, got {}",
          self.cols,
          z.len()
        ),
      });
    }

    let bound = ExtensionBound::<SV, D>::new(lb);
    let num_rows = self.indptr.len() - 1;

    let compute_row = |row: usize| -> Result<SV, SpartanError> {
      let mut acc = <F as DelayedReduction<SV>>::Accumulator::zero();
      let ptrs = [self.indptr[row], self.indptr[row + 1]];
      for (val, col_idx) in self.get_row_unchecked(&ptrs) {
        <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc, val, &z[*col_idx]);
      }
      let field_result = <F as DelayedReduction<SV>>::reduce(&acc);
      bound
        .try_to_small(&field_result)
        .ok_or_else(|| SpartanError::SmallValueOverflow {
          value: format!("0x{}", hex::encode(field_result.to_repr().as_ref())),
          context: format!(
            "SparseMatrix::multiply_vec_small row {} exceeds extension bound for D={}, lb={}",
            row, D, lb
          ),
        })
    };

    if num_rows <= PARALLEL_THRESHOLD {
      (0..num_rows).map(compute_row).collect()
    } else {
      (0..num_rows).into_par_iter().map(compute_row).collect()
    }
  }

  /// returns a custom iterator
  pub fn iter(&self) -> Iter<'_, F> {
    let mut row = 0;
    while row + 1 < self.indptr.len() && self.indptr[row + 1] == 0 {
      row += 1;
    }
    let nnz = if self.indptr.is_empty() {
      0
    } else {
      self.indptr[self.indptr.len() - 1]
    };
    Iter {
      matrix: self,
      row,
      i: 0,
      nnz,
    }
  }
}

/// Iterator for sparse matrix
pub struct Iter<'a, F: PrimeField> {
  matrix: &'a SparseMatrix<F>,
  row: usize,
  i: usize,
  nnz: usize,
}

impl<'a, F: PrimeField> Iterator for Iter<'a, F> {
  type Item = (usize, usize, F);

  fn next(&mut self) -> Option<Self::Item> {
    // are we at the end?
    if self.i == self.nnz {
      return None;
    }

    // compute current item
    let curr_item = (
      self.row,
      self.matrix.indices[self.i],
      self.matrix.data[self.i],
    );

    // advance the iterator
    self.i += 1;
    // edge case at the end
    if self.i == self.nnz {
      return Some(curr_item);
    }
    // if `i` has moved to next row
    while self.i >= self.matrix.indptr[self.row + 1] {
      self.row += 1;
    }

    Some(curr_item)
  }
}

// =============================================================================
// SmallSparseMatrix: CSR with native-int coefficients
// =============================================================================

/// CSR sparse matrix whose coefficients have been narrowed from a field
/// element to a native integer type `SV` (e.g. `i64`). Used by the
/// NeutronNova accumulator prep path so the per-row matvec can run as
/// pure `SV × SV → SV::Product` without touching field arithmetic.
#[derive(Clone, Debug)]
pub struct SmallSparseMatrix<SV> {
  /// Non-zero values, narrowed from the source `SparseMatrix<F>` data.
  pub data: Vec<SV>,
  /// Column index of each non-zero (cloned from the source matrix).
  pub indices: Vec<usize>,
  /// Row offsets (cloned from the source matrix).
  pub indptr: Vec<usize>,
  /// Number of columns.
  pub cols: usize,
}

impl<SV: Copy + Send + Sync> SmallSparseMatrix<SV> {
  /// Try to narrow each coefficient of `m` to `SV`, parallel over the
  /// `data` array. Returns `SmallValueOverflow` on the first coefficient
  /// that doesn't fit.
  pub fn try_from_field<F>(
    m: &SparseMatrix<F>,
    matrix_name: &'static str,
  ) -> Result<Self, SpartanError>
  where
    F: SmallValueField<SV> + Sync,
  {
    let data: Vec<SV> = m
      .data
      .par_iter()
      .enumerate()
      .map(|(i, v)| {
        F::try_field_to_small(v).ok_or_else(|| SpartanError::SmallValueOverflow {
          value: format!("0x{}", hex::encode(v.to_repr().as_ref())),
          context: format!(
            "matrix {} coefficient at nz-index {} does not fit small type",
            matrix_name, i,
          ),
        })
      })
      .collect::<Result<Vec<_>, _>>()?;

    Ok(Self {
      data,
      indices: m.indices.clone(),
      indptr: m.indptr.clone(),
      cols: m.cols,
    })
  }

  #[inline]
  fn get_row_unchecked(&self, ptrs: &[usize; 2]) -> impl Iterator<Item = (&SV, &usize)> {
    self.data[ptrs[0]..ptrs[1]]
      .iter()
      .zip(&self.indices[ptrs[0]..ptrs[1]])
  }
}

/// Triple of A, B, C R1CS matrices with their coefficients narrowed to `SV`.
#[derive(Clone, Debug)]
pub struct SmallR1CSCoefs<SV> {
  pub a: SmallSparseMatrix<SV>,
  pub b: SmallSparseMatrix<SV>,
  pub c: SmallSparseMatrix<SV>,
}

impl<SV: Copy + Send + Sync> SmallR1CSCoefs<SV> {
  /// Convert all three matrices in parallel; bails on the first coefficient
  /// that doesn't fit `SV`. Each inner conversion already parallelizes over
  /// its own data array, so the outer `rayon::join` overlaps the three
  /// sweeps without oversubscribing.
  pub fn try_from_matrices<F>(
    a: &SparseMatrix<F>,
    b: &SparseMatrix<F>,
    c: &SparseMatrix<F>,
  ) -> Result<Self, SpartanError>
  where
    F: SmallValueField<SV> + Sync,
  {
    let (ra, (rb, rc)) = rayon::join(
      || SmallSparseMatrix::try_from_field(a, "A"),
      || {
        rayon::join(
          || SmallSparseMatrix::try_from_field(b, "B"),
          || SmallSparseMatrix::try_from_field(c, "C"),
        )
      },
    );
    Ok(Self {
      a: ra?,
      b: rb?,
      c: rc?,
    })
  }
}

/// Triple `(A·z, B·z, C·z)` matvec where the matrix coefficients **and**
/// the witness vector `z` are both small native integers.
///
/// Inner loop is `*acc += SV::wide_mul(coef, z[col])` per nonzero (via the
/// `ExtensionSmallValue::wide_mul_accumulate` trait method) into a per-row
/// `SV::Product` accumulator. After the row sums are formed, each is
/// bounded by `ExtensionBound::<SV, 2>::new(lb).max_safe()` and narrowed
/// back to `SV` via `try_narrow_from_product`. No field arithmetic, no
/// Montgomery/Barrett reduction.
///
/// `lb` is the same Lagrange-extension exponent that `vec_to_small_for_extension`
/// would use for the post-matvec narrow on the field path (i.e. `l0`).
pub fn multiply_vec_small_small_triple<SV>(
  a: &SmallSparseMatrix<SV>,
  b: &SmallSparseMatrix<SV>,
  c: &SmallSparseMatrix<SV>,
  z: &[SV],
  lb: usize,
) -> Result<(Vec<SV>, Vec<SV>, Vec<SV>), SpartanError>
where
  SV: ExtensionSmallValue,
{
  let num_rows = a.indptr.len() - 1;
  if num_rows != b.indptr.len() - 1 || num_rows != c.indptr.len() - 1 {
    return Err(SpartanError::InvalidInputLength {
      reason: format!(
        "multiply_vec_small_small_triple: A/B/C row counts disagree: {}, {}, {}",
        num_rows,
        b.indptr.len() - 1,
        c.indptr.len() - 1,
      ),
    });
  }
  if a.cols != z.len() || b.cols != z.len() || c.cols != z.len() {
    return Err(SpartanError::InvalidInputLength {
      reason: format!(
        "multiply_vec_small_small_triple: expected z.len()={}, got {}",
        a.cols,
        z.len()
      ),
    });
  }

  let compute_row = |row: usize| -> Result<(SV, SV, SV), SpartanError> {
    let mut acc_a = SV::product_zero();
    let mut acc_b = SV::product_zero();
    let mut acc_c = SV::product_zero();

    let ap = [a.indptr[row], a.indptr[row + 1]];
    for (val, col) in a.get_row_unchecked(&ap) {
      SV::wide_mul_accumulate(&mut acc_a, *val, z[*col]);
    }
    let bp = [b.indptr[row], b.indptr[row + 1]];
    for (val, col) in b.get_row_unchecked(&bp) {
      SV::wide_mul_accumulate(&mut acc_b, *val, z[*col]);
    }
    let cp = [c.indptr[row], c.indptr[row + 1]];
    for (val, col) in c.get_row_unchecked(&cp) {
      SV::wide_mul_accumulate(&mut acc_c, *val, z[*col]);
    }

    let narrow = |acc, which: char| -> Result<SV, SpartanError> {
      SV::try_narrow_from_product::<2>(acc, lb).ok_or_else(|| SpartanError::SmallValueOverflow {
        value: String::new(),
        context: format!(
          "{}z row sum at row {} exceeds extension bound for D=2",
          which, row,
        ),
      })
    };

    Ok((
      narrow(acc_a, 'A')?,
      narrow(acc_b, 'B')?,
      narrow(acc_c, 'C')?,
    ))
  };

  let rows: Vec<(SV, SV, SV)> = if num_rows <= PARALLEL_THRESHOLD {
    (0..num_rows).map(compute_row).collect::<Result<_, _>>()?
  } else {
    (0..num_rows)
      .into_par_iter()
      .map(compute_row)
      .collect::<Result<_, _>>()?
  };

  let mut az = Vec::with_capacity(num_rows);
  let mut bz = Vec::with_capacity(num_rows);
  let mut cz = Vec::with_capacity(num_rows);
  for (a_, b_, c_) in rows {
    az.push(a_);
    bz.push(b_);
    cz.push(c_);
  }
  Ok((az, bz, cz))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::PallasHyraxEngine,
    traits::{Engine, Group},
  };
  use ff::PrimeField;
  use proptest::{
    prelude::*,
    strategy::{BoxedStrategy, Just, Strategy},
  };

  type G = <PallasHyraxEngine as Engine>::GE;
  type Fr = <G as Group>::Scalar;

  /// Wrapper struct around a field element that implements additional traits
  #[derive(Clone, Debug, PartialEq, Eq)]
  pub struct FWrap<F: PrimeField>(pub F);

  impl<F: PrimeField> Copy for FWrap<F> {}

  #[cfg(not(target_arch = "wasm32"))]
  /// Trait implementation for generating `FWrap<F>` instances with proptest
  impl<F: PrimeField> Arbitrary for FWrap<F> {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
      use rand::rngs::StdRng;
      use rand_core::SeedableRng;

      let strategy = any::<[u8; 32]>()
        .prop_map(|seed| FWrap(F::random(StdRng::from_seed(seed))))
        .no_shrink();
      strategy.boxed()
    }
  }

  #[test]
  fn test_matrix_creation() {
    let matrix_data = vec![
      (0, 1, Fr::from(2)),
      (1, 2, Fr::from(3)),
      (2, 0, Fr::from(4)),
    ];
    let sparse_matrix = SparseMatrix::<Fr>::new(&matrix_data, 3, 3);

    assert_eq!(
      sparse_matrix.data,
      vec![Fr::from(2), Fr::from(3), Fr::from(4)]
    );
    assert_eq!(sparse_matrix.indices, vec![1, 2, 0]);
    assert_eq!(sparse_matrix.indptr, vec![0, 1, 2, 3]);
  }

  #[test]
  fn test_multiply_vec_small_matches_field_matvec() {
    use crate::big_num::SmallValueField;

    let matrix = SparseMatrix::<Fr>::new(
      &[
        (0, 0, Fr::from(2u64)),
        (0, 1, -Fr::from(1u64)),
        (1, 1, Fr::from(5u64)),
        (1, 2, Fr::from(3u64)),
        (2, 0, -Fr::from(4u64)),
        (2, 2, Fr::from(7u64)),
      ],
      3,
      3,
    );
    let z_small = vec![3i64, -2, 4];
    let z_field = z_small
      .iter()
      .copied()
      .map(<Fr as SmallValueField<i64>>::small_to_field)
      .collect::<Vec<_>>();

    let expected = matrix.multiply_vec(&z_field).unwrap();
    let got = matrix.multiply_vec_small::<2, i64>(&z_small, 2).unwrap();
    let expected_small = expected
      .iter()
      .map(<Fr as SmallValueField<i64>>::try_field_to_small)
      .collect::<Option<Vec<_>>>()
      .unwrap();

    assert_eq!(got, expected_small);
  }

  #[test]
  fn test_multiply_vec_small_overflow_fails_hard() {
    let matrix = SparseMatrix::<Fr>::new(&[(0, 0, Fr::from(1u64))], 1, 1);
    let err = matrix
      .multiply_vec_small::<2, i64>(&[i64::MAX], 2)
      .unwrap_err();

    assert!(matches!(err, SpartanError::SmallValueOverflow { .. }));
  }

  #[test]
  fn test_small_r1cs_triple_matches_field_matvec() {
    use crate::big_num::{SmallValueField, vec_to_small_for_extension};

    let a = SparseMatrix::<Fr>::new(
      &[
        (0, 0, Fr::from(2u64)),
        (0, 1, -Fr::from(1u64)),
        (1, 1, Fr::from(5u64)),
      ],
      2,
      3,
    );
    let b = SparseMatrix::<Fr>::new(&[(0, 2, Fr::from(3u64)), (1, 0, -Fr::from(4u64))], 2, 3);
    let c = SparseMatrix::<Fr>::new(
      &[
        (0, 0, Fr::from(1u64)),
        (0, 2, Fr::from(7u64)),
        (1, 2, -Fr::from(2u64)),
      ],
      2,
      3,
    );
    let z_small = vec![3i64, -2, 4];
    let z_field = z_small
      .iter()
      .copied()
      .map(<Fr as SmallValueField<i64>>::small_to_field)
      .collect::<Vec<_>>();

    let coefs = SmallR1CSCoefs::<i64>::try_from_matrices(&a, &b, &c).unwrap();
    let (az, bz, cz) =
      multiply_vec_small_small_triple(&coefs.a, &coefs.b, &coefs.c, &z_small, 2).unwrap();

    assert_eq!(
      az,
      vec_to_small_for_extension::<Fr, i64, 2>(&a.multiply_vec(&z_field).unwrap(), 2).unwrap()
    );
    assert_eq!(
      bz,
      vec_to_small_for_extension::<Fr, i64, 2>(&b.multiply_vec(&z_field).unwrap(), 2).unwrap()
    );
    assert_eq!(
      cz,
      vec_to_small_for_extension::<Fr, i64, 2>(&c.multiply_vec(&z_field).unwrap(), 2).unwrap()
    );
  }

  #[test]
  fn test_small_r1cs_triple_row_sum_overflow_fails_hard() {
    let a = SmallSparseMatrix {
      data: vec![1i64],
      indices: vec![0],
      indptr: vec![0, 1],
      cols: 1,
    };
    let empty = SmallSparseMatrix {
      data: Vec::<i64>::new(),
      indices: Vec::new(),
      indptr: vec![0, 0],
      cols: 1,
    };

    let err = multiply_vec_small_small_triple(&a, &empty, &empty, &[i64::MAX], 1).unwrap_err();
    assert!(matches!(err, SpartanError::SmallValueOverflow { .. }));
  }

  #[test]
  fn test_matrix_vector_multiplication() {
    let matrix_data = vec![
      (0, 1, Fr::from(2)),
      (0, 2, Fr::from(7)),
      (1, 2, Fr::from(3)),
      (2, 0, Fr::from(4)),
    ];
    let sparse_matrix = SparseMatrix::<Fr>::new(&matrix_data, 3, 3);
    let vector = vec![Fr::from(1), Fr::from(2), Fr::from(3)];

    let result = sparse_matrix.multiply_vec(&vector);

    assert_eq!(
      result.unwrap(),
      vec![Fr::from(25), Fr::from(9), Fr::from(4)]
    );
  }

  fn coo_strategy() -> BoxedStrategy<Vec<(usize, usize, FWrap<Fr>)>> {
    let coo_strategy = any::<FWrap<Fr>>().prop_flat_map(|f| (0usize..100, 0usize..100, Just(f)));
    proptest::collection::vec(coo_strategy, 10).boxed()
  }

  proptest! {
      #[test]
      fn test_matrix_iter(mut coo_matrix in coo_strategy()) {
        // process the randomly generated coo matrix
        coo_matrix.sort_by_key(|(row, col, _val)| (*row, *col));
        coo_matrix.dedup_by_key(|(row, col, _val)| (*row, *col));
        let coo_matrix = coo_matrix.into_iter().map(|(row, col, val)| { (row, col, val.0) }).collect::<Vec<_>>();

        let matrix = SparseMatrix::new(&coo_matrix, 100, 100);

        prop_assert_eq!(coo_matrix, matrix.iter().collect::<Vec<_>>());
    }
  }
}
