//! Compressed Sparse Row (CSR) storage for variable-length lists.
//!
//! This module provides a memory-efficient data structure for storing
//! N variable-length lists with only 2 allocations total, instead of N+1
//! allocations required by `Vec<Vec<T>>`.

use std::ops::Index;

/// Compressed Sparse Row storage for variable-length lists.
///
/// Stores N variable-length lists in two contiguous arrays:
/// - `offsets[i]..offsets[i+1]` defines the slice for row i
/// - `data` contains all elements back-to-back
///
/// # Benefits over `Vec<Vec<T>>`
/// - 2 allocations total (vs N+1)
/// - Contiguous memory for cache-friendly iteration
/// - No pointer chasing
pub struct Csr<T> {
  offsets: Vec<u32>,
  data: Vec<T>,
}

impl<T> Csr<T> {
  /// Create an empty CSR with pre-allocated capacity.
  ///
  /// # Arguments
  /// * `num_rows` - Expected number of rows
  /// * `total_elements` - Expected total elements across all rows
  pub fn with_capacity(num_rows: usize, total_elements: usize) -> Self {
    let mut offsets = Vec::with_capacity(num_rows + 1);
    offsets.push(0);
    Self {
      offsets,
      data: Vec::with_capacity(total_elements),
    }
  }

  /// Append a new row with the given elements.
  pub fn push(&mut self, elements: &[T])
  where
    T: Clone,
  {
    self.data.extend_from_slice(elements);
    self.offsets.push(self.data.len() as u32);
  }
}

/// Test-only helpers.
#[cfg(test)]
impl<T> Csr<T> {
  /// Number of rows.
  #[inline]
  pub fn num_rows(&self) -> usize {
    self.offsets.len() - 1
  }

  /// Iterate over all rows as (index, slice) pairs.
  pub fn iter_rows(&self) -> impl Iterator<Item = (usize, &[T])> {
    (0..self.num_rows()).map(move |i| (i, &self[i]))
  }
}

impl<T> Index<usize> for Csr<T> {
  type Output = [T];

  #[inline]
  fn index(&self, i: usize) -> &Self::Output {
    let start = self.offsets[i] as usize;
    let end = self.offsets[i + 1] as usize;
    &self.data[start..end]
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_with_capacity() {
    let mut csr = Csr::with_capacity(3, 7);
    csr.push(&[1, 2, 3]);
    csr.push(&[4, 5]);
    csr.push(&[6, 7]);

    assert_eq!(csr.num_rows(), 3);
    assert_eq!(&csr[0], &[1, 2, 3]);
    assert_eq!(&csr[1], &[4, 5]);
    assert_eq!(&csr[2], &[6, 7]);
  }

  #[test]
  fn test_iter_rows() {
    let mut csr = Csr::with_capacity(2, 3);
    csr.push(&[10, 20]);
    csr.push(&[30]);

    let rows: Vec<_> = csr.iter_rows().collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], (0, &[10, 20][..]));
    assert_eq!(rows[1], (1, &[30][..]));
  }
}
