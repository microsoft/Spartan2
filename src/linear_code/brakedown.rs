//! Implementation of linear codes specification described in [GLSTW21].
//! Most part are ported from https://github.com/conroi/lcpc and
//! https://github.com/han0110/plonkish.git.
//!
//! [GLSTW21]: https://eprint.iacr.org/2021/1043.pdf

use itertools::{Itertools, chain};

use crate::linear_code::{
  LinearCodes,
  arithmetic::{horner, steps},
};

use ff::{Field, PrimeField};
use rand::{Rng, RngCore, distributions::Uniform};
use serde::{Deserialize, Serialize};
use std::{
  cmp::{max, min},
  collections::BTreeSet,
  fmt::Debug,
  iter,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrakedownCode<F> {
  row_len: usize,
  codeword_len: usize,
  num_column_opening: usize,
  num_proximity_testing: usize,
  a: Vec<SparseMatrix<F>>,
  b: Vec<SparseMatrix<F>>,
}

impl<F: PrimeField> BrakedownCode<F> {
  pub fn proof_size<S: BrakedownCodeSpec>(n_0: usize, c: usize, r: usize) -> usize {
    let log2_q = F::NUM_BITS as usize;
    let num_ldt = S::num_proximity_testing(log2_q, c, n_0);
    let num_openings = S::num_column_opening();
    let codeword_len = S::codeword_len(log2_q, c, n_0);
    let merkle_path_len = codeword_len.next_power_of_two().ilog2() as usize;

    // Estimate total number of field elements and hashes in the proof
    // (1 + num_ldt) * c: Elements from combined rows
    // num_openings * (r + merkle_path_len): Elements from column openings + Merkle paths
    (1 + num_ldt) * c + num_openings * (r + merkle_path_len)
  }

  pub fn new_multilinear<S: BrakedownCodeSpec>(
    num_vars: usize,
    n_0: usize,
    rng: impl RngCore,
  ) -> Self {
    assert!(1 << num_vars > n_0);

    let log2_q = F::NUM_BITS as usize;
    let min_log2_c = (n_0 + 1).next_power_of_two().ilog2() as usize;

    // Search for the best row_len (c) and num_rows (r)
    // Following BrakingBase [NST24]'s suggestion: set r = O(log n)
    // The current loop naturally favors larger c (smaller r) due to the large weight of num_openings.
    let (_, row_len) =
      (min_log2_c..=num_vars).fold((usize::MAX, 0), |(min_proof_size, best_c), log2_c| {
        let c = 1 << log2_c;
        let r = 1 << (num_vars - log2_c);
        let proof_size = Self::proof_size::<S>(n_0, c, r);
        if proof_size <= min_proof_size {
          (proof_size, c)
        } else {
          (min_proof_size, best_c)
        }
      });
    let codeword_len = S::codeword_len(log2_q, row_len, n_0);
    let num_column_opening = S::num_column_opening();
    let num_proximity_testing = S::num_proximity_testing(log2_q, row_len, n_0);
    let (a, b) = S::matrices(log2_q, row_len, n_0, rng);

    Self {
      row_len,
      codeword_len,
      num_column_opening,
      num_proximity_testing,
      a,
      b,
    }
  }
}

impl<F: PrimeField> LinearCodes<F> for BrakedownCode<F> {
  fn row_len(&self) -> usize {
    self.row_len
  }

  fn codeword_len(&self) -> usize {
    self.codeword_len
  }

  fn num_column_opening(&self) -> usize {
    self.num_column_opening
  }

  fn num_proximity_testing(&self) -> usize {
    self.num_proximity_testing
  }

  fn encode(&self, mut target: impl AsMut<[F]>) {
    let target = target.as_mut();
    assert_eq!(target.len(), self.codeword_len);

    if self.a.is_empty() {
      let (input, output) = target.split_at_mut(self.row_len);
      reed_solomon_into(input, output);
      return;
    }

    let mut input_offset = 0;
    self.a[..self.a.len() - 1].iter().for_each(|a| {
      let (input, output) = target[input_offset..].split_at_mut(a.dimension.n);
      a.dot_into(input, &mut output[..a.dimension.m]);
      input_offset += a.dimension.n;
    });

    let a_last = self.a.last().unwrap();
    let b_last = self.b.last().unwrap();
    let (input, output) = target[input_offset..].split_at_mut(a_last.dimension.n);
    let tmp = a_last.dot(input);
    reed_solomon_into(&tmp, &mut output[..b_last.dimension.n]);
    let mut output_offset = input_offset + a_last.dimension.n + b_last.dimension.n;
    input_offset += a_last.dimension.n + a_last.dimension.m;

    self
      .a
      .iter()
      .rev()
      .zip(self.b.iter().rev())
      .for_each(|(a, b)| {
        input_offset -= a.dimension.m;
        let (input, output) = target.split_at_mut(output_offset);
        b.dot_into(
          &input[input_offset..input_offset + b.dimension.n],
          &mut output[..b.dimension.m],
        );
        output_offset += b.dimension.m;
      });
  }
}

pub trait BrakedownCodeSpec: Debug {
  const LAMBDA: f64;
  const ALPHA: f64;
  const BETA: f64;
  const R: f64;

  fn delta() -> f64 {
    Self::BETA / Self::R
  }

  fn mu() -> f64 {
    Self::R - 1f64 - Self::R * Self::ALPHA
  }

  fn nu() -> f64 {
    Self::BETA + Self::ALPHA * Self::BETA + 0.03
  }

  fn c_n(n: usize) -> usize {
    let alpha = Self::ALPHA;
    let beta = Self::BETA;
    let n = n as f64;
    min(
      max(ceil(1.28 * beta * n), ceil(beta * n) + 4),
      ceil(
        ((110.0 / n) + h(beta) + alpha * h(1.28 * beta / alpha))
          / (beta * (alpha / (1.28 * beta)).log2()),
      ),
    )
  }

  fn d_n(log2_q: usize, n: usize) -> usize {
    let alpha = Self::ALPHA;
    let beta = Self::BETA;
    let r = Self::R;
    let mu = Self::mu();
    let nu = Self::nu();
    let log2_q = log2_q as f64;
    let n = n as f64;
    min(
      ceil((2.0 * beta + ((r - 1.0) + 110.0 / n) / log2_q) * n),
      ceil(
        (r * alpha * h(beta / r) + mu * h(nu / mu) + 110.0 / n) / (alpha * beta * (mu / nu).log2()),
      ),
    )
  }

  fn num_column_opening() -> usize {
    ceil(-Self::LAMBDA / (1.0 - Self::delta() / 3.0).log2())
  }

  fn num_proximity_testing(log2_q: usize, n: usize, n_0: usize) -> usize {
    ceil(Self::LAMBDA / (log2_q as f64 - (Self::codeword_len(log2_q, n, n_0) as f64).log2()))
  }

  fn dimensions(
    log2_q: usize,
    n: usize,
    n_0: usize,
  ) -> (Vec<SparseMatrixDimension>, Vec<SparseMatrixDimension>) {
    assert!(n > n_0);

    let a = iter::successors(Some(n), |n| {
      let m = ceil(*n as f64 * Self::ALPHA);
      if m < *n { Some(m) } else { None }
    })
    .tuple_windows()
    .map(|(n, m)| SparseMatrixDimension::new(n, m, min(Self::c_n(n), m)))
    .take_while(|a| a.n > n_0)
    .collect_vec();
    let b = a
      .iter()
      .map(|a| {
        let n_prime = ceil(a.m as f64 * Self::R);
        let m_prime = ceil(a.n as f64 * Self::R) - a.n - n_prime;
        SparseMatrixDimension::new(n_prime, m_prime, min(Self::d_n(log2_q, a.n), m_prime))
      })
      .collect();

    (a, b)
  }

  fn codeword_len(log2_q: usize, n: usize, n_0: usize) -> usize {
    let (a, b) = Self::dimensions(log2_q, n, n_0);
    if a.is_empty() {
      return ceil(n as f64 * Self::R);
    }
    chain![
      [a[0].n],
      a[..a.len() - 1].iter().map(|a| a.m),
      [b.last().unwrap().n],
      b.iter().map(|b| b.m),
    ]
    .sum()
  }

  fn matrices<F: Field>(
    log2_q: usize,
    n: usize,
    n_0: usize,
    mut rng: impl RngCore,
  ) -> (Vec<SparseMatrix<F>>, Vec<SparseMatrix<F>>) {
    let (a, b) = Self::dimensions(log2_q, n, n_0);
    a.into_iter()
      .zip(b)
      .map(|(a, b)| {
        (
          SparseMatrix::new(a, &mut rng),
          SparseMatrix::new(b, &mut rng),
        )
      })
      .unzip()
  }
}

macro_rules! impl_spec_128 {
    ($(($name:ident, $alpha:literal, $beta:literal, $r:literal)),*) => {
        $(
            #[allow(dead_code)]
            #[derive(Debug)]
            pub struct $name;
            impl BrakedownCodeSpec for $name {
                const LAMBDA: f64 = 128.0;
                const ALPHA: f64 = $alpha;
                const BETA: f64 = $beta;
                const R: f64 = $r;
            }
        )*
    };
}

// Figure 2 in [GLSTW21](https://eprint.iacr.org/2021/1043.pdf).
impl_spec_128!(
  (BrakedownCodeSpec1, 0.1195, 0.0284, 1.420),
  (BrakedownCodeSpec2, 0.1380, 0.0444, 1.470),
  (BrakedownCodeSpec3, 0.1780, 0.0610, 1.521),
  (BrakedownCodeSpec4, 0.2000, 0.0820, 1.640),
  (BrakedownCodeSpec5, 0.2110, 0.0970, 1.616),
  (BrakedownCodeSpec6, 0.2380, 0.1205, 1.720)
);

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SparseMatrixDimension {
  n: usize,
  m: usize,
  d: usize,
}

impl SparseMatrixDimension {
  fn new(n: usize, m: usize, d: usize) -> Self {
    Self { n, m, d }
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseMatrix<F> {
  dimension: SparseMatrixDimension,
  cells: Vec<(usize, F)>,
}

impl<F: Field> SparseMatrix<F> {
  fn new(dimension: SparseMatrixDimension, mut rng: impl RngCore) -> Self {
    let cells = iter::repeat_with(|| {
      let mut columns = BTreeSet::<usize>::new();
      (&mut rng)
        .sample_iter(&Uniform::new(0, dimension.m))
        .filter(|column| columns.insert(*column))
        .take(dimension.d)
        .count();
      columns
        .into_iter()
        .map(|column| (column, F::random(&mut rng)))
        .collect_vec()
    })
    .take(dimension.n)
    .flatten()
    .collect();
    Self { dimension, cells }
  }

  fn rows(&self) -> impl Iterator<Item = &[(usize, F)]> {
    self.cells.chunks(self.dimension.d)
  }

  fn dot_into(&self, array: &[F], mut target: impl AsMut<[F]>) {
    let target = target.as_mut();
    assert_eq!(self.dimension.n, array.len());
    assert_eq!(self.dimension.m, target.len());

    self.rows().zip(array.iter()).for_each(|(cells, item)| {
      cells.iter().for_each(|(column, coeff)| {
        target[*column] += *item * coeff;
      })
    });
  }

  fn dot(&self, array: &[F]) -> Vec<F> {
    let mut target = vec![F::ZERO; self.dimension.m];
    self.dot_into(array, &mut target);
    target
  }
}

fn reed_solomon_into<F: Field>(input: &[F], mut target: impl AsMut<[F]>) {
  target
    .as_mut()
    .iter_mut()
    .zip(steps(F::ONE))
    .for_each(|(target, x)| *target = horner(input, &x));
}

// H(p) = -p \log_2(p) - (1 - p) \log_2(1 - p)
fn h(p: f64) -> f64 {
  assert!(0f64 < p && p < 1f64);
  let one_minus_p = 1f64 - p;
  -p * p.log2() - one_minus_p * one_minus_p.log2()
}

fn ceil(v: f64) -> usize {
  v.ceil() as usize
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;
  use rand_core::OsRng;

  type F = pallas::Scalar;

  #[test]
  fn test_brakedown_encode() {
    let num_vars = 10;
    let n_0 = 20;
    let mut rng = OsRng;
    let brakedown =
      BrakedownCode::<F>::new_multilinear::<BrakedownCodeSpec6>(num_vars, n_0, &mut rng);

    let row_len = brakedown.row_len();
    let codeword_len = brakedown.codeword_len();

    let data = (0..row_len)
      .map(|_| F::random(&mut rng))
      .collect::<Vec<_>>();
    let mut codeword = vec![F::ZERO; codeword_len];
    codeword[..row_len].copy_from_slice(&data);

    brakedown.encode(&mut codeword);

    // Systematic code check: data is preserved in the first part of codeword
    assert_eq!(&codeword[..row_len], &data);
  }

  #[test]
  fn test_brakedown_linearity() {
    let num_vars = 8;
    let n_0 = 20;
    let mut rng = OsRng;
    let brakedown =
      BrakedownCode::<F>::new_multilinear::<BrakedownCodeSpec6>(num_vars, n_0, &mut rng);

    let row_len = brakedown.row_len();
    let codeword_len = brakedown.codeword_len();

    let a = (0..row_len)
      .map(|_| F::random(&mut rng))
      .collect::<Vec<_>>();
    let b = (0..row_len)
      .map(|_| F::random(&mut rng))
      .collect::<Vec<_>>();
    let alpha = F::random(&mut rng);

    let mut codeword_a = vec![F::ZERO; codeword_len];
    codeword_a[..row_len].copy_from_slice(&a);
    brakedown.encode(&mut codeword_a);

    let mut codeword_b = vec![F::ZERO; codeword_len];
    codeword_b[..row_len].copy_from_slice(&b);
    brakedown.encode(&mut codeword_b);

    let mut ab = a.clone();
    for i in 0..row_len {
      ab[i] = a[i] * alpha + b[i];
    }
    let mut codeword_ab = vec![F::ZERO; codeword_len];
    codeword_ab[..row_len].copy_from_slice(&ab);
    brakedown.encode(&mut codeword_ab);

    for i in 0..codeword_len {
      assert_eq!(codeword_ab[i], codeword_a[i] * alpha + codeword_b[i]);
    }
  }
}
