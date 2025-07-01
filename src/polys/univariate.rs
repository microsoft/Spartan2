//! Main components:
//! - `UniPoly`: an univariate dense polynomial in coefficient form (big endian),
//! - `CompressedUniPoly`: a univariate dense polynomial, compressed (omitted linear term), in coefficient form (little endian),
use crate::{
  errors::SpartanError,
  traits::{Group, transcript::TranscriptReprTrait},
};
use ff::PrimeField;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

// ax^2 + bx + c stored as vec![c, b, a]
// ax^3 + bx^2 + cx + d stored as vec![d, c, b, a]
/// A univariate dense polynomial in coefficient form with big endian storage.
///
/// For a polynomial $ax^2 + bx + c$, coefficients are stored as `vec![c, b, a]`.
/// For a polynomial $ax^3 + bx^2 + cx + d$, coefficients are stored as `vec![d, c, b, a]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniPoly<Scalar: PrimeField> {
  pub(crate) coeffs: Vec<Scalar>,
}

// ax^2 + bx + c stored as vec![c, a]
// ax^3 + bx^2 + cx + d stored as vec![d, c, a]
/// A univariate dense polynomial with compressed representation (omitted linear term).
///
/// The linear term coefficient is omitted to save space. For a polynomial $ax^2 + bx + c$,
/// coefficients are stored as `vec![c, a]`. For $ax^3 + bx^2 + cx + d$, stored as `vec![d, c, a]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedUniPoly<Scalar: PrimeField> {
  coeffs_except_linear_term: Vec<Scalar>,
}

impl<Scalar: PrimeField> UniPoly<Scalar> {
  /// Creates a `UniPoly` from its evaluations.
  ///
  /// Given evaluation points at consecutive integers starting from 0,
  /// this function interpolates the unique polynomial of degree `n-1`
  /// using Gaussian elimination.
  ///
  /// # Errors
  /// Returns `SpartanError` if the Gaussian elimination fails due to singular matrix
  /// or invalid input dimensions.
  pub fn from_evals(evals: &[Scalar]) -> Result<Self, SpartanError> {
    let n = evals.len();
    let xs: Vec<Scalar> = (0..n).map(|x| Scalar::from(x as u64)).collect();

    let mut matrix: Vec<Vec<Scalar>> = Vec::with_capacity(n);
    for i in 0..n {
      let mut row = Vec::with_capacity(n);
      let x = xs[i];
      row.push(Scalar::ONE);
      row.push(x);
      for j in 2..n {
        row.push(row[j - 1] * x);
      }
      row.push(evals[i]);
      matrix.push(row);
    }

    let coeffs = gaussian_elimination(&mut matrix)?;
    Ok(Self { coeffs })
  }

  /// Returns the degree of the polynomial.
  pub fn degree(&self) -> usize {
    self.coeffs.len() - 1
  }

  /// Evaluates the polynomial at zero.
  pub fn eval_at_zero(&self) -> Scalar {
    self.coeffs[0]
  }

  /// Evaluates the polynomial at one.
  pub fn eval_at_one(&self) -> Scalar {
    (0..self.coeffs.len())
      .into_par_iter()
      .map(|i| self.coeffs[i])
      .sum()
  }

  /// Evaluates the polynomial at a given point `r`.
  pub fn evaluate(&self, r: &Scalar) -> Scalar {
    let mut eval = self.coeffs[0];
    let mut power = *r;
    for coeff in self.coeffs.iter().skip(1) {
      eval += power * coeff;
      power *= r;
    }
    eval
  }

  /// Compresses the polynomial by omitting the linear coefficient.
  pub fn compress(&self) -> CompressedUniPoly<Scalar> {
    let coeffs_except_linear_term = [&self.coeffs[0..1], &self.coeffs[2..]].concat();
    assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
    CompressedUniPoly {
      coeffs_except_linear_term,
    }
  }
}

impl<Scalar: PrimeField> CompressedUniPoly<Scalar> {
  // we require eval(0) + eval(1) = hint, so we can solve for the linear term as:
  // linear_term = hint - 2 * constant_term - deg2 term - deg3 term
  /// Decompresses the polynomial by reconstructing the linear coefficient.
  ///
  /// # Arguments
  /// * `hint` - A hint value that helps reconstruct the linear term
  ///
  /// # Returns
  /// The full `UniPoly` with all coefficients restored.
  pub fn decompress(&self, hint: &Scalar) -> UniPoly<Scalar> {
    let mut linear_term =
      *hint - self.coeffs_except_linear_term[0] - self.coeffs_except_linear_term[0];
    for i in 1..self.coeffs_except_linear_term.len() {
      linear_term -= self.coeffs_except_linear_term[i];
    }

    let mut coeffs: Vec<Scalar> = Vec::new();
    coeffs.push(self.coeffs_except_linear_term[0]);
    coeffs.push(linear_term);
    coeffs.extend(&self.coeffs_except_linear_term[1..]);
    assert_eq!(self.coeffs_except_linear_term.len() + 1, coeffs.len());
    UniPoly { coeffs }
  }
}

impl<G: Group> TranscriptReprTrait<G> for UniPoly<G::Scalar> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let coeffs = self.compress().coeffs_except_linear_term;
    coeffs
      .iter()
      .flat_map(|&t| t.to_repr().as_ref().to_vec())
      .collect::<Vec<u8>>()
  }
}

// This code is based on code from https://github.com/a16z/jolt/blob/main/jolt-core/src/utils/gaussian_elimination.rs, which itself is
// inspired by https://github.com/TheAlgorithms/Rust/blob/master/src/math/gaussian_elimination.rs
/// Performs Gaussian elimination on a matrix to solve a linear system.
///
/// This function solves for the coefficients of a polynomial given a matrix
/// where each row represents an evaluation point and the last column contains
/// the evaluation values.
///
/// # Arguments
/// * `matrix` - A mutable reference to the augmented matrix
///
/// # Returns
/// A vector containing the solution (polynomial coefficients), or an error if the system cannot be solved.
///
/// # Errors
/// Returns `SpartanError::DivisionByZero` if any diagonal element is zero during the solving process.
pub fn gaussian_elimination<F: PrimeField>(matrix: &mut [Vec<F>]) -> Result<Vec<F>, SpartanError> {
  let size = matrix.len();
  if size != matrix[0].len() - 1 {
    return Err(SpartanError::InvalidInputLength);
  }

  for i in 0..size - 1 {
    for j in i..size - 1 {
      echelon(matrix, i, j)?;
    }
  }

  for i in (1..size).rev() {
    eliminate(matrix, i)?;
  }

  // Disable cargo clippy warnings about needless range loops.
  // Checking the diagonal like this is simpler than any alternative.
  #[allow(clippy::needless_range_loop)]
  for i in 0..size {
    if matrix[i][i] == F::ZERO {
      return Err(SpartanError::DivisionByZero);
    }
  }

  let mut result: Vec<F> = vec![F::ZERO; size];
  for i in 0..size {
    result[i] = div_f(matrix[i][size], matrix[i][i])?;
  }

  Ok(result)
}

fn echelon<F: PrimeField>(matrix: &mut [Vec<F>], i: usize, j: usize) -> Result<(), SpartanError> {
  let size = matrix.len();
  if matrix[i][i] != F::ZERO {
    let factor = div_f(matrix[j + 1][i], matrix[i][i])?;
    (i..size + 1).for_each(|k| {
      let tmp = matrix[i][k];
      matrix[j + 1][k] -= factor * tmp;
    });
  }
  Ok(())
}

fn eliminate<F: PrimeField>(matrix: &mut [Vec<F>], i: usize) -> Result<(), SpartanError> {
  let size = matrix.len();
  if matrix[i][i] != F::ZERO {
    for j in (1..i + 1).rev() {
      let factor = div_f(matrix[j - 1][i], matrix[i][i])?;
      for k in (0..size + 1).rev() {
        let tmp = matrix[i][k];
        matrix[j - 1][k] -= factor * tmp;
      }
    }
  }
  Ok(())
}

/// Division of two prime fields
///
/// # Arguments
/// * `a` - The dividend
/// * `b` - The divisor
///
/// # Returns
/// The result of `a / b` or an error if `b` is zero
///
/// # Errors
/// Returns `SpartanError::DivisionByZero` if `b` is zero (not invertible).
pub fn div_f<F: PrimeField>(a: F, b: F) -> Result<F, SpartanError> {
  let inverse_b = b.invert();

  match inverse_b.into_option() {
    Some(inv) => Ok(a * inv),
    None => Err(SpartanError::DivisionByZero),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;

  fn test_from_evals_quad_with<F: PrimeField>() {
    // polynomial is 2x^2 + 3x + 1
    let e0 = F::ONE;
    let e1 = F::from(6);
    let e2 = F::from(15);
    let evals = vec![e0, e1, e2];
    let poly = UniPoly::from_evals(&evals).unwrap();

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 3);
    assert_eq!(poly.coeffs[0], F::ONE);
    assert_eq!(poly.coeffs[1], F::from(3));
    assert_eq!(poly.coeffs[2], F::from(2));

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e3 = F::from(28);
    assert_eq!(poly.evaluate(&F::from(3)), e3);
  }

  #[test]
  fn test_from_evals_quad() {
    test_from_evals_quad_with::<pallas::Scalar>();
  }

  fn test_from_evals_cubic_with<F: PrimeField>() {
    // polynomial is x^3 + 2x^2 + 3x + 1
    let e0 = F::ONE;
    let e1 = F::from(7);
    let e2 = F::from(23);
    let e3 = F::from(55);
    let evals = vec![e0, e1, e2, e3];
    let poly = UniPoly::from_evals(&evals).unwrap();

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 4);

    assert_eq!(poly.coeffs[1], F::from(3));
    assert_eq!(poly.coeffs[2], F::from(2));
    assert_eq!(poly.coeffs[3], F::from(1));

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e4 = F::from(109);
    assert_eq!(poly.evaluate(&F::from(4)), e4);
  }

  #[test]
  fn test_from_evals_cubic() {
    test_from_evals_cubic_with::<pallas::Scalar>();
  }
  fn test_from_evals_quartic_with<F: PrimeField>() {
    // polynomial is x^4 + 2x^3 + 3x^2 + 4x + 5
    let e0 = F::from(5);
    let e1 = F::from(15);
    let e2 = F::from(57);
    let e3 = F::from(179);
    let e4 = F::from(453);
    let evals = vec![e0, e1, e2, e3, e4];
    let poly = UniPoly::from_evals(&evals).unwrap();

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 5);

    assert_eq!(poly.coeffs[0], F::from(5));
    assert_eq!(poly.coeffs[1], F::from(4));
    assert_eq!(poly.coeffs[2], F::from(3));
    assert_eq!(poly.coeffs[3], F::from(2));
    assert_eq!(poly.coeffs[4], F::from(1));

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e5 = F::from(975);
    assert_eq!(poly.evaluate(&F::from(5)), e5);
  }

  #[test]
  fn test_from_evals_quartic() {
    test_from_evals_quartic_with::<pallas::Scalar>();
  }
}
