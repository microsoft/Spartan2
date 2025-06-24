//! This module defines R1CS related types
use crate::{
  Blind, Commitment, CommitmentKey, DerandKey, PCS, VerifierKey,
  digest::SimpleDigestible,
  errors::SpartanError,
  traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptReprTrait},
};
use core::cmp::max;
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod sparse;
pub(crate) use sparse::SparseMatrix;

/// A type that holds the shape of the R1CS matrices
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSShape<E: Engine> {
  pub(crate) num_cons: usize,
  pub(crate) num_vars: usize,
  pub(crate) num_io: usize,
  pub(crate) A: SparseMatrix<E::Scalar>,
  pub(crate) B: SparseMatrix<E::Scalar>,
  pub(crate) C: SparseMatrix<E::Scalar>,
  #[serde(skip, default = "OnceCell::new")]
  pub(crate) digest: OnceCell<E::Scalar>,
}

impl<E: Engine> SimpleDigestible for R1CSShape<E> {}

/// A type that holds a witness for a given R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSWitness<E: Engine> {
  pub(crate) W: Vec<E::Scalar>,
  pub(crate) r_W: Blind<E>,

  pub(crate) Az: Vec<E::Scalar>,
  pub(crate) Bz: Vec<E::Scalar>,
  pub(crate) Cz: Vec<E::Scalar>,
}

/// A type that holds an R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSInstance<E: Engine> {
  pub(crate) comm_W: Commitment<E>,
  pub(crate) X: Vec<E::Scalar>,
}

impl<E: Engine> R1CSShape<E> {
  /// Create an object of type `R1CSShape` from the explicitly specified R1CS matrices
  pub fn new(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    A: SparseMatrix<E::Scalar>,
    B: SparseMatrix<E::Scalar>,
    C: SparseMatrix<E::Scalar>,
  ) -> Result<R1CSShape<E>, SpartanError> {
    let is_valid = |num_cons: usize,
                    num_vars: usize,
                    num_io: usize,
                    M: &SparseMatrix<E::Scalar>|
     -> Result<Vec<()>, SpartanError> {
      M.iter()
        .map(|(row, col, _val)| {
          if row >= num_cons || col > num_io + num_vars {
            Err(SpartanError::InvalidIndex)
          } else {
            Ok(())
          }
        })
        .collect::<Result<Vec<()>, SpartanError>>()
    };

    is_valid(num_cons, num_vars, num_io, &A)?;
    is_valid(num_cons, num_vars, num_io, &B)?;
    is_valid(num_cons, num_vars, num_io, &C)?;

    Ok(R1CSShape {
      num_cons,
      num_vars,
      num_io,
      A,
      B,
      C,
      digest: OnceCell::new(),
    })
  }

  /// Generates public parameters for a Rank-1 Constraint System (R1CS).
  ///
  /// This function takes into consideration the shape of the R1CS matrices and a hint function
  /// for the number of generators. It returns a `CommitmentKey`.
  ///
  /// # Arguments
  ///
  /// * `S`: The shape of the R1CS matrices.
  /// * `ck_floor`: A function that provides a floor for the number of generators. A good function
  ///   to provide is the ck_floor field defined in the trait `R1CSSNARK`.
  ///
  pub fn commitment_key(&self) -> (CommitmentKey<E>, VerifierKey<E>) {
    let num_cons = self.num_cons;
    let num_vars = self.num_vars;
    E::PCS::setup(b"ck", max(num_cons, num_vars))
  }

  // Checks regularity conditions on the R1CSShape, required in Spartan-class SNARKs
  // Returns false if num_cons or num_vars are not powers of two, or if num_io > num_vars
  #[inline]
  pub(crate) fn is_regular_shape(&self) -> bool {
    let cons_valid = self.num_cons.next_power_of_two() == self.num_cons;
    let vars_valid = self.num_vars.next_power_of_two() == self.num_vars;
    let io_lt_vars = self.num_io < self.num_vars;
    cons_valid && vars_valid && io_lt_vars
  }

  pub fn multiply_vec(
    &self,
    z: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    if z.len() != self.num_io + self.num_vars + 1 {
      return Err(SpartanError::InvalidWitnessLength);
    }

    let (Az, (Bz, Cz)) = rayon::join(
      || self.A.multiply_vec(z),
      || rayon::join(|| self.B.multiply_vec(z), || self.C.multiply_vec(z)),
    );

    Ok((Az, Bz, Cz))
  }

  /// Checks if the R1CS instance is satisfiable given a witness and its shape
  #[allow(dead_code)]
  pub fn is_sat(
    &self,
    ck: &CommitmentKey<E>,
    U: &R1CSInstance<E>,
    W: &R1CSWitness<E>,
  ) -> Result<(), SpartanError> {
    assert_eq!(W.W.len(), self.num_vars);
    assert_eq!(U.X.len(), self.num_io);

    // verify if Az * Bz = u*Cz
    let res_eq = {
      let z = [W.W.clone(), vec![E::Scalar::ONE], U.X.clone()].concat();
      let (Az, Bz, Cz) = self.multiply_vec(&z)?;
      assert_eq!(Az.len(), self.num_cons);
      assert_eq!(Bz.len(), self.num_cons);
      assert_eq!(Cz.len(), self.num_cons);

      (0..self.num_cons).all(|i| Az[i] * Bz[i] == Cz[i])
    };

    // verify if comm_W is a commitment to W
    let res_comm = U.comm_W == PCS::<E>::commit(ck, &W.W, &W.r_W);

    if !res_eq {
      return Err(SpartanError::UnSat {
        reason: "R1CS is unsatisfiable".to_string(),
      });
    }

    if !res_comm {
      return Err(SpartanError::UnSat {
        reason: "Invalid commitment".to_string(),
      });
    }

    Ok(())
  }

  /// Pads the `R1CSShape` so that the shape passes `is_regular_shape`
  /// Renumbers variables to accommodate padded variables
  pub fn pad(&self) -> Self {
    // check if the provided R1CSShape is already as required
    if self.is_regular_shape() {
      return self.clone();
    }

    // equalize the number of variables, constraints, and public IO
    let m = max(max(self.num_vars, self.num_cons), self.num_io).next_power_of_two();

    // check if the number of variables are as expected, then
    // we simply set the number of constraints to the next power of two
    if self.num_vars == m {
      return R1CSShape {
        num_cons: m,
        num_vars: m,
        num_io: self.num_io,
        A: self.A.clone(),
        B: self.B.clone(),
        C: self.C.clone(),
        digest: OnceCell::new(),
      };
    }

    // otherwise, we need to pad the number of variables and renumber variable accesses
    let num_vars_padded = m;
    let num_cons_padded = m;

    let apply_pad = |mut M: SparseMatrix<E::Scalar>| -> SparseMatrix<E::Scalar> {
      M.indices.par_iter_mut().for_each(|c| {
        if *c >= self.num_vars {
          *c += num_vars_padded - self.num_vars
        }
      });

      M.cols += num_vars_padded - self.num_vars;

      let ex = {
        let nnz = M.indptr.last().unwrap();
        vec![*nnz; num_cons_padded - self.num_cons]
      };
      M.indptr.extend(ex);
      M
    };

    let A_padded = apply_pad(self.A.clone());
    let B_padded = apply_pad(self.B.clone());
    let C_padded = apply_pad(self.C.clone());

    R1CSShape {
      num_cons: num_cons_padded,
      num_vars: num_vars_padded,
      num_io: self.num_io,
      A: A_padded,
      B: B_padded,
      C: C_padded,
      digest: OnceCell::new(),
    }
  }
}

impl<E: Engine> R1CSWitness<E> {
  /// A method to create a witness object using a vector of scalars
  pub fn new(
    ck: &CommitmentKey<E>,
    S: &R1CSShape<E>,
    W: &[E::Scalar],
  ) -> Result<(R1CSWitness<E>, Commitment<E>), SpartanError> {
    let mut W = W.to_vec();
    W.resize(S.num_vars, E::Scalar::ZERO);

    let r_W = PCS::<E>::blind(ck);
    let comm_W = PCS::<E>::commit(ck, &W, &r_W);

    let W = R1CSWitness { W, r_W };

    Ok((W, comm_W))
  }

  /// Pads the provided witness to the correct length
  pub fn pad(&self, S: &R1CSShape<E>) -> R1CSWitness<E> {
    let mut W = self.W.clone();
    W.extend(vec![E::Scalar::ZERO; S.num_vars - W.len()]);

    Self {
      W,
      r_W: self.r_W.clone(),
    }
  }

  pub fn derandomize(&self) -> (Self, Blind<E>) {
    (
      R1CSWitness {
        W: self.W.clone(),
        r_W: Blind::<E>::default(),
      },
      self.r_W.clone(),
    )
  }
}

impl<E: Engine> R1CSInstance<E> {
  /// A method to create an instance object using constituent elements
  pub fn new(
    S: &R1CSShape<E>,
    comm_W: &Commitment<E>,
    X: &[E::Scalar],
  ) -> Result<R1CSInstance<E>, SpartanError> {
    if S.num_io != X.len() {
      Err(SpartanError::InvalidInputLength)
    } else {
      Ok(R1CSInstance {
        comm_W: comm_W.clone(),
        X: X.to_owned(),
      })
    }
  }

  pub fn derandomize(&self, dk: &DerandKey<E>, r_W: &Blind<E>) -> R1CSInstance<E> {
    R1CSInstance {
      comm_W: PCS::<E>::derandomize(dk, &self.comm_W, r_W),
      X: self.X.clone(),
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for R1CSInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_W.to_transcript_bytes(),
      self.X.as_slice().to_transcript_bytes(),
    ]
    .concat()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::{PallasHyraxEngine, PallasIPAEngine},
    r1cs::sparse::SparseMatrix,
    traits::Engine,
  };
  use ff::Field;

  fn tiny_r1cs<E: Engine>(num_vars: usize) -> R1CSShape<E> {
    let one = <E::Scalar as Field>::ONE;
    let (num_cons, num_vars, num_io, A, B, C) = {
      let num_cons = 4;
      let num_io = 2;

      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      // The R1CS for this problem consists of the following constraints:
      // `I0 * I0 - Z0 = 0`
      // `Z0 * I0 - Z1 = 0`
      // `(Z1 + I0) * 1 - Z2 = 0`
      // `(Z2 + 5) * 1 - I1 = 0`

      // R1CS is a set of three sparse matrices (A B C), where there is a row for every
      // constraint and a column for every entry in z = (vars, u, inputs)
      // An R1CS instance is satisfiable iff:
      // Az \circ Bz = u \cdot Cz + E, where z = (vars, 1, inputs)
      let mut A: Vec<(usize, usize, E::Scalar)> = Vec::new();
      let mut B: Vec<(usize, usize, E::Scalar)> = Vec::new();
      let mut C: Vec<(usize, usize, E::Scalar)> = Vec::new();

      // constraint 0 entries in (A,B,C)
      // `I0 * I0 - Z0 = 0`
      A.push((0, num_vars + 1, one));
      B.push((0, num_vars + 1, one));
      C.push((0, 0, one));

      // constraint 1 entries in (A,B,C)
      // `Z0 * I0 - Z1 = 0`
      A.push((1, 0, one));
      B.push((1, num_vars + 1, one));
      C.push((1, 1, one));

      // constraint 2 entries in (A,B,C)
      // `(Z1 + I0) * 1 - Z2 = 0`
      A.push((2, 1, one));
      A.push((2, num_vars + 1, one));
      B.push((2, num_vars, one));
      C.push((2, 2, one));

      // constraint 3 entries in (A,B,C)
      // `(Z2 + 5) * 1 - I1 = 0`
      A.push((3, 2, one));
      A.push((3, num_vars, one + one + one + one + one));
      B.push((3, num_vars, one));
      C.push((3, num_vars + 2, one));

      (num_cons, num_vars, num_io, A, B, C)
    };

    // create a shape object
    let rows = num_cons;
    let cols = num_vars + num_io + 1;

    let res = R1CSShape::new(
      num_cons,
      num_vars,
      num_io,
      SparseMatrix::new(&A, rows, cols),
      SparseMatrix::new(&B, rows, cols),
      SparseMatrix::new(&C, rows, cols),
    );
    assert!(res.is_ok());
    res.unwrap()
  }

  fn test_pad_tiny_r1cs_with<E: Engine>() {
    let padded_r1cs = tiny_r1cs::<E>(3).pad();
    assert!(padded_r1cs.is_regular_shape());

    let expected_r1cs = tiny_r1cs::<E>(4);

    assert_eq!(padded_r1cs, expected_r1cs);
  }

  #[test]
  fn test_pad_tiny_r1cs() {
    test_pad_tiny_r1cs_with::<PallasIPAEngine>();
    test_pad_tiny_r1cs_with::<PallasHyraxEngine>();
  }
}
