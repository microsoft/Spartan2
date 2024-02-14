//! Support for generating R1CS using bellperson.

#![allow(non_snake_case)]

use super::{shape_cs::ShapeCS, solver::SatisfyingAssignment, test_shape_cs::TestShapeCS};
use crate::{
  errors::SpartanError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, R1CS},
  traits::Group,
  CommitmentKey,
};
use bellpepper_core::{Index, LinearCombination};
use ff::{Field, PrimeField};

/// `SpartanWitness` provide a method for acquiring an `R1CSInstance` and `R1CSWitness` from implementers.
pub trait SpartanWitness<G: Group> {
  /// Return an instance and witness, given a shape and ck.
  fn r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<G>,
    ck: &CommitmentKey<G>,
  ) -> Result<(R1CSInstance<G>, R1CSWitness<G>), SpartanError>;
}

/// `SpartanShape` provides methods for acquiring `R1CSShape` and `CommitmentKey` from implementers.
pub trait SpartanShape<G: Group> {
  /// Return an appropriate `R1CSShape` and `CommitmentKey` structs.
  fn r1cs_shape(&self) -> (R1CSShape<G>, CommitmentKey<G>);
}

impl<G: Group> SpartanWitness<G> for SatisfyingAssignment<G>
where
  G::Scalar: PrimeField,
{
  fn r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<G>,
    ck: &CommitmentKey<G>,
  ) -> Result<(R1CSInstance<G>, R1CSWitness<G>), SpartanError> {
    let W = R1CSWitness::<G>::new(shape, &self.aux_assignment)?;
    let X = &self.input_assignment[1..];

    let comm_W = W.commit(ck);
    let instance = R1CSInstance::<G>::new(shape, &comm_W, X)?;

    Ok((instance, W))
  }
}

macro_rules! impl_spartan_shape {
  ( $name:ident) => {
    impl<G: Group> SpartanShape<G> for $name<G>
    where
      G::Scalar: PrimeField,
    {
      fn r1cs_shape(&self) -> (R1CSShape<G>, CommitmentKey<G>) {
        let mut A: Vec<(usize, usize, G::Scalar)> = Vec::new();
        let mut B: Vec<(usize, usize, G::Scalar)> = Vec::new();
        let mut C: Vec<(usize, usize, G::Scalar)> = Vec::new();

        let mut num_cons_added = 0;
        let mut X = (&mut A, &mut B, &mut C, &mut num_cons_added);

        let num_inputs = self.num_inputs();
        let num_constraints = self.num_constraints();
        let num_vars = self.num_aux();

        for constraint in self.constraints.iter() {
          add_constraint(
            &mut X,
            num_vars,
            &constraint.0,
            &constraint.1,
            &constraint.2,
          );
        }

        assert_eq!(num_cons_added, num_constraints);

        let S: R1CSShape<G> = {
          // Don't count One as an input for shape's purposes.
          let res = R1CSShape::new(num_constraints, num_vars, num_inputs - 1, &A, &B, &C);
          res.unwrap()
        };

        let ck = R1CS::<G>::commitment_key(&S);

        (S, ck)
      }
    }
  };
}

impl_spartan_shape!(ShapeCS);
impl_spartan_shape!(TestShapeCS);

impl<G: Group> ShapeCS<G> {
  /// r1cs_shape but with extrpolates from one step of a uniform computation 
  pub fn r1cs_shape_uniform(&self, N: usize) -> (R1CSShape<G>, R1CSShape<G>, CommitmentKey<G>) {
    let S_single = self.r1cs_shape().0;

    let mut A: Vec<(usize, usize, G::Scalar)> = Vec::new();
    let mut B: Vec<(usize, usize, G::Scalar)> = Vec::new();
    let mut C: Vec<(usize, usize, G::Scalar)> = Vec::new();

    let mut num_cons_added = 0;
    let mut X = (&mut A, &mut B, &mut C, &mut num_cons_added);

    // HACK(arasuarun): assuming this is = 1
    let num_inputs = self.num_inputs();
    let num_constraints_per_step = self.num_constraints();
    let num_aux_per_step= self.num_aux(); // Arasu: this doesn't include the 1

    let num_constraints_total = num_constraints_per_step * N;
    let num_aux_total = num_aux_per_step * N;

    let span = tracing::span!(tracing::Level::INFO, "r1cs matrix creation");
    let _guard = span.enter();
    for constraint in self.constraints.iter() {
      add_constraint_uniform(
        &mut X,
        num_aux_total,
        &constraint.0,
        &constraint.1,
        &constraint.2,
        N,
        num_aux_per_step,
      );
    }  
    drop(_guard);
    drop(span);

    // assert_eq!(num_cons_added, num_constraints);

    let S: R1CSShape<G> = {
      // Don't count One as an input for shape's purposes.
      // Arasu: num_vars is actually supposed to be num_aux (and not including 1)
      // Witness format is [W || 1 || x]
      let res = R1CSShape::new(num_constraints_total, num_aux_total, num_inputs - 1, &A, &B, &C);
      res.unwrap()
    };


    let ck = R1CS::<G>::commitment_key(&S);

    (S, S_single, ck) 
  }

  /// r1cs_shape but with extrpolates from one step of a uniform computation 
  pub fn r1cs_shape_uniform_variablewise(&self, N: usize) -> (R1CSShape<G>, R1CSShape<G>, CommitmentKey<G>) {
    let S_single = self.r1cs_shape().0;

    let mut A: Vec<(usize, usize, G::Scalar)> = Vec::new();
    let mut B: Vec<(usize, usize, G::Scalar)> = Vec::new();
    let mut C: Vec<(usize, usize, G::Scalar)> = Vec::new();

    let mut num_cons_added = 0;
    let mut X = (&mut A, &mut B, &mut C, &mut num_cons_added);

    // HACK(arasuarun): assuming this is = 1
    let num_inputs = self.num_inputs();
    let num_constraints_per_step = self.num_constraints();
    let num_aux_per_step= self.num_aux(); // Arasu: this doesn't include the 1

    let num_aux_total = num_aux_per_step * N;

    let span = tracing::span!(tracing::Level::INFO, "r1cs matrix creation");
    let _guard = span.enter();
    for constraint in self.constraints.iter() {
      add_constraint_uniform_variablewise(
        &mut X,
        num_aux_total,
        &constraint.0,
        &constraint.1,
        &constraint.2,
        N,
      );
    }  

    let mut num_constraints_total = num_constraints_per_step * N;
    // Add IO consistency constraints 
    // TODO(arasuarun): Hack. Make this a parameter instead. 
    let STATE_LEN = 2; 
    for i in 0..STATE_LEN {
      for step in 0..(N-1) {
        A.push((num_constraints_total + step, num_aux_total, G::Scalar::ONE));
        B.push((num_constraints_total + step, i * N + step, G::Scalar::ONE)); // output of step i
        C.push((num_constraints_total + step, (STATE_LEN + i) * N + step + 1, G::Scalar::ONE)); // input of step I+1
      }
      num_constraints_total += N-1; 
    }

    drop(_guard);
    drop(span);

    // assert_eq!(num_cons_added, num_constraints);

    let S: R1CSShape<G> = {
      // Don't count One as an input for shape's purposes.
      // Arasu: num_vars is actually supposed to be num_aux (and not including 1)
      // Witness format is [W || 1 || x] but as x is assumed to be empty and inputs are provided in aux instead. 
      // So witness is [W || 1] 
      let res = R1CSShape::new(num_constraints_total, num_aux_total, num_inputs-1, &A, &B, &C);
      res.unwrap()
    };


    let ck = R1CS::<G>::commitment_key(&S);

    (S, S_single, ck) 
  }
}

fn add_constraint<S: PrimeField>(
  X: &mut (
    &mut Vec<(usize, usize, S)>,
    &mut Vec<(usize, usize, S)>,
    &mut Vec<(usize, usize, S)>,
    &mut usize,
  ),
  num_vars: usize,
  a_lc: &LinearCombination<S>,
  b_lc: &LinearCombination<S>,
  c_lc: &LinearCombination<S>,
) {
  let (A, B, C, nn) = X;
  let n = **nn;
  let one = S::ONE;

  let add_constraint_component = |index: Index, coeff, V: &mut Vec<_>| {
    match index {
      Index::Input(idx) => {
        // Inputs come last, with input 0, reprsenting 'one',
        // at position num_vars within the witness vector.
        let i = idx + num_vars;
        V.push((n, i, one * coeff))
      }
      Index::Aux(idx) => V.push((n, idx, one * coeff)),
    }
  };


  for (index, coeff) in a_lc.iter() {
    add_constraint_component(index.0, coeff, A);
  }
  for (index, coeff) in b_lc.iter() {
    add_constraint_component(index.0, coeff, B)
  }
  for (index, coeff) in c_lc.iter() {
    add_constraint_component(index.0, coeff, C)
  }


  **nn += 1;
}


fn add_constraint_uniform<S: PrimeField>(
  X: &mut (
    &mut Vec<(usize, usize, S)>,
    &mut Vec<(usize, usize, S)>,
    &mut Vec<(usize, usize, S)>,
    &mut usize,
  ),
  num_vars: usize,
  a_lc: &LinearCombination<S>,
  b_lc: &LinearCombination<S>,
  c_lc: &LinearCombination<S>,
  num_steps: usize, 
  num_aux_per_step: usize,
) {
  let (A, B, C, nn) = X;
  let n = **nn; // Arasu: this is just the row number, I think 
  let one = S::ONE;

  let add_constraint_component = |index: Index, coeff, V: &mut Vec<_>| {
    match index {
      Index::Input(idx) => {
        // Inputs come last, with input 0, reprsenting 'one',
        // at position num_vars within the witness vector.
        let i = idx + num_vars;
        for step in 0..num_steps {
          V.push((n + step, i, one * coeff)); // the column of the input is the same for all steps
        }
      }
      Index::Aux(idx) => {
        for step in 0..num_steps {
          V.push((n + step, idx + num_aux_per_step * step, one * coeff));
        }
      }
    }
  };

  rayon::join(|| {
    a_lc.iter().for_each(|(index, coeff)| {
        add_constraint_component(index.0, coeff, A);
    });
  }, || {
    rayon::join(|| {
        b_lc.iter().for_each(|(index, coeff)| {
            add_constraint_component(index.0, coeff, B);
        });
    }, || {
        c_lc.iter().for_each(|(index, coeff)| {
            add_constraint_component(index.0, coeff, C);
        });
    });
  });

  **nn += num_steps;
}

fn add_constraint_uniform_variablewise<S: PrimeField>(
  X: &mut (
    &mut Vec<(usize, usize, S)>,
    &mut Vec<(usize, usize, S)>,
    &mut Vec<(usize, usize, S)>,
    &mut usize,
  ),
  num_vars: usize,
  a_lc: &LinearCombination<S>,
  b_lc: &LinearCombination<S>,
  c_lc: &LinearCombination<S>,
  num_steps: usize, 
) {
  let (A, B, C, nn) = X;
  let n = **nn; // Arasu: this is just the row number, I think 
  let one = S::ONE;

  let add_constraint_component = |index: Index, coeff, V: &mut Vec<_>| {
    match index {
      Index::Input(idx) => {
        // Inputs come last, with input 0, reprsenting 'one',
        // at position num_vars within the witness vector.
        let i = idx + num_vars;
        for step in 0..num_steps {
          V.push((n + step, i, one * coeff)); // the column of the input is the same for all steps
        }
      }
      Index::Aux(idx) => {
        for step in 0..num_steps {
          V.push((n + step, idx * num_steps + step, one * coeff));
        }
      }
    }
  };

  rayon::join(|| {
    a_lc.iter().for_each(|(index, coeff)| {
        add_constraint_component(index.0, coeff, A);
    });
  }, || {
    rayon::join(|| {
        b_lc.iter().for_each(|(index, coeff)| {
            add_constraint_component(index.0, coeff, B);
        });
    }, || {
        c_lc.iter().for_each(|(index, coeff)| {
            add_constraint_component(index.0, coeff, C);
        });
    });
  });

  **nn += num_steps;
}