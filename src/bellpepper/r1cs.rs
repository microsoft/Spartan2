//! Support for generating R1CS using bellperson.

#![allow(non_snake_case)]

use crate::{
  errors::SpartanError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness},
  traits::Group,
  CommitmentKey,
};
use ark_ff::PrimeField;
use ark_relations::r1cs::ConstraintSystem;

/// `SpartanWitness` provide a method for acquiring an `R1CSInstance` and `R1CSWitness` from implementers.
pub trait SpartanWitness<G: Group> {
  /// Return an instance and witness, given a shape and ck.
  fn r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<G>,
    ck: &CommitmentKey<G>,
  ) -> Result<(R1CSInstance<G>, R1CSWitness<G>), SpartanError>;
}

impl<G: Group> SpartanWitness<G> for ConstraintSystem<G::Scalar>
where
  G::Scalar: PrimeField,
{
  fn r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<G>,
    ck: &CommitmentKey<G>,
  ) -> Result<(R1CSInstance<G>, R1CSWitness<G>), SpartanError> {
    let W = R1CSWitness::<G>::new(shape, &self.witness_assignment)?;
    let X = &self.instance_assignment[1..];

    let comm_W = W.commit(ck);

    let instance = R1CSInstance::<G>::new(shape, &comm_W, X)?;

    Ok((instance, W))
  }
}
