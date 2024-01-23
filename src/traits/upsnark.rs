//! This module defines a collection of traits that define the behavior of a zkSNARK for RelaxedR1CS
use crate::{errors::SpartanError, traits::Group}; //, CommitmentKey, Commitment};
use bellpepper_core::Circuit;
use serde::{Deserialize, Serialize};
use crate::traits::snark::RelaxedR1CSSNARKTrait;

/// A SNARK that derives the R1CS Shape given a single step's shape 
pub trait UniformSNARKTrait<G: Group>:
  Sized + Send + Sync + Serialize + for<'de> Deserialize<'de> + RelaxedR1CSSNARKTrait<G> 
{
  /// Produces the keys, taking the number of steps and the length of the state as input 
  fn setup_uniform<C: Circuit<G::Scalar>>(
    circuit: C,
    num_steps: usize,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError>;
}

/// The witness commitments and generators are passed in externally
pub trait PrecommittedSNARKTrait<G: Group>:
  Sized + Send + Sync + Serialize + for<'de> Deserialize<'de> + UniformSNARKTrait<G> 
{
  /// Setup that takes in the generators used to pre-committed the witness 
  fn setup_precommitted<C: Circuit<G::Scalar>>(
    circuit: C,
    num_steps: usize,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError>;

//   /// Produces a new SNARK for a relaxed R1CS
//   // fn prove_precommitted<C: Circuit<G::Scalar>>(pk: &Self::ProverKey, circuit: C, comm_W: Commitment<G>) -> Result<Self, SpartanError>;
//   fn prove_precommitted<C: Circuit<G::Scalar>>(pk: &Self::ProverKey, circuit: C, comm_W: Commitment<G>) -> Result<Self, SpartanError>;
}