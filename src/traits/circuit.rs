//! This module defines traits that a circuit provider must implement to be used with Spartan.
use crate::traits::Engine;
use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};

/// A helper trait for defining a randomized circuit that Spartan proves.
/// The circuit contains a set of variables that are shared with other circuits.
/// For simplicity, we only allow one round of interaction, so there is a single list of precommitted variables.
/// (1) the prover commits to precommitted variables (and variables that will be made public via inputize)
/// (2) the verifier challenges with a random value
/// (3) the prover commits to aux variables
/// (4) The circuit checks a set of constraints over witness = (shared, precommitted, aux)
/// The public IO includes the challenge and other things made public by the circuit
pub trait SpartanCircuit<E: Engine>: Send + Sync + Clone {
  /// Returns the public values of the circuit, which is the list of values that will be made public
  /// The circuit must make public these values followed by the challenges generated via the transcript
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError>;

  /// Allocated variables in the circuit that are shared with other circuits
  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError>;

  /// Allocates precommitted variables
  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    shared: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError>;

  /// Returns the number of challenges that the circuit expects from the verifier
  /// for randomized checks added in synthesize
  fn num_challenges(&self) -> usize;

  /// Allocate the rest of the variables and constraints in the circuit.
  /// The `shared` and `precommitted` variables are already allocated.
  /// The `challenges` are the challenges from the verifier, if any (these must be allocated as public IO prior to other public IO).
  /// The circuit should use these challenges to synthesize the rest of the circuit.
  /// The circuit should return an error if it cannot synthesize the circuit.
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    shared: &[AllocatedNum<E::Scalar>],
    precommitted: &[AllocatedNum<E::Scalar>],
    challenges: Option<&[E::Scalar]>, // challenges from the verifier
  ) -> Result<(), SynthesisError>;
}
