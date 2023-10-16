//! This module provides an interface to generate spartan proofs with circom circuits.
use std::path::PathBuf;

use crate::{
  errors::SpartanError,
  traits::{self, snark::RelaxedR1CSSNARKTrait, Group},
  ProverKey, VerifierKey, SNARK,
};

use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use ff::PrimeField;

use circom_scotia::{r1cs::R1CS, reader::load_r1cs, witness::WitnessCalculator};

/// Wrapper around an R1CS object to implement the Circuit trait
#[derive(Clone, Debug)]
pub struct SpartanCircuit<F: PrimeField> {
  r1cs: R1CS<F>,
  witness: Option<Vec<F>>, // this is [1 || inputs || witnesses]
}

impl<F: PrimeField> SpartanCircuit<F> {
  fn new(r1cs_path: PathBuf) -> Self {
    SpartanCircuit {
      r1cs: load_r1cs(r1cs_path),
      witness: None,
    }
  }

  fn compute_witness(&mut self, input: Vec<(String, Vec<F>)>, wtns_path: PathBuf) {
    let mut witness_calculator = WitnessCalculator::new(wtns_path).unwrap();
    let witness = witness_calculator
      .calculate_witness(input.clone(), true)
      .expect("msg");

    self.witness = Some(witness);
  }
}

impl<F: PrimeField> Circuit<F> for SpartanCircuit<F> {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let _ = circom_scotia::synthesize(cs, self.r1cs.clone(), self.witness).unwrap();

    Ok(())
  }
}

/// Produces prover and verifier keys
pub fn setup<G: Group, S: RelaxedR1CSSNARKTrait<G>>(
  r1cs_path: PathBuf,
) -> (ProverKey<G, S>, VerifierKey<G, S>) {
  SNARK::<G, S, SpartanCircuit<<G as Group>::Scalar>>::setup(SpartanCircuit::new(r1cs_path))
    .unwrap()
}

/// Produces proof in the form of a SNARK object
pub fn prove<G: Group, S: RelaxedR1CSSNARKTrait<G>>(
  pk: ProverKey<G, S>,
  r1cs_path: PathBuf,
  wtns_path: PathBuf,
  input: Vec<(String, Vec<<G as Group>::Scalar>)>,
) -> Result<SNARK<G, S, SpartanCircuit<<G as traits::Group>::Scalar>>, SpartanError> {
  let mut circuit = SpartanCircuit::new(r1cs_path);
  circuit.compute_witness(input, wtns_path);
  SNARK::prove(&pk, circuit.clone())
}

#[cfg(test)]
mod test {
  use super::{prove, setup};
  use crate::{provider::bn256_grumpkin::bn256, traits::Group};
  use std::env::current_dir;

  #[test]
  fn test_spartan_snark() {
    type G = bn256::Point;
    type EE = crate::provider::ipa_pc::EvaluationEngine<G>;
    type S = crate::spartan::snark::RelaxedR1CSSNARK<G, EE>;

    let root = current_dir().unwrap().join("examples/cube");
    let r1cs_path = root.join("cube.r1cs");
    let wtns_path = root.join("cube.wasm");

    let arg_x = ("x".into(), vec![<G as Group>::Scalar::from(2)]);
    let arg_y = ("y".into(), vec![<G as Group>::Scalar::from(8)]);
    let input = vec![arg_x, arg_y];

    let (pk, vk) = setup::<G, S>(r1cs_path.clone());

    let res = prove::<G, S>(pk, r1cs_path, wtns_path, input);
    assert!(res.is_ok());

    let snark = res.unwrap();
    assert!(snark.verify(&vk).is_ok());
  }

  #[test]
  fn test_spartan_snark_fail() {
    type G = bn256::Point;
    type EE = crate::provider::ipa_pc::EvaluationEngine<G>;
    type S = crate::spartan::snark::RelaxedR1CSSNARK<G, EE>;

    let root = current_dir().unwrap().join("examples/cube");
    let r1cs_path = root.join("cube.r1cs");
    let wtns_path = root.join("cube.wasm");

    let (pk, vk) = setup::<G, S>(r1cs_path.clone());

    // setting y to 9 shouldn't satisfy
    let arg_x = ("x".into(), vec![<G as Group>::Scalar::from(2)]);
    let arg_y = ("y".into(), vec![<G as Group>::Scalar::from(9)]);
    let input = vec![arg_x, arg_y];

    let res = prove::<G, S>(pk, r1cs_path, wtns_path, input);
    assert!(res.is_ok());

    let snark = res.unwrap();
    // check that it fails
    assert!(snark.verify(&vk).is_err());
  }
}
