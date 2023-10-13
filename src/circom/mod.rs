use std::path::PathBuf;

use crate::{
  errors::SpartanError,
  traits::{self, snark::RelaxedR1CSSNARKTrait, Group},
  ProverKey, VerifierKey, SNARK,
};

use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use ff::PrimeField;

use circom_scotia::{r1cs::R1CS, reader::load_r1cs, witness::WitnessCalculator};

#[derive(Clone, Debug)]
pub struct SpartanCircuit<F: PrimeField> {
  r1cs: R1CS<F>,
  witness: Option<Vec<F>>, // this is actually z = [1 || x || w]
}

#[allow(dead_code)]
impl<F: PrimeField> SpartanCircuit<F> {
  pub fn new(r1cs_path: PathBuf) -> Self {
    SpartanCircuit {
      r1cs: load_r1cs(r1cs_path),
      witness: None,
    }
  }

  pub fn compute_witness(&mut self, input: Vec<(String, Vec<F>)>, wtns_path: PathBuf) {
    let mut witness_calculator = WitnessCalculator::new(wtns_path).unwrap();
    let witness = witness_calculator
      .calculate_witness(input.clone(), true)
      .expect("msg");

    self.witness = Some(witness);
  }
}

impl<F: PrimeField> Circuit<F> for SpartanCircuit<F> {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let _ = circom_scotia::synthesize(
      cs, //.namespace(|| "spartan_snark"),
      self.r1cs.clone(),
      self.witness,
    )
    .unwrap();

    Ok(())
  }
}

#[allow(dead_code)]
pub fn generate_keys<G: Group, S: RelaxedR1CSSNARKTrait<G>>(
  circuit: SpartanCircuit<<G as Group>::Scalar>,
) -> (ProverKey<G, S>, VerifierKey<G, S>) {
  SNARK::<G, S, SpartanCircuit<<G as Group>::Scalar>>::setup(circuit).unwrap()
}

#[allow(dead_code)]
pub fn generate_proof<G: Group, S: RelaxedR1CSSNARKTrait<G>>(
  pk: ProverKey<G, S>,
  circuit: SpartanCircuit<<G as Group>::Scalar>,
) -> Result<SNARK<G, S, SpartanCircuit<<G as traits::Group>::Scalar>>, SpartanError> {
  SNARK::prove(&pk, circuit)
}

#[cfg(test)]
mod test {
  use super::{generate_keys, generate_proof, SpartanCircuit};
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
    let mut circuit = SpartanCircuit::new(r1cs_path);

    let (pk, vk) = generate_keys(circuit.clone());

    let arg_x = ("x".into(), vec![<G as Group>::Scalar::from(2)]);
    let arg_y = ("y".into(), vec![<G as Group>::Scalar::from(8)]);
    let input = vec![arg_x, arg_y];

    circuit.compute_witness(input, wtns_path);

    let res = generate_proof::<G, S>(pk, circuit);
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
    let mut circuit = SpartanCircuit::new(r1cs_path);

    let (pk, vk) = generate_keys(circuit.clone());

    // setting y to 9 shouldn't satisfy
    let arg_x = ("x".into(), vec![<G as Group>::Scalar::from(2)]);
    let arg_y = ("y".into(), vec![<G as Group>::Scalar::from(9)]);
    let input = vec![arg_x, arg_y];

    circuit.compute_witness(input, wtns_path);

    let res = generate_proof::<G, S>(pk, circuit);
    assert!(res.is_ok());

    let snark = res.unwrap();
    // check that it fails
    assert!(snark.verify(&vk).is_err());
  }
}
