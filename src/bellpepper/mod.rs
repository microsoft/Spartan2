//! Support for generating R1CS from Bellpepper
pub mod r1cs;
pub mod shape_cs;
pub mod solver;
pub mod test_r1cs;
pub mod test_shape_cs;

#[cfg(test)]
mod tests {
  use crate::{
    bellpepper::{
      solver::SatisfyingAssignment,
      test_r1cs::{TestSpartanShape, TestSpartanWitness},
      test_shape_cs::TestShapeCS,
    },
    provider::PallasHyraxEngine,
    traits::Engine,
  };
  use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
  use ff::PrimeField;

  fn synthesize_alloc_bit<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
  ) -> Result<(), SynthesisError> {
    // get two bits as input and check that they are indeed bits
    let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(Fr::ONE))?;
    let _ = a.inputize(cs.namespace(|| "a is input"));
    cs.enforce(
      || "check a is 0 or 1",
      |lc| lc + CS::one() - a.get_variable(),
      |lc| lc + a.get_variable(),
      |lc| lc,
    );
    let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(Fr::ONE))?;
    let _ = b.inputize(cs.namespace(|| "b is input"));
    cs.enforce(
      || "check b is 0 or 1",
      |lc| lc + CS::one() - b.get_variable(),
      |lc| lc + b.get_variable(),
      |lc| lc,
    );
    Ok(())
  }

  fn test_alloc_bit_with<E: Engine>() {
    // First create the shape
    let mut cs: TestShapeCS<E> = TestShapeCS::new();
    let _ = synthesize_alloc_bit(&mut cs);
    let (shape, ck, _vk) = cs.r1cs_shape().unwrap();

    // Now get the assignment
    let mut cs: SatisfyingAssignment<E> = SatisfyingAssignment::new();
    let _ = synthesize_alloc_bit(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck, true).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  #[test]
  fn test_alloc_bit() {
    test_alloc_bit_with::<PallasHyraxEngine>();
  }
}
