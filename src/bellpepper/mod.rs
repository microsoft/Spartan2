//! Support for generating R1CS from [Bellperson].
//!
//! [Bellperson]: https://github.com/filecoin-project/bellperson

pub mod r1cs;

// Superseded by ark-relations
// pub mod solver;
// pub mod shape_cs;
// pub mod test_shape_cs;

#[cfg(test)]
mod tests {
  use crate::{bellpepper::r1cs::SpartanWitness, traits::Group};

  use crate::r1cs::{R1CSShape, R1CS};
  use ark_ff::Field;
  use ark_relations::lc;
  use ark_relations::r1cs::{ConstraintSystem, SynthesisError, Variable};

  fn synthesize_alloc_bit<F: Field>(cs: &mut ConstraintSystem<F>) -> Result<(), SynthesisError> {
    // Allocate 'a' as a public input
    let a_var = cs.new_input_variable(|| Ok(F::ONE))?;

    // Enforce: a * (1 - a) = 0 (this ensures that 'a' is an 0 or 1)
    cs.enforce_constraint(lc!() + a_var, lc!() + Variable::One - a_var, lc!())?;

    // Allocate 'b' as a public input
    let b_var = cs.new_input_variable(|| Ok(F::ONE))?;

    // Enforce: b * (1 - b) = 0 (this ensures that 'b' is 0 or 1)
    cs.enforce_constraint(lc!() + b_var, lc!() + Variable::One - b_var, lc!())?;

    Ok(())
  }

  fn test_alloc_bit_with<G>()
  where
    G: Group,
  {
    // First create the shape
    let mut cs: ConstraintSystem<G::Scalar> = ConstraintSystem::new();
    let _ = synthesize_alloc_bit(&mut cs);
    let shape: R1CSShape<G> = R1CSShape::from(&cs);
    let ck = R1CS::commitment_key(&shape);

    // Now get the assignment
    let mut cs: ConstraintSystem<G::Scalar> = ConstraintSystem::new();
    let _ = synthesize_alloc_bit(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  #[test]
  fn test_alloc_bit() {
    // test_alloc_bit_with::<pasta_curves::pallas::Point>();
    // test_alloc_bit_with::<crate::provider::bn256_grumpkin::bn256::Point>();
    test_alloc_bit_with::<ark_bls12_381::G1Projective>();
  }
}
