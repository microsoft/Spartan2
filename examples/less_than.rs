use bellpepper_core::{
  boolean::AllocatedBit, num::AllocatedNum, Circuit, ConstraintSystem, LinearCombination,
  SynthesisError,
};
use ff::{PrimeField, PrimeFieldBits};
use spartan2::{
  errors::SpartanError,
  traits::{snark::RelaxedR1CSSNARKTrait, Group},
  SNARK,
};

fn num_to_bits_le_bounded<F: PrimeField + PrimeFieldBits, CS: ConstraintSystem<F>>(
  cs: &mut CS,
  n: AllocatedNum<F>,
  num_bits: u8,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
  let opt_bits = match n.get_value() {
    Some(v) => v
      .to_le_bits()
      .into_iter()
      .take(num_bits as usize)
      .map(|b| Some(b))
      .collect::<Vec<Option<bool>>>(),
    None => vec![None; num_bits as usize],
  };

  // Add one witness per input bit in little-endian bit order
  let bits_circuit = opt_bits.into_iter()
    .enumerate()
    // AllocateBit enforces the value to be 0 or 1 at the constraint level
    .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("b_{}", i)), b).unwrap())
    .collect::<Vec<AllocatedBit>>();

  let mut weighted_sum_lc = LinearCombination::zero();
  let mut pow2 = F::ONE;

  for bit in bits_circuit.iter() {
    weighted_sum_lc = weighted_sum_lc + (pow2, bit.get_variable());
    pow2 = pow2.double();
  }

  cs.enforce(
    || "bit decomposition check",
    |lc| lc + &weighted_sum_lc,
    |lc| lc + CS::one(),
    |lc| lc + n.get_variable(),
  );

  Ok(bits_circuit)
}

// Range check: constrains input < `bound`. The bound must fit into
// `num_bits` bits (this is asserted in the circuit constructor).
// Important: it must be checked elsewhere that the input fits into
// `num_bits` bits - this is NOT constrained by this circuit in order to
// avoid duplications (hence "unsafe")
#[derive(Clone, Debug)]
struct LessThanCircuitUnsafe {
  bound: u64, // Will be a constant in the constraits, not a variable
  input: u64, // Will be an input/output variable
  num_bits: u8,
}

impl LessThanCircuitUnsafe {
  fn new(bound: u64, input: u64, num_bits: u8) -> Self {
    assert!(bound < (1 << num_bits));
    Self {
      bound,
      input,
      num_bits,
    }
  }
}

impl<F: PrimeField + PrimeFieldBits> Circuit<F> for LessThanCircuitUnsafe {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    assert!(F::NUM_BITS > self.num_bits as u32 + 1);

    let input = AllocatedNum::alloc(cs.namespace(|| "input"), || Ok(F::from(self.input)))?;

    let shifted_diff = AllocatedNum::alloc(cs.namespace(|| "shifted_diff"), || {
      Ok(F::from(self.input + (1 << self.num_bits) - self.bound))
    })?;

    cs.enforce(
      || "shifted_diff_computation",
      |lc| lc + input.get_variable() + (F::from((1 << self.num_bits) - self.bound), CS::one()),
      |lc| lc + CS::one(),
      |lc| lc + shifted_diff.get_variable(),
    );

    let shifted_diff_bits = num_to_bits_le_bounded::<F, CS>(cs, shifted_diff, self.num_bits + 1)?;

    // Check that the last (i.e. most sifnificant) bit is 0
    cs.enforce(
      || "bound_check",
      |lc| lc + shifted_diff_bits[self.num_bits as usize].get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + (F::ZERO, CS::one()),
    );

    Ok(())
  }
}

fn verify_circuit<G: Group, S: RelaxedR1CSSNARKTrait<G>>(
  bound: u64,
  input: u64,
  num_bits: u8,
) -> Result<(), SpartanError> {
  let circuit = LessThanCircuitUnsafe::new(bound, input, num_bits);

  // produce keys
  let (pk, vk) = SNARK::<G, S, LessThanCircuitUnsafe>::setup(circuit.clone()).unwrap();

  // produce a SNARK
  let snark = SNARK::prove(&pk, circuit).unwrap();

  // verify the SNARK
  snark.verify(&vk, &[])
}

fn main() {
  type G = pasta_curves::pallas::Point;
  type EE = spartan2::provider::ipa_pc::EvaluationEngine<G>;
  type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G, EE>;

  // Typical exapmle, ok
  assert!(verify_circuit::<G, S>(17, 9, 10).is_ok());
  // Typical example, err
  assert!(verify_circuit::<G, S>(17, 20, 10).is_err());
  // Edge case, err
  assert!(verify_circuit::<G, S>(4, 4, 10).is_err());
  // Edge case, ok
  assert!(verify_circuit::<G, S>(4, 3, 10).is_ok());
  // Minimum number of bits, ok
  assert!(verify_circuit::<G, S>(4, 3, 3).is_ok());
}
