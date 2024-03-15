use bellpepper_core::{
  boolean::AllocatedBit, num::AllocatedNum, Circuit, ConstraintSystem, LinearCombination,
  SynthesisError,
};
use ff::{PrimeField, PrimeFieldBits};
use pasta_curves::Fq;
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
      .map(Some)
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

fn get_msb_index<F: PrimeField + PrimeFieldBits>(n: F) -> u8 {
  n.to_le_bits()
    .into_iter()
    .enumerate()
    .rev()
    .find(|(_, b)| *b)
    .unwrap()
    .0 as u8
}

// Range check: constrains input < `bound`. The bound must fit into
// `num_bits` bits (this is asserted in the circuit constructor).
// Important: it must be checked elsewhere (in an overarching circuit) that the
// input fits into `num_bits` bits - this is NOT constrained by this circuit
// in order to avoid duplications (hence "unsafe"). Cf. LessThanCircuitSafe for
// a safe version.
#[derive(Clone, Debug)]
struct LessThanCircuitUnsafe<F: PrimeField> {
  bound: F, // Will be a constant in the constraits, not a variable
  input: F, // Will be an input/output variable
  num_bits: u8,
}

impl<F: PrimeField + PrimeFieldBits> LessThanCircuitUnsafe<F> {
  fn new(bound: F, input: F, num_bits: u8) -> Self {
    assert!(get_msb_index(bound) < num_bits);
    Self {
      bound,
      input,
      num_bits,
    }
  }
}

impl<F: PrimeField + PrimeFieldBits> Circuit<F> for LessThanCircuitUnsafe<F> {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    assert!(F::NUM_BITS > self.num_bits as u32 + 1);

    let input = AllocatedNum::alloc(cs.namespace(|| "input"), || Ok(self.input))?;

    let shifted_diff = AllocatedNum::alloc(cs.namespace(|| "shifted_diff"), || {
      Ok(self.input + F::from(1 << self.num_bits) - self.bound)
    })?;

    cs.enforce(
      || "shifted_diff_computation",
      |lc| lc + input.get_variable() + (F::from(1 << self.num_bits) - self.bound, CS::one()),
      |lc: LinearCombination<F>| lc + CS::one(),
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

// Range check: constrains input < `bound`. The bound must fit into
// `num_bits` bits (this is asserted in the circuit constructor).
// Furthermore, the input must fit into `num_bits`, which is enforced at the
// constraint level.
#[derive(Clone, Debug)]
struct LessThanCircuitSafe<F: PrimeField + PrimeFieldBits> {
  bound: F,
  input: F,
  num_bits: u8,
}

impl<F: PrimeField + PrimeFieldBits> LessThanCircuitSafe<F> {
  fn new(bound: F, input: F, num_bits: u8) -> Self {
    assert!(get_msb_index(bound) < num_bits);
    Self {
      bound,
      input,
      num_bits,
    }
  }
}

impl<F: PrimeField + PrimeFieldBits> Circuit<F> for LessThanCircuitSafe<F> {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let input = AllocatedNum::alloc(cs.namespace(|| "input"), || Ok(self.input))?;

    // Perform the input bit decomposition check
    num_to_bits_le_bounded::<F, CS>(cs, input, self.num_bits)?;

    // Entering a new namespace to prefix variables in the
    // LessThanCircuitUnsafe, thus avoiding name clashes
    cs.push_namespace(|| "less_than_safe");

    LessThanCircuitUnsafe {
      bound: self.bound,
      input: self.input,
      num_bits: self.num_bits,
    }
    .synthesize(cs)
  }
}

fn verify_circuit_unsafe<G: Group, S: RelaxedR1CSSNARKTrait<G>>(
  bound: G::Scalar,
  input: G::Scalar,
  num_bits: u8,
) -> Result<(), SpartanError> {
  let circuit = LessThanCircuitUnsafe::new(bound, input, num_bits);

  // produce keys
  let (pk, vk) = SNARK::<G, S, LessThanCircuitUnsafe<_>>::setup(circuit.clone()).unwrap();

  // produce a SNARK
  let snark = SNARK::prove(&pk, circuit).unwrap();

  // verify the SNARK
  snark.verify(&vk, &[])
}

fn verify_circuit_safe<G: Group, S: RelaxedR1CSSNARKTrait<G>>(
  bound: G::Scalar,
  input: G::Scalar,
  num_bits: u8,
) -> Result<(), SpartanError> {
  let circuit = LessThanCircuitSafe::new(bound, input, num_bits);

  // produce keys
  let (pk, vk) = SNARK::<G, S, LessThanCircuitSafe<_>>::setup(circuit.clone()).unwrap();

  // produce a SNARK
  let snark = SNARK::prove(&pk, circuit).unwrap();

  // verify the SNARK
  snark.verify(&vk, &[])
}

fn main() {
  type G = pasta_curves::pallas::Point;
  type EE = spartan2::provider::ipa_pc::EvaluationEngine<G>;
  type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G, EE>;

  println!("Executing unsafe circuit...");
  //Typical example, ok
  assert!(verify_circuit_unsafe::<G, S>(Fq::from(17), Fq::from(9), 10).is_ok());
  // Typical example, err
  assert!(verify_circuit_unsafe::<G, S>(Fq::from(17), Fq::from(20), 10).is_err());
  // Edge case, err
  assert!(verify_circuit_unsafe::<G, S>(Fq::from(4), Fq::from(4), 10).is_err());
  // Edge case, ok
  assert!(verify_circuit_unsafe::<G, S>(Fq::from(4), Fq::from(3), 10).is_ok());
  // Minimum number of bits for the bound, ok
  assert!(verify_circuit_unsafe::<G, S>(Fq::from(4), Fq::from(3), 3).is_ok());
  // Insufficient number of bits for the input, but this is not detected by the
  // unsafety of the circuit (compare with the last example below)
  // Note that -Fq::one() is corresponds to q - 1 > bound
  assert!(verify_circuit_unsafe::<G, S>(Fq::from(4), -Fq::one(), 3).is_ok());

  println!("Unsafe circuit OK");

  println!("Executing safe circuit...");
  // Typical example, ok
  assert!(verify_circuit_safe::<G, S>(Fq::from(17), Fq::from(9), 10).is_ok());
  // Typical example, err
  assert!(verify_circuit_safe::<G, S>(Fq::from(17), Fq::from(20), 10).is_err());
  // Edge case, err
  assert!(verify_circuit_safe::<G, S>(Fq::from(4), Fq::from(4), 10).is_err());
  // Edge case, ok
  assert!(verify_circuit_safe::<G, S>(Fq::from(4), Fq::from(3), 10).is_ok());
  // Minimum number of bits for the bound, ok
  assert!(verify_circuit_safe::<G, S>(Fq::from(4), Fq::from(3), 3).is_ok());
  // Insufficient number of bits for the input, err (compare with the last example
  // above).
  // Note that -Fq::one() is corresponds to q - 1 > bound
  assert!(verify_circuit_safe::<G, S>(Fq::from(4), -Fq::one(), 3).is_err());

  println!("Safe circuit OK");
}
