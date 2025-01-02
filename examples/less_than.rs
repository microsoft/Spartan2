use ark_bls12_381::Fr;
use ark_ff::{BigInteger, PrimeField};
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::boolean::AllocatedBool;
use ark_r1cs_std::fields::fp::AllocatedFp;
use ark_relations::r1cs::{
  ConstraintSynthesizer, ConstraintSystemRef, Namespace, SynthesisError, Variable,
};
use ark_relations::{lc, ns};
use num_traits::One;
use spartan2::{
  errors::SpartanError,
  traits::{snark::RelaxedR1CSSNARKTrait, Group},
  SNARK,
};

fn num_to_bits_le_bounded<F: PrimeField>(
  cs: ConstraintSystemRef<F>,
  n: AllocatedFp<F>,
  num_bits: u8,
) -> Result<Vec<AllocatedBool<F>>, SynthesisError> {
  let opt_bits = n
    .value()?
    .into_bigint()
    .to_bits_le()
    .into_iter()
    .take(num_bits as usize)
    .map(Some)
    .collect::<Vec<Option<bool>>>();

  // Add one witness per input bit in little-endian bit order
  let bits_circuit = opt_bits.into_iter()
    .enumerate()
    // AllocatedBool enforces the value to be 0 or 1 at the constraint level
    .map(|(_i, b)| {
      // TODO: Why do I need namespaces here?
      // TODO: Namespace can't use string ids, only const ids
      // let namespaced_cs = Namespace::from(cs.clone());
      // TODO: Is it a "new_input" or a different type of a variable?
        AllocatedBool::<F>::new_input(cs.clone(), || b.ok_or(SynthesisError::AssignmentMissing))
    })
    .collect::<Result<Vec<AllocatedBool<F>>, SynthesisError>>()?;

  let mut weighted_sum_lc = lc!();
  let mut pow2 = F::ONE;

  for bit in bits_circuit.iter() {
    weighted_sum_lc = weighted_sum_lc + (pow2, bit.variable());
    pow2 = pow2.double();
  }

  // weighted_sum_lc == n
  let constraint_lc = weighted_sum_lc - n.variable;

  // Enforce constraint_lc == 0
  let one_lc = lc!() + Variable::One;
  cs.enforce_constraint(constraint_lc, one_lc, lc!())?;

  Ok(bits_circuit)
}

fn get_msb_index<F: PrimeField>(n: F) -> u8 {
  n.into_bigint()
    .to_bits_le()
    .into_iter()
    .enumerate()
    .rev()
    .find(|(_, b)| *b)
    .expect("Index not found")
    .0 as u8
}

// Constrains `input` < `bound`, where the LHS is a witness and the RHS is a
// constant. The bound must fit into `num_bits` bits (this is asserted in the
// circuit constructor).
// Important: it must be checked elsewhere (in an overarching circuit) that the
// input fits into `num_bits` bits - this is NOT constrained by this circuit
// in order to avoid duplications (hence "unsafe"). Cf. LessThanCircuitSafe for
// a safe version.
#[derive(Clone, Debug)]
struct LessThanCircuitUnsafe<F: PrimeField> {
  bound: F, // Will be a constant in the constraints, not a variable
  input: F, // Will be an input/output variable
  num_bits: u8,
}

impl<F: PrimeField> LessThanCircuitUnsafe<F> {
  fn new(bound: F, input: F, num_bits: u8) -> Self {
    assert!(get_msb_index(bound) < num_bits);
    Self {
      bound,
      input,
      num_bits,
    }
  }
}

impl<F: PrimeField> ConstraintSynthesizer<F> for LessThanCircuitUnsafe<F> {
  fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
    assert!(F::MODULUS_BIT_SIZE > self.num_bits as u32 + 1);

    let input_ns = ns!(cs.clone(), "input");
    let input = AllocatedFp::<F>::new_input(input_ns, || Ok(self.input))?;

    let shifted_ns = ns!(cs.clone(), "shifted_diff");
    let shifted_diff = AllocatedFp::<F>::new_witness(shifted_ns, || {
      Ok(self.input + F::from(1 << self.num_bits) - self.bound)
    })?;

    let shifted_diff_lc =
      lc!() + (F::ONE, input.variable) + (F::from(1 << self.num_bits) - self.bound, Variable::One)
        - (F::ONE, shifted_diff.variable);

    // Enforce shifted_diff_lc == 0
    cs.enforce_constraint(shifted_diff_lc, lc!() + (F::ONE, Variable::One), lc!())?;

    let shifted_diff_bits =
      num_to_bits_le_bounded::<F>(cs.clone(), shifted_diff, self.num_bits + 1)?;

    // Check that the most significant bit is 0
    let msb_var = shifted_diff_bits[self.num_bits as usize].variable();

    // Enforce the constraint that the most significant bit is 0
    cs.enforce_constraint(
      lc!() + (F::ONE, msb_var),
      lc!() + (F::ONE, Variable::One),
      lc!(),
    )?;

    Ok(())
  }
}

// Constrains `input` < `bound`, where the LHS is a witness and the RHS is a
// constant. The bound must fit into `num_bits` bits (this is asserted in the
// circuit constructor).
// Furthermore, the input must fit into `num_bits`, which is enforced at the
// constraint level.
#[derive(Clone, Debug)]
struct LessThanCircuitSafe<F: PrimeField> {
  bound: F,
  input: F,
  num_bits: u8,
}

impl<F: PrimeField> LessThanCircuitSafe<F> {
  fn new(bound: F, input: F, num_bits: u8) -> Self {
    assert!(get_msb_index(bound) < num_bits);
    Self {
      bound,
      input,
      num_bits,
    }
  }
}

impl<F: PrimeField> ConstraintSynthesizer<F> for LessThanCircuitSafe<F> {
  fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
    // TODO: Do we need to use a namespace here?
    let input_ns = Namespace::from(cs.clone());
    let input = AllocatedFp::<F>::new_input(input_ns, || Ok(self.input))?;

    // Perform the input bit decomposition check
    num_to_bits_le_bounded::<F>(cs.clone(), input, self.num_bits)?;

    // TODO: Not sure how/why to do this in Arkworks
    // Entering a new namespace to prefix variables in the
    // LessThanCircuitUnsafe, thus avoiding name clashes
    // cs.push_namespace(|| "less_than_safe");

    LessThanCircuitUnsafe {
      bound: self.bound,
      input: self.input,
      num_bits: self.num_bits,
    }
    .generate_constraints(cs)
  }
}

fn verify_circuit_unsafe<G: Group, S: RelaxedR1CSSNARKTrait<G>>(
  bound: G::Scalar,
  input: G::Scalar,
  num_bits: u8,
) -> Result<(), SpartanError> {
  let circuit = LessThanCircuitUnsafe::new(bound, input, num_bits);

  // produce keys
  let (pk, vk) = SNARK::<G, S, LessThanCircuitUnsafe<_>>::setup(circuit.clone())?;

  // produce a SNARK
  let snark = SNARK::prove(&pk, circuit)?;

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
  let (pk, vk) = SNARK::<G, S, LessThanCircuitSafe<_>>::setup(circuit.clone())?;

  // produce a SNARK
  let snark = SNARK::prove(&pk, circuit)?;

  // verify the SNARK
  snark.verify(&vk, &[])
}

fn main() {
  type G = ark_bls12_381::G1Projective;
  type EE = spartan2::provider::ipa_pc::EvaluationEngine<G>;
  type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G, EE>;

  println!("Executing unsafe circuit...");
  //Typical example, ok
  assert!(verify_circuit_unsafe::<G, S>(Fr::from(17), Fr::from(9), 10).is_ok());
  // Typical example, err
  assert!(verify_circuit_unsafe::<G, S>(Fr::from(17), Fr::from(20), 10).is_err());
  // Edge case, err
  assert!(verify_circuit_unsafe::<G, S>(Fr::from(4), Fr::from(4), 10).is_err());
  // Edge case, ok
  assert!(verify_circuit_unsafe::<G, S>(Fr::from(4), Fr::from(3), 10).is_ok());
  // Minimum number of bits for the bound, ok
  assert!(verify_circuit_unsafe::<G, S>(Fr::from(4), Fr::from(3), 3).is_ok());
  // Insufficient number of bits for the input, but this is not detected by the
  // unsafety of the circuit (compare with the last example below)
  // Note that -Fr::one() is corresponds to q - 1 > bound
  assert!(verify_circuit_unsafe::<G, S>(Fr::from(4), -Fr::one(), 3).is_ok());

  println!("Unsafe circuit OK");

  println!("Executing safe circuit...");
  // Typical example, ok
  assert!(verify_circuit_safe::<G, S>(Fr::from(17), Fr::from(9), 10).is_ok());
  // Typical example, err
  assert!(verify_circuit_safe::<G, S>(Fr::from(17), Fr::from(20), 10).is_err());
  // Edge case, err
  assert!(verify_circuit_safe::<G, S>(Fr::from(4), Fr::from(4), 10).is_err());
  // Edge case, ok
  assert!(verify_circuit_safe::<G, S>(Fr::from(4), Fr::from(3), 10).is_ok());
  // Minimum number of bits for the bound, ok
  assert!(verify_circuit_safe::<G, S>(Fr::from(4), Fr::from(3), 3).is_ok());
  // Insufficient number of bits for the input, err (compare with the last example
  // above).
  // Note that -Fr::one() is corresponds to q - 1 > bound
  assert!(verify_circuit_safe::<G, S>(Fr::from(4), -Fr::one(), 3).is_err());

  println!("Safe circuit OK");
}
