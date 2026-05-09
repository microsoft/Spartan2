// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 circuit using small_sha256 gadget (small-value compatible).

use super::hash_to_public_scalars;
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::{PrimeField, PrimeFieldBits};
use std::marker::PhantomData;

use crate::{
  gadgets::{NoBatchEq, SmallBoolean, small_sha256_int},
  small_constraint_system::{SmallConstraintSystem, SmallLinearCombination, SmallToBellpepperCS},
  traits::{
    Engine,
    circuit::{SmallSpartanCircuit, SpartanCircuit},
  },
};
use ff::Field;
#[cfg(debug_assertions)]
use sha2::{Digest, Sha256};

/// SHA-256 circuit using small_sha256 gadget (small-value compatible).
///
/// Uses `SmallMultiEq` to keep coefficients bounded for small-value sumcheck.
#[derive(Debug, Clone)]
pub struct SmallSha256Circuit<Scalar: PrimeField> {
  /// The preimage bytes to hash.
  pub preimage: Vec<u8>,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> SmallSha256Circuit<Scalar> {
  /// Create a new SHA-256 circuit.
  pub fn new(preimage: Vec<u8>) -> Self {
    Self {
      preimage,
      _p: PhantomData,
    }
  }
}

impl<E: Engine> SpartanCircuit<E> for SmallSha256Circuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    Ok(hash_to_public_scalars(&self.preimage))
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // Use SmallToBellpepperCS so the field and integer paths produce the SAME shape.
    let mut small_cs = SmallToBellpepperCS::<E::Scalar, CS>::new(cs);
    let preimage_bits = alloc_preimage_small_bits::<i8, _>(&mut small_cs, &self.preimage)?;
    let mut eq = NoBatchEq::<i8, i32, _>::new(&mut small_cs);
    let hash_bits = small_sha256_int::<i8, _>(&mut eq, &preimage_bits)?;
    drop(eq);

    // Verify against native SHA-256 (debug only)
    #[cfg(debug_assertions)]
    {
      let hash_expected = Sha256::digest(&self.preimage);
      for (i, bit) in hash_bits.iter().enumerate() {
        let byte_idx = i / 8;
        let bit_idx = 7 - (i % 8);
        let expected = (hash_expected[byte_idx] >> bit_idx) & 1 == 1;
        let computed = bit.get_value().unwrap_or(false);
        assert_eq!(computed, expected, "Hash bit {i} mismatch");
      }
    }

    expose_small_hash_bits_as_public::<i8, _>(&mut small_cs, &hash_bits)?;

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
    _: &[AllocatedNum<E::Scalar>],
    _: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

// ── SmallSpartanCircuit impls ─────────────────────────────────────────────

/// Helper: allocate preimage bits as SmallBoolean variables.
pub(crate) fn alloc_preimage_small_bits<W, CS>(
  cs: &mut CS,
  preimage: &[u8],
) -> Result<Vec<SmallBoolean>, SynthesisError>
where
  W: Copy + From<bool>,
  CS: SmallConstraintSystem<W, i32>,
{
  let mut bits = Vec::with_capacity(preimage.len() * 8);
  for (byte_idx, &byte) in preimage.iter().enumerate() {
    for bit_idx in (0..8).rev() {
      let val = (byte >> bit_idx) & 1 == 1;
      let bit = crate::gadgets::small_boolean::SmallBit::alloc(
        &mut cs.namespace(|| format!("preimage_byte{byte_idx}_bit{bit_idx}")),
        Some(val),
      )?;
      bits.push(SmallBoolean::Is(bit));
    }
  }
  Ok(bits)
}

/// Expose SHA-256 digest bits as public inputs and bind each input to the
/// corresponding computed bit.
pub(crate) fn expose_small_hash_bits_as_public<W, CS>(
  cs: &mut CS,
  hash_bits: &[SmallBoolean],
) -> Result<(), SynthesisError>
where
  W: Copy + From<bool>,
  CS: SmallConstraintSystem<W, i32>,
{
  for (i, bit) in hash_bits.iter().enumerate() {
    let public = cs.alloc_input(
      || format!("hash_bit_{i}"),
      || {
        bit
          .get_value()
          .map(W::from)
          .ok_or(SynthesisError::AssignmentMissing)
      },
    )?;

    let mut diff = bit.lc::<i32>();
    diff.add_term(public, -1i32);

    cs.enforce(
      || format!("hash_bit_public_binding_{i}"),
      diff,
      SmallLinearCombination::one(1i32),
      SmallLinearCombination::zero(),
    );
  }

  Ok(())
}

/// SHA-256 circuit for the small-value compiler.
///
/// Uses i8 bit witnesses and i32 constraint coefficients. Shape and witness
/// generation run this same implementation against different backends.
impl<E: Engine> SmallSpartanCircuit<E, i8, i32> for SmallSha256Circuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<i8>, SynthesisError> {
    use crate::sha256_circuits::hash_to_public_scalars;
    let bits: Vec<E::Scalar> = hash_to_public_scalars(&self.preimage);
    Ok(
      bits
        .iter()
        .map(|b| if b.is_zero().into() { 0i8 } else { 1i8 })
        .collect(),
    )
  }

  fn shared<CS: SmallConstraintSystem<i8, i32>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<bellpepper_core::Variable>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: SmallConstraintSystem<i8, i32>>(
    &self,
    cs: &mut CS,
    _shared: &[bellpepper_core::Variable],
  ) -> Result<Vec<bellpepper_core::Variable>, SynthesisError> {
    let preimage_bits = alloc_preimage_small_bits(cs, &self.preimage)?;
    let mut eq = NoBatchEq::<i8, i32, _>::new(cs);
    let hash_bits = small_sha256_int::<i8, _>(&mut eq, &preimage_bits)?;
    drop(eq);

    expose_small_hash_bits_as_public::<i8, _>(cs, &hash_bits)?;

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: SmallConstraintSystem<i8, i32>>(
    &self,
    _cs: &mut CS,
    _shared: &[bellpepper_core::Variable],
    _precommitted: &[bellpepper_core::Variable],
    _challenges: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

impl<E: Engine> SmallSpartanCircuit<E, bool, i32> for SmallSha256Circuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<bool>, SynthesisError> {
    use crate::sha256_circuits::hash_to_public_scalars;
    let bits: Vec<E::Scalar> = hash_to_public_scalars(&self.preimage);
    Ok(bits.iter().map(|b| !bool::from(b.is_zero())).collect())
  }

  fn shared<CS: SmallConstraintSystem<bool, i32>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<bellpepper_core::Variable>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: SmallConstraintSystem<bool, i32>>(
    &self,
    cs: &mut CS,
    _shared: &[bellpepper_core::Variable],
  ) -> Result<Vec<bellpepper_core::Variable>, SynthesisError> {
    let preimage_bits = alloc_preimage_small_bits(cs, &self.preimage)?;
    let mut eq = NoBatchEq::<bool, i32, _>::new(cs);
    let hash_bits = small_sha256_int::<bool, _>(&mut eq, &preimage_bits)?;
    drop(eq);

    expose_small_hash_bits_as_public::<bool, _>(cs, &hash_bits)?;

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: SmallConstraintSystem<bool, i32>>(
    &self,
    _cs: &mut CS,
    _shared: &[bellpepper_core::Variable],
    _precommitted: &[bellpepper_core::Variable],
    _challenges: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

impl<Scalar: PrimeField + PrimeFieldBits> Circuit<Scalar> for SmallSha256Circuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let mut small_cs = SmallToBellpepperCS::<Scalar, CS>::new(cs);
    let preimage_bits = alloc_preimage_small_bits::<i8, _>(&mut small_cs, &self.preimage)?;
    let mut eq = NoBatchEq::<i8, i32, _>::new(&mut small_cs);
    let _ = small_sha256_int::<i8, _>(&mut eq, &preimage_bits)?;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper::gadgets::sha256::sha256 as bellpepper_sha256;
  use bellpepper_core::{ConstraintSystem, boolean::Boolean, test_cs::TestConstraintSystem};
  use ff::Field;
  use sha2::{Digest, Sha256};

  use crate::{
    provider::Bn254Engine,
    small_constraint_system::{
      SmallSatisfyingAssignment, SmallShapeCS, SparseMatrix, circuit::SmallSpartanCircuit,
    },
    traits::Engine,
  };

  type E = Bn254Engine;

  struct FieldReferenceRun {
    private_input_bits: Vec<bool>,
    public_digest_bits: Vec<bool>,
    is_satisfied: bool,
  }

  #[derive(Clone)]
  struct SmallCompilerRun {
    private_input_bits: Vec<bool>,
    public_digest_bits: Vec<bool>,
    is_satisfied: bool,
    z: Vec<i8>,
    first_private_preimage_z_index: usize,
    first_public_digest_z_index: usize,
    num_aux: usize,
    num_inputs: usize,
    num_constraints: usize,
    a: SparseMatrix<i32>,
    b: SparseMatrix<i32>,
    c: SparseMatrix<i32>,
  }

  fn bytes_to_bits_be(bytes: &[u8]) -> Vec<bool> {
    bytes
      .iter()
      .flat_map(|byte| (0..8).rev().map(move |i| ((byte >> i) & 1) == 1))
      .collect()
  }

  fn sha256_bits_be(preimage: &[u8]) -> Vec<bool> {
    bytes_to_bits_be(&Sha256::digest(preimage))
  }

  fn boolean_values(bits: &[Boolean]) -> Vec<bool> {
    bits.iter().map(|bit| bit.get_value().unwrap()).collect()
  }

  fn scalar_to_bit(value: &<E as Engine>::Scalar) -> bool {
    if *value == <E as Engine>::Scalar::ZERO {
      false
    } else {
      assert_eq!(*value, <E as Engine>::Scalar::ONE);
      true
    }
  }

  fn field_reference_sha256(preimage: &[u8]) -> FieldReferenceRun {
    let mut cs = TestConstraintSystem::<<E as Engine>::Scalar>::new();
    let input_bits =
      super::super::alloc_preimage_bits::<<E as Engine>::Scalar, _>(&mut cs, preimage, true)
        .unwrap();
    let digest_bits = bellpepper_sha256(cs.namespace(|| "bellpepper_sha256"), &input_bits).unwrap();
    super::super::expose_hash_bits_as_public::<E, _>(&mut cs, &digest_bits).unwrap();

    let public_digest_bits = cs
      .get_inputs()
      .iter()
      .skip(1)
      .map(|(value, _)| scalar_to_bit(value))
      .collect();

    FieldReferenceRun {
      private_input_bits: boolean_values(&input_bits),
      public_digest_bits,
      is_satisfied: cs.is_satisfied(),
    }
  }

  fn mat_vec(matrix: &SparseMatrix<i32>, z: &[i8]) -> Vec<i128> {
    matrix
      .indptr
      .windows(2)
      .map(|ptrs| {
        (ptrs[0]..ptrs[1])
          .map(|i| i128::from(matrix.data[i]) * i128::from(z[matrix.indices[i]]))
          .sum()
      })
      .collect()
  }

  fn small_r1cs_is_satisfied(
    a: &SparseMatrix<i32>,
    b: &SparseMatrix<i32>,
    c: &SparseMatrix<i32>,
    z: &[i8],
  ) -> bool {
    let az = mat_vec(a, z);
    let bz = mat_vec(b, z);
    let cz = mat_vec(c, z);

    az.iter()
      .zip(bz.iter())
      .zip(cz.iter())
      .all(|((az_i, bz_i), cz_i)| az_i * bz_i == *cz_i)
  }

  fn small_compiler_sha256(preimage: &[u8]) -> SmallCompilerRun {
    let circuit = SmallSha256Circuit::<<E as Engine>::Scalar>::new(preimage.to_vec());

    let mut shape_cs = SmallShapeCS::<i32>::new();
    let shared =
      <SmallSha256Circuit<<E as Engine>::Scalar> as SmallSpartanCircuit<E, i8, i32>>::shared(
        &circuit,
        &mut shape_cs,
      )
      .unwrap();
    let precommitted = <SmallSha256Circuit<<E as Engine>::Scalar> as SmallSpartanCircuit<
      E,
      i8,
      i32,
    >>::precommitted(&circuit, &mut shape_cs, &shared)
    .unwrap();
    <SmallSha256Circuit<<E as Engine>::Scalar> as SmallSpartanCircuit<E, i8, i32>>::synthesize(
      &circuit,
      &mut shape_cs,
      &shared,
      &precommitted,
      None,
    )
    .unwrap();
    let (a, b, c) = shape_cs.to_matrices();

    let mut witness_cs = SmallSatisfyingAssignment::<i8>::new();
    let shared =
      <SmallSha256Circuit<<E as Engine>::Scalar> as SmallSpartanCircuit<E, i8, i32>>::shared(
        &circuit,
        &mut witness_cs,
      )
      .unwrap();
    let precommitted = <SmallSha256Circuit<<E as Engine>::Scalar> as SmallSpartanCircuit<
      E,
      i8,
      i32,
    >>::precommitted(&circuit, &mut witness_cs, &shared)
    .unwrap();
    <SmallSha256Circuit<<E as Engine>::Scalar> as SmallSpartanCircuit<E, i8, i32>>::synthesize(
      &circuit,
      &mut witness_cs,
      &shared,
      &precommitted,
      None,
    )
    .unwrap();

    assert_eq!(shape_cs.num_vars(), witness_cs.aux_assignment.len());
    assert_eq!(shape_cs.num_inputs(), witness_cs.input_assignment.len());
    assert!(
      witness_cs
        .aux_assignment
        .iter()
        .all(|v| matches!(*v, 0 | 1))
    );
    assert!(
      witness_cs
        .input_assignment
        .iter()
        .all(|v| matches!(*v, 0 | 1))
    );

    let z = witness_cs
      .aux_assignment
      .iter()
      .chain(witness_cs.input_assignment.iter())
      .copied()
      .collect::<Vec<_>>();

    let input_bit_count = preimage.len() * 8;
    let private_input_bits = witness_cs.aux_assignment[..input_bit_count]
      .iter()
      .map(|value| *value == 1)
      .collect();
    let public_digest_bits = witness_cs.input_assignment[1..]
      .iter()
      .map(|value| *value == 1)
      .collect();

    SmallCompilerRun {
      private_input_bits,
      public_digest_bits,
      is_satisfied: small_r1cs_is_satisfied(&a, &b, &c, &z),
      z,
      first_private_preimage_z_index: 0,
      first_public_digest_z_index: witness_cs.aux_assignment.len() + 1,
      num_aux: shape_cs.num_vars(),
      num_inputs: shape_cs.num_inputs(),
      num_constraints: shape_cs.num_constraints(),
      a,
      b,
      c,
    }
  }

  #[test]
  fn full_sha256_field_reference_and_small_compiler_match_outputs() {
    let vectors = [
      vec![],
      b"abc".to_vec(),
      vec![0x11; 55],
      vec![0x22; 56],
      vec![0x33; 57],
      vec![0x44; 64],
      vec![0x55; 128],
    ];

    for preimage in vectors {
      let expected_input_bits = bytes_to_bits_be(&preimage);
      let expected_digest_bits = sha256_bits_be(&preimage);
      let field_reference = field_reference_sha256(&preimage);
      let small_compiler = small_compiler_sha256(&preimage);

      assert!(field_reference.is_satisfied);
      assert!(small_compiler.is_satisfied);
      assert_eq!(field_reference.private_input_bits, expected_input_bits);
      assert_eq!(small_compiler.private_input_bits, expected_input_bits);
      assert_eq!(field_reference.public_digest_bits, expected_digest_bits);
      assert_eq!(small_compiler.public_digest_bits, expected_digest_bits);
      assert_eq!(
        field_reference.public_digest_bits,
        small_compiler.public_digest_bits
      );
    }
  }

  #[test]
  fn small_compiler_rejects_wrong_public_digest() {
    let small_compiler = small_compiler_sha256(b"abc");
    let mut bad_z = small_compiler.z.clone();
    bad_z[small_compiler.first_public_digest_z_index] ^= 1;

    assert!(!small_r1cs_is_satisfied(
      &small_compiler.a,
      &small_compiler.b,
      &small_compiler.c,
      &bad_z,
    ));
  }

  #[test]
  fn small_compiler_rejects_wrong_private_preimage_witness() {
    let small_compiler = small_compiler_sha256(b"abc");
    let mut bad_z = small_compiler.z.clone();
    bad_z[small_compiler.first_private_preimage_z_index] ^= 1;

    assert!(!small_r1cs_is_satisfied(
      &small_compiler.a,
      &small_compiler.b,
      &small_compiler.c,
      &bad_z,
    ));
  }

  #[test]
  fn small_compiler_shape_is_value_independent_for_same_length() {
    let zeroes = small_compiler_sha256(&[0u8; 64]);
    let ones = small_compiler_sha256(&[0xff; 64]);
    let patterned = small_compiler_sha256(&(0..64).map(|i| i as u8).collect::<Vec<_>>());

    for other in [&ones, &patterned] {
      assert_eq!(zeroes.num_aux, other.num_aux);
      assert_eq!(zeroes.num_inputs, other.num_inputs);
      assert_eq!(zeroes.num_constraints, other.num_constraints);
      assert_eq!(zeroes.a, other.a);
      assert_eq!(zeroes.b, other.b);
      assert_eq!(zeroes.c, other.c);
    }
  }

  #[test]
  fn small_compiler_same_shape_allows_different_i8_witnesses() {
    let left = small_compiler_sha256(&[0x11u8; 64]);
    let right = small_compiler_sha256(&[0x22u8; 64]);

    assert!(left.is_satisfied);
    assert!(right.is_satisfied);
    assert_eq!(left.a, right.a);
    assert_eq!(left.b, right.b);
    assert_eq!(left.c, right.c);
    assert_ne!(left.z, right.z);
    assert!(left.z.iter().all(|value| matches!(*value, 0 | 1)));
    assert!(right.z.iter().all(|value| matches!(*value, 0 | 1)));
  }
}
