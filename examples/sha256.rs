//! examples/sha256.rs
//! Measure Spartan-2 {setup, prove, verify} times for a SHA-256
//! circuit with varying message lengths
//!
//! Run with: `RUST_LOG=info cargo run --release --example sha256`
#![allow(non_snake_case)]
use bellpepper::gadgets::sha256::sha256;
use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use ff::{Field, PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use spartan2::{
  provider::T256HyraxEngine,
  spartan::R1CSSNARK,
  traits::{Engine, circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use std::{marker::PhantomData, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

type E = T256HyraxEngine;

#[derive(Clone, Debug)]
struct Sha256Circuit<Scalar: PrimeField> {
  preimage: Vec<u8>,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> Sha256Circuit<Scalar> {
  fn new(preimage: Vec<u8>) -> Self {
    Self {
      preimage,
      _p: PhantomData,
    }
  }
}

impl<E: Engine> SpartanCircuit<E> for Sha256Circuit<E::Scalar> {
  fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
    // compute the SHA-256 hash of the preimage
    let mut hasher = Sha256::new();
    hasher.update(&self.preimage);
    let hash = hasher.finalize();
    // convert the hash to a vector of scalars
    let hash_scalars: Vec<<E as Engine>::Scalar> = hash
      .iter()
      .flat_map(|&byte| {
        (0..8).rev().map(move |i| {
          if (byte >> i) & 1 == 1 {
            E::Scalar::ONE
          } else {
            E::Scalar::ZERO
          }
        })
      })
      .collect();
    Ok(hash_scalars)
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // No shared variables in this circuit
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<E::Scalar>], // shared variables, if any
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // No precommitted variables in this circuit
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    // SHA-256 circuit does not expect any challenges
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
    _: &[AllocatedNum<E::Scalar>],
    _: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    // 1. Preimage bits
    let bit_values: Vec<_> = self
      .preimage
      .clone()
      .into_iter()
      .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1 == 1))
      .map(Some)
      .collect();
    assert_eq!(bit_values.len(), self.preimage.len() * 8);

    let preimage_bits = bit_values
      .into_iter()
      .enumerate()
      .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
      .map(|b| b.map(Boolean::from))
      .collect::<Result<Vec<_>, _>>()?;

    // 2. SHA-256 gadget
    let hash_bits = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

    // 3. Sanity-check against Rust SHA-256
    let mut hasher = Sha256::new();
    hasher.update(&self.preimage);
    let expected = hasher.finalize();

    let mut expected_bits = expected
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1));

    for b in &hash_bits {
      match b {
        Boolean::Is(bit) => assert_eq!(expected_bits.next().unwrap(), bit.get_value().unwrap()),
        Boolean::Not(bit) => assert_ne!(expected_bits.next().unwrap(), bit.get_value().unwrap()),
        Boolean::Constant(_) => unreachable!(),
      }
    }

    for (i, bit) in hash_bits.iter().enumerate() {
      // Allocate public input
      let n = AllocatedNum::alloc_input(cs.namespace(|| format!("public num {i}")), || {
        Ok(
          if bit.get_value().ok_or(SynthesisError::AssignmentMissing)? {
            E::Scalar::ONE
          } else {
            E::Scalar::ZERO
          },
        )
      })?;

      // Single equality constraint is enough
      cs.enforce(
        || format!("bit == num {i}"),
        |_| bit.lc(CS::one(), E::Scalar::ONE),
        |lc| lc + CS::one(),
        |lc| lc + n.get_variable(),
      );
    }

    Ok(())
  }
}

fn main() {
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)                // no bold colour codes
    .with_env_filter(EnvFilter::from_default_env())
    .init();

  // Message lengths: 2^10 … 2^11 bytes.
  let circuits: Vec<_> = (10..=11)
    .map(|k| Sha256Circuit::<<E as Engine>::Scalar>::new(vec![0u8; 1 << k]))
    .collect();

  for circuit in circuits {
    let msg_len = circuit.preimage.len();
    let root_span = info_span!("bench", msg_len).entered();
    info!("======= message_len={} bytes =======", msg_len);

    // SETUP
    let t0 = Instant::now();
    let (pk, vk) = R1CSSNARK::<E>::setup(circuit.clone()).expect("setup failed");
    let setup_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = setup_ms, "setup");

    // PREPARE
    let t0 = Instant::now();
    let mut prep_snark =
      R1CSSNARK::<E>::prep_prove(&pk, circuit.clone(), true).expect("prep_prove failed");
    let prep_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prep_ms, "prep_prove");

    // PROVE
    let t0 = Instant::now();
    let proof =
      R1CSSNARK::<E>::prove(&pk, circuit.clone(), &mut prep_snark, true).expect("prove failed");
    let prove_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prove_ms, "prove");

    // VERIFY
    let t0 = Instant::now();
    proof.verify(&vk).expect("verify errored");
    let verify_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = verify_ms, "verify");

    // Summary
    info!(
      "SUMMARY msg={}B, setup={} ms, prep_prove={} ms, prove={} ms, verify={} ms",
      msg_len, setup_ms, prep_ms, prove_ms, verify_ms
    );
    drop(root_span);
  }
}
