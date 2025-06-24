//! examples/sha256.rs
//! Measure Spartan-2 {setup, gen_witness, prove, verify} times for a SHA-256
//! circuit with message lengths 2⁶ … 2¹⁶ bytes.
//!
//! Run with: `RUST_LOG=info cargo run --release --example sha256`
#![allow(non_snake_case)]
use bellpepper::gadgets::{Assignment, sha256::sha256};
use bellpepper_core::{
  Circuit, ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::{AllocatedNum, Num},
};
use ff::{PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use spartan2::{
  R1CSSNARK,
  provider::T256HyraxEngine,
  traits::{Engine, snark::R1CSSNARKTrait},
};
use std::{marker::PhantomData, time::Instant};
use tracing::{info, info_span};

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

impl<Scalar: PrimeField + PrimeFieldBits> Circuit<Scalar> for Sha256Circuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // 1. Preimage bits ----------------------------------------------------
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

    // 2. SHA-256 gadget ---------------------------------------------------
    let hash_bits = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

    // 3. Pack 256 bits into two field elements ---------------------------
    let mut hash_fes = Vec::new();
    for (i, chunk) in hash_bits.chunks(128).enumerate() {
      let mut num = Num::<Scalar>::zero();
      let mut coeff = Scalar::ONE;
      for bit in chunk {
        num = num.add_bool_with_coeff(CS::one(), bit, coeff);
        coeff = coeff.double();
      }

      let hash_fe = AllocatedNum::alloc(cs.namespace(|| format!("hash_fe {i}")), || {
        Ok(*num.get_value().get()?)
      })?;

      // Enforce packing: num * 1 = hash_fe
      cs.enforce(
        || format!("packing constraint {i}"),
        |_| num.lc(Scalar::ONE),
        |lc| lc + CS::one(),
        |lc| lc + hash_fe.get_variable(),
      );
      hash_fes.push(hash_fe);
    }

    // 4. Sanity-check against Rust SHA-256 -------------------------------
    let mut hasher = Sha256::new();
    hasher.update(&self.preimage);
    let expected = hasher.finalize();

    let mut expected_bits = expected
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1));

    for b in hash_bits {
      match b {
        Boolean::Is(bit) => assert_eq!(expected_bits.next().unwrap(), bit.get_value().unwrap()),
        Boolean::Not(bit) => assert_ne!(expected_bits.next().unwrap(), bit.get_value().unwrap()),
        Boolean::Constant(_) => unreachable!(),
      }
    }

    // 5. Expose hash outputs as public inputs ----------------------------
    hash_fes[0].inputize(cs.namespace(|| "hash output 0"))?;
    hash_fes[1].inputize(cs.namespace(|| "hash output 1"))?;
    Ok(())
  }
}

fn main() {
  tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()) // use RUST_LOG
        .with_target(false)
        .init();

  // Message lengths: 2⁶ … 2¹⁶ bytes.
  let circuits: Vec<_> = (6..=12)
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

    // GENERATE WITNESS
    let t0 = Instant::now();
    let (U, W) = R1CSSNARK::<E>::gen_witness(&pk, circuit.clone()).expect("gen_witness failed");
    let gw_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = gw_ms, "gen_witness");

    // PROVE
    let t0 = Instant::now();
    let proof = R1CSSNARK::<E>::prove(&pk, &U, &W).expect("prove failed");
    let prove_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prove_ms, "prove");

    // VERIFY
    let t0 = Instant::now();
    proof.verify(&vk).expect("verify errored");
    let verify_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = verify_ms, "verify");

    // Summary
    info!(
      "SUMMARY msg={}B, setup={} ms, gen_witness={} ms, prove={} ms, verify={} ms",
      msg_len, setup_ms, gw_ms, prove_ms, verify_ms
    );
    drop(root_span);
  }
}
