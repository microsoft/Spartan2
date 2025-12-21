// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/sumcheck_sha256_equivalence.rs
//! Verify that prove_cubic_with_three_inputs and prove_cubic_with_three_inputs_small_value
//! produce identical proofs when used with a real SHA-256 circuit.
//!
//! This tests Algorithm 6 (small-value sumcheck optimization) against the standard method.
//!
//! Run with: `RUST_LOG=info cargo run --release --example sumcheck_sha256_equivalence`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper::gadgets::sha256::sha256;
use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
  ConstraintSystem, SynthesisError,
};
use ff::{Field, PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use spartan2::{
  polys::multilinear::MultilinearPolynomial,
  provider::T256HyraxEngine,
  spartan::SpartanSNARK,
  sumcheck::SumcheckProof,
  traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait, transcript::TranscriptEngineTrait, Engine},
};
use std::{marker::PhantomData, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

type E = T256HyraxEngine;

/// SHA-256 circuit for testing sumcheck equivalence
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
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>], // shared variables, if any
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
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

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    // SHA-256 circuit does not expect any challenges
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

fn main() {
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .init();

  // Use a small message for faster testing (64 bytes = 512 bits)
  let preimage_len = 64;
  let circuit = Sha256Circuit::<<E as Engine>::Scalar>::new(vec![0u8; preimage_len]);

  let _span = info_span!("sumcheck_equivalence", preimage_len).entered();
  info!("Testing sumcheck method equivalence with SHA-256 circuit (preimage_len={} bytes)", preimage_len);

  // SETUP
  info!("Running setup...");
  let t0 = Instant::now();
  let (pk, _vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup failed");
  info!(elapsed_ms = t0.elapsed().as_millis(), "setup");

  // PREP_PROVE
  info!("Running prep_prove...");
  let t0 = Instant::now();
  let prep_snark = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).expect("prep_prove failed");
  info!(elapsed_ms = t0.elapsed().as_millis(), "prep_prove");

  // EXTRACT SUMCHECK INPUTS
  info!("Extracting Az, Bz, Cz, tau from circuit...");
  let t0 = Instant::now();
  let (az, bz, cz, tau) = SpartanSNARK::<E>::extract_outer_sumcheck_inputs(&pk, circuit, &prep_snark)
    .expect("extract_outer_sumcheck_inputs failed");
  info!(elapsed_ms = t0.elapsed().as_millis(), "extract_outer_sumcheck_inputs");

  let num_vars = tau.len();
  info!(num_vars = num_vars, az_len = az.len(), "Extracted polynomials");

  // Clone data for both methods
  let mut az1 = MultilinearPolynomial::new(az.clone());
  let mut bz1 = MultilinearPolynomial::new(bz.clone());
  let mut cz1 = MultilinearPolynomial::new(cz.clone());

  let mut az2 = MultilinearPolynomial::new(az);
  let mut bz2 = MultilinearPolynomial::new(bz);
  let mut cz2 = MultilinearPolynomial::new(cz);

  // Fresh transcripts with same seed for deterministic challenges
  let mut transcript1 = <E as Engine>::TE::new(b"test_equivalence");
  let mut transcript2 = <E as Engine>::TE::new(b"test_equivalence");

  // Claim is zero for satisfying R1CS (Az * Bz = Cz)
  let claim = <E as Engine>::Scalar::ZERO;

  // ===== ORIGINAL METHOD =====
  info!("Running prove_cubic_with_three_inputs (original method)...");
  let t0 = Instant::now();
  let (proof1, r1, evals1) = SumcheckProof::<E>::prove_cubic_with_three_inputs(
    &claim,
    tau.clone(),
    &mut az1,
    &mut bz1,
    &mut cz1,
    &mut transcript1,
  )
  .expect("prove_cubic_with_three_inputs failed");
  let original_us = t0.elapsed().as_micros();
  info!(elapsed_us = original_us, "prove_cubic_with_three_inputs");

  // ===== SMALL-VALUE METHOD (Algorithm 6) =====
  info!("Running prove_cubic_with_three_inputs_small_value (Algorithm 6)...");
  let t0 = Instant::now();
  let (proof2, r2, evals2) = SumcheckProof::<E>::prove_cubic_with_three_inputs_small_value(
    &claim,
    tau,
    &mut az2,
    &mut bz2,
    &mut cz2,
    &mut transcript2,
  )
  .expect("prove_cubic_with_three_inputs_small_value failed");
  let smallvalue_us = t0.elapsed().as_micros();
  info!(elapsed_us = smallvalue_us, "prove_cubic_with_three_inputs_small_value");

  // ===== VERIFY EQUIVALENCE =====
  info!("Verifying equivalence...");

  // Check challenges match
  assert_eq!(r1, r2, "Challenges must match!");
  info!("Challenges match (len={})", r1.len());

  // Check round polynomials match
  assert_eq!(proof1, proof2, "Round polynomials must match!");
  info!("Round polynomials match");

  // Check final evaluations match
  assert_eq!(evals1, evals2, "Final evaluations must match!");
  info!("Final evaluations match");

  // Calculate speedup
  let speedup = if smallvalue_us > 0 {
    original_us as f64 / smallvalue_us as f64
  } else {
    f64::INFINITY
  };

  info!(
    "SUCCESS! Both methods produce identical proofs. Original: {} us, Small-value: {} us, Speedup: {:.2}x",
    original_us, smallvalue_us, speedup
  );
}
