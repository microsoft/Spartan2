// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/sumcheck_sha256_equivalence.rs
//! Verify that prove_cubic_with_three_inputs and prove_cubic_with_three_inputs_small_value
//! produce identical proofs when used with a SmallSha256 circuit.
//!
//! This tests Algorithm 6 (small-value sumcheck optimization) against the standard method.
//! Unlike bellpepper's SHA-256 which has coefficients ~2^237, our SmallSha256Circuit uses
//! SmallMultiEq to keep coefficients within MAX_COEFF_BITS (31 for i32).
//!
//! Run with: `RUST_LOG=info cargo run --release --example sumcheck_sha256_equivalence`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper::gadgets::sha256::sha256 as bellpepper_sha256;
use bellpepper_core::{
  Circuit, ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
  test_cs::TestConstraintSystem,
};
use ff::{Field, PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use spartan2::{
  gadgets::small_sha256,
  polys::multilinear::MultilinearPolynomial,
  provider::PallasHyraxEngine,
  small_field::{I32NoBatch, I64Batch21, SmallMultiEqConfig, SmallValueField},
  spartan::SpartanSNARK,
  sumcheck::SumcheckProof,
  traits::{
    Engine, circuit::SpartanCircuit, snark::R1CSSNARKTrait, transcript::TranscriptEngineTrait,
  },
};
use std::{marker::PhantomData, time::Instant};
use tracing::{info, info_span, instrument};
use tracing_subscriber::{EnvFilter, fmt::time::uptime};

// Use PallasHyraxEngine which has Barrett-optimized SmallLargeMul for Fq
type E = PallasHyraxEngine;

/// SHA-256 circuit using SmallMultiEq for small-value optimization compatibility.
///
/// Unlike bellpepper's SHA-256 which produces coefficients ~2^237,
/// this circuit uses SmallMultiEq to keep coefficients within MAX_COEFF_BITS.
///
/// Generic over `C: SmallMultiEqConfig` to support both:
/// - `I32NoBatch<F>` (i32): Uses no batching, max coeff 2^31
/// - `I64Batch21<F>` (i64): Uses batching with 21 constraints, max coeff 2^21
#[derive(Debug)]
struct SmallSha256Circuit<Scalar: PrimeField, C> {
  preimage: Vec<u8>,
  _p: PhantomData<(Scalar, C)>,
}

// Manual Clone impl that doesn't require C: Clone (C is phantom)
impl<Scalar: PrimeField, C> Clone for SmallSha256Circuit<Scalar, C> {
  fn clone(&self) -> Self {
    Self {
      preimage: self.preimage.clone(),
      _p: PhantomData,
    }
  }
}

impl<Scalar: PrimeField + PrimeFieldBits, C: SmallMultiEqConfig> SmallSha256Circuit<Scalar, C> {
  fn new(preimage: Vec<u8>) -> Self {
    Self {
      preimage,
      _p: PhantomData,
    }
  }
}

impl<E: Engine, C: SmallMultiEqConfig> SpartanCircuit<E> for SmallSha256Circuit<E::Scalar, C>
where
  E::Scalar: SmallValueField<C::SmallValue>,
{
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
    // 1. Preimage bits (big-endian per byte for SHA-256)
    // IMPORTANT: Allocate as witness variables, not constants, so constraints are generated
    let preimage_bits: Vec<Boolean> = self
      .preimage
      .iter()
      .enumerate()
      .flat_map(|(byte_idx, &byte)| {
        (0..8).rev().enumerate().map(move |(bit_idx, i)| {
          let bit_val = (byte >> i) & 1 == 1;
          (byte_idx, bit_idx, bit_val)
        })
      })
      .map(|(byte_idx, bit_idx, bit_val)| {
        AllocatedBit::alloc(
          cs.namespace(|| format!("preimage_{}_{}", byte_idx, bit_idx)),
          Some(bit_val),
        )
        .map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // 2. SmallSHA-256 gadget (uses SmallMultiEq with C config for bounded coefficients)
    let hash_bits = small_sha256::<_, _, C>(cs, &preimage_bits)?;

    // 3. Sanity-check against Rust SHA-256
    let mut hasher = Sha256::new();
    hasher.update(&self.preimage);
    let expected = hasher.finalize();

    let expected_bits: Vec<bool> = expected
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1))
      .collect();

    for (i, (computed, expected_bit)) in hash_bits.iter().zip(expected_bits.iter()).enumerate() {
      let computed_val = match computed {
        Boolean::Is(bit) => bit.get_value().unwrap(),
        Boolean::Not(bit) => !bit.get_value().unwrap(),
        Boolean::Constant(b) => *b,
      };
      assert_eq!(
        computed_val, *expected_bit,
        "Hash bit {} mismatch: computed={}, expected={}",
        i, computed_val, expected_bit
      );
    }

    // 4. Expose hash bits as public inputs
    for (i, bit) in hash_bits.iter().enumerate() {
      let n = AllocatedNum::alloc_input(cs.namespace(|| format!("public num {i}")), || {
        Ok(
          if bit.get_value().ok_or(SynthesisError::AssignmentMissing)? {
            E::Scalar::ONE
          } else {
            E::Scalar::ZERO
          },
        )
      })?;

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

/// Original SHA-256 circuit using bellpepper's implementation.
/// This has coefficients ~2^237 which breaks small-value optimization.
#[derive(Clone)]
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
    // Allocate preimage bits
    let preimage_bits: Vec<Boolean> = self
      .preimage
      .iter()
      .enumerate()
      .flat_map(|(byte_idx, &byte)| {
        (0..8).map(move |i| {
          let bit_val = (byte >> i) & 1 == 1;
          (byte_idx, i, bit_val)
        })
      })
      .map(|(byte_idx, i, bit_val)| {
        AllocatedBit::alloc(
          cs.namespace(|| format!("bit {}_{}", byte_idx, i)),
          Some(bit_val),
        )
        .map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Synthesize bellpepper's SHA-256
    let _ = bellpepper_sha256(cs.namespace(|| "sha256"), &preimage_bits)?;
    Ok(())
  }
}

type F = <E as Engine>::Scalar;

impl<C: SmallMultiEqConfig> Circuit<F> for SmallSha256Circuit<F, C>
where
  F: SmallValueField<C::SmallValue>,
{
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // Allocate preimage bits (big-endian per byte for SHA-256)
    let preimage_bits: Vec<Boolean> = self
      .preimage
      .iter()
      .enumerate()
      .flat_map(|(byte_idx, &byte)| {
        (0..8).rev().map(move |i| {
          let bit_val = (byte >> i) & 1 == 1;
          (byte_idx, 7 - i, bit_val)
        })
      })
      .map(|(byte_idx, i, bit_val)| {
        AllocatedBit::alloc(
          cs.namespace(|| format!("bit {}_{}", byte_idx, i)),
          Some(bit_val),
        )
        .map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Synthesize small SHA-256 with C config
    let _ = small_sha256::<_, _, C>(cs, &preimage_bits)?;
    Ok(())
  }
}

#[instrument(skip_all)]
fn run_setup<C: SmallMultiEqConfig>(
  circuit: SmallSha256Circuit<F, C>,
) -> (
  <SpartanSNARK<E> as R1CSSNARKTrait<E>>::ProverKey,
  <SpartanSNARK<E> as R1CSSNARKTrait<E>>::VerifierKey,
)
where
  F: SmallValueField<C::SmallValue>,
{
  let t0 = Instant::now();
  let result = SpartanSNARK::<E>::setup(circuit).expect("setup failed");
  info!(elapsed_ms = t0.elapsed().as_millis(), "completed");
  result
}

#[instrument(skip_all)]
fn run_prep_prove<C: SmallMultiEqConfig>(
  pk: &<SpartanSNARK<E> as R1CSSNARKTrait<E>>::ProverKey,
  circuit: SmallSha256Circuit<F, C>,
) -> <SpartanSNARK<E> as R1CSSNARKTrait<E>>::PrepSNARK
where
  F: SmallValueField<C::SmallValue>,
{
  let t0 = Instant::now();
  let result = SpartanSNARK::<E>::prep_prove(pk, circuit, true).expect("prep_prove failed");
  info!(elapsed_ms = t0.elapsed().as_millis(), "completed");
  result
}

#[instrument(skip_all)]
fn extract_sumcheck_inputs<C: SmallMultiEqConfig>(
  pk: &<SpartanSNARK<E> as R1CSSNARKTrait<E>>::ProverKey,
  circuit: SmallSha256Circuit<F, C>,
  prep_snark: &<SpartanSNARK<E> as R1CSSNARKTrait<E>>::PrepSNARK,
) -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>)
where
  F: SmallValueField<C::SmallValue>,
{
  let t0 = Instant::now();
  let (az, bz, cz, tau) = SpartanSNARK::<E>::extract_outer_sumcheck_inputs(pk, circuit, prep_snark)
    .expect("extract_outer_sumcheck_inputs failed");
  info!(elapsed_ms = t0.elapsed().as_millis(), "completed");

  // Check small-value compatibility: Az and Bz must fit in the config's Small type
  let mut az_failures = 0;
  let mut bz_failures = 0;
  for (i, val) in az.iter().enumerate() {
    if <F as SmallValueField<C::SmallValue>>::try_field_to_small(val).is_none() {
      if az_failures < 5 {
        info!("Az[{}] doesn't fit in small type: {:?}", i, val);
      }
      az_failures += 1;
    }
  }
  for (i, val) in bz.iter().enumerate() {
    if <F as SmallValueField<C::SmallValue>>::try_field_to_small(val).is_none() {
      if bz_failures < 5 {
        info!("Bz[{}] doesn't fit in small type: {:?}", i, val);
      }
      bz_failures += 1;
    }
  }
  if az_failures > 0 || bz_failures > 0 {
    info!(
      az_failures,
      bz_failures, "Small-value compatibility check FAILED"
    );
  }

  (az, bz, cz, tau)
}

fn test_sumcheck_equivalence_for_message_len<C: SmallMultiEqConfig>(
  preimage_len: usize,
  config_name: &str,
) where
  F: SmallValueField<C::SmallValue> + SmallValueField<i64, IntermediateSmallValue = i128>,
{
  let _span = info_span!("test", msg_len = preimage_len, config = config_name).entered();

  // Create both circuits
  let preimage = vec![0u8; preimage_len];
  let small_circuit = SmallSha256Circuit::<F, C>::new(preimage.clone());
  let bellpepper_circuit = Sha256Circuit::<F>::new(preimage);

  // Synthesize and count constraints for SmallSha256
  // (clone because Circuit::synthesize consumes self)
  let mut cs1 = TestConstraintSystem::<F>::new();
  small_circuit
    .clone()
    .synthesize(&mut cs1)
    .expect("small_sha256 synthesis failed");
  let small_sha256_constraints = cs1.num_constraints();

  // Synthesize and count constraints for bellpepper SHA256
  let mut cs2 = TestConstraintSystem::<F>::new();
  bellpepper_circuit
    .synthesize(&mut cs2)
    .expect("bellpepper synthesis failed");
  let bellpepper_sha256_constraints = cs2.num_constraints();

  info!(
    msg_len = preimage_len,
    small_sha256_constraints, bellpepper_sha256_constraints, "Constraint comparison"
  );

  let (pk, _vk) = run_setup(small_circuit.clone());
  let prep_snark = run_prep_prove(&pk, small_circuit.clone());
  let (az, bz, cz, tau) = extract_sumcheck_inputs(&pk, small_circuit, &prep_snark);

  let num_vars = tau.len();
  info!(
    num_vars,
    poly_len = az.len(),
    "Extracted sumcheck polynomials"
  );

  // Create field-element polynomials for the original method
  let mut az1 = MultilinearPolynomial::new(az.clone());
  let mut bz1 = MultilinearPolynomial::new(bz.clone());
  let mut cz1 = MultilinearPolynomial::new(cz.clone());

  // Try to create small-value polynomials for the optimized method
  // This will return None if any witness value is too large to fit in i64
  // For SHA-256, we use i64/i128 config because positional coefficients are ~2^34
  let az_poly = MultilinearPolynomial::new(az.clone());
  let bz_poly = MultilinearPolynomial::new(bz.clone());
  let az_small_opt = MultilinearPolynomial::<i64>::try_from_field(&az_poly);
  let bz_small_opt = MultilinearPolynomial::<i64>::try_from_field(&bz_poly);

  // Claim is zero for satisfying R1CS (Az * Bz = Cz)
  let claim = <E as Engine>::Scalar::ZERO;

  // ===== ORIGINAL METHOD =====
  info!("Running prove_cubic_with_three_inputs (original method)...");
  let mut transcript1 = <E as Engine>::TE::new(b"test_equivalence");
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
  // Only run if witness values fit in i64
  match (az_small_opt, bz_small_opt) {
    (Some(az_small), Some(bz_small)) => {
      info!("Witness values fit in i64, running small-value optimization with i64/i128 config...");

      // Verify i64 polynomials match field polynomials (debug check)
      let az_back: MultilinearPolynomial<F> = az_small.to_field();
      let bz_back: MultilinearPolynomial<F> = bz_small.to_field();
      let mut mismatch_count = 0;
      for i in 0..az.len() {
        if az[i] != az_back[i] {
          mismatch_count += 1;
          if mismatch_count <= 5 {
            info!(
              "Az mismatch at index {}: field={:?}, i64_to_field={:?}",
              i, az[i], az_back[i]
            );
          }
        }
        if bz[i] != bz_back[i] {
          mismatch_count += 1;
          if mismatch_count <= 5 {
            info!(
              "Bz mismatch at index {}: field={:?}, i64_to_field={:?}",
              i, bz[i], bz_back[i]
            );
          }
        }
      }
      if mismatch_count > 0 {
        panic!(
          "Polynomial mismatch: {} values differ. i64 conversion is lossy!",
          mismatch_count
        );
      }
      info!("Verified: i64 polynomials match field polynomials");

      let mut az2 = MultilinearPolynomial::new(az);
      let mut bz2 = MultilinearPolynomial::new(bz);
      let mut cz2 = MultilinearPolynomial::new(cz);
      let mut transcript2 = <E as Engine>::TE::new(b"test_equivalence");

      info!("Running prove_cubic_with_three_inputs_small_value (Algorithm 6)...");
      let t0 = Instant::now();
      let (proof2, r2, evals2) = SumcheckProof::<E>::prove_cubic_with_three_inputs_small_value(
        &claim,
        tau,
        &az_small,
        &bz_small,
        &mut az2,
        &mut bz2,
        &mut cz2,
        &mut transcript2,
      )
      .expect("prove_cubic_with_three_inputs_small_value failed");
      let smallvalue_us = t0.elapsed().as_micros();
      info!(
        elapsed_us = smallvalue_us,
        "prove_cubic_with_three_inputs_small_value"
      );

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
        msg_len = preimage_len,
        small_sha256_constraints,
        bellpepper_sha256_constraints,
        original_sumcheck_us = original_us,
        small_value_sumcheck_us = smallvalue_us,
        speedup = format!("{:.2}x", speedup),
        "PASSED: proofs are equivalent"
      );
    }
    _ => {
      panic!(
        "Az/Bz values too large for small-value optimization (don't fit in i64). \
         Config: {}. The small-value optimization requires all Az and Bz values to fit in i64.",
        config_name
      );
    }
  }
}

fn main() {
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_timer(uptime())
    .with_env_filter(EnvFilter::from_default_env())
    .init();

  info!("Testing sumcheck method equivalence with SmallSha256Circuit");
  info!("SmallSha256Circuit uses SmallMultiEq to keep coefficients bounded");
  info!("Note: Small-value sumcheck (Algorithm 6) requires even num_vars");

  // Message lengths: 1024 bytes produces num_vars=20 (even), 2048 bytes produces num_vars=21 (odd)
  // Algorithm 6 requires even num_vars, so we only test 1024 bytes
  // TODO: Extend Algorithm 6 to support odd num_vars
  let preimage_len = 1024;

  // Test with I32NoBatch (i32, no batching) - direct constraints, max coeff 2^31
  info!("===== Testing with I32NoBatch (i32, no batching) =====");
  test_sumcheck_equivalence_for_message_len::<I32NoBatch<F>>(preimage_len, "I32NoBatch");

  // Test with I64Batch21 (i64, batch 21) - batched constraints, max coeff 2^21
  info!("===== Testing with I64Batch21 (i64, batch 21) =====");
  test_sumcheck_equivalence_for_message_len::<I64Batch21<F>>(preimage_len, "I64Batch21");

  info!("All tests passed for both I32NoBatch and I64Batch21!");
}
