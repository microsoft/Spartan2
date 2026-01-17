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
//! SmallMultiEq to keep coefficients within bounded ranges.
//!
//! Run with: `RUST_LOG=info cargo run --release --example sumcheck_sha256_equivalence`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[path = "circuits/mod.rs"]
mod circuits;

use bellpepper_core::{Circuit, test_cs::TestConstraintSystem};
use circuits::{Sha256Circuit, SmallSha256Circuit};
use ff::Field;
use spartan2::{
  polys::multilinear::MultilinearPolynomial,
  provider::PallasHyraxEngine,
  small_field::SmallValueField,
  spartan::SpartanSNARK,
  sumcheck::SumcheckProof,
  traits::{Engine, snark::R1CSSNARKTrait, transcript::TranscriptEngineTrait},
};
use std::time::Instant;
use tracing::{info, info_span, instrument};
use tracing_subscriber::{EnvFilter, fmt::time::uptime};

// Use PallasHyraxEngine which has Barrett-optimized SmallLargeMul for Fq
type E = PallasHyraxEngine;
type F = <E as Engine>::Scalar;

#[instrument(skip_all)]
fn run_setup(
  circuit: SmallSha256Circuit<F>,
) -> (
  <SpartanSNARK<E> as R1CSSNARKTrait<E>>::ProverKey,
  <SpartanSNARK<E> as R1CSSNARKTrait<E>>::VerifierKey,
) {
  let t0 = Instant::now();
  let result = SpartanSNARK::<E>::setup(circuit).expect("setup failed");
  info!(elapsed_ms = t0.elapsed().as_millis(), "completed");
  result
}

#[instrument(skip_all)]
fn run_prep_prove(
  pk: &<SpartanSNARK<E> as R1CSSNARKTrait<E>>::ProverKey,
  circuit: SmallSha256Circuit<F>,
) -> <SpartanSNARK<E> as R1CSSNARKTrait<E>>::PrepSNARK {
  let t0 = Instant::now();
  let result = SpartanSNARK::<E>::prep_prove(pk, circuit, true).expect("prep_prove failed");
  info!(elapsed_ms = t0.elapsed().as_millis(), "completed");
  result
}

#[instrument(skip_all)]
fn extract_sumcheck_inputs(
  pk: &<SpartanSNARK<E> as R1CSSNARKTrait<E>>::ProverKey,
  circuit: SmallSha256Circuit<F>,
  prep_snark: &<SpartanSNARK<E> as R1CSSNARKTrait<E>>::PrepSNARK,
) -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>) {
  let t0 = Instant::now();
  let (az, bz, cz, tau) = SpartanSNARK::<E>::extract_outer_sumcheck_inputs(pk, circuit, prep_snark)
    .expect("extract_outer_sumcheck_inputs failed");
  info!(elapsed_ms = t0.elapsed().as_millis(), "completed");

  // Check small-value compatibility: Az and Bz must fit in i64
  let mut az_failures = 0;
  let mut bz_failures = 0;
  for (i, val) in az.iter().enumerate() {
    if <F as SmallValueField<i64>>::try_field_to_small(val).is_none() {
      if az_failures < 5 {
        info!("Az[{}] doesn't fit in i64: {:?}", i, val);
      }
      az_failures += 1;
    }
  }
  for (i, val) in bz.iter().enumerate() {
    if <F as SmallValueField<i64>>::try_field_to_small(val).is_none() {
      if bz_failures < 5 {
        info!("Bz[{}] doesn't fit in i64: {:?}", i, val);
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

fn test_sumcheck_equivalence_for_message_len(preimage_len: usize, use_batching: bool)
where
  F: SmallValueField<i64, IntermediateSmallValue = i128>,
{
  let config_name = if use_batching {
    "BatchingEq<21>"
  } else {
    "NoBatchEq"
  };
  let _span = info_span!("test", msg_len = preimage_len, config = config_name).entered();

  // Create both circuits
  let preimage = vec![0u8; preimage_len];
  let small_circuit = SmallSha256Circuit::<F>::new(preimage.clone(), use_batching);
  let bellpepper_circuit = Sha256Circuit::<F>::new(preimage);

  // Synthesize and count constraints for SmallSha256
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

  // Claim is zero for satisfying R1CS (Az * Bz = Cz)
  let claim = F::ZERO;

  // ===== ORIGINAL METHOD =====
  // Run in scope so memory is freed before next benchmark
  let (proof1, r1, evals1, original_us) = {
    let mut az1 = MultilinearPolynomial::new(az.clone());
    let mut bz1 = MultilinearPolynomial::new(bz.clone());
    let mut cz1 = MultilinearPolynomial::new(cz.clone());

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
    let elapsed = t0.elapsed().as_micros();
    info!(elapsed_us = elapsed, "prove_cubic_with_three_inputs");
    (proof1, r1, evals1, elapsed)
  }; // az1, bz1, cz1 dropped here

  // Try to create small-value polynomials for the optimized method
  let az_poly = MultilinearPolynomial::new(az.clone());
  let bz_poly = MultilinearPolynomial::new(bz.clone());
  let az_small_opt = MultilinearPolynomial::<i64>::try_from_field(&az_poly);
  let bz_small_opt = MultilinearPolynomial::<i64>::try_from_field(&bz_poly);

  // ===== SMALL-VALUE METHOD (Algorithm 6) =====
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

      // Run in scope so memory is freed after benchmark
      let (proof2, r2, evals2, smallvalue_us) = {
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
        let elapsed = t0.elapsed().as_micros();
        info!(
          elapsed_us = elapsed,
          "prove_cubic_with_three_inputs_small_value"
        );
        (proof2, r2, evals2, elapsed)
      };

      // ===== VERIFY EQUIVALENCE =====
      info!("Verifying equivalence...");

      assert_eq!(r1, r2, "Challenges must match!");
      info!("Challenges match (len={})", r1.len());

      assert_eq!(proof1, proof2, "Round polynomials must match!");
      info!("Round polynomials match");

      assert_eq!(evals1, evals2, "Final evaluations must match!");
      info!("Final evaluations match");

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

  // Message lengths: 1024 bytes produces num_vars=20
  let preimage_len = 1024;

  // Test with NoBatchEq (i32 path) - direct constraints
  info!("===== Testing with NoBatchEq (i32 path, no batching) =====");
  test_sumcheck_equivalence_for_message_len(preimage_len, false);

  // Test with BatchingEq<21> (i64 path) - batched constraints
  info!("===== Testing with BatchingEq<21> (i64 path, batch 21) =====");
  test_sumcheck_equivalence_for_message_len(preimage_len, true);

  info!("All tests passed for both NoBatchEq and BatchingEq<21>!");
}
