// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/sumcheck_sweep.rs
//! Sweep benchmark for sumcheck methods across polynomial sizes 2^10 to 2^30
//! Produces CSV output for chart generation
//!
//! Run with: RUST_LOG=info cargo run --release --example sumcheck_sweep
//! Or for CSV only: cargo run --release --example sumcheck_sweep 2>/dev/null > results.csv

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use spartan2::{
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  provider::PallasHyraxEngine,
  sumcheck::SumcheckProof,
  traits::{transcript::TranscriptEngineTrait, Engine},
};
use std::io::Write;
use std::time::Instant;
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

type E = PallasHyraxEngine;
type F = <E as Engine>::Scalar;

fn run_single_benchmark(num_vars: usize) -> (u128, u128) {
  let _span = info_span!("benchmark", num_vars).entered();
  let n = 1usize << num_vars;

  // Setup: deterministic polynomials
  let az_vals: Vec<F> = (0..n).map(|i| F::from((i + 1) as u64)).collect();
  let bz_vals: Vec<F> = (0..n).map(|i| F::from((i + 3) as u64)).collect();
  let cz_vals: Vec<F> = az_vals
    .iter()
    .zip(&bz_vals)
    .map(|(a, b)| *a * *b - F::from(5u64))
    .collect();
  let taus: Vec<F> = (0..num_vars).map(|i| F::from((i + 2) as u64)).collect();

  // Compute claim
  let eq_evals = EqPolynomial::evals_from_points(&taus);
  let claim: F = (0..n)
    .map(|i| eq_evals[i] * (az_vals[i] * bz_vals[i] - cz_vals[i]))
    .sum();

  // ===== ORIGINAL METHOD =====
  let mut az1 = MultilinearPolynomial::new(az_vals.clone());
  let mut bz1 = MultilinearPolynomial::new(bz_vals.clone());
  let mut cz1 = MultilinearPolynomial::new(cz_vals.clone());
  let mut transcript1 = <E as Engine>::TE::new(b"bench");

  let t0 = Instant::now();
  let (_proof1, r1, evals1) = SumcheckProof::<E>::prove_cubic_with_three_inputs(
    &claim,
    taus.clone(),
    &mut az1,
    &mut bz1,
    &mut cz1,
    &mut transcript1,
  )
  .unwrap();
  let original_us = t0.elapsed().as_micros();
  info!(elapsed_us = original_us, "prove_cubic_with_three_inputs");

  // ===== SMALL-VALUE METHOD =====
  let mut az2 = MultilinearPolynomial::new(az_vals);
  let mut bz2 = MultilinearPolynomial::new(bz_vals);
  let mut cz2 = MultilinearPolynomial::new(cz_vals);
  let mut transcript2 = <E as Engine>::TE::new(b"bench");

  let t0 = Instant::now();
  let (_proof2, r2, evals2) = SumcheckProof::<E>::prove_cubic_with_three_inputs_small_value(
    &claim,
    taus,
    &mut az2,
    &mut bz2,
    &mut cz2,
    &mut transcript2,
  )
  .unwrap();
  let smallvalue_us = t0.elapsed().as_micros();
  info!(
    elapsed_us = smallvalue_us,
    "prove_cubic_with_three_inputs_small_value"
  );

  // Verify equivalence
  assert_eq!(r1, r2, "challenges must match");
  assert_eq!(evals1, evals2, "final evaluations must match");

  (original_us, smallvalue_us)
}

fn main() {
  // Initialize tracing (logs to stderr so CSV can go to stdout)
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .with_writer(std::io::stderr)
    .init();

  // Print CSV header
  println!("num_vars,n,original_us,smallvalue_us,speedup");

  // Sweep from 2^10 to 2^30
  // Note: 2^30 requires ~8GB RAM for 3 polynomials, adjust max based on available memory
  let max_vars = std::env::var("MAX_VARS")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(24); // Default to 2^24 (~16M elements) for reasonable runtime

  // Small-value method requires:
  // 1. l0 <= num_vars / 2 where l0 = 3 (default), so num_vars >= 6
  // 2. num_vars must be even (Algorithm 6 split expects even ℓ)
  // We start at 10 and step by 2 to satisfy both constraints.
  for num_vars in (10..=max_vars).step_by(2) {

    let n = 1usize << num_vars;

    // Run multiple iterations for small sizes to reduce variance
    let iterations = if num_vars <= 15 { 5 } else { 1 };

    let mut original_total = 0u128;
    let mut smallvalue_total = 0u128;

    for _ in 0..iterations {
      let (orig, sv) = run_single_benchmark(num_vars);
      original_total += orig;
      smallvalue_total += sv;
    }

    let original_us = original_total / iterations as u128;
    let smallvalue_us = smallvalue_total / iterations as u128;
    let speedup = if smallvalue_us > 0 {
      original_us as f64 / smallvalue_us as f64
    } else {
      f64::INFINITY
    };

    println!(
      "{},{},{},{},{:.3}",
      num_vars, n, original_us, smallvalue_us, speedup
    );

    // Flush to see progress
    std::io::stdout().flush().ok();
  }
}
