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
//!
//! CLI modes:
//!   single 26               - Run only 26 variables (for profiling)
//!   range-sweep             - Sweep 10-24 (default)
//!   range-sweep --min 20 --max 26 - Custom range

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use clap::{Parser, Subcommand};
use spartan2::{
  polys::multilinear::MultilinearPolynomial,
  provider::PallasHyraxEngine,
  sumcheck::SumcheckProof,
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use std::{io::Write, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(about = "Sumcheck benchmark sweep")]
struct Args {
  #[command(subcommand)]
  command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
  /// Run a single level (for profiling)
  Single { vars: usize },
  /// Run a range sweep
  RangeSweep {
    #[arg(long, default_value = "10")]
    min: usize,
    #[arg(long, default_value = "24")]
    max: usize,
  },
}

type E = PallasHyraxEngine;
type F = <E as Engine>::Scalar;

/// Helper to create field-element polynomials from i32 values
#[inline]
fn make_field_polys(az_i32: &[i32], bz_i32: &[i32]) -> (Vec<F>, Vec<F>, Vec<F>) {
  let az_vals: Vec<F> = az_i32.iter().map(|&v| F::from(v as u64)).collect();
  let bz_vals: Vec<F> = bz_i32.iter().map(|&v| F::from(v as u64)).collect();
  let cz_vals: Vec<F> = az_vals.iter().zip(&bz_vals).map(|(a, b)| *a * *b).collect();
  (az_vals, bz_vals, cz_vals)
}

/// Returns (original_us, i32_small_us, i64_small_us)
/// - original_us: full sumcheck with field elements
/// - i32_small_us: full sumcheck with i32/i64 small-value optimization
/// - i64_small_us: full sumcheck with i64/i128 small-value optimization
///
/// Memory-optimized: each benchmark runs in its own scope so memory is freed
/// before the next benchmark starts. This reduces peak memory from ~78GB to ~26GB
/// for num_vars=28.
fn run_single_benchmark(num_vars: usize) -> (u128, u128, u128) {
  let _span = info_span!("benchmark", num_vars).entered();
  let n = 1usize << num_vars;

  // Setup: deterministic polynomials with satisfying witness (Cz = Az * Bz)
  // build_accumulators_spartan assumes Az·Bz = Cz on binary points
  //
  // Create small i32 values - these are small and kept for all benchmarks
  let az_i32: Vec<i32> = (0..n).map(|i| (i + 1) as i32).collect();
  let bz_i32: Vec<i32> = (0..n).map(|i| (i + 3) as i32).collect();
  let taus: Vec<F> = (0..num_vars).map(|i| F::from((i + 2) as u64)).collect();

  // Claim = 0 for satisfying witness
  let claim: F = F::from(0u64);

  // ===== ORIGINAL METHOD =====
  // Run in scope so memory is freed before next benchmark
  let (original_us, r1, evals1) = {
    let (az_vals, bz_vals, cz_vals) = make_field_polys(&az_i32, &bz_i32);
    let mut az1 = MultilinearPolynomial::new(az_vals);
    let mut bz1 = MultilinearPolynomial::new(bz_vals);
    let mut cz1 = MultilinearPolynomial::new(cz_vals);
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
    let elapsed = t0.elapsed().as_micros();
    info!(elapsed_us = elapsed, "prove_cubic_with_three_inputs");
    (elapsed, r1, evals1)
  }; // az1, bz1, cz1, az_vals, bz_vals, cz_vals dropped here

  // ===== SMALL-VALUE METHOD (i32/i64) =====
  // Run in scope so memory is freed before next benchmark
  let (smallvalue_us, r2, evals2) = {
    // Create small-value polynomials directly from i32 (no field conversion!)
    // This is the key optimization: ss multiplications use native i64 arithmetic
    let az_small = MultilinearPolynomial::new(az_i32.clone());
    let bz_small = MultilinearPolynomial::new(bz_i32.clone());

    // Also need field-element polynomials for binding in later rounds
    let (az_vals, bz_vals, cz_vals) = make_field_polys(&az_i32, &bz_i32);
    let mut az2 = MultilinearPolynomial::new(az_vals);
    let mut bz2 = MultilinearPolynomial::new(bz_vals);
    let mut cz2 = MultilinearPolynomial::new(cz_vals);
    let mut transcript2 = <E as Engine>::TE::new(b"bench");

    let t0 = Instant::now();
    let (_proof2, r2, evals2) = SumcheckProof::<E>::prove_cubic_with_three_inputs_small_value(
      &claim,
      taus.clone(),
      &az_small,
      &bz_small,
      &mut az2,
      &mut bz2,
      &mut cz2,
      &mut transcript2,
    )
    .unwrap();
    let elapsed = t0.elapsed().as_micros();
    info!(
      elapsed_us = elapsed,
      "prove_cubic_with_three_inputs_small_value (i32/i64)"
    );
    (elapsed, r2, evals2)
  }; // az_small, bz_small, az2, bz2, cz2, etc. dropped here

  // Verify equivalence
  assert_eq!(r1, r2, "challenges must match");
  assert_eq!(evals1, evals2, "final evaluations must match");

  // ===== SMALL-VALUE METHOD (i64/i128) =====
  let (i64_smallvalue_us, r3, evals3) = {
    // Create i64 small-value polynomials
    let az_i64: Vec<i64> = az_i32.iter().map(|&v| v as i64).collect();
    let bz_i64: Vec<i64> = bz_i32.iter().map(|&v| v as i64).collect();
    let az_small_i64 = MultilinearPolynomial::new(az_i64);
    let bz_small_i64 = MultilinearPolynomial::new(bz_i64);

    // Need fresh field-element polynomials for binding in later rounds
    let (az_vals, bz_vals, cz_vals) = make_field_polys(&az_i32, &bz_i32);
    let mut az3 = MultilinearPolynomial::new(az_vals);
    let mut bz3 = MultilinearPolynomial::new(bz_vals);
    let mut cz3 = MultilinearPolynomial::new(cz_vals);
    let mut transcript3 = <E as Engine>::TE::new(b"bench");

    let t0 = Instant::now();
    let (_proof3, r3, evals3) = SumcheckProof::<E>::prove_cubic_with_three_inputs_small_value(
      &claim,
      taus,
      &az_small_i64,
      &bz_small_i64,
      &mut az3,
      &mut bz3,
      &mut cz3,
      &mut transcript3,
    )
    .unwrap();
    let elapsed = t0.elapsed().as_micros();
    info!(
      elapsed_us = elapsed,
      "prove_cubic_with_three_inputs_small_value (i64/i128)"
    );
    (elapsed, r3, evals3)
  };

  // Verify i64 also matches
  assert_eq!(r1, r3, "i64 challenges must match");
  assert_eq!(evals1, evals3, "i64 final evaluations must match");

  (original_us, smallvalue_us, i64_smallvalue_us)
}

fn run_sumcheck_sweep(min_vars: usize, max_vars: usize) {
  // Print CSV header
  // - original_us: full sumcheck with field elements
  // - i32_small_us: full sumcheck with i32/i64 optimization
  // - i64_small_us: full sumcheck with i64/i128 optimization
  // - orig_vs_i32: speedup of i32 small-value over original
  // - orig_vs_i64: speedup of i64 small-value over original
  println!("num_vars,n,original_us,i32_small_us,i64_small_us,orig_vs_i32,orig_vs_i64");

  // Small-value method requires: l0 <= num_vars / 2 where l0 = 3 (default), so num_vars >= 6
  for num_vars in min_vars..=max_vars {
    let n = 1usize << num_vars;

    // Run multiple iterations for small sizes to reduce variance
    let iterations = if num_vars <= 15 { 5 } else { 1 };

    let mut original_total = 0u128;
    let mut i32_smallvalue_total = 0u128;
    let mut i64_smallvalue_total = 0u128;

    for _ in 0..iterations {
      let (orig, i32_sv, i64_sv) = run_single_benchmark(num_vars);
      original_total += orig;
      i32_smallvalue_total += i32_sv;
      i64_smallvalue_total += i64_sv;
    }

    let original_us = original_total / iterations as u128;
    let i32_smallvalue_us = i32_smallvalue_total / iterations as u128;
    let i64_smallvalue_us = i64_smallvalue_total / iterations as u128;

    let orig_vs_i32 = if i32_smallvalue_us > 0 {
      original_us as f64 / i32_smallvalue_us as f64
    } else {
      f64::INFINITY
    };

    let orig_vs_i64 = if i64_smallvalue_us > 0 {
      original_us as f64 / i64_smallvalue_us as f64
    } else {
      f64::INFINITY
    };

    println!(
      "{},{},{},{},{},{:.3},{:.3}",
      num_vars, n, original_us, i32_smallvalue_us, i64_smallvalue_us, orig_vs_i32, orig_vs_i64
    );

    // Flush to see progress
    std::io::stdout().flush().ok();
  }
}

fn main() {
  // Initialize tracing (logs to stderr so CSV can go to stdout)
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .with_writer(std::io::stderr)
    .init();

  let args = Args::parse();

  let (min_vars, max_vars) = match args.command {
    Some(Command::Single { vars }) => (vars, vars),
    Some(Command::RangeSweep { min, max }) => (min, max),
    None => (10, 24), // Default: sweep 10-24
  };

  eprintln!(
    "Running sumcheck benchmark (min={}, max={})...",
    min_vars, max_vars
  );
  run_sumcheck_sweep(min_vars, max_vars);
}
