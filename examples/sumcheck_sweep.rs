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
//!   --method i64 single 26        - Run only 26 variables with i64 method (for profiling)
//!   --method base range-sweep     - Sweep 10-24 with base method
//!   --method i64 range-sweep --min 20 --max 26 - Custom range with i64 method

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use clap::{Parser, Subcommand, ValueEnum};
use spartan2::{
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  provider::{Bn254Engine, PallasHyraxEngine, VestaHyraxEngine},
  small_field::{DelayedReduction, SmallValueField},
  small_sumcheck::prove_cubic_small_value,
  sumcheck::SumcheckProof,
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use ff::Field;
use std::{io::Write, ops::Mul, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

// ============================================================================
// Generic poly construction (uses closure for conversion, std Mul for products)
// ============================================================================

/// Generic poly construction using closure for conversion
fn make_polys<T, F>(
  az_i32: &[i32],
  bz_i32: &[i32],
  convert: F,
) -> (
  MultilinearPolynomial<T>,
  MultilinearPolynomial<T>,
  MultilinearPolynomial<T>,
)
where
  T: Clone + Mul<Output = T>,
  F: Fn(i32) -> T,
{
  let az: Vec<T> = az_i32.iter().map(|&v| convert(v)).collect();
  let bz: Vec<T> = bz_i32.iter().map(|&v| convert(v)).collect();
  let cz: Vec<T> = az
    .iter()
    .zip(&bz)
    .map(|(a, b)| a.clone() * b.clone())
    .collect();
  (
    MultilinearPolynomial::new(az),
    MultilinearPolynomial::new(bz),
    MultilinearPolynomial::new(cz),
  )
}

// ============================================================================
// SumcheckBenchmark Trait
// ============================================================================

/// Trait for benchmarkable sumcheck methods
trait SumcheckBenchmark<E: Engine, T: Clone + Mul<Output = T>> {
  /// Human-readable name for CSV output
  fn name(&self) -> &'static str;

  /// Conversion function for i32 -> T
  fn convert(v: i32) -> T;

  /// Run the prover with polynomials of type T
  fn prove(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    az: MultilinearPolynomial<T>,
    bz: MultilinearPolynomial<T>,
    cz: MultilinearPolynomial<T>,
    transcript: &mut E::TE,
  ) -> (SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>);
}

// ============================================================================
// Shared Verification
// ============================================================================

/// Shared verification logic (not duplicated in each impl)
fn verify_sumcheck_proof<E: Engine>(
  proof: &SumcheckProof<E>,
  claim: &E::Scalar,
  num_vars: usize,
  taus: &[E::Scalar],
  expected_r: &[E::Scalar],
  expected_evals: &[E::Scalar],
) {
  let mut transcript_v = E::TE::new(b"bench");
  let (final_claim, r_v) = proof
    .verify(*claim, num_vars, 3, &mut transcript_v)
    .unwrap();
  assert_eq!(r_v, expected_r, "Verify challenges must match prover");
  let tau_eval = EqPolynomial::new(taus.to_vec()).evaluate(&r_v);
  let expected = tau_eval * (expected_evals[0] * expected_evals[1] - expected_evals[2]);
  assert_eq!(final_claim, expected, "Sumcheck final claim mismatch");
}

// ============================================================================
// Benchmark Implementations
// ============================================================================

/// Baseline cubic sumcheck using prove_cubic_with_three_inputs
struct BaseCubic;

impl<E: Engine> SumcheckBenchmark<E, E::Scalar> for BaseCubic {
  fn name(&self) -> &'static str {
    "base"
  }

  fn convert(v: i32) -> E::Scalar {
    E::Scalar::from(v as u64)
  }

  fn prove(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    mut az: MultilinearPolynomial<E::Scalar>,
    mut bz: MultilinearPolynomial<E::Scalar>,
    mut cz: MultilinearPolynomial<E::Scalar>,
    transcript: &mut E::TE,
  ) -> (SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>) {
    SumcheckProof::<E>::prove_cubic_with_three_inputs(
      claim,
      taus,
      &mut az,
      &mut bz,
      &mut cz,
      transcript,
    )
    .unwrap()
  }
}

/// Split-eq with delayed modular reduction
struct SplitEqDmr;

impl<E> SumcheckBenchmark<E, E::Scalar> for SplitEqDmr
where
  E: Engine,
  E::Scalar: DelayedReduction<E::Scalar>,
{
  fn name(&self) -> &'static str {
    "split_eq_dmr"
  }

  fn convert(v: i32) -> E::Scalar {
    E::Scalar::from(v as u64)
  }

  fn prove(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    mut az: MultilinearPolynomial<E::Scalar>,
    mut bz: MultilinearPolynomial<E::Scalar>,
    mut cz: MultilinearPolynomial<E::Scalar>,
    transcript: &mut E::TE,
  ) -> (SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>) {
    SumcheckProof::<E>::prove_cubic_with_three_inputs_split_eq_delayed(
      claim,
      taus,
      &mut az,
      &mut bz,
      &mut cz,
      transcript,
    )
    .unwrap()
  }
}

/// Small-value i64 optimization
struct SmallValueI64;

impl<E> SumcheckBenchmark<E, i64> for SmallValueI64
where
  E: Engine,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  fn name(&self) -> &'static str {
    "i64"
  }

  fn convert(v: i32) -> i64 {
    v as i64
  }

  fn prove(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    az: MultilinearPolynomial<i64>,
    bz: MultilinearPolynomial<i64>,
    cz: MultilinearPolynomial<i64>,
    transcript: &mut E::TE,
  ) -> (SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>) {
    prove_cubic_small_value::<E, _, 3>(claim, taus, &az, &bz, &cz, transcript).unwrap()
  }
}

/// Small-value i32 optimization
struct SmallValueI32;

impl<E> SumcheckBenchmark<E, i32> for SmallValueI32
where
  E: Engine,
  E::Scalar: SmallValueField<i32>
    + DelayedReduction<i32>
    + DelayedReduction<i64>
    + DelayedReduction<E::Scalar>,
{
  fn name(&self) -> &'static str {
    "i32"
  }

  fn convert(v: i32) -> i32 {
    v
  }

  fn prove(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    az: MultilinearPolynomial<i32>,
    bz: MultilinearPolynomial<i32>,
    cz: MultilinearPolynomial<i32>,
    transcript: &mut E::TE,
  ) -> (SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>) {
    prove_cubic_small_value::<E, _, 3>(claim, taus, &az, &bz, &cz, transcript).unwrap()
  }
}

// ============================================================================
// Generic Runner
// ============================================================================

/// Benchmark result with separate setup and prove times
#[derive(Clone, Copy, Default)]
struct BenchResult {
  setup_us: u128,
  prove_us: u128,
}

/// Run a single benchmark and return the result
fn run_single<E, T, B>(
  bench: &B,
  num_vars: usize,
  az_i32: &[i32],
  bz_i32: &[i32],
  taus: &[E::Scalar],
  claim: &E::Scalar,
) -> BenchResult
where
  E: Engine,
  T: Clone + Mul<Output = T>,
  B: SumcheckBenchmark<E, T>,
{
  // Setup: convert to polys using B::convert (timed)
  let t_setup = Instant::now();
  let (az, bz, cz) = make_polys(az_i32, bz_i32, B::convert);
  let mut transcript = E::TE::new(b"bench");
  let setup_us = t_setup.elapsed().as_micros();

  // Prove via trait (timed)
  let t_prove = Instant::now();
  let (proof, r, evals) = B::prove(claim, taus.to_vec(), az, bz, cz, &mut transcript);
  let prove_us = t_prove.elapsed().as_micros();

  // Verify (shared, not timed)
  verify_sumcheck_proof::<E>(&proof, claim, num_vars, taus, &r, &evals);

  info!(setup_us, prove_us, method = bench.name(), "benchmark");
  BenchResult { setup_us, prove_us }
}

/// Helper trait for running benchmarks with type erasure
trait BenchmarkRunner<E: Engine> {
  fn name(&self) -> &'static str;
  fn run(
    &self,
    num_vars: usize,
    az_i32: &[i32],
    bz_i32: &[i32],
    taus: &[E::Scalar],
    claim: &E::Scalar,
  ) -> BenchResult;
}

/// Wrapper to implement BenchmarkRunner for any SumcheckBenchmark
struct BenchWrapper<B, T>(B, std::marker::PhantomData<T>);

impl<B, T> BenchWrapper<B, T> {
  fn new(bench: B) -> Self {
    Self(bench, std::marker::PhantomData)
  }
}

impl<E, T, B> BenchmarkRunner<E> for BenchWrapper<B, T>
where
  E: Engine,
  T: Clone + Mul<Output = T>,
  B: SumcheckBenchmark<E, T>,
{
  fn name(&self) -> &'static str {
    self.0.name()
  }

  fn run(
    &self,
    num_vars: usize,
    az_i32: &[i32],
    bz_i32: &[i32],
    taus: &[E::Scalar],
    claim: &E::Scalar,
  ) -> BenchResult {
    run_single::<E, T, B>(&self.0, num_vars, az_i32, bz_i32, taus, claim)
  }
}

/// Run sweep with multiple methods, outputting combined CSV rows
fn run_sweep_multi<E: Engine>(
  benchmarks: Vec<Box<dyn BenchmarkRunner<E>>>,
  min_vars: usize,
  max_vars: usize,
  num_trials: usize,
) {
  // Build header
  let mut header = vec!["num_vars".to_string(), "n".to_string(), "trial".to_string()];
  for bench in &benchmarks {
    header.push(format!("{}_setup_us", bench.name()));
    header.push(format!("{}_prove_us", bench.name()));
  }
  // Add speedup columns if base is included
  let has_base = benchmarks.iter().any(|b| b.name() == "base");
  if has_base {
    for bench in &benchmarks {
      if bench.name() != "base" {
        header.push(format!("prove_speedup_{}", bench.name()));
      }
    }
  }
  println!("{}", header.join(","));
  std::io::stdout().flush().ok();

  for num_vars in min_vars..=max_vars {
    let _span = info_span!("benchmark", num_vars).entered();
    let n = 1usize << num_vars;

    // Common: create i32 test data
    let az_i32: Vec<i32> = (0..n).map(|i| (i + 1) as i32).collect();
    let bz_i32: Vec<i32> = (0..n).map(|i| (i + 3) as i32).collect();
    let taus: Vec<E::Scalar> = (0..num_vars)
      .map(|i| E::Scalar::from((i + 2) as u64))
      .collect();
    let claim = E::Scalar::ZERO;

    for trial in 1..=num_trials {
      // Run each benchmark
      let results: Vec<(&str, BenchResult)> = benchmarks
        .iter()
        .map(|b| (b.name(), b.run(num_vars, &az_i32, &bz_i32, &taus, &claim)))
        .collect();

      // Build row
      let mut row: Vec<String> = vec![num_vars.to_string(), n.to_string(), trial.to_string()];
      for (_, result) in &results {
        row.push(result.setup_us.to_string());
        row.push(result.prove_us.to_string());
      }

      // Add speedup columns if base is included
      if has_base {
        let base_prove = results
          .iter()
          .find(|(name, _)| *name == "base")
          .map(|(_, r)| r.prove_us)
          .unwrap_or(1);
        for (name, result) in &results {
          if *name != "base" {
            let speedup = if result.prove_us > 0 {
              base_prove as f64 / result.prove_us as f64
            } else {
              f64::INFINITY
            };
            row.push(format!("{:.3}", speedup));
          }
        }
      }

      println!("{}", row.join(","));
      std::io::stdout().flush().ok();
    }
  }
}

// ============================================================================
// CLI
// ============================================================================

/// Field choice for benchmarks
#[derive(ValueEnum, Clone, Default, Debug)]
enum FieldChoice {
  /// Pallas curve scalar field (Fq)
  #[default]
  PallasFq,
  /// Vesta curve scalar field (Fp)
  VestaFp,
  /// BN254 curve scalar field (Fr)
  Bn254Fr,
}

/// Sumcheck method to benchmark
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
enum SumcheckMethod {
  /// Baseline cubic sumcheck
  Base,
  /// Split-eq with delayed modular reduction
  SplitEqDmr,
  /// Small-value i32 optimization
  I32,
  /// Small-value i64 optimization
  I64,
}

#[derive(Parser)]
#[command(about = "Sumcheck benchmark sweep")]
struct Args {
  /// Field to use for benchmarks
  #[arg(long, value_enum, default_value = "pallas-fq")]
  field: FieldChoice,

  /// Methods to benchmark (comma-separated)
  #[arg(
    long,
    value_enum,
    default_value = "base,i64",
    value_delimiter = ','
  )]
  methods: Vec<SumcheckMethod>,

  /// Number of trials per num_vars (each recorded separately)
  #[arg(long, default_value = "1")]
  trials: usize,

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

// ============================================================================
// Main
// ============================================================================

/// Build benchmark runners for Pallas field
fn build_pallas_benchmarks(
  methods: &[SumcheckMethod],
) -> Vec<Box<dyn BenchmarkRunner<PallasHyraxEngine>>> {
  let mut benchmarks: Vec<Box<dyn BenchmarkRunner<PallasHyraxEngine>>> = Vec::new();
  for method in methods {
    match method {
      SumcheckMethod::Base => benchmarks.push(Box::new(BenchWrapper::new(BaseCubic))),
      SumcheckMethod::SplitEqDmr => benchmarks.push(Box::new(BenchWrapper::new(SplitEqDmr))),
      SumcheckMethod::I64 => benchmarks.push(Box::new(BenchWrapper::new(SmallValueI64))),
      SumcheckMethod::I32 => benchmarks.push(Box::new(BenchWrapper::new(SmallValueI32))),
    }
  }
  benchmarks
}

/// Build benchmark runners for Vesta field
fn build_vesta_benchmarks(
  methods: &[SumcheckMethod],
) -> Vec<Box<dyn BenchmarkRunner<VestaHyraxEngine>>> {
  let mut benchmarks: Vec<Box<dyn BenchmarkRunner<VestaHyraxEngine>>> = Vec::new();
  for method in methods {
    match method {
      SumcheckMethod::Base => benchmarks.push(Box::new(BenchWrapper::new(BaseCubic))),
      SumcheckMethod::SplitEqDmr => benchmarks.push(Box::new(BenchWrapper::new(SplitEqDmr))),
      SumcheckMethod::I64 => benchmarks.push(Box::new(BenchWrapper::new(SmallValueI64))),
      SumcheckMethod::I32 => benchmarks.push(Box::new(BenchWrapper::new(SmallValueI32))),
    }
  }
  benchmarks
}

/// Build benchmark runners for BN254 field (no i32 support)
fn build_bn254_benchmarks(
  methods: &[SumcheckMethod],
) -> Vec<Box<dyn BenchmarkRunner<Bn254Engine>>> {
  let mut benchmarks: Vec<Box<dyn BenchmarkRunner<Bn254Engine>>> = Vec::new();
  for method in methods {
    match method {
      SumcheckMethod::Base => benchmarks.push(Box::new(BenchWrapper::new(BaseCubic))),
      SumcheckMethod::SplitEqDmr => benchmarks.push(Box::new(BenchWrapper::new(SplitEqDmr))),
      SumcheckMethod::I64 => benchmarks.push(Box::new(BenchWrapper::new(SmallValueI64))),
      SumcheckMethod::I32 => {
        eprintln!("Warning: i32 method not supported for BN254 field, skipping");
      }
    }
  }
  benchmarks
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
    "Running sumcheck benchmark (field={:?}, methods={:?}, min={}, max={}, trials={})...",
    args.field, args.methods, min_vars, max_vars, args.trials
  );

  // Dispatch based on field
  match args.field {
    FieldChoice::PallasFq => {
      let benchmarks = build_pallas_benchmarks(&args.methods);
      run_sweep_multi::<PallasHyraxEngine>(benchmarks, min_vars, max_vars, args.trials);
    }
    FieldChoice::VestaFp => {
      let benchmarks = build_vesta_benchmarks(&args.methods);
      run_sweep_multi::<VestaHyraxEngine>(benchmarks, min_vars, max_vars, args.trials);
    }
    FieldChoice::Bn254Fr => {
      let benchmarks = build_bn254_benchmarks(&args.methods);
      run_sweep_multi::<Bn254Engine>(benchmarks, min_vars, max_vars, args.trials);
    }
  }
}
