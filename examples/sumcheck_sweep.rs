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
use ff::Field;
use spartan2::{
  cli::FieldChoice,
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  provider::{Bn254Engine, PallasHyraxEngine, VestaHyraxEngine},
  small_field::{DelayedReduction, SmallValueField},
  small_sumcheck::prove_cubic_small_value,
  sumcheck::SumcheckProof,
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use std::{io::Write, ops::Mul, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

// ============================================================================
// SumcheckBenchmark Trait and BenchResult
// ============================================================================

/// Benchmark result with separate setup and prove times
#[derive(Clone, Copy, Default)]
struct BenchResult {
  setup_us: u128,
  prove_us: u128,
}

/// Trait for sumcheck benchmarks with associated Value type.
trait SumcheckBenchmark<E: Engine> {
  /// The value type for polynomial evaluations (E::Scalar, i64, or i32)
  type Value: Clone + Mul<Output = Self::Value>;

  /// Convert i32 witness value to Self::Value
  fn convert(v: i32) -> Self::Value;

  /// Run the sumcheck prover with given polynomials
  fn prove(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    az: MultilinearPolynomial<Self::Value>,
    bz: MultilinearPolynomial<Self::Value>,
    cz: MultilinearPolynomial<Self::Value>,
    transcript: &mut E::TE,
  ) -> (SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>);
}

/// Standalone benchmark runner - shared logic for all benchmark types
fn run_benchmark<E, B>(
  name: &'static str,
  num_vars: usize,
  az_i32: &[i32],
  bz_i32: &[i32],
  taus: &[E::Scalar],
  claim: &E::Scalar,
) -> BenchResult
where
  E: Engine,
  B: SumcheckBenchmark<E>,
{
  let mut transcript = E::TE::new(b"bench");

  let t_setup = Instant::now();
  let (az, bz, cz) = make_polys(az_i32, bz_i32, B::convert);
  let setup_us = t_setup.elapsed().as_micros();

  let t_prove = Instant::now();
  let (proof, r, evals) = B::prove(claim, taus.to_vec(), az, bz, cz, &mut transcript);
  let prove_us = t_prove.elapsed().as_micros();

  verify_sumcheck_proof::<E>(&proof, claim, num_vars, taus, &r, &evals);
  info!(setup_us, prove_us, method = name, "benchmark");
  BenchResult { setup_us, prove_us }
}

// ============================================================================
// Benchmark Implementations
// ============================================================================

/// Baseline cubic sumcheck using prove_cubic_with_three_inputs
struct BaseCubic;

impl<E: Engine> SumcheckBenchmark<E> for BaseCubic {
  type Value = E::Scalar;

  fn convert(v: i32) -> Self::Value {
    E::Scalar::from(v as u64)
  }

  fn prove(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    mut az: MultilinearPolynomial<Self::Value>,
    mut bz: MultilinearPolynomial<Self::Value>,
    mut cz: MultilinearPolynomial<Self::Value>,
    transcript: &mut E::TE,
  ) -> (SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>) {
    SumcheckProof::<E>::prove_cubic_with_three_inputs(claim, taus, &mut az, &mut bz, &mut cz, transcript)
      .unwrap()
  }
}

/// Split-eq with delayed modular reduction
struct SplitEqDmr;

impl<E> SumcheckBenchmark<E> for SplitEqDmr
where
  E: Engine,
  E::Scalar: DelayedReduction<E::Scalar>,
{
  type Value = E::Scalar;

  fn convert(v: i32) -> Self::Value {
    E::Scalar::from(v as u64)
  }

  fn prove(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    mut az: MultilinearPolynomial<Self::Value>,
    mut bz: MultilinearPolynomial<Self::Value>,
    mut cz: MultilinearPolynomial<Self::Value>,
    transcript: &mut E::TE,
  ) -> (SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>) {
    SumcheckProof::<E>::prove_cubic_with_three_inputs_split_eq_delayed(
      claim, taus, &mut az, &mut bz, &mut cz, transcript,
    )
    .unwrap()
  }
}

/// Small-value i64 optimization
struct SmallValueI64;

impl<E> SumcheckBenchmark<E> for SmallValueI64
where
  E: Engine,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  type Value = i64;

  fn convert(v: i32) -> Self::Value {
    v as i64
  }

  fn prove(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    az: MultilinearPolynomial<Self::Value>,
    bz: MultilinearPolynomial<Self::Value>,
    cz: MultilinearPolynomial<Self::Value>,
    transcript: &mut E::TE,
  ) -> (SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>) {
    prove_cubic_small_value::<E, _, 3>(claim, taus, &az, &bz, &cz, transcript).unwrap()
  }
}

/// Small-value i32 optimization
struct SmallValueI32;

impl<E> SumcheckBenchmark<E> for SmallValueI32
where
  E: Engine,
  E::Scalar: SmallValueField<i32>
    + DelayedReduction<i32>
    + DelayedReduction<i64>
    + DelayedReduction<E::Scalar>,
{
  type Value = i32;

  fn convert(v: i32) -> Self::Value {
    v
  }

  fn prove(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    az: MultilinearPolynomial<Self::Value>,
    bz: MultilinearPolynomial<Self::Value>,
    cz: MultilinearPolynomial<Self::Value>,
    transcript: &mut E::TE,
  ) -> (SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>) {
    prove_cubic_small_value::<E, _, 3>(claim, taus, &az, &bz, &cz, transcript).unwrap()
  }
}

/// Get method name for CSV output
fn method_name(method: &SumcheckMethod) -> &'static str {
  match method {
    SumcheckMethod::Base => "base",
    SumcheckMethod::SplitEqDmr => "split_eq_dmr",
    SumcheckMethod::I64 => "i64",
    SumcheckMethod::I32 => "i32",
  }
}

/// Run sweep with multiple methods, outputting combined CSV rows
fn run_sweep_multi<E>(
  methods: &[SumcheckMethod],
  min_vars: usize,
  max_vars: usize,
  num_trials: usize,
) where
  E: Engine,
  E::Scalar: SmallValueField<i64>
    + SmallValueField<i32>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<i32>
    + DelayedReduction<E::Scalar>,
{
  // Build header
  let mut header = vec!["num_vars".to_string(), "n".to_string(), "trial".to_string()];
  for method in methods {
    let name = method_name(method);
    header.push(format!("{}_setup_us", name));
    header.push(format!("{}_prove_us", name));
  }
  // Add speedup columns if base is included
  let has_base = methods.contains(&SumcheckMethod::Base);
  if has_base {
    for method in methods {
      let name = method_name(method);
      if name != "base" {
        header.push(format!("prove_speedup_{}", name));
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
      // Run each benchmark via direct dispatch
      let results: Vec<(&str, BenchResult)> = methods
        .iter()
        .map(|method| {
          let name = method_name(method);
          let result = match method {
            SumcheckMethod::Base => {
              run_benchmark::<E, BaseCubic>(name, num_vars, &az_i32, &bz_i32, &taus, &claim)
            }
            SumcheckMethod::SplitEqDmr => {
              run_benchmark::<E, SplitEqDmr>(name, num_vars, &az_i32, &bz_i32, &taus, &claim)
            }
            SumcheckMethod::I64 => {
              run_benchmark::<E, SmallValueI64>(name, num_vars, &az_i32, &bz_i32, &taus, &claim)
            }
            SumcheckMethod::I32 => {
              run_benchmark::<E, SmallValueI32>(name, num_vars, &az_i32, &bz_i32, &taus, &claim)
            }
          };
          (name, result)
        })
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
  #[arg(long, value_enum, default_value = "base,i64", value_delimiter = ',')]
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
      run_sweep_multi::<PallasHyraxEngine>(&args.methods, min_vars, max_vars, args.trials);
    }
    FieldChoice::VestaFp => {
      run_sweep_multi::<VestaHyraxEngine>(&args.methods, min_vars, max_vars, args.trials);
    }
    FieldChoice::Bn254Fr => {
      run_sweep_multi::<Bn254Engine>(&args.methods, min_vars, max_vars, args.trials);
    }
  }
}

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
// Shared Verification
// ============================================================================

/// Shared verification logic
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
