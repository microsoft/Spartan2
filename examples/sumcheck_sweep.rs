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

use clap::{Parser, Subcommand, ValueEnum};
use spartan2::{
  polys::multilinear::MultilinearPolynomial,
  provider::{Bn254Engine, PallasHyraxEngine, VestaHyraxEngine},
  small_field::{DelayedReduction, SmallValueField},
  sumcheck::SumcheckProof,
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use std::{io::Write, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

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

#[derive(Parser)]
#[command(about = "Sumcheck benchmark sweep")]
struct Args {
  /// Field to use for benchmarks
  #[arg(long, value_enum, default_value = "pallas-fq")]
  field: FieldChoice,

  /// Methods to benchmark (comma-separated: base,i32,i64). Default: all
  #[arg(long, default_value = "base,i32,i64", value_delimiter = ',')]
  methods: Vec<String>,

  /// Number of trials per num_vars (each recorded separately)
  #[arg(long, default_value = "1")]
  trials: usize,

  #[command(subcommand)]
  command: Option<Command>,
}

/// Tracks which benchmark methods to run
struct BenchMethods {
  base: bool,
  split_eq_dmr: bool,
  i32_small: bool,
  i64_small: bool,
}

impl BenchMethods {
  fn from_args(methods: &[String]) -> Self {
    Self {
      base: methods.iter().any(|m| m == "base"),
      split_eq_dmr: methods.iter().any(|m| m == "split-eq-dmr"),
      i32_small: methods.iter().any(|m| m == "i32"),
      i64_small: methods.iter().any(|m| m == "i64"),
    }
  }

  /// Returns a copy with i32 disabled (for fields that don't support it)
  fn without_i32(&self) -> Self {
    Self {
      base: self.base,
      split_eq_dmr: self.split_eq_dmr,
      i32_small: false,
      i64_small: self.i64_small,
    }
  }
}

/// Benchmark result with separate setup and prove times
#[derive(Clone, Copy, Default)]
struct BenchResult {
  setup_us: u128,
  prove_us: u128,
}

/// Result of a single trial for a given num_vars
struct TrialResult {
  num_vars: usize,
  n: usize,
  trial: usize,
  base: Option<BenchResult>,
  split_eq_dmr: Option<BenchResult>,
  i32_small: Option<BenchResult>,
  i64_small: Option<BenchResult>,
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

/// Helper to create field-element polynomials from i32 values
#[inline]
fn make_field_polys<F: ff::PrimeField>(az_i32: &[i32], bz_i32: &[i32]) -> (Vec<F>, Vec<F>, Vec<F>) {
  let az_vals: Vec<F> = az_i32.iter().map(|&v| F::from(v as u64)).collect();
  let bz_vals: Vec<F> = bz_i32.iter().map(|&v| F::from(v as u64)).collect();
  let cz_vals: Vec<F> = az_vals.iter().zip(&bz_vals).map(|(a, b)| *a * *b).collect();
  (az_vals, bz_vals, cz_vals)
}

/// Returns (base_result, split_eq_dmr_result, i64_result) as Options based on selected methods.
/// Each BenchResult contains separate setup_us and prove_us timings.
///
/// Memory-optimized: each benchmark runs in its own scope so memory is freed
/// before the next benchmark starts. This reduces peak memory from ~78GB to ~26GB
/// for num_vars=28.
fn run_single_benchmark<E>(
  num_vars: usize,
  methods: &BenchMethods,
) -> (Option<BenchResult>, Option<BenchResult>, Option<BenchResult>)
where
  E: Engine,
  E::Scalar: SmallValueField<i64, IntermediateSmallValue = i128> + DelayedReduction<i64>,
{
  type F<E> = <E as Engine>::Scalar;

  let _span = info_span!("benchmark", num_vars).entered();
  let n = 1usize << num_vars;

  // Setup: deterministic polynomials with satisfying witness (Cz = Az * Bz)
  // build_accumulators_spartan assumes Az·Bz = Cz on binary points
  //
  // Create small i32 values - these are small and kept for all benchmarks
  let az_i32: Vec<i32> = (0..n).map(|i| (i + 1) as i32).collect();
  let bz_i32: Vec<i32> = (0..n).map(|i| (i + 3) as i32).collect();
  let taus: Vec<F<E>> = (0..num_vars)
    .map(|i| F::<E>::from((i + 2) as u64))
    .collect();

  // Claim = 0 for satisfying witness
  let claim: F<E> = F::<E>::from(0u64);

  // ===== ORIGINAL METHOD =====
  // Run in scope so memory is freed before next benchmark
  let (result_base, r1, evals1) = if methods.base {
    let t_setup = Instant::now();
    let (az_vals, bz_vals, cz_vals) = make_field_polys::<F<E>>(&az_i32, &bz_i32);
    let mut az1 = MultilinearPolynomial::new(az_vals);
    let mut bz1 = MultilinearPolynomial::new(bz_vals);
    let mut cz1 = MultilinearPolynomial::new(cz_vals);
    let mut transcript1 = E::TE::new(b"bench");
    let setup_us = t_setup.elapsed().as_micros();

    let t_prove = Instant::now();
    let (_proof1, r1, evals1) = SumcheckProof::<E>::prove_cubic_with_three_inputs(
      &claim,
      taus.clone(),
      &mut az1,
      &mut bz1,
      &mut cz1,
      &mut transcript1,
    )
    .unwrap();
    let prove_us = t_prove.elapsed().as_micros();
    info!(setup_us, prove_us, "prove_cubic_with_three_inputs");
    (
      Some(BenchResult { setup_us, prove_us }),
      Some(r1),
      Some(evals1),
    )
  } else {
    (None, None, None)
  }; // az1, bz1, cz1, az_vals, bz_vals, cz_vals dropped here

  // ===== SPLIT-EQ DELAYED MODULAR REDUCTION METHOD =====
  let (result_split_eq_dmr, r2, evals2) = if methods.split_eq_dmr {
    let t_setup = Instant::now();
    let (az_vals, bz_vals, cz_vals) = make_field_polys::<F<E>>(&az_i32, &bz_i32);
    let mut az2 = MultilinearPolynomial::new(az_vals);
    let mut bz2 = MultilinearPolynomial::new(bz_vals);
    let mut cz2 = MultilinearPolynomial::new(cz_vals);
    let mut transcript2 = E::TE::new(b"bench");
    let setup_us = t_setup.elapsed().as_micros();

    let t_prove = Instant::now();
    let (_proof2, r2, evals2) =
      SumcheckProof::<E>::prove_cubic_with_three_inputs_split_eq_delayed::<i64>(
        &claim,
        taus.clone(),
        &mut az2,
        &mut bz2,
        &mut cz2,
        &mut transcript2,
      )
      .unwrap();
    let prove_us = t_prove.elapsed().as_micros();
    info!(
      setup_us,
      prove_us, "prove_cubic_with_three_inputs_split_eq_delayed"
    );
    (
      Some(BenchResult { setup_us, prove_us }),
      Some(r2),
      Some(evals2),
    )
  } else {
    (None, None, None)
  };

  // Note: i32 benchmark is handled separately in run_i32_benchmark() for fields that support it

  // ===== SMALL-VALUE METHOD (i64/i128) =====
  let (result_i64, r3, evals3) = if methods.i64_small {
    let t_setup = Instant::now();
    // Create i64 small-value polynomials
    let az_i64: Vec<i64> = az_i32.iter().map(|&v| v as i64).collect();
    let bz_i64: Vec<i64> = bz_i32.iter().map(|&v| v as i64).collect();
    let az_small_i64 = MultilinearPolynomial::new(az_i64);
    let bz_small_i64 = MultilinearPolynomial::new(bz_i64);

    // Need fresh field-element polynomials for binding in later rounds
    let (az_vals, bz_vals, cz_vals) = make_field_polys::<F<E>>(&az_i32, &bz_i32);
    let mut az3 = MultilinearPolynomial::new(az_vals);
    let mut bz3 = MultilinearPolynomial::new(bz_vals);
    let mut cz3 = MultilinearPolynomial::new(cz_vals);
    let mut transcript3 = E::TE::new(b"bench");
    let setup_us = t_setup.elapsed().as_micros();

    let t_prove = Instant::now();
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
    let prove_us = t_prove.elapsed().as_micros();
    info!(
      setup_us,
      prove_us, "prove_cubic_with_three_inputs_small_value (i64/i128)"
    );
    (
      Some(BenchResult { setup_us, prove_us }),
      Some(r3),
      Some(evals3),
    )
  } else {
    (None, None, None)
  };

  // Verify split_eq_dmr matches base (only when both methods were run)
  if let (Some(r1), Some(r2)) = (&r1, &r2) {
    assert_eq!(r1, r2, "split_eq_dmr challenges must match base");
  }
  if let (Some(e1), Some(e2)) = (&evals1, &evals2) {
    assert_eq!(e1, e2, "split_eq_dmr final evaluations must match base");
  }

  // Verify i64 matches base (only when both methods were run)
  if let (Some(r1), Some(r3)) = (&r1, &r3) {
    assert_eq!(r1, r3, "i64 challenges must match");
  }
  if let (Some(e1), Some(e3)) = (&evals1, &evals3) {
    assert_eq!(e1, e3, "i64 final evaluations must match");
  }

  (result_base, result_split_eq_dmr, result_i64)
}

/// Run i32 benchmark separately (only for fields that support SmallValueField<i32>)
fn run_i32_benchmark<E>(num_vars: usize, az_i32: &[i32], bz_i32: &[i32]) -> Option<BenchResult>
where
  E: Engine,
  E::Scalar: SmallValueField<i32, IntermediateSmallValue = i64> + DelayedReduction<i32>,
{
  type F<E> = <E as Engine>::Scalar;

  let taus: Vec<F<E>> = (0..num_vars)
    .map(|i| F::<E>::from((i + 2) as u64))
    .collect();
  let claim: F<E> = F::<E>::from(0u64);

  let t_setup = Instant::now();
  let az_small = MultilinearPolynomial::new(az_i32.to_vec());
  let bz_small = MultilinearPolynomial::new(bz_i32.to_vec());

  let (az_vals, bz_vals, cz_vals) = make_field_polys::<F<E>>(az_i32, bz_i32);
  let mut az2 = MultilinearPolynomial::new(az_vals);
  let mut bz2 = MultilinearPolynomial::new(bz_vals);
  let mut cz2 = MultilinearPolynomial::new(cz_vals);
  let mut transcript2 = E::TE::new(b"bench");
  let setup_us = t_setup.elapsed().as_micros();

  let t_prove = Instant::now();
  let (_proof2, _r2, _evals2) = SumcheckProof::<E>::prove_cubic_with_three_inputs_small_value(
    &claim,
    taus,
    &az_small,
    &bz_small,
    &mut az2,
    &mut bz2,
    &mut cz2,
    &mut transcript2,
  )
  .unwrap();
  let prove_us = t_prove.elapsed().as_micros();
  info!(
    setup_us,
    prove_us, "prove_cubic_with_three_inputs_small_value (i32/i64)"
  );
  Some(BenchResult { setup_us, prove_us })
}

/// Build dynamic CSV header based on selected methods
fn build_csv_header(methods: &BenchMethods) -> String {
  let mut cols = vec!["num_vars", "n", "trial"];
  if methods.base {
    cols.push("base_setup_us");
    cols.push("base_prove_us");
  }
  if methods.split_eq_dmr {
    cols.push("split_eq_dmr_setup_us");
    cols.push("split_eq_dmr_prove_us");
  }
  if methods.i32_small {
    cols.push("i32_setup_us");
    cols.push("i32_prove_us");
  }
  if methods.i64_small {
    cols.push("i64_setup_us");
    cols.push("i64_prove_us");
  }
  // Add speedup columns only when comparing base with others (based on prove time)
  if methods.base && methods.split_eq_dmr {
    cols.push("prove_speedup_split_eq_dmr");
  }
  if methods.base && methods.i32_small {
    cols.push("prove_speedup_i32");
  }
  if methods.base && methods.i64_small {
    cols.push("prove_speedup_i64");
  }
  cols.join(",")
}

/// Format a trial result as a CSV row
fn format_csv_row(result: &TrialResult, methods: &BenchMethods) -> String {
  let mut vals: Vec<String> = vec![
    result.num_vars.to_string(),
    result.n.to_string(),
    result.trial.to_string(),
  ];

  if methods.base {
    let r = result.base.unwrap();
    vals.push(r.setup_us.to_string());
    vals.push(r.prove_us.to_string());
  }
  if methods.split_eq_dmr {
    let r = result.split_eq_dmr.unwrap();
    vals.push(r.setup_us.to_string());
    vals.push(r.prove_us.to_string());
  }
  if methods.i32_small {
    let r = result.i32_small.unwrap();
    vals.push(r.setup_us.to_string());
    vals.push(r.prove_us.to_string());
  }
  if methods.i64_small {
    let r = result.i64_small.unwrap();
    vals.push(r.setup_us.to_string());
    vals.push(r.prove_us.to_string());
  }

  // Add speedup columns only when comparing base with others (based on prove time)
  if methods.base && methods.split_eq_dmr {
    let base_prove = result.base.unwrap().prove_us as f64;
    let dmr_prove = result.split_eq_dmr.unwrap().prove_us as f64;
    let speedup = if dmr_prove > 0.0 {
      base_prove / dmr_prove
    } else {
      f64::INFINITY
    };
    vals.push(format!("{:.3}", speedup));
  }
  if methods.base && methods.i32_small {
    let base_prove = result.base.unwrap().prove_us as f64;
    let i32_prove = result.i32_small.unwrap().prove_us as f64;
    let speedup = if i32_prove > 0.0 {
      base_prove / i32_prove
    } else {
      f64::INFINITY
    };
    vals.push(format!("{:.3}", speedup));
  }
  if methods.base && methods.i64_small {
    let base_prove = result.base.unwrap().prove_us as f64;
    let i64_prove = result.i64_small.unwrap().prove_us as f64;
    let speedup = if i64_prove > 0.0 {
      base_prove / i64_prove
    } else {
      f64::INFINITY
    };
    vals.push(format!("{:.3}", speedup));
  }

  vals.join(",")
}

/// Run sweep with full i32+i64 support (for Pallas/Vesta)
/// Prints each trial result immediately after completion
fn run_sumcheck_sweep_with_i32<E>(
  min_vars: usize,
  max_vars: usize,
  num_trials: usize,
  methods: &BenchMethods,
) where
  E: Engine,
  E::Scalar: SmallValueField<i32, IntermediateSmallValue = i64>
    + SmallValueField<i64, IntermediateSmallValue = i128>
    + DelayedReduction<i32>
    + DelayedReduction<i64>,
{
  println!("{}", build_csv_header(methods));
  std::io::stdout().flush().ok();

  for num_vars in min_vars..=max_vars {
    let n = 1usize << num_vars;

    // Create test data once per num_vars
    let az_i32: Vec<i32> = (0..n).map(|i| (i + 1) as i32).collect();
    let bz_i32: Vec<i32> = (0..n).map(|i| (i + 3) as i32).collect();

    for trial in 1..=num_trials {
      let (base_r, split_eq_dmr_r, i64_r) = run_single_benchmark::<E>(num_vars, methods);
      let i32_r = if methods.i32_small {
        run_i32_benchmark::<E>(num_vars, &az_i32, &bz_i32)
      } else {
        None
      };

      let result = TrialResult {
        num_vars,
        n,
        trial,
        base: base_r,
        split_eq_dmr: split_eq_dmr_r,
        i32_small: i32_r,
        i64_small: i64_r,
      };
      println!("{}", format_csv_row(&result, methods));
      std::io::stdout().flush().ok();
    }
  }
}

/// Run sweep with i64 only (for BN254 which doesn't support i32)
/// Prints each trial result immediately after completion
fn run_sumcheck_sweep_i64_only<E>(
  min_vars: usize,
  max_vars: usize,
  num_trials: usize,
  methods: &BenchMethods,
) where
  E: Engine,
  E::Scalar: SmallValueField<i64, IntermediateSmallValue = i128> + DelayedReduction<i64>,
{
  // Use methods without i32 (will be skipped)
  let methods = methods.without_i32();

  println!("{}", build_csv_header(&methods));
  std::io::stdout().flush().ok();

  for num_vars in min_vars..=max_vars {
    let n = 1usize << num_vars;

    for trial in 1..=num_trials {
      let (base_r, split_eq_dmr_r, i64_r) = run_single_benchmark::<E>(num_vars, &methods);

      let result = TrialResult {
        num_vars,
        n,
        trial,
        base: base_r,
        split_eq_dmr: split_eq_dmr_r,
        i32_small: None,
        i64_small: i64_r,
      };
      println!("{}", format_csv_row(&result, &methods));
      std::io::stdout().flush().ok();
    }
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
  let methods = BenchMethods::from_args(&args.methods);

  let (min_vars, max_vars) = match args.command {
    Some(Command::Single { vars }) => (vars, vars),
    Some(Command::RangeSweep { min, max }) => (min, max),
    None => (10, 24), // Default: sweep 10-24
  };

  eprintln!(
    "Running sumcheck benchmark (field={:?}, min={}, max={}, trials={}, methods={:?})...",
    args.field, min_vars, max_vars, args.trials, args.methods
  );

  // Run benchmarks, printing each trial as it completes
  match args.field {
    FieldChoice::PallasFq => {
      run_sumcheck_sweep_with_i32::<PallasHyraxEngine>(min_vars, max_vars, args.trials, &methods);
    }
    FieldChoice::VestaFp => {
      run_sumcheck_sweep_with_i32::<VestaHyraxEngine>(min_vars, max_vars, args.trials, &methods);
    }
    FieldChoice::Bn254Fr => {
      if methods.i32_small {
        eprintln!("Note: i32 benchmarks not supported for BN254, skipping i32 method");
      }
      run_sumcheck_sweep_i64_only::<Bn254Engine>(min_vars, max_vars, args.trials, &methods);
    }
  }
}
