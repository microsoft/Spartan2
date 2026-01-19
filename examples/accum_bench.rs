// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/accum_bench.rs
//!
//! Benchmark for Lagrange accumulator building and l0 sumcheck rounds.
//!
//! Measures:
//! - Accumulator generation time (`build_accumulators_spartan`)
//! - L0 sumcheck rounds time (using precomputed accumulators)
//! - Total time (accum + l0)
//!
//! Compares DelayedModularReductionEnabled vs DelayedModularReductionDisabled.
//!
//! Run with:
//!   cargo run --release --example accum_bench -- single 22
//!   cargo run --release --example accum_bench -- --delayed-modular-reduction both --l0 5 single 22
//!   cargo run --release --example accum_bench -- --show accum-only range-sweep --min 16 --max 24
//!   cargo run --release --example accum_bench -- --show l0-only range-sweep --min 16 --max 24
//!   cargo run --release --example accum_bench -- --show total-only range-sweep --min 16 --max 24

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use clap::{Parser, Subcommand, ValueEnum};
use ff::Field;
use spartan2::{
  lagrange_accumulator::{
    DelayedModularReductionDisabled, DelayedModularReductionEnabled, DelayedModularReductionMode,
    MatVecMLE, SPARTAN_T_DEGREE, build_accumulators_spartan, derive_t1,
  },
  polys::{multilinear::MultilinearPolynomial, univariate::UniPoly},
  provider::{Bn254Engine, PallasHyraxEngine, VestaHyraxEngine},
  small_field::{DelayedReduction, SmallValueField},
  sumcheck::lagrange_sumcheck::SmallValueSumCheck,
  traits::Engine,
};
use std::{io::Write, time::Instant};
use tracing::info;
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

/// Witness type for benchmarks
#[derive(ValueEnum, Clone, Default, Debug)]
enum WitnessType {
  /// i32 witness coefficients
  I32,
  /// i64 witness coefficients
  #[default]
  I64,
  /// Full field element coefficients
  Field,
}

/// Delayed modular reduction mode selection for benchmarks
#[derive(ValueEnum, Clone, Default, Debug)]
enum DelayedModularReductionChoice {
  /// Delayed modular reduction enabled
  Enabled,
  /// Delayed modular reduction disabled (immediate reduction)
  Disabled,
  /// Run both and compare
  #[default]
  Both,
}

/// Which timings to show in output
#[derive(ValueEnum, Clone, Default, Debug)]
enum ShowTimings {
  /// Show only accumulator build time
  AccumOnly,
  /// Show only l0 sumcheck rounds time
  L0Only,
  /// Show only total time (accum + l0)
  TotalOnly,
  /// Show all timings (accum, l0, and total)
  #[default]
  All,
}

#[derive(Parser)]
#[command(about = "Accumulator benchmark with delayed modular reduction toggle")]
struct Args {
  /// Field to use for benchmarks
  #[arg(long, value_enum, default_value = "pallas-fq")]
  field: FieldChoice,

  /// Witness type (i32, i64, or field)
  #[arg(long, value_enum, default_value = "i64")]
  witness: WitnessType,

  /// Number of small-value rounds (default: num_vars / 4)
  #[arg(long)]
  l0: Option<usize>,

  /// Delayed modular reduction mode (enabled, disabled, or both)
  #[arg(long, value_enum, default_value = "both")]
  delayed_modular_reduction: DelayedModularReductionChoice,

  /// Which timings to show (accum-only, l0-only, total-only, or all)
  #[arg(long, value_enum, default_value = "all")]
  show: ShowTimings,

  /// Number of trials per num_vars
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
    #[arg(long, default_value = "16")]
    min: usize,
    #[arg(long, default_value = "24")]
    max: usize,
  },
}

/// Benchmark result for one configuration
#[derive(Clone, Copy, Default)]
struct BenchResult {
  accum_us: u128,
  l0_rounds_us: u128,
}

impl BenchResult {
  fn total_us(&self) -> u128 {
    self.accum_us + self.l0_rounds_us
  }
}

/// Result of a single trial
struct TrialResult {
  num_vars: usize,
  n: usize,
  l0: usize,
  trial: usize,
  delayed_modular_reduction_enabled: Option<BenchResult>,
  delayed_modular_reduction_disabled: Option<BenchResult>,
}

fn build_csv_header(
  delayed_modular_reduction: &DelayedModularReductionChoice,
  show: &ShowTimings,
) -> String {
  let mut cols = vec!["num_vars", "n", "l0", "trial"];

  match delayed_modular_reduction {
    DelayedModularReductionChoice::Enabled => match show {
      ShowTimings::AccumOnly => cols.push("dmr_accum_us"),
      ShowTimings::L0Only => cols.push("dmr_l0_us"),
      ShowTimings::TotalOnly => cols.push("dmr_total_us"),
      ShowTimings::All => cols.extend(["dmr_accum_us", "dmr_l0_us", "dmr_total_us"]),
    },
    DelayedModularReductionChoice::Disabled => match show {
      ShowTimings::AccumOnly => cols.push("no_dmr_accum_us"),
      ShowTimings::L0Only => cols.push("no_dmr_l0_us"),
      ShowTimings::TotalOnly => cols.push("no_dmr_total_us"),
      ShowTimings::All => cols.extend(["no_dmr_accum_us", "no_dmr_l0_us", "no_dmr_total_us"]),
    },
    DelayedModularReductionChoice::Both => match show {
      ShowTimings::AccumOnly => {
        cols.extend(["dmr_accum_us", "no_dmr_accum_us", "accum_speedup"]);
      }
      ShowTimings::L0Only => {
        cols.extend(["dmr_l0_us", "no_dmr_l0_us", "l0_speedup"]);
      }
      ShowTimings::TotalOnly => {
        cols.extend(["dmr_total_us", "no_dmr_total_us", "total_speedup"]);
      }
      ShowTimings::All => {
        cols.extend([
          "dmr_accum_us",
          "dmr_l0_us",
          "dmr_total_us",
          "no_dmr_accum_us",
          "no_dmr_l0_us",
          "no_dmr_total_us",
          "accum_speedup",
          "l0_speedup",
          "total_speedup",
        ]);
      }
    },
  }
  cols.join(",")
}

fn format_csv_row(
  result: &TrialResult,
  delayed_modular_reduction: &DelayedModularReductionChoice,
  show: &ShowTimings,
) -> String {
  let mut row = format!(
    "{},{},{},{}",
    result.num_vars, result.n, result.l0, result.trial
  );

  match delayed_modular_reduction {
    DelayedModularReductionChoice::Enabled => {
      if let Some(r) = result.delayed_modular_reduction_enabled {
        match show {
          ShowTimings::AccumOnly => row.push_str(&format!(",{}", r.accum_us)),
          ShowTimings::L0Only => row.push_str(&format!(",{}", r.l0_rounds_us)),
          ShowTimings::TotalOnly => row.push_str(&format!(",{}", r.total_us())),
          ShowTimings::All => row.push_str(&format!(
            ",{},{},{}",
            r.accum_us,
            r.l0_rounds_us,
            r.total_us()
          )),
        }
      }
    }
    DelayedModularReductionChoice::Disabled => {
      if let Some(r) = result.delayed_modular_reduction_disabled {
        match show {
          ShowTimings::AccumOnly => row.push_str(&format!(",{}", r.accum_us)),
          ShowTimings::L0Only => row.push_str(&format!(",{}", r.l0_rounds_us)),
          ShowTimings::TotalOnly => row.push_str(&format!(",{}", r.total_us())),
          ShowTimings::All => row.push_str(&format!(
            ",{},{},{}",
            r.accum_us,
            r.l0_rounds_us,
            r.total_us()
          )),
        }
      }
    }
    DelayedModularReductionChoice::Both => {
      if let (Some(enabled_r), Some(disabled_r)) = (
        result.delayed_modular_reduction_enabled,
        result.delayed_modular_reduction_disabled,
      ) {
        match show {
          ShowTimings::AccumOnly => {
            let speedup = disabled_r.accum_us as f64 / enabled_r.accum_us as f64;
            row.push_str(&format!(
              ",{},{},{:.2}",
              enabled_r.accum_us, disabled_r.accum_us, speedup
            ));
          }
          ShowTimings::L0Only => {
            let speedup = disabled_r.l0_rounds_us as f64 / enabled_r.l0_rounds_us as f64;
            row.push_str(&format!(
              ",{},{},{:.2}",
              enabled_r.l0_rounds_us, disabled_r.l0_rounds_us, speedup
            ));
          }
          ShowTimings::TotalOnly => {
            let speedup = disabled_r.total_us() as f64 / enabled_r.total_us() as f64;
            row.push_str(&format!(
              ",{},{},{:.2}",
              enabled_r.total_us(),
              disabled_r.total_us(),
              speedup
            ));
          }
          ShowTimings::All => {
            let accum_speedup = disabled_r.accum_us as f64 / enabled_r.accum_us as f64;
            let l0_speedup = disabled_r.l0_rounds_us as f64 / enabled_r.l0_rounds_us as f64;
            let total_speedup = disabled_r.total_us() as f64 / enabled_r.total_us() as f64;
            row.push_str(&format!(
              ",{},{},{},{},{},{},{:.2},{:.2},{:.2}",
              enabled_r.accum_us,
              enabled_r.l0_rounds_us,
              enabled_r.total_us(),
              disabled_r.accum_us,
              disabled_r.l0_rounds_us,
              disabled_r.total_us(),
              accum_speedup,
              l0_speedup,
              total_speedup
            ));
          }
        }
      }
    }
  }
  row
}

/// Generate satisfying witness polynomials (Az * Bz = Cz on boolean hypercube)
fn generate_witness_i64<F: ff::PrimeField + SmallValueField<i64>>(
  num_vars: usize,
) -> (
  MultilinearPolynomial<i64>,
  MultilinearPolynomial<i64>,
  Vec<F>,
) {
  let n = 1usize << num_vars;
  let az_vals: Vec<i64> = (0..n).map(|i| ((i % 1000) + 1) as i64).collect();
  let bz_vals: Vec<i64> = (0..n).map(|i| (((i * 7) % 1000) + 1) as i64).collect();
  let taus: Vec<F> = (0..num_vars).map(|i| F::from((i * 7 + 3) as u64)).collect();

  (
    MultilinearPolynomial::new(az_vals),
    MultilinearPolynomial::new(bz_vals),
    taus,
  )
}

/// Run accumulator benchmark with specified delayed modular reduction mode
fn run_accumulator_bench<E, P, Mode>(az: &P, bz: &P, taus: &[E::Scalar], l0: usize) -> BenchResult
where
  E: Engine,
  E::Scalar: SmallValueField<i64, IntermediateSmallValue = i128> + DelayedReduction<i64>,
  P: MatVecMLE<E::Scalar>,
  Mode: DelayedModularReductionMode<E::Scalar, P, SPARTAN_T_DEGREE>,
{
  // Phase 1: Measure accumulator build time
  let start = Instant::now();
  let accumulators = build_accumulators_spartan::<_, _, Mode>(az, bz, taus, l0);
  let accum_us = start.elapsed().as_micros();

  // Phase 2: Measure l0 sumcheck rounds time
  let mut small_value =
    SmallValueSumCheck::<E::Scalar, SPARTAN_T_DEGREE>::from_accumulators(accumulators);

  // Initial claim = 0 for satisfying witness (Az * Bz - Cz = 0)
  let mut claim_per_round = E::Scalar::ZERO;

  let start = Instant::now();

  for round in 0..l0 {
    // 1. Get t evaluations from accumulators
    let t_all = small_value.eval_t_all_u(round);
    let t_inf = t_all.at_infinity();
    let t0 = t_all.at_zero();

    // 2. Get eq round values
    let li = small_value.eq_round_values(taus[round]);

    // 3. Derive t1 from sumcheck constraint
    let t1 = derive_t1(li.at_zero(), li.at_one(), claim_per_round, t0)
      .expect("l1 non-zero for valid witness");

    // 4. Build univariate polynomial
    // s_i(X) = ℓ_i(X) * t_i(X) where ℓ_i(X) = ℓ_∞·X + ℓ_0 and t_i(X) is degree-2
    // We compute evaluations at 0, 1, 2, 3 and interpolate
    let l0_val = li.at_zero();
    let linf = li.at_infinity();

    // t_i(X) = t_inf * X^2 + b*X + t0 where b = t1 - t_inf - t0
    let b = t1 - t_inf - t0;

    // Evaluate s(X) = ℓ(X) * t(X) at X = 0, 1, 2, 3
    // ℓ(X) = linf * X + l0_val
    // t(X) = t_inf * X^2 + b * X + t0
    let eval_s = |x: u64| -> E::Scalar {
      let x_f = E::Scalar::from(x);
      let l_x = linf * x_f + l0_val;
      let t_x = t_inf * x_f * x_f + b * x_f + t0;
      l_x * t_x
    };

    let evals = vec![eval_s(0), eval_s(1), eval_s(2), eval_s(3)];
    let poly = UniPoly::from_evals(&evals).expect("valid polynomial");

    // 5. Simulate verifier challenge (deterministic for benchmark)
    let r_i = E::Scalar::from((round + 7) as u64);

    // 6. Advance to next round
    small_value.advance(&li, r_i);
    claim_per_round = poly.evaluate(&r_i);
  }

  let l0_rounds_us = start.elapsed().as_micros();

  BenchResult {
    accum_us,
    l0_rounds_us,
  }
}

/// Run benchmark for a single num_vars configuration
fn run_single<E>(
  num_vars: usize,
  l0: usize,
  delayed_modular_reduction: &DelayedModularReductionChoice,
) -> TrialResult
where
  E: Engine,
  E::Scalar: SmallValueField<i64, IntermediateSmallValue = i128> + DelayedReduction<i64>,
{
  let (az, bz, taus) = generate_witness_i64::<E::Scalar>(num_vars);
  let n = 1usize << num_vars;

  let delayed_modular_reduction_enabled = match delayed_modular_reduction {
    DelayedModularReductionChoice::Enabled | DelayedModularReductionChoice::Both => {
      Some(run_accumulator_bench::<
        E,
        _,
        DelayedModularReductionEnabled<i64>,
      >(&az, &bz, &taus, l0))
    }
    DelayedModularReductionChoice::Disabled => None,
  };

  let delayed_modular_reduction_disabled = match delayed_modular_reduction {
    DelayedModularReductionChoice::Disabled | DelayedModularReductionChoice::Both => {
      Some(run_accumulator_bench::<E, _, DelayedModularReductionDisabled>(&az, &bz, &taus, l0))
    }
    DelayedModularReductionChoice::Enabled => None,
  };

  TrialResult {
    num_vars,
    n,
    l0,
    trial: 0,
    delayed_modular_reduction_enabled,
    delayed_modular_reduction_disabled,
  }
}

fn run_benchmark<E>(args: &Args)
where
  E: Engine,
  E::Scalar: SmallValueField<i64, IntermediateSmallValue = i128> + DelayedReduction<i64>,
{
  let (min_vars, max_vars) = match &args.command {
    Some(Command::Single { vars }) => (*vars, *vars),
    Some(Command::RangeSweep { min, max }) => (*min, *max),
    None => (16, 24),
  };

  // Print CSV header
  println!(
    "{}",
    build_csv_header(&args.delayed_modular_reduction, &args.show)
  );

  for num_vars in min_vars..=max_vars {
    let l0 = args.l0.unwrap_or(num_vars / 4).max(1).min(num_vars - 1);

    // Warmup
    if num_vars == min_vars {
      info!("Warmup run for num_vars={}", num_vars);
      let _ = run_single::<E>(num_vars, l0, &args.delayed_modular_reduction);
    }

    for trial in 0..args.trials {
      let mut result = run_single::<E>(num_vars, l0, &args.delayed_modular_reduction);
      result.trial = trial;
      println!(
        "{}",
        format_csv_row(&result, &args.delayed_modular_reduction, &args.show)
      );
      std::io::stdout().flush().unwrap();

      // Log progress
      if let (Some(enabled), Some(disabled)) = (
        result.delayed_modular_reduction_enabled,
        result.delayed_modular_reduction_disabled,
      ) {
        let accum_speedup = disabled.accum_us as f64 / enabled.accum_us as f64;
        let l0_speedup = disabled.l0_rounds_us as f64 / enabled.l0_rounds_us as f64;
        let total_speedup = disabled.total_us() as f64 / enabled.total_us() as f64;
        info!(
          "num_vars={} l0={}: accum({:.0}μs vs {:.0}μs, {:.2}x), l0({:.0}μs vs {:.0}μs, {:.2}x), total({:.0}μs vs {:.0}μs, {:.2}x)",
          num_vars,
          l0,
          enabled.accum_us,
          disabled.accum_us,
          accum_speedup,
          enabled.l0_rounds_us,
          disabled.l0_rounds_us,
          l0_speedup,
          enabled.total_us(),
          disabled.total_us(),
          total_speedup
        );
      }
    }
  }
}

fn main() {
  // Initialize tracing
  tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::from_default_env())
    .with_target(false)
    .init();

  let args = Args::parse();

  info!("Accumulator benchmark");
  info!(
    "Field: {:?}, Witness: {:?}, delayed_modular_reduction: {:?}, show: {:?}",
    args.field, args.witness, args.delayed_modular_reduction, args.show
  );

  match args.field {
    FieldChoice::PallasFq => run_benchmark::<PallasHyraxEngine>(&args),
    FieldChoice::VestaFp => run_benchmark::<VestaHyraxEngine>(&args),
    FieldChoice::Bn254Fr => run_benchmark::<Bn254Engine>(&args),
  }
}
