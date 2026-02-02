// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/sha256_chain_benchmark.rs
//! Benchmark SHA-256 hash chains comparing:
//! - Original sumcheck (baseline)
//! - Small-value sumcheck with BatchingEq<21>
//!
//! Run with: `RUST_LOG=info cargo run --release --no-default-features --example sha256_chain_benchmark`
//! Or for CSV only: `cargo run --release --no-default-features --example sha256_chain_benchmark 2>/dev/null`
//!
//! CLI modes:
//!   spartan 20                              - Full Spartan prove breakdown, small vs large
//!   single-sumcheck 26                      - Single sumcheck-only run (for profiling)
//!   range-sumcheck-sweep --min 16 --max 20  - Sumcheck-only CSV sweep

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[path = "circuits/mod.rs"]
mod circuits;
mod spartan_timing_phases;
mod timing;

use circuits::SmallSha256ChainCircuit;
use clap::{Parser, Subcommand};
use ff::Field;
use spartan_timing_phases::{PHASES, print_table};
use spartan2::{
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  provider::Bn254Engine,
  small_field::SmallValueField,
  spartan::SpartanSNARK,
  sumcheck::SumcheckProof,
  traits::{Engine, snark::R1CSSNARKTrait, transcript::TranscriptEngineTrait},
};
use std::{io::Write, time::Instant};
use timing::{TimingLayer, clear_timings, snapshot_timings};
use tracing::{info, info_span};
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt, util::SubscriberInitExt};

// Use PallasHyraxEngine which has Barrett-optimized SmallLargeMul for Fq
type E = Bn254Engine;
type F = <E as Engine>::Scalar;

#[derive(Parser)]
#[command(about = "SHA-256 chain benchmark: original vs small-value sumcheck")]
struct Args {
  #[command(subcommand)]
  command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
  /// Full Spartan prove breakdown with small vs large comparison table
  Spartan { num_vars: usize },
  /// Run a single sumcheck-only benchmark (for profiling)
  SingleSumcheck { num_vars: usize },
  /// Sumcheck-only CSV sweep across a range of num_vars
  RangeSumcheckSweep {
    #[arg(long, default_value = "16")]
    min: usize,
    #[arg(long, default_value = "26")]
    max: usize,
  },
}

/// Convert num_vars to chain_length.
/// num_vars=16 → chain=2, num_vars=18 → chain=8, etc.
/// Formula: chain_length = 2^(num_vars - 15)
fn num_vars_to_chain_length(num_vars: usize) -> usize {
  1 << (num_vars - 15)
}

// ─── Spartan full-prove benchmark ───

fn run_spartan_benchmark(
  input: [u8; 32],
  num_vars: usize,
  timing_data: &timing::TimingData,
  constraints_data: &timing::ConstraintsData,
) {
  let chain_length = num_vars_to_chain_length(num_vars);
  let small_circuit = SmallSha256ChainCircuit::<F>::new(input, chain_length);

  let root_span = info_span!("bench", num_vars, chain_length).entered();
  info!(
    "======= num_vars={}, chain_length={} =======",
    num_vars, chain_length
  );

  // SETUP (once per circuit)
  let t0 = Instant::now();
  let (pk, vk) = SpartanSNARK::<E>::setup(small_circuit.clone()).expect("setup failed");
  let setup_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = setup_ms, "setup");

  let mut small_timings: Vec<u64> = Vec::new();
  let mut large_timings: Vec<u64> = Vec::new();

  for is_small in [true, false] {
    let mode = if is_small { "small" } else { "large" };
    let _mode_span = info_span!("mode", mode).entered();
    info!("--- is_small={} ---", is_small);

    clear_timings(timing_data);

    // PREPARE
    let t0 = Instant::now();
    let prep_snark = SpartanSNARK::<E>::prep_prove(&pk, small_circuit.clone(), is_small)
      .expect("prep_prove failed");
    let prep_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prep_ms, "prep_prove");

    // PROVE
    let t0 = Instant::now();
    let proof = SpartanSNARK::<E>::prove(&pk, small_circuit.clone(), &prep_snark, is_small)
      .expect("prove failed");
    let prove_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prove_ms, "prove");

    let timings = snapshot_timings(timing_data, PHASES);
    if is_small {
      small_timings = timings;
    } else {
      large_timings = timings;
    }

    // VERIFY
    let t0 = Instant::now();
    proof.verify(&vk).expect("verify failed");
    let verify_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = verify_ms, "verify");

    info!(
      "SUMMARY num_vars={}, chain={}, is_small={}, setup={} ms, prep={} ms, prove={} ms, verify={} ms",
      num_vars, chain_length, is_small, setup_ms, prep_ms, prove_ms, verify_ms
    );
  }

  let constraints = constraints_data.lock().unwrap().take();
  let header = match constraints {
    Some(c) => format!(
      "===== chain={}, num_vars={}, constraints={} =====",
      chain_length, num_vars, c
    ),
    None => format!("===== chain={}, num_vars={} =====", chain_length, num_vars),
  };
  print_table(&header, &small_timings, &large_timings);

  drop(root_span);
}

// ─── Sumcheck-only benchmark (existing logic) ───

struct BenchmarkResult {
  chain_length: usize,
  num_vars: usize,
  num_constraints: usize,
  witness_ms: u128,
  extract_ms: u128,
  orig_sumcheck_ms: u128,
  small_sumcheck_ms: u128,
}

fn run_chain_benchmark(
  input: [u8; 32],
  chain_length: usize,
  expected_num_vars: usize,
) -> BenchmarkResult
where
  F: SmallValueField<i64, IntermediateSmallValue = i128>,
{
  let small_circuit = SmallSha256ChainCircuit::<F>::new(input, chain_length);

  let t0 = Instant::now();
  let (pk, _vk) = SpartanSNARK::<E>::setup(small_circuit.clone()).expect("setup failed");
  let setup_ms = t0.elapsed().as_millis();
  let num_constraints = pk.sizes()[4];
  info!(setup_ms, num_constraints, "setup");

  let t0 = Instant::now();
  let prep_snark =
    SpartanSNARK::<E>::prep_prove(&pk, small_circuit.clone(), true).expect("prep_prove failed");
  let witness_ms = t0.elapsed().as_millis();
  info!(witness_ms, "witness synthesis");

  let t0 = Instant::now();
  let (az, bz, cz, tau) =
    SpartanSNARK::<E>::extract_outer_sumcheck_inputs(&pk, small_circuit, &prep_snark)
      .expect("extract_outer_sumcheck_inputs failed");
  let extract_ms = t0.elapsed().as_millis();
  info!(extract_ms, "extract inputs");

  let num_vars = tau.len();

  assert_eq!(
    num_vars, expected_num_vars,
    "Expected num_vars={} but got {}. Adjust chain_length.",
    expected_num_vars, num_vars
  );

  let claim = F::ZERO;
  let tau_for_verify = tau.clone();

  // ===== ORIGINAL SUMCHECK =====
  let (proof1, r1, evals1, orig_sumcheck_ms) = {
    let mut az1 = MultilinearPolynomial::new(az.clone());
    let mut bz1 = MultilinearPolynomial::new(bz.clone());
    let mut cz1 = MultilinearPolynomial::new(cz);
    let mut transcript1 = <E as Engine>::TE::new(b"sha256_chain_bench");

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
    let elapsed = t0.elapsed().as_millis();
    info!(orig_sumcheck_ms = elapsed, "original sumcheck");
    (proof1, r1, evals1, elapsed)
  };

  // ===== SMALL-VALUE SUMCHECK =====
  let (proof2, r2, evals2, small_sumcheck_ms) = {
    let az_small_vals: Vec<i64> = az
      .iter()
      .map(|v| <F as SmallValueField<i64>>::try_field_to_small(v).expect("Az too large for i64"))
      .collect();
    let bz_small_vals: Vec<i64> = bz
      .iter()
      .map(|v| <F as SmallValueField<i64>>::try_field_to_small(v).expect("Bz too large for i64"))
      .collect();
    let cz_small_vals: Vec<i64> = az_small_vals
      .iter()
      .zip(bz_small_vals.iter())
      .map(|(&a, &b)| a * b)
      .collect();
    let az_small = MultilinearPolynomial::new(az_small_vals);
    let bz_small = MultilinearPolynomial::new(bz_small_vals);
    let cz_small = MultilinearPolynomial::new(cz_small_vals);

    let mut transcript2 = <E as Engine>::TE::new(b"sha256_chain_bench");

    let t0 = Instant::now();
    let (proof2, r2, evals2) = SumcheckProof::<E>::prove_cubic_with_three_inputs_small_value(
      &claim,
      tau,
      &az_small,
      &bz_small,
      &cz_small,
      &mut transcript2,
    )
    .expect("prove_cubic_with_three_inputs_small_value failed");
    let elapsed = t0.elapsed().as_millis();
    info!(small_sumcheck_ms = elapsed, "small-value sumcheck");
    (proof2, r2, evals2, elapsed)
  };

  assert_eq!(r1, r2, "Challenges must match!");
  assert_eq!(proof1, proof2, "Round polynomials must match!");
  assert_eq!(evals1, evals2, "Final evaluations must match!");

  // ===== VERIFY SUMCHECK PROOF =====
  {
    let mut transcript_v = <E as Engine>::TE::new(b"sha256_chain_bench");
    let (final_claim, r_v) = proof1
      .verify(claim, num_vars, 3, &mut transcript_v)
      .expect("sumcheck verify failed");
    assert_eq!(r_v, r1, "Verify challenges must match prover challenges");
    // Check: final_claim == tau(r_x) * (Az(r_x)*Bz(r_x) - Cz(r_x))
    let tau_eval = EqPolynomial::new(tau_for_verify.clone()).evaluate(&r_v);
    let expected = tau_eval * (evals1[0] * evals1[1] - evals1[2]);
    assert_eq!(
      final_claim, expected,
      "Sumcheck final claim must match expected"
    );
    info!("sumcheck verification passed");
  }

  BenchmarkResult {
    chain_length,
    num_vars,
    num_constraints,
    witness_ms,
    extract_ms,
    orig_sumcheck_ms,
    small_sumcheck_ms,
  }
}

fn print_csv_header() {
  println!(
    "chain_length,num_vars,log2_constraints,num_constraints,witness_ms,orig_sumcheck_ms,small_sumcheck_ms,total_proving_ms,speedup,witness_pct"
  );
}

fn print_csv_row(result: &BenchmarkResult) {
  let speedup = result.orig_sumcheck_ms as f64 / result.small_sumcheck_ms as f64;
  let total_ms = result.witness_ms + result.extract_ms + result.small_sumcheck_ms;
  let witness_pct = (result.witness_ms as f64 / total_ms as f64) * 100.0;
  let log2_constraints = (result.num_constraints as f64).log2();

  println!(
    "{},{},{:.3},{},{},{},{},{},{:.2},{:.1}",
    result.chain_length,
    result.num_vars,
    log2_constraints,
    result.num_constraints,
    result.witness_ms,
    result.orig_sumcheck_ms,
    result.small_sumcheck_ms,
    total_ms,
    speedup,
    witness_pct
  );
  std::io::stdout().flush().ok();
}

fn init_simple_tracing() {
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .with_writer(std::io::stderr)
    .init();
}

fn main() {
  let args = Args::parse();

  // Use a deterministic input
  let input: [u8; 32] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
  ];

  match args.command {
    Some(Command::Spartan { num_vars }) => {
      let (timing_layer, timing_data, constraints_data) = TimingLayer::new();

      tracing_subscriber::registry()
        .with(timing_layer)
        .with(
          tracing_subscriber::fmt::layer()
            .with_target(false)
            .with_ansi(true)
            .with_writer(std::io::stderr)
            .with_filter(EnvFilter::from_default_env()),
        )
        .init();

      run_spartan_benchmark(input, num_vars, &timing_data, &constraints_data);
    }
    Some(Command::SingleSumcheck { num_vars }) => {
      init_simple_tracing();
      let chain_length = num_vars_to_chain_length(num_vars);
      print_csv_header();
      let result = run_chain_benchmark(input, chain_length, num_vars);
      print_csv_row(&result);
    }
    Some(Command::RangeSumcheckSweep { min, max }) => {
      init_simple_tracing();
      print_csv_header();
      for num_vars in min..=max {
        let chain_length = num_vars_to_chain_length(num_vars);
        let result = run_chain_benchmark(input, chain_length, num_vars);
        print_csv_row(&result);
      }
    }
    None => {
      init_simple_tracing();
      print_csv_header();
      for num_vars in 16..=26 {
        let chain_length = num_vars_to_chain_length(num_vars);
        let result = run_chain_benchmark(input, chain_length, num_vars);
        print_csv_row(&result);
      }
    }
  }
}
