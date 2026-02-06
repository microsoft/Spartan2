// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/sha256.rs
//! Measure Spartan-2 {setup, prove, verify} times for a SHA-256
//! circuit with varying message lengths
//!
//! Run with: `RUST_LOG=info cargo run --release --example sha256`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use clap::Parser;
use spartan2::{
  cli::FieldChoice,
  provider::{Bn254Engine, PallasHyraxEngine, VestaHyraxEngine},
  sha256_circuits::SmallSha256Circuit,
  small_field::{DelayedReduction, SmallValueField},
  spartan::SpartanSNARK,
  timing::{ConstraintsData, SPARTAN_PHASES, TimingData, TimingLayer, clear_timings, print_table, snapshot_timings},
  traits::{Engine, snark::R1CSSNARKTrait},
};
use std::{collections::HashMap, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(about = "SHA-256 Spartan benchmark")]
struct Args {
  #[arg(long, value_enum, default_value = "bn254-fr")]
  field: FieldChoice,
}

fn run_benchmark<E: Engine>(timing_data: &TimingData, constraints_data: &ConstraintsData)
where
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  // Message lengths: 2^10 … 2^11 bytes.
  let circuits: Vec<_> = (10..=11)
    .map(|k| SmallSha256Circuit::<E::Scalar>::new(vec![0u8; 1 << k], true))
    .collect();

  for circuit in circuits {
    let msg_len = circuit.preimage.len();
    let root_span = info_span!("bench", msg_len).entered();
    info!("======= message_len={} bytes =======", msg_len);

    // SETUP (once per circuit)
    let t0 = Instant::now();
    let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup failed");
    let setup_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = setup_ms, "setup");

    let mut small_timings = HashMap::new();
    let mut large_timings = HashMap::new();

    for is_small in [true, false] {
      let mode = if is_small { "small" } else { "large" };
      let _mode_span = info_span!("mode", mode).entered();
      info!("--- is_small={} ---", is_small);

      // Clear timing data before prove
      clear_timings(timing_data);

      // PREPARE
      let t0 = Instant::now();
      let prep_snark =
        SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), is_small).expect("prep_prove failed");
      let prep_ms = t0.elapsed().as_millis();
      info!(elapsed_ms = prep_ms, "prep_prove");

      // PROVE
      let t0 = Instant::now();
      let proof = SpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep_snark, is_small)
        .expect("prove failed");
      let prove_ms = t0.elapsed().as_millis();
      info!(elapsed_ms = prove_ms, "prove");

      // Snapshot timings from prove
      let timings = snapshot_timings(timing_data, SPARTAN_PHASES);
      if is_small {
        small_timings = timings;
      } else {
        large_timings = timings;
      }

      // VERIFY
      let t0 = Instant::now();
      proof.verify(&vk).expect("verify errored");
      let verify_ms = t0.elapsed().as_millis();
      info!(elapsed_ms = verify_ms, "verify");

      info!(
        "SUMMARY msg={}B, is_small={}, setup={} ms, prep_prove={} ms, prove={} ms, verify={} ms",
        msg_len, is_small, setup_ms, prep_ms, prove_ms, verify_ms
      );
    }

    // Print comparison table
    let constraints = constraints_data.lock().unwrap().take();
    let header = match constraints {
      Some(c) => format!("===== msg={}B, constraints={} =====", msg_len, c),
      None => format!("===== msg={}B =====", msg_len),
    };
    print_table(&header, SPARTAN_PHASES, &small_timings, &large_timings);

    drop(root_span);
  }
}

fn main() {
  let args = Args::parse();

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

  match args.field {
    FieldChoice::Bn254Fr => run_benchmark::<Bn254Engine>(&timing_data, &constraints_data),
    FieldChoice::PallasFq => run_benchmark::<PallasHyraxEngine>(&timing_data, &constraints_data),
    FieldChoice::VestaFp => run_benchmark::<VestaHyraxEngine>(&timing_data, &constraints_data),
  }
}
