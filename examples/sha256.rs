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

#[path = "circuits/mod.rs"]
mod circuits;
#[path = "common/mod.rs"]
mod common;

use circuits::SmallSha256Circuit;
use common::spartan_timing_phases::{PHASES, print_table};
use common::timing::{TimingLayer, clear_timings, snapshot_timings};
use spartan2::{
  provider::Bn254Engine,
  spartan::SpartanSNARK,
  traits::{Engine, snark::R1CSSNARKTrait},
};
use std::time::Instant;
use tracing::{info, info_span};
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt, util::SubscriberInitExt};

type E = Bn254Engine;

fn main() {
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

  // Message lengths: 2^10 … 2^11 bytes.
  let circuits: Vec<_> = (10..=11)
    .map(|k| SmallSha256Circuit::<<E as Engine>::Scalar>::new(vec![0u8; 1 << k], true))
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

    let mut small_timings: Vec<u64> = Vec::new();
    let mut large_timings: Vec<u64> = Vec::new();

    for is_small in [true, false] {
      let mode = if is_small { "small" } else { "large" };
      let _mode_span = info_span!("mode", mode).entered();
      info!("--- is_small={} ---", is_small);

      // Clear timing data before prove
      clear_timings(&timing_data);

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
      let timings = snapshot_timings(&timing_data, PHASES);
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
    print_table(&header, &small_timings, &large_timings);

    drop(root_span);
  }
}
