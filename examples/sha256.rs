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

use circuits::Sha256Circuit;
use spartan2::{
  provider::T256HyraxEngine,
  spartan::SpartanSNARK,
  traits::{Engine, snark::R1CSSNARKTrait},
};
use std::time::Instant;
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

type E = T256HyraxEngine;

fn main() {
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .init();

  // Message lengths: 2^10 … 2^11 bytes.
  let circuits: Vec<_> = (10..=11)
    .map(|k| Sha256Circuit::<<E as Engine>::Scalar>::new(vec![0u8; 1 << k]))
    .collect();

  for circuit in circuits {
    let msg_len = circuit.preimage.len();
    let root_span = info_span!("bench", msg_len).entered();
    info!("======= message_len={} bytes =======", msg_len);

    // SETUP
    let t0 = Instant::now();
    let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup failed");
    let setup_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = setup_ms, "setup");

    // PREPARE
    let t0 = Instant::now();
    let prep_snark =
      SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).expect("prep_prove failed");
    let prep_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prep_ms, "prep_prove");

    // PROVE
    let t0 = Instant::now();
    let proof =
      SpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep_snark, true).expect("prove failed");
    let prove_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prove_ms, "prove");

    // VERIFY
    let t0 = Instant::now();
    proof.verify(&vk).expect("verify errored");
    let verify_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = verify_ms, "verify");

    // Summary
    info!(
      "SUMMARY msg={}B, setup={} ms, prep_prove={} ms, prove={} ms, verify={} ms",
      msg_len, setup_ms, prep_ms, prove_ms, verify_ms
    );
    drop(root_span);
  }
}
