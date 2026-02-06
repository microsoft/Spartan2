// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/neutronnova_sha256_benchmark.rs
//! Benchmark NeutronNova NIFS folding of SHA-256 hash chains comparing:
//! - Small-value NIFS sumcheck
//! - Large-value (vanilla) NIFS sumcheck
//!
//! Run with: `RUST_LOG=info cargo run --release --example neutronnova_sha256_benchmark`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use clap::{Parser, ValueEnum};
use spartan2::{
  bellpepper::{
    r1cs::{MultiRoundSpartanWitness, SpartanWitness},
    solver::SatisfyingAssignment,
  },
  cli::FieldChoice,
  math::Math,
  neutronnova_zk::{
    NeutronNovaNIFS, NeutronNovaPrepZkSNARK, NeutronNovaProverKey, NeutronNovaVerifierKey,
    NeutronNovaZkSNARK,
  },
  provider::{Bn254Engine, PallasHyraxEngine, VestaHyraxEngine},
  r1cs::{R1CSInstance, R1CSWitness},
  sha256_circuits::SmallSha256ChainCircuit,
  small_field::{DelayedReduction, SmallValueField},
  timing::{
    NEUTRONNOVA_PHASES, NEUTRONNOVA_ZK_PROVE_PHASES, TimingData, TimingLayer, clear_timings,
    print_table, snapshot_timings,
  },
  traits::{Engine, circuit::SpartanCircuit, pcs::FoldingEngineTrait, transcript::TranscriptEngineTrait},
  zk::NeutronNovaVerifierCircuit,
};
use std::{collections::HashMap, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt, util::SubscriberInitExt};

/// Benchmark mode
#[derive(ValueEnum, Clone, Default, Debug)]
enum BenchMode {
  /// Benchmark NIFS folding only
  #[default]
  Nifs,
  /// Benchmark full NeutronNovaZkSNARK::prove
  ZkProve,
}

#[derive(Parser)]
#[command(about = "NeutronNova benchmark: small vs large sumcheck")]
struct Args {
  #[arg(long, default_value = "4")]
  instances: usize,
  #[arg(long, default_value = "4")]
  chain_length: usize,
  #[arg(long, value_enum, default_value = "bn254-fr")]
  field: FieldChoice,
  #[arg(long, value_enum, default_value = "nifs")]
  mode: BenchMode,
}

/// Generate step circuits with distinct inputs
fn make_circuits<F: ff::PrimeField + ff::PrimeFieldBits>(
  num_instances: usize,
  chain_length: usize,
) -> Vec<SmallSha256ChainCircuit<F>> {
  (0..num_instances)
    .map(|i| {
      let mut input = [0u8; 32];
      input[0] = i as u8;
      input[1] = (i >> 8) as u8;
      SmallSha256ChainCircuit::new(input, chain_length)
    })
    .collect()
}

/// Generate instances and witnesses from circuits using the NeutronNova pipeline.
fn generate_instances_and_witnesses<E, C>(
  pk: &NeutronNovaProverKey<E>,
  prep: &NeutronNovaPrepZkSNARK<E>,
  step_circuits: &[C],
  is_small: bool,
) -> (Vec<R1CSInstance<E>>, Vec<R1CSWitness<E>>)
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  C: SpartanCircuit<E>,
{
  let mut ps_step: Vec<_> = prep.ps_step.clone();

  let (instances, witnesses): (Vec<_>, Vec<_>) = ps_step
    .iter_mut()
    .zip(step_circuits.iter().enumerate())
    .map(|(pre_state, (i, circuit))| {
      let mut transcript = E::TE::new(b"neutronnova_prove");
      transcript.absorb(b"vk", &pk.vk_digest);
      transcript.absorb(
        b"num_circuits",
        &E::Scalar::from(step_circuits.len() as u64),
      );
      transcript.absorb(b"circuit_index", &E::Scalar::from(i as u64));

      let public_values = circuit.public_values().expect("public_values failed");
      transcript.absorb(b"public_values", &public_values.as_slice());

      SatisfyingAssignment::r1cs_instance_and_witness(
        pre_state,
        &pk.S_step,
        &pk.ck,
        circuit,
        is_small,
        &mut transcript,
      )
      .expect("r1cs_instance_and_witness failed")
    })
    .unzip();

  let instances_regular: Vec<_> = instances
    .iter()
    .map(|u| u.to_regular_instance().expect("to_regular_instance failed"))
    .collect();

  (instances_regular, witnesses)
}

/// Run NeutronNovaNIFS::prove with fresh vc/vc_state/transcript.
fn nifs_prove_single<E: Engine>(
  pk: &NeutronNovaProverKey<E>,
  instances: &[R1CSInstance<E>],
  witnesses: &[R1CSWitness<E>],
  is_small: bool,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let n_padded = instances.len().next_power_of_two();
  let num_vars = pk.S_step.num_shared + pk.S_step.num_precommitted + pk.S_step.num_rest;
  let num_rounds_b = n_padded.log_2();
  let num_rounds_x = pk.S_step.num_cons.log_2();
  let num_rounds_y = num_vars.log_2() + 1;

  let mut vc = NeutronNovaVerifierCircuit::<E>::default(num_rounds_b, num_rounds_x, num_rounds_y);
  let mut vc_state =
    SatisfyingAssignment::<E>::initialize_multiround_witness(&pk.vc_shape).expect("init vc_state");

  let mut transcript = E::TE::new(b"neutronnova_prove");
  transcript.absorb(b"vk", &pk.vk_digest);

  NeutronNovaNIFS::<E>::prove(
    &pk.S_step,
    instances,
    witnesses,
    &mut vc,
    &mut vc_state,
    &pk.vc_shape,
    &pk.vc_ck,
    &mut transcript,
    is_small,
  )
  .expect("NeutronNovaNIFS::prove failed");
}

/// Verify a SNARK and print result.
fn verify_snark<E: Engine>(
  pk: &NeutronNovaProverKey<E>,
  vk: &NeutronNovaVerifierKey<E>,
  circuits: &[SmallSha256ChainCircuit<E::Scalar>],
  core_circuit: &SmallSha256ChainCircuit<E::Scalar>,
  num_instances: usize,
  is_small: bool,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let mode = if is_small { "small-value" } else { "large-value" };
  let prep =
    NeutronNovaZkSNARK::<E>::prep_prove(pk, circuits, core_circuit, is_small).expect("prep_prove");
  let snark =
    NeutronNovaZkSNARK::<E>::prove(pk, circuits, core_circuit, &prep, is_small).expect("prove");
  let res = snark.verify(vk, num_instances);
  assert!(res.is_ok(), "Verification failed: {:?}", res.err());
  eprintln!("  verified: yes ({})", mode);
}

fn benchmark_nifs_prove<E: Engine>(
  num_instances: usize,
  chain_length: usize,
  timing_data: &TimingData,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let circuits = make_circuits::<E::Scalar>(num_instances, chain_length);
  let core_circuit = circuits[0].clone();

  eprintln!(
    "Setting up NeutronNova for {} instances, chain_length={}...",
    num_instances, chain_length
  );
  let t0 = Instant::now();
  let (pk, vk) =
    NeutronNovaZkSNARK::<E>::setup(&circuits[0], &core_circuit, num_instances).expect("setup");
  let setup_ms = t0.elapsed().as_millis();
  eprintln!("Setup done in {} ms", setup_ms);

  let mut small_timings = HashMap::new();
  let mut large_timings = HashMap::new();

  for is_small in [true, false] {
    let mode = if is_small { "small" } else { "large" };
    let _mode_span = info_span!("mode", mode).entered();

    clear_timings(timing_data);

    let t_total = Instant::now();

    // Witness generation: prep_prove
    let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core_circuit, is_small)
      .expect("prep_prove");

    // Witness generation: synthesize instances
    let (instances, witnesses) =
      generate_instances_and_witnesses(&pk, &prep, &circuits, is_small);

    // NIFS prove
    nifs_prove_single(&pk, &instances, &witnesses, is_small);

    let total_ms = t_total.elapsed().as_millis();
    info!(elapsed_ms = total_ms as u64, "end_to_end_total");

    let timings = snapshot_timings(timing_data, NEUTRONNOVA_PHASES);
    if is_small {
      small_timings = timings;
    } else {
      large_timings = timings;
    }
  }

  // Verify using the full ZkSNARK pipeline
  verify_snark(&pk, &vk, &circuits, &core_circuit, num_instances, true);
  verify_snark(&pk, &vk, &circuits, &core_circuit, num_instances, false);

  let header = format!(
    "===== NeutronNova NIFS: instances={}, chain_length={}, constraints={} =====",
    num_instances, chain_length, pk.S_step.num_cons,
  );
  print_table(&header, NEUTRONNOVA_PHASES, &small_timings, &large_timings);
}

fn benchmark_zk_prove<E: Engine>(
  num_instances: usize,
  chain_length: usize,
  timing_data: &TimingData,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let circuits = make_circuits::<E::Scalar>(num_instances, chain_length);
  let core_circuit = circuits[0].clone();

  eprintln!(
    "Setting up NeutronNova for {} instances, chain_length={}...",
    num_instances, chain_length
  );
  let t0 = Instant::now();
  let (pk, vk) =
    NeutronNovaZkSNARK::<E>::setup(&circuits[0], &core_circuit, num_instances).expect("setup");
  let setup_ms = t0.elapsed().as_millis();
  eprintln!("Setup done in {} ms", setup_ms);

  let mut small_timings = HashMap::new();
  let mut large_timings = HashMap::new();

  for is_small in [true, false] {
    let mode = if is_small { "small" } else { "large" };
    let _mode_span = info_span!("mode", mode).entered();

    clear_timings(timing_data);

    // Full ZK prove
    let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core_circuit, is_small)
      .expect("prep_prove");
    let snark = NeutronNovaZkSNARK::<E>::prove(&pk, &circuits, &core_circuit, &prep, is_small)
      .expect("prove");

    let timings = snapshot_timings(timing_data, NEUTRONNOVA_ZK_PROVE_PHASES);
    if is_small {
      small_timings = timings;
    } else {
      large_timings = timings;
    }

    // Verify
    let res = snark.verify(&vk, num_instances);
    assert!(res.is_ok(), "Verification failed: {:?}", res.err());
    eprintln!("  verified: yes ({})", mode);
  }

  let header = format!(
    "===== NeutronNova ZkProve: instances={}, chain_length={}, constraints={} =====",
    num_instances, chain_length, pk.S_step.num_cons,
  );
  print_table(
    &header,
    NEUTRONNOVA_ZK_PROVE_PHASES,
    &small_timings,
    &large_timings,
  );
}

fn main() {
  let args = Args::parse();

  let (timing_layer, timing_data, _constraints_data) = TimingLayer::new();

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

  match (args.field, args.mode) {
    (FieldChoice::Bn254Fr, BenchMode::Nifs) => {
      benchmark_nifs_prove::<Bn254Engine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::Bn254Fr, BenchMode::ZkProve) => {
      benchmark_zk_prove::<Bn254Engine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::PallasFq, BenchMode::Nifs) => {
      benchmark_nifs_prove::<PallasHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::PallasFq, BenchMode::ZkProve) => {
      benchmark_zk_prove::<PallasHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::VestaFp, BenchMode::Nifs) => {
      benchmark_nifs_prove::<VestaHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::VestaFp, BenchMode::ZkProve) => {
      benchmark_zk_prove::<VestaHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
  }
}
