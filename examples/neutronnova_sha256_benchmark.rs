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
    normalize_parallel_timings, print_table, snapshot_timings,
  },
  traits::{
    Engine, circuit::SpartanCircuit, pcs::FoldingEngineTrait, transcript::TranscriptEngineTrait,
  },
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
  /// Benchmark decoupled NIFS with configurable l0
  Decoupled,
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
  /// Number of small-value sumcheck rounds (for decoupled mode)
  #[arg(long)]
  l0: Option<usize>,
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
/// - `l0 = 0`: large-value mode (no small-value optimization)
/// - `l0 > 0`: decoupled mode with l0 small-value rounds
fn nifs_prove<E: Engine>(
  pk: &NeutronNovaProverKey<E>,
  instances: &[R1CSInstance<E>],
  witnesses: &[R1CSWitness<E>],
  l0: usize,
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

  NeutronNovaNIFS::<E>::prove::<i64>(
    &pk.S_step,
    instances,
    witnesses,
    l0,
    &mut vc,
    &mut vc_state,
    &pk.vc_shape,
    &pk.vc_ck,
    &mut transcript,
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
  l0: usize,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let mode = if l0 > 0 { "small-value" } else { "large-value" };
  let prep =
    NeutronNovaZkSNARK::<E>::prep_prove(pk, circuits, core_circuit, l0).expect("prep_prove");
  let snark =
    NeutronNovaZkSNARK::<E>::prove(pk, circuits, core_circuit, &prep, l0).expect("prove");
  let res = snark.verify(vk, num_instances);
  assert!(res.is_ok(), "Verification failed: {:?}", res.err());
  eprintln!("  verified: yes ({})", mode);
}

/// Parallel spans that need to be normalized by dividing by parallelism factor.
/// These spans are called once per circuit and run in parallel.
const PARALLEL_SPANS: &[&str] = &["precom_syn", "commit_pre"];

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
  let num_cores = rayon::current_num_threads();
  // +1 for core circuit which also runs precommitted_witness
  let parallel_divisor = (num_instances + 1).min(num_cores);

  let circuits = make_circuits::<E::Scalar>(num_instances, chain_length);
  let core_circuit = circuits[0].clone();

  eprintln!(
    "Setting up NeutronNova for {} instances, chain_length={}, cores={}...",
    num_instances, chain_length, num_cores
  );
  let t0 = Instant::now();
  let (pk, vk) =
    NeutronNovaZkSNARK::<E>::setup(&circuits[0], &core_circuit, num_instances).expect("setup");
  let setup_ms = t0.elapsed().as_millis();
  eprintln!("Setup done in {} ms", setup_ms);

  let ell_b = num_instances.next_power_of_two().trailing_zeros() as usize;
  let mut small_timings = HashMap::new();
  let mut large_timings = HashMap::new();

  // l0 = ell_b for small (all rounds small-value), l0 = 0 for large
  for l0 in [ell_b, 0] {
    let mode = if l0 > 0 { "small" } else { "large" };
    let _mode_span = info_span!("mode", mode).entered();

    clear_timings(timing_data);

    let t_total = Instant::now();

    // Witness generation: prep_prove
    let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core_circuit, l0)
      .expect("prep_prove");

    // Witness generation: synthesize instances
    let is_small = l0 > 0;
    let (instances, witnesses) = generate_instances_and_witnesses(&pk, &prep, &circuits, is_small);

    // NIFS prove
    nifs_prove(&pk, &instances, &witnesses, l0);

    let total_ms = t_total.elapsed().as_millis();
    info!(elapsed_ms = total_ms as u64, "end_to_end_total");

    let mut timings = snapshot_timings(timing_data, NEUTRONNOVA_PHASES);
    // Normalize parallel spans to approximate wall-clock time
    normalize_parallel_timings(&mut timings, PARALLEL_SPANS, parallel_divisor);
    if l0 > 0 {
      small_timings = timings;
    } else {
      large_timings = timings;
    }
  }

  // Verify using the full ZkSNARK pipeline
  verify_snark(&pk, &vk, &circuits, &core_circuit, num_instances, ell_b);
  verify_snark(&pk, &vk, &circuits, &core_circuit, num_instances, 0);

  let header = format!(
    "===== NeutronNova NIFS: instances={}, chain_length={}, constraints={}, cores={} =====",
    num_instances, chain_length, pk.S_step.num_cons, num_cores,
  );
  print_table(&header, NEUTRONNOVA_PHASES, &small_timings, &large_timings);
}

fn benchmark_zk_prove<E: Engine>(
  num_instances: usize,
  chain_length: usize,
  l0: usize,
  timing_data: &TimingData,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let num_cores = rayon::current_num_threads();
  let parallel_divisor = (num_instances + 1).min(num_cores);

  let circuits = make_circuits::<E::Scalar>(num_instances, chain_length);
  let core_circuit = circuits[0].clone();

  let ell_b = num_instances.next_power_of_two().trailing_zeros() as usize;
  eprintln!(
    "Setting up NeutronNova for {} instances (ℓ_b={}), chain_length={}, l0={}, cores={}...",
    num_instances, ell_b, chain_length, l0, num_cores
  );
  let t0 = Instant::now();
  let (pk, vk) =
    NeutronNovaZkSNARK::<E>::setup(&circuits[0], &core_circuit, num_instances).expect("setup");
  let setup_ms = t0.elapsed().as_millis();
  eprintln!("Setup done in {} ms", setup_ms);

  clear_timings(timing_data);

  let mode = if l0 > 0 { "small-value" } else { "large-value" };
  let _mode_span = info_span!("mode", mode).entered();

  let t_total = Instant::now();

  // Full ZK prove
  let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core_circuit, l0)
    .expect("prep_prove");
  let snark = NeutronNovaZkSNARK::<E>::prove(&pk, &circuits, &core_circuit, &prep, l0)
    .expect("prove");

  let total_ms = t_total.elapsed().as_millis();
  info!(elapsed_ms = total_ms as u64, "end_to_end_total");

  let timings = snapshot_timings(timing_data, NEUTRONNOVA_ZK_PROVE_PHASES);

  // Verify
  let res = snark.verify(&vk, num_instances);
  assert!(res.is_ok(), "Verification failed: {:?}", res.err());
  eprintln!("  verified: yes ({})", mode);

  // Print timing table
  let header = format!(
    "===== NeutronNova ZkProve: instances={}, l0={}, chain_length={}, constraints={}, cores={} =====",
    num_instances, l0, chain_length, pk.S_step.num_cons, num_cores,
  );
  eprintln!("\n{}", header);
  eprintln!("{:-<80}", "");
  eprintln!("{:<40} {:>15}", "Phase", "Time (ms)");
  eprintln!("{:-<80}", "");
  for (_span_name, display_name) in NEUTRONNOVA_ZK_PROVE_PHASES {
    let time = timings.get(*display_name).copied().unwrap_or(0);
    eprintln!("{:<40} {:>15}", display_name, time);
  }
  eprintln!("{:-<80}", "");
  eprintln!("{:<40} {:>15}", "TOTAL", total_ms);
}

/// Decoupled NIFS phases for timing
const DECOUPLED_PHASES: &[(&str, &str)] = &[
  ("convert_to_small", "convert_small"),
  ("matrix_vector_multiply_small", "mat_vec_small"),
  ("phase1_small_value", "phase1_small"),
  ("transition_fold", "transition"),
  ("fold_witnesses", "fold_W"),
  ("fold_instances", "fold_U"),
  ("phase2_sumfold", "phase2_sumfold"),
  ("final_commitment_fold", "final_msm"),
  ("nifs_prove", "total_nifs"),
];

fn benchmark_decoupled<E: Engine>(
  num_instances: usize,
  chain_length: usize,
  l0: usize,
  timing_data: &TimingData,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let num_cores = rayon::current_num_threads();
  let parallel_divisor = (num_instances + 1).min(num_cores);

  let circuits = make_circuits::<E::Scalar>(num_instances, chain_length);
  let core_circuit = circuits[0].clone();

  let ell_b = num_instances.next_power_of_two().trailing_zeros() as usize;
  eprintln!(
    "Setting up NeutronNova for {} instances (ℓ_b={}), chain_length={}, l0={}, cores={}...",
    num_instances, ell_b, chain_length, l0, num_cores
  );

  let t0 = Instant::now();
  let (pk, _vk) =
    NeutronNovaZkSNARK::<E>::setup(&circuits[0], &core_circuit, num_instances).expect("setup");
  let setup_ms = t0.elapsed().as_millis();
  eprintln!("Setup done in {} ms", setup_ms);

  clear_timings(timing_data);

  let _mode_span = info_span!("mode", mode = "decoupled").entered();

  let t_total = Instant::now();

  // Witness generation: prep_prove (l0 > 0 requires small-value compatible witnesses)
  let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core_circuit, l0)
    .expect("prep_prove");

  // Witness generation: synthesize instances
  let is_small = l0 > 0;
  let (instances, witnesses) = generate_instances_and_witnesses(&pk, &prep, &circuits, is_small);

  // NIFS prove with l0
  nifs_prove(&pk, &instances, &witnesses, l0);

  let total_ms = t_total.elapsed().as_millis();
  info!(elapsed_ms = total_ms as u64, "end_to_end_total");

  let mut timings = snapshot_timings(timing_data, DECOUPLED_PHASES);
  normalize_parallel_timings(&mut timings, PARALLEL_SPANS, parallel_divisor);

  let header = format!(
    "===== NeutronNova Decoupled: instances={}, l0={}, chain_length={}, constraints={}, cores={} =====",
    num_instances, l0, chain_length, pk.S_step.num_cons, num_cores,
  );

  // Print single-column table for decoupled mode
  eprintln!("\n{}", header);
  eprintln!("{:-<80}", "");
  eprintln!("{:<40} {:>15}", "Phase", "Time (ms)");
  eprintln!("{:-<80}", "");
  for (_span_name, display_name) in DECOUPLED_PHASES {
    let time = timings.get(*display_name).copied().unwrap_or(0);
    eprintln!("{:<40} {:>15}", display_name, time);
  }
  eprintln!("{:-<80}", "");
  eprintln!("{:<40} {:>15}", "TOTAL", total_ms);
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
      let l0 = args.l0.expect("--l0 is required for zk-prove mode");
      benchmark_zk_prove::<Bn254Engine>(args.instances, args.chain_length, l0, &timing_data)
    }
    (FieldChoice::Bn254Fr, BenchMode::Decoupled) => {
      let l0 = args.l0.expect("--l0 is required for decoupled mode");
      benchmark_decoupled::<Bn254Engine>(args.instances, args.chain_length, l0, &timing_data)
    }
    (FieldChoice::PallasFq, BenchMode::Nifs) => {
      benchmark_nifs_prove::<PallasHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::PallasFq, BenchMode::ZkProve) => {
      let l0 = args.l0.expect("--l0 is required for zk-prove mode");
      benchmark_zk_prove::<PallasHyraxEngine>(args.instances, args.chain_length, l0, &timing_data)
    }
    (FieldChoice::PallasFq, BenchMode::Decoupled) => {
      let l0 = args.l0.expect("--l0 is required for decoupled mode");
      benchmark_decoupled::<PallasHyraxEngine>(args.instances, args.chain_length, l0, &timing_data)
    }
    (FieldChoice::VestaFp, BenchMode::Nifs) => {
      benchmark_nifs_prove::<VestaHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::VestaFp, BenchMode::ZkProve) => {
      let l0 = args.l0.expect("--l0 is required for zk-prove mode");
      benchmark_zk_prove::<VestaHyraxEngine>(args.instances, args.chain_length, l0, &timing_data)
    }
    (FieldChoice::VestaFp, BenchMode::Decoupled) => {
      let l0 = args.l0.expect("--l0 is required for decoupled mode");
      benchmark_decoupled::<VestaHyraxEngine>(args.instances, args.chain_length, l0, &timing_data)
    }
  }
}
