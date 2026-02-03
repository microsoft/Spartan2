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
//! Run with: `RUST_LOG=info cargo run --release --example neutronnova_sha256_benchmark -- full-nifs-prove`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use spartan2::sha256_circuits::SmallSha256ChainCircuit;
use clap::{Parser, Subcommand};
use spartan2::timing::{NEUTRONNOVA_PHASES, NEUTRONNOVA_SHORT_NAMES, TimingLayer, clear_timings, snapshot_timings};
use spartan2::{
  bellpepper::{
    r1cs::{MultiRoundSpartanWitness, SpartanWitness},
    solver::SatisfyingAssignment,
  },
  math::Math,
  neutronnova_zk::{NeutronNovaNIFS, NeutronNovaZkSNARK},
  provider::{Bn254Engine, PallasHyraxEngine},
  small_field::{DelayedReduction, SmallValueField},
  traits::{
    Engine,
    circuit::SpartanCircuit,
    pcs::FoldingEngineTrait,
    transcript::TranscriptEngineTrait,
  },
  zk::NeutronNovaVerifierCircuit,
};
use std::time::Instant;
use tracing::{info, info_span};
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(about = "NeutronNova NIFS folding benchmark: small vs large sumcheck")]
struct Args {
  #[command(subcommand)]
  command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
  /// Full NIFS prove with phase breakdown table (small vs large)
  FullNifsProve {
    #[arg(long, default_value = "4")]
    instances: usize,
    #[arg(long, default_value = "4")]
    chain_length: usize,
    #[arg(long, default_value = "1")]
    rounds: usize,
    #[arg(long, default_value = "bn254")]
    field: String,
  },
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
/// Returns (step_instances_regular, step_witnesses).
fn generate_instances_and_witnesses<E, C>(
  pk: &spartan2::neutronnova_zk::NeutronNovaProverKey<E>,
  prep: &spartan2::neutronnova_zk::NeutronNovaPrepZkSNARK<E>,
  step_circuits: &[C],
  is_small: bool,
) -> (Vec<spartan2::r1cs::R1CSInstance<E>>, Vec<spartan2::r1cs::R1CSWitness<E>>)
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
/// Returns (E_eq, Az, Bz, Cz, folded_W, folded_U).
#[allow(clippy::type_complexity)]
fn run_nifs_prove<E: Engine>(
  pk: &spartan2::neutronnova_zk::NeutronNovaProverKey<E>,
  instances: &[spartan2::r1cs::R1CSInstance<E>],
  witnesses: &[spartan2::r1cs::R1CSWitness<E>],
  is_small: bool,
) -> (
  Vec<E::Scalar>,
  Vec<E::Scalar>,
  Vec<E::Scalar>,
  Vec<E::Scalar>,
  spartan2::r1cs::R1CSWitness<E>,
  spartan2::r1cs::R1CSInstance<E>,
)
where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64, IntermediateSmallValue = i128>
    + DelayedReduction<i64, IntermediateSmallValue = i128>,
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

  let (e_eq, az, bz, cz, folded_w, folded_u) = if is_small {
    NeutronNovaNIFS::<E>::prove_small_value(
      &pk.S_step,
      instances,
      witnesses,
      &mut vc,
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &mut transcript,
    )
    .expect("NeutronNovaNIFS::prove_small_value failed")
  } else {
    NeutronNovaNIFS::<E>::prove(
      &pk.S_step,
      instances,
      witnesses,
      &mut vc,
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &mut transcript,
    )
    .expect("NeutronNovaNIFS::prove failed")
  };

  (e_eq, az, bz, cz, folded_w, folded_u)
}

fn print_hierarchical_table(
  header: &str,
  phases: &[&str],
  short_names: &[&str],
  small: &[u64],
  large: &[u64],
) {
  let col_w = 12;

  eprintln!("\n{}", header);

  eprint!("{:<14}", "");
  for name in short_names {
    eprint!("{:>width$}", name, width = col_w);
  }
  eprintln!();

  eprint!("{:<14}", "small");
  for v in small {
    eprint!("{:>width$}", v, width = col_w);
  }
  eprintln!();

  eprint!("{:<14}", "large");
  for v in large {
    eprint!("{:>width$}", v, width = col_w);
  }
  eprintln!();

  eprint!("{:<14}", "speedup");
  for (s, l) in small.iter().zip(large.iter()) {
    if *s == 0 {
      eprint!("{:>width$}", "-", width = col_w);
    } else {
      eprint!("{:>width$.2}x", *l as f64 / *s as f64, width = col_w - 1);
    }
  }
  eprintln!();

  // Also print phase name mapping for reference
  eprintln!("\nPhase key:");
  for (short, full) in short_names.iter().zip(phases.iter()) {
    eprintln!("  {:<12} = {}", short, full);
  }
}

fn run_full_nifs_prove<E: Engine>(
  num_instances: usize,
  chain_length: usize,
  rounds: usize,
  timing_data: &spartan2::timing::TimingData,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64, IntermediateSmallValue = i128>
    + DelayedReduction<i64, IntermediateSmallValue = i128>,
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

  let mut small_timings_all: Vec<Vec<u64>> = Vec::new();
  let mut large_timings_all: Vec<Vec<u64>> = Vec::new();

  for round in 0..rounds {
    let _round_span = info_span!("round", round).entered();

    for is_small in [true, false] {
      let mode = if is_small { "small" } else { "large" };
      let _mode_span = info_span!("mode", mode).entered();

      clear_timings(timing_data);

      let t_total = Instant::now();

      // Witness generation: prep_prove
      let t0 = Instant::now();
      let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core_circuit, is_small)
        .expect("prep_prove");
      let prep_ms = t0.elapsed().as_millis();

      // Witness generation: synthesize instances
      let t0 = Instant::now();
      let (instances, witnesses) =
        generate_instances_and_witnesses(&pk, &prep, &circuits, is_small);
      let gen_ms = t0.elapsed().as_millis();

      // NIFS prove
      let t0 = Instant::now();
      let (_e_eq, _az, _bz, _cz, _folded_w, _folded_u) =
        run_nifs_prove(&pk, &instances, &witnesses, is_small);
      let nifs_ms = t0.elapsed().as_millis();

      let total_ms = t_total.elapsed().as_millis();
      info!(elapsed_ms = total_ms as u64, "end_to_end_total");

      let timings = snapshot_timings(timing_data, NEUTRONNOVA_PHASES);
      if is_small {
        small_timings_all.push(timings);
      } else {
        large_timings_all.push(timings);
      }

      eprintln!(
        "  [round {}] mode={}, prep={} ms, gen={} ms, nifs={} ms, total={} ms",
        round, mode, prep_ms, gen_ms, nifs_ms, total_ms,
      );
    }

    // Verify using the full ZkSNARK pipeline for one run
    if round == 0 {
      let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core_circuit, true)
        .expect("prep_prove for verify");
      let snark = NeutronNovaZkSNARK::<E>::prove(&pk, &circuits, &core_circuit, &prep, true)
        .expect("prove for verify");
      let res = snark.verify(&vk, num_instances);
      assert!(res.is_ok(), "Verification failed: {:?}", res.err());
      eprintln!("  verified: yes (small-value)");

      let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core_circuit, false)
        .expect("prep_prove for verify");
      let snark = NeutronNovaZkSNARK::<E>::prove(&pk, &circuits, &core_circuit, &prep, false)
        .expect("prove for verify");
      let res = snark.verify(&vk, num_instances);
      assert!(res.is_ok(), "Verification failed: {:?}", res.err());
      eprintln!("  verified: yes (large-value)");
    }
  }

  // Use last round's timings (or min across rounds if multiple)
  let small_timings = if rounds == 1 {
    small_timings_all.into_iter().next().unwrap()
  } else {
    // Take element-wise min across rounds
    let n = small_timings_all[0].len();
    (0..n)
      .map(|i| small_timings_all.iter().map(|t| t[i]).min().unwrap())
      .collect()
  };
  let large_timings = if rounds == 1 {
    large_timings_all.into_iter().next().unwrap()
  } else {
    let n = large_timings_all[0].len();
    (0..n)
      .map(|i| large_timings_all.iter().map(|t| t[i]).min().unwrap())
      .collect()
  };

  let header = format!(
    "===== NeutronNova NIFS: instances={}, chain_length={}, constraints={}, rounds={} =====",
    num_instances, chain_length, pk.S_step.num_cons, rounds,
  );
  print_hierarchical_table(
    &header,
    NEUTRONNOVA_PHASES,
    NEUTRONNOVA_SHORT_NAMES,
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

  match args.command {
    Some(Command::FullNifsProve {
      instances,
      chain_length,
      rounds,
      field,
    }) => match field.as_str() {
      "bn254" => run_full_nifs_prove::<Bn254Engine>(instances, chain_length, rounds, &timing_data),
      "pallas" => {
        run_full_nifs_prove::<PallasHyraxEngine>(instances, chain_length, rounds, &timing_data)
      }
      _ => panic!("Unknown field: {}. Use 'bn254' or 'pallas'.", field),
    },
    None => {
      // Default: full-nifs-prove with default params
      run_full_nifs_prove::<Bn254Engine>(4, 4, 1, &timing_data);
    }
  }
}
