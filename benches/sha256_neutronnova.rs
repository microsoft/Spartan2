// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! benches/sha256_neutronnova.rs
//! Criterion benchmarks for NeutronNova on a batch of SHA-256 single-block
//! compressions.
//!
//! Run with: `RUSTFLAGS="-C target-cpu=native" cargo bench --bench sha256_neutronnova`
//! Override thread counts with `BENCH_THREADS=1,4,8`.
//! Set `NEUTRONNOVA_L0_SWEEP_SUMMARY=1` to print a coarse per-size `l0` sweep
//! summary before Criterion runs.

#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use ff::Field;
use spartan2::{
  errors::SpartanError,
  gadgets::{SmallUInt32, small_sha256_compression_function},
  neutronnova_zk::{
    NeutronNovaPrepZkSNARK, NeutronNovaProverKey, NeutronNovaVerifierKey, NeutronNovaZkSNARK,
  },
  provider::T256HyraxEngine,
  traits::{Engine, circuit::SpartanCircuit},
};
use std::{
  cmp::Ordering,
  marker::PhantomData,
  time::{Duration, Instant},
};
use tracing_subscriber::EnvFilter;

type E = T256HyraxEngine;

/// Sizes in bytes to benchmark: 1/2/4/8 KB.
const SIZES: &[usize] = &[1024, 2048, 4096, 8192];

/// SHA-256 block size in bytes.
const BLOCK_BYTES: usize = 64;

/// Coarse success target for the printed sweep summary.
const SPEEDUP_TARGET: f64 = 1.5;

type BenchResult<T> = Result<T, SpartanError>;

fn num_steps_for_size(size: usize) -> usize {
  size / BLOCK_BYTES
}

fn ell_b_for_steps(num_steps: usize) -> usize {
  num_steps.next_power_of_two().ilog2() as usize
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchMode {
  Baseline,
  L0(usize),
}

impl BenchMode {
  fn label(self) -> String {
    match self {
      BenchMode::Baseline => "baseline".to_string(),
      BenchMode::L0(l0) => format!("l0-{l0}"),
    }
  }
}

fn sweep_modes(num_steps: usize) -> Vec<BenchMode> {
  let ell_b = ell_b_for_steps(num_steps);
  std::iter::once(BenchMode::Baseline)
    .chain((1..=ell_b).map(BenchMode::L0))
    .collect()
}

fn control_modes(num_steps: usize) -> Vec<BenchMode> {
  let ell_b = ell_b_for_steps(num_steps);
  vec![BenchMode::Baseline, BenchMode::L0(ell_b)]
}

/// Step circuit: proves one SHA-256 compression on a 64-byte block.
#[derive(Clone, Debug)]
struct Sha256StepCircuit<Eng: Engine> {
  block: [u8; BLOCK_BYTES],
  _p: PhantomData<Eng>,
}

impl<Eng: Engine> Sha256StepCircuit<Eng> {
  fn new(block: [u8; BLOCK_BYTES]) -> Self {
    Self {
      block,
      _p: PhantomData,
    }
  }
}

impl<Eng: Engine> SpartanCircuit<Eng> for Sha256StepCircuit<Eng> {
  fn public_values(&self) -> Result<Vec<Eng::Scalar>, SynthesisError> {
    Ok(vec![Eng::Scalar::ZERO])
  }

  fn shared<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<Eng::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<Eng::Scalar>],
  ) -> Result<Vec<AllocatedNum<Eng::Scalar>>, SynthesisError> {
    let input_bits: Vec<Boolean> = self
      .block
      .iter()
      .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1u8 == 1u8))
      .enumerate()
      .map(|(i, b)| {
        AllocatedBit::alloc(cs.namespace(|| format!("block bit {i}")), Some(b)).map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(input_bits.len(), 512);

    const IV: [u32; 8] = [
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
      0x5be0cd19,
    ];
    let current_hash: Vec<SmallUInt32> = IV.iter().map(|&v| SmallUInt32::constant(v)).collect();

    let _next = small_sha256_compression_function(
      cs.namespace(|| "sha256 compression"),
      &input_bits,
      &current_hash,
    )?;

    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(Eng::Scalar::ZERO))?;
    x.inputize(cs.namespace(|| "inputize x"))?;

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<Eng::Scalar>],
    _: &[AllocatedNum<Eng::Scalar>],
    _: Option<&[Eng::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

/// Trivial core circuit: just exposes a single public input.
#[derive(Clone, Debug)]
struct CoreCircuit<Eng: Engine>(PhantomData<Eng>);

impl<Eng: Engine> CoreCircuit<Eng> {
  fn new() -> Self {
    Self(PhantomData)
  }
}

impl<Eng: Engine> SpartanCircuit<Eng> for CoreCircuit<Eng> {
  fn public_values(&self) -> Result<Vec<Eng::Scalar>, SynthesisError> {
    Ok(vec![Eng::Scalar::ZERO])
  }

  fn shared<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<Eng::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<Eng::Scalar>],
  ) -> Result<Vec<AllocatedNum<Eng::Scalar>>, SynthesisError> {
    let input_bits: Vec<Boolean> = (0..512)
      .map(|i| {
        AllocatedBit::alloc(cs.namespace(|| format!("core bit {i}")), Some(false))
          .map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;

    const IV: [u32; 8] = [
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
      0x5be0cd19,
    ];
    let current_hash: Vec<SmallUInt32> = IV.iter().map(|&v| SmallUInt32::constant(v)).collect();

    let _next = small_sha256_compression_function(
      cs.namespace(|| "core sha256 compression"),
      &input_bits,
      &current_hash,
    )?;

    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(Eng::Scalar::ZERO))?;
    x.inputize(cs.namespace(|| "inputize x"))?;
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<Eng::Scalar>],
    _: &[AllocatedNum<Eng::Scalar>],
    _: Option<&[Eng::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

fn thread_counts() -> Vec<usize> {
  if let Ok(val) = std::env::var("BENCH_THREADS") {
    val
      .split(',')
      .filter_map(|s| s.trim().parse().ok())
      .collect()
  } else {
    let max = std::thread::available_parallelism()
      .map(|n| n.get())
      .unwrap_or(4);
    vec![1, 2, 4, 8, 16]
      .into_iter()
      .filter(|&t| t <= max)
      .collect()
  }
}

fn summary_enabled() -> bool {
  std::env::var_os("NEUTRONNOVA_L0_SWEEP_SUMMARY").is_some()
}

fn summary_reps() -> usize {
  std::env::var("NEUTRONNOVA_SUMMARY_REPS")
    .ok()
    .and_then(|s| s.parse().ok())
    .filter(|&n| n > 0)
    .unwrap_or(1)
}

fn make_step_circuits(num_steps: usize) -> Vec<Sha256StepCircuit<E>> {
  (0..num_steps)
    .map(|i| Sha256StepCircuit::<E>::new([i as u8; BLOCK_BYTES]))
    .collect()
}

fn build_inputs(num_steps: usize) -> (Vec<Sha256StepCircuit<E>>, CoreCircuit<E>) {
  (make_step_circuits(num_steps), CoreCircuit::<E>::new())
}

fn setup_keypair(num_steps: usize) -> (NeutronNovaProverKey<E>, NeutronNovaVerifierKey<E>) {
  let step_proto = Sha256StepCircuit::<E>::new([0u8; BLOCK_BYTES]);
  let core_proto = CoreCircuit::<E>::new();
  NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps).unwrap()
}

fn prep_for_mode(
  pk: &NeutronNovaProverKey<E>,
  step_circuits: &[Sha256StepCircuit<E>],
  core_circuit: &CoreCircuit<E>,
  mode: BenchMode,
) -> BenchResult<NeutronNovaPrepZkSNARK<E>> {
  match mode {
    BenchMode::Baseline => {
      NeutronNovaZkSNARK::<E>::prep_prove(pk, step_circuits, core_circuit, true)
    }
    BenchMode::L0(l0) => {
      NeutronNovaZkSNARK::<E>::prep_prove_with_l0(pk, step_circuits, core_circuit, l0)
    }
  }
}

fn prove_for_mode(
  pk: &NeutronNovaProverKey<E>,
  step_circuits: &[Sha256StepCircuit<E>],
  core_circuit: &CoreCircuit<E>,
  prep: NeutronNovaPrepZkSNARK<E>,
  mode: BenchMode,
) -> BenchResult<(NeutronNovaZkSNARK<E>, NeutronNovaPrepZkSNARK<E>)> {
  match mode {
    BenchMode::Baseline => {
      NeutronNovaZkSNARK::<E>::prove(pk, step_circuits, core_circuit, prep, true)
    }
    BenchMode::L0(l0) => {
      NeutronNovaZkSNARK::<E>::prove_with_l0::<i64>(pk, step_circuits, core_circuit, prep, l0)
    }
  }
}

fn bench_nifs_for_mode(
  pk: &NeutronNovaProverKey<E>,
  step_circuits: &[Sha256StepCircuit<E>],
  core_circuit: &CoreCircuit<E>,
  prep: NeutronNovaPrepZkSNARK<E>,
  mode: BenchMode,
) -> BenchResult<NeutronNovaPrepZkSNARK<E>> {
  match mode {
    BenchMode::Baseline => {
      NeutronNovaZkSNARK::<E>::bench_nifs(pk, step_circuits, core_circuit, prep, true)
    }
    BenchMode::L0(l0) => {
      NeutronNovaZkSNARK::<E>::bench_nifs_with_l0::<i64>(pk, step_circuits, core_circuit, prep, l0)
    }
  }
}

fn build_proof_for_mode(num_steps: usize, mode: BenchMode) -> BenchResult<NeutronNovaZkSNARK<E>> {
  let (pk, _vk) = setup_keypair(num_steps);
  let (step_circuits, core_circuit) = build_inputs(num_steps);
  let prep = prep_for_mode(&pk, &step_circuits, &core_circuit, mode)?;
  Ok(prove_for_mode(&pk, &step_circuits, &core_circuit, prep, mode)?.0)
}

fn is_skippable_mode_error(err: &SpartanError) -> bool {
  matches!(err, SpartanError::SmallValueOverflow { .. })
}

fn probe_mode(num_steps: usize, mode: BenchMode) -> BenchResult<()> {
  build_proof_for_mode(num_steps, mode).map(|_| ())
}

fn collect_valid_modes(
  num_steps: usize,
  modes: impl IntoIterator<Item = BenchMode>,
) -> (Vec<BenchMode>, Vec<(BenchMode, SpartanError)>) {
  let mut valid = Vec::new();
  let mut skipped = Vec::new();

  for mode in modes {
    match probe_mode(num_steps, mode) {
      Ok(()) => valid.push(mode),
      Err(err) if is_skippable_mode_error(&err) => skipped.push((mode, err)),
      Err(err) => panic!(
        "failed to probe mode {} for {num_steps} steps: {err}",
        mode.label()
      ),
    }
  }

  (valid, skipped)
}

fn median_duration(mut samples: Vec<Duration>) -> Duration {
  samples.sort_unstable();
  samples[samples.len() / 2]
}

fn measure_median<F>(reps: usize, mut f: F) -> Duration
where
  F: FnMut() -> Duration,
{
  median_duration((0..reps).map(|_| f()).collect())
}

fn measure_prove_duration(
  pool: &rayon::ThreadPool,
  num_steps: usize,
  mode: BenchMode,
  reps: usize,
) -> Duration {
  measure_median(reps, || {
    pool.install(|| {
      let (pk, _vk) = setup_keypair(num_steps);
      let (step_circuits, core_circuit) = build_inputs(num_steps);
      let prep = prep_for_mode(&pk, &step_circuits, &core_circuit, mode).unwrap();
      let prep_back = prove_for_mode(&pk, &step_circuits, &core_circuit, prep, mode)
        .unwrap()
        .1;
      let start = Instant::now();
      let _ = prove_for_mode(&pk, &step_circuits, &core_circuit, prep_back, mode).unwrap();
      start.elapsed()
    })
  })
}

fn measure_prep_and_prove_duration(
  pool: &rayon::ThreadPool,
  num_steps: usize,
  mode: BenchMode,
  reps: usize,
) -> Duration {
  measure_median(reps, || {
    pool.install(|| {
      let (pk, _vk) = setup_keypair(num_steps);
      let (step_circuits, core_circuit) = build_inputs(num_steps);
      let start = Instant::now();
      let prep = prep_for_mode(&pk, &step_circuits, &core_circuit, mode).unwrap();
      let _ = prove_for_mode(&pk, &step_circuits, &core_circuit, prep, mode).unwrap();
      start.elapsed()
    })
  })
}

fn duration_ratio(a: Duration, b: Duration) -> f64 {
  a.as_secs_f64() / b.as_secs_f64()
}

fn geometric_mean(values: &[f64]) -> f64 {
  let sum_logs = values.iter().map(|v| v.ln()).sum::<f64>();
  (sum_logs / values.len() as f64).exp()
}

fn cmp_f64(a: f64, b: f64) -> Ordering {
  a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn report_l0_sweep_summary(thread_counts: &[usize]) {
  let reps = summary_reps();
  println!("== NeutronNova l0 sweep summary (reps={reps}) ==");

  for &size in SIZES {
    let num_steps = num_steps_for_size(size);
    let (valid_modes, skipped_modes) = collect_valid_modes(num_steps, sweep_modes(num_steps));
    let l0s: Vec<usize> = valid_modes
      .into_iter()
      .filter_map(|mode| match mode {
        BenchMode::Baseline => None,
        BenchMode::L0(l0) => Some(l0),
      })
      .collect();

    if !skipped_modes.is_empty() {
      println!("size={}B skipped_modes:", size);
      for (mode, err) in &skipped_modes {
        println!("  mode={} skipped: {}", mode.label(), err);
      }
    }

    if l0s.is_empty() {
      println!(
        "size={}B num_steps={} has no valid accumulator l0 values for i64; skipping summary",
        size, num_steps
      );
      continue;
    }

    let per_thread = thread_counts
      .iter()
      .map(|&nthreads| {
        let pool = rayon::ThreadPoolBuilder::new()
          .num_threads(nthreads)
          .build()
          .expect("failed to build rayon pool");
        let baseline_prove = measure_prove_duration(&pool, num_steps, BenchMode::Baseline, reps);
        let baseline_total =
          measure_prep_and_prove_duration(&pool, num_steps, BenchMode::Baseline, reps);
        let candidates = l0s
          .iter()
          .map(|&l0| {
            let mode = BenchMode::L0(l0);
            let prove = measure_prove_duration(&pool, num_steps, mode, reps);
            let total = measure_prep_and_prove_duration(&pool, num_steps, mode, reps);
            (l0, prove, total)
          })
          .collect::<Vec<_>>();
        (nthreads, baseline_prove, baseline_total, candidates)
      })
      .collect::<Vec<_>>();

    let best = l0s
      .iter()
      .map(|&l0| {
        let mut ratios = Vec::with_capacity(per_thread.len() * 2);
        let mut per_thread_ratios = Vec::with_capacity(per_thread.len());
        for (nthreads, baseline_prove, baseline_total, candidates) in &per_thread {
          let (_, prove, total) = candidates
            .iter()
            .find(|(candidate_l0, _, _)| *candidate_l0 == l0)
            .expect("candidate l0 missing");
          let prove_ratio = duration_ratio(*baseline_prove, *prove);
          let total_ratio = duration_ratio(*baseline_total, *total);
          ratios.push(prove_ratio);
          ratios.push(total_ratio);
          per_thread_ratios.push((*nthreads, prove_ratio, total_ratio));
        }
        let worst_case = ratios.iter().copied().fold(f64::INFINITY, f64::min);
        let mean_ratio = geometric_mean(&ratios);
        (l0, worst_case, mean_ratio, per_thread_ratios)
      })
      .max_by(|a, b| {
        cmp_f64(a.1, b.1)
          .then_with(|| cmp_f64(a.2, b.2))
          .then_with(|| a.0.cmp(&b.0))
      })
      .expect("at least one accumulator l0");

    let pass = best.1 >= SPEEDUP_TARGET;
    println!(
      "size={}B num_steps={} best_l0={} worst_case_speedup={:.3} geo_mean_speedup={:.3} pass={}",
      size, num_steps, best.0, best.1, best.2, pass
    );
    for (nthreads, prove_ratio, total_ratio) in &best.3 {
      println!(
        "  threads={} prove_speedup={:.3} prep_and_prove_speedup={:.3}",
        nthreads, prove_ratio, total_ratio
      );
    }
  }
}

fn neutronnova_benches(c: &mut Criterion) {
  if std::env::var_os("NEUTRONNOVA_BENCH_TRACE").is_some() || std::env::var_os("RUST_LOG").is_some()
  {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();
  }

  let thread_counts = thread_counts();
  if summary_enabled() {
    report_l0_sweep_summary(&thread_counts);
  }

  for &size in SIZES {
    let num_steps = num_steps_for_size(size);
    let (valid_modes, skipped_modes) = collect_valid_modes(num_steps, sweep_modes(num_steps));
    for (mode, err) in skipped_modes {
      println!(
        "Skipping NeutronNova SHA-256 mode={} size={}B num_steps={}: {}",
        mode.label(),
        size,
        num_steps,
        err
      );
    }

    for mode in valid_modes {
      let proof = build_proof_for_mode(num_steps, mode).unwrap();
      let proof_bytes = bincode::serialize(&proof).unwrap();
      println!(
        "NeutronNova SHA-256 mode={} size={}B num_steps={}: proof_size={} bytes",
        mode.label(),
        size,
        num_steps,
        proof_bytes.len()
      );
    }
  }

  let mut g = c.benchmark_group("neutronnova_sha256");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(3));

  for &size in SIZES {
    let num_steps = num_steps_for_size(size);
    let (sweep_modes, skipped_modes) = collect_valid_modes(num_steps, sweep_modes(num_steps));
    for (mode, err) in skipped_modes {
      println!(
        "Skipping benchmark mode={} size={}B num_steps={}: {}",
        mode.label(),
        size,
        num_steps,
        err
      );
    }
    let verify_modes: Vec<BenchMode> = control_modes(num_steps)
      .into_iter()
      .filter(|mode| sweep_modes.contains(mode))
      .collect();
    g.throughput(Throughput::Bytes(size as u64));

    for &nthreads in &thread_counts {
      let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .expect("failed to build rayon pool");

      g.bench_function(format!("setup/{size}/t{nthreads}"), |b| {
        b.iter(|| {
          pool.install(|| {
            let _ = setup_keypair(num_steps);
          });
        });
      });

      for &mode in &sweep_modes {
        g.bench_function(
          format!("prep_and_prove/{}/{size}/t{nthreads}", mode.label()),
          |b| {
            b.iter_batched(
              || {
                pool.install(|| {
                  let (pk, _vk) = setup_keypair(num_steps);
                  let (step_circuits, core_circuit) = build_inputs(num_steps);
                  (pk, step_circuits, core_circuit)
                })
              },
              |(pk, step_circuits, core_circuit)| {
                pool.install(|| {
                  let prep = prep_for_mode(&pk, &step_circuits, &core_circuit, mode).unwrap();
                  let _ = prove_for_mode(&pk, &step_circuits, &core_circuit, prep, mode).unwrap();
                });
              },
              BatchSize::LargeInput,
            );
          },
        );

        g.bench_function(format!("prove/{}/{size}/t{nthreads}", mode.label()), |b| {
          b.iter_batched(
            || {
              pool.install(|| {
                let (pk, _vk) = setup_keypair(num_steps);
                let (step_circuits, core_circuit) = build_inputs(num_steps);
                let prep = prep_for_mode(&pk, &step_circuits, &core_circuit, mode).unwrap();
                let prep_back = prove_for_mode(&pk, &step_circuits, &core_circuit, prep, mode)
                  .unwrap()
                  .1;
                (pk, step_circuits, core_circuit, prep_back)
              })
            },
            |(pk, step_circuits, core_circuit, prep)| {
              pool.install(|| {
                let _ = prove_for_mode(&pk, &step_circuits, &core_circuit, prep, mode).unwrap();
              });
            },
            BatchSize::LargeInput,
          );
        });

        g.bench_function(
          format!("nifs_pipeline/{}/{size}/t{nthreads}", mode.label()),
          |b| {
            b.iter_batched(
              || {
                pool.install(|| {
                  let (pk, _vk) = setup_keypair(num_steps);
                  let (step_circuits, core_circuit) = build_inputs(num_steps);
                  let prep = prep_for_mode(&pk, &step_circuits, &core_circuit, mode).unwrap();
                  let prep_back =
                    bench_nifs_for_mode(&pk, &step_circuits, &core_circuit, prep, mode).unwrap();
                  (pk, step_circuits, core_circuit, prep_back)
                })
              },
              |(pk, step_circuits, core_circuit, prep)| {
                pool.install(|| {
                  let _ =
                    bench_nifs_for_mode(&pk, &step_circuits, &core_circuit, prep, mode).unwrap();
                });
              },
              BatchSize::LargeInput,
            );
          },
        );
      }

      for &mode in &verify_modes {
        g.bench_function(format!("verify/{}/{size}/t{nthreads}", mode.label()), |b| {
          b.iter_batched(
            || {
              pool.install(|| {
                let (pk, vk) = setup_keypair(num_steps);
                let (step_circuits, core_circuit) = build_inputs(num_steps);
                let prep = prep_for_mode(&pk, &step_circuits, &core_circuit, mode).unwrap();
                let proof = prove_for_mode(&pk, &step_circuits, &core_circuit, prep, mode)
                  .unwrap()
                  .0;
                (vk, proof)
              })
            },
            |(vk, proof)| {
              pool.install(|| {
                proof.verify(&vk, num_steps).unwrap();
              });
            },
            BatchSize::LargeInput,
          );
        });
      }
    }
  }

  g.finish();
}

criterion_group!(benches, neutronnova_benches);
criterion_main!(benches);
