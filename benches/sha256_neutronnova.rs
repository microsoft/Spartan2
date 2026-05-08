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
//!
//! The bench measures two prover variants against their *natural* circuits:
//! - `BaselineMode` runs the standard NeutronNova prover on a SHA-256 circuit
//!   built from `bellpepper::gadgets::sha256::sha256_compression_function`
//!   (full-field `UInt32`s).
//! - `AccumulatorMode { l0 }` runs the small-value accumulator prover on a
//!   SHA-256 circuit built from `small_sha256_compression_function`
//!   (`SmallUInt32`).
//!
//! The choice of compression gadget is pinned by the `Sha256Gadget` trait
//! and threaded through a shared `Sha256StepCircuit<E, G>` /
//! `Sha256CoreCircuit<E, G>` so neither circuit body is duplicated.

use rayon::ThreadPool;
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper::gadgets::{sha256::sha256_compression_function, uint32::UInt32};
use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use criterion::{
  BatchSize, BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main,
  measurement::WallTime,
};
use ff::{Field, PrimeField};
use spartan2::{
  errors::SpartanError,
  gadgets::{SmallUInt32, small_sha256_compression_function},
  neutronnova_zk::{
    NeutronNovaAccumulatorPrepZkSNARK, NeutronNovaPrepZkSNARK, NeutronNovaProverKey,
    NeutronNovaVerifierKey, NeutronNovaZkSNARK,
  },
  provider::T256HyraxEngine,
  traits::{Engine, circuit::SpartanCircuit},
};
use std::{fmt::Debug, marker::PhantomData, time::Duration};
use tracing_subscriber::EnvFilter;

type E = T256HyraxEngine;

/// Sizes in bytes to benchmark: 1/2/4/8 KB.
const SIZES: &[usize] = &[1024, 2048, 4096, 8192];

/// SHA-256 block size in bytes.
const BLOCK_BYTES: usize = 64;

/// Standard SHA-256 initial hash values (used by both gadgets).
const SHA256_IV: [u32; 8] = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Hard ceiling on `l0` for SHA-256 + `SV = i64` + `BatchingEq<21>`.
///
/// `vec_to_small_for_extension::<_, i64, 2>(&Az, l0)` requires
/// `|Az[i]| ≤ 2^62 / 3^l0`. SHA-256 row sums reach ~`2^53.7` empirically
/// (matrix coefficients up to `2^21`, witness up to `2^32`). At `l0 = 5`
/// the bound is `~2^54.1` — ~1.3× margin; `l0 = 6` fails with
/// `SmallValueOverflow`. Lifting this requires switching `SV` to a wider
/// type or shrinking `BatchingEq<K>` in `small_sha256_compression_function`.
const MAX_L0: usize = 5;

// ---------------------------------------------------------------------------
// Sha256Gadget: the compression-function plug-in point.
// ---------------------------------------------------------------------------

/// Picks which SHA-256 compression gadget the circuit uses.
///
/// Each impl owns the entire compression call, including the IV element type
/// (`UInt32` vs `SmallUInt32`), so the generic circuit body never names those
/// types directly.
trait Sha256Gadget<Scalar: PrimeField>: 'static + Clone + Debug + Send + Sync {
  fn run<CS: ConstraintSystem<Scalar>>(
    cs: CS,
    input_bits: &[Boolean],
  ) -> Result<(), SynthesisError>;
}

/// Full-field SHA-256 (bellpepper's `sha256_compression_function` + `UInt32`).
#[derive(Clone, Copy, Debug, Default)]
struct FullFieldSha256;

impl<S: PrimeField> Sha256Gadget<S> for FullFieldSha256 {
  fn run<CS: ConstraintSystem<S>>(
    mut cs: CS,
    input_bits: &[Boolean],
  ) -> Result<(), SynthesisError> {
    let current_hash: Vec<UInt32> = SHA256_IV.iter().map(|&v| UInt32::constant(v)).collect();
    let _ = sha256_compression_function(
      cs.namespace(|| "sha256 compression"),
      input_bits,
      &current_hash,
    )?;
    Ok(())
  }
}

/// Small-value SHA-256 (local `small_sha256_compression_function` + `SmallUInt32`).
#[derive(Clone, Copy, Debug, Default)]
struct SmallValueSha256;

impl<S: PrimeField> Sha256Gadget<S> for SmallValueSha256 {
  fn run<CS: ConstraintSystem<S>>(
    mut cs: CS,
    input_bits: &[Boolean],
  ) -> Result<(), SynthesisError> {
    let current_hash: Vec<SmallUInt32> = SHA256_IV
      .iter()
      .map(|&v| SmallUInt32::constant(v))
      .collect();
    let _ = small_sha256_compression_function(
      cs.namespace(|| "sha256 compression"),
      input_bits,
      &current_hash,
    )?;
    Ok(())
  }
}

// ---------------------------------------------------------------------------
// Circuits: one shape, parameterized over the gadget.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Sha256StepCircuit<Eng: Engine, G: Sha256Gadget<Eng::Scalar>> {
  block: [u8; BLOCK_BYTES],
  _p: PhantomData<(Eng, G)>,
}

impl<Eng: Engine, G: Sha256Gadget<Eng::Scalar>> Sha256StepCircuit<Eng, G> {
  fn new(block: [u8; BLOCK_BYTES]) -> Self {
    Self {
      block,
      _p: PhantomData,
    }
  }
}

impl<Eng: Engine, G: Sha256Gadget<Eng::Scalar>> SpartanCircuit<Eng> for Sha256StepCircuit<Eng, G> {
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

    G::run(cs.namespace(|| "compress"), &input_bits)?;

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

#[derive(Clone, Debug)]
struct Sha256CoreCircuit<Eng: Engine, G: Sha256Gadget<Eng::Scalar>>(PhantomData<(Eng, G)>);

impl<Eng: Engine, G: Sha256Gadget<Eng::Scalar>> Sha256CoreCircuit<Eng, G> {
  fn new() -> Self {
    Self(PhantomData)
  }
}

impl<Eng: Engine, G: Sha256Gadget<Eng::Scalar>> SpartanCircuit<Eng> for Sha256CoreCircuit<Eng, G> {
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

    G::run(cs.namespace(|| "core compress"), &input_bits)?;

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

// ---------------------------------------------------------------------------
// BenchMode: prover + circuit pairing.
// ---------------------------------------------------------------------------

trait BenchMode: Send + Sync {
  type StepCircuit: SpartanCircuit<E> + Clone + Debug + Send + Sync;
  type CoreCircuit: SpartanCircuit<E> + Clone + Debug + Send + Sync;
  type Prep: Send;

  fn label(&self) -> String;

  fn make_step_circuit(block: [u8; BLOCK_BYTES]) -> Self::StepCircuit;
  fn make_core_circuit() -> Self::CoreCircuit;

  fn setup_keypair(num_steps: usize) -> (NeutronNovaProverKey<E>, NeutronNovaVerifierKey<E>) {
    let step_proto = Self::make_step_circuit([0u8; BLOCK_BYTES]);
    let core_proto = Self::make_core_circuit();
    NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps).unwrap()
  }

  fn prep_prove(
    &self,
    pk: &NeutronNovaProverKey<E>,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
  ) -> Result<Self::Prep, SpartanError>;

  fn prove(
    &self,
    pk: &NeutronNovaProverKey<E>,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
    prep: Self::Prep,
  ) -> Result<(NeutronNovaZkSNARK<E>, Self::Prep), SpartanError>;

  fn bench_nifs(
    &self,
    pk: &NeutronNovaProverKey<E>,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
    prep: Self::Prep,
  ) -> Result<Self::Prep, SpartanError>;
}

#[derive(Clone, Copy, Debug)]
struct BaselineMode;

impl BenchMode for BaselineMode {
  type StepCircuit = Sha256StepCircuit<E, FullFieldSha256>;
  type CoreCircuit = Sha256CoreCircuit<E, FullFieldSha256>;
  type Prep = NeutronNovaPrepZkSNARK<E>;

  fn label(&self) -> String {
    "baseline".to_string()
  }

  fn make_step_circuit(block: [u8; BLOCK_BYTES]) -> Self::StepCircuit {
    Sha256StepCircuit::new(block)
  }

  fn make_core_circuit() -> Self::CoreCircuit {
    Sha256CoreCircuit::new()
  }

  fn prep_prove(
    &self,
    pk: &NeutronNovaProverKey<E>,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
  ) -> Result<Self::Prep, SpartanError> {
    NeutronNovaZkSNARK::<E>::prep_prove(pk, steps, core, true)
  }

  fn prove(
    &self,
    pk: &NeutronNovaProverKey<E>,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
    prep: Self::Prep,
  ) -> Result<(NeutronNovaZkSNARK<E>, Self::Prep), SpartanError> {
    NeutronNovaZkSNARK::<E>::prove(pk, steps, core, prep, true)
  }

  fn bench_nifs(
    &self,
    pk: &NeutronNovaProverKey<E>,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
    prep: Self::Prep,
  ) -> Result<Self::Prep, SpartanError> {
    NeutronNovaZkSNARK::<E>::bench_nifs(pk, steps, core, prep, true)
  }
}

#[derive(Clone, Copy, Debug)]
struct AccumulatorMode {
  l0: usize,
}

impl AccumulatorMode {
  fn check_l0(&self) {
    assert!(
      (1..=MAX_L0).contains(&self.l0),
      "AccumulatorMode l0={} out of range 1..={MAX_L0}; see MAX_L0 docs for the bound math",
      self.l0,
    );
  }
}

impl BenchMode for AccumulatorMode {
  type StepCircuit = Sha256StepCircuit<E, SmallValueSha256>;
  type CoreCircuit = Sha256CoreCircuit<E, SmallValueSha256>;
  type Prep = NeutronNovaAccumulatorPrepZkSNARK<E, i64>;

  fn label(&self) -> String {
    format!("l0-{}", self.l0)
  }

  fn make_step_circuit(block: [u8; BLOCK_BYTES]) -> Self::StepCircuit {
    Sha256StepCircuit::new(block)
  }

  fn make_core_circuit() -> Self::CoreCircuit {
    Sha256CoreCircuit::new()
  }

  fn prep_prove(
    &self,
    pk: &NeutronNovaProverKey<E>,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
  ) -> Result<Self::Prep, SpartanError> {
    self.check_l0();
    NeutronNovaZkSNARK::<E>::prep_prove_accumulator_with_l0::<i64>(pk, steps, core, self.l0)
  }

  fn prove(
    &self,
    pk: &NeutronNovaProverKey<E>,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
    prep: Self::Prep,
  ) -> Result<(NeutronNovaZkSNARK<E>, Self::Prep), SpartanError> {
    self.check_l0();
    NeutronNovaZkSNARK::<E>::prove_accumulator_with_l0::<i64>(pk, steps, core, prep, self.l0)
  }

  fn bench_nifs(
    &self,
    pk: &NeutronNovaProverKey<E>,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
    prep: Self::Prep,
  ) -> Result<Self::Prep, SpartanError> {
    self.check_l0();
    NeutronNovaZkSNARK::<E>::bench_accumulator_nifs_with_l0::<i64>(pk, steps, core, prep, self.l0)
  }
}

// ---------------------------------------------------------------------------
// Generic per-mode helpers.
// ---------------------------------------------------------------------------

fn num_steps_for_size(size: usize) -> usize {
  size / BLOCK_BYTES
}

fn ell_b_for_steps(num_steps: usize) -> usize {
  num_steps.next_power_of_two().ilog2() as usize
}

fn make_step_circuits<M: BenchMode>(num_steps: usize) -> Vec<M::StepCircuit> {
  (0..num_steps)
    .map(|i| M::make_step_circuit([i as u8; BLOCK_BYTES]))
    .collect()
}

fn build_inputs<M: BenchMode>(num_steps: usize) -> (Vec<M::StepCircuit>, M::CoreCircuit) {
  (make_step_circuits::<M>(num_steps), M::make_core_circuit())
}

fn build_proof_for_mode<M: BenchMode>(
  num_steps: usize,
  mode: &M,
) -> Result<NeutronNovaZkSNARK<E>, SpartanError> {
  let (pk, _vk) = M::setup_keypair(num_steps);
  let (steps, core) = build_inputs::<M>(num_steps);
  let prep = mode.prep_prove(&pk, &steps, &core)?;
  Ok(mode.prove(&pk, &steps, &core, prep)?.0)
}

/// Accumulator `l0` values to bench at `num_steps`.
///
/// The cap is `min(ell_b, MAX_L0)` — anything above `MAX_L0` is known to
/// overflow `vec_to_small_for_extension`, so we skip it a priori instead of
/// probing at runtime. See `MAX_L0` for the bound math.
fn accumulator_l0_values(num_steps: usize) -> Vec<usize> {
  (1..=ell_b_for_steps(num_steps).min(MAX_L0)).collect()
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

// ---------------------------------------------------------------------------
// Criterion bench registration.
// ---------------------------------------------------------------------------

/// Register the `prep_and_prove`, `prove`, and `nifs_pipeline` benches for one
/// mode at one (size, threads) cell. The `setup` bench is registered once per
/// (size, threads) by the caller — it only depends on the circuit shape, not
/// the prover variant.
fn register_mode_benches<M: BenchMode + Copy>(
  g: &mut BenchmarkGroup<'_, WallTime>,
  pool: &ThreadPool,
  mode: M,
  num_steps: usize,
  size: usize,
  nthreads: usize,
) {
  let label = mode.label();

  g.bench_function(format!("prep_and_prove/{label}/{size}/t{nthreads}"), |b| {
    b.iter_batched(
      || {
        pool.install(|| {
          let (pk, _vk) = M::setup_keypair(num_steps);
          let (steps, core) = build_inputs::<M>(num_steps);
          (pk, steps, core)
        })
      },
      |(pk, steps, core)| {
        pool.install(|| {
          let prep = mode.prep_prove(&pk, &steps, &core).unwrap();
          let _ = mode.prove(&pk, &steps, &core, prep).unwrap();
        });
      },
      BatchSize::LargeInput,
    );
  });

  g.bench_function(format!("prove/{label}/{size}/t{nthreads}"), |b| {
    b.iter_batched(
      || {
        pool.install(|| {
          let (pk, _vk) = M::setup_keypair(num_steps);
          let (steps, core) = build_inputs::<M>(num_steps);
          let prep = mode.prep_prove(&pk, &steps, &core).unwrap();
          let prep_back = mode.prove(&pk, &steps, &core, prep).unwrap().1;
          (pk, steps, core, prep_back)
        })
      },
      |(pk, steps, core, prep)| {
        pool.install(|| {
          let _ = mode.prove(&pk, &steps, &core, prep).unwrap();
        });
      },
      BatchSize::LargeInput,
    );
  });

  g.bench_function(format!("nifs_pipeline/{label}/{size}/t{nthreads}"), |b| {
    b.iter_batched(
      || {
        pool.install(|| {
          let (pk, _vk) = M::setup_keypair(num_steps);
          let (steps, core) = build_inputs::<M>(num_steps);
          let prep = mode.prep_prove(&pk, &steps, &core).unwrap();
          let prep_back = mode.bench_nifs(&pk, &steps, &core, prep).unwrap();
          (pk, steps, core, prep_back)
        })
      },
      |(pk, steps, core, prep)| {
        pool.install(|| {
          let _ = mode.bench_nifs(&pk, &steps, &core, prep).unwrap();
        });
      },
      BatchSize::LargeInput,
    );
  });
}

fn register_verify_bench<M: BenchMode + Copy>(
  g: &mut BenchmarkGroup<'_, WallTime>,
  pool: &ThreadPool,
  mode: M,
  num_steps: usize,
  size: usize,
  nthreads: usize,
) {
  let label = mode.label();
  g.bench_function(format!("verify/{label}/{size}/t{nthreads}"), |b| {
    b.iter_batched(
      || {
        pool.install(|| {
          let (pk, vk) = M::setup_keypair(num_steps);
          let (steps, core) = build_inputs::<M>(num_steps);
          let prep = mode.prep_prove(&pk, &steps, &core).unwrap();
          let proof = mode.prove(&pk, &steps, &core, prep).unwrap().0;
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

fn print_proof_size<M: BenchMode>(mode: &M, size: usize, num_steps: usize) {
  let proof = build_proof_for_mode(num_steps, mode).unwrap();
  let proof_bytes = bincode::serialize(&proof).unwrap();
  println!(
    "NeutronNova SHA-256 mode={} size={size}B num_steps={num_steps}: proof_size={} bytes",
    mode.label(),
    proof_bytes.len()
  );
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

  // Print proof sizes once per (size, mode) combo. Off by default — each call
  // to `print_proof_size` runs a full prove just to serialize the output, so
  // skipping it shaves ~20s off bench startup. Set `NEUTRONNOVA_PROOF_SIZES=1`
  // to opt back in.
  if std::env::var_os("NEUTRONNOVA_PROOF_SIZES").is_some() {
    for &size in SIZES {
      let num_steps = num_steps_for_size(size);
      print_proof_size(&BaselineMode, size, num_steps);
      for l0 in accumulator_l0_values(num_steps) {
        print_proof_size(&AccumulatorMode { l0 }, size, num_steps);
      }
    }
  }

  let mut g = c.benchmark_group("neutronnova_sha256");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(3));

  for &size in SIZES {
    let num_steps = num_steps_for_size(size);
    let valid_l0s = accumulator_l0_values(num_steps);
    g.throughput(Throughput::Bytes(size as u64));

    // Verify exactly one accumulator mode per size (the deepest l0 we sweep).
    let verify_max_l0 = valid_l0s.last().copied();

    for &nthreads in &thread_counts {
      let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .expect("failed to build rayon pool");

      // setup is per-circuit-shape, but baseline and accumulator share the
      // same setup signature; report under "setup" label without per-mode
      // duplication. We use the baseline circuit because it's the most
      // expensive shape (full-field SHA-256).
      g.bench_function(format!("setup/{size}/t{nthreads}"), |b| {
        b.iter(|| {
          pool.install(|| {
            let _ = BaselineMode::setup_keypair(num_steps);
          });
        });
      });

      register_mode_benches(&mut g, &pool, BaselineMode, num_steps, size, nthreads);
      register_verify_bench(&mut g, &pool, BaselineMode, num_steps, size, nthreads);

      for &l0 in &valid_l0s {
        let mode = AccumulatorMode { l0 };
        register_mode_benches(&mut g, &pool, mode, num_steps, size, nthreads);
        if Some(l0) == verify_max_l0 {
          register_verify_bench(&mut g, &pool, mode, num_steps, size, nthreads);
        }
      }
    }
  }

  g.finish();
}

criterion_group!(benches, neutronnova_benches);
criterion_main!(benches);
