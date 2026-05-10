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
//! Emit one-shot stage timing CSV with `NEUTRONNOVA_STAGE_TRACE=1` (use
//! `TRACE_L0=0,s,3,4` to select FullField, SmallValue, and accumulator l0s).
//!
//! Three modes ship in the sweep:
//! - `FullField` — bellpepper UInt32 SHA-256 through the standard NeutronNova
//!   prover with the small-witness optimizations *disabled*
//!   (`is_small = false`). The literal "no smallness exploited anywhere" cell.
//! - `SmallValue` — same circuit, same prover, but with `is_small = true`.
//!   Exposes the gain from the standard prover's small-witness commit/matvec
//!   optimizations on a witness whose values fit in machine words.
//! - `SmallAccumulator { l0 }` — small-value SHA-256 built directly on
//!   `SmallConstraintSystem<bool, i32>`, run through the small-accumulator
//!   prover in `src/small_neutronnova_zk.rs`.
//!
//! `FullField → SmallValue` isolates the standard-prover small-witness path;
//! `SmallValue → SmallAccumulator` isolates the additional accumulator gain.

use rayon::ThreadPool;
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper::gadgets::{sha256::sha256_compression_function, uint32::UInt32};
use bellpepper_core::{
  ConstraintSystem, SynthesisError, Variable,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use criterion::{
  BatchSize, BenchmarkGroup, Criterion, Throughput, black_box, criterion_group,
  measurement::WallTime,
};
use ff::Field;
use spartan2::{
  errors::SpartanError,
  gadgets::{
    NoBatchEq, SmallBit, SmallBoolean, SmallUInt32, small_sha256_compression_function_int,
  },
  neutronnova_zk::{
    NeutronNovaPrepZkSNARK, NeutronNovaProverKey, NeutronNovaSmallAccumulatorPrepZkSNARK,
    NeutronNovaSmallProverKey, NeutronNovaVerifierKey, NeutronNovaZkSNARK,
  },
  provider::T256HyraxEngine,
  small_constraint_system::{SmallConstraintSystem, SmallLinearCombination},
  traits::{
    Engine,
    circuit::{SmallSpartanCircuit, SpartanCircuit},
  },
};
use std::{
  fmt::{self, Debug},
  fs::File,
  io::{self, Write},
  marker::PhantomData,
  path::PathBuf,
  sync::{Arc, Mutex},
  time::{Duration, Instant},
};
use tracing::{
  Subscriber,
  field::{Field as TracingField, Visit},
  info,
};
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

type E = T256HyraxEngine;

/// Sizes in bytes to benchmark: 1/2/4/8 KB.
const SIZES: &[usize] = &[1024, 2048, 4096, 8192];

/// SHA-256 block size in bytes.
const BLOCK_BYTES: usize = 64;

/// Standard SHA-256 initial hash values (used by both gadgets).
const SHA256_IV: [u32; 8] = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Conservative hard ceiling on `l0` for SHA-256 + `SV = i32`.
///
/// The accumulator prep path evaluates `Az/Bz/Cz` via the generic matvec with
/// `i32 * bool -> i32`, so SHA-256 row values stay in native `i32` and products
/// later widen as `i32 * i32 -> i64`. The extension bound for `SV = i32` at
/// `l0 = 5` is still comfortably above the observed compression row sums; keep
/// this cap conservative unless the row-bound test and stage trace are refreshed.
const MAX_L0: usize = 5;

/// Accumulator `l0` values to compare in the benchmark sweep.
///
/// These are the useful SHA-256 operating points for the small accumulator:
/// `l0 = 3` exercises the mixed prefix/suffix path and `l0 = 4` also covers
/// the full-prefix case for the 1 KiB benchmark, where `ell_b = 4`.
const BENCH_L0_VALUES: &[usize] = &[3, 4];

// ---------------------------------------------------------------------------
// Standard-prover circuits: bellpepper UInt32 SHA-256.
// ---------------------------------------------------------------------------

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
    let alloc_bits_t = Instant::now();
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
    info!(
      elapsed_ms = %alloc_bits_t.elapsed().as_millis(),
      "bench_alloc_block_bits"
    );

    let current_hash: Vec<UInt32> = SHA256_IV.iter().map(|&v| UInt32::constant(v)).collect();
    let compression_t = Instant::now();
    let _ = sha256_compression_function(
      cs.namespace(|| "sha256 compression"),
      &input_bits,
      &current_hash,
    )?;
    info!(
      elapsed_ms = %compression_t.elapsed().as_millis(),
      "bench_sha256_compression"
    );

    let inputize_t = Instant::now();
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(Eng::Scalar::ZERO))?;
    x.inputize(cs.namespace(|| "inputize x"))?;
    info!(
      elapsed_ms = %inputize_t.elapsed().as_millis(),
      "bench_inputize_public_zero"
    );

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
struct Sha256CoreCircuit<Eng: Engine>(PhantomData<Eng>);

impl<Eng: Engine> Sha256CoreCircuit<Eng> {
  fn new() -> Self {
    Self(PhantomData)
  }
}

impl<Eng: Engine> SpartanCircuit<Eng> for Sha256CoreCircuit<Eng> {
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
    let alloc_bits_t = Instant::now();
    let input_bits: Vec<Boolean> = (0..512)
      .map(|i| {
        AllocatedBit::alloc(cs.namespace(|| format!("core bit {i}")), Some(false))
          .map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;
    info!(
      elapsed_ms = %alloc_bits_t.elapsed().as_millis(),
      "bench_alloc_block_bits"
    );

    let current_hash: Vec<UInt32> = SHA256_IV.iter().map(|&v| UInt32::constant(v)).collect();
    let compression_t = Instant::now();
    let _ = sha256_compression_function(
      cs.namespace(|| "core sha256 compression"),
      &input_bits,
      &current_hash,
    )?;
    info!(
      elapsed_ms = %compression_t.elapsed().as_millis(),
      "bench_sha256_compression"
    );

    let inputize_t = Instant::now();
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(Eng::Scalar::ZERO))?;
    x.inputize(cs.namespace(|| "inputize x"))?;
    info!(
      elapsed_ms = %inputize_t.elapsed().as_millis(),
      "bench_inputize_public_zero"
    );
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
struct SmallSha256StepCircuit<Eng: Engine> {
  block: [u8; BLOCK_BYTES],
  _p: PhantomData<Eng>,
}

impl<Eng: Engine> SmallSha256StepCircuit<Eng> {
  fn new(block: [u8; BLOCK_BYTES]) -> Self {
    Self {
      block,
      _p: PhantomData,
    }
  }
}

#[derive(Clone, Debug)]
struct SmallSha256CoreCircuit<Eng: Engine>(PhantomData<Eng>);

impl<Eng: Engine> SmallSha256CoreCircuit<Eng> {
  fn new() -> Self {
    Self(PhantomData)
  }
}

fn run_small_sha256_compression<CS: SmallConstraintSystem<bool, i32>>(
  cs: &mut CS,
  input_bits: &[SmallBoolean],
) -> Result<(), SynthesisError> {
  let current_hash = SHA256_IV
    .iter()
    .map(|&v| SmallUInt32::constant(v))
    .collect::<Vec<_>>();
  let mut eq = NoBatchEq::<bool, i32, _>::new(cs);
  let _ = small_sha256_compression_function_int::<bool, _>(&mut eq, input_bits, &current_hash)?;
  Ok(())
}

fn inputize_small_zero<CS: SmallConstraintSystem<bool, i32>>(
  cs: &mut CS,
) -> Result<(), SynthesisError> {
  let x = cs.alloc(|| "x", || Ok(false))?;
  let x_public = cs.alloc_input(|| "inputize x", || Ok(false))?;
  if !cs.is_witness_generator() {
    let mut diff = SmallLinearCombination::from_variable(x, 1i32);
    diff.add_term(x_public, -1i32);
    cs.enforce(
      || "x equals public input",
      diff,
      SmallLinearCombination::one(1i32),
      SmallLinearCombination::zero(),
    );
  }
  Ok(())
}

fn alloc_small_block_bits<CS: SmallConstraintSystem<bool, i32>>(
  cs: &mut CS,
  block: &[u8; BLOCK_BYTES],
  label: &'static str,
) -> Result<Vec<SmallBoolean>, SynthesisError> {
  block
    .iter()
    .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1u8 == 1u8))
    .enumerate()
    .map(|(i, b)| {
      SmallBit::alloc(&mut cs.namespace(|| format!("{label} bit {i}")), Some(b))
        .map(SmallBoolean::Is)
    })
    .collect()
}

impl<Eng: Engine> SmallSpartanCircuit<Eng, bool, i32> for SmallSha256StepCircuit<Eng> {
  fn public_values(&self) -> Result<Vec<bool>, SynthesisError> {
    Ok(vec![false])
  }

  fn shared<CS: SmallConstraintSystem<bool, i32>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<Variable>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: SmallConstraintSystem<bool, i32>>(
    &self,
    cs: &mut CS,
    _: &[Variable],
  ) -> Result<Vec<Variable>, SynthesisError> {
    let alloc_bits_t = Instant::now();
    let input_bits = alloc_small_block_bits(cs, &self.block, "block")?;
    info!(
      elapsed_ms = %alloc_bits_t.elapsed().as_millis(),
      "bench_small_alloc_block_bits"
    );

    let compression_t = Instant::now();
    run_small_sha256_compression(cs, &input_bits)?;
    info!(
      elapsed_ms = %compression_t.elapsed().as_millis(),
      "bench_small_sha256_compression"
    );

    let inputize_t = Instant::now();
    inputize_small_zero(cs)?;
    info!(
      elapsed_ms = %inputize_t.elapsed().as_millis(),
      "bench_small_inputize_public_zero"
    );
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: SmallConstraintSystem<bool, i32>>(
    &self,
    _: &mut CS,
    _: &[Variable],
    _: &[Variable],
    _: Option<&[Eng::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

impl<Eng: Engine> SmallSpartanCircuit<Eng, bool, i32> for SmallSha256CoreCircuit<Eng> {
  fn public_values(&self) -> Result<Vec<bool>, SynthesisError> {
    Ok(vec![false])
  }

  fn shared<CS: SmallConstraintSystem<bool, i32>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<Variable>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: SmallConstraintSystem<bool, i32>>(
    &self,
    cs: &mut CS,
    _: &[Variable],
  ) -> Result<Vec<Variable>, SynthesisError> {
    let block = [0u8; BLOCK_BYTES];
    let alloc_bits_t = Instant::now();
    let input_bits = alloc_small_block_bits(cs, &block, "core")?;
    info!(
      elapsed_ms = %alloc_bits_t.elapsed().as_millis(),
      "bench_small_alloc_block_bits"
    );

    let compression_t = Instant::now();
    run_small_sha256_compression(cs, &input_bits)?;
    info!(
      elapsed_ms = %compression_t.elapsed().as_millis(),
      "bench_small_sha256_compression"
    );

    let inputize_t = Instant::now();
    inputize_small_zero(cs)?;
    info!(
      elapsed_ms = %inputize_t.elapsed().as_millis(),
      "bench_small_inputize_public_zero"
    );
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: SmallConstraintSystem<bool, i32>>(
    &self,
    _: &mut CS,
    _: &[Variable],
    _: &[Variable],
    _: Option<&[Eng::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

// ---------------------------------------------------------------------------
// BenchMode: prover + circuit pairing.
// ---------------------------------------------------------------------------

trait BenchMode: Send + Sync {
  type ProverKey: Send + Sync;
  type StepCircuit: Clone + Debug + Send + Sync;
  type CoreCircuit: Clone + Debug + Send + Sync;
  type Prep: Send;

  fn label(&self) -> String;

  fn make_step_circuit(block: [u8; BLOCK_BYTES]) -> Self::StepCircuit;
  fn make_core_circuit() -> Self::CoreCircuit;

  fn setup_keypair(num_steps: usize) -> (Self::ProverKey, NeutronNovaVerifierKey<E>);

  fn prep_prove(
    &self,
    pk: &Self::ProverKey,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
  ) -> Result<Self::Prep, SpartanError>;

  fn prove(
    &self,
    pk: &Self::ProverKey,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
    prep: Self::Prep,
  ) -> Result<(NeutronNovaZkSNARK<E>, Self::Prep), SpartanError>;
}

/// Standard NeutronNova prover with the bellpepper UInt32 SHA-256 circuit.
///
/// `is_small` selects between the field-element prover path (`false` →
/// `FullField`) and the small-witness optimized path (`true` → `SmallValue`).
/// Same circuit, same prover key, same setup — only the flag passed to
/// `prep_prove`/`prove` differs.
#[derive(Clone, Copy, Debug)]
struct StandardMode {
  is_small: bool,
  label: &'static str,
}

impl StandardMode {
  fn assert_wiring(self) {
    match self.label {
      "full-field" => assert!(
        !self.is_small,
        "FullField benchmark must call the standard prover with is_small=false",
      ),
      "small-value" => assert!(
        self.is_small,
        "SmallValue benchmark must call the standard prover with is_small=true",
      ),
      label => panic!("unexpected standard benchmark mode label {label}"),
    }
  }
}

const FULL_FIELD: StandardMode = StandardMode {
  is_small: false,
  label: "full-field",
};

const SMALL_VALUE: StandardMode = StandardMode {
  is_small: true,
  label: "small-value",
};

impl BenchMode for StandardMode {
  type ProverKey = NeutronNovaProverKey<E>;
  type StepCircuit = Sha256StepCircuit<E>;
  type CoreCircuit = Sha256CoreCircuit<E>;
  type Prep = NeutronNovaPrepZkSNARK<E>;

  fn label(&self) -> String {
    self.label.to_string()
  }

  fn make_step_circuit(block: [u8; BLOCK_BYTES]) -> Self::StepCircuit {
    Sha256StepCircuit::new(block)
  }

  fn make_core_circuit() -> Self::CoreCircuit {
    Sha256CoreCircuit::new()
  }

  fn setup_keypair(num_steps: usize) -> (Self::ProverKey, NeutronNovaVerifierKey<E>) {
    let step_proto = Self::make_step_circuit([0u8; BLOCK_BYTES]);
    let core_proto = Self::make_core_circuit();
    NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps).unwrap()
  }

  fn prep_prove(
    &self,
    pk: &Self::ProverKey,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
  ) -> Result<Self::Prep, SpartanError> {
    self.assert_wiring();
    NeutronNovaZkSNARK::<E>::prep_prove(pk, steps, core, self.is_small)
  }

  fn prove(
    &self,
    pk: &Self::ProverKey,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
    prep: Self::Prep,
  ) -> Result<(NeutronNovaZkSNARK<E>, Self::Prep), SpartanError> {
    self.assert_wiring();
    NeutronNovaZkSNARK::<E>::prove(pk, steps, core, prep, self.is_small)
  }
}

#[derive(Clone, Copy, Debug)]
struct SmallAccumulator {
  l0: usize,
}

impl SmallAccumulator {
  fn check_l0(&self) {
    assert!(
      (1..=MAX_L0).contains(&self.l0),
      "SmallAccumulator l0={} out of range 1..={MAX_L0}; see MAX_L0 docs for the bound math",
      self.l0,
    );
  }
}

impl BenchMode for SmallAccumulator {
  type ProverKey = NeutronNovaSmallProverKey<E, i32>;
  type StepCircuit = SmallSha256StepCircuit<E>;
  type CoreCircuit = SmallSha256CoreCircuit<E>;
  type Prep = NeutronNovaSmallAccumulatorPrepZkSNARK<E, i32, bool>;

  fn label(&self) -> String {
    format!("small-accum/l0-{}", self.l0)
  }

  fn make_step_circuit(block: [u8; BLOCK_BYTES]) -> Self::StepCircuit {
    SmallSha256StepCircuit::new(block)
  }

  fn make_core_circuit() -> Self::CoreCircuit {
    SmallSha256CoreCircuit::new()
  }

  fn setup_keypair(num_steps: usize) -> (Self::ProverKey, NeutronNovaVerifierKey<E>) {
    let step_proto = Self::make_step_circuit([0u8; BLOCK_BYTES]);
    let core_proto = Self::make_core_circuit();
    NeutronNovaZkSNARK::<E>::setup_small(&step_proto, &core_proto, num_steps).unwrap()
  }

  fn prep_prove(
    &self,
    pk: &Self::ProverKey,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
  ) -> Result<Self::Prep, SpartanError> {
    self.check_l0();
    NeutronNovaSmallAccumulatorPrepZkSNARK::<E, i32, bool>::prep_prove_small(
      pk, steps, core, self.l0,
    )
  }

  fn prove(
    &self,
    pk: &Self::ProverKey,
    steps: &[Self::StepCircuit],
    core: &Self::CoreCircuit,
    prep: Self::Prep,
  ) -> Result<(NeutronNovaZkSNARK<E>, Self::Prep), SpartanError> {
    self.check_l0();
    prep.prove_small(pk, steps, core)
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
fn accumulator_l0_values(num_steps: usize) -> Vec<usize> {
  let ell_b = ell_b_for_steps(num_steps);
  BENCH_L0_VALUES
    .iter()
    .copied()
    .filter(|&l0| l0 <= ell_b && l0 <= MAX_L0)
    .collect()
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
// One-shot stage trace mode.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TraceContext {
  byte_size: usize,
  num_steps: usize,
  mode: String,
  l0: usize,
  threads: usize,
  phase: String,
}

#[derive(Clone, Debug)]
struct TraceRow {
  byte_size: usize,
  num_steps: usize,
  mode: String,
  l0: usize,
  threads: usize,
  phase: String,
  stage: String,
  elapsed_ms: u128,
}

#[derive(Clone, Default)]
struct StageTrace {
  rows: Arc<Mutex<Vec<TraceRow>>>,
  context: Arc<Mutex<Option<TraceContext>>>,
}

impl StageTrace {
  fn set_context(&self, context: TraceContext) {
    *self.context.lock().expect("trace context lock poisoned") = Some(context);
  }

  fn clear_context(&self) {
    *self.context.lock().expect("trace context lock poisoned") = None;
  }

  fn record_manual(&self, stage: &str, elapsed_ms: u128) {
    let Some(context) = self
      .context
      .lock()
      .expect("trace context lock poisoned")
      .clone()
    else {
      return;
    };
    self.push_row(context, stage.to_string(), elapsed_ms);
  }

  fn push_row(&self, context: TraceContext, stage: String, elapsed_ms: u128) {
    self
      .rows
      .lock()
      .expect("trace rows lock poisoned")
      .push(TraceRow {
        byte_size: context.byte_size,
        num_steps: context.num_steps,
        mode: context.mode,
        l0: context.l0,
        threads: context.threads,
        phase: context.phase,
        stage,
        elapsed_ms,
      });
  }

  fn rows(&self) -> Vec<TraceRow> {
    self.rows.lock().expect("trace rows lock poisoned").clone()
  }
}

#[derive(Default)]
struct TraceEventVisitor {
  elapsed_ms: Option<u128>,
  message: Option<String>,
}

impl TraceEventVisitor {
  fn parse_elapsed(&mut self, raw: &str) {
    if self.elapsed_ms.is_some() {
      return;
    }
    let trimmed = raw.trim().trim_matches('"');
    if let Ok(value) = trimmed.parse::<u128>() {
      self.elapsed_ms = Some(value);
    }
  }

  fn record_message(&mut self, raw: &str) {
    let trimmed = raw.trim().trim_matches('"');
    if !trimmed.is_empty() {
      self.message = Some(trimmed.to_string());
    }
  }
}

impl Visit for TraceEventVisitor {
  fn record_u64(&mut self, field: &TracingField, value: u64) {
    if field.name() == "elapsed_ms" {
      self.elapsed_ms = Some(u128::from(value));
    }
  }

  fn record_i64(&mut self, field: &TracingField, value: i64) {
    if field.name() == "elapsed_ms" && value >= 0 {
      self.elapsed_ms = Some(value as u128);
    }
  }

  fn record_str(&mut self, field: &TracingField, value: &str) {
    match field.name() {
      "elapsed_ms" => self.parse_elapsed(value),
      "message" => self.record_message(value),
      _ => {}
    }
  }

  fn record_debug(&mut self, field: &TracingField, value: &dyn fmt::Debug) {
    let raw = format!("{value:?}");
    match field.name() {
      "elapsed_ms" => self.parse_elapsed(&raw),
      "message" => self.record_message(&raw),
      _ => {}
    }
  }
}

impl<S> Layer<S> for StageTrace
where
  S: Subscriber,
{
  fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
    let mut visitor = TraceEventVisitor::default();
    event.record(&mut visitor);
    let Some(elapsed_ms) = visitor.elapsed_ms else {
      return;
    };
    let Some(context) = self
      .context
      .lock()
      .expect("trace context lock poisoned")
      .clone()
    else {
      return;
    };
    let stage = visitor
      .message
      .unwrap_or_else(|| event.metadata().name().to_string());
    self.push_row(context, stage, elapsed_ms);
  }
}

fn parse_trace_values(name: &str, default: &[usize]) -> Vec<usize> {
  std::env::var(name)
    .ok()
    .map(|raw| {
      raw
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
          item
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("{name} must contain comma-separated integers, got {raw:?}"))
        })
        .collect::<Vec<_>>()
    })
    .filter(|values| !values.is_empty())
    .unwrap_or_else(|| default.to_vec())
}

/// One trace selector parsed from `TRACE_L0`.
///
/// `0` selects `FullField`, the literal `s` selects `SmallValue`, and a
/// positive integer selects `SmallAccumulator { l0: n }`.
#[derive(Clone, Copy, Debug)]
enum TraceMode {
  FullField,
  SmallValue,
  Accumulator(usize),
}

fn parse_trace_modes(default: &[TraceMode]) -> Vec<TraceMode> {
  let Some(raw) = std::env::var("TRACE_L0").ok() else {
    return default.to_vec();
  };
  let parsed: Vec<TraceMode> = raw
    .split(',')
    .map(str::trim)
    .filter(|item| !item.is_empty())
    .map(|item| match item {
      "0" => TraceMode::FullField,
      "s" => TraceMode::SmallValue,
      _ => match item.parse::<usize>() {
        Ok(0) => TraceMode::FullField,
        Ok(n) => TraceMode::Accumulator(n),
        Err(_) => panic!(
          "TRACE_L0 entry must be `0` (full-field), `s` (small-value), or a positive integer (small-accum l0); got {item:?}"
        ),
      },
    })
    .collect();
  if parsed.is_empty() {
    default.to_vec()
  } else {
    parsed
  }
}

fn trace_context(
  mode: &str,
  byte_size: usize,
  l0: usize,
  threads: usize,
  phase: &str,
) -> TraceContext {
  TraceContext {
    byte_size,
    num_steps: num_steps_for_size(byte_size),
    mode: mode.to_string(),
    l0,
    threads,
    phase: phase.to_string(),
  }
}

fn run_stage_trace_case<M: BenchMode + Copy>(
  trace: &StageTrace,
  mode: M,
  byte_size: usize,
  l0: usize,
  threads: usize,
) {
  assert!(
    byte_size.is_multiple_of(BLOCK_BYTES),
    "TRACE_BYTES value {byte_size} is not divisible by {BLOCK_BYTES}",
  );
  let num_steps = num_steps_for_size(byte_size);
  let mode_label = mode.label();
  let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(threads)
    .build()
    .expect("failed to build rayon pool");

  trace.set_context(trace_context(&mode_label, byte_size, l0, threads, "setup"));
  let setup_start = Instant::now();
  let (pk, vk) = pool.install(|| M::setup_keypair(num_steps));
  trace.record_manual("total", setup_start.elapsed().as_millis());

  let (steps, core) = pool.install(|| build_inputs::<M>(num_steps));

  trace.set_context(trace_context(&mode_label, byte_size, l0, threads, "prep"));
  let prep_start = Instant::now();
  let prep = pool
    .install(|| mode.prep_prove(&pk, &steps, &core))
    .unwrap();
  trace.record_manual("total", prep_start.elapsed().as_millis());

  trace.set_context(trace_context(&mode_label, byte_size, l0, threads, "prove"));
  let prove_start = Instant::now();
  let (proof, prep_back) = pool
    .install(|| mode.prove(&pk, &steps, &core, prep))
    .unwrap();
  trace.record_manual("total", prove_start.elapsed().as_millis());

  trace.set_context(trace_context(&mode_label, byte_size, l0, threads, "verify"));
  let verify_start = Instant::now();
  pool.install(|| proof.verify(&vk, num_steps).unwrap());
  trace.record_manual("total", verify_start.elapsed().as_millis());

  trace.clear_context();
  black_box(proof);
  black_box(prep_back);
}

fn write_csv_field<W: Write>(out: &mut W, value: &str) -> io::Result<()> {
  if value
    .chars()
    .any(|ch| matches!(ch, ',' | '"' | '\n' | '\r'))
  {
    write!(out, "\"{}\"", value.replace('"', "\"\""))
  } else {
    write!(out, "{value}")
  }
}

fn write_trace_csv<W: Write>(mut out: W, rows: &[TraceRow]) -> io::Result<()> {
  writeln!(
    out,
    "bytes,num_steps,mode,l0,threads,phase,stage,elapsed_ms"
  )?;
  for row in rows {
    write!(out, "{},{},", row.byte_size, row.num_steps)?;
    write_csv_field(&mut out, &row.mode)?;
    write!(out, ",{},{},", row.l0, row.threads)?;
    write_csv_field(&mut out, &row.phase)?;
    write!(out, ",")?;
    write_csv_field(&mut out, &row.stage)?;
    writeln!(out, ",{}", row.elapsed_ms)?;
  }
  Ok(())
}

fn write_stage_trace_output(rows: &[TraceRow]) -> io::Result<()> {
  if let Some(path) = std::env::var_os("TRACE_OUTPUT") {
    let mut file = File::create(PathBuf::from(path))?;
    write_trace_csv(&mut file, rows)
  } else {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    write_trace_csv(&mut out, rows)
  }
}

fn run_stage_trace() {
  let trace = StageTrace::default();
  tracing_subscriber::registry()
    .with(trace.clone())
    .try_init()
    .expect("failed to install NeutronNova stage trace subscriber");

  let byte_sizes = parse_trace_values("TRACE_BYTES", &[1024, 2048]);
  let trace_modes = parse_trace_modes(&[
    TraceMode::FullField,
    TraceMode::SmallValue,
    TraceMode::Accumulator(3),
    TraceMode::Accumulator(4),
  ]);
  let thread_counts = parse_trace_values("TRACE_THREADS", &[1, 2, 4, 8]);

  for byte_size in byte_sizes {
    let num_steps = num_steps_for_size(byte_size);
    let ell_b = ell_b_for_steps(num_steps);
    for &mode in &trace_modes {
      for &threads in &thread_counts {
        match mode {
          TraceMode::FullField => {
            run_stage_trace_case(&trace, FULL_FIELD, byte_size, 0, threads);
          }
          TraceMode::SmallValue => {
            run_stage_trace_case(&trace, SMALL_VALUE, byte_size, 0, threads);
          }
          TraceMode::Accumulator(l0) => {
            assert!(
              l0 <= ell_b && l0 <= MAX_L0,
              "TRACE_L0 value {l0} is invalid for {byte_size} bytes; expected 1..={}",
              ell_b.min(MAX_L0),
            );
            run_stage_trace_case(&trace, SmallAccumulator { l0 }, byte_size, l0, threads);
          }
        }
      }
    }
  }

  write_stage_trace_output(&trace.rows()).expect("failed to write NeutronNova stage trace CSV");
}

// ---------------------------------------------------------------------------
// Criterion bench registration.
// ---------------------------------------------------------------------------

/// Register the `prep_and_prove` and `prove` benches for one mode at one
/// (size, threads) cell. The `setup` bench is registered once per
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
      print_proof_size(&FULL_FIELD, size, num_steps);
      print_proof_size(&SMALL_VALUE, size, num_steps);
      for l0 in accumulator_l0_values(num_steps) {
        print_proof_size(&SmallAccumulator { l0 }, size, num_steps);
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

      // FullField and SmallValue share the same setup (same circuit shape,
      // same prover key); the accumulator setup is distinct.
      g.bench_function(format!("setup/standard/{size}/t{nthreads}"), |b| {
        b.iter(|| {
          pool.install(|| {
            let _ = <StandardMode as BenchMode>::setup_keypair(num_steps);
          });
        });
      });
      g.bench_function(format!("setup/small-accum/{size}/t{nthreads}"), |b| {
        b.iter(|| {
          pool.install(|| {
            let _ = <SmallAccumulator as BenchMode>::setup_keypair(num_steps);
          });
        });
      });

      register_mode_benches(&mut g, &pool, FULL_FIELD, num_steps, size, nthreads);
      register_verify_bench(&mut g, &pool, FULL_FIELD, num_steps, size, nthreads);

      register_mode_benches(&mut g, &pool, SMALL_VALUE, num_steps, size, nthreads);
      register_verify_bench(&mut g, &pool, SMALL_VALUE, num_steps, size, nthreads);

      for &l0 in &valid_l0s {
        let mode = SmallAccumulator { l0 };
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

fn main() {
  if std::env::var_os("NEUTRONNOVA_STAGE_TRACE").is_some() {
    run_stage_trace();
  } else {
    benches();
    Criterion::default().configure_from_args().final_summary();
  }
}
