use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use criterion::black_box;
use ff::Field;
use spartan2::{
  gadgets::{SmallUInt32, small_sha256_compression_function},
  neutronnova_zk::NeutronNovaZkSNARK,
  provider::T256HyraxEngine,
  traits::{Engine, circuit::SpartanCircuit},
};
use std::{env, marker::PhantomData, time::Instant};
use tracing_subscriber::EnvFilter;

type E = T256HyraxEngine;

const BLOCK_BYTES: usize = 64;

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

fn parse_env<T>(name: &str, default: T) -> T
where
  T: std::str::FromStr + Copy,
{
  env::var(name)
    .ok()
    .and_then(|v| v.parse::<T>().ok())
    .unwrap_or(default)
}

fn make_step_circuits(num_steps: usize) -> Vec<Sha256StepCircuit<E>> {
  (0..num_steps)
    .map(|i| Sha256StepCircuit::<E>::new([i as u8; BLOCK_BYTES]))
    .collect()
}

fn main() {
  let _ = tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .try_init();

  let num_steps = parse_env("TRACE_NUM_STEPS", 16usize);
  let l0 = parse_env("TRACE_L0", 0usize);
  let threads = parse_env("TRACE_THREADS", 1usize);

  let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(threads)
    .build()
    .expect("failed to build rayon pool");

  pool.install(|| {
    let step_proto = Sha256StepCircuit::<E>::new([0u8; BLOCK_BYTES]);
    let core_proto = CoreCircuit::<E>::new();
    let (pk, _vk) = NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps).unwrap();
    let step_circuits = make_step_circuits(num_steps);
    let core_circuit = CoreCircuit::<E>::new();

    eprintln!(
      "trace_case num_steps={} l0={} threads={}",
      num_steps, l0, threads
    );

    let (proof, prep_elapsed, prove_elapsed) = if l0 == 0 {
      let prep_start = Instant::now();
      let prep =
        NeutronNovaZkSNARK::<E>::prep_prove(&pk, &step_circuits, &core_circuit, true).unwrap();
      let prep_elapsed = prep_start.elapsed();

      let prove_start = Instant::now();
      let (proof, _prep_back) =
        NeutronNovaZkSNARK::<E>::prove(&pk, &step_circuits, &core_circuit, prep, true).unwrap();
      (proof, prep_elapsed, prove_start.elapsed())
    } else {
      let prep_start = Instant::now();
      let prep = NeutronNovaZkSNARK::<E>::prep_prove_accumulator_with_l0::<i64>(
        &pk,
        &step_circuits,
        &core_circuit,
        l0,
      )
      .unwrap();
      let prep_elapsed = prep_start.elapsed();

      let prove_start = Instant::now();
      let (proof, _prep_back) = NeutronNovaZkSNARK::<E>::prove_accumulator_with_l0::<i64>(
        &pk,
        &step_circuits,
        &core_circuit,
        prep,
        l0,
      )
      .unwrap();
      (proof, prep_elapsed, prove_start.elapsed())
    };

    black_box(proof);
    eprintln!(
      "trace_case_done prep_ms={} prove_ms={} total_ms={}",
      prep_elapsed.as_millis(),
      prove_elapsed.as_millis(),
      (prep_elapsed + prove_elapsed).as_millis()
    );
  });
}
