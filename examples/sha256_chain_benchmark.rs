// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/sha256_chain_benchmark.rs
//! Benchmark SHA-256 hash chains comparing:
//! - Original sumcheck (baseline)
//! - Small-value sumcheck with I64Batch21
//!
//! Run with: `RUST_LOG=info cargo run --release --no-default-features --example sha256_chain_benchmark`
//! Or for CSV only: `cargo run --release --no-default-features --example sha256_chain_benchmark 2>/dev/null`
//!
//! CLI modes:
//!   single 26               - Run only 2^26 (for profiling)
//!   range-sweep             - Sweep 16-26 (default)
//!   range-sweep --min 16 --max 20 - Custom range

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper_core::{
  Circuit, ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use clap::{Parser, Subcommand};
use ff::{Field, PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use spartan2::{
  gadgets::small_sha256_with_prefix,
  polys::multilinear::MultilinearPolynomial,
  provider::PallasHyraxEngine,
  small_field::{I64Batch21, SmallMultiEqConfig, SmallValueField},
  spartan::SpartanSNARK,
  sumcheck::SumcheckProof,
  traits::{
    Engine, circuit::SpartanCircuit, snark::R1CSSNARKTrait, transcript::TranscriptEngineTrait,
  },
};
use std::{io::Write, marker::PhantomData, time::Instant};
use tracing::info;
use tracing_subscriber::EnvFilter;

// Use PallasHyraxEngine which has Barrett-optimized SmallLargeMul for Fq
type E = PallasHyraxEngine;
type F = <E as Engine>::Scalar;

#[derive(Parser)]
#[command(about = "SHA-256 chain benchmark: original vs small-value sumcheck")]
struct Args {
  #[command(subcommand)]
  command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
  /// Run a single num_vars value (for profiling)
  Single { num_vars: usize },
  /// Run a range sweep
  RangeSweep {
    #[arg(long, default_value = "16")]
    min: usize,
    #[arg(long, default_value = "26")]
    max: usize,
  },
}

/// Convert num_vars to chain_length.
/// num_vars=16 → chain=2, num_vars=18 → chain=8, etc.
/// Formula: chain_length = 2^(num_vars - 15)
fn num_vars_to_chain_length(num_vars: usize) -> usize {
  1 << (num_vars - 15)
}

/// Benchmark result for a single chain length
struct BenchmarkResult {
  chain_length: usize,
  num_vars: usize,
  num_constraints: usize,
  witness_ms: u128,
  extract_ms: u128, // included in total but not printed separately
  orig_sumcheck_ms: u128,
  small_sumcheck_ms: u128,
}

/// SHA-256 chain circuit using SmallMultiEq for small-value optimization compatibility.
///
/// Chains `chain_length` SHA-256 hashes starting from a 256-bit input.
/// Hash[0] = SHA-256(input), Hash[i] = SHA-256(Hash[i-1])
#[derive(Debug)]
struct SmallSha256ChainCircuit<Scalar: PrimeField, C> {
  /// 32-byte (256-bit) input to start the chain
  input: [u8; 32],
  /// Number of SHA-256 hashes in the chain
  chain_length: usize,
  _p: PhantomData<(Scalar, C)>,
}

// Manual Clone impl that doesn't require C: Clone (C is phantom)
impl<Scalar: PrimeField, C> Clone for SmallSha256ChainCircuit<Scalar, C> {
  fn clone(&self) -> Self {
    Self {
      input: self.input,
      chain_length: self.chain_length,
      _p: PhantomData,
    }
  }
}

impl<Scalar: PrimeField + PrimeFieldBits, C: SmallMultiEqConfig>
  SmallSha256ChainCircuit<Scalar, C>
{
  fn new(input: [u8; 32], chain_length: usize) -> Self {
    Self {
      input,
      chain_length,
      _p: PhantomData,
    }
  }

  /// Compute the expected final hash by applying SHA-256 chain_length times
  fn expected_output(&self) -> [u8; 32] {
    let mut current = self.input;
    for _ in 0..self.chain_length {
      let mut hasher = Sha256::new();
      hasher.update(current);
      current = hasher.finalize().into();
    }
    current
  }
}

impl<E: Engine, C: SmallMultiEqConfig> SpartanCircuit<E> for SmallSha256ChainCircuit<E::Scalar, C>
where
  E::Scalar: SmallValueField<C::SmallValue>,
{
  fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
    // The final hash is the public output
    let output = self.expected_output();
    let hash_scalars: Vec<<E as Engine>::Scalar> = output
      .iter()
      .flat_map(|&byte| {
        (0..8).rev().map(move |i| {
          if (byte >> i) & 1 == 1 {
            E::Scalar::ONE
          } else {
            E::Scalar::ZERO
          }
        })
      })
      .collect();
    Ok(hash_scalars)
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // 1. Allocate input bits (256 bits)
    let mut current_bits: Vec<Boolean> = self
      .input
      .iter()
      .enumerate()
      .flat_map(|(byte_idx, &byte)| {
        (0..8).rev().enumerate().map(move |(bit_idx, i)| {
          let bit_val = (byte >> i) & 1 == 1;
          (byte_idx, bit_idx, bit_val)
        })
      })
      .map(|(byte_idx, bit_idx, bit_val)| {
        AllocatedBit::alloc(
          cs.namespace(|| format!("input_{}_{}", byte_idx, bit_idx)),
          Some(bit_val),
        )
        .map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // 2. Chain SHA-256 hashes
    for chain_idx in 0..self.chain_length {
      let prefix = format!("c{}_", chain_idx);
      let hash_bits = small_sha256_with_prefix::<_, _, C>(cs, &current_bits, &prefix)?;
      current_bits = hash_bits;
    }

    // 3. Sanity-check against expected output
    let expected = self.expected_output();
    let expected_bits: Vec<bool> = expected
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1))
      .collect();

    for (i, (computed, expected_bit)) in current_bits.iter().zip(expected_bits.iter()).enumerate() {
      let computed_val = match computed {
        Boolean::Is(bit) => bit.get_value().unwrap(),
        Boolean::Not(bit) => !bit.get_value().unwrap(),
        Boolean::Constant(b) => *b,
      };
      assert_eq!(
        computed_val, *expected_bit,
        "Hash bit {} mismatch: computed={}, expected={}",
        i, computed_val, expected_bit
      );
    }

    // 4. Expose final hash bits as public inputs
    for (i, bit) in current_bits.iter().enumerate() {
      let n = AllocatedNum::alloc_input(cs.namespace(|| format!("public num {i}")), || {
        Ok(
          if bit.get_value().ok_or(SynthesisError::AssignmentMissing)? {
            E::Scalar::ONE
          } else {
            E::Scalar::ZERO
          },
        )
      })?;

      cs.enforce(
        || format!("bit == num {i}"),
        |_| bit.lc(CS::one(), E::Scalar::ONE),
        |lc| lc + CS::one(),
        |lc| lc + n.get_variable(),
      );
    }

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
    _: &[AllocatedNum<E::Scalar>],
    _: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

impl<C: SmallMultiEqConfig> Circuit<F> for SmallSha256ChainCircuit<F, C>
where
  F: SmallValueField<C::SmallValue>,
{
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // Allocate input bits (256 bits, big-endian per byte)
    let mut current_bits: Vec<Boolean> = self
      .input
      .iter()
      .enumerate()
      .flat_map(|(byte_idx, &byte)| {
        (0..8).rev().map(move |i| {
          let bit_val = (byte >> i) & 1 == 1;
          (byte_idx, 7 - i, bit_val)
        })
      })
      .map(|(byte_idx, i, bit_val)| {
        AllocatedBit::alloc(
          cs.namespace(|| format!("bit {}_{}", byte_idx, i)),
          Some(bit_val),
        )
        .map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Chain SHA-256 hashes
    for chain_idx in 0..self.chain_length {
      let prefix = format!("c{}_", chain_idx);
      let hash_bits = small_sha256_with_prefix::<_, _, C>(cs, &current_bits, &prefix)?;
      current_bits = hash_bits;
    }

    Ok(())
  }
}

fn run_chain_benchmark<C: SmallMultiEqConfig>(
  input: [u8; 32],
  chain_length: usize,
  expected_num_vars: usize,
) -> BenchmarkResult
where
  F: SmallValueField<C::SmallValue> + SmallValueField<i64, IntermediateSmallValue = i128>,
{
  // Create the circuit
  let small_circuit = SmallSha256ChainCircuit::<F, C>::new(input, chain_length);

  // === SETUP (one-time, not included in proving time) ===
  let t0 = Instant::now();
  let (pk, _vk) = SpartanSNARK::<E>::setup(small_circuit.clone()).expect("setup failed");
  let setup_ms = t0.elapsed().as_millis();
  let num_constraints = pk.sizes()[4];
  info!(setup_ms, num_constraints, "setup");

  // === WITNESS SYNTHESIS ===
  let t0 = Instant::now();
  let prep_snark =
    SpartanSNARK::<E>::prep_prove(&pk, small_circuit.clone(), true).expect("prep_prove failed");
  let witness_ms = t0.elapsed().as_millis();
  info!(witness_ms, "witness synthesis");

  // === EXTRACT SUMCHECK INPUTS ===
  let t0 = Instant::now();
  let (az, bz, cz, tau) =
    SpartanSNARK::<E>::extract_outer_sumcheck_inputs(&pk, small_circuit, &prep_snark)
      .expect("extract_outer_sumcheck_inputs failed");
  let extract_ms = t0.elapsed().as_millis();
  info!(extract_ms, "extract inputs");

  let num_vars = tau.len();

  // Verify we got the expected num_vars
  assert_eq!(
    num_vars, expected_num_vars,
    "Expected num_vars={} but got {}. Adjust chain_length.",
    expected_num_vars, num_vars
  );

  // Create field-element polynomials for the original method
  let mut az1 = MultilinearPolynomial::new(az.clone());
  let mut bz1 = MultilinearPolynomial::new(bz.clone());
  let mut cz1 = MultilinearPolynomial::new(cz.clone());

  // Create small-value polynomials for the optimized method (using i64)
  let az_poly = MultilinearPolynomial::new(az.clone());
  let bz_poly = MultilinearPolynomial::new(bz.clone());
  let az_small =
    MultilinearPolynomial::<i64>::try_from_field(&az_poly).expect("Az values too large for i64");
  let bz_small =
    MultilinearPolynomial::<i64>::try_from_field(&bz_poly).expect("Bz values too large for i64");

  let claim = F::ZERO;

  // ===== ORIGINAL SUMCHECK =====
  let mut transcript1 = <E as Engine>::TE::new(b"sha256_chain_bench");
  let t0 = Instant::now();
  let (proof1, r1, evals1) = SumcheckProof::<E>::prove_cubic_with_three_inputs(
    &claim,
    tau.clone(),
    &mut az1,
    &mut bz1,
    &mut cz1,
    &mut transcript1,
  )
  .expect("prove_cubic_with_three_inputs failed");
  let orig_sumcheck_ms = t0.elapsed().as_millis();
  info!(orig_sumcheck_ms, "original sumcheck");

  // ===== SMALL-VALUE SUMCHECK =====
  let mut az2 = MultilinearPolynomial::new(az);
  let mut bz2 = MultilinearPolynomial::new(bz);
  let mut cz2 = MultilinearPolynomial::new(cz);
  let mut transcript2 = <E as Engine>::TE::new(b"sha256_chain_bench");

  let t0 = Instant::now();
  let (proof2, r2, evals2) = SumcheckProof::<E>::prove_cubic_with_three_inputs_small_value(
    &claim,
    tau,
    &az_small,
    &bz_small,
    &mut az2,
    &mut bz2,
    &mut cz2,
    &mut transcript2,
  )
  .expect("prove_cubic_with_three_inputs_small_value failed");
  let small_sumcheck_ms = t0.elapsed().as_millis();
  info!(small_sumcheck_ms, "small-value sumcheck");

  // Verify equivalence
  assert_eq!(r1, r2, "Challenges must match!");
  assert_eq!(proof1, proof2, "Round polynomials must match!");
  assert_eq!(evals1, evals2, "Final evaluations must match!");

  BenchmarkResult {
    chain_length,
    num_vars,
    num_constraints,
    witness_ms,
    extract_ms,
    orig_sumcheck_ms,
    small_sumcheck_ms,
  }
}

fn print_csv_header() {
  println!(
    "chain_length,num_vars,num_constraints,witness_ms,orig_sumcheck_ms,small_sumcheck_ms,total_proving_ms,speedup,witness_pct"
  );
}

fn print_csv_row(result: &BenchmarkResult) {
  let speedup = result.orig_sumcheck_ms as f64 / result.small_sumcheck_ms as f64;

  // Total proving time (excluding one-time setup)
  let total_ms = result.witness_ms + result.extract_ms + result.small_sumcheck_ms;
  let witness_pct = (result.witness_ms as f64 / total_ms as f64) * 100.0;

  println!(
    "{},{},{},{},{},{},{},{:.2},{:.1}",
    result.chain_length,
    result.num_vars,
    result.num_constraints,
    result.witness_ms,
    result.orig_sumcheck_ms,
    result.small_sumcheck_ms,
    total_ms,
    speedup,
    witness_pct
  );
  std::io::stdout().flush().ok();
}

fn main() {
  // Initialize tracing to stderr so CSV output goes to stdout cleanly
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .with_writer(std::io::stderr)
    .init();

  let args = Args::parse();

  // Determine which num_vars values to run (must be even for Algorithm 6)
  let num_vars_list: Vec<usize> = match args.command {
    Some(Command::Single { num_vars }) => {
      assert!(
        num_vars % 2 == 0,
        "num_vars must be even (Algorithm 6 requirement)"
      );
      vec![num_vars]
    }
    Some(Command::RangeSweep { min, max }) => (min..=max).step_by(2).collect(),
    None => vec![16, 18, 20, 22, 24, 26], // default
  };

  // Use a deterministic input
  let input: [u8; 32] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
  ];

  print_csv_header();

  for num_vars in num_vars_list {
    let chain_length = num_vars_to_chain_length(num_vars);
    let result = run_chain_benchmark::<I64Batch21<F>>(input, chain_length, num_vars);
    print_csv_row(&result);
  }
}
