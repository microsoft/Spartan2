// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! benches/sha256_neutronnova.rs
//! Criterion benchmarks for NeutronNova {setup, prep_prove, prove, verify}
//! on a batch of 32 SHA-256 step circuits (2048 bytes total).
//!
//! Run with: `RUSTFLAGS="-C target-cpu=native" cargo bench --bench sha256_neutronnova`
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper::gadgets::sha256::sha256;
use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use ff::Field;
use spartan2::{
  neutronnova_zk::NeutronNovaZkSNARK,
  provider::T256HyraxEngine,
  traits::{Engine, circuit::SpartanCircuit},
};
use std::{marker::PhantomData, time::Duration};

type E = T256HyraxEngine;

#[derive(Clone, Debug)]
struct Sha256Circuit<Eng: Engine> {
  preimage: Vec<u8>,
  _p: PhantomData<Eng>,
}

impl<Eng: Engine> Sha256Circuit<Eng> {
  fn new(preimage: Vec<u8>) -> Self {
    Self {
      preimage,
      _p: PhantomData,
    }
  }
}

impl<Eng: Engine> SpartanCircuit<Eng> for Sha256Circuit<Eng> {
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
    let bit_values: Vec<_> = self
      .preimage
      .clone()
      .into_iter()
      .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
      .map(Some)
      .collect();
    assert_eq!(bit_values.len(), self.preimage.len() * 8);

    let preimage_bits = bit_values
      .into_iter()
      .enumerate()
      .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
      .map(|b| b.map(Boolean::from))
      .collect::<Result<Vec<_>, _>>()?;

    let _ = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

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

/// Thread counts to benchmark. Override with BENCH_THREADS env var (comma-separated).
fn thread_counts() -> Vec<usize> {
  if let Ok(val) = std::env::var("BENCH_THREADS") {
    val
      .split(',')
      .filter_map(|s| s.trim().parse().ok())
      .collect()
  } else {
    vec![1, 4, 8, 16]
  }
}

fn neutronnova_benches(c: &mut Criterion) {
  let num_steps = 32;
  let msg_len = 64;
  let total_bytes = num_steps * msg_len;
  let thread_counts = thread_counts();

  // Report proof size once (outside measurements)
  {
    let step_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
    let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
    let (pk, _vk) =
      NeutronNovaZkSNARK::<E>::setup(&step_circuit, &core_circuit, num_steps).unwrap();
    let step_circuits: Vec<_> = (0..num_steps)
      .map(|i| Sha256Circuit::<E>::new(vec![i as u8; msg_len]))
      .collect();
    let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
    let prep =
      NeutronNovaZkSNARK::<E>::prep_prove(&pk, &step_circuits, &core_circuit, true).unwrap();
    let (proof, _) =
      NeutronNovaZkSNARK::<E>::prove(&pk, &step_circuits, &core_circuit, prep, true).unwrap();
    let proof_bytes = bincode::serialize(&proof).unwrap();
    println!(
      "NeutronNova SHA-256 steps={} total_hashed={}B: proof_size={} bytes",
      num_steps,
      total_bytes,
      proof_bytes.len()
    );
  }

  let mut g = c.benchmark_group("neutronnova_sha256");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(10));
  g.throughput(Throughput::Bytes(total_bytes as u64));

  for &nthreads in &thread_counts {
    let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(nthreads)
      .build()
      .expect("failed to build rayon pool");

    g.bench_function(format!("setup/{total_bytes}/t{nthreads}"), |b| {
      b.iter(|| {
        pool.install(|| {
          let step_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
          let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
          let _ =
            NeutronNovaZkSNARK::<E>::setup(&step_circuit, &core_circuit, num_steps).unwrap();
        });
      });
    });

    g.bench_function(format!("prep_prove/{total_bytes}/t{nthreads}"), |b| {
      b.iter_batched(
        || {
          pool.install(|| {
            let step_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
            let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
            NeutronNovaZkSNARK::<E>::setup(&step_circuit, &core_circuit, num_steps)
              .unwrap()
              .0
          })
        },
        |pk| {
          pool.install(|| {
            let step_circuits: Vec<_> = (0..num_steps)
              .map(|i| Sha256Circuit::<E>::new(vec![i as u8; msg_len]))
              .collect();
            let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
            let _ =
              NeutronNovaZkSNARK::<E>::prep_prove(&pk, &step_circuits, &core_circuit, true)
                .unwrap();
          });
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("prove/{total_bytes}/t{nthreads}"), |b| {
      b.iter_batched(
        || {
          pool.install(|| {
            let step_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
            let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
            let (pk, _vk) =
              NeutronNovaZkSNARK::<E>::setup(&step_circuit, &core_circuit, num_steps).unwrap();
            let step_circuits: Vec<_> = (0..num_steps)
              .map(|i| Sha256Circuit::<E>::new(vec![i as u8; msg_len]))
              .collect();
            let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
            let prep =
              NeutronNovaZkSNARK::<E>::prep_prove(&pk, &step_circuits, &core_circuit, true)
                .unwrap();
            // Warm-up prove
            let (_proof, prep_back) =
              NeutronNovaZkSNARK::<E>::prove(&pk, &step_circuits, &core_circuit, prep, true)
                .unwrap();
            (pk, step_circuits, core_circuit, prep_back)
          })
        },
        |(pk, step_circuits, core_circuit, prep)| {
          pool.install(|| {
            let _ =
              NeutronNovaZkSNARK::<E>::prove(&pk, &step_circuits, &core_circuit, prep, true)
                .unwrap();
          });
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{total_bytes}/t{nthreads}"), |b| {
      b.iter_batched(
        || {
          pool.install(|| {
            let step_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
            let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
            let (pk, vk) =
              NeutronNovaZkSNARK::<E>::setup(&step_circuit, &core_circuit, num_steps).unwrap();
            let step_circuits: Vec<_> = (0..num_steps)
              .map(|i| Sha256Circuit::<E>::new(vec![i as u8; msg_len]))
              .collect();
            let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
            let prep =
              NeutronNovaZkSNARK::<E>::prep_prove(&pk, &step_circuits, &core_circuit, true)
                .unwrap();
            let (proof, _) =
              NeutronNovaZkSNARK::<E>::prove(&pk, &step_circuits, &core_circuit, prep, true)
                .unwrap();
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
  g.finish();
}

criterion_group!(benches, neutronnova_benches);
criterion_main!(benches);
