// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/sha256_neutronnova.rs
//! Measure NeutronNova {setup, prep_prove, prove, verify} times for a batch
//! of 32 SHA-256 step circuits folded together with a core SHA-256 circuit.
//!
//! Run with: `RUST_LOG=info cargo bench --bench sha256_neutronnova`
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
use ff::Field;
use spartan2::{
  neutronnova_zk::NeutronNovaZkSNARK,
  provider::T256HyraxEngine,
  traits::{Engine, circuit::SpartanCircuit},
};
use std::{marker::PhantomData, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

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

fn main() {
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .init();

  let num_steps = 32;
  let msg_len = 64;
  let total_bytes = num_steps * msg_len;

  let root_span = info_span!("bench", num_steps, total_bytes).entered();
  info!(
    "======= NeutronNova: {} step circuits, total_hashed={} bytes =======",
    num_steps, total_bytes
  );

  let step_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);
  let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);

  // SETUP
  let t0 = Instant::now();
  let (pk, vk) =
    NeutronNovaZkSNARK::<E>::setup(&step_circuit, &core_circuit, num_steps).expect("setup failed");
  let setup_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = setup_ms, "setup");

  let step_circuits: Vec<_> = (0..num_steps)
    .map(|i| Sha256Circuit::<E>::new(vec![i as u8; msg_len]))
    .collect();
  let core_circuit = Sha256Circuit::<E>::new(vec![0u8; msg_len]);

  // PREP PROVE
  let t0 = Instant::now();
  let prep_snark = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &step_circuits, &core_circuit, true)
    .expect("prep_prove failed");
  let prep_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = prep_ms, "prep_prove");

  // PROVE
  let t0 = Instant::now();
  let (proof, _prep_snark) =
    NeutronNovaZkSNARK::<E>::prove(&pk, &step_circuits, &core_circuit, prep_snark, true)
      .expect("prove failed");
  let prove_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = prove_ms, "prove");

  // VERIFY
  let t0 = Instant::now();
  proof.verify(&vk, num_steps).expect("verify failed");
  let verify_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = verify_ms, "verify");

  // Summary
  info!(
    "SUMMARY steps={}, total_hashed={}B, setup={} ms, prep_prove={} ms, prove={} ms, verify={} ms",
    num_steps, total_bytes, setup_ms, prep_ms, prove_ms, verify_ms
  );
  drop(root_span);
}
