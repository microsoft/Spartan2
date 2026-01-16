// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/logup_lookup.rs
//! Example circuit demonstrating split-commitments and challenges with a lookup table
//! using the logup randomized check.
//!
//! The circuit has:
//! - Shared section: A table T containing all values from 0 to 255 (256 entries)
//! - Precommitted section: 
//!   - L: lookup values (1024 entries) claimed to be in the table
//!   - counts: for each table entry, how many times it appears in L (256 entries)
//! - Challenge: c (depends on shared and precommitted sections)
//! - Synthesize: Verifies sum_i {1/(L[i] + c)} = sum_j {counts[j]/(T[j] + c)}
//!
//! Run with: `RUST_LOG=info cargo run --release --example logup_lookup`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::{Field, PrimeField};
use spartan2::{
  provider::T256HyraxEngine,
  spartan::SpartanSNARK,
  traits::{Engine, circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use std::{marker::PhantomData, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

type E = T256HyraxEngine;

const TABLE_SIZE: usize = 256;
const LOOKUP_SIZE: usize = 1024;

#[derive(Clone, Debug)]
struct LookupCircuit<Scalar: PrimeField> {
  // The lookup values (each should be in range 0..256)
  lookup_values: Vec<u64>,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField> LookupCircuit<Scalar> {
  fn new(lookup_values: Vec<u64>) -> Self {
    assert_eq!(lookup_values.len(), LOOKUP_SIZE);
    // Ensure all lookup values are in the table range
    for &val in &lookup_values {
      assert!(val < TABLE_SIZE as u64, "Lookup value must be < {}", TABLE_SIZE);
    }
    Self {
      lookup_values,
      _p: PhantomData,
    }
  }

  // Compute the counts array: how many times each table entry appears in lookup_values
  fn compute_counts(&self) -> Vec<u64> {
    let mut counts = vec![0u64; TABLE_SIZE];
    for &val in &self.lookup_values {
      counts[val as usize] += 1;
    }
    counts
  }
}

impl<E: Engine> SpartanCircuit<E> for LookupCircuit<E::Scalar> {
  fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
    // Return a dummy public value (the size of the table)
    Ok(vec![E::Scalar::from(TABLE_SIZE as u64)])
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // Allocate the table T in the shared section
    // T contains all values from 0 to 255
    let mut table = Vec::with_capacity(TABLE_SIZE);
    for i in 0..TABLE_SIZE {
      let val = E::Scalar::from(i as u64);
      let allocated = AllocatedNum::alloc(cs.namespace(|| format!("table[{}]", i)), || Ok(val))?;
      table.push(allocated);
    }
    Ok(table)
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    let mut result = Vec::new();

    // Add a public input (table size)
    let table_size_scalar = E::Scalar::from(TABLE_SIZE as u64);
    let table_size_public = AllocatedNum::alloc_input(
      cs.namespace(|| "table_size_public"),
      || Ok(table_size_scalar),
    )?;
    // Add a constraint to ensure it's the correct value
    cs.enforce(
      || "table_size_check",
      |lc| lc + table_size_public.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + (table_size_scalar, CS::one()),
    );

    // Allocate the lookup values L
    for (i, &val) in self.lookup_values.iter().enumerate() {
      let scalar_val = E::Scalar::from(val);
      let allocated =
        AllocatedNum::alloc(cs.namespace(|| format!("lookup[{}]", i)), || Ok(scalar_val))?;
      result.push(allocated);
    }

    // Allocate the counts array
    let counts = self.compute_counts();
    for (i, &count) in counts.iter().enumerate() {
      let scalar_count = E::Scalar::from(count);
      let allocated =
        AllocatedNum::alloc(cs.namespace(|| format!("count[{}]", i)), || Ok(scalar_count))?;
      result.push(allocated);
    }

    Ok(result)
  }

  fn num_challenges(&self) -> usize {
    // We need one challenge for the logup check
    1
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    shared: &[AllocatedNum<E::Scalar>],
    precommitted: &[AllocatedNum<E::Scalar>],
    challenges: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    // shared contains the table T
    let table = shared;
    assert_eq!(table.len(), TABLE_SIZE);

    // precommitted contains L (lookup values) followed by counts
    let lookup_values = &precommitted[0..LOOKUP_SIZE];
    let counts = &precommitted[LOOKUP_SIZE..LOOKUP_SIZE + TABLE_SIZE];
    assert_eq!(lookup_values.len(), LOOKUP_SIZE);
    assert_eq!(counts.len(), TABLE_SIZE);

    // Get the challenge value - use ZERO during setup
    let challenge = challenges.and_then(|c| c.first().copied()).unwrap_or(E::Scalar::ZERO);

    // Allocate inverse hints for LHS: h_i = 1/(L[i] + c)
    let mut lhs_inverses = Vec::with_capacity(LOOKUP_SIZE);
    for (i, lookup_val) in lookup_values.iter().enumerate() {
      // Compute the hint value: 1 / (L[i] + c)
      let hint_value = lookup_val
        .get_value()
        .and_then(|v| {
          let denominator = v + challenge;
          Some(denominator.invert().unwrap_or(E::Scalar::ZERO))
        });

      // Allocate the hint
      let hint =
        AllocatedNum::alloc(cs.namespace(|| format!("lhs_inv[{}]", i)), || {
          hint_value.ok_or(SynthesisError::AssignmentMissing)
        })?;

      // Add constraint: hint * (lookup_val + challenge) = 1
      // Rewrite as: hint * lookup_val = 1 - hint * challenge
      cs.enforce(
        || format!("lhs_inv_check[{}]", i),
        |lc| lc + hint.get_variable(),
        |lc| lc + lookup_val.get_variable(),
        |lc| lc + CS::one() - (challenge, hint.get_variable()),
      );

      lhs_inverses.push(hint);
    }

    // Allocate inverse hints for RHS: g_j = counts[j]/(T[j] + c)
    let mut rhs_numerators = Vec::with_capacity(TABLE_SIZE);
    for (j, (table_val, count)) in table.iter().zip(counts.iter()).enumerate() {
      // Compute the hint value: counts[j] / (T[j] + c)
      let hint_value = table_val
        .get_value()
        .and_then(|t| count.get_value().map(|c_val| (t, c_val)))
        .and_then(|(t, c_val)| {
          let denominator = t + challenge;
          let inv = denominator.invert().unwrap_or(E::Scalar::ZERO);
          Some(c_val * inv)
        });

      // Allocate the hint
      let hint =
        AllocatedNum::alloc(cs.namespace(|| format!("rhs_num[{}]", j)), || {
          hint_value.ok_or(SynthesisError::AssignmentMissing)
        })?;

      // Add constraint: hint * (table_val + challenge) = count
      // Rewrite as: hint * table_val = count - hint * challenge
      cs.enforce(
        || format!("rhs_num_check[{}]", j),
        |lc| lc + hint.get_variable(),
        |lc| lc + table_val.get_variable(),
        |lc| lc + count.get_variable() - (challenge, hint.get_variable()),
      );

      rhs_numerators.push(hint);
    }

    // Now check that sum of LHS inverses equals sum of RHS numerators
    // sum_i h_i = sum_j g_j
    // Allocate a variable for the LHS sum
    let lhs_sum_value = lhs_inverses
      .iter()
      .try_fold(E::Scalar::ZERO, |acc, inv| {
        inv.get_value().map(|v| acc + v)
      });
    let lhs_sum = AllocatedNum::alloc(cs.namespace(|| "lhs_sum"), || {
      lhs_sum_value.ok_or(SynthesisError::AssignmentMissing)
    })?;

    // Constrain lhs_sum = sum of lhs_inverses
    cs.enforce(
      || "lhs_sum_constraint",
      |lc| {
        lhs_inverses
          .iter()
          .fold(lc, |lc, inv| lc + inv.get_variable())
      },
      |lc| lc + CS::one(),
      |lc| lc + lhs_sum.get_variable(),
    );

    // Allocate a variable for the RHS sum
    let rhs_sum_value = rhs_numerators
      .iter()
      .try_fold(E::Scalar::ZERO, |acc, num| {
        num.get_value().map(|v| acc + v)
      });
    let rhs_sum = AllocatedNum::alloc(cs.namespace(|| "rhs_sum"), || {
      rhs_sum_value.ok_or(SynthesisError::AssignmentMissing)
    })?;

    // Constrain rhs_sum = sum of rhs_numerators
    cs.enforce(
      || "rhs_sum_constraint",
      |lc| {
        rhs_numerators
          .iter()
          .fold(lc, |lc, num| lc + num.get_variable())
      },
      |lc| lc + CS::one(),
      |lc| lc + rhs_sum.get_variable(),
    );

    // Final constraint: lhs_sum = rhs_sum
    cs.enforce(
      || "equality_check",
      |lc| lc + lhs_sum.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + rhs_sum.get_variable(),
    );

    Ok(())
  }
}

fn main() {
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .init();

  // Create a lookup circuit with random lookup values
  // For simplicity, we'll use a deterministic pattern
  let mut lookup_values = Vec::with_capacity(LOOKUP_SIZE);
  for i in 0..LOOKUP_SIZE {
    // Use a pattern that ensures all table entries are used at least once
    // and some multiple times
    lookup_values.push((i % TABLE_SIZE) as u64);
  }

  let circuit = LookupCircuit::<<E as Engine>::Scalar>::new(lookup_values);

  let root_span = info_span!("logup_lookup").entered();
  info!("======= Logup Lookup Circuit =======");
  info!("Table size: {}", TABLE_SIZE);
  info!("Lookup size: {}", LOOKUP_SIZE);

  // SETUP
  let t0 = Instant::now();
  let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup failed");
  let setup_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = setup_ms, "setup");

  // PREPARE
  let t0 = Instant::now();
  let prep_snark =
    SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).expect("prep_prove failed");
  let prep_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = prep_ms, "prep_prove");

  // PROVE
  let t0 = Instant::now();
  let proof =
    SpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep_snark, true).expect("prove failed");
  let prove_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = prove_ms, "prove");

  // VERIFY
  let t0 = Instant::now();
  proof.verify(&vk).expect("verify errored");
  let verify_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = verify_ms, "verify");

  // Summary
  info!(
    "SUMMARY table_size={}, lookup_size={}, setup={} ms, prep_prove={} ms, prove={} ms, verify={} ms",
    TABLE_SIZE, LOOKUP_SIZE, setup_ms, prep_ms, prove_ms, verify_ms
  );
  drop(root_span);

  info!("Logup lookup circuit verification PASSED!");
}
