//! Benchmarks Spartan's prover for proving SHA-256 with varying sized messages.
//! This code invokes a hand-written SHA-256 gadget from bellpepper.
#![allow(non_snake_case)]
use bellpepper::gadgets::{Assignment, sha256::sha256};
use bellpepper_core::{
  Circuit, ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::{AllocatedNum, Num},
};
use core::{marker::PhantomData, time::Duration};
use criterion::*;
use ff::{PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use spartan2::{R1CSSNARK, provider::T256HyraxEngine, traits::snark::R1CSSNARKTrait};

type E = T256HyraxEngine;

#[derive(Clone, Debug)]
struct Sha256Circuit<Scalar: PrimeField> {
  preimage: Vec<u8>,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> Sha256Circuit<Scalar> {
  pub fn new(preimage: Vec<u8>) -> Self {
    Self {
      preimage,
      _p: PhantomData,
    }
  }
}

impl<Scalar: PrimeField + PrimeFieldBits> Circuit<Scalar> for Sha256Circuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
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

    let hash_bits = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

    let mut hash_fe = Vec::new();
    for (i, hash_bits) in hash_bits.chunks(128_usize).enumerate() {
      let mut num = Num::<Scalar>::zero();
      let mut coeff = Scalar::ONE;
      for bit in hash_bits {
        num = num.add_bool_with_coeff(CS::one(), bit, coeff);

        coeff = coeff.double();
      }

      let hash = AllocatedNum::alloc(cs.namespace(|| format!("input {i}")), || {
        Ok(*num.get_value().get()?)
      })?;

      // num * 1 = hash
      cs.enforce(
        || format!("packing constraint {i}"),
        |_| num.lc(Scalar::ONE),
        |lc| lc + CS::one(),
        |lc| lc + hash.get_variable(),
      );
      hash_fe.push(hash);
    }

    // sanity check with the hasher
    let mut hasher = Sha256::new();
    hasher.update(&self.preimage);
    let hash_result = hasher.finalize();

    let mut s = hash_result
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1u8 == 1u8));

    for b in hash_bits {
      match b {
        Boolean::Is(b) => {
          assert!(s.next().unwrap() == b.get_value().unwrap());
        }
        Boolean::Not(b) => {
          assert!(s.next().unwrap() != b.get_value().unwrap());
        }
        Boolean::Constant(_b) => {
          panic!("Can't reach here")
        }
      }
    }

    hash_fe[0].inputize(cs.namespace(|| "hash output 0"))?;
    hash_fe[1].inputize(cs.namespace(|| "hash output 1"))?;

    Ok(())
  }
}

fn bench_spartan(c: &mut Criterion) {
  // Test vectors
  let circuits = vec![
    Sha256Circuit::new(vec![0u8; 1 << 6]),
    Sha256Circuit::new(vec![0u8; 1 << 7]),
    Sha256Circuit::new(vec![0u8; 1 << 8]),
    Sha256Circuit::new(vec![0u8; 1 << 9]),
    Sha256Circuit::new(vec![0u8; 1 << 10]),
    Sha256Circuit::new(vec![0u8; 1 << 11]),
    Sha256Circuit::new(vec![0u8; 1 << 12]),
    Sha256Circuit::new(vec![0u8; 1 << 13]),
    Sha256Circuit::new(vec![0u8; 1 << 14]),
    Sha256Circuit::new(vec![0u8; 1 << 15]),
    Sha256Circuit::new(vec![0u8; 1 << 16]),
  ];

  for circuit in circuits {
    let mut group = c.benchmark_group(format!(
      "Spartan-Sha256-message-len-{}",
      circuit.preimage.len()
    ));
    group.sample_size(10);

    // Produce public parameters
    let (pk, _) =
      R1CSSNARK::<E>::setup(circuit.clone()).expect("Failed to produce public parameters");

    // generate instance-witness pair
    let (U, W) = R1CSSNARK::<E>::gen_witness(&pk, circuit.clone()).unwrap();

    group.bench_function("Prove", |b| {
      b.iter(|| {
        let res = R1CSSNARK::<E>::prove(black_box(&pk), black_box(&U), black_box(&W));
        assert!(res.is_ok());
      })
    });
    group.finish();
  }
}

criterion_group! {
name = spartan;
config = Criterion::default().warm_up_time(Duration::from_millis(3000));
targets = bench_spartan
}

criterion_main!(spartan);
