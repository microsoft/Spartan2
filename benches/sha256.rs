//! Benchmarks Spartan's prover for proving SHA-256 with varying sized messages.
//! This code invokes a hand-written SHA-256 gadget from bellman/bellperson.
//! It also uses code from bellman/bellperson to compare circuit-generated digest with sha2 crate's output
#![allow(non_snake_case)]
use bellperson::gadgets::{sha256::sha256, Assignment};
use bellperson::{gadgets::{
  boolean::{AllocatedBit, Boolean},
  num::{AllocatedNum, Num}},
  Circuit,ConstraintSystem, SynthesisError,
};
use core::time::Duration;
use criterion::*;
use ff::{PrimeField, PrimeFieldBits};
use spartan2::{traits::Group,SNARK};
use sha2::{Digest, Sha256};
use std::marker::PhantomData;

type G = pasta_curves::pallas::Point;
type EE = spartan2::provider::hyrax_pc::HyraxEvaluationEngine<G>;
type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G, EE>;

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

impl<Scalar: PrimeField> Circuit<Scalar> for Sha256Circuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(
    self, 
    cs: &mut CS,
  ) -> Result<(), SynthesisError> {
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

    for (i, hash_bits) in hash_bits.chunks(256_usize).enumerate() {
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
    Ok(())
  }
}

criterion_group! {
name = snark;
config = Criterion::default().warm_up_time(Duration::from_millis(3000));
targets = bench_snark
}

criterion_main!(snark);

fn bench_snark(c: &mut Criterion) {
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
      "SpartanProve-Sha256-message-len-{}",
      circuit.preimage.len()
    ));
    group.sample_size(10);

    // produce keys
    let (pk, _vk) =
      SNARK::<G, S, Sha256Circuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

    group.bench_function("Prove", |b| {
      let res = b.iter(|| {
        SNARK::<G, S, Sha256Circuit<<G as Group>::Scalar>>::prove(
          black_box(&pk),
          black_box(circuit.clone()),
        );
      });
    });
    group.finish();
  }
}
