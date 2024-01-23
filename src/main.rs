fn main() {
    let circuits = vec![
        Sha256Circuit::new(vec![0u8; 1 << 12]),
    ];

    let (chrome_layer, _guard) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();

    for circuit in circuits {
        let span = tracing::info_span!("SpartanProve-Sha256-message-len", len = circuit.preimage.len());
        let _enter = span.enter();

        run_circuit(circuit);
    }

    drop(_guard);
}

fn run_circuit(circuit: Sha256Circuit<<G as Group>::Scalar>) {
    let start = std::time::Instant::now();

    // produce keys
    let (pk, _vk) =
        SNARK::<G, S, Sha256Circuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

    let _ = SNARK::<G, S, Sha256Circuit<<G as Group>::Scalar>>::prove(
        &pk,
        circuit.clone(),
    );

    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);
}

use bellpepper::gadgets::{sha256::sha256, Assignment};
use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  num::{AllocatedNum, Num},
  Circuit, ConstraintSystem, SynthesisError,
};
use ff::{PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use spartan2::{SNARK, traits::Group};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use std::marker::PhantomData;

type G = pasta_curves::pallas::Point;
type EE = spartan2::provider::hyrax_pc::HyraxEvaluationEngine<G>;
type S = spartan2::spartan::upsnark::R1CSSNARK<G, EE>;

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
