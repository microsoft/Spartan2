//! examples/sha256_bls12_381_dory.rs
//!
//! Demonstrates SHA-256 circuit proving using BLS12-381 curve with Dory-PC backend.
//! This example showcases the BLS12381DoryEngine implementation.
//!
//! Based on the original sha256.rs example from Spartan2.
//!
//! Run with: `RUST_LOG=info cargo run --release --example sha256_bls12_381_dory --features dory`

#[cfg(feature = "dory")]
mod inner {
    use bellpepper::gadgets::sha256::sha256;
    use bellpepper_core::{
        boolean::{AllocatedBit, Boolean},
        num::AllocatedNum,
        ConstraintSystem, SynthesisError,
    };
    use ff::{Field, PrimeField, PrimeFieldBits};
    use sha2::{Digest, Sha256};
    use spartan2::{
        provider::BLS12381DoryEngine,
        spartan::SpartanSNARK,
        traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait, Engine},
    };
    use std::{marker::PhantomData, time::Instant};
    use tracing::{info, info_span};
    use tracing_subscriber::EnvFilter;

    type E = BLS12381DoryEngine;

    #[derive(Clone, Debug)]
    struct Sha256Circuit<Scalar: PrimeField> {
        preimage: Vec<u8>,
        _p: PhantomData<Scalar>,
    }

    impl<Scalar: PrimeField + PrimeFieldBits> Sha256Circuit<Scalar> {
        fn new(preimage: Vec<u8>) -> Self {
            Self {
                preimage,
                _p: PhantomData,
            }
        }
    }

    impl<E: Engine> SpartanCircuit<E> for Sha256Circuit<E::Scalar> {
        fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
            let mut hasher = Sha256::new();
            hasher.update(&self.preimage);
            let hash = hasher.finalize();
            let hash_scalars: Vec<<E as Engine>::Scalar> = hash
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
            let bit_values: Vec<_> = self
                .preimage
                .clone()
                .into_iter()
                .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1 == 1))
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

            let mut hasher = Sha256::new();
            hasher.update(&self.preimage);
            let expected = hasher.finalize();

            let mut expected_bits = expected
                .iter()
                .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1));

            for b in &hash_bits {
                match b {
                    Boolean::Is(bit) => {
                        assert_eq!(expected_bits.next().unwrap(), bit.get_value().unwrap())
                    }
                    Boolean::Not(bit) => {
                        assert_ne!(expected_bits.next().unwrap(), bit.get_value().unwrap())
                    }
                    Boolean::Constant(_) => unreachable!(),
                }
            }

            for (i, bit) in hash_bits.iter().enumerate() {
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

    pub fn run() {
        tracing_subscriber::fmt()
            .with_target(false)
            .with_ansi(true)
            .with_env_filter(EnvFilter::from_default_env())
            .init();

        println!("BLS12-381 + Dory-PC SHA-256 Example");
        println!("===================================");
        println!();

        // Use smaller message for demonstration (Dory setup is fast but prove is slower)
        let circuits: Vec<_> = (8..=9)
            .map(|k| Sha256Circuit::<<E as Engine>::Scalar>::new(vec![0u8; 1 << k]))
            .collect();

        for circuit in circuits {
            let msg_len = circuit.preimage.len();
            let root_span = info_span!("bench", msg_len).entered();
            info!("Message length: {} bytes", msg_len);

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
            proof.verify(&vk).expect("verify failed");
            let verify_ms = t0.elapsed().as_millis();
            info!(elapsed_ms = verify_ms, "verify");

            // Summary
            println!();
            println!("Results for {} byte message:", msg_len);
            println!("  Setup:      {} ms", setup_ms);
            println!("  Prep:       {} ms", prep_ms);
            println!("  Prove:      {} ms", prove_ms);
            println!("  Verify:     {} ms", verify_ms);
            println!();

            drop(root_span);
        }

        println!("All proofs verified successfully.");
    }
}

#[cfg(feature = "dory")]
fn main() {
    inner::run();
}

#[cfg(not(feature = "dory"))]
fn main() {
    eprintln!("This example requires the 'dory' feature.");
    eprintln!("Run with: cargo run --release --example sha256_bls12_381_dory --features dory");
    std::process::exit(1);
}

