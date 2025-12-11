//! Benchmark comparing Hyrax-PC vs Dory-PC on BLS12-381
//!
//! Run with: cargo bench --bench pcs_comparison --features dory
//!
//! This benchmark validates:
//! 1. Both PCS implementations produce valid proofs
//! 2. Verification returns correct public outputs
//! 3. Tampered proofs are rejected (soundness sanity check)

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ff::Field;
use spartan2::{
    provider::{BLS12381HyraxEngine, BLS12381DoryEngine},
    spartan::SpartanSNARK,
    traits::{Engine, circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};

// Type aliases for cleaner code
type HyraxScalar = <BLS12381HyraxEngine as Engine>::Scalar;
type DoryScalar = <BLS12381DoryEngine as Engine>::Scalar;

/// Expected public output: x^3 + x + 5 where x=2 → 8 + 2 + 5 = 15
const EXPECTED_OUTPUT: u64 = 15;

/// A cubic circuit: x^3 + x + 5 = y
#[derive(Clone, Debug, Default)]
struct CubicCircuit;

impl<E: Engine> SpartanCircuit<E> for CubicCircuit {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
        Ok(vec![E::Scalar::from(15u64)])
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
        &self,
        _cs: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
        Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
        &self,
        _cs: &mut CS,
        _shared: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
        Ok(vec![])
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<E::Scalar>],
        _precommitted: &[AllocatedNum<E::Scalar>],
        _challenges: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
        // x^3 + x + 5 = y, where x = 2
        let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
            Ok(E::Scalar::ONE + E::Scalar::ONE)
        })?;
        let x_sq = x.square(cs.namespace(|| "x_sq"))?;
        let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
        let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
            Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + E::Scalar::from(5u64))
        })?;

        cs.enforce(
            || "y = x^3 + x + 5",
            |lc| {
                lc + x_cu.get_variable()
                    + x.get_variable()
                    + CS::one()
                    + CS::one()
                    + CS::one()
                    + CS::one()
                    + CS::one()
            },
            |lc| lc + CS::one(),
            |lc| lc + y.get_variable(),
        );

        let _ = y.inputize(cs.namespace(|| "output"));
        Ok(())
    }
}

/// Validate that both PCS implementations are correct before benchmarking

fn validate_implementations() {
    let circuit = CubicCircuit;

    // ============ Hyrax Validation ============
    {
        type E = BLS12381HyraxEngine;
        type S = SpartanSNARK<E>;
        
        let (pk, vk) = S::setup(circuit.clone()).expect("Hyrax setup should succeed");
        let prep = S::prep_prove(&pk, circuit.clone(), false).expect("Hyrax prep should succeed");
        let proof = S::prove(&pk, circuit.clone(), &prep, false).expect("Hyrax prove should succeed");
        
        // Verify returns correct public output
        let public_io = proof.verify(&vk).expect("Hyrax verify should succeed");
        assert_eq!(
            public_io.len(), 1,
            "Hyrax: Expected 1 public output, got {}", public_io.len()
        );
        assert_eq!(
            public_io[0], HyraxScalar::from(EXPECTED_OUTPUT),
            "Hyrax: Public output mismatch"
        );
        
        // Soundness check: serialize, tamper, deserialize should fail verify
        let proof_bytes = bincode::serialize(&proof).expect("serialize");
        if proof_bytes.len() > 100 {
            let mut tampered = proof_bytes.clone();
            // Tamper with middle of proof (avoiding headers)
            tampered[proof_bytes.len() / 2] ^= 0xFF;
            if let Ok(tampered_proof) = bincode::deserialize::<S>(&tampered) {
                // If deserialization succeeds, verification MUST fail
                assert!(
                    tampered_proof.verify(&vk).is_err(),
                    "Hyrax: Tampered proof should NOT verify"
                );
            }
            // If deserialization fails, that's also acceptable (proof integrity)
        }
        
        println!("✓ Hyrax validation passed (correctness + soundness)");
    }

    // ============ Dory Validation ============
    {
        type E = BLS12381DoryEngine;
        type S = SpartanSNARK<E>;
        
        let (pk, vk) = S::setup(circuit.clone()).expect("Dory setup should succeed");
        let prep = S::prep_prove(&pk, circuit.clone(), false).expect("Dory prep should succeed");
        let proof = S::prove(&pk, circuit.clone(), &prep, false).expect("Dory prove should succeed");
        
        // Verify returns correct public output
        let public_io = proof.verify(&vk).expect("Dory verify should succeed");
        assert_eq!(
            public_io.len(), 1,
            "Dory: Expected 1 public output, got {}", public_io.len()
        );
        assert_eq!(
            public_io[0], DoryScalar::from(EXPECTED_OUTPUT),
            "Dory: Public output mismatch"
        );
        
        // Soundness check
        let proof_bytes = bincode::serialize(&proof).expect("serialize");
        if proof_bytes.len() > 100 {
            let mut tampered = proof_bytes.clone();
            tampered[proof_bytes.len() / 2] ^= 0xFF;
            if let Ok(tampered_proof) = bincode::deserialize::<S>(&tampered) {
                assert!(
                    tampered_proof.verify(&vk).is_err(),
                    "Dory: Tampered proof should NOT verify"
                );
            }
        }
        
        println!("✓ Dory validation passed (correctness + soundness)");
    }

    println!("\n=== Both implementations validated. Starting benchmarks... ===\n");
}

/// Benchmark setup phase
fn bench_setup(c: &mut Criterion) {
    // Run validation ONCE before any benchmarks
    validate_implementations();
    let mut group = c.benchmark_group("PCS Setup (BLS12-381)");
    let circuit = CubicCircuit;

    group.bench_function("Hyrax", |b| {
        b.iter(|| {
            type S = SpartanSNARK<BLS12381HyraxEngine>;
            let _ = black_box(S::setup(circuit.clone()).unwrap());
        })
    });

    group.bench_function("Dory", |b| {
        b.iter(|| {
            type S = SpartanSNARK<BLS12381DoryEngine>;
            let _ = black_box(S::setup(circuit.clone()).unwrap());
        })
    });

    group.finish();
}

/// Benchmark prove phase
fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCS Prove (BLS12-381)");
    let circuit = CubicCircuit;

    // Setup Hyrax
    type EHyrax = BLS12381HyraxEngine;
    type SHyrax = SpartanSNARK<EHyrax>;
    let (pk_h, _vk_h) = SHyrax::setup(circuit.clone()).unwrap();
    let prep_h = SHyrax::prep_prove(&pk_h, circuit.clone(), false).unwrap();

    group.bench_function("Hyrax", |b| {
        b.iter(|| {
            let _ = black_box(SHyrax::prove(&pk_h, circuit.clone(), &prep_h, false).unwrap());
        })
    });

    // Setup Dory
    type EDory = BLS12381DoryEngine;
    type SDory = SpartanSNARK<EDory>;
    let (pk_d, _vk_d) = SDory::setup(circuit.clone()).unwrap();
    let prep_d = SDory::prep_prove(&pk_d, circuit.clone(), false).unwrap();

    group.bench_function("Dory", |b| {
        b.iter(|| {
            let _ = black_box(SDory::prove(&pk_d, circuit.clone(), &prep_d, false).unwrap());
        })
    });

    group.finish();
}

/// Benchmark verify phase
fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCS Verify (BLS12-381)");
    let circuit = CubicCircuit;

    // Setup and prove with Hyrax
    type EHyrax = BLS12381HyraxEngine;
    type SHyrax = SpartanSNARK<EHyrax>;
    let (pk_h, vk_h) = SHyrax::setup(circuit.clone()).unwrap();
    let prep_h = SHyrax::prep_prove(&pk_h, circuit.clone(), false).unwrap();
    let proof_h = SHyrax::prove(&pk_h, circuit.clone(), &prep_h, false).unwrap();

    group.bench_function("Hyrax", |b| {
        b.iter(|| {
            let result = proof_h.verify(&vk_h).expect("Hyrax verify failed in bench");
            // Ensure we're actually verifying, not just measuring overhead
            assert_eq!(result[0], HyraxScalar::from(EXPECTED_OUTPUT));
            black_box(result)
        })
    });

    // Setup and prove with Dory
    type EDory = BLS12381DoryEngine;
    type SDory = SpartanSNARK<EDory>;
    let (pk_d, vk_d) = SDory::setup(circuit.clone()).unwrap();
    let prep_d = SDory::prep_prove(&pk_d, circuit.clone(), false).unwrap();
    let proof_d = SDory::prove(&pk_d, circuit.clone(), &prep_d, false).unwrap();

    group.bench_function("Dory", |b| {
        b.iter(|| {
            let result = proof_d.verify(&vk_d).expect("Dory verify failed in bench");
            assert_eq!(result[0], DoryScalar::from(EXPECTED_OUTPUT));
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark full pipeline (setup + prove + verify)
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full Pipeline (BLS12-381)");
    let circuit = CubicCircuit;

    group.bench_function("Hyrax", |b| {
        b.iter(|| {
            type S = SpartanSNARK<BLS12381HyraxEngine>;
            let (pk, vk) = S::setup(circuit.clone()).unwrap();
            let prep = S::prep_prove(&pk, circuit.clone(), false).unwrap();
            let proof = S::prove(&pk, circuit.clone(), &prep, false).unwrap();
            let result = proof.verify(&vk).expect("Hyrax full pipeline verify failed");
            assert_eq!(result[0], HyraxScalar::from(EXPECTED_OUTPUT));
            black_box(result)
        })
    });

    group.bench_function("Dory", |b| {
        b.iter(|| {
            type S = SpartanSNARK<BLS12381DoryEngine>;
            let (pk, vk) = S::setup(circuit.clone()).unwrap();
            let prep = S::prep_prove(&pk, circuit.clone(), false).unwrap();
            let proof = S::prove(&pk, circuit.clone(), &prep, false).unwrap();
            let result = proof.verify(&vk).expect("Dory full pipeline verify failed");
            assert_eq!(result[0], DoryScalar::from(EXPECTED_OUTPUT));
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_setup,
    bench_prove,
    bench_verify,
    bench_full_pipeline,
);
criterion_main!(benches);

