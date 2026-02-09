//! Benchmarks for sumcheck with split-eq delayed modular reduction.
//!
//! Run with: cargo bench --bench sumcheck
//! Filter:   cargo bench --bench sumcheck -- "/20"
//! Sizes:    BENCH_SIZES=24,26 cargo bench --bench sumcheck
//! Field:    BENCH_FIELD=pallas cargo bench --bench sumcheck
//!           Options: bn254 (default), pallas, vesta, t256

use criterion::{BatchSize, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ff::Field;
use rand::{rngs::StdRng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use spartan2::{
  big_num::DelayedReduction,
  polys::multilinear::MultilinearPolynomial,
  provider::{Bn254Engine, PallasHyraxEngine, T256HyraxEngine, VestaHyraxEngine},
  sumcheck::SumcheckProof,
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use std::time::Duration;

criterion_group! {
    name = sumcheck;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(20))
        .sample_size(10);
    targets = bench_sumcheck_split_eq
}

criterion_main!(sumcheck);

fn bench_sumcheck_with_engine<E: Engine>(c: &mut Criterion, field_name: &str)
where
  E::Scalar: DelayedReduction<E::Scalar>,
{
  // Read sizes from env, default to 16..=26
  let sizes: Vec<usize> = std::env::var("BENCH_SIZES")
    .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
    .unwrap_or_else(|_| (16..=26).collect());

  let max_vars = *sizes.iter().max().unwrap_or(&26);
  let max_len = 1 << max_vars;

  // Pre-generate random test data at maximum size using seeded RNG for reproducibility
  // Using random data avoids masking branch misprediction or cache effects
  const SEED: u64 = 0xDEADBEEF_CAFEBABE;

  let az: Vec<E::Scalar> = (0..max_len)
    .into_par_iter()
    .map_init(
      || StdRng::seed_from_u64(SEED),
      |rng, _| E::Scalar::random(&mut *rng),
    )
    .collect();

  let bz: Vec<E::Scalar> = (0..max_len)
    .into_par_iter()
    .map_init(
      || StdRng::seed_from_u64(SEED.wrapping_add(1)),
      |rng, _| E::Scalar::random(&mut *rng),
    )
    .collect();

  let cz: Vec<E::Scalar> = (0..max_len)
    .into_par_iter()
    .map_init(
      || StdRng::seed_from_u64(SEED.wrapping_add(2)),
      |rng, _| E::Scalar::random(&mut *rng),
    )
    .collect();

  let mut tau_rng = StdRng::seed_from_u64(SEED.wrapping_add(3));
  let taus: Vec<E::Scalar> = (0..max_vars)
    .map(|_| E::Scalar::random(&mut tau_rng))
    .collect();

  let group_name = format!("Sumcheck/{field_name}");
  let mut group = c.benchmark_group(&group_name);

  for num_vars in sizes {
    let len = 1 << num_vars;

    group.throughput(Throughput::Elements(len as u64));

    group.bench_with_input(
      BenchmarkId::new("split_eq_delayed", num_vars),
      &num_vars,
      |b, &num_vars| {
        b.iter_batched(
          || {
            // Setup: allocate polynomials and transcript (not timed)
            (
              MultilinearPolynomial::new(az[..len].to_vec()),
              MultilinearPolynomial::new(bz[..len].to_vec()),
              MultilinearPolynomial::new(cz[..len].to_vec()),
              taus[..num_vars].to_vec(),
              E::TE::new(b"bench"),
            )
          },
          |(mut poly_az, mut poly_bz, mut poly_cz, tau_vec, mut transcript)| {
            // Only this part is timed
            SumcheckProof::<E>::prove_cubic_with_three_inputs(
              &E::Scalar::ZERO,
              tau_vec,
              &mut poly_az,
              &mut poly_bz,
              &mut poly_cz,
              &mut transcript,
            )
            .unwrap()
          },
          BatchSize::LargeInput,
        );
      },
    );
  }

  group.finish();
}

fn bench_sumcheck_split_eq(c: &mut Criterion) {
  let field = std::env::var("BENCH_FIELD").unwrap_or_else(|_| "bn254".to_string());

  match field.to_lowercase().as_str() {
    "bn254" => bench_sumcheck_with_engine::<Bn254Engine>(c, "bn254"),
    "pallas" => bench_sumcheck_with_engine::<PallasHyraxEngine>(c, "pallas"),
    "vesta" => bench_sumcheck_with_engine::<VestaHyraxEngine>(c, "vesta"),
    "t256" => bench_sumcheck_with_engine::<T256HyraxEngine>(c, "t256"),
    _ => {
      eprintln!(
        "Unknown field '{}'. Options: bn254 (default), pallas, vesta, t256",
        field
      );
      bench_sumcheck_with_engine::<Bn254Engine>(c, "bn254")
    }
  }
}
