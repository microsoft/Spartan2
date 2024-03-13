extern crate core;

use criterion::{criterion_group, criterion_main, Criterion};
use ff::Field;
use pasta_curves::Fp;

use spartan2::spartan::polys::eq::EqPolynomial;

fn benchmarks_evaluate_incremental(c: &mut Criterion) {
  let mut group = c.benchmark_group("evaluate_incremental");
  (1..=20).step_by(2).for_each(|i| {
    let random_point: Vec<Fp> = (0..2usize.pow(i))
      .map(|_| Fp::random(&mut rand::thread_rng()))
      .collect();
    let random_polynomial = EqPolynomial::new(
      (0..2usize.pow(i))
        .map(|_| Fp::random(&mut rand::thread_rng()))
        .collect(),
    );
    group.bench_with_input(format!("2^{}", i), &i, |b, &_i| {
      b.iter(|| {
        random_polynomial.evaluate(random_point.as_slice());
      });
    });
  });
}

criterion_group!(benches, benchmarks_evaluate_incremental);
criterion_main!(benches);
