extern crate core;

use criterion::{criterion_group, criterion_main, Criterion};
use ff::Field;
use pasta_curves::Fp;

use spartan2::spartan::polys::eq::EqPolynomial;

fn benchmarks_evaluate_incremental(c: &mut Criterion) {
  let mut group = c.benchmark_group("evaluate_incremental");
  (1..=20).step_by(2).for_each(|i| {
    group.bench_with_input(format!("2^{}", i), &i, |b, &i| {
      let eq_polynomial =
        EqPolynomial::new(vec![Fp::random(&mut rand::thread_rng()); 2usize.pow(i)]);
      b.iter(|| {
        eq_polynomial.evaluate(vec![Fp::random(&mut rand::thread_rng()); 2usize.pow(i)].as_slice())
      });
    });
  });
}

criterion_group!(benches, benchmarks_evaluate_incremental);
criterion_main!(benches);
