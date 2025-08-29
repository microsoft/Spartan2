use spartan2::{
  provider::PallasMerkleMleEngine as E,
  provider::pcs::merkle_mle_pc::HashMlePCS,
  traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait},
  polys::multilinear::MultilinearPolynomial,
};
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use proptest::prelude::*;

fn rs(rng: &mut StdRng) -> <E as Engine>::Scalar { 
  <E as Engine>::Scalar::from(rng.r#gen::<u64>()) 
}

#[test]
fn mle_roundtrip_fixed() {
  let m = 5usize; let n = 1usize << m;
  let mut rng = StdRng::seed_from_u64(7);
  let poly: Vec<<E as Engine>::Scalar> = (0..n).map(|_| rs(&mut rng)).collect();
  let point: Vec<<E as Engine>::Scalar> = (0..m).map(|_| rs(&mut rng)).collect();

  let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"t", n);
  let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);
  let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();

  let mut tp = <E as Engine>::TE::new(b"t");
  let (eval, arg) = <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tp, &com, &poly, &blind, &point).unwrap();

  let mut tv = <E as Engine>::TE::new(b"t");
  <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tv, &com, &point, &eval, &arg).unwrap();

  assert_eq!(eval, MultilinearPolynomial::new(poly).evaluate(&point));
}

proptest! {
  #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]
  #[test]
  fn mle_roundtrip_random(m in 1usize..=5, seed in any::<u64>()) {
    let n = 1usize << m;
    let mut rng = StdRng::seed_from_u64(seed);
    let poly: Vec<<E as Engine>::Scalar> = (0..n).map(|_| rs(&mut rng)).collect();
    let point: Vec<<E as Engine>::Scalar> = (0..m).map(|_| rs(&mut rng)).collect();

    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"p", n);
    let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();

    let mut tp = <E as Engine>::TE::new(b"p");
    let (eval, arg) = <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tp, &com, &poly, &blind, &point).unwrap();

    let mut tv = <E as Engine>::TE::new(b"p");
    prop_assert!(<HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tv, &com, &point, &eval, &arg).is_ok());
  }
}

#[test]
fn commit_rejects_non_power_of_two() {
  let n = 12usize;
  let (ck, _vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"x", n);
  let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);
  let poly: Vec<<E as Engine>::Scalar> = (0..n).map(|i| <E as Engine>::Scalar::from(i as u64)).collect();
  assert!(<HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).is_err());
}
