//! This module implements Spartan's traits using the following configuration:
//! `CommitmentEngine` with Pedersen's commitments
//! `Group` with pasta curves and BN256/Grumpkin
//! `RO` traits with Poseidon
//! `EvaluationEngine` with an IPA-based polynomial evaluation argument

// pub mod bn256_grumpkin; // Replaced by ark-bls12-381, TODO: Fix later?
pub mod ipa_pc;
pub mod keccak;
// pub mod pasta; // Replaced by ark-bls12-381, TODO: Fix later?
pub mod ark;
pub mod ark_serde;
pub mod pedersen;
// pub mod secp_secq; // Replaced by ark-bls12-381, TODO: Fix later?

use ark_ec::AffineRepr;
use ark_ff::{AdditiveGroup, PrimeField};
use num_traits::Zero;

// TODO: Replaced by VariableMSM from Arkworks, remove?
/// Native implementation of fast multiexp
/// Adapted from zcash/halo2
fn cpu_multiexp_serial<C>(coeffs: &[C::ScalarField], bases: &[C], acc: &mut C::Group)
where
  C: AffineRepr,
  C::ScalarField: PrimeField,
{
  let coeffs: Vec<_> = coeffs.iter().map(|a| *a).collect();

  let c = if bases.len() < 4 {
    1
  } else if bases.len() < 32 {
    3
  } else {
    f64::from(bases.len() as u32).ln().ceil() as usize
  };

  /// Returns the `c`-bit integer value at the specified segment from the given bytes.
  fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::BigInt) -> usize {
    // Calculate the bit position and check bounds
    let skip_bits = segment * c;
    if skip_bits / 8 >= bytes.as_ref().len() {
      return 0;
    }
    // Process up to 8 bytes and extract relevant bits
    let mut tmp = 0u64;
    for (i, &byte) in bytes.as_ref()[skip_bits / 8..].iter().take(8).enumerate() {
      tmp |= (byte as u64) << (i * 8);
    }
    (tmp >> (skip_bits % 8)) as usize & ((1 << c) - 1)
  }

  let segments = (256 / c) + 1;

  for current_segment in (0..segments).rev() {
    for _ in 0..c {
      *acc = acc.double();
    }

    #[derive(Clone, Copy)]
    enum Bucket<C: AffineRepr> {
      None,
      Affine(C),
      Projective(C::Group),
    }

    impl<C: AffineRepr> Bucket<C> {
      fn add_assign(&mut self, other: &C) {
        *self = match *self {
          Bucket::None => Bucket::Affine(*other),
          Bucket::Affine(a) => Bucket::Projective(a + *other),
          Bucket::Projective(mut a) => {
            a += *other;
            Bucket::Projective(a)
          }
        }
      }

      fn add(self, mut other: C::Group) -> C::Group {
        match self {
          Bucket::None => other,
          Bucket::Affine(a) => {
            other += a;
            other
          }
          Bucket::Projective(a) => other + a,
        }
      }
    }

    let mut buckets: Vec<Bucket<C>> = vec![Bucket::None; (1 << c) - 1];

    for (coeff, base) in coeffs.iter().zip(bases.iter()) {
      let coeff_bigint = coeff.into_bigint();
      let coeff = get_at::<C::ScalarField>(current_segment, c, &coeff_bigint);
      if coeff != 0 {
        buckets[coeff - 1].add_assign(base);
      }
    }

    // Summation by parts
    // e.g. 3a + 2b + 1c = a +
    //                    (a) + b +
    //                    ((a) + b) + c
    let mut running_sum = C::Group::zero();
    for exp in buckets.into_iter().rev() {
      running_sum = exp.add(running_sum);
      *acc += running_sum;
    }
  }
}

/// Curve ops
#[macro_export]
macro_rules! impl_traits {
  (
    $name:ident,
    $name_compressed:ident,
    $name_curve:ident,
    $name_curve_affine:ident,
    $order_str:literal
  ) => {
    impl Group for $name::Point {
      type Base = $name::Base;
      type Scalar = $name::Scalar;
      type CompressedGroupElement = $name_compressed;
      type PreprocessedGroupElement = $name::Affine;
      type TE = Keccak256Transcript<Self>;
      type CE = CommitmentEngine<Self>;

      fn vartime_multiscalar_mul<C: AffineRepr>(scalars: &[Self::Scalar], bases: &[C]) -> C::Group {
        VariableBaseMSM::multi_scalar_mul(bases, scalars)
      }

      fn preprocessed(&self) -> Self::PreprocessedGroupElement {
        self.to_affine()
      }

      fn compress(&self) -> Self::CompressedGroupElement {
        self.to_bytes()
      }

      fn from_label<C: AffineRepr>(label: &'static [u8], n: usize) -> Vec<C> {
        let mut shake = Shake256::default();
        shake.update(label);
        let mut reader = shake.finalize_xof();
        let mut uniform_bytes_vec = Vec::new();
        for _ in 0..n {
          let mut uniform_bytes = [0u8; 32];
          reader.read_exact(&mut uniform_bytes).unwrap();
          uniform_bytes_vec.push(uniform_bytes);
        }
        let gens_proj: Vec<$name_curve> = (0..n)
          .into_par_iter()
          .map(|i| {
            let hash = $name_curve::hash_to_curve("from_uniform_bytes");
            hash(&uniform_bytes_vec[i])
          })
          .collect();

        let num_threads = rayon::current_num_threads();
        if gens_proj.len() > num_threads {
          let chunk = (gens_proj.len() as f64 / num_threads as f64).ceil() as usize;
          (0..num_threads)
            .into_par_iter()
            .flat_map(|i| {
              let start = i * chunk;
              let end = if i == num_threads - 1 {
                gens_proj.len()
              } else {
                core::cmp::min((i + 1) * chunk, gens_proj.len())
              };
              if end > start {
                let mut gens = vec![$name_curve_affine::identity(); end - start];
                <Self as Curve>::batch_normalize(&gens_proj[start..end], &mut gens);
                gens
              } else {
                vec![]
              }
            })
            .collect()
        } else {
          let mut gens = vec![$name_curve_affine::identity(); n];
          <Self as Curve>::batch_normalize(&gens_proj, &mut gens);
          gens
        }
      }

      fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
        // see: grumpkin implementation at src/provider/bn256_grumpkin.rs
        let coordinates = self.to_affine().coordinates();
        if coordinates.is_some().unwrap_u8() == 1
          && (Self::PreprocessedGroupElement::identity() != self.to_affine())
        {
          (*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
        } else {
          (Self::Base::zero(), Self::Base::zero(), true)
        }
      }

      fn get_curve_params() -> (Self::Base, Self::Base, BigInt) {
        let A = $name::Point::a();
        let B = $name::Point::b();
        let order = BigInt::from_str_radix($order_str, 16).unwrap();

        (A, B, order)
      }

      fn zero() -> Self {
        $name::Point::identity()
      }

      fn get_generator() -> Self {
        $name::Point::generator()
      }
    }

    impl PrimeFieldExt for $name::Scalar {
      fn from_uniform(bytes: &[u8]) -> Self {
        let bytes_arr: [u8; 64] = bytes.try_into().unwrap();
        $name::Scalar::from_uniform_bytes(&bytes_arr)
      }
    }

    impl<G: Group> TranscriptReprTrait<G> for $name_compressed {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.as_ref().to_vec()
      }
    }

    impl CompressedGroup for $name_compressed {
      type GroupElement = $name::Point;

      fn decompress(&self) -> Option<$name::Point> {
        Some($name_curve::from_bytes(&self).unwrap())
      }
    }
  };
}
