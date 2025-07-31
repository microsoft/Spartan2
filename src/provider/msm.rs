//! This module provides a multi-scalar multiplication routine
//! The generic implementation is adapted from halo2; we add an optimization to commit to bits more efficiently
//! The specialized implementations are adapted from jolt, with additional optimizations and parallelization.
use crate::{errors::SpartanError, start_span};
use ff::{Field, PrimeField};
use halo2curves::{CurveAffine, group::Group};
use num_integer::Integer;
use num_traits::{ToPrimitive, Zero};
use rayon::{current_num_threads, prelude::*};
use std::time::Instant;
use tracing::{info, info_span};

#[derive(Clone, Copy)]
enum Bucket<C: CurveAffine> {
  None,
  Affine(C),
  Projective(C::Curve),
}

impl<C: CurveAffine> Bucket<C> {
  fn add_assign(&mut self, other: &C) {
    *self = match *self {
      Bucket::None => Bucket::Affine(*other),
      Bucket::Affine(a) => Bucket::Projective(a + *other),
      Bucket::Projective(a) => Bucket::Projective(a + other),
    }
  }

  fn add(self, other: C::Curve) -> C::Curve {
    match self {
      Bucket::None => other,
      Bucket::Affine(a) => other + a,
      Bucket::Projective(a) => other + a,
    }
  }
}

/// Single-threaded Pippenger MSM with scalar pre-processing.
///
/// * Scalars are assumed to be 256-bit (4 × u64 limbs).
fn cpu_msm_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  let (_msm_span, msm_t) = start_span!("msm_serial", size = coeffs.len());
  assert_eq!(coeffs.len(), bases.len());

  // Pick window size (in bits) – same heuristic as before
  let c: usize = if bases.len() < 4_000 { 8 } else { 16 };
  let window_mask: u64 = (1u64 << c) - 1;
  let segments: usize = (256 + c - 1) / c; // 256-bit scalars

  // Boolean scalars: accumulated and separated from non-Boolean scalars
  let mut boolean_sum = C::Curve::identity();
  let mut non_boolean = Vec::with_capacity(coeffs.len());

  for (s, b) in coeffs.iter().zip(bases) {
    if *s == C::Scalar::ONE {
      boolean_sum += b;
    } else if *s != C::Scalar::ZERO {
      non_boolean.push((*s, *b));
    }
  }

  if non_boolean.is_empty() {
    return boolean_sum;
  }
  let (coeffs, bases): (Vec<_>, Vec<_>) = non_boolean.into_iter().unzip();

  // Pre-split every scalar into 4 little-endian u64 limbs
  let limbs: Vec<[u64; 4]> = coeffs
    .iter()
    .map(|s| {
      let bytes = s.to_repr(); // little-endian bytes
      let bytes = bytes.as_ref(); // &[u8; 32] → &[u8]

      assert_eq!(bytes.len(), 32); // Ensure we only have 32 bytes

      let mut out = [0u64; 4];
      for i in 0..4 {
        let start = i * 8;
        out[i] = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
      }
      out // out[0] = 64 LSB, out[3] = 64 MSB
    })
    .collect();

  // Scratch buffers reused for every segment
  let mut buckets: Vec<C::Curve> = vec![C::Curve::identity(); (1 << c) - 1];
  let mut acc = C::Curve::identity();

  // Pippenger main loop – highest segment first
  for seg in (0..segments).rev() {
    // multiply accumulator by 2^c
    for _ in 0..c {
      acc = acc.double();
    }

    // reset the buckets
    buckets.fill(C::Curve::identity());

    // which 64-bit word / bit offset stores this window?
    let word = (seg * c) / 64;
    let shift = (seg * c) & 63;

    // fill the buckets
    for (limb, base) in limbs.iter().zip(&bases) {
      let slice = ((limb[word] >> shift) & window_mask) as usize;
      if slice != 0 {
        buckets[slice - 1] += *base;
      }
    }

    // summation-by-parts to fold the buckets into `acc`
    let mut running = C::Curve::identity();
    for b in buckets.iter().rev() {
      running += b;
      acc += &running;
    }
  }

  info!(elapsed_ms = %msm_t.elapsed().as_millis(), size = coeffs.len(), "msm_serial");

  // Add the fast Boolean part back in and return
  acc + boolean_sum
}

/// Performs a multi-scalar-multiplication operation without GPU acceleration.
///
/// This will use multithreading if beneficial.
/// Adapted from zcash/halo2
///
/// # Errors
/// Returns `SpartanError::InvalidInputLength` if coeffs and bases have different lengths.
pub fn msm<C: CurveAffine>(
  coeffs: &[C::Scalar],
  bases: &[C],
  use_parallelism_internally: bool,
) -> Result<C::Curve, SpartanError> {
  let (_msm_span, msm_t) = start_span!("msm", size = coeffs.len());

  if coeffs.len() != bases.len() {
    return Err(SpartanError::InvalidInputLength);
  }

  let num_threads = if coeffs.len() < 512 {
    // If the number of coefficients is small, we use a single thread.
    // Otherwise, we use parallelism.
    // This is a heuristic to avoid overhead from parallelism for small inputs.
    1
  } else if use_parallelism_internally {
    current_num_threads()
  } else {
    1
  };

  info!(num_threads = num_threads, "msm");

  let result = if coeffs.len() > num_threads {
    let chunk = coeffs.len() / num_threads;
    coeffs
      .par_chunks(chunk)
      .zip(bases.par_chunks(chunk))
      .map(|(coeffs, bases)| cpu_msm_serial(coeffs, bases))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    cpu_msm_serial(coeffs, bases)
  };

  info!(elapsed_ms = %msm_t.elapsed().as_millis(), size = coeffs.len(), "msm");
  Ok(result)
}

fn num_bits(n: usize) -> usize {
  if n == 0 { 0 } else { (n.ilog2() + 1) as usize }
}

/// Multi-scalar multiplication using the best algorithm for the given scalars.
///
/// # Errors
/// Returns `SpartanError::InvalidInputLength` if bases and scalars have different lengths.
/// Returns `SpartanError::InternalError` if scalars contain values that cannot be processed.
pub fn msm_small<C: CurveAffine, T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
  scalars: &[T],
  bases: &[C],
  use_parallelism_internally: bool,
) -> Result<C::Curve, SpartanError> {
  let (_msm_small_span, msm_small_t) = start_span!("msm_small", size = scalars.len());

  if bases.len() != scalars.len() {
    return Err(SpartanError::InvalidInputLength);
  }

  let max_scalar = scalars.iter().max().ok_or(SpartanError::InternalError)?;
  let max_scalar_usize = max_scalar.to_usize().ok_or(SpartanError::InternalError)?;
  let max_num_bits = num_bits(max_scalar_usize);
  let result = match max_num_bits {
    0 => C::identity().into(),
    1 => {
      let (_binary_span, binary_t) = start_span!("msm_binary");
      let result = msm_binary(scalars, bases, use_parallelism_internally);
      if binary_t.elapsed().as_millis() != 0 {
        info!(elapsed_ms = %binary_t.elapsed().as_millis(), size = scalars.len(), "msm_binary");
      }
      result
    }
    2..=10 => {
      let (_msm_10_span, msm_10_t) = start_span!("msm_10", max_bits = max_num_bits);
      let result = msm_10(scalars, bases, max_num_bits, use_parallelism_internally);
      info!(elapsed_ms = %msm_10_t.elapsed().as_millis(), max_bits = max_num_bits, "msm_10");
      result
    }
    _ => {
      let (_msm_rest_span, msm_rest_t) = start_span!("msm_small_rest", max_bits = max_num_bits);
      let result = msm_small_rest(scalars, bases, max_num_bits, use_parallelism_internally);
      info!(elapsed_ms = %msm_rest_t.elapsed().as_millis(), max_bits = max_num_bits, "msm_small_rest");
      result
    }
  };

  if msm_small_t.elapsed().as_millis() != 0 {
    info!(elapsed_ms = %msm_small_t.elapsed().as_millis(), size = scalars.len(), max_bits = max_num_bits, "msm_small");
  }
  Ok(result)
}

#[inline(always)]
fn msm_binary<C: CurveAffine, T: Integer + Sync>(
  scalars: &[T],
  bases: &[C],
  use_parallelism_internally: bool,
) -> C::Curve {
  assert_eq!(scalars.len(), bases.len());
  let num_threads = if use_parallelism_internally {
    current_num_threads()
  } else {
    1
  };
  let process_chunk = |scalars: &[T], bases: &[C]| {
    let mut acc = C::Curve::identity();
    scalars
      .iter()
      .zip(bases.iter())
      .filter(|(scalar, _)| (!scalar.is_zero()))
      .for_each(|(_, base)| {
        acc += *base;
      });
    acc
  };

  if scalars.len() > num_threads {
    let chunk = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk)
      .zip(bases.par_chunks(chunk))
      .map(|(scalars, bases)| process_chunk(scalars, bases))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    process_chunk(scalars, bases)
  }
}

/// MSM optimized for up to 10-bit scalars
#[inline(always)]
fn msm_10<C: CurveAffine, T: Into<u64> + Zero + Copy + Sync>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
  use_parallelism_internally: bool,
) -> C::Curve {
  fn msm_10_serial<C: CurveAffine, T: Into<u64> + Zero + Copy>(
    scalars: &[T],
    bases: &[C],
    max_num_bits: usize,
  ) -> C::Curve {
    let num_buckets: usize = 1 << max_num_bits;
    let mut buckets = vec![Bucket::None; num_buckets];

    scalars
      .iter()
      .zip(bases.iter())
      .filter(|(scalar, _base)| !scalar.is_zero())
      .for_each(|(scalar, base)| {
        let bucket_index: u64 = (*scalar).into();
        buckets[bucket_index as usize].add_assign(base);
      });

    let mut result = C::Curve::identity();
    let mut running_sum = C::Curve::identity();
    buckets.iter().skip(1).rev().for_each(|exp| {
      running_sum = exp.add(running_sum);
      result += &running_sum;
    });
    result
  }

  let num_threads = if use_parallelism_internally {
    current_num_threads()
  } else {
    1
  };
  if scalars.len() > num_threads {
    let chunk_size = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk_size)
      .zip(bases.par_chunks(chunk_size))
      .map(|(scalars_chunk, bases_chunk)| msm_10_serial(scalars_chunk, bases_chunk, max_num_bits))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    msm_10_serial(scalars, bases, max_num_bits)
  }
}

#[inline(always)]
fn msm_small_rest<C: CurveAffine, T: Into<u64> + Zero + Copy + Sync>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
  use_parallelism_internally: bool,
) -> C::Curve {
  fn msm_small_rest_serial<C: CurveAffine, T: Into<u64> + Zero + Copy>(
    scalars: &[T],
    bases: &[C],
    max_num_bits: usize,
  ) -> C::Curve {
    let c = if bases.len() < 4_000 { 8 } else { 16 }; // divisor of 64

    let zero = C::Curve::identity();

    let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _base)| !s.is_zero());
    let window_starts = (0..max_num_bits).step_by(c);

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // in parallel process each such window.
    let window_sums: Vec<_> = window_starts
      .map(|w_start| {
        let mut res = zero;
        // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
        let mut buckets = vec![zero; (1 << c) - 1];
        // This clone is cheap, because the iterator contains just a
        // pointer and an index into the original vectors.
        scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
          let scalar: u64 = scalar.into();
          if scalar == 1 {
            // We only process unit scalars once in the first window.
            if w_start == 0 {
              res += base;
            }
          } else {
            let mut scalar = scalar;

            // We right-shift by w_start, thus getting rid of the
            // lower bits.
            scalar >>= w_start;

            // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
            scalar %= 1 << c;

            // If the scalar is non-zero, we update the corresponding
            // bucket.
            // (Recall that `buckets` doesn't have a zero bucket.)
            if scalar != 0 {
              buckets[(scalar - 1) as usize] += base;
            }
          }
        });

        // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
        // This is computed below for b buckets, using 2b curve additions.
        //
        // We could first normalize `buckets` and then use mixed-addition
        // here, but that's slower for the kinds of groups we care about
        // (Short Weierstrass curves and Twisted Edwards curves).
        // In the case of Short Weierstrass curves,
        // mixed addition saves ~4 field multiplications per addition.
        // However normalization (with the inversion batched) takes ~6
        // field multiplications per element,
        // hence batch normalization is a slowdown.

        // `running_sum` = sum_{j in i..num_buckets} bucket[j],
        // where we iterate backward from i = num_buckets to 0.
        let mut running_sum = C::Curve::identity();
        buckets.into_iter().rev().for_each(|b| {
          running_sum += &b;
          res += &running_sum;
        });
        res
      })
      .collect();

    // We store the sum for the lowest window.
    let lowest = window_sums.first().copied().unwrap_or(zero);

    // We're traversing windows from high to low.
    lowest
      + window_sums[1..]
        .iter()
        .rev()
        .fold(zero, |mut total, sum_i| {
          total += sum_i;
          for _ in 0..c {
            total = total.double();
          }
          total
        })
  }

  let num_threads = if use_parallelism_internally {
    current_num_threads()
  } else {
    1
  };
  if scalars.len() > num_threads {
    let chunk_size = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk_size)
      .zip(bases.par_chunks(chunk_size))
      .map(|(scalars_chunk, bases_chunk)| {
        msm_small_rest_serial(scalars_chunk, bases_chunk, max_num_bits)
      })
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    msm_small_rest_serial(scalars, bases, max_num_bits)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::{pallas, vesta};
  use ff::Field;
  use halo2curves::{CurveAffine, group::Group};
  use rand_core::OsRng;

  fn test_general_msm_with<F: Field, A: CurveAffine<ScalarExt = F>>() {
    let n = 8;
    let coeffs = (0..n).map(|_| F::random(OsRng)).collect::<Vec<_>>();
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();

    assert_eq!(coeffs.len(), bases.len());
    let naive = coeffs
      .iter()
      .zip(bases.iter())
      .fold(A::CurveExt::identity(), |acc, (coeff, base)| {
        acc + *base * coeff
      });
    let msm = msm(&coeffs, &bases, true);

    assert_eq!(naive, msm.unwrap())
  }

  #[test]
  fn test_general_msm() {
    test_general_msm_with::<pallas::Scalar, pallas::Affine>();
    test_general_msm_with::<vesta::Scalar, vesta::Affine>();
  }

  fn test_msm_ux_with<F: PrimeField, A: CurveAffine<ScalarExt = F>>() {
    let n = 8;
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();

    for bit_width in [1, 4, 8, 10, 16, 20, 32, 40, 64] {
      println!("bit_width: {bit_width}");
      assert!(bit_width <= 64); // Ensure we don't overflow F::from
      let coeffs: Vec<u64> = (0..n)
        .map(|_| rand::random::<u64>() % (1 << bit_width))
        .collect::<Vec<_>>();
      let coeffs_scalar: Vec<F> = coeffs.iter().map(|b| F::from(*b)).collect::<Vec<_>>();
      let general = msm(&coeffs_scalar, &bases, true);
      let integer = msm_small(&coeffs, &bases, true);

      assert_eq!(general.unwrap(), integer.unwrap());
    }
  }

  #[test]
  fn test_msm_ux() {
    test_msm_ux_with::<pallas::Scalar, pallas::Affine>();
    test_msm_ux_with::<vesta::Scalar, vesta::Affine>();
  }
}
