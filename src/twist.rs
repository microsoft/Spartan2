//! Implements the Twist alternative algorithm.
//!
//! Val evaluation sum-check ported from: https://github.com/a16z/jolt/blob/fd91935e7a6c4fcd729c561b2feb5371d7ec8ec0/jolt-core/src/subprotocols/twist.rs
#![allow(clippy::too_many_arguments)]
use crate::{
  errors::SpartanError,
  math::Math,
  polys::{
    eq::EqPolynomial,
    multilinear::{BindingOrder, MultilinearPolynomial},
    univariate::{CompressedUniPoly, UniPoly},
  },
  sumcheck::SumcheckProof,
  traits::{Engine, PrimeFieldExt, transcript::TranscriptEngineTrait},
};
use ff::Field;
use itertools::Itertools;
use rayon::prelude::*;

/// SNARK for memory consistency checks
pub struct TwistSNARK<E: Engine> {
  /// Proof for the read-checking and write-checking sum-checks
  /// (steps 3 and 4 of Figure 9).
  read_write_checking_snark: ReadWriteCheckingSNARK<E>,
  /// Proof of the Val-evaluation sum-check (step 6 of Figure 9).
  val_evaluation_snark: ValEvaluationSNARK<E>,
}

impl<E: Engine> TwistSNARK<E> {
  /// Alternative prover algorithm for the Read-write checking sum-check
  #[tracing::instrument(skip_all, name = "TwistSNARK::prove")]
  pub fn prove(
    read_addresses: Vec<usize>,
    read_values: Vec<u32>,
    write_addresses: Vec<usize>,
    write_values: Vec<u32>,
    write_increments: Vec<i64>,
    r: Vec<E::Scalar>,
    r_prime: Vec<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<TwistSNARK<E>, SpartanError> {
    let (read_write_checking_snark, r_address, r_cycle) = ReadWriteCheckingSNARK::prove(
      read_addresses,
      read_values,
      &write_addresses,
      write_values,
      &write_increments,
      r,
      r_prime,
      transcript,
    )?;

    let (val_evaluation_snark, _r_cycle_prime) = prove_val_evaluation(
      write_addresses,
      write_increments,
      r_address,
      r_cycle,
      read_write_checking_snark.val_claim,
      transcript,
    )?;

    // TODO: Connect PCS

    Ok(TwistSNARK {
      read_write_checking_snark,
      val_evaluation_snark,
    })
  }

  /// Verifies the twist proof
  pub fn verify(
    &self,
    r: Vec<E::Scalar>,
    r_prime: Vec<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<(), SpartanError> {
    let log_T = r_prime.len();

    let r_cycle = self
      .read_write_checking_snark
      .verify(r, r_prime, transcript)?;

    let (sumcheck_claim, r_cycle_prime) = self.val_evaluation_snark.sumcheck_proof.verify(
      self.read_write_checking_snark.val_claim,
      log_T,
      2,
      transcript,
    )?;

    // Compute LT(r_cycle', r_cycle)
    let mut lt_eval = E::Scalar::ZERO;
    let mut eq_term = E::Scalar::ONE;
    for (x, y) in r_cycle_prime.iter().rev().zip(r_cycle.iter()) {
      lt_eval += (E::Scalar::ONE - x) * y * eq_term;
      eq_term *= E::Scalar::ONE - x - y + *x * y + *x * y;
    }

    assert_eq!(
      sumcheck_claim,
      lt_eval * self.val_evaluation_snark.inc_claim,
      "Val evaluation sumcheck failed"
    );

    // TODO: Prove Inc claim with PCS

    Ok(())
  }
}

/// ReadWrite checking sum-check proof
pub struct ReadWriteCheckingSNARK<E: Engine> {
  /// Joint sumcheck proof for the read-checking and write-checking sumchecks
  /// (steps 3 and 4 of Figure 9).
  sumcheck_proof: SumcheckProof<E>,
  /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
  /// checking sumcheck.
  ra_claim: E::Scalar,
  /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
  rv_claim: E::Scalar,
  /// The claimed evaluation wa(r_address, r_cycle) output by the read/write-
  /// checking sumcheck.
  wa_claim: E::Scalar,
  /// The claimed evaluation wv(r_address, r_cycle) output by the read/write-
  /// checking sumcheck.
  wv_claim: E::Scalar,
  /// The claimed evaluation val(r_address, r_cycle) output by the read/write-
  /// checking sumcheck.
  val_claim: E::Scalar,
  /// The claimed evaluation Inc(r, r') proven by the write-checking sumcheck.
  inc_claim: E::Scalar,
}

impl<E: Engine> ReadWriteCheckingSNARK<E> {
  /// Prover implementation for read-write checking sum-check
  #[tracing::instrument(skip_all, name = "ReadWriteCheckingSNARK::prove")]
  pub fn prove(
    read_addresses: Vec<usize>,
    read_values: Vec<u32>,
    write_addresses: &[usize],
    write_values: Vec<u32>,
    write_increments: &[i64],
    r: Vec<E::Scalar>,
    r_prime: Vec<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<(ReadWriteCheckingSNARK<E>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    const DEGREE: usize = 3;
    let T = write_increments.len();
    let K = 2usize.pow(r.len() as u32);
    assert!(T.is_power_of_two());
    let num_rounds = T.log_2() + K.log_2();
    debug_assert_eq!(read_addresses.len(), T);
    debug_assert_eq!(read_values.len(), T);
    debug_assert_eq!(write_addresses.len(), T);
    debug_assert_eq!(write_values.len(), T);
    debug_assert_eq!(write_increments.len(), T);

    let mut compressed_polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::with_capacity(num_rounds);
    let mut r_address: Vec<E::Scalar> = Vec::with_capacity(K.log_2());
    let mut r_cycle: Vec<E::Scalar> = Vec::with_capacity(T.log_2());

    // Used to batch the read-checking and write-checking sumcheck
    // (see Section 4.2.1)
    let z: E::Scalar = transcript.squeeze(b"z")?;
    let mut B = MultilinearPolynomial::new(EqPolynomial::new(r_prime.clone()).evals());
    let mut A = MultilinearPolynomial::new(EqPolynomial::new(r.clone()).evals());
    let rv = MultilinearPolynomial::new(
      read_values
        .iter()
        .map(|&v| E::Scalar::from(v as u64))
        .collect_vec(),
    );
    let mut wv = MultilinearPolynomial::new(
      write_values
        .iter()
        .map(|&v| E::Scalar::from(v as u64))
        .collect_vec(),
    );
    let inc_claim = write_addresses
      .iter()
      .zip(write_increments.iter())
      .enumerate()
      .map(|(j, (&k, &inc))| A[k] * B[j] * E::Scalar::from_i64(inc))
      .sum();
    let rv_claim = rv.evaluate(&r_prime);
    let mut claim_per_round: E::Scalar = rv_claim + z * inc_claim;
    let eq_km_c = [
      [
        E::Scalar::ONE,          // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
        E::Scalar::from_i64(-1), // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2) = -1
        E::Scalar::from_i64(-2), // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3) = -2
      ],
      [
        E::Scalar::ZERO,    // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
        E::Scalar::from(2), // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2) = 2
        E::Scalar::from(3), // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3) = 3
      ],
    ];
    let mut D = vec![E::Scalar::ONE; K];
    let mut C_row_0 = vec![E::Scalar::ZERO; K];
    let read_comb_func =
      |eq: E::Scalar, ra: E::Scalar, val: E::Scalar| -> E::Scalar { eq * ra * val };
    let write_comb_func = |eq_addr: E::Scalar,
                           eq_cycle: E::Scalar,
                           wa: E::Scalar,
                           wv: E::Scalar,
                           val: E::Scalar|
     -> E::Scalar { eq_addr * eq_cycle * (wa * (wv - val)) };

    for round in 0..K.log_2() {
      let mut uni_poly_evals = [E::Scalar::ZERO; DEGREE];
      let mut C_row_j = C_row_0.clone();
      for (j, ((&inc, &wa_usize), &ra_usize)) in write_increments
        .iter()
        .zip_eq(write_addresses.iter())
        .zip_eq(read_addresses.iter())
        .enumerate()
      {
        // Get evals for read-checking sum-check
        let m = round + 1;

        // Get val evals for read-checking sum-check
        // k' is the high bits
        let read_k_prime = ra_usize >> m;
        let c0 = C_row_j[2 * read_k_prime];
        let m_c = C_row_j[2 * read_k_prime + 1] - C_row_j[2 * read_k_prime];
        let c2 = C_row_j[2 * read_k_prime + 1] + m_c;
        let c3 = c2 + m_c;

        // k* = (k1, ... km)
        // km is the high bit of k*
        let k_star = ra_usize % (1 << m);
        let km = k_star >> round;
        // low (binded) bits
        let k_d = k_star % (1 << round);
        let ra_evals = [
          D[k_d] * eq_km_c[km][0],
          D[k_d] * eq_km_c[km][1],
          D[k_d] * eq_km_c[km][2],
        ];

        // write-checking sum-check evals
        let write_k_prime = wa_usize >> m;
        let wc0 = C_row_j[2 * write_k_prime];
        let wm_c = C_row_j[2 * write_k_prime + 1] - C_row_j[2 * write_k_prime];
        let wc2 = C_row_j[2 * write_k_prime + 1] + wm_c;
        let wc3 = wc2 + wm_c;

        let eq_r_address_evals = A.sumcheck_evals(write_k_prime, DEGREE, BindingOrder::LowToHigh);

        // k* = (k1, ... km)
        let wk_star = wa_usize % (1 << m);
        // low (binded) bits
        let wk_d = wk_star % (1 << round);
        // km is the high bit of k*
        let wkm = wk_star >> round;
        let wa_0 = D[wk_d] * eq_km_c[wkm][0];
        let wa_2 = D[wk_d] * eq_km_c[wkm][1];
        let wa_3 = D[wk_d] * eq_km_c[wkm][2];

        uni_poly_evals[0] += read_comb_func(B[j], ra_evals[0], c0)
          + z * write_comb_func(eq_r_address_evals[0], B[j], wa_0, wv[j], wc0);
        uni_poly_evals[1] += read_comb_func(B[j], ra_evals[1], c2)
          + z * write_comb_func(eq_r_address_evals[1], B[j], wa_2, wv[j], wc2);
        uni_poly_evals[2] += read_comb_func(B[j], ra_evals[2], c3)
          + z * write_comb_func(eq_r_address_evals[2], B[j], wa_3, wv[j], wc3);

        // Update the j'th row of C
        // higher bits
        let wkc_prime = wa_usize >> round;
        // Get lower bits
        let wkc_double_prime = wa_usize % (1 << round);
        C_row_j[wkc_prime] += E::Scalar::from_i64(inc) * D[wkc_double_prime];
      }

      let uni_poly = UniPoly::from_evals(&[
        uni_poly_evals[0],
        claim_per_round - uni_poly_evals[0],
        uni_poly_evals[1],
        uni_poly_evals[2],
      ])?;

      // append the prover's message to the transcript
      transcript.absorb(b"p", &uni_poly);

      //derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r_address.push(r_i);
      compressed_polys.push(uni_poly.compress());

      // Set up next round
      claim_per_round = uni_poly.evaluate(&r_i);

      // This only works when init_val is all 0's; otherwise we would need to bind it
      C_row_0.truncate(K >> (round + 1));
      A.bind_poly_var_bot(&r_i);
      let (left_evals, right_evals) = D.split_at_mut(1 << round);
      left_evals
        .par_iter_mut()
        .zip(right_evals.par_iter_mut())
        .for_each(|(left, right)| {
          *right = r_i * *left;
          *left -= *right;
        });
    }
    let mut ra = MultilinearPolynomial::new((0..T).map(|j| D[read_addresses[j]]).collect_vec());
    let mut wa = MultilinearPolynomial::new((0..T).map(|j| D[write_addresses[j]]).collect_vec());
    let eq_r_address_term = A[0];
    // All address variables binded
    // compute val_r_address
    let mut val_r_address: Vec<E::Scalar> = vec![E::Scalar::ZERO; T];
    for j in 1..T {
      let (k, inc) = (write_addresses[j - 1], write_increments[j - 1]);
      val_r_address[j] = val_r_address[j - 1] + E::Scalar::from_i64(inc) * D[k];
    }
    let mut val_r_address = MultilinearPolynomial::new(val_r_address);
    let write_comb_func =
      |eq_cycle: E::Scalar, wa: E::Scalar, wv: E::Scalar, val: E::Scalar| -> E::Scalar {
        eq_cycle * (wa * (wv - val))
      };
    // last log T rounds
    for _round in 0..T.log_2() {
      let half_poly_len = ra.len() / 2;

      let uni_poly_evals: [E::Scalar; DEGREE] = (0..half_poly_len)
        .into_par_iter()
        .map(|i| {
          let eq_evals = B.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
          let ra_evals = ra.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
          let val_evals = val_r_address.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
          let wa_evals = wa.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
          let wv_evals = wv.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
          [
            read_comb_func(eq_evals[0], ra_evals[0], val_evals[0])
              + z
                * eq_r_address_term
                * write_comb_func(eq_evals[0], wa_evals[0], wv_evals[0], val_evals[0]),
            read_comb_func(eq_evals[1], ra_evals[1], val_evals[1])
              + z
                * eq_r_address_term
                * write_comb_func(eq_evals[1], wa_evals[1], wv_evals[1], val_evals[1]),
            read_comb_func(eq_evals[2], ra_evals[2], val_evals[2])
              + z
                * eq_r_address_term
                * write_comb_func(eq_evals[2], wa_evals[2], wv_evals[2], val_evals[2]),
          ]
        })
        .reduce(
          || [E::Scalar::ZERO; 3],
          |running, new| {
            [
              running[0] + new[0],
              running[1] + new[1],
              running[2] + new[2],
            ]
          },
        );
      let uni_poly = UniPoly::from_evals(&[
        uni_poly_evals[0],
        claim_per_round - uni_poly_evals[0],
        uni_poly_evals[1],
        uni_poly_evals[2],
      ])?;
      transcript.absorb(b"p", &uni_poly);
      let r_i = transcript.squeeze(b"c")?;
      r_cycle.push(r_i);
      compressed_polys.push(uni_poly.compress());
      claim_per_round = uni_poly.evaluate(&r_i);
      rayon::join(
        || {
          rayon::join(|| B.bind_poly_var_bot(&r_i), || ra.bind_poly_var_bot(&r_i));
        },
        || {
          rayon::join(
            || val_r_address.bind_poly_var_bot(&r_i),
            || rayon::join(|| wa.bind_poly_var_bot(&r_i), || wv.bind_poly_var_bot(&r_i)),
          );
        },
      );
    }

    assert_eq!(
      claim_per_round,
      B[0] * ra[0] * val_r_address[0]
        + z * eq_r_address_term * B[0] * (wa[0] * (wv[0] - val_r_address[0]))
    );
    Ok((
      ReadWriteCheckingSNARK {
        sumcheck_proof: SumcheckProof { compressed_polys },
        ra_claim: ra[0],
        rv_claim,
        wa_claim: wa[0],
        wv_claim: wv[0],
        val_claim: val_r_address[0],
        inc_claim,
      },
      r_address,
      r_cycle,
    ))
  }

  /// Verifies the read-write checking sum-check
  pub fn verify(
    &self,
    r: Vec<E::Scalar>,
    r_prime: Vec<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<Vec<E::Scalar>, SpartanError> {
    let K = 2usize.pow(r.len() as u32);
    let T = 2usize.pow(r_prime.len() as u32);
    let z: E::Scalar = transcript.squeeze(b"z")?;

    let (sumcheck_claim, r_sumcheck) = self.sumcheck_proof.verify(
      self.rv_claim + z * self.inc_claim,
      T.log_2() + K.log_2(),
      3,
      transcript,
    )?;

    let (r_address, r_cycle) = r_sumcheck.split_at(K.log_2());
    let mut r_cycle = r_cycle.to_vec();
    r_cycle.reverse();
    let mut r_address = r_address.to_vec();
    r_address.reverse();
    let eq_eval_cycle = EqPolynomial::new(r_prime).evaluate(&r_cycle);
    let eq_eval_address = EqPolynomial::new(r).evaluate(&r_address);
    assert_eq!(
      sumcheck_claim,
      eq_eval_cycle * self.val_claim * self.ra_claim
        + z * eq_eval_address * eq_eval_cycle * (self.wa_claim * (self.wv_claim - self.val_claim)),
    );

    Ok(r_cycle.to_vec())
  }
}

/// Proof for the ValEvaluation sum-check
pub struct ValEvaluationSNARK<E: Engine> {
  /// Sumcheck proof for the Val-evaluation sumcheck (steps 6 of Figure 9).
  sumcheck_proof: SumcheckProof<E>,
  /// The claimed evaluation Inc(r_address, r_cycle') output by the Val-evaluation sumcheck.
  inc_claim: E::Scalar,
}

/// Implements the sumcheck prover for the Val-evaluation sumcheck described in
/// Section 8.1 and Appendix B of the Twist+Shout paper
#[tracing::instrument(skip_all)]
pub fn prove_val_evaluation<E: Engine>(
  write_addresses: Vec<usize>,
  write_increments: Vec<i64>,
  mut r_address: Vec<E::Scalar>,
  mut r_cycle: Vec<E::Scalar>,
  claimed_evaluation: E::Scalar,
  transcript: &mut E::TE,
) -> Result<(ValEvaluationSNARK<E>, Vec<E::Scalar>), SpartanError> {
  r_address.reverse();
  r_cycle.reverse();
  let T = 2usize.pow(r_cycle.len() as u32);

  // Compute the size-K table storing all eq(r_address, k) evaluations for
  // k \in {0, 1}^log(K)
  let eq_r_address = EqPolynomial::new(r_address.clone()).evals();

  let span = tracing::span!(tracing::Level::INFO, "compute Inc");
  let _guard = span.enter();

  // Compute the Inc polynomial using the above table
  let inc: Vec<E::Scalar> = write_addresses
    .par_iter()
    .zip(write_increments.par_iter())
    .map(|(k, increment)| eq_r_address[*k] * E::Scalar::from_i64(*increment))
    .collect();
  let mut inc = MultilinearPolynomial::new(inc);

  drop(_guard);
  drop(span);

  let span = tracing::span!(tracing::Level::INFO, "compute LT");
  let _guard = span.enter();

  let mut lt: Vec<E::Scalar> = vec![E::Scalar::ZERO; T];
  for (i, r) in r_cycle.iter().rev().enumerate() {
    let (evals_left, evals_right) = lt.split_at_mut(1 << i);
    evals_left
      .par_iter_mut()
      .zip(evals_right.par_iter_mut())
      .for_each(|(x, y)| {
        *y = *x * r;
        *x += *r - *y;
      });
  }
  let mut lt = MultilinearPolynomial::new(lt);

  drop(_guard);
  drop(span);

  let num_rounds = T.log_2();
  let mut previous_claim = claimed_evaluation;
  let mut r_cycle_prime: Vec<E::Scalar> = Vec::with_capacity(num_rounds);

  const DEGREE: usize = 2;

  let span = tracing::span!(tracing::Level::INFO, "Val-evaluation sumcheck");
  let _guard = span.enter();

  let mut compressed_polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::with_capacity(num_rounds);
  for _round in 0..num_rounds {
    #[cfg(test)]
    {
      let expected: E::Scalar = (0..inc.len()).map(|j| inc[j] * lt[j]).sum::<E::Scalar>();
      assert_eq!(
        expected, previous_claim,
        "Sumcheck sanity check failed in round {_round}"
      );
    }

    let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
    let _inner_guard = inner_span.enter();

    let univariate_poly_evals: [E::Scalar; 2] = (0..inc.len() / 2)
      .into_par_iter()
      .map(|i| {
        let inc_evals = inc.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
        let lt_evals = lt.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

        [inc_evals[0] * lt_evals[0], inc_evals[1] * lt_evals[1]]
      })
      .reduce(
        || [E::Scalar::ZERO; 2],
        |running, new| [running[0] + new[0], running[1] + new[1]],
      );

    let univariate_poly = UniPoly::from_evals(&[
      univariate_poly_evals[0],
      previous_claim - univariate_poly_evals[0],
      univariate_poly_evals[1],
    ])?;

    drop(_inner_guard);
    drop(inner_span);

    let compressed_poly = univariate_poly.compress();
    transcript.absorb(b"p", &univariate_poly);
    compressed_polys.push(compressed_poly);

    let r_j = transcript.squeeze(b"c")?;
    r_cycle_prime.push(r_j);

    previous_claim = univariate_poly.evaluate(&r_j);

    // Bind polynomials
    rayon::join(
      || inc.bind_poly_var_bot(&r_j),
      || lt.bind_poly_var_bot(&r_j),
    );
  }

  let proof = ValEvaluationSNARK {
    sumcheck_proof: SumcheckProof { compressed_polys },
    inc_claim: inc[0],
  };

  Ok((proof, r_cycle_prime))
}

#[cfg(test)]
mod tests {
  use crate::{
    math::Math,
    polys::multilinear::MultilinearPolynomial,
    traits::{Engine, transcript::TranscriptEngineTrait},
    twist::{ReadWriteCheckingSNARK, TwistSNARK, prove_val_evaluation},
  };
  use itertools::Itertools;
  use rand::{RngCore, SeedableRng, rngs::StdRng};

  type E = crate::provider::T256HyraxEngine;
  type F = <E as Engine>::Scalar;

  #[test]
  fn twist_e2e() {
    const K: usize = 16;
    const T: usize = 1 << 8;

    let mut rng = StdRng::seed_from_u64(2048);

    let mut registers = [0u32; K];
    let mut read_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut read_values: Vec<u32> = Vec::with_capacity(T);
    let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut write_values: Vec<u32> = Vec::with_capacity(T);
    let mut write_increments: Vec<i64> = Vec::with_capacity(T);
    for _ in 0..T {
      // Random read register
      let read_address = rng.next_u32() as usize % K;
      // Random write register
      let write_address = rng.next_u32() as usize % K;
      read_addresses.push(read_address);
      write_addresses.push(write_address);
      // Read the value currently in the read register
      read_values.push(registers[read_address]);
      // Random write value
      let write_value = rng.next_u32();
      write_values.push(write_value);
      // The increment is the difference between the new value and the old value
      let write_increment = (write_value as i64) - (registers[write_address] as i64);
      write_increments.push(write_increment);
      // Write the new value to the write register
      registers[write_address] = write_value;
    }

    let mut prover_transcript = <E as Engine>::TE::new(b"test_transcript");
    let r = prover_transcript
      .squeeze_scalar_powers(K.log_2(), b"r")
      .unwrap();
    let r_prime = prover_transcript
      .squeeze_scalar_powers(T.log_2(), b"r_prime")
      .unwrap();

    let proof: TwistSNARK<E> = TwistSNARK::prove(
      read_addresses,
      read_values,
      write_addresses,
      write_values,
      write_increments,
      r.clone(),
      r_prime.clone(),
      &mut prover_transcript,
    )
    .unwrap();

    let mut verifier_transcript = <E as Engine>::TE::new(b"test_transcript");
    let r = verifier_transcript
      .squeeze_scalar_powers(K.log_2(), b"r")
      .unwrap();
    let r_prime = verifier_transcript
      .squeeze_scalar_powers(T.log_2(), b"r_prime")
      .unwrap();

    let verification_result = proof.verify(r, r_prime, &mut verifier_transcript);
    assert!(
      verification_result.is_ok(),
      "Verification failed with error: {:?}",
      verification_result.err()
    );
  }

  #[test]
  fn val_evaluation_sumcheck() {
    const K: usize = 64;
    const T: usize = 1 << 8;

    let mut rng = StdRng::seed_from_u64(42);

    let mut registers = [0u32; K];
    let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut write_increments: Vec<i64> = Vec::with_capacity(T);
    let mut val: Vec<u32> = Vec::with_capacity(K * T);
    for _ in 0..T {
      val.extend(registers.iter());
      // Random write register
      let write_address = rng.next_u32() as usize % K;
      write_addresses.push(write_address);
      // Random write value
      let write_value = rng.next_u32();
      // The increment is the difference between the new value and the old value
      let write_increment = (write_value as i64) - (registers[write_address] as i64);
      write_increments.push(write_increment);
      // Write the new value to the write register
      registers[write_address] = write_value;
    }
    let val = MultilinearPolynomial::new(val.iter().map(|&v| F::from(v as u64)).collect_vec());

    let mut prover_transcript = <E as Engine>::TE::new(b"test_transcript");
    let r_cycle = prover_transcript
      .squeeze_scalar_powers(T.log_2(), b"r_cycle")
      .unwrap();
    let r_address = prover_transcript
      .squeeze_scalar_powers(K.log_2(), b"r_address")
      .unwrap();
    let mut r_val = [r_address.clone(), r_cycle.clone()].concat();
    r_val.reverse();
    let val_evaluation = val.evaluate(&r_val);
    let (proof, _) = prove_val_evaluation::<E>(
      write_addresses,
      write_increments,
      r_address,
      r_cycle,
      val_evaluation,
      &mut prover_transcript,
    )
    .unwrap();

    let mut verifier_transcript = <E as Engine>::TE::new(b"test_transcript");
    let _r_cycle = verifier_transcript
      .squeeze_scalar_powers(T.log_2(), b"r_cycle")
      .unwrap();
    let _r_address = verifier_transcript
      .squeeze_scalar_powers(K.log_2(), b"r_address")
      .unwrap();

    let verification_result =
      proof
        .sumcheck_proof
        .verify(val_evaluation, T.log_2(), 2, &mut verifier_transcript);
    assert!(
      verification_result.is_ok(),
      "Verification failed with error: {:?}",
      verification_result.err()
    );
  }

  #[test]
  fn read_write_checking_sumcheck_local() {
    const K: usize = 16;
    const T: usize = 1 << 8;

    let mut rng = StdRng::seed_from_u64(0);

    let mut registers = [0u32; K];
    let mut read_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut read_values: Vec<u32> = Vec::with_capacity(T);
    let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut write_values: Vec<u32> = Vec::with_capacity(T);
    let mut write_increments: Vec<i64> = Vec::with_capacity(T);
    for _ in 0..T {
      // Random read register
      let read_address = rng.next_u32() as usize % K;
      // Random write register
      let write_address = rng.next_u32() as usize % K;
      read_addresses.push(read_address);
      write_addresses.push(write_address);
      // Read the value currently in the read register
      read_values.push(registers[read_address]);
      // Random write value
      let write_value = rng.next_u32();
      write_values.push(write_value);
      // The increment is the difference between the new value and the old value
      let write_increment = (write_value as i64) - (registers[write_address] as i64);
      write_increments.push(write_increment);
      // Write the new value to the write register
      registers[write_address] = write_value;
    }

    let mut prover_transcript = <E as Engine>::TE::new(b"test_transcript");
    let r = prover_transcript
      .squeeze_scalar_powers(K.log_2(), b"r")
      .unwrap();
    let r_prime = prover_transcript
      .squeeze_scalar_powers(T.log_2(), b"r_prime")
      .unwrap();

    let (proof, _, _) = ReadWriteCheckingSNARK::<E>::prove(
      read_addresses,
      read_values,
      &write_addresses,
      write_values,
      &write_increments,
      r,
      r_prime,
      &mut prover_transcript,
    )
    .unwrap();

    let mut verifier_transcript = <E as Engine>::TE::new(b"test_transcript");
    let r = verifier_transcript
      .squeeze_scalar_powers(K.log_2(), b"r")
      .unwrap();
    let r_prime = verifier_transcript
      .squeeze_scalar_powers(T.log_2(), b"r_prime")
      .unwrap();

    proof.verify(r, r_prime, &mut verifier_transcript).unwrap();
  }
}
