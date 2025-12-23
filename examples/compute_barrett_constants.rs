//! Compute Barrett reduction constants for Pallas Fp.
//! Run with: cargo run --example compute_barrett_constants

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};

fn main() {
  // Pallas Fp modulus (base field)
  let p = BigUint::parse_bytes(
    b"40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
    16,
  )
  .unwrap();

  // Barrett parameters
  let np = 255u32; // bit-length of p
  let r_prime = BigUint::one() << np; // R' = 2^255
  let r = BigUint::one() << 64u32; // r = 2^64

  // Compute constants
  let two_p = &p * 2u32;
  let mu = (&r * &r_prime) / &two_p;
  let two_pow_64_mod_p = (BigUint::one() << 64u32) % &p;

  // Print as Rust constants
  println!("// Pallas Fp (Base field) Barrett constants");
  println!("// p = 0x{:x}", p);
  println!();
  print_const("PALLAS_FP", &p, 4);
  print_const("PALLAS_FP_2P", &two_p, 5);
  print_const("PALLAS_FP_MU", &mu, 5);
  print_const("TWO_POW_64_MOD_FP", &two_pow_64_mod_p, 4);

  // Also compute Fq constants
  compute_pallas_fq();
}

fn print_const(name: &str, value: &BigUint, num_limbs: usize) {
  let limbs: Vec<u64> = (0..num_limbs)
    .map(|i| {
      let shifted = value >> (64 * i);
      let masked = &shifted & BigUint::from(u64::MAX);
      masked.to_u64().unwrap_or(0)
    })
    .collect();

  let formatted: Vec<String> = limbs.iter().map(|x| format!("0x{:016x}", x)).collect();
  println!(
    "const {}: [u64; {}] = [{}];",
    name,
    num_limbs,
    formatted.join(", ")
  );
}

// Also compute for Pallas Fq (scalar field) = Vesta Fp
fn compute_pallas_fq() {
  // Pallas Fq modulus (scalar field)
  let q = BigUint::parse_bytes(
    b"40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
    16,
  )
  .unwrap();

  let nq = 255u32;
  let r_prime = BigUint::one() << nq;
  let r = BigUint::one() << 64u32;

  let two_q = &q * 2u32;
  let mu = (&r * &r_prime) / &two_q;
  let two_pow_64_mod_q = (BigUint::one() << 64u32) % &q;

  println!();
  println!("// Pallas Fq (Scalar field) Barrett constants");
  println!("// q = 0x{:x}", q);
  println!();
  print_const("PALLAS_FQ", &q, 4);
  print_const("PALLAS_FQ_2Q", &two_q, 5);
  print_const("PALLAS_FQ_MU", &mu, 5);
  print_const("TWO_POW_64_MOD_FQ", &two_pow_64_mod_q, 4);
}
