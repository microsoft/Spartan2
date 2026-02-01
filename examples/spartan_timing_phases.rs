//! Shared Spartan prove-phase constants and table printing.
//! Used by `sha256.rs` and `sha256_chain_benchmark.rs`.

pub const PHASES: &[&str] = &[
  "precommitted_witness_synthesize",
  "commit_witness_precommitted",
  "r1cs_instance_and_witness",
  "commit_witness_rest",
  "matrix_vector_multiply",
  "prepare_multilinear_polys",
  "outer_sumcheck",
  "compute_eval_rx",
  "compute_eval_table_sparse",
  "prepare_poly_ABC",
  "prepare_poly_z",
  "inner_sumcheck",
  "pcs_prove",
  "spartan_snark_prove",
];

pub const SHORT_NAMES: &[&str] = &[
  "synth_pre", "commit_pre", "r1cs_rest", "commit_rest",
  "mat_vec", "prep_ml", "outer_sc", "eval_rx",
  "eval_sparse", "poly_ABC", "poly_z", "inner_sc", "pcs", "total",
];

pub fn print_table(header: &str, small: &[u64], large: &[u64]) {
  let col_w = 12;

  eprintln!("\n{}", header);

  eprint!("{:<10}", "");
  for name in SHORT_NAMES {
    eprint!("{:>width$}", name, width = col_w);
  }
  eprintln!();

  eprint!("{:<10}", "small");
  for v in small {
    eprint!("{:>width$}", v, width = col_w);
  }
  eprintln!();

  eprint!("{:<10}", "large");
  for v in large {
    eprint!("{:>width$}", v, width = col_w);
  }
  eprintln!();

  eprint!("{:<10}", "speedup");
  for (s, l) in small.iter().zip(large.iter()) {
    if *s == 0 {
      eprint!("{:>width$}", "-", width = col_w);
    } else {
      eprint!("{:>width$.2}x", *l as f64 / *s as f64, width = col_w - 1);
    }
  }
  eprintln!();
}
