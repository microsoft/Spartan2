//! Timing utilities for benchmarks and examples.
//!
//! This module provides:
//! - [`TimingLayer`]: A tracing layer that captures `elapsed_ms` and `constraints` fields
//! - Phase constants for Spartan and NeutronNova prove phases
//! - Helper functions for collecting and displaying timing data

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::Subscriber;
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

/// Shared timing data storage (phase name -> elapsed milliseconds).
pub type TimingData = Arc<Mutex<HashMap<String, u64>>>;

/// Shared constraint count storage.
pub type ConstraintsData = Arc<Mutex<Option<u64>>>;

/// A tracing layer that captures timing events with `elapsed_ms` fields.
pub struct TimingLayer {
  data: TimingData,
  constraints: ConstraintsData,
}

impl TimingLayer {
  /// Create a new timing layer and return handles to the collected data.
  pub fn new() -> (Self, TimingData, ConstraintsData) {
    let data: TimingData = Arc::new(Mutex::new(HashMap::new()));
    let constraints: ConstraintsData = Arc::new(Mutex::new(None));
    (
      Self {
        data: data.clone(),
        constraints: constraints.clone(),
      },
      data,
      constraints,
    )
  }
}

impl Default for TimingLayer {
  fn default() -> Self {
    Self::new().0
  }
}

struct ElapsedVisitor {
  elapsed_ms: Option<u64>,
  constraints: Option<u64>,
  message: Option<String>,
}

impl Visit for ElapsedVisitor {
  fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
    match field.name() {
      "elapsed_ms" => {
        let s = format!("{:?}", value);
        self.elapsed_ms = s.parse().ok();
      }
      "constraints" => {
        let s = format!("{:?}", value);
        self.constraints = s.parse().ok();
      }
      "message" => {
        self.message = Some(format!("{:?}", value));
      }
      _ => {}
    }
  }

  fn record_str(&mut self, field: &Field, value: &str) {
    if field.name() == "message" {
      self.message = Some(value.to_string());
    }
  }

  fn record_u64(&mut self, field: &Field, value: u64) {
    match field.name() {
      "elapsed_ms" => self.elapsed_ms = Some(value),
      "constraints" => self.constraints = Some(value),
      _ => {}
    }
  }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for TimingLayer {
  fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
    let mut visitor = ElapsedVisitor {
      elapsed_ms: None,
      constraints: None,
      message: None,
    };
    event.record(&mut visitor);

    if let (Some(ms), Some(msg)) = (visitor.elapsed_ms, visitor.message) {
      if let Ok(mut map) = self.data.lock() {
        *map.entry(msg).or_insert(0) += ms;
      }
    }
    if let Some(c) = visitor.constraints {
      if let Ok(mut lock) = self.constraints.lock() {
        *lock = Some(c);
      }
    }
  }
}

/// Extract timing values for the given phases from collected data.
pub fn snapshot_timings(data: &TimingData, phases: &[&str]) -> Vec<u64> {
  let map = data.lock().unwrap();
  phases.iter().map(|k| map.get(*k).copied().unwrap_or(0)).collect()
}

/// Clear all collected timing data.
pub fn clear_timings(data: &TimingData) {
  data.lock().unwrap().clear();
}

// ============================================================================
// Spartan prove-phase constants
// ============================================================================

/// Spartan prove phase names (for timing collection).
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

/// Short names for Spartan phases (for table display).
pub const SHORT_NAMES: &[&str] = &[
  "synth_pre", "commit_pre", "r1cs_rest", "commit_rest",
  "mat_vec", "prep_ml", "outer_sc", "eval_rx",
  "eval_sparse", "poly_ABC", "poly_z", "inner_sc", "pcs", "total",
];

// ============================================================================
// NeutronNova NIFS prove-phase constants
// ============================================================================

/// NeutronNova NIFS prove phase names.
pub const NEUTRONNOVA_PHASES: &[&str] = &[
  "generate_shared_witness",
  "generate_precommitted_witnesses",
  "commit_witness_precommitted",
  "matrix_vector_multiply_instances",
  "nifs_folding_rounds",
  "fold_witnesses",
  "fold_instances",
  "end_to_end_total",
];

/// Short names for NeutronNova phases.
pub const NEUTRONNOVA_SHORT_NAMES: &[&str] = &[
  "shared_syn", "precom_syn", "commit_pre",
  "mat_vec", "nifs_fold",
  "fold_W", "fold_U", "total",
];

/// Print a comparison table of timing data.
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
