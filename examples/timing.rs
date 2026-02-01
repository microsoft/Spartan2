//! Tracing layer that captures `elapsed_ms` and `constraints` fields from events.
//! Used by benchmark examples to collect timing data programmatically.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::Subscriber;
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

pub type TimingData = Arc<Mutex<HashMap<String, u64>>>;
pub type ConstraintsData = Arc<Mutex<Option<u64>>>;

pub struct TimingLayer {
  data: TimingData,
  constraints: ConstraintsData,
}

impl TimingLayer {
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
        map.insert(msg, ms);
      }
    }
    if let Some(c) = visitor.constraints {
      if let Ok(mut lock) = self.constraints.lock() {
        *lock = Some(c);
      }
    }
  }
}

pub fn snapshot_timings(data: &TimingData, phases: &[&str]) -> Vec<u64> {
  let map = data.lock().unwrap();
  phases.iter().map(|k| map.get(*k).copied().unwrap_or(0)).collect()
}

pub fn clear_timings(data: &TimingData) {
  data.lock().unwrap().clear();
}
