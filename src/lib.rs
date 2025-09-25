//! This library implements Spartan, a high-speed SNARK.
//! We currently implement a non-preprocessing version of Spartan
//! that is generic over the polynomial commitment and evaluation argument (i.e., a PCS).
#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::type_complexity)]
#![forbid(unsafe_code)]

// private modules
mod math;
mod r1cs;

#[macro_use]
mod macros;

// public modules
pub mod bellpepper;
pub mod digest;
pub mod errors;
pub mod neutronnova;
pub mod nifs;
pub mod polys;
pub mod provider;
pub mod spartan;
pub mod sumcheck;
pub mod traits;
pub mod zk;

/// Start a span + timer, return `(Span, Instant)`.
macro_rules! start_span {
    ($name:expr $(, $($fmt:tt)+)?) => {{
        let span       = tracing::info_span!($name $(, $($fmt)+)?);
        let span_clone = span.clone();    // lives as long as the guard
        let _guard      = span_clone.enter();
        (span, std::time::Instant::now())
    }};
}
pub(crate) use start_span;

/// The width used for per-round commitments in the multiround protocol.
/// This affects the commitment scheme structure and padding calculations.
pub const MULTIROUND_COMMITMENT_WIDTH: usize = 4;

use traits::{Engine, pcs::PCSEngineTrait};
type CommitmentKey<E> = <<E as traits::Engine>::PCS as PCSEngineTrait<E>>::CommitmentKey;
type VerifierKey<E> = <<E as traits::Engine>::PCS as PCSEngineTrait<E>>::VerifierKey;
type Commitment<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::Commitment;
type PartialCommitment<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::PartialCommitment;
type PCS<E> = <E as Engine>::PCS;
type Blind<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::Blind;
