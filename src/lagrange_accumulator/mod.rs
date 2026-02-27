// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Lagrange accumulator algorithm for small-value sumcheck optimization (Algorithm 6).
//!
//! This module implements the Lagrange domain extension technique from IACR 2025/1117,
//! which accelerates sumcheck provers when witness coefficients are small integers.
//!
//! # Module Structure
//!
//! - [`domain`]: Domain types (LagrangePoint, LagrangeHatPoint, LagrangeIndex)
//! - [`evals`]: Evaluation containers (LagrangeEvals, LagrangeHatEvals)
//! - [`basis`]: Lagrange basis computation (LagrangeBasisFactory, LagrangeCoeff)
//! - [`extension`]: Multilinear polynomial extension (Procedure 6)
//! - [`accumulator`]: Accumulator data structures
//! - [`accumulator_builder`]: Accumulator construction (Procedure 9)
//! - [`index`]: Index mapping (Definition A.5)
//! - [`thread_state`]: Thread-local buffers for parallel execution
//! - [`eq_round`]: Per-round equality factor tracking

mod accumulator;
mod accumulator_builder;
mod basis;
mod csr;
mod domain;
mod eq_round;
mod evals;
pub(crate) mod extension;
mod index;
mod thread_state;

// Re-exports for public API
#[allow(unused_imports)]
pub use domain::{LagrangeHatPoint, LagrangeIndex, LagrangePoint, ValueOneExcluded};

// Evaluation containers
pub use evals::{LagrangeEvals, LagrangeHatEvals};

// Basis computation
pub use basis::{LagrangeBasisFactory, LagrangeCoeff};

// Accumulators
#[allow(unused_imports)]
pub use accumulator::{LagrangeAccumulators, RoundAccumulator};

// Builder functions
pub use accumulator_builder::{SPARTAN_T_DEGREE, build_accumulators_spartan};

// Eq round factor and derivation
pub use eq_round::{EqRoundFactor, derive_t1};
