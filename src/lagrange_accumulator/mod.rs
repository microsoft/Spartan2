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
//! - [`witness`]: Witness polynomial abstraction (MatVecMLE trait)

mod accumulator;
mod accumulator_builder;
mod basis;
mod delay_modular_reduction_mode;
mod domain;
mod eq_round;
mod evals;
mod extension;
mod index;
mod mat_vec_mle;
mod thread_state;

// Domain types
pub use domain::{LagrangeHatPoint, LagrangeIndex, LagrangePoint, ValueOneExcluded};

// Evaluation containers
pub use evals::{LagrangeEvals, LagrangeHatEvals};

// Basis computation
pub use basis::{LagrangeBasisFactory, LagrangeCoeff};

// Extension
pub use extension::LagrangeEvaluatedMultilinearPolynomial;

// Accumulators
pub use accumulator::{LagrangeAccumulators, RoundAccumulator};

// Builder functions
pub use accumulator_builder::{SPARTAN_T_DEGREE, build_accumulators, build_accumulators_spartan};

// Delayed modular reduction mode selection
pub use delay_modular_reduction_mode::{
  AccumulateProduct, DelayedModularReductionDisabled, DelayedModularReductionEnabled,
  DelayedModularReductionMode,
};

// Index computation
pub use index::{AccumulatorPrefixIndex, CachedPrefixIndex, compute_idx4};

// Eq round factor and derivation
pub use eq_round::{EqRoundFactor, derive_t1};
pub use mat_vec_mle::MatVecMLE;
