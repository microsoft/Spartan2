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
//! - [`basis`]: Domain eval containers, basis computation, and tensor coefficients
//! - [`extension`]: Multilinear polynomial extension (Procedure 6)
//! - [`accumulator`]: Accumulator data structures
//! - [`accumulator_builder`]: Accumulator construction (Procedure 9)
//! - [`index`]: Index mapping (Definition A.5)
//! - [`thread_state`]: Thread-local buffers for parallel execution

mod accumulator;
mod accumulator_builder;
mod basis;
mod csr;
mod domain;
pub(crate) mod extension;
mod index;
mod thread_state;

// Crate-internal surface used by the small-value sumcheck implementation.
pub(crate) use accumulator::LagrangeAccumulators;
pub(crate) use accumulator_builder::{
  SPARTAN_T_DEGREE, build_accumulators_neutronnova,
  build_accumulators_neutronnova_from_prefix_ext_workspace,
  build_accumulators_neutronnova_preextended, build_accumulators_spartan,
};
pub(crate) use basis::{
  LagrangeBasisFactory, LagrangeCoeff, LagrangeDomainEvals, ReducedLagrangeDomainEvals,
};
