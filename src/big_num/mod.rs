// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Big-number arithmetic for optimized sumcheck.
//!
//! This module provides wide-integer accumulators and delayed modular reduction
//! for field × field products, enabling faster sumcheck by batching Montgomery
//! reduction operations.

mod delayed_reduction;
pub mod field_reduction_constants;
pub mod limbs;
pub mod macros;
pub mod montgomery;

pub use delayed_reduction::DelayedReduction;
pub use field_reduction_constants::FieldReductionConstants;
