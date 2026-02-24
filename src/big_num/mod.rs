// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Big-number arithmetic for optimized sumcheck.
//!
//! This module provides wide-integer accumulators and delayed modular reduction
//! for field × field products, enabling faster sumcheck by batching Montgomery
//! reduction operations.

pub(crate) mod delayed_reduction;
pub(crate) mod field_reduction_constants;
pub(crate) mod macros;
pub(crate) mod montgomery;

mod limbs;

pub use delayed_reduction::DelayedReduction;
pub use field_reduction_constants::FieldReductionConstants;
