// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Big-number arithmetic for optimized sumcheck.
//!
//! This module provides wide-integer accumulators and delayed modular reduction
//! for field products, enabling faster sumcheck by batching reduction operations.
//!
//! ## Reduction Strategies
//!
//! - **Montgomery reduction**: Used for field × field products. Values are already
//!   in Montgomery form, so we use `montgomery_reduce_9` after accumulation.
//!
//! - **Barrett reduction**: Used for field × small_int products. Small integers
//!   are NOT in Montgomery form, so Barrett reduction works directly on raw limbs.

pub(crate) mod barrett;
pub(crate) mod delayed_reduction;
pub(crate) mod field_reduction_constants;
pub(crate) mod macros;
pub(crate) mod montgomery;
pub(crate) mod small_value_field;
pub(crate) mod wide_mul;

mod limbs;

// Re-exports: traits
pub use delayed_reduction::DelayedReduction;
pub use field_reduction_constants::{BarrettReductionConstants, FieldReductionConstants};
pub use small_value_field::SmallValueField;
#[allow(unused_imports)]
pub use wide_mul::WideMul;

// Re-exports: types
#[allow(unused_imports)]
pub use limbs::{SignedWideLimbs, SubMagResult, WideLimbs, sub_mag};
