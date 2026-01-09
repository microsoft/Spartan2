// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Circuit gadgets optimized for small-value sumcheck.
//!
//! This module provides circuit gadgets that are designed to work with the
//! small-value sumcheck optimization. The key difference from bellpepper's
//! gadgets is that constraint coefficients are bounded to fit in native integers.
//!
//! # Available Gadgets
//!
//! - [`SmallMultiEq`]: Trait for batched equality constraints with bounded coefficients
//! - [`NoBatchEq`]: Direct enforcement (for i32 path)
//! - [`BatchingEq`]: Batched enforcement (for i64 path)
//! - [`SmallUInt32`]: 32-bit unsigned integer with bit operations
//! - [`small_sha256`]: SHA-256 function using small-value compatible gadgets

mod addmany;
mod small_multi_eq;
mod small_sha256;
mod small_uint32;

pub use small_multi_eq::{BatchingEq, NoBatchEq, SmallMultiEq};
pub use small_sha256::{small_sha256, small_sha256_with_prefix, small_sha256_with_small_multi_eq};
pub use small_uint32::SmallUInt32;
