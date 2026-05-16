// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Circuit gadgets optimized for small-value NeutronNova benchmarks.
//!
//! These gadgets avoid Bellpepper's large-coefficient `MultiEq` batching for
//! SHA-256 compression, keeping generated constraint coefficients compatible
//! with full-small accumulator NIFS.

mod addmany;
pub mod small_boolean;
mod small_multi_eq;
mod small_sha256;
mod small_uint32;

pub use small_boolean::{SmallBit, SmallBoolean};
pub use small_multi_eq::{NoBatchEq, SmallMultiEq};
pub use small_sha256::{
  small_sha256_compression_function_int, small_sha256_compression_function_int_with_prefix,
  small_sha256_int, small_sha256_int_with_prefix,
};
pub use small_uint32::SmallUInt32;
