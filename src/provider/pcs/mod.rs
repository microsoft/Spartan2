// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module provides implementations of polynomial commitment schemes (PCS).

// helper code for polynomial commitment schemes
pub mod ipa;

// implementations of polynomial commitment schemes
pub mod hyrax_pc;

/// Dory-PC polynomial commitment scheme adapter using quarks-zk
#[cfg(feature = "dory")]
pub mod dory_pc;
