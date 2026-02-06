// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! CLI utilities for examples and benchmarks.

use clap::ValueEnum;

/// Field choice for benchmarks
#[derive(ValueEnum, Clone, Default, Debug)]
pub enum FieldChoice {
  /// Pallas curve scalar field (Fq)
  #[default]
  PallasFq,
  /// Vesta curve scalar field (Fp)
  VestaFp,
  /// BN254 curve scalar field (Fr)
  Bn254Fr,
}
