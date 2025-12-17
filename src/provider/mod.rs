// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module implements Spartan's traits using the following several different combinations

// public modules to be used as an commitment engine with Spartan
pub mod bls12_381;
pub mod keccak;
pub mod pasta;
pub mod pcs;
pub mod pt256;
pub mod traits;

mod msm;

use crate::{
  provider::{
    bls12_381::g1,
    keccak::Keccak256Transcript,
    pasta::{pallas, vesta},
    pcs::hyrax_pc::HyraxPCS,
    pt256::{p256, t256},
  },
  traits::Engine,
};
use core::fmt::Debug;
use serde::{Deserialize, Serialize};

#[cfg(feature = "dory")]
use crate::provider::pcs::dory_pc::DoryPCS;

/// An implementation of the Spartan Engine trait with Pallas curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasHyraxEngine;

/// An implementation of the Spartan Engine trait with Vesta curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaHyraxEngine;

/// An implementation of the Spartan Engine trait with P256 curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct P256HyraxEngine;

/// An implementation of the Spartan Engine trait with T256 curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct T256HyraxEngine;

/// An implementation of the Spartan Engine trait with BLS12-381 G1 curve and Hyrax commitment scheme
///
/// BLS12-381 is a pairing-friendly curve offering ~128 bits of security.
/// This engine uses the G1 subgroup for commitments with Hyrax-PC.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BLS12381HyraxEngine;

impl Engine for PallasHyraxEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

impl Engine for VestaHyraxEngine {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type GE = vesta::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

impl Engine for P256HyraxEngine {
  type Base = p256::Base;
  type Scalar = p256::Scalar;
  type GE = p256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

impl Engine for T256HyraxEngine {
  type Base = t256::Base;
  type Scalar = t256::Scalar;
  type GE = t256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

impl Engine for BLS12381HyraxEngine {
  type Base = g1::Base;
  type Scalar = g1::Scalar;
  type GE = g1::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

/// An implementation of the Spartan Engine trait with BLS12-381 curve and Dory commitment scheme
///
/// Dory-PC provides O(log n) verification complexity using pairings.
/// This engine wraps quarks-zk's DoryPCS implementation.
#[cfg(feature = "dory")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BLS12381DoryEngine;

#[cfg(feature = "dory")]
impl Engine for BLS12381DoryEngine {
  type Base = g1::Base;
  type Scalar = g1::Scalar;
  type GE = g1::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = DoryPCS<Self>;
}
