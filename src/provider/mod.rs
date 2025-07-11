//! This module implements Spartan's traits using the following several different combinations

// public modules to be used as an commitment engine with Spartan
pub mod keccak;
pub mod pasta;
pub mod pcs;
pub mod pt256;
pub mod traits;

mod msm;

use crate::{
  provider::{
    keccak::Keccak256Transcript,
    pasta::{pallas, vesta},
    pcs::hyrax_pc::HyraxPCS,
    pt256::{p256, t256},
  },
  traits::Engine,
};
use core::fmt::Debug;
use serde::{Deserialize, Serialize};

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

/*
/// An implementation of the Spartan Engine trait with Pallas curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasIPAEngine;

/// An implementation of the Spartan Engine trait with Vesta curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaIPAEngine;

/// An implementation of the Spartan Engine trait with P256 curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct P256IPAEngine;

/// An implementation of the Spartan Engine trait with T256 curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct T256IPAEngine;

impl Engine for PallasIPAEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = IPAPCS<Self>;
}

impl Engine for VestaIPAEngine {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type GE = vesta::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = IPAPCS<Self>;
}

impl Engine for P256IPAEngine {
  type Base = p256::Base;
  type Scalar = p256::Scalar;
  type GE = p256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = IPAPCS<Self>;
}

impl Engine for T256IPAEngine {
  type Base = t256::Base;
  type Scalar = t256::Scalar;
  type GE = t256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = IPAPCS<Self>;
}
*/
