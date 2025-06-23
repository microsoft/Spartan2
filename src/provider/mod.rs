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
    pcs::{
      hyrax_pc::{HyraxCommitmentEngine, HyraxEvaluationEngine},
      ipa_pc::{CommitmentEngine as IPACommitmentEngine, EvaluationEngine as IPAEvaluationEngine},
    },
    pt256::{p256, t256},
  },
  traits::Engine,
};
use serde::{Deserialize, Serialize};

/// An implementation of the Spartan `Engine` trait with Pallas curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasHyraxEngine;

/// An implementation of the Spartan `Engine` trait with Vesta curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaHyraxEngine;

/// An implementation of the Spartan `Engine` trait with P256 curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct P256HyraxEngine;

/// An implementation of the Spartan `Engine` trait with T256 curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct T256HyraxEngine;

impl Engine for PallasHyraxEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type TE = Keccak256Transcript<Self>;
  type CE = HyraxCommitmentEngine<Self>;
  type EE = HyraxEvaluationEngine<Self>;
}

impl Engine for VestaHyraxEngine {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type GE = vesta::Point;
  type TE = Keccak256Transcript<Self>;
  type CE = HyraxCommitmentEngine<Self>;
  type EE = HyraxEvaluationEngine<Self>;
}

impl Engine for P256HyraxEngine {
  type Base = p256::Base;
  type Scalar = p256::Scalar;
  type GE = p256::Point;
  type TE = Keccak256Transcript<Self>;
  type CE = HyraxCommitmentEngine<Self>;
  type EE = HyraxEvaluationEngine<Self>;
}

impl Engine for T256HyraxEngine {
  type Base = t256::Base;
  type Scalar = t256::Scalar;
  type GE = t256::Point;
  type TE = Keccak256Transcript<Self>;
  type CE = HyraxCommitmentEngine<Self>;
  type EE = HyraxEvaluationEngine<Self>;
}

/// An implementation of the Spartan `Engine` trait with Pallas curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasIPAEngine;

/// An implementation of the Spartan `Engine` trait with Vesta curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaIPAEngine;

/// An implementation of the Spartan `Engine` trait with P256 curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct P256IPAEngine;

/// An implementation of the Spartan `Engine` trait with T256 curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct T256IPAEngine;

impl Engine for PallasIPAEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type TE = Keccak256Transcript<Self>;
  type CE = IPACommitmentEngine<Self>;
  type EE = IPAEvaluationEngine<Self>;
}

impl Engine for VestaIPAEngine {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type GE = vesta::Point;
  type TE = Keccak256Transcript<Self>;
  type CE = IPACommitmentEngine<Self>;
  type EE = IPAEvaluationEngine<Self>;
}

impl Engine for P256IPAEngine {
  type Base = p256::Base;
  type Scalar = p256::Scalar;
  type GE = p256::Point;
  type TE = Keccak256Transcript<Self>;
  type CE = IPACommitmentEngine<Self>;
  type EE = IPAEvaluationEngine<Self>;
}

impl Engine for T256IPAEngine {
  type Base = t256::Base;
  type Scalar = t256::Scalar;
  type GE = t256::Point;
  type TE = Keccak256Transcript<Self>;
  type CE = IPACommitmentEngine<Self>;
  type EE = IPAEvaluationEngine<Self>;
}

#[cfg(test)]
mod tests {
  use crate::provider::{pasta::pallas, traits::DlogGroup};
  use digest::{ExtendableOutput, Update};
  use halo2curves::{CurveExt, group::Curve};
  use sha3::Shake256;
  use std::io::Read;

  macro_rules! impl_cycle_pair_test {
    ($curve:ident) => {
      fn from_label_serial(label: &'static [u8], n: usize) -> Vec<$curve::Affine> {
        let mut shake = Shake256::default();
        shake.update(label);
        let mut reader = shake.finalize_xof();
        (0..n)
          .map(|_| {
            let mut uniform_bytes = [0u8; 32];
            reader.read_exact(&mut uniform_bytes).unwrap();
            let hash = $curve::Point::hash_to_curve("from_uniform_bytes");
            hash(&uniform_bytes).to_affine()
          })
          .collect()
      }

      let label = b"test_from_label";
      for n in [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1021,
      ] {
        let ck_par = <$curve::Point as DlogGroup>::from_label(label, n);
        let ck_ser = from_label_serial(label, n);
        assert_eq!(ck_par.len(), n);
        assert_eq!(ck_ser.len(), n);
        assert_eq!(ck_par, ck_ser);
      }
    };
  }

  #[test]
  fn test_pallas_from_label() {
    impl_cycle_pair_test!(pallas);
  }
}
