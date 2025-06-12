//! This module implements Nova's traits using the following several different combinations

// public modules to be used as an evaluation engine with Spartan
pub mod ipa_pc;
pub mod pasta;

pub(crate) mod keccak;
pub(crate) mod pedersen;
pub(crate) mod traits;

mod msm;

use crate::{
  provider::{
    keccak::Keccak256Transcript,
    pasta::{pallas, vesta},
    pedersen::CommitmentEngine as PedersenCommitmentEngine,
  },
  traits::Engine,
};
use serde::{Deserialize, Serialize};

/// An implementation of the Nova `Engine` trait with Pallas curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasEngine;

/// An implementation of the Nova `Engine` trait with Vesta curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaEngine;

impl Engine for PallasEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type TE = Keccak256Transcript<Self>;
  type CE = PedersenCommitmentEngine<Self>;
}

impl Engine for VestaEngine {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type GE = vesta::Point;
  type TE = Keccak256Transcript<Self>;
  type CE = PedersenCommitmentEngine<Self>;
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
