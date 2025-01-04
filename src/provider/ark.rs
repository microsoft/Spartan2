use crate::provider::keccak::Keccak256Transcript;
use crate::provider::pedersen::CommitmentEngine;
use crate::traits::{CompressedGroup, Group, PrimeFieldExt, TranscriptReprTrait};
use ark_bls12_381::g1::Config as G1Config;
use ark_bls12_381::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::hashing::{
  curve_maps::wb::WBMap, map_to_curve_hasher::MapToCurveBasedHasher, HashToCurve,
};
use ark_ec::short_weierstrass::SWCurveConfig;
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::field_hashers::DefaultFieldHasher;
use ark_ff::{AdditiveGroup, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_bigint::BigInt;
use num_traits::Zero;
use rayon::prelude::*;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha3::{
  digest::{ExtendableOutput, Update},
  Shake256,
};
use std::io::Read;

/// Compressed representation of BLS12-381 group elements
#[derive(Clone, Copy, Debug, Eq, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub struct BLS12CompressedElement {
  repr: [u8; 48],
}

impl Serialize for BLS12CompressedElement {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    let mut state = serializer.serialize_struct("BLS12CompressedElement", 1)?;
    state.serialize_field("repr", &self.repr.as_slice())?;
    state.end()
  }
}

impl<'de> Deserialize<'de> for BLS12CompressedElement {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let repr: [u8; 48] = {
      let repr_vec = Vec::<u8>::deserialize(deserializer)?;
      if repr_vec.len() != 48 {
        return Err(serde::de::Error::custom(
          "Invalid length for BLS12CompressedElement repr",
        ));
      }
      let mut repr_arr = [0u8; 48];
      repr_arr.copy_from_slice(&repr_vec);
      repr_arr
    };
    Ok(BLS12CompressedElement::new(repr))
  }
}

impl BLS12CompressedElement {
  /// Creates a new `BLS12CompressedElement`
  pub const fn new(repr: [u8; 48]) -> Self {
    Self { repr }
  }
}

impl<G: Group> TranscriptReprTrait<G> for BLS12CompressedElement {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.repr.to_vec()
  }
}

impl<G: Group> TranscriptReprTrait<G> for Fq {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut serialized_data = Vec::new();
    self
      .serialize_compressed(&mut serialized_data)
      .expect("Serialization failed");
    serialized_data
  }
}

impl<G: Group> TranscriptReprTrait<G> for Fr {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut serialized_data = Vec::new();
    self
      .serialize_compressed(&mut serialized_data)
      .expect("Serialization failed");
    serialized_data
  }
}

impl Group for G1Projective {
  type Base = Fq;
  type Scalar = Fr;
  type CompressedGroupElement = BLS12CompressedElement;
  type PreprocessedGroupElement = G1Affine;
  type TE = Keccak256Transcript<Self>;
  type CE = CommitmentEngine<Self>;

  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self {
    // TODO: Properly handle error
    VariableBaseMSM::msm(bases, scalars).expect("Failed to perform MSM")
  }

  fn compress(&self) -> Self::CompressedGroupElement {
    let mut repr = Vec::new();
    self
      .into_affine()
      .serialize_compressed(&mut repr)
      .expect("Serialization should not fail");
    BLS12CompressedElement::new(
      repr
        .try_into()
        .expect("Serialized data has incorrect length"),
    )
  }

  fn preprocessed(&self) -> Self::PreprocessedGroupElement {
    self.into_affine()
  }

  fn from_label(label: &[u8], n: usize) -> Vec<G1Affine> {
    let mut shake = Shake256::default();
    shake.update(label);

    let mut reader = shake.finalize_xof();

    let mut uniform_bytes_vec = Vec::new();
    for _ in 0..n {
      let mut uniform_bytes = [0u8; 32];
      reader
        .read_exact(&mut uniform_bytes)
        .expect("Failed to read bytes from XOF");
      uniform_bytes_vec.push(uniform_bytes);
    }

    let hasher = MapToCurveBasedHasher::<
      G1Projective,
      DefaultFieldHasher<sha2::Sha256, 128>,
      WBMap<G1Config>,
    >::new(b"from_uniform_bytes")
    .expect("Failed to create MapToCurveBasedHasher");

    let ck_proj = uniform_bytes_vec
      .into_par_iter()
      .map(|bytes| hasher.hash(&bytes).expect("Hashing to curve failed"))
      .collect::<Vec<_>>();

    let mut ck_affine = vec![G1Affine::identity(); n];
    for (proj, affine) in ck_proj.iter().zip(ck_affine.iter_mut()) {
      *affine = (*proj).into();
    }

    ck_affine
  }

  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
    let affine = self.into_affine();
    if affine.is_zero() {
      (Self::Base::zero(), Self::Base::zero(), true)
    } else {
      let coords = affine
        .xy()
        .expect("Point is not at infinity; coordinates must exist");
      (coords.0, coords.1, false)
    }
  }

  fn zero() -> Self {
    G1Projective::ZERO
  }
  fn get_generator() -> Self {
    G1Projective::generator()
  }

  fn get_curve_params() -> (Self::Base, Self::Base, BigInt) {
    let a = G1Config::COEFF_A;
    let b = G1Config::COEFF_B;
    let order = Fr::MODULUS.into();
    (a, b, order)
  }
}

impl CompressedGroup for BLS12CompressedElement {
  type GroupElement = G1Projective;

  fn decompress(&self) -> Option<G1Projective> {
    G1Affine::deserialize_compressed(&self.repr[..])
      .ok()
      .map(Into::into)
  }
}

impl PrimeFieldExt for Fr {
  fn from_uniform(bytes: &[u8]) -> Self {
    Self::from_be_bytes_mod_order(bytes.try_into().unwrap())
  }
}
