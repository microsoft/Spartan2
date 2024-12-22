//! Implements serialization helpers for compatibility with ark-serialize

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Deserializer, Serializer};
use serde_with::{DeserializeAs, SerializeAs};

pub struct Canonical<T>(std::marker::PhantomData<T>);

impl<T> SerializeAs<T> for Canonical<T>
where
  T: CanonicalSerialize,
{
  fn serialize_as<S>(source: &T, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    let mut buffer = Vec::new();
    source
      .serialize_compressed(&mut buffer)
      .map_err(serde::ser::Error::custom)?;
    serializer.serialize_bytes(&buffer)
  }
}

impl<'de, T> DeserializeAs<'de, T> for Canonical<T>
where
  T: CanonicalDeserialize,
{
  fn deserialize_as<D>(deserializer: D) -> Result<T, D::Error>
  where
    D: Deserializer<'de>,
  {
    let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
    T::deserialize_compressed(&*bytes).map_err(serde::de::Error::custom)
  }
}
