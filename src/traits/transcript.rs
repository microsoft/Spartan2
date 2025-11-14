//! This module provides the trait definitions for transcript functionality in the Spartan2 library.
//! Transcripts are used for Fiat-Shamir transformations to make interactive proof systems non-interactive.
use crate::{
  errors::SpartanError,
  traits::{Engine, Group},
};

/// This trait allows types to implement how they want to be added to `TranscriptEngine`
pub trait TranscriptReprTrait<G: Group>: Send + Sync {
  /// returns a byte representation of self to be added to the transcript
  fn to_transcript_bytes(&self) -> Vec<u8>;
}

/// This trait defines the behavior of a transcript engine compatible with Spartan
pub trait TranscriptEngineTrait<E: Engine>: Send + Sync {
  /// initializes the transcript
  fn new(label: &'static [u8]) -> Self;

  /// returns a scalar element of the group as a challenge
  fn squeeze(&mut self, label: &'static [u8]) -> Result<E::Scalar, SpartanError>;

  /// returns a vector of scalar element's of the group as a challenge
  fn squeeze_scalar_powers(
    &mut self,
    len: usize,
    label: &'static [u8],
  ) -> Result<Vec<E::Scalar>, SpartanError> {
    let r = self.squeeze(label)?;
    let mut r_vec = vec![r; len];
    for i in 1..len {
      r_vec[i] = r_vec[i - 1] * r;
    }
    Ok(r_vec)
  }

  /// absorbs any type that implements `TranscriptReprTrait` under a label
  fn absorb<T: TranscriptReprTrait<E::GE>>(&mut self, label: &'static [u8], o: &T);

  /// adds a domain separator
  fn dom_sep(&mut self, bytes: &'static [u8]);
}

impl<G: Group, T: TranscriptReprTrait<G>> TranscriptReprTrait<G> for &[T] {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self
      .iter()
      .flat_map(|t| t.to_transcript_bytes())
      .collect::<Vec<u8>>()
  }
}
