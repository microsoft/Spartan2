mod arithmetic;
pub mod brakedown;

pub trait LinearCodes<F>: Sync + Send {
  fn row_len(&self) -> usize;

  fn codeword_len(&self) -> usize;

  fn num_column_opening(&self) -> usize;

  fn num_proximity_testing(&self) -> usize;

  fn encode(&self, input: impl AsMut<[F]>);
}
