use ff::Field;

pub fn horner<F: Field>(coeffs: &[F], x: &F) -> F {
  coeffs
    .iter()
    .rev()
    .fold(F::ZERO, |acc, coeff| acc * x + coeff)
}
pub fn steps<F: Field>(start: F) -> impl Iterator<Item = F> {
  steps_by(start, F::ONE)
}
pub fn steps_by<F: Field>(start: F, step: F) -> impl Iterator<Item = F> {
  std::iter::successors(Some(start), move |state| Some(step + state))
}
