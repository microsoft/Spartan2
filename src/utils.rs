use num_integer::Integer;

/// Parallel utilities for processing large vectors in parallel using Rayon. The parallelization is controlled by the "parallel" feature.
pub mod parallel;

/// Integer division that rounds up
pub fn div_ceil(dividend: usize, divisor: usize) -> usize {
  Integer::div_ceil(&dividend, &divisor)
}
