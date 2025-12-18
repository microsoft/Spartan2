# Spartan2 Coding Style Guide

This document summarizes the coding conventions and style patterns used in this codebase.

## File Headers

All Rust source files must include this copyright header:

```rust
// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2
```

## Project Structure

```
src/
├── lib.rs              # Crate root with lint config
├── spartan.rs          # Main Spartan SNARK
├── spartan_zk.rs       # Spartan with zero-knowledge
├── neutronnova_zk.rs   # NeutronNova folding scheme
├── sumcheck.rs         # Sum-check protocol
├── polys/              # Polynomial implementations
├── r1cs/               # R1CS constraint system
├── nifs/               # Non-interactive folding
├── provider/           # Crypto backends (curves, PCS)
├── traits/             # Core traits
├── errors.rs           # Error definitions
└── macros.rs           # Helper macros
examples/               # Example circuits
benches/                # Benchmarks
```

## Lint Configuration

```rust
#![deny(warnings, unused, future_incompatible, nonstandard_style, rust_2018_idioms, missing_docs)]
#![forbid(unsafe_code)]
#![allow(non_snake_case)]           // For mathematical variables (A, B, C matrices)
#![allow(clippy::upper_case_acronyms)]  // For PCS, R1CS, etc.
#![allow(clippy::type_complexity)]      // Complex associated types are expected
#![allow(clippy::too_many_arguments)]   // Protocol functions have many params
```

## Formatting (rustfmt.toml)

```toml
edition                  = "2024"
style_edition            = "2024"
newline_style            = "Unix"
tab_spaces               = 2          # 2-space indentation
max_width                = 100
use_try_shorthand        = true
reorder_imports          = true
merge_derives            = true
use_field_init_shorthand = true
imports_granularity      = "Crate"
```

## Naming Conventions

### Types
- PascalCase: `SpartanSNARK`, `MultilinearPolynomial`, `SumcheckProof`
- Traits end with `Trait`: `R1CSSNARKTrait`, `PCSEngineTrait`

### Functions
- snake_case with verb prefix: `compute_eval_points_quad`, `bind_poly_var_top`
- Common prefixes: `compute_`, `evaluate_`, `bind_`, `prove_`, `verify_`, `prep_`

### Variables
- snake_case for locals: `claim_per_round`, `num_rounds_x`
- Single letters for math: `r`, `tau`, `e`, `x`
- SCREAMING_SNAKE_CASE for constants: `PAR_THRESHOLD`

### Generics
- Single capitals: `E` (Engine), `F` (Field), `R` (RngCore)
- Type aliases for complex associated types:
```rust
type CommitmentKey<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::CommitmentKey;
```

## Documentation

### Module-level (`//!`)
```rust
//! This module implements the sum-check protocol used in Spartan.
//!
//! The sum-check protocol allows a prover to convince a verifier...
```

### Type/Function-level (`///`)
```rust
/// Proves the sum-check relation for a quadratic polynomial.
///
/// # Arguments
/// * `claim` - The claimed sum value
/// * `poly` - The multilinear polynomial
///
/// # Returns
/// A tuple of (proof, challenges, evaluations)
```

### Inline comments
- Explain complex algorithms with paper references
- Document invariants: `// Invariant: self.round is always >= 1`
- Note optimizations: `// Use incremental computation to avoid O(n) work`

## Error Handling

Use `thiserror` with a single error enum:

```rust
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SpartanError {
  #[error("InvalidIndex")]
  InvalidIndex,

  #[error("UnSat: {reason}")]
  UnSat { reason: String },

  #[error("InvalidVectorSize")]
  InvalidVectorSize { actual: usize, max: usize },
}
```

Return `Result<T, SpartanError>` from fallible functions.

## Import Organization

1. Crate imports first (grouped hierarchically)
2. External crates alphabetically
3. Standard library last

```rust
use crate::{
  errors::SpartanError,
  polys::multilinear::MultilinearPolynomial,
  traits::Engine,
  start_span,
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
```

## Parallelism

- Use Rayon for parallel iteration
- Check threshold before spawning: `if len > PAR_THRESHOLD { par_iter() } else { iter() }`
- Use `rayon::join()` for two-way parallelism
- Custom `zip_with!` macro for clean parallel zips

```rust
// Good
rayon::join(|| compute_left(), || compute_right());

// Good
evals.par_iter_mut().for_each(|e| *e = e.square());
```

## Serialization

```rust
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]  // For generic types
pub struct Proof<E: Engine> {
  commitment: Commitment<E>,
  #[serde(skip, default = "OnceCell::new")]
  cached_digest: OnceCell<E::Scalar>,
}
```

## Tracing/Timing

```rust
use tracing::{debug, info};

let (_span, timer) = start_span!("operation_name");
// ... work ...
info!(elapsed_ms = %timer.elapsed().as_millis(), "operation_name");
```

## Performance Patterns

- `#[inline]` for small hot functions
- Pre-allocate: `Vec::with_capacity(n)`
- In-place mutation: `fn bind(&mut self, r: F)`
- Incremental computation over recomputation
- Comments explaining algorithmic complexity

## Testing

```rust
#[cfg(test)]
mod tests {
  use super::*;

  // Generic test helper
  fn test_snark_with<E: Engine, S: R1CSSNARKTrait<E>>() {
    // Setup tracing
    let _ = tracing_subscriber::fmt()
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();

    // Test logic...
  }

  #[test]
  fn test_spartan_snark() {
    test_snark_with::<PallasHyraxEngine, SpartanSNARK<_>>();
  }
}
```

## Mathematical Code

- `#![allow(non_snake_case)]` permits uppercase for matrices: `A`, `B`, `C`, `Z`
- Use LaTeX-style notation in comments: `// eq(τ, x) = Π_i ((1-τ_i)(1-x_i) + τ_i·x_i)`
- Reference papers: `// See Algorithm 5 in [BDDT24]`
