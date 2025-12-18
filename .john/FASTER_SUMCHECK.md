# Algorithm 6: Faster Sumcheck for Spartan

## Overview

This document describes the implementation plan for Algorithm 6 (EqPoly-SmallValueSC) from the paper "Speeding Up Sum-Check Proving". The algorithm combines:
- **Small-value optimization** (accumulators for rounds 1-ℓ₀)
- **Eq-poly optimization** (Algorithm 5 for rounds ℓ₀+2 onwards)

### Target Form (Spartan Outer Sumcheck)

```
g(X) = eq(τ, X) · (Az(X) · Bz(X) - Cz(X))
```

- Degree: 3 (cubic)
- Evaluation points: `U_d = {∞, 0, 1, 2}` where `d = 3`, `|U_d| = 4`

---

## Protocol Flow: τ vs r Challenges

**Critical insight:** τ (taus) are NOT set by the prover arbitrarily. They are derived from the **transcript via Fiat-Shamir BEFORE the sumcheck begins**.

### Timeline in Spartan (`src/spartan.rs`)

```rust
// 1. Prover commits to witness W
let (comm_W, blinds_W) = W.commit::<E::CE>(ck)?;

// 2. Commitment absorbed into transcript
transcript.absorb(b"C", &comm_W);

// 3. τ squeezed from transcript BEFORE sumcheck starts
let mut tau = transcript.squeeze(b"t")?;   // τ[0..log_m]
tau.extend(transcript.squeeze(b"t")?);      // τ[log_m..2*log_m]
// ^^^ ALL of τ is now known! ^^^

// 4. Compute Az, Bz, Cz (prover's work)
let (poly_Az, poly_Bz, poly_Cz) = pk.S.multiply_vec_par(&z)?;

// 5. NOW the sumcheck begins
let (sc_proof_outer, r_x, claims_outer) =
    SumcheckProof::prove_cubic_with_three_inputs(
        &E::Scalar::ZERO,
        tau,                // <-- ALL of τ passed in
        &mut poly_Az,
        &mut poly_Bz,
        &mut poly_Cz,
        &mut transcript,
    )?;
```

### Two Types of Challenges

| Challenge | When Known | Source | Used For |
|-----------|------------|--------|----------|
| **τ (taus)** | BEFORE sumcheck | Transcript (after commitment) | eq(τ, X) polynomial |
| **r_i** | Round i of sumcheck | Transcript (after prover message) | Binding variables |

### Why This Enables Precomputation

Since **ALL of τ is known before the sumcheck starts**, we can:

1. **Precompute E_in and E_out tables** - These only depend on τ
2. **Precompute ALL accumulators A_1, ..., A_ℓ₀** - These depend on τ and input polynomials (both known)
3. **Precompute eq_tau_0_2_3** - The eq(τ_i, 0/2/3) factors for each round

The **R_i vector** is the only thing that depends on r challenges, and it's built incrementally as rounds progress.

### API Design: Match Existing Interface

The new Algorithm 6 prover has the **exact same signature** as the existing prover:

```rust
// Existing (Algorithm 5)
pub fn prove_cubic_with_three_inputs(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,           // ← τ passed in (already known)
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    poly_C: &mut MultilinearPolynomial<E::Scalar>,
    transcript: &mut E::TE,
) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError>

// New (Algorithm 6) - SAME signature, drop-in replacement
pub fn prove_cubic_with_three_inputs_alg6(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,           // ← Same: τ passed in
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    poly_C: &mut MultilinearPolynomial<E::Scalar>,
    transcript: &mut E::TE,
) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError>
```

Returns:
- `SumcheckProof` - compressed round polynomials (identical format)
- `Vec<E::Scalar>` - r challenges collected during protocol
- `Vec<E::Scalar>` - final evaluations `[Az(r), Bz(r), Cz(r)]`

---

## Evaluation Domains: U_d and Û_d

The algorithm uses two related evaluation domains:

| Domain | Definition | Size | Used For |
|--------|------------|------|----------|
| **U_d** | {∞, 0, 1, 2, ..., d-1} | d+1 | Prefix v in accumulators, extended poly evals |
| **Û_d** | U_d \ {1} = {∞, 0, 2, ..., d-1} | d | Coordinate u in accumulators |

For d=3 (cubic):
- **U_d = {∞, 0, 1, 2}** — 4 elements
- **Û_d = {∞, 0, 2}** — 3 elements (1 is excluded)

**Why exclude 1 from Û_d?** The evaluation at u=1 can be recovered from the sum-check constraint `s(0) + s(1) = claim`, so we don't need to store/compute it explicitly.

---

## The ∞ Point (Leading Coefficient)

The ∞ point is **NOT** a literal field element — it represents the **leading coefficient** of a polynomial.

### For a Linear Polynomial (degree 1)

A linear polynomial `p(X) = aX + b` has:
- `p(0) = b`
- `p(1) = a + b`
- `p(∞) = a` ← the leading coefficient (slope)

In practice: **`p(∞) = p(1) - p(0)`**

### For a Degree-d Polynomial

For `p(X) = a_d X^d + a_{d-1} X^{d-1} + ... + a_0`:
- `p(∞) = a_d` (the leading coefficient)

Intuitively: as X → ∞, the highest-degree term dominates, so `p(X)/X^d → a_d`.

### Why Include ∞?

To uniquely determine a degree-d polynomial, we need d+1 points. Using {∞, 0, 1, ..., d-1} is convenient because for multilinear polynomials, `p(∞) = p(1) - p(0)` requires no extra computation.

### Index Encoding

Our encoding maps indices to evaluation points:
```
Index 0 → ∞ (leading coefficient)
Index 1 → evaluation at 0
Index 2 → evaluation at 1
Index 3 → evaluation at 2
...
Index k+1 → evaluation at k
```

In `extend_top_var_to_ud`, the first chunk computes the ∞ evaluations:
```rust
// left = p(0, ...), right = p(1, ...)
// Index 0 represents ∞:
result.push(right[i] - left[i]);  // This IS p(∞, suffix)
```

---

## Core Data Structures

### 0. Domain Types

The evaluation domain U_d = {∞, 0, 1, ..., d-1} uses ∞ as a special point representing the "leading coefficient" of a polynomial. To avoid confusion between index 0 and the value 0, we use explicit types.

We define two domain types:
- `UdPoint` — a point in U_d = {∞, 0, 1, ..., d-1} (d+1 elements)
- `UdHatPoint` — a point in Û_d = U_d \ {1} = {∞, 0, 2, ..., d-1} (d elements)

The reduced domain Û_d excludes value 1 because s(1) can be recovered from the sum-check constraint s(0) + s(1) = claim.

```rust
/// A point in the domain U_d = {∞, 0, 1, ..., d-1}
///
/// The domain has d+1 points. The ∞ point represents evaluation of the
/// leading coefficient (see Lemma 2.2 in the paper).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UdPoint {
    /// The point at infinity — represents leading coefficient
    Infinity,
    /// A finite field value 0, 1, ..., d-1
    Finite(usize),
}

impl UdPoint {
    /// Convert to flat index for array access.
    /// Infinity → 0, Finite(v) → v + 1
    #[inline]
    pub fn to_index(self) -> usize {
        match self {
            UdPoint::Infinity => 0,
            UdPoint::Finite(v) => v + 1,
        }
    }

    /// Convert from flat index.
    /// 0 → Infinity, k → Finite(k - 1)
    #[inline]
    pub fn from_index(idx: usize) -> Self {
        if idx == 0 {
            UdPoint::Infinity
        } else {
            UdPoint::Finite(idx - 1)
        }
    }

    /// Is this a binary point (0 or 1)?
    #[inline]
    pub fn is_binary(self) -> bool {
        matches!(self, UdPoint::Finite(0) | UdPoint::Finite(1))
    }

    /// Convert to field element. Returns `None` for Infinity.
    #[inline]
    pub fn to_field<F: PrimeField>(self) -> Option<F> {
        match self {
            UdPoint::Infinity => None,
            UdPoint::Finite(v) => Some(F::from(v as u64)),
        }
    }

    /// Convert to Û_d point (the reduced domain excluding value 1).
    ///
    /// Returns `None` for Finite(1) since 1 ∉ Û_d.
    #[inline]
    pub fn to_ud_hat(self) -> Option<UdHatPoint> {
        UdHatPoint::try_from(self).ok()
    }
}

/// Error returned when trying to convert `Finite(1)` to `UdHatPoint`.
///
/// The value 1 is excluded from Û_d because s(1) can be recovered
/// from the sum-check constraint s(0) + s(1) = claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueOneExcluded;

impl std::fmt::Display for ValueOneExcluded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "value 1 is not in Û_d (excluded from reduced domain)")
    }
}

impl std::error::Error for ValueOneExcluded {}

/// A point in the reduced domain Û_d = U_d \ {1} = {∞, 0, 2, 3, ..., d-1}
///
/// This domain has d elements (one less than U_d).
/// Value 1 is excluded because s(1) can be recovered from s(0) + s(1) = claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UdHatPoint {
    /// The point at infinity — represents leading coefficient
    Infinity,
    /// A finite field value: 0, 2, 3, ... (never 1)
    Finite(usize),
}

impl UdHatPoint {
    // === Construction ===

    /// Create the infinity point
    pub const fn infinity() -> Self {
        UdHatPoint::Infinity
    }

    /// Create the zero point
    pub const fn zero() -> Self {
        UdHatPoint::Finite(0)
    }

    /// Create a finite point. Returns None for v=1 (not in Û_d).
    pub fn finite(v: usize) -> Option<Self> {
        if v == 1 {
            None
        } else {
            Some(UdHatPoint::Finite(v))
        }
    }

    // === Index Conversion ===

    /// Convert to array index.
    /// Mapping: ∞ → 0, 0 → 1, 2 → 2, 3 → 3, ...
    #[inline]
    pub fn to_index(self) -> usize {
        match self {
            UdHatPoint::Infinity => 0,
            UdHatPoint::Finite(0) => 1,
            UdHatPoint::Finite(k) => k,  // 2→2, 3→3, etc.
        }
    }

    /// Create from array index.
    /// Mapping: 0 → ∞, 1 → 0, 2 → 2, 3 → 3, ...
    #[inline]
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => UdHatPoint::Infinity,
            1 => UdHatPoint::Finite(0),
            k => UdHatPoint::Finite(k),
        }
    }

    // === Domain Conversion ===

    /// Convert to UdPoint (U_d point)
    #[inline]
    pub fn to_ud_point(self) -> UdPoint {
        match self {
            UdHatPoint::Infinity => UdPoint::Infinity,
            UdHatPoint::Finite(v) => UdPoint::Finite(v),
        }
    }

    /// Convert to field element. Returns None for Infinity.
    #[inline]
    pub fn to_field<F: PrimeField>(self) -> Option<F> {
        match self {
            UdHatPoint::Infinity => None,
            UdHatPoint::Finite(v) => Some(F::from(v as u64)),
        }
    }

    // === Properties ===

    /// Is this the infinity point?
    #[inline]
    pub fn is_infinity(self) -> bool {
        matches!(self, UdHatPoint::Infinity)
    }

    /// Is this the zero point?
    #[inline]
    pub fn is_zero(self) -> bool {
        matches!(self, UdHatPoint::Finite(0))
    }

    // === Iteration ===

    /// Iterate over all points in Û_d for degree d.
    /// Yields: ∞, 0, 2, 3, ..., d-1 (total of d elements)
    pub fn iter(d: usize) -> impl Iterator<Item = UdHatPoint> {
        (0..d).map(UdHatPoint::from_index)
    }
}

// === Trait Implementations ===

impl From<UdHatPoint> for UdPoint {
    fn from(p: UdHatPoint) -> Self {
        p.to_ud_point()
    }
}

impl TryFrom<UdPoint> for UdHatPoint {
    type Error = ValueOneExcluded;

    fn try_from(p: UdPoint) -> Result<Self, Self::Error> {
        match p {
            UdPoint::Infinity => Ok(UdHatPoint::Infinity),
            UdPoint::Finite(1) => Err(ValueOneExcluded),
            UdPoint::Finite(v) => Ok(UdHatPoint::Finite(v)),
        }
    }
}

/// A tuple β ∈ U_d^k — an index into the extended domain.
///
/// Used to index into LagrangeEvals which stores evaluations over U_d^ℓ₀.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UdTuple(pub Vec<UdPoint>);

impl UdTuple {
    /// Number of coordinates
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Check if all coordinates are binary (0 or 1, no ∞)
    ///
    /// Useful for Spartan optimization: binary points yield zero by R1CS relation.
    pub fn is_all_binary(&self) -> bool {
        self.0.iter().all(|p| p.is_binary())
    }

    /// Check if any coordinate is ∞
    ///
    /// Useful for Spartan optimization: Cz term vanishes when any coord is ∞.
    pub fn has_infinity(&self) -> bool {
        self.0.iter().any(|p| matches!(p, UdPoint::Infinity))
    }

    /// Convert to flat index for array access (mixed-radix encoding)
    pub fn to_flat_index(&self, base: usize) -> usize {
        self.0.iter().fold(0, |acc, p| acc * base + p.to_index())
    }

    /// Convert from flat index (mixed-radix decoding)
    pub fn from_flat_index(mut idx: usize, base: usize, len: usize) -> Self {
        let mut points = vec![UdPoint::Infinity; len];
        for i in (0..len).rev() {
            points[i] = UdPoint::from_index(idx % base);
            idx /= base;
        }
        UdTuple(points)
    }
}
```

**Index encoding:**
| UdPoint | Flat Index | Field Value |
|---------------------|------------|-------------|
| `Infinity`          | 0          | (leading coeff) |
| `Finite(0)`         | 1          | 0 |
| `Finite(1)`         | 2          | 1 |
| `Finite(2)`         | 3          | 2 |

### Design Goal: Thread-Local Accumulators with Efficient Merge

For parallel execution, each thread maintains its own `SmallValueAccumulators`. After processing, thread-local results are merged via element-wise addition. This requires:
1. **Nested storage** `Vec<Vec<Scalar>>` matching the mathematical structure: `data[v][u]`
2. **O(1) indexed writes** for the bucketing step
3. **Fast merge** via element-wise addition

### 1. RoundAccumulator (Single Round)

```rust
/// A single round's accumulator A_i(v, u).
///
/// For round i (0-indexed), this stores:
/// - (d+1)^i prefixes (one per v ∈ U_d^i)
/// - Each prefix has d values (one per u ∈ Û_d = {∞, 0, 2, ..., d-1})
///
/// Structure: data[v_idx][u_idx] = A_i(v, u)
/// - Outer Vec: indexed by v ∈ U_d^{i-1} (Lagrange domain prefixes)
/// - Inner Vec: indexed by u ∈ Û_d (evaluation points, excluding u=1)
pub struct RoundAccumulator<Scalar: PrimeField> {
    /// data[v_idx][u_idx] = A_i(v, u)
    data: Vec<Vec<Scalar>>,
    /// Degree parameter (size of Û_d)
    d: usize,
}

impl<Scalar: PrimeField> RoundAccumulator<Scalar> {
    pub fn new(num_prefixes: usize, d: usize) -> Self {
        Self {
            data: vec![vec![Scalar::ZERO; d]; num_prefixes],
            d,
        }
    }

    fn base(&self) -> usize {
        self.d + 1
    }

    /// O(1) indexed accumulation (performance path)
    #[inline]
    pub fn accumulate(&mut self, v_idx: usize, u_idx: usize, value: Scalar) {
        self.data[v_idx][u_idx] += value;
    }

    /// O(1) indexed read (performance path)
    #[inline]
    pub fn get(&self, v_idx: usize, u_idx: usize) -> Scalar {
        self.data[v_idx][u_idx]
    }

    /// Accumulate by domain types (type-safe path)
    #[inline]
    pub fn accumulate_by_domain(
        &mut self,
        v: &UdTuple,
        u: UdHatPoint,
        value: Scalar,
    ) {
        let v_idx = v.to_flat_index(self.base());
        let u_idx = u.to_index();
        self.data[v_idx][u_idx] += value;
    }

    /// Read by domain types (type-safe path)
    #[inline]
    pub fn get_by_domain(&self, v: &UdTuple, u: UdHatPoint) -> Scalar {
        let v_idx = v.to_flat_index(self.base());
        let u_idx = u.to_index();
        self.data[v_idx][u_idx]
    }

    /// Element-wise merge
    pub fn merge(&mut self, other: &Self) {
        for (a_row, b_row) in self.data.iter_mut().zip(&other.data) {
            for (a, b) in a_row.iter_mut().zip(b_row) {
                *a += b;
            }
        }
    }

    /// Number of prefix entries
    pub fn num_prefixes(&self) -> usize {
        self.data.len()
    }
}
```

**Index mapping for u (Û_d with d=3):**
```
u_idx=0 → u=∞ (leading coefficient)
u_idx=1 → u=0
u_idx=2 → u=2
(u=1 is skipped — recovered from constraint)
```

### 2. SmallValueAccumulators (All Rounds - Thread-Local)

```rust
/// Collection of accumulators for all ℓ₀ rounds.
///
/// Each thread gets its own copy during parallel execution.
/// After processing, thread-local copies are merged via `merge()`.
pub struct SmallValueAccumulators<Scalar: PrimeField> {
    /// Number of rounds using small-value optimization
    l0: usize,
    /// Degree parameter (d=3 for cubic)
    d: usize,
    /// rounds[i] contains A_{i+1} (the accumulator for 1-indexed round i+1)
    rounds: Vec<RoundAccumulator<Scalar>>,
}

impl<Scalar: PrimeField> SmallValueAccumulators<Scalar> {
    /// Create a fresh accumulator (used per-thread in fold)
    pub fn new(l0: usize, d: usize) -> Self {
        let base = d + 1;
        let rounds = (0..l0)
            .map(|i| {
                let num_prefixes = base.pow(i as u32);
                RoundAccumulator::new(num_prefixes, d)
            })
            .collect();
        Self { l0, d, rounds }
    }

    /// O(1) accumulation into bucket (round, v_idx, u_idx)
    #[inline]
    pub fn accumulate(&mut self, round: usize, v_idx: usize, u_idx: usize, value: Scalar) {
        self.rounds[round].accumulate(v_idx, u_idx, value);
    }

    /// Read A_i(v, u)
    #[inline]
    pub fn get(&self, round: usize, v_idx: usize, u_idx: usize) -> Scalar {
        self.rounds[round].get(v_idx, u_idx)
    }

    /// Merge another accumulator into this one (for reduce phase)
    pub fn merge(&mut self, other: &Self) {
        for (self_round, other_round) in self.rounds.iter_mut().zip(other.rounds.iter()) {
            self_round.merge(other_round);
        }
    }

    /// Accumulate by domain types (type-safe path)
    ///
    /// # Arguments
    /// * `round` - Round index (0-indexed)
    /// * `v` - Prefix tuple in U_d^round
    /// * `u` - Coordinate in Û_d
    /// * `value` - Value to accumulate
    #[inline]
    pub fn accumulate_by_domain(
        &mut self,
        round: usize,
        v: &UdTuple,
        u: UdHatPoint,
        value: Scalar,
    ) {
        self.rounds[round].accumulate_by_domain(v, u, value);
    }

    /// Read A_i(v, u) by domain types (type-safe path)
    #[inline]
    pub fn get_by_domain(
        &self,
        round: usize,
        v: &UdTuple,
        u: UdHatPoint,
    ) -> Scalar {
        self.rounds[round].get_by_domain(v, u)
    }
}
```

**Sizes for d=3, ℓ₀=3:**

| Round (0-idx) | num_prefixes = (d+1)^i | d (Û_d size) | Total elements |
|---------------|------------------------|--------------|----------------|
| 0             | 4^0 = 1                | 3            | 3              |
| 1             | 4^1 = 4                | 3            | 12             |
| 2             | 4^2 = 16               | 3            | 48             |
| **Total**     |                        |              | **63**         |

### 3. Parallel Fold-Reduce Pattern

The `build_accumulators` function uses Rayon's fold-reduce pattern for efficient parallelization:

```rust
fn build_accumulators<Scalar: PrimeField + Send + Sync>(
    poly_az: &[Scalar],
    poly_bz: &[Scalar],
    poly_cz: &[Scalar],
    taus: &[Scalar],
    l0: usize,
) -> SmallValueAccumulators<Scalar> {

    (0..num_x_out)
        .into_par_iter()
        .fold(
            // Each thread gets its own fresh accumulator
            || SmallValueAccumulators::new(l0, d),

            // Process x_out into thread-local accumulator
            |mut local_acc, x_out| {
                // ... compute tA for this x_out (see full implementation) ...

                // Distribute to buckets via idx4
                for beta_idx in 0..num_betas {
                    let t_a_val = tA[beta_idx];

                    for route in &routes[beta_idx] {
                        let weight = e_y[route.round_0idx()][route.y_idx] * e_xout[x_out];

                        // Direct indexed write - THIS IS THE BUCKETING
                        local_acc.accumulate(
                            route.round_0idx(),
                            route.v_idx,
                            route.u_idx,
                            weight * t_a_val
                        );
                    }
                }
                local_acc
            },
        )
        .reduce(
            // Identity for reduce
            || SmallValueAccumulators::new(l0, d),

            // Merge thread-local results
            |mut a, b| { a.merge(&b); a }
        )
}
```

**Why fold (not map):**

| Pattern | Accumulators Created | Memory |
|---------|---------------------|--------|
| `.map(\|x\| new Acc)` | One per x_out item | O(x_out_size) |
| `.fold(\|\| new Acc, \|acc, x\|)` | One per thread | O(num_threads) |

With `fold`, Rayon reuses the same accumulator across all x_out values assigned to that thread.

**Memory with fold:** 8 threads × 63 elements × 32 bytes ≈ **16 KB** (regardless of x_out_size)

### 4. Lagrange Evaluations (Extended Domain)

```rust
/// Evaluations over U_d^ℓ₀ for Lagrange interpolation
///
/// Extended evaluations from the boolean hypercube {0,1}^ℓ₀ to U_d^ℓ₀,
/// enabling efficient round polynomial computation via Lagrange interpolation.
pub struct LagrangeEvals<Scalar: PrimeField> {
    evals: Vec<Scalar>,  // size (d+1)^ℓ₀
    l0: usize,
    d: usize,
}

impl<Scalar: PrimeField> LagrangeEvals<Scalar> {
    #[inline]
    fn base(&self) -> usize {
        self.d + 1
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.evals.len()
    }

    /// Get evaluation by flat index (performance path)
    #[inline]
    pub fn get(&self, idx: usize) -> Scalar {
        self.evals[idx]
    }

    /// Get evaluation by domain tuple (type-safe path)
    #[inline]
    pub fn get_by_domain(&self, tuple: &UdTuple) -> Scalar {
        self.evals[tuple.to_flat_index(self.base())]
    }

    /// Convert flat index to domain tuple (for debugging/clarity)
    pub fn to_domain_tuple(&self, flat_idx: usize) -> UdTuple {
        UdTuple::from_flat_index(flat_idx, self.base(), self.l0)
    }
}
```

### 5. LagrangeBasisCoeffs (R_i Tensor)

```rust
/// Tracks R_i = ⊗_{j<i} (L_{U_d,k}(r_j))_k, the tensor product of
/// Lagrange basis evaluations at previous challenges.
///
/// After round i, R_i has size (d+1)^{i-1} entries.
/// R_i[nat_{d+1}(v)] = Π_{j=1}^{i-1} L_{v_j}(r_j) for v ∈ U_d^{i-1}
///
/// Key property: When r_j is a domain point k, L_k(r_j) = 1 and L_m(r_j) = 0
/// for m ≠ k, making R_i a one-hot selector vector.
pub struct LagrangeBasisCoeffs<Scalar: PrimeField> {
    d: usize,
    /// Current R_i vector. Starts as [1] for R_1.
    coeffs: Vec<Scalar>,
}

impl<Scalar: PrimeField> LagrangeBasisCoeffs<Scalar> {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            coeffs: vec![Scalar::ONE],  // R_1 = [1]
        }
    }

    #[inline]
    fn base(&self) -> usize {
        self.d + 1
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Scalar {
        self.coeffs[idx]
    }

    /// After receiving challenge r_i, extend: R_{i+1} = R_i ⊗ L_{U_d}(r_i)
    pub fn extend(&mut self, r: &Scalar) {
        let lagrange = lagrange_basis_evals(r, self.d);
        let base = self.base();

        let mut new_coeffs = Vec::with_capacity(self.coeffs.len() * base);
        for &rv in &self.coeffs {
            for &lu in &lagrange {
                new_coeffs.push(rv * lu);
            }
        }
        self.coeffs = new_coeffs;
    }

    pub fn coeffs(&self) -> &[Scalar] {
        &self.coeffs
    }
}
```

---

## Index Encoding/Decoding

### Binary Hypercube Indexing

For `MultilinearPolynomial`, index `i` corresponds to point `(b_{m-1}, ..., b_1, b_0)`:
- Highest bit = top variable (x_1)
- `Z[i]` = evaluation at binary representation of `i`

```rust
/// Encode binary tuple to index
fn binary_tuple_to_index(bits: &[usize]) -> usize {
    bits.iter().fold(0, |acc, &b| (acc << 1) | b)
}

/// Decode index to binary tuple
fn index_to_binary_tuple(mut idx: usize, len: usize) -> Vec<usize> {
    let mut bits = vec![0; len];
    for i in (0..len).rev() {
        bits[i] = idx & 1;
        idx >>= 1;
    }
    bits
}
```

### Mixed-Radix Indexing (U_d^k)

For tuples over `U_d = {∞, 0, 1, ..., d-1}`:
- `∞ → 0`, `0 → 1`, `1 → 2`, ..., `d-1 → d`

See "Helper Functions" section for: `tuple_to_flat_index`, `flat_index_to_tuple`

---

## Lagrange Basis Functions (NEW)

The codebase uses Gaussian elimination for interpolation but does NOT have Lagrange basis evaluation. We need this for the R_i vector update.

```rust
/// Compute L_{U_d,i}(r) for all i, where U_d = {∞, 0, 1, ..., d-1}
///
/// L_{U,i}(X) = Π_{j≠i} (X - x_j) / (x_i - x_j)
///
/// Returns [L_∞(r), L_0(r), L_1(r), ..., L_{d-1}(r)]
pub fn lagrange_basis_evals<Scalar: PrimeField>(r: &Scalar, d: usize) -> Vec<Scalar> {
    let mut result = Vec::with_capacity(d + 1);

    // Precompute (r - j) for j = 0, 1, ..., d-1
    let r_minus: Vec<Scalar> = (0..d)
        .map(|j| *r - Scalar::from(j as u64))
        .collect();

    // L_∞(r): weight for the leading coefficient s(∞) = lc(s)
    // When s(∞) represents the leading coefficient of X^d, we have:
    // s(r) = s(∞) · P(r) + Σ_{k=0}^{d-1} L_k(r) · s(k)
    // where P(r) = Π_{j=0}^{d-1} (r - j)
    //
    // So L_∞(r) = P(r) = Π_{j=0}^{d-1} (r - j)  (no factorial division!)
    //
    // Sanity check for d=2: if s(X) = aX² + bX + c, then s(∞) = a, and
    // s(r) = a·r(r-1) + (1-r)·s(0) + r·s(1) ✓
    let l_inf: Scalar = r_minus.iter().product();
    result.push(l_inf);

    // L_k(r) for k = 0, 1, ..., d-1 (finite points)
    for k in 0..d {
        let mut l_k = Scalar::ONE;
        for j in 0..d {
            if j != k {
                let denom = Scalar::from(k as u64) - Scalar::from(j as u64);
                l_k *= r_minus[j] * denom.invert().unwrap();
            }
        }
        result.push(l_k);
    }

    result
}
```

---

## Procedure 5: Extend Linear Polynomial

Extends evaluations from `{0, 1}` to `U_d = {∞, 0, 1, ..., d-1}`.

**This pattern already exists in the codebase!** See `src/sumcheck/mod.rs:183` and `src/sumcheck/mod.rs:315-330`.

### Existing Pattern (inline in sumcheck)

```rust
// From compute_eval_points_quad (line 183)
// eval 2: -A(low) + 2*A(high)
let a_bound = a_high + a_high - a_low;

// From compute_eval_points_cubic_with_additive_term (lines 315-330)
// eval 2: -A(low) + 2*A(high)
let poly_A_bound_point = a_high + a_high - a_low;

// eval 3: -2*A(low) + 3*A(high), computed incrementally from eval 2
let poly_A_bound_point = poly_A_bound_point + a_high - a_low;
```

### Mathematical Formula

Given `low = p(0)` and `high = p(1)`:

| Point | Formula | Equivalent |
|-------|---------|------------|
| p(∞) | `high - low` | Leading coefficient (slope) |
| p(0) | `low` | Given |
| p(1) | `high` | Given |
| p(2) | `2*high - low` | `low + 2*(high - low)` |
| p(3) | `3*high - 2*low` | `p(2) + (high - low)` |
| p(k) | `k*high - (k-1)*low` | `p(k-1) + (high - low)` |

### Incremental Computation

The existing code uses an incremental approach:
```rust
let diff = high - low;           // p(∞) = leading coeff
let mut val = 2*high - low;      // p(2)
// For p(3), p(4), ...:
val += diff;                     // p(k) = p(k-1) + diff
```

This avoids multiplication by computing each successive point as `p(k) = p(k-1) + diff`.

---

## Procedure 6: Extend Multilinear Polynomial

Extends evaluations from `{0,1}^ℓ₀` to `U_d^ℓ₀` by repeatedly extending one dimension.

### Algorithm

```
Input: Evaluations {p(y)}_{y∈{0,1}^ℓ₀}, stored in CurrentEvals
Output: Evaluations {p(β)}_{β∈U_d^ℓ₀}

for j = 1 to ℓ₀ do                           // Extend dimension j
    for β_prefix ∈ U_d^{j-1} do
        for y_suffix ∈ {0,1}^{ℓ₀-j} do
            p0 := CurrentEvals[(β_prefix, 0, y_suffix)]
            p1 := CurrentEvals[(β_prefix, 1, y_suffix)]

            // Extend via Procedure 5
            for γ ∈ U_d do
                NextEvals[(β_prefix, γ, y_suffix)] := extrapolate(p0, p1, γ)
            end for
        end for
    end for
    CurrentEvals := NextEvals
end for
return CurrentEvals
```

### Standalone Function (Avoids Clone)

```rust
/// Extend evaluations from {0,1}^ℓ₀ to U_d^ℓ₀ for Lagrange interpolation.
///
/// Takes ownership of input to avoid expensive clone.
///
/// # Arguments
/// * `evals` - 2^ℓ₀ evaluations over boolean hypercube (consumed)
/// * `l0` - Number of variables to extend
/// * `d` - Degree parameter for extended domain U_d
fn extend_to_lagrange_domain<Scalar: PrimeField>(
    evals: Vec<Scalar>,
    l0: usize,
    d: usize,
) -> LagrangeEvals<Scalar> {
    debug_assert_eq!(evals.len(), 1 << l0, "Input size must be 2^l0");
    let mut current = evals;  // Take ownership, no clone!

    for _ in 0..l0 {
        current = extend_one_variable(&current, d);
    }

    LagrangeEvals {
        evals: current,
        l0,
        d,
    }
}

/// Extend the first variable from {0,1} to U_d = {∞, 0, 1, ..., d-1}.
///
/// Input: evaluations of size 2 * suffix_size (over {0,1} × suffix_domain)
/// Output: evaluations of size (d+1) * suffix_size (over U_d × suffix_domain)
fn extend_one_variable<Scalar: PrimeField>(evals: &[Scalar], d: usize) -> Vec<Scalar> {
    let half = evals.len() / 2;
    let (left, right) = evals.split_at(half);  // left = p(0, ...), right = p(1, ...)

    let mut result = Vec::with_capacity((d + 1) * half);

    // Index 0 → ∞: leading coefficient = p(1) - p(0)
    for i in 0..half {
        result.push(right[i] - left[i]);
    }

    // Index 1 → 0: p(0, suffix)
    result.extend_from_slice(left);

    // Index 2 → 1: p(1, suffix)
    result.extend_from_slice(right);

    // Index 3..=d → 2..=d-1: extrapolate p(k) = p(0) + k * (p(1) - p(0))
    for k in 2..d {
        let k_scalar = Scalar::from(k as u64);
        for i in 0..half {
            let diff = right[i] - left[i];
            result.push(left[i] + k_scalar * diff);
        }
    }

    result
}
```

### Concrete Example (ℓ₀=2, d=3)

```
Input: [p(0,0)=1, p(0,1)=3, p(1,0)=5, p(1,1)=11]  (4 values over {0,1}²)

Round 1 - extend X:
  For y=0: p0=1, p1=5, diff=4 → [p(∞,0)=4, p(0,0)=1, p(1,0)=5, p(2,0)=9]
  For y=1: p0=3, p1=11, diff=8 → [p(∞,1)=8, p(0,1)=3, p(1,1)=11, p(2,1)=19]

  After Round 1: 8 values over U_d × {0,1}

Round 2 - extend Y:
  For x=∞: p0=4, p1=8, diff=4 → [4, 4, 8, 12]
  For x=0: p0=1, p1=3, diff=2 → [2, 1, 3, 5]
  For x=1: p0=5, p1=11, diff=6 → [6, 5, 11, 17]
  For x=2: p0=9, p1=19, diff=10 → [10, 9, 19, 29]

Output (16 values over U_d²):
┌─────────┬────────┬────────┬────────┬────────┐
│  p(x,y) │  y=∞   │  y=0   │  y=1   │  y=2   │
├─────────┼────────┼────────┼────────┼────────┤
│   x=∞   │   4    │   4    │   8    │   12   │
│   x=0   │   2    │ **1**  │ **3**  │   5    │
│   x=1   │   6    │ **5**  │ **11** │   17   │
│   x=2   │  10    │   9    │   19   │   29   │
└─────────┴────────┴────────┴────────┴────────┘
              ** = original boolean points **
```

### Unit Tests for `extend_to_lagrange_domain`

```rust
#[cfg(test)]
mod extend_to_lagrange_domain_tests {
    use super::*;
    use rand_core::OsRng;

    // ==================== Test 1: Output Size ====================

    #[test]
    fn test_output_size() {
        let d = 3;

        for l0 in 1..=4 {
            let input_size = 1 << l0;
            let input: Vec<Fr> = (0..input_size).map(|i| Fr::from(i as u64)).collect();

            let extended: LagrangeEvals<Fr> = extend_to_lagrange_domain(input, l0, d);

            let expected_size = (d + 1).pow(l0 as u32);
            assert_eq!(extended.len(), expected_size);
            assert_eq!(extended.l0, l0);
            assert_eq!(extended.d, d);
        }
    }

    // ==================== Test 2: Boolean Points Preserved ====================

    #[test]
    fn test_preserves_boolean() {
        let l0 = 3;
        let d = 3;
        let base = d + 1;

        let input: Vec<Fr> = (0..(1 << l0)).map(|_| Fr::random(&mut OsRng)).collect();

        let extended: LagrangeEvals<Fr> = extend_to_lagrange_domain(input.clone(), l0, d);

        // In U_d indexing: 0 → index 1, 1 → index 2
        for b in 0..(1 << l0) {
            let mut ud_idx = 0;
            for j in 0..l0 {
                let bit = (b >> (l0 - 1 - j)) & 1;
                let ud_val = bit + 1;
                ud_idx = ud_idx * base + ud_val;
            }

            assert_eq!(extended.get(ud_idx), input[b]);
        }
    }

    // ==================== Test 3: Single Variable ====================

    #[test]
    fn test_single_var() {
        let d = 3;
        let p0 = Fr::from(7);
        let p1 = Fr::from(19);

        let l0 = 1;  // single variable
        let input = vec![p0, p1];
        let extended: LagrangeEvals<Fr> = extend_to_lagrange_domain(input, l0, d);

        // U_d = {∞, 0, 1, 2} with indices 0, 1, 2, 3
        assert_eq!(extended.get(0), p1 - p0, "p(∞) = leading coeff");
        assert_eq!(extended.get(1), p0, "p(0)");
        assert_eq!(extended.get(2), p1, "p(1)");
        assert_eq!(extended.get(3), p1.double() - p0, "p(2) = 2*p1 - p0");
    }

    // ==================== Test 4: Matches Direct Evaluation ====================

    #[test]
    fn test_matches_direct() {
        let l0 = 3;
        let d = 3;
        let base = d + 1;

        let input: Vec<Fr> = (0..(1 << l0)).map(|_| Fr::random(&mut OsRng)).collect();
        let extended: LagrangeEvals<Fr> = extend_to_lagrange_domain(input.clone(), l0, d);

        // Check all finite points via direct multilinear evaluation
        for idx in 0..extended.len() {
            let tuple = index_to_tuple(idx, base, l0);

            // Skip infinity points (index 0 in any coordinate)
            if tuple.iter().any(|&t| t == 0) {
                continue;
            }

            // Convert U_d indices to field values: index k → value k-1
            let point: Vec<Fr> = tuple.iter()
                .map(|&t| Fr::from((t - 1) as u64))
                .collect();

            let direct = evaluate_multilinear(&input, &point);
            assert_eq!(extended.get(idx), direct);
        }
    }

    // ==================== Test 5: Infinity = Leading Coefficient ====================

    #[test]
    fn test_infinity_leading_coeff() {
        let l0 = 3;
        let d = 3;
        let base = d + 1;

        let input: Vec<Fr> = (0..(1 << l0)).map(|_| Fr::random(&mut OsRng)).collect();
        let extended: LagrangeEvals<Fr> = extend_to_lagrange_domain(input.clone(), l0, d);

        // p(∞, y₂, y₃) = p(1, y₂, y₃) - p(0, y₂, y₃)
        for y2 in 0..2usize {
            for y3 in 0..2usize {
                let idx_0 = (0 << 2) | (y2 << 1) | y3;  // p(0, y2, y3)
                let idx_1 = (1 << 2) | (y2 << 1) | y3;  // p(1, y2, y3)

                let expected = input[idx_1] - input[idx_0];
                let ext_idx = 0 * base * base + (y2 + 1) * base + (y3 + 1);

                assert_eq!(extended.get(ext_idx), expected);
            }
        }
    }

    // ==================== Test 6: Known Polynomial ====================

    #[test]
    fn test_known_polynomial() {
        // p(X, Y, Z) = X + 2Y + 4Z
        let d = 3;
        let l0 = 3;
        let base = d + 1;

        let mut input = Vec::with_capacity(8);
        for x in 0..2u64 {
            for y in 0..2u64 {
                for z in 0..2u64 {
                    input.push(Fr::from(x + 2*y + 4*z));
                }
            }
        }

        let extended: LagrangeEvals<Fr> = extend_to_lagrange_domain(input, l0, d);

        // Finite points: p(a,b,c) = a + 2b + 4c
        for a in 0..d {
            for b in 0..d {
                for c in 0..d {
                    let idx = (a+1) * base * base + (b+1) * base + (c+1);
                    let expected = Fr::from(a as u64 + 2 * b as u64 + 4 * c as u64);
                    assert_eq!(extended.get(idx), expected);
                }
            }
        }

        // Infinity points = variable coefficients
        assert_eq!(extended.get(0 * base * base + 1 * base + 1), Fr::ONE, "p(∞,0,0) = coeff of X");
        assert_eq!(extended.get(1 * base * base + 0 * base + 1), Fr::from(2), "p(0,∞,0) = coeff of Y");
        assert_eq!(extended.get(1 * base * base + 1 * base + 0), Fr::from(4), "p(0,0,∞) = coeff of Z");
        assert_eq!(extended.get(0), Fr::ZERO, "p(∞,∞,∞) = 0 (no XYZ term)");
    }

    // ==================== Helpers ====================

    fn index_to_tuple(mut idx: usize, base: usize, len: usize) -> Vec<usize> {
        let mut tuple = vec![0; len];
        for i in (0..len).rev() {
            tuple[i] = idx % base;
            idx /= base;
        }
        tuple
    }

    /// Direct multilinear evaluation: p(r) = Σ_x p(x) · eq(x, r)
    fn evaluate_multilinear<F: PrimeField>(evals: &[F], point: &[F]) -> F {
        let l = point.len();
        let mut result = F::ZERO;
        for (i, &val) in evals.iter().enumerate() {
            let mut eq_term = F::ONE;
            for j in 0..l {
                let bit = (i >> (l - 1 - j)) & 1;
                if bit == 1 {
                    eq_term *= point[j];
                } else {
                    eq_term *= F::ONE - point[j];
                }
            }
            result += val * eq_term;
        }
        result
    }
}
```

### Test Summary

| Test | What it verifies |
|------|------------------|
| `test_output_size` | Returns (d+1)^ℓ₀ elements, correct metadata |
| `test_preserves_boolean` | Original {0,1}^ℓ₀ values unchanged |
| `test_single_var` | Base case: p(∞), p(0), p(1), p(2) correct |
| `test_matches_direct` | Random poly matches direct evaluation at finite points |
| `test_infinity_leading_coeff` | p(∞,...)=p(1,...)-p(0,...) |
| `test_known_polynomial` | p(X,Y,Z)=X+2Y+4Z: finite points + infinity coefficients |

---

## Gather Prefix Evaluations

Bridges the gap between full polynomials (size 2^ℓ) and Procedure 6 (which expects size 2^ℓ₀).

### The Problem

In Spartan, `poly_Az`, `poly_Bz`, `poly_Cz` are each of size **2^ℓ** (evaluations over all ℓ variables).

Procedure 6 expects evaluations over **{0,1}^ℓ₀** (the first ℓ₀ variables only).

For Procedure 9, we need to iterate over all suffix values and extract the prefix evaluations for each.

### Index Layout

For a polynomial over ℓ variables:
```
index = (prefix << suffix_vars) | suffix

Where:
- prefix = first ℓ₀ bits (high bits) — the variables we extend
- suffix = last (ℓ - ℓ₀) bits (low bits) — the variables we iterate over
- suffix_vars = ℓ - ℓ₀
```

### Concrete Example

```
ℓ = 5 total variables
ℓ₀ = 2 prefix variables
suffix_vars = 3

Index bits: [p₁ p₀ | s₂ s₁ s₀]
             prefix   suffix

For suffix s = 5 (binary: 101), gather all prefix values:

prefix=0 (00): index = (0 << 3) | 5 = 0b00_101 = 5
prefix=1 (01): index = (1 << 3) | 5 = 0b01_101 = 13
prefix=2 (10): index = (2 << 3) | 5 = 0b10_101 = 21
prefix=3 (11): index = (3 << 3) | 5 = 0b11_101 = 29

result = [poly[5], poly[13], poly[21], poly[29]]
```

### Visual

```
poly = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, ...]
                        │               │                │
For suffix=5:           ●               ●                ●        ...
                        │               │                │
                        5               13               21
                        │               │                │
                        ▼               ▼                ▼
              result = [poly[5],    poly[13],       poly[21],    ...]

Stride between reads = 2^{suffix_vars} = 8
```

### Implementation

```rust
/// Gather evaluations for all prefix values given a fixed suffix.
///
/// For a polynomial over ℓ variables, extracts the 2^{ℓ₀} evaluations
/// where the last (ℓ - ℓ₀) variables are fixed to `suffix`.
///
/// Runtime: O(2^ℓ₀)
fn gather_prefix_evals<Scalar: Copy>(
    poly: &[Scalar],
    l0: usize,      // number of prefix variables
    suffix: usize,  // fixed suffix value
) -> Vec<Scalar> {
    let l = poly.len().log_2();
    let suffix_vars = l - l0;
    let prefix_size = 1 << l0;

    (0..prefix_size)
        .map(|prefix| {
            let idx = (prefix << suffix_vars) | suffix;
            poly[idx]
        })
        .collect()
}
```

### Runtime Analysis

| Operation | Work per suffix | Total for all suffixes |
|-----------|-----------------|------------------------|
| Gather | O(2^ℓ₀) | O(2^{ℓ-ℓ₀} × 2^ℓ₀) = O(2^ℓ) |
| Extend | O((d+1)^ℓ₀) | O(2^{ℓ-ℓ₀} × (d+1)^ℓ₀) |

Total reads of original polynomials: O(2^ℓ) — we read each element exactly once across all suffixes.

### Unit Tests for `gather_prefix_evals`

```rust
#[cfg(test)]
mod gather_prefix_evals_tests {
    use super::*;

    // ==================== Test 1: Output Size ====================

    #[test]
    fn test_gather_output_size() {
        let l = 5;
        let l0 = 2;
        let poly: Vec<Fr> = (0..(1 << l)).map(|i| Fr::from(i as u64)).collect();

        let gathered = gather_prefix_evals(&poly, l0, 0);

        assert_eq!(gathered.len(), 1 << l0);  // 2^ℓ₀ = 4
    }

    // ==================== Test 2: Correct Indices ====================

    #[test]
    fn test_gather_correct_indices() {
        let l = 5;
        let l0 = 2;
        let suffix_vars = l - l0;  // 3

        // Use index as value so we can verify which indices were read
        let poly: Vec<Fr> = (0..(1 << l)).map(|i| Fr::from(i as u64)).collect();

        let suffix = 5;  // binary: 101
        let gathered = gather_prefix_evals(&poly, l0, suffix);

        // Verify we got the right indices
        // prefix=0: (0 << 3) | 5 = 5
        // prefix=1: (1 << 3) | 5 = 13
        // prefix=2: (2 << 3) | 5 = 21
        // prefix=3: (3 << 3) | 5 = 29
        assert_eq!(gathered[0], Fr::from(5u64));
        assert_eq!(gathered[1], Fr::from(13u64));
        assert_eq!(gathered[2], Fr::from(21u64));
        assert_eq!(gathered[3], Fr::from(29u64));
    }

    // ==================== Test 3: All Suffixes Cover All Indices ====================

    #[test]
    fn test_gather_all_suffixes_cover_polynomial() {
        let l = 4;
        let l0 = 2;
        let suffix_vars = l - l0;
        let suffix_size = 1 << suffix_vars;

        let poly: Vec<Fr> = (0..(1 << l)).map(|i| Fr::from(i as u64)).collect();

        // Gather for all suffixes and collect all indices
        let mut all_values: Vec<Fr> = Vec::new();
        for suffix in 0..suffix_size {
            let gathered = gather_prefix_evals(&poly, l0, suffix);
            all_values.extend(gathered);
        }

        // Should have read every element exactly once
        all_values.sort_by_key(|f| f.to_repr().as_ref().to_vec());
        let expected: Vec<Fr> = (0..(1 << l)).map(|i| Fr::from(i as u64)).collect();

        // Verify we got all values (order may differ)
        assert_eq!(all_values.len(), poly.len());
    }

    // ==================== Test 4: Suffix Zero ====================

    #[test]
    fn test_gather_suffix_zero() {
        let l = 4;
        let l0 = 2;
        let suffix_vars = l - l0;

        let poly: Vec<Fr> = (0..(1 << l)).map(|i| Fr::from(i as u64)).collect();

        let gathered = gather_prefix_evals(&poly, l0, 0);

        // suffix=0: indices are 0, 4, 8, 12 (stride = 2^{suffix_vars} = 4)
        assert_eq!(gathered[0], Fr::from(0u64));
        assert_eq!(gathered[1], Fr::from(4u64));
        assert_eq!(gathered[2], Fr::from(8u64));
        assert_eq!(gathered[3], Fr::from(12u64));
    }

    // ==================== Test 5: Single Prefix Variable ====================

    #[test]
    fn test_gather_single_prefix_var() {
        let l = 4;
        let l0 = 1;  // Only 1 prefix variable
        let suffix_vars = l - l0;  // 3 suffix variables

        let poly: Vec<Fr> = (0..(1 << l)).map(|i| Fr::from(i as u64)).collect();

        let suffix = 3;
        let gathered = gather_prefix_evals(&poly, l0, suffix);

        // 2^ℓ₀ = 2 prefix values
        assert_eq!(gathered.len(), 2);
        // prefix=0: (0 << 3) | 3 = 3
        // prefix=1: (1 << 3) | 3 = 11
        assert_eq!(gathered[0], Fr::from(3u64));
        assert_eq!(gathered[1], Fr::from(11u64));
    }
}
```

### Test Summary

| Test | What it verifies |
|------|------------------|
| `test_gather_output_size` | Returns 2^ℓ₀ elements |
| `test_gather_correct_indices` | Reads from correct strided indices |
| `test_gather_all_suffixes_cover_polynomial` | All suffixes together cover entire polynomial |
| `test_gather_suffix_zero` | Edge case: suffix = 0 |
| `test_gather_single_prefix_var` | Edge case: ℓ₀ = 1 |

---

## compute_idx4 Mapping (Definition A.5)

Maps evaluation prefix `β ∈ U_d^ℓ₀` to all accumulators it contributes to.

### AccumulatorPrefixIndex Struct

```rust
/// A single contribution from β to an accumulator A_i(v, u).
///
/// Represents the decomposition of β ∈ U_d^ℓ₀ into:
/// - Round i (which accumulator)
/// - Prefix v = (β₁, ..., β_{i-1}) ∈ U_d^{i-1}
/// - Coordinate u = βᵢ ∈ Û_d
/// - Binary suffix y = (β_{i+1}, ..., β_{ℓ₀}) ∈ {0,1}^{ℓ₀-i}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccumulatorPrefixIndex {
    /// Total number of small-value rounds (ℓ₀)
    pub l0: usize,

    /// Round index i ∈ [1, ℓ₀] (1-indexed as in paper)
    pub round: usize,

    /// Prefix v = (β₁, ..., β_{i-1}) as flat index in U_d^{i-1}
    pub v_idx: usize,

    /// Coordinate u = βᵢ ∈ Û_d (type-safe, excludes value 1)
    pub u: UdHatPoint,

    /// Binary suffix y = (β_{i+1}, ..., β_{ℓ₀}) as flat index in {0,1}^{ℓ₀-i}
    pub y_idx: usize,
}

impl AccumulatorPrefixIndex {
    /// Round as 0-indexed (for array access)
    #[inline]
    pub fn round_0idx(&self) -> usize {
        self.round - 1
    }

    /// Length of prefix v
    #[inline]
    pub fn prefix_len(&self) -> usize {
        self.round - 1
    }

    /// Length of binary suffix y
    #[inline]
    pub fn suffix_len(&self) -> usize {
        self.l0 - self.round
    }
}
```

### compute_idx4 Function

```rust
/// Computes accumulator indices for β ∈ U_d^ℓ₀.
///
/// For each round i where:
/// 1. The suffix β[i..] is binary (values in {0,1})
/// 2. The coordinate u = β[i-1] is in Û_d (i.e., u ≠ 1)
///
/// Returns an `AccumulatorPrefixIndex` describing the contribution.
///
/// # Type-Safe API
///
/// This function accepts `UdTuple` for type safety. For performance-critical
/// code using flat indices, convert via `UdTuple::from_flat_index`.
///
/// # Arguments
/// * `beta` - Tuple in U_d^ℓ₀ as a UdTuple
/// * `l0` - Number of small-value rounds
/// * `d` - Degree parameter
pub fn compute_idx4(beta: &UdTuple, l0: usize, d: usize) -> Vec<AccumulatorPrefixIndex> {
    let base = d + 1;
    let mut result = Vec::new();

    for i in 1..=l0 {
        // Check if suffix β[i..] is all binary
        // Binary means the value is Finite(0) or Finite(1)
        let suffix_is_binary = beta.0[i..].iter().all(|p| p.is_binary());

        if !suffix_is_binary {
            continue;
        }

        // u = β[i-1] — try to convert to Û_d point
        // If u = Finite(1), to_ud_hat() returns None and we skip this round
        let u = beta.0[i - 1];
        let Some(u_hat) = u.to_ud_hat() else {
            continue;  // u = Finite(1), not in Û_d
        };

        // v = prefix β[0..i-1] as flat index
        let prefix = UdTuple(beta.0[0..(i - 1)].to_vec());
        let v_idx = prefix.to_flat_index(base);

        // y = suffix converted to binary index
        // Finite(0) → bit 0, Finite(1) → bit 1
        let suffix = &beta.0[i..];
        let y_idx = suffix.iter().fold(0usize, |acc, p| {
            let bit = match p {
                UdPoint::Finite(0) => 0,
                UdPoint::Finite(1) => 1,
                _ => unreachable!("suffix should be binary"),
            };
            (acc << 1) | bit
        });

        result.push(AccumulatorPrefixIndex {
            l0,
            round: i,  // 1-indexed
            v_idx,
            u: u_hat,
            y_idx,
        });
    }

    result
}
```

### Concrete Examples (d=3, ℓ₀=3)

**Encoding Reminder:**

For U_d = {∞, 0, 1, 2} with d=3:
- Index 0 → ∞ (leading coefficient)
- Index 1 → evaluation at 0
- Index 2 → evaluation at 1
- Index 3 → evaluation at 2

Binary means index ∈ {1, 2} (i.e., evaluation at 0 or 1).

---

**Example 1: β = (1, 2, 1) → point (0, 1, 0)**

All values are binary, but round 2 is filtered because u = 1 ∉ Û_d:

| Round i | v = β[0..i-1] | u = β[i-1] | u value | u ∈ Û_d? | y = β[i..] | y binary? | v_idx | u | y_idx |
|---------|---------------|------------|---------|----------|------------|-----------|-------|---|-------|
| 1       | ()            | 1          | 0       | ✓        | (2, 1)     | ✓         | 0     | Finite(0) | 2     |
| 2       | (1,)          | 2          | **1**   | **✗**    | (1,)       | ✓         | -     | -         | -     |
| 3       | (1, 2)        | 1          | 0       | ✓        | ()         | ✓         | 6     | Finite(0) | 0     |

`compute_idx4(&[1,2,1], 3, 3)` returns **2** `AccumulatorPrefixIndex` (round 2 filtered):
- `{ l0: 3, round: 1, v_idx: 0, u: UdHatPoint::Finite(0), y_idx: 2 }`
- `{ l0: 3, round: 3, v_idx: 6, u: UdHatPoint::Finite(0), y_idx: 0 }`

y_idx calculation for round 1: y=(2,1) → bits (1,0) → 1×2 + 0 = 2

---

**Example 2: β = (0, 1, 2) → point (∞, 0, 1)**

Has ∞ at first position, round 3 filtered because u = 1 ∉ Û_d:

| Round i | v = β[0..i-1] | u = β[i-1] | u value | u ∈ Û_d? | y = β[i..] | y binary? | v_idx | u | y_idx |
|---------|---------------|------------|---------|----------|------------|-----------|-------|---|-------|
| 1       | ()            | 0          | ∞       | ✓        | (1, 2)     | ✓         | 0     | Infinity | 1     |
| 2       | (0,)          | 1          | 0       | ✓        | (2,)       | ✓         | 0     | Finite(0) | 1     |
| 3       | (0, 1)        | 2          | **1**   | **✗**    | ()         | ✓         | -     | -         | -     |

`compute_idx4(&[0,1,2], 3, 3)` returns **2** `AccumulatorPrefixIndex` (round 3 filtered):
- `{ l0: 3, round: 1, v_idx: 0, u: UdHatPoint::Infinity, y_idx: 1 }`
- `{ l0: 3, round: 2, v_idx: 0, u: UdHatPoint::Finite(0), y_idx: 1 }`

---

**Example 3: β = (0, 0, 1) → point (∞, ∞, 0)**

Two ∞ values — round 1 skipped due to non-binary suffix:

| Round i | v = β[0..i-1] | u = β[i-1] | u value | u ∈ Û_d? | y = β[i..] | y binary?   | v_idx | u | y_idx |
|---------|---------------|------------|---------|----------|------------|-------------|-------|---|-------|
| 1       | ()            | 0          | ∞       | ✓        | (0, 1)     | ✗ (has 0=∞) | -     | -         | -     |
| 2       | (0,)          | 0          | ∞       | ✓        | (1,)       | ✓           | 0     | Infinity  | 0     |
| 3       | (0, 0)        | 1          | 0       | ✓        | ()         | ✓           | 0     | Finite(0) | 0     |

`compute_idx4(&[0,0,1], 3, 3)` returns 2 `AccumulatorPrefixIndex`:
- `{ l0: 3, round: 2, v_idx: 0, u: UdHatPoint::Infinity, y_idx: 0 }`
- `{ l0: 3, round: 3, v_idx: 0, u: UdHatPoint::Finite(0), y_idx: 0 }`

Only 2 contributions — round 1 skipped because suffix (0, 1) contains ∞ (value not in {0,1}).

---

**Example 4: β = (3, 3, 3) → point (2, 2, 2)**

All non-binary (extrapolated points), only last round contributes:

| Round i | v = β[0..i-1] | u = β[i-1] | u value | u ∈ Û_d? | y = β[i..] | y binary? | v_idx | u | y_idx |
|---------|---------------|------------|---------|----------|------------|-----------|-------|---|-------|
| 1       | ()            | 3          | 2       | ✓        | (3, 3)     | ✗         | -     | -         | -     |
| 2       | (3,)          | 3          | 2       | ✓        | (3,)       | ✗         | -     | -         | -     |
| 3       | (3, 3)        | 3          | 2       | ✓        | ()         | ✓ (empty) | 15    | Finite(2) | 0     |

`compute_idx4(&[3,3,3], 3, 3)` returns 1 `AccumulatorPrefixIndex`:
- `{ l0: 3, round: 3, v_idx: 15, u: UdHatPoint::Finite(2), y_idx: 0 }`

Only 1 contribution — only the last round (empty suffix is vacuously binary).

---

**Example 5: β = (2, 2, 2) → point (1, 1, 1)**

All values are binary, but **ALL rounds filtered** because u = 1 ∉ Û_d at every position:

| Round i | v = β[0..i-1] | u = β[i-1] | u value | u ∈ Û_d? | y = β[i..] | y binary? | v_idx | u | y_idx |
|---------|---------------|------------|---------|----------|------------|-----------|-------|---|-------|
| 1       | ()            | 2          | **1**   | **✗**    | (2, 2)     | ✓         | -     | - | -     |
| 2       | (2,)          | 2          | **1**   | **✗**    | (2,)       | ✓         | -     | - | -     |
| 3       | (2, 2)        | 2          | **1**   | **✗**    | ()         | ✓         | -     | - | -     |

`compute_idx4(&[2,2,2], 3, 3)` returns **0** `AccumulatorPrefixIndex` (all rounds filtered).

**Key insight:** The point (1, 1, 1) has u = 1 at every round, and 1 ∉ Û_d. This is correct because:
- Accumulators A_i(v, u) only store values for u ∈ Û_d
- The evaluation at u = 1 is recovered from the sum-check constraint s(0) + s(1) = claim
- Therefore, products at points where βᵢ = 1 don't need to be accumulated directly

---

### The Key Insight

The `compute_idx4` function tells us which accumulators a product term ∏ₖ pₖ(β, x'') contributes to. Two conditions must be met:

1. **Binary suffix:** y = β[i..] ∈ {0,1}^{ℓ₀-i} because accumulators sum over the boolean hypercube
2. **u ∈ Û_d:** The coordinate u = βᵢ must be in Û_d = U_d \ {1} because we don't store evaluations at u = 1

$$A_i(v, u) = \sum_{y \in \{0,1\}^{\ell_0-i}} \sum_{x''} \prod_k p_k(v, u, y, x'')$$

**Why exclude u = 1?** The sum-check constraint s(0) + s(1) = claim allows the verifier to derive s(1) from s(0). Therefore:
- We only need to compute s(u) for u ∈ Û_d = {∞, 0, 2, ..., d-1}
- The accumulator stores `d` values per row (for Û_d), not `d+1` (for U_d)
- Products where βᵢ = 1 would contribute to A_i(v, 1), which we don't store

---

### Unit Tests for compute_idx4

```rust
// Helper: construct UdTuple from flat indices
// 0 → ∞, 1 → 0, 2 → 1, 3 → 2, ...
fn tuple_from_indices(indices: &[usize]) -> UdTuple {
    UdTuple(
        indices.iter().map(|&i| UdPoint::from_index(i)).collect()
    )
}

#[test]
fn test_compute_idx4_example_mixed_binary() {
    // β = (Finite(0), Finite(1), Finite(0)) → point (0, 1, 0)
    // Round 2 filtered because u = value 1 ∉ Û_d
    let l0 = 3;
    let d = 3;

    let beta = tuple_from_indices(&[1, 2, 1]);  // indices → (0, 1, 0)
    let contributions = compute_idx4(&beta, l0, d);

    assert_eq!(contributions.len(), 2);  // Round 2 filtered

    // Round 1: v=(), u=value 0, y=(Finite(1),Finite(0)) → y_idx = 2
    let round1 = contributions.iter().find(|c| c.round == 1).unwrap();
    assert_eq!(round1.v_idx, 0);
    assert_eq!(round1.u, UdHatPoint::Finite(0));
    assert_eq!(round1.y_idx, 2);  // bits (1,0) = 2
    assert_eq!(round1.prefix_len(), 0);
    assert_eq!(round1.suffix_len(), 2);

    // Round 2: FILTERED (u = value 1 ∉ Û_d)
    assert!(contributions.iter().all(|c| c.round != 2));

    // Round 3: v=(Finite(0),Finite(1)), u=value 0, y=() → y_idx = 0
    let round3 = contributions.iter().find(|c| c.round == 3).unwrap();
    assert_eq!(round3.v_idx, 6);  // 1*4 + 2 = 6
    assert_eq!(round3.u, UdHatPoint::Finite(0));
    assert_eq!(round3.y_idx, 0);
    assert_eq!(round3.prefix_len(), 2);
    assert_eq!(round3.suffix_len(), 0);
}

#[test]
fn test_compute_idx4_example_with_infinity() {
    // β = (∞, 0, 1) → point (∞, 0, 1)
    // Round 3 filtered because u = value 1 ∉ Û_d
    let l0 = 3;
    let d = 3;

    let beta = tuple_from_indices(&[0, 1, 2]);  // indices → (∞, 0, 1)
    let contributions = compute_idx4(&beta, l0, d);

    assert_eq!(contributions.len(), 2);  // Round 3 filtered

    // Round 1: v=(), u=∞, y=(Finite(0),Finite(1)) → y_idx = 1
    let round1 = contributions.iter().find(|c| c.round == 1).unwrap();
    assert_eq!(round1.v_idx, 0);
    assert_eq!(round1.u, UdHatPoint::Infinity);
    assert_eq!(round1.y_idx, 1);  // bits (0,1) = 1

    // Round 2: v=(∞,), u=value 0, y=(Finite(1),) → y_idx = 1
    let round2 = contributions.iter().find(|c| c.round == 2).unwrap();
    assert_eq!(round2.v_idx, 0);
    assert_eq!(round2.u, UdHatPoint::Finite(0));
    assert_eq!(round2.y_idx, 1);  // bits (1,) = 1

    // Round 3: FILTERED (u = value 1 ∉ Û_d)
    assert!(contributions.iter().all(|c| c.round != 3));
}

#[test]
fn test_compute_idx4_example_double_infinity() {
    // β = (∞, ∞, 0) → point (∞, ∞, 0)
    // Round 1 skipped because suffix contains ∞
    let l0 = 3;
    let d = 3;

    let beta = tuple_from_indices(&[0, 0, 1]);  // indices → (∞, ∞, 0)
    let contributions = compute_idx4(&beta, l0, d);

    assert_eq!(contributions.len(), 2);

    // Round 1 should be missing
    assert!(contributions.iter().all(|c| c.round != 1));

    // Round 2: v=(∞,), u=∞, y=(Finite(0),) → y_idx = 0
    let round2 = contributions.iter().find(|c| c.round == 2).unwrap();
    assert_eq!(round2.v_idx, 0);
    assert_eq!(round2.u, UdHatPoint::Infinity);
    assert_eq!(round2.y_idx, 0);

    // Round 3: v=(∞,∞), u=Finite(0), y=() → y_idx = 0
    let round3 = contributions.iter().find(|c| c.round == 3).unwrap();
    assert_eq!(round3.v_idx, 0);  // 0*4 + 0 = 0
    assert_eq!(round3.u, UdHatPoint::Finite(0));
    assert_eq!(round3.y_idx, 0);
}

#[test]
fn test_compute_idx4_example_all_extrapolated() {
    // β = (Finite(2), Finite(2), Finite(2)) → point (2, 2, 2)
    // Only last round (empty suffix is vacuously binary)
    let l0 = 3;
    let d = 3;

    let beta = tuple_from_indices(&[3, 3, 3]);  // indices → (2, 2, 2)
    let contributions = compute_idx4(&beta, l0, d);

    assert_eq!(contributions.len(), 1);

    let only = &contributions[0];
    assert_eq!(only.round, 3);
    assert_eq!(only.v_idx, 15);  // 3*4 + 3 = 15
    assert_eq!(only.u, UdHatPoint::Finite(2));
    assert_eq!(only.y_idx, 0);
    assert_eq!(only.suffix_len(), 0);
}

#[test]
fn test_compute_idx4_contribution_counts() {
    // Not every β contributes! β where all elements are Finite(1)
    // has NO contributions because u=1 ∉ Û_d for all rounds.
    let l0 = 3;
    let d = 3;
    let base = d + 1;
    let prefix_ud_size = base.pow(l0 as u32);

    let mut zero_contribution_count = 0;

    for beta_idx in 0..prefix_ud_size {
        let beta = UdTuple::from_flat_index(beta_idx, base, l0);
        let contributions = compute_idx4(&beta, l0, d);

        // All contributions should have correct l0
        for c in &contributions {
            assert_eq!(c.l0, l0);
        }

        if contributions.is_empty() {
            zero_contribution_count += 1;
        }
    }

    // β = (Finite(1), Finite(1), Finite(1)) is the only β with zero contributions
    assert_eq!(zero_contribution_count, 1, "Only β=(1,1,1) should have zero contributions");
}

#[test]
fn test_compute_idx4_all_ones_has_no_contributions() {
    // β = (Finite(1), Finite(1), Finite(1)) → point (1, 1, 1)
    // ALL rounds filtered because u = value 1 ∉ Û_d at every position
    let l0 = 3;
    let d = 3;

    let beta = tuple_from_indices(&[2, 2, 2]);  // indices → (1, 1, 1)
    let contributions = compute_idx4(&beta, l0, d);

    assert!(contributions.is_empty(), "β=(1,1,1) should have no contributions");
}

#[test]
fn test_compute_idx4_binary_beta_filtered_by_u() {
    // Binary β (Finite(0) or Finite(1)) DON'T all contribute to all rounds!
    // Round i is filtered when β[i-1] = Finite(1) (i.e., u = value 1 ∉ Û_d)
    let l0 = 3;
    let d = 3;

    // Iterate over all binary β
    for b in 0..(1 << l0) {
        // Construct beta with Finite(0) or Finite(1) values
        let indices: Vec<usize> = (0..l0)
            .map(|j| ((b >> (l0 - 1 - j)) & 1) + 1)  // 0→index 1 (Finite(0)), 1→index 2 (Finite(1))
            .collect();
        let beta = tuple_from_indices(&indices);

        let contributions = compute_idx4(&beta, l0, d);

        // Count how many positions have Finite(1) (value 1)
        let num_ones = beta.0.iter().filter(|&&p| p == UdPoint::Finite(1)).count();

        // Expected contributions = l0 - num_ones (each position with value 1 filters that round)
        let expected_len = l0 - num_ones;
        assert_eq!(
            contributions.len(), expected_len,
            "β={:?} should have {} contributions (filtering {} rounds with u=1)",
            beta, expected_len, num_ones
        );

        // Verify rounds are correctly filtered
        for (i, &point) in beta.0.iter().enumerate() {
            let round = i + 1;
            let has_round = contributions.iter().any(|c| c.round == round);
            if point == UdPoint::Finite(1) {
                // u = value 1 ∉ Û_d → round should be missing
                assert!(!has_round, "β={:?} should NOT have round {} (u=1)", beta, round);
            } else {
                // u ∈ Û_d → round should be present (suffix is always binary for binary β)
                assert!(has_round, "β={:?} should have round {} (u≠1)", beta, round);
            }
        }
    }
}

#[test]
fn test_compute_idx4_index_bounds() {
    let l0 = 4;
    let d = 3;
    let base = d + 1;
    let prefix_ud_size = base.pow(l0 as u32);

    for beta_idx in 0..prefix_ud_size {
        let beta = UdTuple::from_flat_index(beta_idx, base, l0);
        let contributions = compute_idx4(&beta, l0, d);

        for c in contributions {
            // l0 should match
            assert_eq!(c.l0, l0);

            // Round bound: i ∈ [1, ℓ₀]
            assert!(c.round >= 1 && c.round <= l0,
                "round {} out of [1, {}] for β={:?}", c.round, l0, beta);

            // v_idx bound: v ∈ U_d^{i-1}, so v_idx < (d+1)^{i-1}
            let max_v_idx = base.pow(c.prefix_len() as u32);
            assert!(c.v_idx < max_v_idx,
                "v_idx {} >= {} for round {} β={:?}", c.v_idx, max_v_idx, c.round, beta);

            // u bound: u ∈ Û_d, so u.to_index() < d
            // Û_d has d elements: {∞, 0, 2, 3, ..., d-1}
            assert!(c.u.to_index() < d,
                "u.to_index() {} >= {} (Û_d size) for β={:?}", c.u.to_index(), d, beta);

            // y_idx bound: y ∈ {0,1}^{ℓ₀-i}, so y_idx < 2^{ℓ₀-i}
            let max_y_idx = 1usize << c.suffix_len();
            assert!(c.y_idx < max_y_idx,
                "y_idx {} >= {} for round {} β={:?}", c.y_idx, max_y_idx, c.round, beta);
        }
    }
}

#[test]
fn test_compute_idx4_v_u_decode() {
    let l0 = 3;
    let d = 3;
    let base = d + 1;

    // Test cases that have at least some contributions
    // Note: all Finite(1) has NO contributions so we exclude it
    let test_cases: Vec<UdTuple> = vec![
        tuple_from_indices(&[1, 1, 1]),  // all u=Finite(0), 3 contributions
        tuple_from_indices(&[0, 1, 1]),  // u=∞,Finite(0),Finite(0), 3 contributions
        tuple_from_indices(&[3, 0, 1]),  // u=Finite(2),∞,Finite(0), 3 contributions
        tuple_from_indices(&[3, 3, 3]),  // u=Finite(2)×3, only round 3 due to non-binary suffix
    ];

    for beta in test_cases {
        let contributions = compute_idx4(&beta, l0, d);

        for c in contributions {
            // v should be β[0..round-1]
            let expected_v = UdTuple(beta.0[0..c.prefix_len()].to_vec());
            let expected_v_idx = expected_v.to_flat_index(base);
            assert_eq!(c.v_idx, expected_v_idx,
                "v_idx mismatch for round {} β={:?}", c.round, beta);

            // u should be the Û_d point from β[round-1]
            let u = beta.0[c.round - 1];
            let expected_u = u.to_ud_hat().unwrap();
            assert_eq!(c.u, expected_u,
                "u mismatch for round {} β={:?}: expected {:?} (from {:?})",
                c.round, beta, expected_u, u);
        }
    }
}

#[test]
fn test_compute_idx4_y_idx_encoding() {
    let l0 = 4;
    let d = 3;

    // β = (Finite(0), Finite(1), Finite(0), Finite(1)) → point (0, 1, 0, 1)
    // Rounds 2 and 4 filtered because u = value 1 ∉ Û_d
    let beta = tuple_from_indices(&[1, 2, 1, 2]);
    let contributions = compute_idx4(&beta, l0, d);

    assert_eq!(contributions.len(), 2);  // Only rounds 1 and 3

    // Round 1: u=Finite(0) ∈ Û_d, y = suffix → bits (1,0,1) → y_idx = 5
    let c1 = contributions.iter().find(|c| c.round == 1).unwrap();
    assert_eq!(c1.y_idx, 0b101);
    assert_eq!(c1.suffix_len(), 3);
    assert_eq!(c1.u, UdHatPoint::Finite(0));

    // Round 2: FILTERED (u = Finite(1) ∉ Û_d)
    assert!(contributions.iter().all(|c| c.round != 2));

    // Round 3: u=Finite(0) ∈ Û_d, y = suffix → bits (1,) → y_idx = 1
    let c3 = contributions.iter().find(|c| c.round == 3).unwrap();
    assert_eq!(c3.y_idx, 0b1);
    assert_eq!(c3.suffix_len(), 1);
    assert_eq!(c3.u, UdHatPoint::Finite(0));

    // Round 4: FILTERED (u = Finite(1) ∉ Û_d)
    assert!(contributions.iter().all(|c| c.round != 4));
}

#[test]
fn test_compute_idx4_y_idx_encoding_no_filtering() {
    // Test y_idx encoding with a β that has no filtering
    let l0 = 4;
    let d = 3;

    // β = (Finite(0), Finite(0), Finite(0), Finite(0)) → all zeros
    // All u = Finite(0) ∈ Û_d, so all rounds present
    let beta = tuple_from_indices(&[1, 1, 1, 1]);
    let contributions = compute_idx4(&beta, l0, d);

    assert_eq!(contributions.len(), l0);

    // Round 1: y = suffix → bits (0,0,0) → y_idx = 0
    let c1 = contributions.iter().find(|c| c.round == 1).unwrap();
    assert_eq!(c1.y_idx, 0b000);
    assert_eq!(c1.suffix_len(), 3);

    // Round 2: y = suffix → bits (0,0) → y_idx = 0
    let c2 = contributions.iter().find(|c| c.round == 2).unwrap();
    assert_eq!(c2.y_idx, 0b00);
    assert_eq!(c2.suffix_len(), 2);

    // Round 3: y = suffix → bits (0,) → y_idx = 0
    let c3 = contributions.iter().find(|c| c.round == 3).unwrap();
    assert_eq!(c3.y_idx, 0b0);
    assert_eq!(c3.suffix_len(), 1);

    // Round 4: y = suffix → () → y_idx = 0
    let c4 = contributions.iter().find(|c| c.round == 4).unwrap();
    assert_eq!(c4.y_idx, 0);
    assert_eq!(c4.suffix_len(), 0);
}

#[test]
fn test_compute_idx4_suffix_must_be_binary_values() {
    // Two conditions filter rounds:
    // 1. The suffix y must consist only of binary values (Finite(0) or Finite(1))
    // 2. The coordinate u must be in Û_d (i.e., u ≠ Finite(1))

    let l0 = 3;
    let d = 3;

    // β with non-binary value (Finite(2)) in suffix position
    // indices (1, 3, 2) → values (Finite(0), Finite(2), Finite(1))
    let beta = tuple_from_indices(&[1, 3, 2]);
    let contributions = compute_idx4(&beta, l0, d);

    // Round 1: u=Finite(0) ∈ Û_d ✓, suffix has Finite(2) → SKIP (non-binary suffix)
    // Round 2: u=Finite(2) ∈ Û_d ✓, suffix has Finite(1) binary ✓ → OK
    // Round 3: u=Finite(1) ∉ Û_d ✗ → SKIP (u not in Û_d)
    assert_eq!(contributions.len(), 1);  // Only round 2
    assert!(contributions.iter().any(|c| c.round == 2));
    assert!(contributions.iter().all(|c| c.round != 1 && c.round != 3));

    // β with ∞ in suffix position
    // indices (2, 0, 1) → values (Finite(1), ∞, Finite(0))
    let beta = tuple_from_indices(&[2, 0, 1]);
    let contributions = compute_idx4(&beta, l0, d);

    // Round 1: u=Finite(1) ∉ Û_d ✗ → SKIP (u not in Û_d)
    // Round 2: u=∞ ∈ Û_d ✓, suffix has Finite(0) binary ✓ → OK
    // Round 3: u=Finite(0) ∈ Û_d ✓, suffix=() binary ✓ → OK
    assert_eq!(contributions.len(), 2);  // Rounds 2 and 3
    assert!(contributions.iter().all(|c| c.round != 1));
    assert!(contributions.iter().any(|c| c.round == 2));
    assert!(contributions.iter().any(|c| c.round == 3));
}

#[test]
fn test_accumulator_prefix_index_helpers() {
    let idx = AccumulatorPrefixIndex {
        l0: 5,
        round: 3,
        v_idx: 10,
        u_idx: 2,
        y_idx: 3,
    };

    assert_eq!(idx.round_0idx(), 2);      // 3 - 1
    assert_eq!(idx.prefix_len(), 2);      // 3 - 1
    assert_eq!(idx.suffix_len(), 2);      // 5 - 3
}
```

---

## Eq-Poly Precomputation (Reusing Existing Code)

Procedure 9 requires precomputing eq-poly evaluations for E_in and E_out. The codebase has two approaches:

### Procedure 2 vs Procedure 3

| Procedure | Returns | Use Case |
|-----------|---------|----------|
| **Procedure 2** | `Vec<Scalar>` (single flat vector, size 2^ℓ) | When you only need the final evaluations |
| **Procedure 3** | `Vec<Vec<Scalar>>` (pyramid of all levels) | When you need intermediate levels for lookups |

### Procedure 2: `EqPolynomial::evals_from_points`

```rust
// src/polys/eq.rs:60-79
pub fn evals_from_points(r: &[Scalar]) -> Vec<Scalar>
```

Given `r = [r_1, ..., r_m]`, computes all `2^m` evaluations of `eq(r, x)` for all `x ∈ {0,1}^m` in O(2^m) time using a doubling algorithm:

```rust
pub fn evals_from_points(r: &[Scalar]) -> Vec<Scalar> {
    let l = r.len();
    let mut evals: Vec<Scalar> = vec![Scalar::ZERO; (2_usize).pow(l as u32)];
    let mut size = 1;
    evals[0] = Scalar::ONE;

    for r in r.iter().rev() {
        let (evals_left, evals_right) = evals.split_at_mut(size);
        let (evals_right, _) = evals_right.split_at_mut(size);

        // evals_right[j] = evals_left[j] * r  (for x_i = 1)
        // evals_left[j] -= evals_right[j]     (for x_i = 0)
        zip_with_for_each!(par_iter_mut, (evals_left, evals_right), |x, y| {
            *y = *x * r;
            *x -= &*y;
        });

        size *= 2;
    }
    evals
}
```

### Procedure 3: `compute_eq_polynomials` (pyramid)

```rust
// src/sumcheck/mod.rs:898-917 (inside EqSumCheckInstance::new)
let compute_eq_polynomials = |taus: Vec<&E::Scalar>| -> Vec<Vec<E::Scalar>> {
    let len = taus.len();
    let mut result = Vec::with_capacity(len + 1);

    result.push(vec![E::Scalar::ONE]);  // Level 0: size 1

    for i in 0..len {
        let tau = taus[i];
        let prev = &result[i];
        let mut v_next = prev.to_vec();
        v_next.par_extend(prev.par_iter().map(|v| *v * tau));
        let (first, last) = v_next.split_at_mut(prev.len());
        first.par_iter_mut().zip(last).for_each(|(a, b)| *a -= *b);
        result.push(v_next);
    }

    result  // Returns Vec<Vec<>> - pyramid of ALL intermediate levels
};
```

Returns a pyramid where `result[i]` has size `2^i`:
- `result[0]` = `[1]`
- `result[1]` = `[1-τ₁, τ₁]`
- `result[2]` = `[(1-τ₁)(1-τ₂), (1-τ₁)τ₂, τ₁(1-τ₂), τ₁τ₂]`
- ...

### Which to Use for Procedure 9

For **E_in** and **E_out** precomputation: Use **Procedure 2** (`evals_from_points`).

These are one-time precomputations where we only need the final level of evaluations, not intermediate levels. The pyramid approach (Procedure 3) is used by `EqSumCheckInstance` for the runtime lookups during rounds, but that's a different context.

### E_in Precomputation

From the paper (Procedure 9 input):
$$E_{in} := \{eq(w_{[l0+1:l0+l/2]}, x_{in})\}_{x_{in} \in \{0,1\}^{l/2}}$$

**Note:** The paper's x_in domain size is ℓ/2 (the slice `w[ℓ₀+1 : ℓ₀+ℓ/2]` has length ℓ/2 in 1-indexed notation).

In code (0-indexed):
```rust
let half = l / 2;
let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + half]);
// Size: 2^{half} = 2^{ℓ/2}  (NOT 2^{half - l0}!)
```

**Note:** The slice `taus[l0..l0 + half]` has length `half`, so `e_in.len() == 2^half`.

### E_out Precomputation

From the paper (Procedure 9 input), for each round `i ∈ [1, ℓ₀]`:
$$E_{out,i} := \{eq((w_{[i+1:l0]}, w_{[l/2+l0+1:]}), (y, x_{out}))\}_{(y, x_{out}) \in \{0,1\}^{l0-i} \times \{0,1\}^{l/2-l0}}$$

This concatenates two tau slices:
- `w[i+1:ℓ₀]` = remaining first-half taus after position i
- `w[ℓ/2+ℓ₀+1:]` = second-half taus from position (ℓ/2+ℓ₀+1) onwards

In code (0-indexed):
```rust
let e_out: Vec<Vec<Scalar>> = (1..=l0)
    .map(|i| {
        // Concatenate w[i..l0] and w[(half + l0)..]
        let combined: Vec<Scalar> = taus[i..l0]
            .iter()
            .chain(taus[(half + l0)..].iter())
            .cloned()
            .collect();
        EqPolynomial::evals_from_points(&combined)
    })
    .collect();
// E_out[i-1] has size 2^{(l0 - i) + (l - half - l0)} = 2^{ℓ/2 - i}
```

### Index Correspondence

The output of `evals_from_points` uses binary indexing:
- `evals[idx]` = `eq(r, x)` where `x` is the binary representation of `idx`
- For E_out, the index encodes `(y, x_out)` as: `idx = (y_bits << (half - l0)) | x_out_bits`

### E_y Precomputation (Optimized Suffix Pyramid)

**Problem:** E_y[i] = eq(τ[i+1:ℓ₀], y) for each round i needs **suffix** slices of τ.

**Naive approach** (O(ℓ₀ · 2^ℓ₀) total work):
```rust
let e_y: Vec<Vec<S>> = (0..l0)
    .map(|i| EqPolynomial::evals_from_points(&taus[i + 1..l0]))
    .collect();
```

This recomputes overlapping suffixes redundantly.

**Optimized approach** (O(2^ℓ₀) total work): Build a **suffix pyramid** in one pass.

```
Suffix slices for ℓ₀=4, τ = [τ₀, τ₁, τ₂, τ₃]:

E_y[0] = eq(τ[1:4], ·) = eq([τ₁, τ₂, τ₃], ·)  size 2³ = 8
E_y[1] = eq(τ[2:4], ·) = eq([τ₂, τ₃], ·)      size 2² = 4
E_y[2] = eq(τ[3:4], ·) = eq([τ₃], ·)          size 2¹ = 2
E_y[3] = eq(τ[4:4], ·) = eq([], ·) = [1]      size 2⁰ = 1
```

Build backwards from the empty suffix:

```
Step 0: E_y[3] = [1]                                    (empty eq)
Step 1: E_y[2] = extend E_y[3] with τ₃                  → [1-τ₃, τ₃]
Step 2: E_y[1] = extend E_y[2] with τ₂                  → 4 elements
Step 3: E_y[0] = extend E_y[1] with τ₁                  → 8 elements
```

**Implementation:**

```rust
/// Compute suffix eq-polynomials: E_y[i] = eq(τ[i+1:ℓ₀], y) for all i ∈ [0, ℓ₀)
///
/// Uses pyramid approach: build from the end (τ[ℓ₀-1]) backwards to τ[1].
/// Each step extends the previous suffix by prepending one more τ value.
///
/// # Arguments
/// * `taus` - τ values for the first ℓ₀ variables (τ[0:ℓ₀])
/// * `l0` - number of small-value rounds
///
/// # Returns
/// Vec of length ℓ₀, where result[i] has 2^{ℓ₀-i-1} elements.
/// result[i][y] = eq(τ[i+1:ℓ₀], y) for y ∈ {0,1}^{ℓ₀-i-1}
///
/// # Complexity
/// O(2^ℓ₀) total field multiplications (vs O(ℓ₀ · 2^ℓ₀) naive)
pub fn compute_suffix_eq_pyramid<S: PrimeField>(taus: &[S], l0: usize) -> Vec<Vec<S>> {
    assert!(taus.len() >= l0, "taus must have at least l0 elements");

    let mut result: Vec<Vec<S>> = vec![vec![]; l0];

    // Base case: E_y[l0-1] = eq([], ·) = [1] (empty suffix)
    result[l0 - 1] = vec![S::ONE];

    // Build backwards: each step prepends one τ value
    // E_y[i] = eq(τ[i+1:l0], ·) is built from E_y[i+1] = eq(τ[i+2:l0], ·)
    // by prepending τ[i+1]
    for i in (0..l0 - 1).rev() {
        let tau = taus[i + 1];
        let prev = &result[i + 1];
        let prev_len = prev.len();

        // New table has 2× the entries (prepending a new variable)
        let mut next = Vec::with_capacity(prev_len * 2);

        // For multilinear indexing: first variable is high bit
        // new_idx = new_bit * prev_len + old_idx
        //
        // new_bit = 0: eq factor is (1 - τ)
        // new_bit = 1: eq factor is τ

        // First half: new_bit = 0
        for &v in prev.iter() {
            next.push(v * (S::ONE - tau));
        }

        // Second half: new_bit = 1
        for &v in prev.iter() {
            next.push(v * tau);
        }

        result[i] = next;
    }

    result
}

#[cfg(test)]
mod suffix_pyramid_tests {
    use super::*;

    #[test]
    fn test_suffix_pyramid_sizes() {
        let l0 = 4;
        let taus: Vec<Fr> = (0..l0).map(|i| Fr::from(i as u64 + 2)).collect();
        let pyramid = compute_suffix_eq_pyramid(&taus, l0);

        assert_eq!(pyramid.len(), l0);
        for i in 0..l0 {
            let expected_size = 1 << (l0 - i - 1);
            assert_eq!(pyramid[i].len(), expected_size,
                "E_y[{}] should have size {}", i, expected_size);
        }
    }

    #[test]
    fn test_suffix_pyramid_base_case() {
        let l0 = 3;
        let taus: Vec<Fr> = (0..l0).map(|_| Fr::random(&mut OsRng)).collect();
        let pyramid = compute_suffix_eq_pyramid(&taus, l0);

        // E_y[l0-1] = [1] (empty suffix)
        assert_eq!(pyramid[l0 - 1].len(), 1);
        assert_eq!(pyramid[l0 - 1][0], Fr::ONE);
    }

    #[test]
    fn test_suffix_pyramid_single_tau() {
        let l0 = 3;
        let taus: Vec<Fr> = (0..l0).map(|_| Fr::random(&mut OsRng)).collect();
        let pyramid = compute_suffix_eq_pyramid(&taus, l0);

        // E_y[l0-2] = eq([τ_{l0-1}], ·) = [1-τ, τ]
        let tau_last = taus[l0 - 1];
        assert_eq!(pyramid[l0 - 2].len(), 2);
        assert_eq!(pyramid[l0 - 2][0], Fr::ONE - tau_last, "eq(τ, 0) = 1-τ");
        assert_eq!(pyramid[l0 - 2][1], tau_last, "eq(τ, 1) = τ");
    }

    #[test]
    fn test_suffix_pyramid_matches_naive() {
        // Verify pyramid matches independent computation
        let l0 = 4;
        let taus: Vec<Fr> = (0..l0).map(|_| Fr::random(&mut OsRng)).collect();

        let pyramid = compute_suffix_eq_pyramid(&taus, l0);

        for i in 0..l0 {
            let naive = if i + 1 >= l0 {
                vec![Fr::ONE]
            } else {
                EqPolynomial::evals_from_points(&taus[i + 1..l0])
            };

            assert_eq!(pyramid[i].len(), naive.len(),
                "Size mismatch at i={}", i);

            for (j, (&p, &n)) in pyramid[i].iter().zip(naive.iter()).enumerate() {
                assert_eq!(p, n,
                    "Value mismatch at E_y[{}][{}]", i, j);
            }
        }
    }

    #[test]
    fn test_suffix_pyramid_indexing() {
        // Verify index semantics: pyramid[i][y] = eq(τ[i+1:l0], y)
        let l0 = 3;
        let tau1 = Fr::from(5);
        let tau2 = Fr::from(7);
        let tau0 = Fr::from(3);  // Not used in any E_y suffix
        let taus = vec![tau0, tau1, tau2];

        let pyramid = compute_suffix_eq_pyramid(&taus, l0);

        // E_y[0] = eq([τ₁, τ₂], y) for y ∈ {0,1}²
        // Index 0 = (0,0): eq = (1-τ₁)(1-τ₂)
        // Index 1 = (0,1): eq = (1-τ₁)(τ₂)
        // Index 2 = (1,0): eq = (τ₁)(1-τ₂)
        // Index 3 = (1,1): eq = (τ₁)(τ₂)
        assert_eq!(pyramid[0][0], (Fr::ONE - tau1) * (Fr::ONE - tau2));
        assert_eq!(pyramid[0][1], (Fr::ONE - tau1) * tau2);
        assert_eq!(pyramid[0][2], tau1 * (Fr::ONE - tau2));
        assert_eq!(pyramid[0][3], tau1 * tau2);

        // E_y[1] = eq([τ₂], y) for y ∈ {0,1}
        assert_eq!(pyramid[1][0], Fr::ONE - tau2);
        assert_eq!(pyramid[1][1], tau2);

        // E_y[2] = eq([], ·) = [1]
        assert_eq!(pyramid[2][0], Fr::ONE);
    }
}
```

**Complexity comparison:**

| Approach | Time | Space |
|----------|------|-------|
| Naive (ℓ₀ separate calls) | O(ℓ₀ · 2^ℓ₀) | O(2^ℓ₀) per call |
| Suffix pyramid | O(2^ℓ₀) | O(2^ℓ₀) total |

For ℓ₀=8, this is **8× fewer multiplications**.

---

##
: Build Accumulators (Parallel)

Parallelized over the `x_out` loop using Rayon's **fold-reduce** pattern.

### Why fold instead of map?

| Pattern | Allocations | Memory |
|---------|-------------|--------|
| `.map(\|x\| { new Acc; ... })` | One per item | O(x_out_size) worst case |
| `.fold(\|\| new Acc, \|acc, x\| ...)` | One per thread | O(num_threads) |

With `fold`, each thread **reuses** its accumulator across all items it processes:

```
Rayon Thread Pool (fixed size, e.g., 8 threads)

Thread 0: acc_0  ← processes x_out=0, then x_out=8, then x_out=16, ...
Thread 1: acc_1  ← processes x_out=1, then x_out=9, then x_out=17, ...
...

Total accumulators = num_threads (e.g., 8)
NOT x_out_size (e.g., 4096)
```

Memory: 8 threads × 2.8 MB = **~22 MB** (regardless of x_out_size)

### Eq-Poly Table Structure

The original paper combines E_out,i into a single table over (y, x_out). We split it for clarity:

| Table | Formula | Depends on round? | Size |
|-------|---------|-------------------|------|
| E_in | eq(τ[l0 : l0+half], x_in) | No | 2^(ℓ/2) |
| E_xout | eq(τ[half+l0 : l], x_out) | No | 2^(ℓ/2-ℓ₀) |
| E_y[i] | eq(τ[i+1 : l0], y) | Yes (shrinks) | 2^(ℓ₀-i-1) |

**Key insight:** E_xout is constant across rounds, so precompute once. E_y shrinks each round (suffix slices of τ).

The original E_out,i[y, x_out] = E_y[i][y] × E_xout[x_out].

### Implementation

```rust
/// Procedure 9: Build accumulators A_i(v, u) for Spartan's first sum-check (Algorithm 6).
///
/// Computes accumulators for: g(X) = eq(τ, X) · (Az(X) · Bz(X) - Cz(X))
///
/// For Spartan, d = 3 (cubic polynomial: eq · Az · Bz contributes degree 3).
///
/// Parallelism strategy:
/// - Outer parallel loop over x_out values (using Rayon fold-reduce)
/// - Each thread maintains thread-local accumulators
/// - Final reduction merges all thread-local results via element-wise addition
///
/// FUTURE OPTIMIZATIONS (not yet implemented):
/// 1. Binary points yield zero: For β ∈ {0,1}^l0, Az(β)·Bz(β) - Cz(β) = 0
///    by the R1CS relation. These β values can be skipped entirely.
/// 2. Cz term vanishes at ∞: When β contains any ∞ coordinate, only Az·Bz
///    contributes (Cz is lower degree). Can skip Cz extension for those β.
pub fn build_accumulators<S: PrimeField + Send + Sync>(
    poly_az: &[S],   // Az evaluations over {0,1}^ℓ (size 2^ℓ)
    poly_bz: &[S],   // Bz evaluations over {0,1}^ℓ (size 2^ℓ)
    poly_cz: &[S],   // Cz evaluations over {0,1}^ℓ (size 2^ℓ)
    taus: &[S],      // τ vector of length ℓ
    l0: usize,
) -> SmallValueAccumulators<S> {
    let d = 3;  // Spartan: degree 3 (eq · Az · Bz)
    let l = poly_az.len().log_2();
    let half = l / 2;
    let num_x_out = 1 << (half - l0);
    let num_x_in = 1 << half;
    let num_prefix = 1 << l0;
    let num_betas = (d + 1).pow(l0 as u32);

    // ===== Precompute eq-poly tables =====

    // E_in = eq(τ[l0 : l0+half], x_in) — same for all rounds
    let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + half]);

    // E_xout = eq(τ[half+l0 : l], x_out) — same for all rounds
    let e_xout = EqPolynomial::evals_from_points(&taus[half + l0..l]);

    // E_y[i] = eq(τ[i+1 : l0], y) — different per round (shrinks)
    // Use optimized suffix pyramid approach (see compute_suffix_eq_pyramid)
    let e_y = compute_suffix_eq_pyramid(&taus[..l0], l0);

    // ===== Precompute routing table for idx4 =====
    // CRITICAL: Do NOT decode β or allocate inside the β loop!
    // Precompute all routes once: routes[beta_idx] = [AccumulatorPrefixIndex, ...]
    let d_plus_1 = d + 1;
    let routes: Vec<Vec<AccumulatorPrefixIndex>> = (0..num_betas)
        .map(|beta_idx| {
            let beta = UdTuple::from_flat_index(beta_idx, d_plus_1, l0);
            compute_idx4(&beta, l0, d)
        })
        .collect();

    // ===== Parallel fold-reduce over x_out =====

    (0..num_x_out)
        .into_par_iter()
        .fold(
            || SmallValueAccumulators::new(l0, d),
            |mut local_acc, x_out| {
                // Temporary accumulators tA[β] for all β ∈ U_d^l0
                let mut tA = vec![S::ZERO; num_betas];

                // ----- Inner loop: accumulate into tA -----
                // NOTE: This loop can be chunked for additional parallelism.
                // Each chunk would have its own tA, then merge via element-wise addition.
                // Size is 2^(ℓ/2), so chunking may help for large ℓ.
                for x_in in 0..num_x_in {
                    // Step 1: Gather prefix evaluations for Az, Bz, Cz
                    // For each polynomial p ∈ {Az, Bz, Cz}, gather p(b, x_in, x_out) for b ∈ {0,1}^l0
                    let suffix = x_in * num_x_out + x_out;
                    let az_prefix = gather_prefix_evals(poly_az, l0, suffix);
                    let bz_prefix = gather_prefix_evals(poly_bz, l0, suffix);
                    let cz_prefix = gather_prefix_evals(poly_cz, l0, suffix);

                    // Step 2: Extend each polynomial from {0,1}^l0 to U_d^l0 (Procedure 6)
                    // This gives us Az(β), Bz(β), Cz(β) for all β ∈ U_d^l0
                    let az_ext = extend_to_lagrange_domain(az_prefix, l0, d);
                    let bz_ext = extend_to_lagrange_domain(bz_prefix, l0, d);
                    let cz_ext = extend_to_lagrange_domain(cz_prefix, l0, d);

                    // Step 3: Compute Az·Bz - Cz at each β and accumulate into tA
                    let e_in_val = e_in[x_in];
                    for beta_idx in 0..num_betas {
                        let val = az_ext.get(beta_idx) * bz_ext.get(beta_idx)
                                - cz_ext.get(beta_idx);
                        tA[beta_idx] += e_in_val * val;
                    }
                }

                // ----- Distribute tA to final accumulators via idx4 -----
                let e_xout_val = e_xout[x_out];

                for beta_idx in 0..num_betas {
                    if tA[beta_idx].is_zero() {
                        continue;
                    }

                    // Use precomputed routes — NO allocation or β decoding here!
                    for route in &routes[beta_idx] {
                        let round_0idx = route.round_0idx();
                        let e_y_val = e_y[round_0idx][route.y_idx];
                        local_acc.accumulate(
                            round_0idx, route.v_idx, route.u_idx,
                            e_y_val * e_xout_val * tA[beta_idx]
                        );
                    }
                }

                local_acc
            }
        )
        .reduce(
            || SmallValueAccumulators::new(l0, d),
            |mut a, b| { a.merge(&b); a }
        )
}
```

### Helper Functions

```rust
// Functions used in build_accumulators (defined in their respective sections):
//
// gather_prefix_evals: See "Gather Prefix Evaluations" section
//   fn gather_prefix_evals<S: Copy>(poly: &[S], l0: usize, suffix: usize) -> Vec<S>
//
// extend_to_lagrange_domain: See "Procedure 6" section
//   fn extend_to_lagrange_domain<S>(evals: Vec<S>, l0: usize, d: usize) -> LagrangeEvals<S>
//
// compute_idx4: See "compute_idx4 Mapping" section
//   fn compute_idx4(beta: &[usize], l0: usize, d: usize) -> Vec<AccumulatorPrefixIndex>
//
// flat_index_to_tuple, tuple_to_flat_index: See below

// ===== Tuple Indexing (Mixed-Radix) =====
// Canonical definitions for converting between flat indices and tuples.
// Used throughout for β ∈ U_d^k indexing.

/// Convert tuple β ∈ U_d^k to flat index (mixed-radix encoding)
fn tuple_to_flat_index(tuple: &[usize], base: usize) -> usize {
    tuple.iter().fold(0, |acc, &v| acc * base + v)
}

/// Convert flat index to tuple β ∈ U_d^k (mixed-radix decoding)
fn flat_index_to_tuple(mut idx: usize, base: usize, len: usize) -> Vec<usize> {
    let mut tuple = vec![0; len];
    for i in (0..len).rev() {
        tuple[i] = idx % base;
        idx /= base;
    }
    tuple
}
```

### How Bucketing Works

The "group by (i, v, u)" happens via **direct indexed writes** — no HashMap needed:

```rust
for idx in compute_idx4(&beta, l0, d) {
    *partial.get_mut(idx.round_0idx(), idx.v_idx, idx.u_idx) += e_out_val * t_a_val;
    //       ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    //       Direct array indexing to the correct bucket
}
```

**Inside one thread** (processing multiple x_out values):

```
x_out=0: β=(1,2,1) → idx4 says write to (round=0, v=0, u=1)
                   → partial.rounds[0].data[0][1] += contribution

x_out=8: β=(1,2,1) → idx4 says write to (round=0, v=0, u=1)  (same slot!)
                   → partial.rounds[0].data[0][1] += contribution

         β=(0,1,2) → idx4 says write to (round=0, v=0, u=0)  (different slot)
                   → partial.rounds[0].data[0][0] += contribution
```

Multiple β values and multiple x_out values accumulate into the same slots via `+=`.

**Across threads** (reduce phase):

```
Thread 0: partial_0.rounds[0].data[0][1] = 150
Thread 1: partial_1.rounds[0].data[0][1] = 230
Thread 2: partial_2.rounds[0].data[0][1] = 85
...
────────────────────────────────────────────────
After reduce:
final.rounds[0].data[0][1] = 150 + 230 + 85 + ... = A₁(v=0, u=1)
```

### Mathematical Justification

The accumulator sum decomposes over x_out:

$$A_i(v, u) = \sum_{x_{out}} \underbrace{\sum_{\beta: (i,v,u,\_) \in idx4(\beta)} E_{out,i}[y, x_{out}] \cdot t_A[\beta]}_{\text{partial}_{x_{out}}[i][v][u]}$$

Each x_out contributes **additively**, so we can:
1. **fold**: Each thread accumulates its x_out values into one accumulator
2. **reduce**: Sum thread-local accumulators element-wise

---

## Algorithm 6: Main Prover

Three phases:
1. **Phase 1 (rounds 1..ℓ₀):** Accumulator-based with R_i vector
2. **Phase 2 (round ℓ₀+1):** Streaming transition
3. **Phase 3 (rounds ℓ₀+2..ℓ):** Delegate to EqSumCheckInstance

```rust
impl<E: Engine> SumcheckProof<E> {
    /// Prove cubic sumcheck using Algorithm 6 (EqPoly-SmallValueSC).
    pub fn prove_cubic_with_three_inputs_alg6(
        claim: &E::Scalar,
        taus: Vec<E::Scalar>,
        poly_A: &mut MultilinearPolynomial<E::Scalar>,
        poly_B: &mut MultilinearPolynomial<E::Scalar>,
        poly_C: &mut MultilinearPolynomial<E::Scalar>,
        transcript: &mut E::TE,
        l0: Option<usize>,
    ) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
        let num_rounds = taus.len();
        let d = 3;  // cubic
        let l0 = l0.unwrap_or_else(|| select_l0(num_rounds, d));

        let mut r: Vec<E::Scalar> = Vec::new();
        let mut polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::new();
        let mut claim_per_round = *claim;

        // Precompute accumulators (Procedure 9)
        let accumulators = build_accumulators(
            &poly_A.Z, &poly_B.Z, &poly_C.Z,
            &taus, l0,
        );

        let mut lagrange_coeffs = LagrangeBasisCoeffs::new(d);

        // ========== Phase 1: Accumulator-based rounds ==========
        for round in 0..l0 {
            // Compute t_i(u) from accumulators
            let t_evals = compute_t_from_accumulators(&accumulators, &lagrange_coeffs, round, d);

            // Compute round polynomial (needs adaptation for eq factor)
            let poly = compute_round_poly_from_t(&t_evals, &taus, &r, round, claim_per_round, d)?;

            transcript.absorb(b"p", &poly);
            let r_i = transcript.squeeze(b"c")?;
            r.push(r_i);
            polys.push(poly.compress());

            claim_per_round = poly.evaluate(&r_i);
            lagrange_coeffs.extend(&r_i);
        }

        // ========== Phase 2: Streaming transition (round ℓ₀+1) ==========
        // Bind polynomials with challenges from Phase 1
        for r_i in r.iter().take(l0) {
            rayon::join(
                || poly_A.bind_poly_var_top(r_i),
                || rayon::join(
                    || poly_B.bind_poly_var_top(r_i),
                    || poly_C.bind_poly_var_top(r_i),
                ),
            );
        }

        // One streaming round
        // ... (similar to existing prove_cubic_with_three_inputs)

        // ========== Phase 3: Delegate to EqSumCheckInstance ==========
        let remaining_taus = taus[(l0 + 1)..].to_vec();
        let mut eq_instance = eq_sumcheck::EqSumCheckInstance::<E>::new(remaining_taus);

        for round in (l0 + 1)..num_rounds {
            // ... use eq_instance.evaluation_points_cubic_with_three_inputs
        }

        Ok((
            SumcheckProof { compressed_polys: polys },
            r,
            vec![poly_A[0], poly_B[0], poly_C[0]],
        ))
    }
}
```

---

## Per-Round State and Data Lifetime

### When Each Data Structure is Used

| Data | When Built | When Used | When Discarded |
|------|------------|-----------|----------------|
| `E_in` | Precomputation | Precomputation only | After precomputation |
| `E_out,i` | Precomputation | Precomputation only | After precomputation |
| `A_1..A_ℓ₀` | Precomputation (all at once) | A_i read in round i | After Phase 1 completes |
| `R_i` | Round i-1 | Round i | Extended to R_{i+1} |

### Accumulator Usage Pattern

Each `A_i` is used **ONLY in round i**. However, we **precompute and store ALL of them** upfront since building accumulators requires streaming over the full polynomials.

```
PRECOMPUTATION:
  Build A_1, A_2, ..., A_ℓ₀ in ONE pass over poly_A, poly_B, poly_C
  Store all in memory (~87K elements for d=3, ℓ₀=8)

Round 1: Read A_1(v, u) where v ∈ U_d^0 = {∅}, u ∈ Û_d
         → A_1 has (d+1)^0 = 1 prefix, each with d u-values
         → For d=3: 1 × 3 = 3 entries total
         → Dot product with R_1 = [1]

Round 2: Read A_2(v, u) where v ∈ U_d^1, u ∈ Û_d
         → A_2 has (d+1)^1 = 4 prefixes, each with d u-values
         → For d=3: 4 × 3 = 12 entries total
         → Dot product with R_2 (size 4)

Round 3: Read A_3(v, u) where v ∈ U_d^2, u ∈ Û_d
         → A_3 has (d+1)^2 = 16 prefixes, each with d u-values
         → For d=3: 16 × 3 = 48 entries total
         → Dot product with R_3 (size 16)

...

Round ℓ₀: Read A_ℓ₀(v, u)
          → Dot product with R_ℓ₀ (size (d+1)^{ℓ₀-1})
          → After this, all accumulators can be dropped
```

### E_out Tables - Precomputation Only

```
┌────────────────────────────────────────────────────────────┐
│                 PRECOMPUTATION (Procedure 9)               │
├────────────────────────────────────────────────────────────┤
│  E_in, E_out,1, E_out,2, ..., E_out,ℓ₀                    │
│           ↓                                                │
│  Build accumulators A_1, A_2, ..., A_ℓ₀                   │
│           ↓                                                │
│  DISCARD E_in and all E_out,i  ← No longer needed!        │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                 PROVING (Rounds 1..ℓ₀)                     │
├────────────────────────────────────────────────────────────┤
│  Only use: A_i (for current round i) + R_i                │
│  E_out,i not needed anymore                                │
└────────────────────────────────────────────────────────────┘
```

### What's Persisted Across Rounds

| Variable | Type | Per-Round Change |
|----------|------|------------------|
| `accumulators` | `SmallValueAccumulators` | Read-only (precomputed) |
| `R_i` | `LagrangeBasisCoeffs` | **Grows**: size $(d+1)^{i-1}$ → $(d+1)^i$ |
| `polys` | `Vec<CompressedUniPoly>` | Append 1 compressed poly |
| `r` | `Vec<Scalar>` | Append 1 challenge |
| `claim_per_round` | `Scalar` | Updated |
| `poly_A, B, C` | `MultilinearPolynomial` | **Untouched** until Phase 2 |

### Comparison: Algorithm 5 vs Algorithm 6 (Phase 1)

| Aspect | Algorithm 5 (Current) | Algorithm 6 (Phase 1) |
|--------|----------------------|----------------------|
| `poly_A, B, C` | **Halved each round** | **Untouched** until Phase 2 |
| `eq_instance` | Updated via `bound()` | Not used (use accumulators) |
| `R_i` vector | ❌ Doesn't exist | ✅ Grows each round |
| Accumulators | ❌ Don't exist | ✅ Precomputed, read-only |
| Per-round work | $O(2^{l-i})$ | $O((d+1)^{i-1} \cdot (d+1))$ |

### Memory Strategy: Precompute All Accumulators

**Decision: We precompute and store ALL accumulators A_1, A_2, ..., A_ℓ₀ upfront.**

Rationale:
1. Accumulator building requires a streaming pass over the full polynomials (expensive)
2. Per-round usage is just a dot product with R_i (cheap)
3. Total storage is manageable: $\sum_{i=1}^{l0} (d+1)^i = O((d+1)^{l0})$

```rust
// Build ALL accumulators upfront (one expensive pass)
let accumulators = build_accumulators(&poly_az, &poly_bz, &poly_cz, &taus, l0);

// Each round i just reads accumulators.data[i] - O((d+1)^i) dot product
for i in 0..l0 {
    let t_evals = compute_t(&accumulators.data[i], &R);
    // ... rest of round
}
```

#### Memory Footprint

For $d=3$, $l0=8$:
- A_1: 4 elements
- A_2: 16 elements
- A_3: 64 elements
- ...
- A_8: 65,536 elements
- **Total: ~87K field elements ≈ 2.8 MB** (for 32-byte field elements)

This fits comfortably in L3 cache, making per-round access fast.

---

## Integration with Spartan

### Option: Conditional Call

```rust
// In SpartanSNARK::prove()
const ALG6_MIN_ROUNDS: usize = 16;

let (sc_proof_outer, r_x, claims_outer) = if num_rounds_x >= ALG6_MIN_ROUNDS {
    SumcheckProof::prove_cubic_with_three_inputs_alg6(
        &E::Scalar::ZERO,
        tau,
        &mut poly_Az,
        &mut poly_Bz,
        &mut poly_Cz,
        &mut transcript,
        None,  // auto-select ℓ₀
    )?
} else {
    SumcheckProof::prove_cubic_with_three_inputs(
        &E::Scalar::ZERO,
        tau,
        &mut poly_Az,
        &mut poly_Bz,
        &mut poly_Cz,
        &mut transcript,
    )?
};
```

### Variable Partitioning Constraint

**Critical Constraint: ℓ₀ ≤ ℓ/2**

Algorithm 6 (Procedure 9) partitions the ℓ sumcheck variables into three segments:

| Segment | Length | Description | Domain |
|---------|--------|-------------|--------|
| **(v, u, y)** | ℓ₀ | Prefix variables for small-value optimization | Extended to U_d^ℓ₀ |
| **x_in** | ℓ/2 | Inner sum variables (E_in weighted) | Binary {0,1}^{ℓ/2} |
| **x_out** | ℓ/2 - ℓ₀ | Outer sum variables (E_out weighted) | Binary {0,1}^{ℓ/2-ℓ₀} |
| **Total** | **ℓ** | | |

For this partitioning to be valid:

1. **x_in must have non-negative length:** ℓ/2 ≥ 0 (always true)
2. **x_out must have non-negative length:** ℓ/2 - ℓ₀ ≥ 0 → **ℓ₀ ≤ ℓ/2**
3. **Total must equal ℓ:** ℓ₀ + ℓ/2 + (ℓ/2 - ℓ₀) = ℓ ✓

```
Variable layout for ℓ=20, ℓ₀=3:

┌─────────────────┬──────────────────────────┬─────────────────────┐
│    Prefix       │         x_in             │       x_out         │
│   (v, u, y)     │    (inner sum)           │    (outer sum)      │
├─────────────────┼──────────────────────────┼─────────────────────┤
│     3 vars      │       10 vars            │       7 vars        │
│   (ℓ₀ = 3)      │     (ℓ/2 = 10)           │  (ℓ/2 - ℓ₀ = 7)     │
├─────────────────┼──────────────────────────┼─────────────────────┤
│ Extended to     │ Binary, weighted by E_in │ Binary, weighted by │
│ U_d^3 = 64 pts  │ (size 2^10 = 1024)       │ E_out (size 2^7)    │
└─────────────────┴──────────────────────────┴─────────────────────┘
                              Total: 20 variables
```

**Why this constraint exists:**

The paper's eq-poly optimization (Section 5.2) splits the equality polynomial:
$$\tilde{eq}(w, X) = \tilde{eq}(w_L, x_L) \cdot \tilde{eq}(w_R, x_R)$$

where $x_L$ corresponds to x_in (ℓ/2 variables) and $x_R$ corresponds to x_out (ℓ/2 - ℓ₀ variables).
For the inner sum over x_in to be independent of the round, x_in must have exactly ℓ/2 bits.
This forces x_out = ℓ - ℓ₀ - ℓ/2 = ℓ/2 - ℓ₀, which requires ℓ₀ ≤ ℓ/2.

### Heuristic for ℓ₀ Selection

```rust
/// Select optimal ℓ₀ for Algorithm 6.
///
/// # Constraints
/// 1. **Variable partitioning:** ℓ₀ ≤ ℓ/2 (required for x_out ≥ 0)
/// 2. **Memory:** Accumulators fit in L2/L3 cache (≈2-4 MB)
/// 3. **Minimum rounds:** Need at least 2 rounds after ℓ₀ (streaming + final)
///
/// # Memory Analysis
/// Total accumulator size: Σᵢ (d+1)ⁱ × d ≈ (d+1)^ℓ₀ × d field elements
/// For d=3, ℓ₀=8: 4^8 × 3 × 32 bytes ≈ 6 MB (fits in L3)
///
/// # Returns
/// The selected ℓ₀ value, guaranteed to satisfy all constraints.
pub fn select_l0(num_rounds: usize, d: usize) -> usize {
    // Constraint 1: Memory-based maximum
    let memory_max_l0 = match d {
        2 => 10,  // 3^10 × 2 × 32 bytes ≈ 3.8 MB
        3 => 8,   // 4^8 × 3 × 32 bytes ≈ 6 MB
        4 => 7,   // 5^7 × 4 × 32 bytes ≈ 10 MB
        _ => 6,
    };

    // Constraint 2: Variable partitioning (ℓ₀ ≤ ℓ/2)
    let partition_max_l0 = num_rounds / 2;

    // Constraint 3: Need at least 2 rounds after ℓ₀ for streaming + final phases
    // This means ℓ₀ ≤ ℓ - 2, i.e., num_rounds - 2
    let phase_max_l0 = num_rounds.saturating_sub(2);

    // Take the minimum of all constraints
    let l0 = memory_max_l0
        .min(partition_max_l0)
        .min(phase_max_l0);

    // Ensure at least 1 round of small-value optimization (otherwise don't use Algorithm 6)
    l0.max(1)
}

/// Check if Algorithm 6 is applicable for the given parameters.
///
/// Returns `None` if Algorithm 6 shouldn't be used, otherwise returns the suggested ℓ₀.
pub fn should_use_algorithm6(num_rounds: usize, d: usize) -> Option<usize> {
    // Minimum viable: need at least 4 rounds (ℓ₀=1, streaming, 2 final rounds)
    if num_rounds < 4 {
        return None;
    }

    let l0 = select_l0(num_rounds, d);

    // Only use Algorithm 6 if we get at least 2 rounds of small-value optimization
    // (otherwise the precomputation overhead isn't worth it)
    if l0 >= 2 {
        Some(l0)
    } else {
        None
    }
}

#[cfg(test)]
mod select_l0_tests {
    use super::*;

    #[test]
    fn test_partition_constraint() {
        // ℓ₀ must be ≤ ℓ/2
        assert!(select_l0(10, 3) <= 5);  // ℓ=10 → ℓ₀ ≤ 5
        assert!(select_l0(20, 3) <= 10); // ℓ=20 → ℓ₀ ≤ 10
        assert!(select_l0(6, 3) <= 3);   // ℓ=6 → ℓ₀ ≤ 3
    }

    #[test]
    fn test_memory_constraint() {
        // Large ℓ should still be bounded by memory
        assert!(select_l0(100, 3) <= 8);  // d=3 memory limit
        assert!(select_l0(100, 2) <= 10); // d=2 memory limit
    }

    #[test]
    fn test_minimum_rounds() {
        // Very small ℓ should still work
        assert_eq!(select_l0(4, 3), 1);   // ℓ=4 → ℓ₀=min(8, 2, 2)=2, but .max(1)=2...
        // Actually: partition_max=2, phase_max=2, memory=8 → min=2
        assert!(select_l0(4, 3) >= 1);
    }

    #[test]
    fn test_should_use_algorithm6() {
        // Too few rounds
        assert_eq!(should_use_algorithm6(3, 3), None);

        // Borderline case
        assert!(should_use_algorithm6(6, 3).is_some());

        // Normal case
        let l0 = should_use_algorithm6(20, 3);
        assert!(l0.is_some());
        assert!(l0.unwrap() >= 2);
    }

    #[test]
    fn test_x_out_non_negative() {
        // For any valid l0, x_out = ℓ/2 - ℓ₀ must be ≥ 0
        for num_rounds in 4..=30 {
            let l0 = select_l0(num_rounds, 3);
            let x_out_len = (num_rounds / 2).saturating_sub(l0);
            assert!(x_out_len >= 0,
                "x_out negative for ℓ={}, ℓ₀={}", num_rounds, l0);
        }
    }
}
```

---

## Files to Create/Modify

> **Note:** We are implementing incrementally, keeping changes isolated from the
> existing `src/sumcheck.rs` file until Algorithm 6 integration.

### Phase 1: Procedure 9 (build_accumulators) — Current Focus

**`src/polys/eq.rs`** (extend existing)
- `compute_suffix_eq_pyramid()` — E_y precomputation for suffix eq tables

**`src/lagrange.rs`** (new file)
- `UdPoint`, `UdHatPoint`, `UdTuple` — domain types
- `ValueOneExcluded` — error type for Û_d conversion
- `LagrangeEvals` — extended evaluation storage
- `extend_one_variable()`, `extend_to_lagrange_domain()` — Procedures 5 & 6
- `lagrange_basis_evals()` — Lagrange basis evaluation (for R_i update later)

**`src/accumulators.rs`** (new file)
- `RoundAccumulator`, `SmallValueAccumulators` — data structures
- `AccumulatorPrefixIndex`, `compute_idx4()` — index mapping
- `gather_prefix_evals()` — gather operation
- `build_accumulators()` — main Procedure 9 function

### Phase 2: Algorithm 6 Integration — Later

1. **Modify:** `src/sumcheck.rs`
   - `LagrangeBasisCoeffs` (for R_i tensor tracking)
   - `prove_cubic_with_three_inputs_alg6` (full Algorithm 6 prover)

2. **Modify:** `src/spartan.rs`
   - Add conditional call to Algorithm 6

---

## Implementation Chunks (Phase 1)

Incremental implementation order for Procedure 9 (`build_accumulators`):

### Chunk 1: Domain Types (`src/lagrange.rs`)
- `UdPoint` enum — points in U_d = {∞, 0, 1, ..., d-1}
- `UdHatPoint` enum — points in Û_d = U_d \ {1}
- `UdTuple` struct — tuples β ∈ U_d^k
- `ValueOneExcluded` error type
- Unit tests for conversions (`to_index`, `from_index`, `to_ud_hat`, `to_flat_index`, etc.)

### Chunk 2: Lagrange Extension (`src/lagrange.rs`)
- `extend_one_variable()` — extends one var from {0,1} to U_d
- `extend_to_lagrange_domain()` — extends all ℓ₀ vars
- `LagrangeEvals` struct
- Unit tests (output size, boolean points preserved, matches direct evaluation)

### Chunk 3: Accumulator Data Structures (`src/accumulators.rs`)
- `RoundAccumulator` struct
- `SmallValueAccumulators` struct
- `merge()` operation for parallel reduction
- Unit tests

### Chunk 4: Index Mapping (`src/accumulators.rs`)
- `AccumulatorPrefixIndex` struct
- `compute_idx4()` function
- Unit tests for all edge cases (binary suffix filtering, u∈Û_d filtering)

### Chunk 5: Eq Pyramid (`src/polys/eq.rs`)
- `compute_suffix_eq_pyramid()` for E_y tables
- Unit tests

### Chunk 6: Gather Function (`src/accumulators.rs`)
- `gather_prefix_evals()` — gathers p(b, x_in, x_out) for all b ∈ {0,1}^ℓ₀
- Unit tests

### Chunk 7: build_accumulators (`src/accumulators.rs`)
- Assemble all pieces
- Parallel fold-reduce with Rayon
- Integration tests comparing against naive computation

**Dependency graph:**
```
Chunk 1 (Domain Types)
    ↓
Chunk 2 (Lagrange Extension)
    ↓
Chunk 3 (Accumulator Structs) ← Chunk 4 (Index Mapping)
    ↓                              ↓
    └──────────┬───────────────────┘
               ↓
Chunk 5 (Eq Pyramid) + Chunk 6 (Gather)
               ↓
         Chunk 7 (build_accumulators)
```

---

## Testing Strategy

### Overview

1. **Unit tests:** Each struct and helper function
2. **Property tests:** Verify `prove_alg6` produces same challenges/evals as `prove_cubic_with_three_inputs`
3. **Integration test:** Full Spartan prove/verify with Algorithm 6
4. **Benchmarks:** Compare performance at various constraint sizes

---

### Unit Tests for Procedure 6 (Extend Multilinear to U_d^ℓ₀)

```rust
#[test]
fn test_extend_output_size() {
    // Output should have (d+1)^ℓ₀ elements
    let d = 3;
    for l0 in 1..=4 {
        let input_size = 1 << l0;
        let input: Vec<Fr> = (0..input_size).map(|i| Fr::from(i as u64)).collect();
        let poly = MultilinearPolynomial::new(input);

        let extended = extend_to_ud_full(&poly, l0, d);

        let expected_size = (d + 1).pow(l0 as u32);
        assert_eq!(extended.len(), expected_size);
    }
}

#[test]
fn test_extend_preserves_boolean_points() {
    // Evaluations at {0,1}^ℓ₀ should match original
    let l0 = 3;
    let d = 3;
    let base = d + 1;

    let input: Vec<Fr> = (0..(1 << l0)).map(|_| Fr::random(&mut OsRng)).collect();
    let poly = MultilinearPolynomial::new(input.clone());
    let extended = extend_to_ud_full(&poly, l0, d);

    for b in 0..(1 << l0) {
        // Convert boolean index to U_d index: 0→1, 1→2
        let mut ud_idx = 0;
        for j in 0..l0 {
            let bit = (b >> (l0 - 1 - j)) & 1;
            ud_idx = ud_idx * base + (bit + 1);
        }
        assert_eq!(extended[ud_idx], input[b]);
    }
}

#[test]
fn test_extend_single_variable() {
    // Base case: Procedure 5 for linear polynomial
    let d = 3;
    let p0 = Fr::from(7);
    let p1 = Fr::from(19);

    let input = vec![p0, p1];
    let poly = MultilinearPolynomial::new(input);
    let extended = extend_to_ud_full(&poly, 1, d);

    // U_d = {∞, 0, 1, 2} at indices 0, 1, 2, 3
    assert_eq!(extended[0], p1 - p0, "p(∞) = leading coeff");
    assert_eq!(extended[1], p0, "p(0)");
    assert_eq!(extended[2], p1, "p(1)");
    assert_eq!(extended[3], p1.double() - p0, "p(2) = 2*p1 - p0");
}

#[test]
fn test_extend_bilinear() {
    // p(X, Y) = X * Y
    let d = 3;
    let l0 = 2;
    let base = d + 1;

    let input = vec![Fr::ZERO, Fr::ZERO, Fr::ZERO, Fr::ONE];
    let poly = MultilinearPolynomial::new(input);
    let extended = extend_to_ud_full(&poly, l0, d);

    // Check p(a, b) = a * b for finite points
    for a in 0..d {
        for b in 0..d {
            let idx = (a + 1) * base + (b + 1);
            let expected = Fr::from(a as u64) * Fr::from(b as u64);
            assert_eq!(extended[idx], expected, "p({}, {}) wrong", a, b);
        }
    }
}
```

| Test | What it verifies |
|------|------------------|
| `test_extend_output_size` | Output has (d+1)^ℓ₀ elements |
| `test_extend_preserves_boolean_points` | Original {0,1}^ℓ₀ evaluations unchanged |
| `test_extend_single_variable` | Procedure 5 (base case) is correct |
| `test_extend_bilinear` | Known p(X,Y)=XY extends correctly |

---

### Unit Tests for LagrangeBasisCoeffs (R_i Tensor)

```rust
#[test]
fn test_lagrange_basis_coeffs_initial() {
    // R_1 = [1]
    let d = 3;
    let coeffs = LagrangeBasisCoeffs::<Fr>::new(d);

    assert_eq!(coeffs.len(), 1);
    assert_eq!(coeffs.get(0), Fr::ONE);
}

#[test]
fn test_lagrange_basis_coeffs_size_growth() {
    // Size = (d+1)^{i-1} after i-1 extensions
    let d = 3;
    let base = d + 1;
    let mut coeffs = LagrangeBasisCoeffs::<Fr>::new(d);

    assert_eq!(coeffs.len(), 1);  // (d+1)^0

    coeffs.extend(&Fr::from(5));
    assert_eq!(coeffs.len(), base);  // (d+1)^1

    coeffs.extend(&Fr::from(7));
    assert_eq!(coeffs.len(), base * base);  // (d+1)^2
}

#[test]
fn test_lagrange_basis_coeffs_tensor_product() {
    // R_{i+1} = R_i ⊗ L(r_i)
    let d = 3;
    let base = d + 1;

    let r1 = Fr::from(7);
    let r2 = Fr::from(11);

    let mut coeffs = LagrangeBasisCoeffs::<Fr>::new(d);
    coeffs.extend(&r1);
    coeffs.extend(&r2);

    let l_r1 = lagrange_basis_evals(&r1, d);
    let l_r2 = lagrange_basis_evals(&r2, d);

    for i in 0..base {
        for j in 0..base {
            let idx = i * base + j;
            assert_eq!(coeffs.get(idx), l_r1[i] * l_r2[j]);
        }
    }
}

#[test]
fn test_lagrange_basis_coeffs_at_domain_points() {
    // When r = domain point, R becomes one-hot (selector)
    let d = 3;
    let base = d + 1;

    let mut coeffs = LagrangeBasisCoeffs::<Fr>::new(d);

    // r_1 = 0 (index 1 in U_d) → one-hot at index 1
    coeffs.extend(&Fr::ZERO);
    assert_eq!(coeffs.get(0), Fr::ZERO);  // L_∞(0) = 0
    assert_eq!(coeffs.get(1), Fr::ONE);   // L_0(0) = 1
    assert_eq!(coeffs.get(2), Fr::ZERO);  // L_1(0) = 0
    assert_eq!(coeffs.get(3), Fr::ZERO);  // L_2(0) = 0

    // r_2 = 1 (index 2 in U_d) → one-hot at 1*4+2 = 6
    coeffs.extend(&Fr::ONE);
    for i in 0..base*base {
        if i == 6 {
            assert_eq!(coeffs.get(i), Fr::ONE);
        } else {
            assert_eq!(coeffs.get(i), Fr::ZERO);
        }
    }
}

#[test]
fn test_lagrange_basis_coeffs_selects_accumulator() {
    // Key property: t_i(u) = Σ_v R_i[v] · A_i(v, u) selects correct entry
    // when challenges are domain points
    let d = 3;
    let base = d + 1;
    let l0 = 2;

    // Build accumulator with distinct values
    let mut acc = SmallValueAccumulators::<Fr>::new(l0, d);
    for v_idx in 0..base {
        for u_idx in 0..d {
            *acc.get_mut(1, v_idx, u_idx) = Fr::from((v_idx * 100 + u_idx) as u64);
        }
    }

    // r_1 = 1 (index 2 in U_d) → selects v_idx = 2
    let mut coeffs = LagrangeBasisCoeffs::<Fr>::new(d);
    coeffs.extend(&Fr::ONE);

    for u_idx in 0..d {
        let t_u: Fr = (0..base)
            .map(|v_idx| coeffs.get(v_idx) * acc.get(1, v_idx, u_idx))
            .sum();

        // Should equal A_2(v=2, u) since R selects v_idx = 2
        assert_eq!(t_u, acc.get(1, 2, u_idx));
    }
}
```

| Test | What it verifies |
|------|------------------|
| `test_lagrange_basis_coeffs_initial` | R_1 = [1] |
| `test_lagrange_basis_coeffs_size_growth` | Size = (d+1)^{i-1} after i-1 extensions |
| `test_lagrange_basis_coeffs_tensor_product` | R_{i+1} = R_i ⊗ L(r_i) |
| `test_lagrange_basis_coeffs_at_domain_points` | One-hot when r is a domain point |
| `test_lagrange_basis_coeffs_selects_accumulator` | t_i(u) = Σ_v R_i[v] · A_i(v, u) selects correct entry |

---

### Unit Tests for Procedure 9 (Build Accumulators)

```rust
#[test]
fn test_accumulator_dimensions() {
    // rounds[i] has (d+1)^i prefixes, each with d u-values
    // data[v_idx][u_idx] structure: outer Vec has num_prefixes, inner Vec has d
    let l0 = 3;
    let d = 3;
    let acc = SmallValueAccumulators::<Fr>::new(l0, d);

    assert_eq!(acc.rounds.len(), l0);

    // Round i (0-indexed) has (d+1)^i prefixes
    assert_eq!(acc.rounds[0].data.len(), 1);   // (d+1)^0 = 1 prefix
    assert_eq!(acc.rounds[1].data.len(), 4);   // (d+1)^1 = 4 prefixes
    assert_eq!(acc.rounds[2].data.len(), 16);  // (d+1)^2 = 16 prefixes

    // Each prefix has d u-values
    assert_eq!(acc.rounds[0].data[0].len(), d);
    assert_eq!(acc.rounds[1].data[0].len(), d);
    assert_eq!(acc.rounds[2].data[0].len(), d);
}

#[test]
fn test_parallel_equals_sequential() {
    // Parallel fold-reduce should match sequential reference computation
    let l0 = 2;
    let d = 3;  // Spartan degree (hardcoded in build_accumulators)
    let l = 6;
    let size = 1 << l;

    let poly_az: Vec<Fr> = (0..size).map(|_| Fr::random(&mut OsRng)).collect();
    let poly_bz: Vec<Fr> = (0..size).map(|_| Fr::random(&mut OsRng)).collect();
    let poly_cz: Vec<Fr> = (0..size).map(|_| Fr::random(&mut OsRng)).collect();
    let taus: Vec<Fr> = (0..l).map(|_| Fr::random(&mut OsRng)).collect();

    let acc = build_accumulators(&poly_az, &poly_bz, &poly_cz, &taus, l0);
    let acc_reference = build_accumulators_reference(&poly_az, &poly_bz, &poly_cz, &taus, l0);

    for round in 0..l0 {
        for v_idx in 0..acc.rounds[round].num_prefixes() {
            for u_idx in 0..d {
                assert_eq!(
                    acc.get(round, v_idx, u_idx),
                    acc_reference.get(round, v_idx, u_idx)
                );
            }
        }
    }
}
```

| Test | What it verifies |
|------|------------------|
| `test_accumulator_dimensions` | Correct sizes for each round |
| `test_parallel_equals_sequential` | Parallelization is correct |

---

### End-to-End Integration Tests for Procedure 9

These tests verify Procedure 9 produces correct accumulators by comparing against a naive reference implementation.

#### Naive Reference Implementation

```rust
/// Naive O(2^ℓ) reference implementation - directly computes the accumulator formula
/// Used only for testing, NOT for production
fn naive_compute_accumulators<S: PrimeField>(
    poly_evals: &[Vec<S>],  // d polynomials, each with 2^ℓ evaluations over {0,1}^ℓ
    taus: &[S],             // τ vector of length ℓ
    l0: usize,
    l: usize,
) -> SmallValueAccumulators<S> {
    let d = poly_evals.len();
    let half = l / 2;
    let mut accumulators = SmallValueAccumulators::new(l0, d);

    // Precompute E_in table (same as Procedure 9)
    let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + half]);

    // For each accumulator A_i(v, u), directly compute the sum from the formula
    for i in 0..l0 {
        let num_prefixes = pow(d + 1, i);

        for v_idx in 0..num_prefixes {
            let v = index_to_ud_tuple(v_idx, d + 1, i);

            for u_idx in 0..d {
                let u = ud_hat_index_to_point(u_idx);

                // Sum over y ∈ {0,1}^(ℓ₀-i-1) and x_out ∈ {0,1}^(ℓ/2-ℓ₀)
                let mut acc_sum = S::ZERO;

                let y_len = l0 - i - 1;
                let x_out_len = half - l0;

                for y in 0..(1usize << y_len) {
                    for x_out in 0..(1usize << x_out_len) {
                        // Compute E_out,i[y, x_out]
                        let e_out = compute_e_out_at_point(i, y, x_out, taus, l0, half, l);

                        // Inner sum over x_in ∈ {0,1}^(ℓ/2)
                        let mut inner_sum = S::ZERO;
                        for x_in in 0..(1usize << half) {
                            // Compute ∏_k p_k(v, u, y, x_in, x_out) via extension
                            let product = compute_extended_product_at_point(
                                &v, u, y, x_in, x_out, poly_evals, l0, half, l
                            );
                            inner_sum += e_in[x_in] * product;
                        }

                        acc_sum += e_out * inner_sum;
                    }
                }

                accumulators.accumulate(i, v_idx, u_idx, acc_sum);
            }
        }
    }

    accumulators
}

/// Compute E_out,i at a specific point (y, x_out)
/// E_out,i uses τ[(i+1):ℓ₀] concatenated with τ[(ℓ/2+ℓ₀):]
fn compute_e_out_at_point<S: PrimeField>(
    i: usize,
    y: usize,
    x_out: usize,
    taus: &[S],
    l0: usize,
    half: usize,
    l: usize,
) -> S {
    let mut result = S::ONE;

    // τ[(i+1):ℓ₀] paired with y bits
    let y_len = l0 - i - 1;
    for j in 0..y_len {
        let tau_j = taus[i + 1 + j];
        let bit = ((y >> j) & 1) == 1;
        result *= if bit { tau_j } else { S::ONE - tau_j };
    }

    // τ[(ℓ/2+ℓ₀):] paired with x_out bits
    let x_out_len = half - l0;
    for j in 0..x_out_len {
        let tau_j = taus[half + l0 + j];
        let bit = ((x_out >> j) & 1) == 1;
        result *= if bit { tau_j } else { S::ONE - tau_j };
    }

    result
}

/// Compute ∏_k p_k(v, u, y, x_in, x_out) using multilinear extension
/// The point (v, u, y) is in U_d^ℓ₀, and (x_in, x_out) is in {0,1}^(ℓ-ℓ₀)
fn compute_extended_product_at_point<S: PrimeField>(
    v: &[usize],      // prefix in U_d^(i-1), using U_d point values
    u: usize,         // value in Û_d, using U_d point value
    y: usize,         // suffix bits in {0,1}^(ℓ₀-i-1)
    x_in: usize,      // inner bits in {0,1}^(ℓ/2)
    x_out: usize,     // outer bits in {0,1}^(ℓ/2-ℓ₀)
    poly_evals: &[Vec<S>],
    l0: usize,
    half: usize,
    l: usize,
) -> S {
    let d = poly_evals.len();
    let mut product = S::ONE;

    for k in 0..d {
        // Build the full ℓ₀-length prefix: (v, u, y)
        // v has length i-1, u is single value, y has length ℓ₀-i-1
        // Total: (i-1) + 1 + (ℓ₀-i-1) = ℓ₀ - 1... wait, that's wrong
        // Actually: v has length i, u is position i+1, y fills the rest
        // Let me reconsider: the paper says β = (v, u, y) where
        // v ∈ U_d^{i-1}, u ∈ Û_d, y ∈ {0,1}^{ℓ₀-i}
        // So total length = (i-1) + 1 + (ℓ₀-i) = ℓ₀

        // Evaluate p_k at the extended point using multilinear extension formula
        let eval = evaluate_multilinear_extension_at_point(
            &poly_evals[k], v, u, y, x_in, x_out, l0, half, l
        );
        product *= eval;
    }

    product
}

/// Evaluate multilinear polynomial at a point where first ℓ₀ coords may be non-binary
fn evaluate_multilinear_extension_at_point<S: PrimeField>(
    evals: &[S],      // 2^ℓ evaluations over {0,1}^ℓ
    v: &[usize],      // U_d point values for first |v| variables
    u: usize,         // U_d point value for next variable
    y: usize,         // binary bits for next (ℓ₀ - |v| - 1) variables
    x_in: usize,      // binary bits for next ℓ/2 variables
    x_out: usize,     // binary bits for remaining variables
    l0: usize,
    half: usize,
    l: usize,
) -> S {
    // Use multilinear extension formula:
    // p(r) = Σ_{b ∈ {0,1}^ℓ} p(b) · eq(r, b)
    // where eq(r, b) = ∏_j (r_j · b_j + (1-r_j)(1-b_j))

    let mut result = S::ZERO;
    let suffix_len = l - l0;
    let y_len = l0 - v.len() - 1;

    // Sum over all binary points b ∈ {0,1}^ℓ
    for b in 0..(1usize << l) {
        let p_b = evals[b];

        // Compute eq(point, b) where point = (v, u, y_bits, x_in_bits, x_out_bits)
        let mut eq_val = S::ONE;
        let mut bit_idx = 0;

        // First |v| coordinates: v[j] paired with b[j]
        for &v_j in v.iter() {
            let b_j = ((b >> bit_idx) & 1) == 1;
            eq_val *= eq_factor(ud_point_to_field(v_j), b_j);
            bit_idx += 1;
        }

        // Next coordinate: u paired with b[bit_idx]
        let b_u = ((b >> bit_idx) & 1) == 1;
        eq_val *= eq_factor(ud_point_to_field(u), b_u);
        bit_idx += 1;

        // Next y_len coordinates: y bits paired with b bits
        for j in 0..y_len {
            let y_j = ((y >> j) & 1) == 1;
            let b_j = ((b >> bit_idx) & 1) == 1;
            eq_val *= eq_factor_binary(y_j, b_j);
            bit_idx += 1;
        }

        // Next half coordinates: x_in bits paired with b bits
        for j in 0..half {
            let x_j = ((x_in >> j) & 1) == 1;
            let b_j = ((b >> bit_idx) & 1) == 1;
            eq_val *= eq_factor_binary(x_j, b_j);
            bit_idx += 1;
        }

        // Remaining coordinates: x_out bits paired with b bits
        let x_out_len = l - l0 - half;
        for j in 0..x_out_len {
            let x_j = ((x_out >> j) & 1) == 1;
            let b_j = ((b >> bit_idx) & 1) == 1;
            eq_val *= eq_factor_binary(x_j, b_j);
            bit_idx += 1;
        }

        result += p_b * eq_val;
    }

    result
}

/// eq(r, b) factor for a single coordinate where r may be non-binary
fn eq_factor<S: PrimeField>(r: S, b: bool) -> S {
    if b { r } else { S::ONE - r }
}

/// eq(a, b) factor for binary inputs
fn eq_factor_binary<S: PrimeField>(a: bool, b: bool) -> S {
    if a == b { S::ONE } else { S::ZERO }
}

/// Convert U_d point value to field element
/// ∞ → handled specially (leading coefficient), 0,1,2,... → field elements
fn ud_point_to_field<S: PrimeField>(point: usize) -> S {
    if point == INFINITY {
        // For ∞, the eq factor represents leading coefficient extraction
        // This is handled in the extension logic, not here directly
        panic!("∞ point should be handled via extension, not direct eq evaluation");
    }
    S::from(point as u64)
}
```

#### Test 1: Procedure 9 Matches Naive Implementation

```rust
#[test]
fn test_procedure9_matches_naive() {
    // Test with various sizes
    for (l, l0) in [(8, 2), (10, 3), (12, 3)] {
        let d = 2;

        // Random small-value polynomial evaluations (simulating R1CS trace)
        let mut rng = OsRng;
        let poly_evals: Vec<Vec<Fr>> = (0..d)
            .map(|_| (0..(1 << l))
                .map(|_| Fr::from(rng.next_u32() as u64))
                .collect())
            .collect();

        // Random τ challenges
        let taus: Vec<Fr> = (0..l)
            .map(|_| Fr::random(&mut rng))
            .collect();

        // Compute via Procedure 9 (optimized)
        let actual = procedure_9(&poly_evals, &taus, l0, d);

        // Compute via naive reference
        let expected = naive_compute_accumulators(&poly_evals, &taus, l0, l);

        // Compare all accumulators
        for i in 0..l0 {
            for v_idx in 0..pow(d + 1, i) {
                for u_idx in 0..d {
                    assert_eq!(
                        actual.get(i, v_idx, u_idx),
                        expected.get(i, v_idx, u_idx),
                        "Mismatch: l={}, l0={}, i={}, v_idx={}, u_idx={}",
                        l, l0, i, v_idx, u_idx
                    );
                }
            }
        }
    }
}
```

#### Test 2: Parallel vs Sequential

```rust
#[test]
fn test_procedure9_parallel_matches_sequential() {
    let l = 12;
    let l0 = 3;
    let d = 2;

    let mut rng = OsRng;
    let poly_evals: Vec<Vec<Fr>> = (0..d)
        .map(|_| (0..(1 << l))
            .map(|_| Fr::from(rng.next_u32() as u64))
            .collect())
        .collect();

    let taus: Vec<Fr> = (0..l)
        .map(|_| Fr::random(&mut rng))
        .collect();

    // Both should produce identical results
    let sequential = procedure_9_sequential(&poly_evals, &taus, l0, d);
    let parallel = procedure_9_parallel(&poly_evals, &taus, l0, d);

    for i in 0..l0 {
        for v_idx in 0..pow(d + 1, i) {
            for u_idx in 0..d {
                assert_eq!(
                    sequential.get(i, v_idx, u_idx),
                    parallel.get(i, v_idx, u_idx),
                    "Parallel/sequential mismatch at i={}, v_idx={}, u_idx={}",
                    i, v_idx, u_idx
                );
            }
        }
    }
}
```

#### Test 3: Accumulators Produce Correct Round Polynomials

This verifies the accumulators work correctly when used in the sumcheck protocol:

```rust
#[test]
fn test_accumulators_produce_correct_round_polynomials() {
    let l = 10;
    let l0 = 3;
    let d = 2;

    let mut rng = OsRng;
    let poly_evals: Vec<Vec<Fr>> = (0..d)
        .map(|_| (0..(1 << l))
            .map(|_| Fr::from(rng.next_u32() as u64))
            .collect())
        .collect();

    let taus: Vec<Fr> = (0..l)
        .map(|_| Fr::random(&mut rng))
        .collect();

    // Build accumulators via Procedure 9
    let accumulators = procedure_9(&poly_evals, &taus, l0, d);

    // Verify each round polynomial matches direct computation
    let mut lagrange_coeffs = LagrangeBasisCoeffs::new(d);
    let mut challenges: Vec<Fr> = Vec::new();

    for round in 0..l0 {
        // Compute t_i(u) using accumulators + Lagrange coefficients
        // t_i(u) = Σ_v R_i[v] · A_i(v, u)
        let t_from_acc: Vec<Fr> = (0..d)
            .map(|u_idx| {
                let mut sum = Fr::ZERO;
                for v_idx in 0..pow(d + 1, round) {
                    sum += lagrange_coeffs.get(v_idx) * accumulators.get(round, v_idx, u_idx);
                }
                sum
            })
            .collect();

        // Compute t_i(u) directly via naive O(2^(ℓ-i)) summation
        let t_direct = naive_compute_round_poly_t(
            round, &poly_evals, &taus, &challenges, l0, l
        );

        // Compare at all evaluation points in Û_d
        for u_idx in 0..d {
            assert_eq!(
                t_from_acc[u_idx],
                t_direct[u_idx],
                "Round {} mismatch at u_idx={}", round, u_idx
            );
        }

        // Simulate verifier challenge for next round
        let r = Fr::random(&mut rng);
        challenges.push(r);
        lagrange_coeffs.extend(&r);
    }
}

/// Directly compute t_i(u) for all u ∈ Û_d at round i
fn naive_compute_round_poly_t<S: PrimeField>(
    round: usize,
    poly_evals: &[Vec<S>],
    taus: &[S],
    challenges: &[S],  // r_1, ..., r_{i-1}
    l0: usize,
    l: usize,
) -> Vec<S> {
    let d = poly_evals.len();
    let half = l / 2;

    // t_i(u) = Σ_{x' ∈ {0,1}^{ℓ-i}} E_out(x') · E_in(x'_in) · ∏_k p_k(r_{<i}, u, x')
    // where x' = (y, x_in, x_out) with appropriate splits

    (0..d).map(|u_idx| {
        let u = ud_hat_index_to_point(u_idx);

        let mut sum = S::ZERO;
        // Sum over all x' ∈ {0,1}^{ℓ-round-1}
        for x_prime in 0..(1usize << (l - round - 1)) {
            // Compute eq factors and product term
            // ... (similar structure to naive_compute_accumulators)
            let contribution = compute_round_term(
                round, u, x_prime, poly_evals, taus, challenges, l0, half, l
            );
            sum += contribution;
        }
        sum
    }).collect()
}
```

#### Test Summary

| Test | What it verifies |
|------|------------------|
| `test_procedure9_matches_naive` | Accumulators match direct formula computation |
| `test_procedure9_parallel_matches_sequential` | Parallel fold-reduce produces identical results |
| `test_accumulators_produce_correct_round_polynomials` | Accumulators work correctly in sumcheck context |

---

## Summary

### Phase 1: Procedure 9 Components

| Component | File | Notes |
|-----------|------|-------|
| `UdPoint` | `src/lagrange.rs` | Domain point enum for U_d = {∞, 0, 1, ..., d-1} |
| `UdHatPoint` | `src/lagrange.rs` | Domain point enum for Û_d = U_d \ {1} |
| `UdTuple` | `src/lagrange.rs` | Tuple β ∈ U_d^k |
| `ValueOneExcluded` | `src/lagrange.rs` | Error type for Û_d conversion |
| `LagrangeEvals` | `src/lagrange.rs` | Extended domain evaluations |
| `extend_one_variable` | `src/lagrange.rs` | Procedure 5 |
| `extend_to_lagrange_domain` | `src/lagrange.rs` | Procedure 6 |
| `lagrange_basis_evals` | `src/lagrange.rs` | For R_i update (Phase 2) |
| `compute_suffix_eq_pyramid` | `src/polys/eq.rs` | E_y precomputation |
| `RoundAccumulator` | `src/accumulators.rs` | Single round's A_i(v, u) matrix |
| `SmallValueAccumulators` | `src/accumulators.rs` | Collection of all round accumulators |
| `AccumulatorPrefixIndex` | `src/accumulators.rs` | Struct for idx4 output (uses `u: UdHatPoint`) |
| `compute_idx4` | `src/accumulators.rs` | Accumulator index mapping |
| `gather_prefix_evals` | `src/accumulators.rs` | Gather p(b, x_in, x_out) for b ∈ {0,1}^ℓ₀ |
| `build_accumulators` | `src/accumulators.rs` | Main Procedure 9 (Rayon fold-reduce) |

### Phase 2: Algorithm 6 Components (Later)

| Component | File | Notes |
|-----------|------|-------|
| `LagrangeBasisCoeffs` | `src/sumcheck.rs` | R_i tensor (⊗ of Lagrange basis evals) |
| `prove_cubic_with_three_inputs_alg6` | `src/sumcheck.rs` | Full Algorithm 6 prover |

### Reused Components

| Component | File | Notes |
|-----------|------|-------|
| `EqPolynomial::evals_from_points` | `src/polys/eq.rs` | For E_in, E_out tables |
| `EqSumCheckInstance` | `src/sumcheck.rs` | For Phase 3 (rounds ℓ₀+2 onwards) |
