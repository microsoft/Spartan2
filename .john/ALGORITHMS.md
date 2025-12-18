# Spartan2 Algorithm Documentation

This document describes the core algorithms implemented in Spartan2, including mathematical expressions and corresponding code references.

---

## Table of Contents
1. [Multilinear Extensions & Polynomial Representations](#1-multilinear-extensions--polynomial-representations)
2. [Sumcheck Protocol](#2-sumcheck-protocol)
3. [Spartan SNARK](#3-spartan-snark)
4. [NeutronNova Folding](#4-neutronnova-folding)

---

## 1. Multilinear Extensions & Polynomial Representations

**Source**: `src/polys/` directory

### 1.1 MultilinearPolynomial - Dense Representation

A multilinear polynomial $\tilde{Z}(x_1, \ldots, x_m)$ is one where each variable has degree at most 1. The multilinear extension (MLE) of a function $Z: \{0,1\}^m \to \mathbb{F}$ is:

$$\tilde{Z}(x_1, \ldots, x_m) = \sum_{e \in \{0,1\}^m} Z(e) \cdot \prod_{i=1}^{m} \left( x_i \cdot e_i + (1 - x_i) \cdot (1 - e_i) \right)$$

**Struct** (`src/polys/multilinear.rs`):
```rust
pub struct MultilinearPolynomial<Scalar: PrimeField> {
  pub(crate) Z: Vec<Scalar>,  // evaluations over {0,1}^m
}
```

**Storage**: Evaluations are stored in lexicographic order of the binary representation:
- Index 0 → evaluation at $(0, 0, \ldots, 0)$
- Index 1 → evaluation at $(0, 0, \ldots, 1)$
- Index $2^m - 1$ → evaluation at $(1, 1, \ldots, 1)$

### 1.2 Variable Binding: `bind_poly_var_top`

Binding the top variable $x_1$ to a challenge $r$ reduces an $m$-variate polynomial to an $(m-1)$-variate one:

$$\tilde{Z}(r, x_2, \ldots, x_m) = (1 - r) \cdot \tilde{Z}(0, x_2, \ldots, x_m) + r \cdot \tilde{Z}(1, x_2, \ldots, x_m)$$

**Code** (`src/polys/multilinear.rs`):
```rust
pub fn bind_poly_var_top(&mut self, r: &Scalar) {
  let n = self.Z.len() / 2;
  let (left, right) = self.Z.split_at_mut(n);

  // left[i] = left[i] + r * (right[i] - left[i])
  //         = (1-r) * left[i] + r * right[i]
  zip_with_for_each!((left.par_iter_mut(), right.par_iter()), |a, b| {
    *a += *r * (*b - *a);
  });

  self.Z.truncate(n);  // reduce from 2^m to 2^{m-1}
}
```

### 1.3 EqPolynomial - Equality Polynomial

The equality polynomial evaluates to 1 when inputs match, 0 otherwise:

$$eq(\tau, x) = \prod_{i=1}^{m} \left( \tau_i \cdot x_i + (1 - \tau_i) \cdot (1 - x_i) \right)$$

**Key Method**: `evals_from_points` computes all $2^m$ evaluations in $O(2^m)$ time using a doubling algorithm:

```rust
pub fn evals_from_points(r: &[Scalar]) -> Vec<Scalar> {
  let mut evals: Vec<Scalar> = vec![Scalar::ZERO; 2_usize.pow(r.len() as u32)];
  let mut size = 1;
  evals[0] = Scalar::ONE;

  for r in r.iter().rev() {
    let (evals_left, evals_right) = evals.split_at_mut(size);
    zip_with_for_each!(par_iter_mut, (evals_left, evals_right), |x, y| {
      *y = *x * r;
      *x -= &*y;  // x = x * (1 - r)
    });
    size *= 2;
  }
  evals
}
```

### 1.4 PowPolynomial - Power Polynomial

Used in NeutronNova for batching constraints:

$$pow(\tau, x) = \prod_{i=1}^{\l} \left( 1 + (\tau^{2^i} - 1) \cdot x_i \right)$$

**Struct**: Stores $[\tau, \tau^2, \tau^4, \ldots, \tau^{2^{\l-1}}]$ (squares).

### 1.5 Dimension Reduction: Multilinear → Univariate

In each sumcheck round, we reduce an $m$-variate polynomial to a univariate polynomial by fixing one variable.

**Process**:
1. **Split** $2^m$ evaluations into:
   - `low` = indices $[0, 2^{m-1})$: evaluations where top variable = 0
   - `high` = indices $[2^{m-1}, 2^m)$: evaluations where top variable = 1

2. **Compute univariate evaluations**:
   - At $X = 0$: $s(0) = \sum_i g(\text{low}_i)$
   - At $X = 1$: $s(1) = \text{claim} - s(0)$ (from sumcheck constraint)
   - At $X = 2$: Use extrapolation formula

3. **Extrapolation formula** (for linear interpolation in one variable):
   $$\tilde{p}(X) = (1 - X) \cdot \tilde{p}(0) + X \cdot \tilde{p}(1)$$

   At $X = 2$:
   $$\tilde{p}(2) = -\tilde{p}(0) + 2 \cdot \tilde{p}(1)$$

**Code** (`src/sumcheck.rs:158-198`):
```rust
fn compute_eval_points_quad<F>(
  poly_A: &MultilinearPolynomial<E::Scalar>,
  poly_B: &MultilinearPolynomial<E::Scalar>,
  comb_func: &F,
) -> (E::Scalar, E::Scalar) {
  let len = poly_A.Z.len() / 2;
  par_for(len, |i| {
    let a_low = poly_A[i];
    let a_high = poly_A[len + i];

    // eval at 0: use low values directly
    let eval0 = comb_func(&a_low, &b_low);

    // eval at 2: extrapolate using -low + 2*high
    let a_bound = a_high + a_high - a_low;
    let b_bound = b_high + b_high - b_low;
    let eval2 = comb_func(&a_bound, &b_bound);

    (eval0, eval2)
  }, /* reduce */, /* identity */)
}
```

### 1.6 UniPoly / CompressedUniPoly

**Source**: `src/polys/univariate.rs`

**Storage**: Coefficients in little-endian order. For $ax^2 + bx + c$: `coeffs = [c, b, a]`

**Interpolation**: `UniPoly::from_evals` takes evaluations at points 0, 1, 2, ... and uses Gaussian elimination to find coefficients.

#### Compression Scheme

The sumcheck constraint $s(0) + s(1) = \text{claim}$ creates redundancy. For a degree-$d$ polynomial with $d+1$ coefficients, we can recover one coefficient from the others.

**Prover compresses** by omitting the linear term (`src/polys/univariate.rs:100-107`):
```rust
// For ax² + bx + c, store only [c, a] (omit b)
pub fn compress(&self) -> CompressedUniPoly<Scalar> {
  let coeffs_except_linear_term = [&self.coeffs[0..1], &self.coeffs[2..]].concat();
  CompressedUniPoly { coeffs_except_linear_term }
}
```

**Verifier decompresses** using the claim as a hint (`src/polys/univariate.rs:120-133`):
```rust
// Recover linear term from: s(0) + s(1) = hint
// s(0) = c
// s(1) = a + b + c  (sum of all coefficients)
// hint = c + (a + b + c) = 2c + b + a
// Therefore: b = hint - 2c - a
pub fn decompress(&self, hint: &Scalar) -> UniPoly<Scalar> {
  // Start with: linear = hint - 2*constant
  let mut linear_term = *hint - self.coeffs_except_linear_term[0]
                              - self.coeffs_except_linear_term[0];

  // Subtract higher-degree coefficients (they contribute to s(1))
  for i in 1..self.coeffs_except_linear_term.len() {
    linear_term -= self.coeffs_except_linear_term[i];
  }

  // Reconstruct: [constant, linear, quadratic, ...]
  let mut coeffs = vec![self.coeffs_except_linear_term[0]];
  coeffs.push(linear_term);
  coeffs.extend(&self.coeffs_except_linear_term[1..]);
  UniPoly { coeffs }
}
```

**Example** (quadratic $s(X) = 2X^2 + 3X + 5$):
- Full: `coeffs = [5, 3, 2]` (c=5, b=3, a=2)
- Compressed: `[5, 2]` (omit b=3)
- Hint = s(0) + s(1) = 5 + 10 = 15
- Recovery: b = 15 - 2(5) - 2 = 3 ✓

**Savings**: 1 field element per sumcheck round.

---

## 2. Sumcheck Protocol

**Source**: `src/sumcheck.rs`

### 2.1 Mathematical Background

**Goal**: Prove that $\sum_{x \in \{0,1\}^\l} g(x) = C$ without the verifier computing the sum.

**Round polynomial**: In round $i$, the prover sends:
$$s_i(X) = \sum_{x_{i+1}, \ldots, x_\l \in \{0,1\}} g(r_1, \ldots, r_{i-1}, X, x_{i+1}, \ldots, x_\l)$$

**Key constraint**: $s_i(0) + s_i(1) = C_{i-1}$ where $C_0 = C$.

### 2.2 Prover Protocol (`prove_quad`)

**Input**: Claim $C$, multilinear polynomials $A, B$, combination function $g(a, b)$

**For each round $i = 1, \ldots, \l$**:

1. **Compute evaluations** at points 0 and 2:
   $$s_i(0) = \sum_{x_{rest}} g(A(0, x_{rest}), B(0, x_{rest}))$$
   $$s_i(2) = \sum_{x_{rest}} g(A(2, x_{rest}), B(2, x_{rest}))$$

2. **Derive point 1** from constraint:
   $$s_i(1) = C_{i-1} - s_i(0)$$

3. **Interpolate** to get univariate polynomial $s_i(X)$

4. **Send** compressed polynomial to transcript

5. **Receive challenge** $r_i$ from transcript

6. **Bind** polynomials: $A \leftarrow A(r_i, \cdot)$, $B \leftarrow B(r_i, \cdot)$

7. **Update claim**: $C_i = s_i(r_i)$

**Code** (`src/sumcheck.rs:227-261`):
```rust
for round in 0..num_rounds {
  let (eval_point_0, eval_point_2) =
    Self::compute_eval_points_quad(poly_A, poly_B, &comb_func);

  let evals = vec![eval_point_0, claim_per_round - eval_point_0, eval_point_2];
  let poly = UniPoly::from_evals(&evals)?;

  transcript.absorb(b"p", &poly);
  let r_i = transcript.squeeze(b"c")?;

  claim_per_round = poly.evaluate(&r_i);

  rayon::join(
    || poly_A.bind_poly_var_top(&r_i),
    || poly_B.bind_poly_var_top(&r_i),
  );
}
```

### 2.3 Verifier Protocol (`verify`)

**For each round $i = 1, \ldots, \l$**:

1. **Decompress** polynomial using claim as hint
2. **Check degree** bound
3. **Verify** $s_i(0) + s_i(1) = e$ (implicit in decompress)
4. **Squeeze challenge** $r_i$ from transcript
5. **Update evaluation**: $e = s_i(r_i)$

**Code** (`src/sumcheck.rs:112-139`):
```rust
for i in 0..num_rounds {
  let poly = self.compressed_polys[i].decompress(&e);

  if poly.degree() != degree_bound {
    return Err(SpartanError::InvalidSumcheckProof);
  }

  transcript.absorb(b"p", &poly);
  let r_i = transcript.squeeze(b"c")?;

  e = poly.evaluate(&r_i);
}
```

### 2.3.1 Per-Round State and Memory

#### What's Persisted Across Rounds

| Variable | Type | What it stores |
|----------|------|----------------|
| `polys` | `Vec<CompressedUniPoly>` | Compressed round polynomials (proof) |
| `r` | `Vec<Scalar>` | All verifier challenges |
| `claim_per_round` | `Scalar` | Current claim (updated each round) |
| `poly_A, B, C` | `MultilinearPolynomial` | Mutated in-place (halved each round) |
| `eq_instance` | `EqSumCheckInstance` | Eq-poly state (updated via `bound()`) |

#### Memory Usage Per Round

| Data | Size | Stored in proof? |
|------|------|------------------|
| `CompressedUniPoly` | 3 field elements (cubic) | ✅ Yes |
| Challenge `r_i` | 1 field element | ✅ Yes |
| `poly_A`, `poly_B`, `poly_C` | Halved each round | ❌ No (mutated) |
| `eq_instance` tables | Shrinking indices | ❌ No (internal state) |

#### Final Output

```rust
Ok((
  SumcheckProof { compressed_polys: polys },  // All round polynomials
  r,                                           // All challenges [r_0, ..., r_{ℓ-1}]
  vec![poly_A[0], poly_B[0], poly_C[0]],      // Final evaluations
))
```

### 2.4 Equality Polynomial Optimization (Algorithm 5)

**Source**: `src/sumcheck.rs:867-1027` (eq_sumcheck module)

For proving $\sum_x eq(\tau, x) \cdot f(x) = 0$, use square-root decomposition.

**Key insight**: Split $\tau$ into left and right halves:
$$eq(\tau, x) = eq(\tau_L, x_L) \cdot eq(\tau_R, x_R)$$

**Precomputation**: Build $O(2^{m/2})$ sized lookup tables for each half.

**Struct**:
```rust
pub struct EqSumCheckInstance<E: Engine> {
  poly_eq_left: Vec<Vec<E::Scalar>>,   // tables of size [1, 2, 4, ..., 2^{first_half-1}]
  poly_eq_right: Vec<Vec<E::Scalar>>,  // tables of size [1, 2, 4, ..., 2^{second_half}]
  eval_eq_left: E::Scalar,             // running product for bound variables
}
```

#### Building the Eq-Poly Tables

The `compute_eq_polynomials` closure builds tables incrementally using a **doubling algorithm** (`src/sumcheck.rs:898-916`):

```rust
let compute_eq_polynomials = |taus: Vec<&E::Scalar>| -> Vec<Vec<E::Scalar>> {
  let len = taus.len();
  let mut result = Vec::with_capacity(len + 1);

  result.push(vec![E::Scalar::ONE]);  // result[0] = [1]

  for i in 0..len {
    let tau = taus[i];
    let prev = &result[i];             // size 2^i

    let mut v_next = prev.to_vec();    // copy prev
    // Append prev[j] * tau for all j (parallel)
    v_next.par_extend(prev.par_iter().map(|v| *v * tau));
    // Now v_next = [prev[0], ..., prev[2^i-1], prev[0]*τ, ..., prev[2^i-1]*τ]

    let (first, last) = v_next.split_at_mut(prev.len());
    // first[j] = prev[j] * (1 - τ)  (for x_new = 0)
    // last[j]  = prev[j] * τ        (for x_new = 1)
    first.par_iter_mut().zip(last).for_each(|(a, b)| *a -= *b);

    result.push(v_next);  // size 2^{i+1}
  }
  result
};
```

**Result structure**: `result[i][j]` = $eq((\tau_0, \ldots, \tau_{i-1}), \text{binary}(j))$

#### Splitting the Tau Vector

```rust
let l = taus.len();               // total variables
let first_half = l / 2;
let second_half = l - first_half;

let (left_taus, right_taus) = taus.split_at(first_half);

// Left: skip τ_0 (tracked separately in eval_eq_left), then reverse
let left_taus = left_taus.iter().skip(1).rev().collect::<Vec<_>>();
// → [τ_{first_half-1}, τ_{first_half-2}, ..., τ_1]

// Right: just reverse
let right_taus = right_taus.iter().rev().collect::<Vec<_>>();
// → [τ_{l-1}, τ_{l-2}, ..., τ_{first_half}]

let (poly_eq_left, poly_eq_right) = rayon::join(
  || compute_eq_polynomials(left_taus),
  || compute_eq_polynomials(right_taus),
);
```

#### Table Dimensions (Example: $\l = 6$)

```
taus = [τ_0, τ_1, τ_2 | τ_3, τ_4, τ_5]
        └─ left ────┘   └── right ──┘

poly_eq_left (from [τ_2, τ_1], skipping τ_0):
  [0]: [1]                                    // 1 element
  [1]: [eq(τ_2, 0), eq(τ_2, 1)]              // 2 elements
  [2]: [eq((τ_2,τ_1), (0,0)), ..., (1,1)]    // 4 elements

poly_eq_right (from [τ_5, τ_4, τ_3]):
  [0]: [1]                                    // 1 element
  [1]: [eq(τ_5, 0), eq(τ_5, 1)]              // 2 elements
  [2]: [eq((τ_5,τ_4), x) for x in {0,1}²]    // 4 elements
  [3]: [eq((τ_5,τ_4,τ_3), x) for x in {0,1}³] // 8 elements
```

**Why reversed?** As rounds progress, we index into *smaller* tables (the later taus get "consumed" first).

### 2.5 ZK Variants

**`prove_quad_zk`** (`src/sumcheck.rs:595-641`): Interacts with a verifier circuit to commit to polynomial coefficients.

Key difference: Instead of just sending polynomials to transcript, coefficients are exposed to a constraint system for commitment.

---

## 3. Spartan SNARK

**Source**: `src/spartan.rs`

### 3.1 Mathematical Background

**R1CS**: Given matrices $A, B, C$ and witness $z = (W, 1, X)$:
$$(Az) \circ (Bz) = Cz$$

where $\circ$ denotes element-wise (Hadamard) product.

**Spartan reduces R1CS to two sumcheck instances**:
1. **Outer sumcheck**: Proves the R1CS constraint holds
2. **Inner sumcheck**: Proves matrix-vector products are correct

### 3.2 Prover Protocol (`prove`)

#### Phase 1: Setup
```rust
let mut z = [W.W.clone(), vec![E::Scalar::ONE], U.public_values.clone()].concat();
let tau = (0..num_rounds_x).map(|_| transcript.squeeze(b"t")).collect()?;
let (Az, Bz, Cz) = pk.S.multiply_vec(&z)?;
```

#### Phase 2: Outer Sumcheck

**Claim**:
$$\sum_{x \in \{0,1\}^n} eq(\tau, x) \cdot \left( Az(x) \cdot Bz(x) - Cz(x) \right) = 0$$

**Code** (`src/spartan.rs:243-256`):
```rust
let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_three_inputs(
  &E::Scalar::ZERO,  // claim is zero (R1CS satisfiability)
  tau,
  &mut poly_Az,
  &mut poly_Bz,
  &mut poly_Cz,
  &mut transcript,
)?;

let (claim_Az, claim_Bz, claim_Cz) = (claims_outer[0], claims_outer[1], claims_outer[2]);
```

#### Phase 3: Inner Sumcheck

**Combined claim** using random $r$:
$$\text{claim} = Az(r_x) + r \cdot Bz(r_x) + r^2 \cdot Cz(r_x)$$

**Prove**:
$$\sum_{y \in \{0,1\}^m} \left( A(r_x, y) + r \cdot B(r_x, y) + r^2 \cdot C(r_x, y) \right) \cdot z(y) = \text{claim}$$

**Code** (`src/spartan.rs:300-309`):
```rust
let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;

let (sc_proof_inner, r_y, claims_inner) = SumcheckProof::prove_quad(
  &claim_inner_joint,
  num_rounds_y,
  &mut MultilinearPolynomial::new(poly_ABC),
  &mut MultilinearPolynomial::new(poly_z),
  comb_func,  // |a, b| a * b
  &mut transcript,
)?;
```

#### Phase 4: PCS Proof

Prove evaluation of witness polynomial at $r_y$ using polynomial commitment scheme.

### 3.3 Verifier Protocol (`verify`)

1. **Reconstruct** $\tau$ from transcript
2. **Verify outer sumcheck**, check final claim:
   $$eq(\tau, r_x) \cdot (Az(r_x) \cdot Bz(r_x) - Cz(r_x)) = \text{claim}_{outer}$$

3. **Verify inner sumcheck**, check:
   $$(A(r_x, r_y) + r \cdot B(r_x, r_y) + r^2 \cdot C(r_x, r_y)) \cdot z(r_y) = \text{claim}_{inner}$$

4. **Verify PCS** evaluation argument

**Code** (`src/spartan.rs:392-443`):
```rust
// Outer sumcheck verification
let taus_bound_rx = tau.evaluate(&r_x);
let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
if claim_outer_final != claim_outer_final_expected {
  return Err(SpartanError::InvalidSumcheckProof);
}

// Inner sumcheck verification
let (eval_A, eval_B, eval_C) = vk.S.evaluate_with_tables(&T_x, &T_y);
let claim_inner_final_expected = (eval_A + r * eval_B + r * r * eval_C) * eval_Z;
if claim_inner_final != claim_inner_final_expected {
  return Err(SpartanError::InvalidSumcheckProof);
}
```

---

## 4. NeutronNova Folding

**Source**: `src/neutronnova_zk.rs`

### 4.1 Mathematical Background

**Goal**: Batch $N$ R1CS instances into one, then prove with Spartan.

**Multi-folding**: Combine instances using random linear combination:
$$W_{fold} = \sum_{i=0}^{N-1} w_i \cdot W_i$$

where $w_i$ are barycentric weights derived from challenges.

### 4.2 NIFS Folding Protocol (`NeutronNovaNIFS::prove`)

**Input**: $N$ R1CS instances $(U_0, W_0), \ldots, (U_{N-1}, W_{N-1})$

**For each round $t = 0, \ldots, \log_2 N - 1$**:

1. **Pair instances**: Group into pairs $(U_{2i}, W_{2i})$ and $(U_{2i+1}, W_{2i+1})$

2. **Compute cubic polynomial** for each pair:
   $$T_t(X) = eq(\rho, \cdot) \cdot \left( Az_{pair}(X) \cdot Bz_{pair}(X) - Cz_{pair}(X) \right)$$

3. **Send** polynomial to transcript

4. **Receive challenge** $r_t$

5. **Fold pairs**:
   $$U_{fold} = (1 - r_t) \cdot U_{2i} + r_t \cdot U_{2i+1}$$

**Code** (`src/neutronnova_zk.rs:276-369`):
```rust
for t in 0..ell_b {
  let rho_t = rhos[t];

  // Compute polynomial for this folding round
  let (e0, quad_coeff) = Self::prove_helper(/* ... */);
  let poly_t = UniPoly { coeffs: vec![new_d, new_c, new_b, new_a] };

  // Get challenge from verifier circuit
  let chals = SatisfyingAssignment::<E>::process_round(/* ... */)?;
  let r_b = chals[0];

  // Fold: halve the number of instances
  // layer[i] = (1-r_b)*layer[2i] + r_b*layer[2i+1]
}
```

### 4.3 Full Proving Protocol

1. **Rerandomization**: Add fresh blinding to commitments for ZK

2. **Multi-folding NIFS**: Fold $N$ step instances into 1 in $\log N$ rounds

3. **Batched Outer Sumcheck**: Prove step + core circuits together using $\tau^k$ polynomial

4. **Batched Inner Sumcheck**: Prove witness evaluations

5. **Verifier Circuit NIFS**: Fold verifier circuit with random instance for ZK

6. **PCS Proof**: Final evaluation argument

### 4.4 Zero-Knowledge Mechanism

1. **Rerandomization** (`src/neutronnova_zk.rs:607-621`):
   - Fresh blinding factors added to commitments
   - Hides actual witness values

2. **Verifier Circuit** (`src/zk.rs`):
   - Encodes verification checks as R1CS constraints
   - Allows prover to commit to intermediate values

3. **Nova NIFS** (`src/nifs.rs`):
   - Folds verifier circuit with random instance
   - Cross-term $T = Az \circ Bz - u \cdot Cz - E$ committed separately

### 4.5 Verification Protocol

1. **Validate** instances, check shared commitments match
2. **Reconstruct** challenges from transcript
3. **Verify multi-folding**: Recompute folded instance
4. **Verify verifier circuit NIFS**
5. **Verify matrix evaluations** match public values:
   $$\text{public} = [\tau(r_x), X_{step}(r_y), X_{core}(r_y), eq(\rho, r_b), \text{quotients}]$$
6. **Verify PCS** evaluation argument

---

## Complexity Summary

| Algorithm | Prover | Verifier | Proof Size |
|-----------|--------|----------|------------|
| Sumcheck ($\l$ rounds) | $O(2^\l)$ | $O(\l)$ | $O(\l)$ |
| Spartan | $O(\|C\| + \|W\| \log \|W\|)$ | $O(\log \|C\| + \log \|W\|)$ | $O(\log \|C\| + \log \|W\|)$ |
| NeutronNova (N instances) | $O(N \cdot \|C\| + N \log N)$ | $O(\log \|C\| + \log N)$ | $O(\log \|C\| + \log N)$ |

where $\|C\|$ = number of constraints, $\|W\|$ = witness size.
