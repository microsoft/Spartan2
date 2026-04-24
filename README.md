# Spartan and NeutronNova: Fast client-side zero-knowledge proving systems

A client-side zkSNARK library built on the Spartan sum-check proof
system and NeutronNova's folding scheme. Spartan2 powers
[Vega](https://eprint.iacr.org/2025/2094).

## What this library provides

- **Spartan zkSNARK** — a PCS-generic Rust implementation of
  [Spartan](https://eprint.iacr.org/2019/550), a sum-check-based zkSNARK
  with a linear-time prover. Accepts R1CS circuits written with
  [bellpepper](https://github.com/lurk-lab/bellpepper). Spartan is
  PCS-agnostic and works with any multilinear polynomial commitment
  scheme (Hyrax, HyperKZG, Binius, WHIR, BaseFold, Dory, PST13, …); the
  choice determines field size, security model (pre- or post-quantum),
  setup assumptions (transparent or universal), and commitment style
  (hash- or curve-based). While Spartan supports R1CS, Plonkish, AIR,
  and CCS in principle (with lookup constraints fitting natively via
  Spartan's internal lookup arguments), this library currently exposes
  the R1CS frontend. Zero-knowledge is obtained via Nova's folding
  scheme. The Spark optimization is not implemented, so verifier work
  is proportional to the number of non-zero R1CS entries.

- **NeutronNova zkSNARK** — a non-recursive implementation of
  [NeutronNova](https://eprint.iacr.org/2024/1606) folding for uniform
  computations: given many instances of a single **step circuit**, all
  R1CS instances are multi-folded into one and the folded instance is
  proved with Spartan, amortizing the prover across the batch. An
  optional **core circuit** can tie the batch together (e.g., to
  enforce cross-step consistency).

- **Precomputable / online witness split.** Both protocols expose
  `setup` → `prep_prove` → `prove`. `setup` produces circuit-shape key
  material. `prep_prove` processes the *precomputable* witness — the
  portion known ahead of proving time — synthesizing and committing to
  it; for NeutronNova it also caches the per-step matrix-vector
  products (`Az`, `Bz`, `Cz`). `prove` consumes fresh *online* witness
  data (challenges, rest-witness, fresh randomness), runs NeutronNova's
  multi-folding rounds where applicable, and produces the final proof.
  The `prep_prove` state can be reused across multiple `prove` calls,
  so amortizable work is paid once. This is the pattern
  [Vega](https://eprint.iacr.org/2025/2094) relies on for low-latency
  proving.

- **Criterion benchmarks** — `benches/sha256_spartan.rs` and
  `benches/sha256_neutronnova.rs` measure setup, prep_prove, prove, and
  verify across message sizes and thread counts, and report proof
  sizes.

## Running benchmarks

The `benches/` directory contains SHA-256 benchmarks for both protocols using [Criterion](https://github.com/bheisler/criterion.rs). Each benchmark measures setup, prep_prove, prove, and verify times across multiple iterations and thread counts, and reports proof sizes.

```bash
# Spartan: SHA-256 over 1 KiB and 2 KiB messages
RUSTFLAGS="-C target-cpu=native" cargo bench --bench sha256_spartan

# NeutronNova: 32 SHA-256 step circuits (2048 bytes total)
RUSTFLAGS="-C target-cpu=native" cargo bench --bench sha256_neutronnova
```

Override thread counts with `BENCH_THREADS` (comma-separated):

```bash
BENCH_THREADS=1,8 RUSTFLAGS="-C target-cpu=native" cargo bench --bench sha256_spartan
```

## References

[Spartan: Efficient and general-purpose zkSNARKs without trusted setup](https://eprint.iacr.org/2019/550) \
Srinath Setty \
CRYPTO 2020

[NeutronNova: Folding everything that reduces to zero-check](https://eprint.iacr.org/2024/1606) \
Abhiram Kothapalli, Srinath Setty \
IACR ePrint 2024/1606

[Vega: Low-latency zero-knowledge proofs over existing credentials](https://eprint.iacr.org/2025/2094) \
Darya Kaviani, Srinath Setty \
IEEE S&P 2026

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.