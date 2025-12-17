# Add BLS12-381 Curve Provider and Dory-PC Backend

## Summary

This PR adds support for:
1. **BLS12-381 curve** as a new curve provider
2. **Dory-PC** as an alternative polynomial commitment scheme

## Motivation

BLS12-381 is widely used in production systems (Ethereum 2.0, Zcash, etc.) and offers strong security guarantees. Dory-PC provides O(1) commitment size and O(log n) verification, making it suitable for large circuits.

## Changes

### New Files
- `src/provider/bls12_381.rs` - BLS12-381 curve implementation via `halo2curves`
- `src/provider/pcs/dory_pc.rs` - Dory-PC adapter wrapping `quarks-zk` crate
- `benches/pcs_comparison.rs` - Benchmark comparing Hyrax vs Dory
- `examples/BENCHMARK_RESULTS.md` - Benchmark results

### Modified Files
- `src/provider/mod.rs` - Register `BLS12381HyraxEngine` and `BLS12381DoryEngine`
- `src/provider/pcs/mod.rs` - Export `dory_pc` module
- `Cargo.toml` - Add dependencies (`quarks-zk`, `ark-*` crates)

## New Engines

```rust
// BLS12-381 with Hyrax-PC (existing PCS)
pub struct BLS12381HyraxEngine;

// BLS12-381 with Dory-PC (new PCS)
pub struct BLS12381DoryEngine;
```

## Benchmark Results

| Phase | Hyrax | Dory | Notes |
|-------|-------|------|-------|
| Setup | 104.58 ms | 376.75 µs | Dory 278x faster |
| Prove | 32.12 ms | 502.53 ms | Hyrax 15.6x faster |
| Verify | 30.77 ms | 254.45 ms | Hyrax 8.3x faster |

**Conclusion**: Dory excels at setup (O(1)), Hyrax is faster for small circuits. Dory scales better for large circuits.

## Testing

- 16 new tests for BLS12-381 provider (group properties, MSM, transcript)
- 11 new tests for Dory-PC adapter (correctness, soundness, binding)
- Benchmark validates both implementations before measuring

```bash
# Run tests
cargo test --features dory

# Run benchmarks
cargo bench --bench pcs_comparison --features dory
```

## Dependencies

New optional dependencies (behind `dory` feature):
- `quarks-zk = "0.1.1"` - Dory-PC implementation
- `ark-serialize`, `ark-ff`, `ark-bls12-381` - Serialization support

## Breaking Changes

None. This is purely additive.


