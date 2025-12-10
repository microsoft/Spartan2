# Hyrax vs Dory PCS Benchmark (BLS12-381)

Benchmark results comparing Hyrax-PC and Dory-PC implementations on BLS12-381 curve.

## Circuit
- **Type**: Cubic constraint (x^3 + x + 5 = y)
- **Input**: x = 2
- **Output**: y = 15

## Results

| Phase | Hyrax | Dory | Ratio |
|-------|-------|------|-------|
| **Setup** | 104.58 ms | 376.75 µs | Dory **278x faster** |
| **Prove** | 32.12 ms | 502.53 ms | Hyrax **15.6x faster** |
| **Verify** | 30.77 ms | 254.45 ms | Hyrax **8.3x faster** |
| **Full Pipeline** | 203.21 ms | 818.19 ms | Hyrax **4x faster** |

## Analysis

### Dory Advantages
- **O(1) Setup**: 278x faster setup time (microseconds vs milliseconds)
- **Constant commitment size**: Independent of polynomial degree
- **Better scaling**: Logarithmic verification for large circuits

### Hyrax Advantages
- **Faster Prove**: 15.6x faster for small circuits
- **Faster Verify**: 8.3x faster for small circuits
- **No pairing operations**: Uses only group operations

## When to Use Each

| Use Case | Recommended PCS |
|----------|-----------------|
| Small circuits (< 1K constraints) | Hyrax |
| Large circuits (> 10K constraints) | Dory |
| Frequent setup changes | Dory |
| Verification-critical applications | Hyrax (small) / Dory (large) |

## Validation
Both implementations passed:
- Correctness check (public output = 15)
- Soundness check (tampered proofs rejected)

## Reproducibility
```bash
cd external/Spartan2
cargo bench --bench pcs_comparison --features dory
```

---
*Generated: 2024-12-10*
*Platform: Linux x86_64*
*Rust: 1.90.0*

