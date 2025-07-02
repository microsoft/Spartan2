# Arkworks Integration

This directory contains the arkworks integration for Spartan2, which implements the standard `ark-snark` `SNARK` trait to enable compatibility with the broader arkworks ecosystem.

## Overview

The arkworks integration provides:

- **`SpartanSNARK`**: A wrapper that implements the `ark-snark::SNARK` trait
- **Wrapper Types**: `ArkworksProvingKey`, `ArkworksVerifyingKey`, and `ArkworksProof` that wrap Spartan's native types
- **`FieldBridge` Trait**: A trait for converting between arkworks and Spartan field types
- **Error Handling**: `ArkworksError` that bridges between error systems

## Current Status

This is an initial implementation that establishes the interface and structure. The main components are:

### ✅ Completed
- Basic arkworks SNARK trait implementation structure
- Wrapper types for keys and proofs  
- Error type bridging
- Feature flag (`arkworks`) for optional dependency
- Tests demonstrating the interface

### 🚧 In Progress / TODO
- **Circuit Conversion**: Bridge between `ark_relations::r1cs::ConstraintSynthesizer` and `bellpepper_core::Circuit`
- **Field Bridging**: Implement `FieldBridge` for specific field types that exist in both ecosystems
- **Serialization**: Complete arkworks-compatible serialization for wrapper types
- **Full Implementation**: Replace placeholder methods with actual conversions

## Usage

Enable the arkworks feature in your `Cargo.toml`:

```toml
spartan2 = { version = "0.1.0", features = ["arkworks"] }
```

Then use the arkworks SNARK trait:

```rust
use spartan2::arkworks::snark::SpartanSNARK;
use ark_snark::SNARK;

// Once fully implemented:
// type MySNARK = SpartanSNARK<PallasIPAEngine, MyField>;
// let (pk, vk) = MySNARK::circuit_specific_setup(circuit, &mut rng)?;
```

## Architecture

The integration works by:

1. **Trait Implementation**: `SpartanSNARK` implements `ark-snark::SNARK`
2. **Type Wrapping**: Arkworks-compatible wrappers around Spartan types
3. **Circuit Adaptation**: (TODO) Convert between constraint system representations
4. **Field Bridging**: (TODO) Convert between field type representations

## Next Steps

1. **Implement Field Bridging**: Create `FieldBridge` implementations for common field types
2. **Circuit System Bridge**: Complete the constraint system conversion
3. **Real-world Testing**: Test with actual arkworks circuits and field types
4. **Performance Optimization**: Optimize conversions for minimal overhead

This enables Spartan to be used in arkworks-based projects like [Sonobe](https://github.com/privacy-scaling-explorations/sonobe/) and other zkSNARK frameworks that expect the standard arkworks interface.