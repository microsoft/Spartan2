# Spartan: High-speed zero-knowledge SNARKs without trusted setup
Spartan is a sum-check-based zkSNARK with an extremely efficient prover (a zkSNARK is type cryptographic proof system that enables a prover to prove a mathematical statement to a verifier with a short proof and succinct verification, and without revealing anything beyond the validity of the statement). Spartan also features several unique properties that are particularly relevant for applications where zero-knowledge is essential. Here are some highlights:

* Spartan provides a linear-time polynomial IOP that when combined with a polynomial commitment scheme provides a succinct interactive argument. It is made non-interactive using the Fiat-Shamir transform.

* Spartan can be instantiated with any multilinear polynomial commitment scheme (e.g., Binius, HyperKZG, Samaritan, WHIR, Mercury, BaseFold, PST13, Dory, Hyrax). Depending on the polynomial commitment scheme used, one can achieve different properties (e.g., small fields or big fields, post-quantum or pre-quantum security, transparent or universal setup, hash-based or curve-based, binary fields or prime fields).

* Spartan is flexible with respect to arithmetization: it can support R1CS, Plonkish, AIR, and their generalization CCS. Spartan protocol itself internally uses lookup arguments, so one can additionally prove lookup constraints with Spartan.

* The prover's work naturally splits into a witness-dependent part and a witness-independent part (a significant chunk, up to 90%, of the prover's work is incurred in the witness-independent part). The latter part can be offloaded to any untrusted entity without violating zero-knowledge. Note that such a clean decomposition between witness-dependent part and witness-independent part is not featured by other popular zkSNARKs (e.g., Plonk, HyperPlonk, Honk).

* The witness-dependent work of the Spartan prover is shown to be MPC-friendly by more recent works, allowing the whole Spartan prover to be delegated.

* For uniform constraint systems, Spartan's prover can be optimized further by eliminating the witness-independent work of the prover, which constitutes about 90% of the prover's work.

## About this library
Compared to an earlier implementation of [Spartan](https://github.com/Microsoft/Spartan), this project provides an implementation of Spartan that is generic over the polynomial commitment scheme. This version also accepts circuits expressed with bellpepper, which supports R1CS. In the future, we plan to add support for other circuit formats (e.g., Plonkish, CCS). The first version of this code is derived from Nova's open-source code.

The proofs are *not* zero-knowledge (we plan to add it in the near future). Also, the current implementation does not implement the Spark protocol, so the verifier's work is proportional to the number of non-zero entries in the R1CS matrices.

### Supported polynomial commitment schemes
- [ ] Elliptic-curve based schemes
  - [x] Bulletproofs-based PCS
  - [x] Hyrax PCS
  - [ ] Dory
  - [ ] Sona
  - [ ] HyperKZG (requires a universal trusted setup)
  - [ ] Mercury / Samaritan (require a universal trusted setup)
- [ ] Hash-based schemes
  - [ ] Basefold
  - [ ] WHIR
  - [ ] Brakedown
  - [ ] Binius
  - [ ] Ligero
- [ ] Lattice-based schemes
  - [ ] Greyhound

## References
The following paper, which appeared at CRYPTO 2020, provides details of the Spartan proof system:

[Spartan: Efficient and general-purpose zkSNARKs without trusted setup](https://eprint.iacr.org/2019/550) \
Srinath Setty \
IACR CRYPTO 2020

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