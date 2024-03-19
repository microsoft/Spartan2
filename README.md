# Spartan2: High-speed zero-knowledge SNARKs.

Spartan is a high-speed zkSNARK, where a zkSNARK is type cryptographic proof system that enables a prover to prove a mathematical statement to a verifier with a short proof and succinct verification, and without revealing anything beyond the validity of the statement. Spartan provides a linear-time polynomial IOP that when combined with a polynomial commitment scheme provides a succinct interactive argument. It is made non-interactive using the Fiat-Shamir transform.

Compared to an earlier implementation of Spartan, the goal of this project is to provide an implementation of Spartan that is generic over the polynomial commitment scheme. This version also accepts circuits expressed with bellperson. We currently support R1CS. In the future, we plan to add support for other circuit formats (e.g., Plonkish, CCS). The proofs are *not* zero-knowledge (we plan to add it in the near future). The first version of this code is derived from Nova's open-source code.

## References
The following paper, which appeared at CRYPTO 2020, provides details of the Spartan proof system:

[Spartan: Efficient and general-purpose zkSNARKs without trusted setup](https://eprint.iacr.org/2019/550) \
Srinath Setty \
CRYPTO 2020

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

## Examples

Run `cargo run --example EXAMPLE_NAME` to run the corresponding example. Leave `EXAMPLE_NAME` empty for a list of available examples.
