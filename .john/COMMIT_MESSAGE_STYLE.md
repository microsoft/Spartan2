# Commit Message Style

## Format

```
<Verb> <what changed>

<Optional body explaining why>
```

## Rules

1. **Subject line**: Start with an action verb in imperative mood
   - `Add`, `Remove`, `Implement`, `Fix`, `Improve`, `Expand`, `Update`

2. **Case**: Use sentence case (not Title Case)

3. **No trailing period** on the subject line

4. **PR numbers**: When merging PRs, append `(#XX)` at the end

5. **Colons**: Use to separate high-level summary from details
   - `Improve X: do Y and Z`

6. **Body** (optional): Explain *why* the change was made if not obvious from the subject

## Examples

```
Improve documentation coverage: fix rustdoc issues and add internal documentation (#93)
Add copyright and license headers to all Rust source files (#91)
Remove evaluation to speedup the prover (#86)
Implement sum-check optimizations for equality polynomials (#84)
fix blind handling (#69)
```

With body:

```
Allow non_snake_case lint to suppress rust-analyzer warnings

Variable names often match academic notation (e.g., A, B, C for R1CS
matrices), which triggers non_snake_case warnings in rust-analyzer.
```
