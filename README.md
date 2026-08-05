# Amortized Proofs — Master's Thesis

> **Paper:** [*Amortized Multi-Verifier Proofs from Reductions of Knowledge*](https://eprint.iacr.org/2026/1553)
> — Nikitas Paslis, Carla Ràfols, Alexandros Zacharakis.
> Cryptology ePrint Archive, Paper 2026/1553.

This repository is the reference implementation and benchmark harness for the
construction in that paper.

A Rust implementation of a **promise-carrying local folding scheme** and the
machinery needed to benchmark it against the naïve baseline. The central
question the code answers empirically is:

> When a Server must produce proofs for `K` delegated jobs, is it cheaper to
> prove each job independently, or to **fold** the `K` jobs into one and
> discharge that single folded claim with a single SNARK?

- **Side 1 (baseline):** `K` independent Marlin inner-linchecks.
- **Side 2 (amortization):** fold `K` claims → 1, discharge the fold's
  "promises", then run **one** lincheck on the folded claim.

The crossover `K*` where the two costs meet is the headline result.

Two concerns live side by side in this repo, and it's worth keeping them apart:

1. **The amortization contribution** — the folding scheme and its promise
   discharge (`src/reductions/`, `src/relations/`). This is the thesis's own work.
2. **The Marlin discharge** — a vendored/adapted Marlin inner-lincheck
   (`src/marlin_ahp/`) used as the final SNARK **and** as the Side-1 baseline.
   It is a *pluggable backend*, not part of the contribution.

The rest of this document follows that split: the amortization pipeline first,
the Marlin discharge second, then a full reference for everything else.

---

## Cryptographic setting

- **Curve:** BLS12-381 (`ark-bls12-381`), scalar field `Fr`.
- **Commitment scheme:** KZG (`src/pc/`) — additively homomorphic, which is
  what makes folding by random linear combination possible.
- **Fiat–Shamir:** Blake3 transcript with domain-separated `fork` (`src/transcript.rs`).
- **Merkle commitments:** used by the FS/MT transform to bind `K` leaves before
  squeezing fold challenges (`src/merkle.rs`).
- Built on **arkworks 0.5**, Rust edition 2024.

---

## The amortization pipeline (in execution order)

This is the Side-2 path, exactly as `examples/bench_csv.rs` drives it. Each
numbered step names the reduction it calls and the relation it produces; the
lower-level reductions those steps compose internally are deferred to the
[reference](#reference-full-relation--reduction-catalogue) below. The pipeline
stops once everything is folded into a single `R_P` claim — discharging that
claim with the Marlin backend is [its own section](#the-marlin-discharge-vendored-backend).

```
                 ① AbcFold::prove
  K · R_{A,B,C}  ───────────────►  1 root R_{A,B,C}   +   K·κ  R_PCC promises
                                        │                      │
                                        │            ② + per-leaf leaf-encoding
                                        │               R_PCC promise (abc_leaf_pcc)
                                        │                      │
                                        │            ③ prove_discharge
                                        │       ┌──────────────┴───────────────┐
                                        │       FsMtPcc              PcoFold
                                        │   bundle → R_PCO@x   →   1 root R_PCO
                                        │                              │
                                        │                        1 KZG opening
                                   ④ RokP::reduce
                                        │
                                   1 R_P claim  ───►  Marlin discharge (own section)
```

| Step | Call                                    | Aggregate map                                | What it does                                                                 |
|------|-----------------------------------------|----------------------------------------------|-----------------------------------------------------------------------------|
| ①    | `AbcFold::prove`                        | `K R_{A,B,C} → 1 R_{A,B,C} + K·κ R_PCC`      | Folds `K = 2^κ` leaves through a binary tree; emits a promise per fold node. |
| ②    | `full_pcc_bundles` / `leaf_correctness_pcc` | `K·κ R_PCC → K·(κ+1) R_PCC`              | Appends each leaf's *encoding-correctness* promise (one per leaf).           |
| ③    | `prove_discharge`                       | `K·(κ+1) R_PCC → 1 R_PCO` (+ 1 KZG opening)  | Collapses every leaf's bundle to one `R_PCO` at a shared `x`, folds the `K` down to one root `R_PCO`, opens it once. |
| ④    | `RokP::reduce`                          | `1 R_{A,B,C} → 1 R_P + 1 R_PCO`             | Reduces the folded root to the final `R_P` claim (its `R_PCO` is opened in-proof); the `R_P` is handed to the Marlin discharge (below). |

The whole batch funnels down to a single statement. Running total of open
statements after each step:

```text
  start        K · R_{A,B,C}
  ① AbcFold    1 · R_{A,B,C}  +  K·κ     · R_PCC
  ② append     1 · R_{A,B,C}  +  K·(κ+1) · R_PCC
  ③ discharge  1 · R_{A,B,C}                        (K·(κ+1) R_PCC → 1 R_PCO, opened)
  ④ RokP       1 · R_P                              (R_PCO opened inside the RokP proof)
               └──►  handed to the Marlin discharge (own section), which closes it
```

The relation *types* threaded through this path, in order:
**`R_{A,B,C}` → `R_PCC` → `R_PCO` → `R_P`.**

A single local **verifier** checks only *its own* slice: its fold path, its
reconstructed promise, the shared discharge (`FsMtPcc` + `PcoFold` + the one
opening), and the `RokP` reduction — independent of `K` up to a `log K` path
term. (Verifying the final `R_P` claim is the Marlin discharge's job, below.)

---

## The Marlin discharge (vendored backend)

Everything under `src/marlin_ahp/` is adapted from
[`arkworks-rs/marlin`](https://github.com/arkworks-rs/marlin) (MIT/Apache-2.0)
and re-typed for this crate. It is a *pluggable backend*, not part of the
amortization contribution, and plays two roles in the benchmark.

**Step ⑤ — Side 2 discharge.** The amortization pipeline (above) ends by handing
over a single `R_P` claim; one Marlin inner-lincheck closes it — this is the
step that finishes the funnel:

```text
  ⑤ InnerLincheck::prove    1 R_P → ∅     (Marlin inner sumcheck)
```

**Side 1 — baseline.** The same `InnerLincheck::prove`, run `K` times (once per
delegated job, no folding) — the baseline that Side 2 has to beat.

| File                 | Role                                                                 |
|----------------------|----------------------------------------------------------------------|
| `arithmetize.rs`     | Index `(A,B,C)` over a shared domain `K` (`row/col/val/row_col`).     |
| `bivariate.rs`       | Unnormalized bivariate Lagrange poly `u_H(X,Y)` helper.              |
| `inner_lincheck.rs`  | Marlin inner sumcheck — the SNARK that discharges `R_P`.              |

Because it is a *backend*, `R_P` (`src/relations/p.rs`) is deliberately left
without an `is_satisfied` impl: it is the seam where a different discharging
SNARK could be swapped in.

---

## Reference: full relation & reduction catalogue

The steps above only name the reductions the benchmark calls *directly*. Those
compose finer-grained reductions internally; this section lists everything.

### Relations (`src/relations/`)

| Relation      | File                     | On path? | Statement asserts                                                                                  |
|---------------|--------------------------|----------|----------------------------------------------------------------------------------------------------|
| `R_{A,B,C}`   | `abc.rs`                 | ✅ leaf   | Three bilinear forms `y_X = uᵀ·X·v` (`X ∈ {A,B,C}`), `u=λ(α)`, `v=λ(β)` as rational `(N,T)` pairs.  |
| `R_PCC`       | `pcc.rs`                 | ✅        | Commitments `c_i` + degree bounds `d_i` + polynomial constraints `Q_j(X, p(X)) = 0`. The promises. |
| `R_PCO`       | `pco.rs`                 | ✅        | A single KZG evaluation claim `p(point) = value`. `R_PCO^x` = many sharing one `x`.                 |
| `R_P`         | `p.rs`                   | ✅ output | `y = P_A(α,β) + η·P_B(α,β) + η²·P_C(α,β)`. The benchmark boundary → Marlin.                          |
| `R_PCC-D`     | `pcc_d.rs`               | internal | `R_PCC` augmented with degree-test shifted commitments (used inside `FsMtPcc`).                     |

Plus `abc_leaf_pcc.rs`, which builds the per-leaf leaf-encoding-correctness
`R_PCC` promise (step ②).

### Reductions (`src/reductions/`)

**Called directly by the benchmark:**

| Reduction   | Paper name                 | Maps (aggregate over the batch)                         |
|-------------|----------------------------|---------------------------------------------------------|
| `AbcFold`   | `LfscPcc(Π_{ABC}, κ)`      | `K · R_{A,B,C} → 1 R_{A,B,C} + K·κ · R_PCC`             |
| `discharge` | —                          | `K·(κ+1) · R_PCC → 1 R_PCO` (composition, + KZG opening) |
| `RokP`      | `Π_P`                      | `1 R_{A,B,C} → 1 R_P + 1 R_PCO`                         |

**Composed internally (not called by the benchmark directly):**

| Reduction   | Paper name                 | Maps                                             | Composed inside |
|-------------|----------------------------|-------------------------------------------------|-----------------|
| `RokAbc`    | `Π_{ABC}`                  | `R_{A,B,C}² → R_{A,B,C} × R_PCC`                 | `AbcFold`       |
| `FsMtPcc`   | `FsMt(Π_PC ∘ Π_DT, κ)`     | per-leaf `(κ+1) R_PCC → R_PCO` at shared `x`    | `discharge`     |
| `RokDt`     | `Π_DT`                     | `R_PCC → R_PCC-D`                               | `FsMtPcc`       |
| `RokPcc`    | `Π_PC`                     | `R_PCC-D → R_PCO`                               | `FsMtPcc`       |
| `PcoFold`   | `LfscPcc(RokPcoFold, κ)`   | `K · R_PCO^x → R_PCO`                            | `discharge`     |
| `RokPco`    | `Π_PCO`                    | `(R_PCO^x)^ℓ → R_PCO^x`                          | `PcoFold`       |

---

## Module map

```
src/
├── reductions/           ── amortization contribution ──
├── relations/            the relation zoo + fold promises
│
├── marlin_ahp/           ── vendored Marlin discharge (backend / baseline) ──
│   ├── arithmetize.rs    index A,B,C over shared domain K
│   ├── bivariate.rs      unnormalized bivariate Lagrange poly u_H(X,Y)
│   └── inner_lincheck.rs Marlin inner sumcheck — discharges R_P
│
├── pc/                   ── infrastructure ──
├── transcript.rs         KZG · Blake3 Fiat–Shamir · binary Merkle tree
├── merkle.rs
└── core/relation.rs      Relation trait (Params / Statement / Witness / is_satisfied)

benches/marlin_lincheck.rs   Criterion bench: prover-side Side 1 vs Side 2
examples/bench_csv.rs        Full pipeline (prover + verifier), CSV output
scripts/plot_amortization.py Plots a results CSV (prover + verifier panels)
results.csv                  Checked-in sweep, n = 2^10…2^18, K = 1…128
amortization.png/.pdf        The plot of that sweep
```

---

## Building & running

Unit tests (relation round-trips, forgery-rejection, arithmetization):

```bash
cargo test
```

Criterion benchmarks (prover-side, HTML reports under `target/criterion/`):

```bash
cargo bench
```

Full `(n, K)` sweep with prover **and** verifier timings, as CSV:

```bash
cargo run --release --example bench_csv -- \
    --ns 8,16,32 --ks 1,2,4,8,16,32 --reps 5 --out results.csv
```

Defaults: `--ns 8,16 --ks 1,2,4,8,16 --reps 5`, output to stdout. Columns are
`n,k,side,prove_ms,verify_ms`.

Plotting the sweep (needs `pandas` + `matplotlib`):

```bash
python scripts/plot_amortization.py results.csv --out amortization.pdf
```

---

## Results

The checked-in [`results.csv`](results.csv) is the sweep behind
[`amortization.png`](amortization.png), covering `n = 2^10 … 2^18` and
`K = 1 … 128` at 5 reps. Reproduce it with:

```bash
cargo run --release --example bench_csv -- \
    --ns 1024,4096,16384,65536,262144 --ks 1,2,4,8,16,32,64,128 \
    --reps 5 --out results.csv
```

The headline number is the crossover `K*` — the smallest batch size at which the
Server's Side-2 (fold + one lincheck) proving time beats Side-1 (`K` independent
linchecks):

| `n`      | `K*` | Server time at `K = 128` (Side 1 → Side 2) | Speed-up |
|----------|------|--------------------------------------------|----------|
| `2^10`   | —    | 3.1 s → 7.6 s                              | 0.41×    |
| `2^12`   | 4    | 10.6 s → 8.1 s                             | 1.3×     |
| `2^14`   | 4    | 40.2 s → 9.2 s                             | 4.4×     |
| `2^16`   | 4    | 149.9 s → 14.3 s                           | 10.5×    |
| `2^18`   | 4    | 522.7 s → 28.6 s                           | 18.3×    |

Side-1 proving is linear in `K`; Side-2 is dominated by the single lincheck on
the folded claim, so the gap widens with both `n` and `K`. At `n = 2^10` the
fold's fixed cost never amortizes within `K ≤ 128` — the per-job lincheck is
already too cheap to be worth folding.

The trade is on the verifier: Side-1 verification is essentially flat in both
`n` and `K` (~4 ms, one lincheck), while Side-2 costs a fold path + `RokP` +
lincheck and grows with `log K` — 8 ms at `K = 1` up to 513 ms at `K = 128`
(measured at `n = 2^18`).

---

## Status / caveats

- **This repository is still under active development.** Interfaces, module
  layout, and benchmark numbers are all subject to change without notice — pin
  a commit if you depend on any of it.
- `R_P` is intentionally left without an `is_satisfied` impl — it is the seam
  where different discharging SNARKs plug in (see `relations/p.rs`).
- SRS is sized for the largest `(n, K)` in a run; sizing logic and the
  degree-growth argument are documented in `examples/bench_csv.rs`.
- This is research/thesis code: correctness and legibility over production
  hardening.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

Code under `src/marlin_ahp/` is adapted from
[`arkworks-rs/marlin`](https://github.com/arkworks-rs/marlin) and retains its
original MIT/Apache-2.0 licensing.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

---

## Citing

```bibtex
@misc{cryptoeprint:2026/1553,
      author = {Nikitas Paslis and Carla Ràfols and Alexandros Zacharakis},
      title = {Amortized Multi-Verifier Proofs from Reductions of Knowledge},
      howpublished = {Cryptology {ePrint} Archive, Paper 2026/1553},
      year = {2026},
      url = {https://eprint.iacr.org/2026/1553}
}
```

---

*Crate: `amortized-proofs-masters-thesis` · research code accompanying
[ePrint 2026/1553](https://eprint.iacr.org/2026/1553).*
