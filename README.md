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
├── marlin_ahp/           ── vendored Marlin (backend / baseline) ──
│   ├── arithmetize.rs    index A,B,C over shared domain K
│   ├── bivariate.rs      unnormalized bivariate Lagrange poly u_H(X,Y)
│   ├── r1cs.rs           satisfiable instance for the outer sumcheck
│   ├── outer_sumcheck.rs Marlin rounds 1-2 — the witness-dependent half
│   └── inner_lincheck.rs Marlin inner sumcheck (round 3) — discharges R_P
│
├── pc/                   ── infrastructure ──
├── transcript.rs         KZG · Blake3 Fiat–Shamir · binary Merkle tree
├── merkle.rs
└── core/relation.rs      Relation trait (Params / Statement / Witness / is_satisfied)

benches/marlin_lincheck.rs   Criterion bench: prover-side Side 1 vs Side 2
examples/bench_csv.rs        Full pipeline (prover + verifier), CSV output
examples/bench_verifier_csv.rs  Verifier stages + delegation baselines, CSV output
examples/bench_marlin_split.rs  Full Marlin prover, split WD / WI, CSV output
scripts/plot_amortization.py    Plots a results CSV (prover + verifier panels)
scripts/plot_verifier_amortization.py  Proof-of-inclusion verifier cost
scripts/plot_delegation_gain.py        Prove-it-yourself vs. delegate-and-verify
scripts/plot_verifier_aggregate.py     K linchecks vs. one amortized proof
scripts/plot_marlin_split.py           Witness-dependent vs. -independent Marlin
scripts/plot_marlin_density.py         The same split vs. matrix density
scripts/plot_delegation_full_marlin.py Delegating inside a whole Marlin proof
results/                     ── every CSV and figure lands here ──
├── results.csv              Checked-in sweep, n = 2^10…2^20, K = 1…128
├── results_verifier.csv     Verifier-side sweep, same grid
├── results_marlin_split.csv Full-Marlin prover split over n, both ZK settings
├── results_marlin_density.csv  Full-Marlin prover split over matrix density
│
├── amortization.*           Side 1 vs Side 2, prover and verifier rows
├── amortization_prover.*    Prover row alone
├── amortization_verifier.*  Verifier row alone
├── verifier_amortization.*  Proof-of-inclusion verifier cost (part 1)
├── delegation_gain.*        Delegation gain for the local party
├── verifier_aggregate.*     Aggregate verification work vs. K linchecks
├── marlin_split.*           Witness-dependent vs. -independent prover split
├── marlin_density.*         The same split vs. matrix density
├── delegation_full_marlin.* Delegating inside a whole Marlin proof
│                            (each figure is written as both .pdf and .png)
└── thesis/                  ── variants shaped for embedding in the text ──
    ├── amortization_prover_thesis.*    all six n, as 2 rows of 3 panels
    ├── amortization_verifier_thesis.*  all six n, as 2 rows of 3 panels
    ├── marlin_split_thesis.*           composition panel alone
    └── marlin_density_thesis.*         composition panel alone
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
    --ns 8,16,32 --ks 1,2,4,8,16,32 --reps 5 --out results/results.csv
```

Defaults: `--ns 8,16 --ks 1,2,4,8,16 --reps 5`, output to stdout. Columns are
`n,k,side,prove_ms,verify_ms`.

Plotting the sweep (needs `pandas` + `matplotlib`):

```bash
python scripts/plot_amortization.py results/results.csv --out results/amortization.pdf

# or as two files, one row each
python scripts/plot_amortization.py results/results.csv --side prover \
    --out results/amortization_prover.pdf
python scripts/plot_amortization.py results/results.csv --side verifier \
    --out results/amortization_verifier.pdf
```

No flags: [`results/results.csv`](results/results.csv) is checked in **plot-ready**, and
every figure above is exactly what it contains.

That is worth stating precisely, because its two `verify_ms` columns are *not*
what a fresh `bench_csv` sweep emits — both were replaced by dedicated
measurements, for different reasons:

* **Side 2** was measured before the square-and-multiply fix in
  `rok_pcc::evaluate_constraint` and overstated the verifier by up to ~14× at
  `n = 2^18`. It now carries the post-fix stage sum
  (`v_fold + v_discharge + v_rokp + v_lincheck`) from
  [`results/results_verifier.csv`](results/results_verifier.csv). Statement preparation is
  deliberately not in that sum — see `patch_side2_verify`.
* **Side 1** is one lincheck verify and nothing else, so its cost is
  `K`-independent — but in the sweep that millisecond-scale number is timed
  right after a `K`-lincheck prover run that can take half an hour, and it
  inherits whatever frequency state the machine drifted into (7.5 ms at
  `n = 2^20, K = 32` against a flat ~3.5 ms). It now carries a dedicated 9-rep
  measurement, one value per `n` broadcast across `K`.

`prove_ms` is untouched in both cases — the naive exponentiation was only ever
on the verifier's path, and the prover timings never had the drift exposure.

After re-running `bench_csv` you get raw verifier columns back, and need both
substitutions before the figures mean anything. `--verifier-csv` and
`--side1-csv` apply them (they are no-ops against the checked-in CSV). The
Side-1 input is itself a `bench_csv` run — `--ks 1` suffices, since the quantity
does not depend on `K` — at ~13 min for all six `n`:

```bash
cargo run --release --example bench_csv -- \
    --ns 1024,4096,16384,65536,262144,1048576 --ks 1 --reps 9 \
    --out results/side1_verify.csv

python scripts/plot_amortization.py results/results.csv --side verifier \
    --side1-csv results/side1_verify.csv \
    --verifier-csv results/results_verifier.csv \
    --out results/amortization_verifier.pdf
```

One measurement note for anyone comparing the two files: the Side-1 column sits
0–22% above `v_lincheck_ms` in `results_verifier.csv`, which times the *same*
`InnerLincheck::verify` call. There it runs back-to-back with itself and stays
warm in cache; in `bench_csv` it runs once, immediately after a lincheck *prove*
has evicted it. The cold number is the realistic one for a party that has just
been handed a proof, so that is the one plotted.

Verifier-side sweep — the same grid, but the Side-2 prover runs **once** per
`(n, K)` and only the local party's verification is re-timed, split into its
stages. Minutes rather than hours, since the `K`-independent-lincheck prover
sweep is not needed:

```bash
cargo run --release --example bench_verifier_csv -- \
    --ns 1024,4096,16384,65536,262144,1048576 --ks 1,2,4,8,16,32,64,128 \
    --reps 9 --out results/results_verifier.csv

python scripts/plot_verifier_amortization.py results/results_verifier.csv \
    --out results/verifier_amortization.pdf
python scripts/plot_delegation_gain.py results/results_verifier.csv \
    --out results/delegation_gain.pdf
python scripts/plot_verifier_aggregate.py results/results_verifier.csv \
    --out results/verifier_aggregate.pdf
```

Columns are `n,k` then the local party's stages (`v_stmt_prep_ms`, `v_fold_ms`,
`v_discharge_ms`, `v_rokp_ms`, `v_lincheck_ms`, and the derived `v_amort_ms`,
`v_total_ms`), then the no-delegation baselines (`p_local_rokp_ms`,
`p_local_lincheck_ms`) and the no-proof reference `ref_direct_eval_ms`.

Full-Marlin prover split — how much of a whole proof is witness-dependent (and
therefore *not* delegatable) versus witness-independent:

```bash
cargo run --release --example bench_marlin_split -- \
    --ns 1024,4096,16384,65536,262144 --densities 2 --reps 3 --zk both \
    --out results/results_marlin_split.csv

python scripts/plot_marlin_split.py results/results_marlin_split.csv \
    --out results/marlin_split.pdf
python scripts/plot_delegation_full_marlin.py \
    results/results_marlin_split.csv results/results_verifier.csv \
    --out results/delegation_full_marlin.pdf
```

Sweeping matrix density instead of `n` — `--densities d` puts `d·n` non-zeros
across `(A, B, C)` jointly:

```bash
cargo run --release --example bench_marlin_split -- \
    --ns 4096,16384,65536 --densities 2,3,4,5,6,8 --reps 5 --zk both \
    --out results/results_marlin_density.csv

python scripts/plot_marlin_density.py results/results_marlin_density.csv \
    --n 16384 --out results/marlin_density.pdf
```

Variants shaped for embedding in the thesis text — `--ncols` wraps the per-`n`
panels into a grid instead of one wide row (a page is taller than it is wide),
`--n` takes a subset of the sweep, `--panels composition` emits the left panel
alone:

```bash
python scripts/plot_amortization.py results/results.csv --side prover \
    --ncols 3 --out results/thesis/amortization_prover_thesis.pdf
python scripts/plot_amortization.py results/results.csv --side verifier \
    --ncols 3 --out results/thesis/amortization_verifier_thesis.pdf
python scripts/plot_marlin_split.py results/results_marlin_split.csv \
    --panels composition --out results/thesis/marlin_split_thesis.pdf
python scripts/plot_marlin_density.py results/results_marlin_density.csv \
    --n 16384 --panels composition --out results/thesis/marlin_density_thesis.pdf
```

Use `--reps 3` or more: `median` takes `xs[len/2]`, which at `reps = 2` returns
the larger sample rather than a median.

This one sizes the SRS at `D = 4n`, not the `D = 2n` the other benchmarks use:
with zero-knowledge on, Marlin's mask polynomial has degree `3n − 1` and must be
committable, and running both ZK settings at one `D` keeps the ZK comparison
from being confounded with SRS size. Consequently its `round3_ms` is **not**
comparable with `p_local_lincheck_ms` — the `g_2` degree shift and the batched
opening both scale with `D` — so any ratio against `wd_ms` must use this file's
own `round3_ms`.

---

## Results

The checked-in [`results/results.csv`](results/results.csv) is the sweep behind
[`results/amortization.png`](results/amortization.png), covering `n = 2^10 … 2^20` and
`K = 1 … 128` at 5 reps. Reproduce it with:

```bash
cargo run --release --example bench_csv -- \
    --ns 1024,4096,16384,65536,262144,1048576 --ks 1,2,4,8,16,32,64,128 \
    --reps 5 --out results/results.csv
```

Budget most of a day for the full grid: `n = 2^20` alone is ~8 h, three quarters
of it in the two `K ∈ {64, 128}` Side-1 cells (`K` linchecks at ~13.5 s each).
Peak RSS there is ~22 GB — `AbcWitness` carries `u, v` as dense length-`n`
vectors, so 128 leaves is 8.6 GB and the bench clones the leaf set once per rep.

The headline number is the crossover `K*` — the smallest batch size at which the
Server's Side-2 (fold + one lincheck) proving time beats Side-1 (`K` independent
linchecks):

| `n`      | `K*` | Server time at `K = 128` (Side 1 → Side 2) | Speed-up |
|----------|------|--------------------------------------------|----------|
| `2^10`   | —    | 3.1 s → 7.6 s                              | 0.41×    |
| `2^12`   | 4    | 10.6 s → 8.1 s                             | 1.3×     |
| `2^14`   | 4    | 40.2 s → 9.2 s                             | 4.4×     |
| `2^16`   | 4    | 143.5 s → 15.2 s                           | 9.5×     |
| `2^18`   | 4    | 522.7 s → 28.6 s                           | 18.3×    |
| `2^20`   | 4    | 1730.0 s → 81.3 s                          | 21.3×    |

Side-1 proving is linear in `K`; Side-2 is dominated by the single lincheck on
the folded claim, so the gap widens with both `n` and `K`. At `n = 2^10` the
fold's fixed cost never amortizes within `K ≤ 128` — the per-job lincheck is
already too cheap to be worth folding.

The trade is on the verifier: Side-1 verification is essentially flat in both
`n` and `K` (3.3–3.9 ms, one lincheck), while Side-2 costs a fold path + `RokP` +
lincheck and grows with `log K` — 8.2 ms at `K = 1` up to 34.7 ms at `K = 128`
(measured at `n = 2^18`, from [`results/results_verifier.csv`](results/results_verifier.csv)).

> **Note.** Both `verify_ms` columns of [`results/results.csv`](results/results.csv) are
> substituted measurements, not what `bench_csv` emitted — Side 2 because the
> sweep predates the `rok_pcc::evaluate_constraint` fix, Side 1 because a
> millisecond timing taken inside a half-hour prover cell inherits the machine's
> frequency drift. [Plotting the sweep](#building--running) above gives the full
> account and the two commands that reproduce them. `prove_ms` is untouched.

> **`n = 2^16` was re-measured.** The original block was taken while the machine
> dropped frequency partway through: the cost of one Side-1 lincheck, which must
> be flat in `K`, ran 1001 ms at `K ≤ 8` but 1848 ms at `K = 64`, and a ~2×
> excursion spanning `(K=32, Side 2)` and `(K=64, Side 1)` — adjacent cells in
> execution order — left Side 2 *non-monotone* in `K` (8.6 s at `K = 32` against
> 7.2 s at `K = 64`). The row was re-run whole, in one window, so it is
> internally consistent: 1013–1145 ms per lincheck across the sweep, Side 2
> monotone. `K*` is unchanged at 4. Its absolute times sit ~13% above the old
> `K ≤ 8` cells, so — as with the density benchmark below — **compare within a
> row, not across rows.**

> **`n = 2^20` measurement quality.** The block is sound — Side 2 is strictly
> monotone in `K`, `K*` = 4 with a 14% margin — but it carries the same mild
> drift: one Side-1 lincheck ranges 13007–14571 ms across the row (12%, peaking
> at `K = 32`, whose raw Side-1 `verify_ms` also doubled — that column has since
> been re-measured, see the note above). No cell is off by the ~2×
> that made the original `n = 2^16` row unusable, and no conclusion moves, so it
> was kept as measured. A related artifact: `bench_csv`'s `K = 1` Side-1 time
> (13007 ms) and `bench_verifier_csv`'s `p_local_lincheck_ms` (11365 ms) are the
> *same operation* measured 15% apart in two processes — at `n = 2^18` the two
> agree to 0.4%. Both are reported above, in their own tables; do not divide one
> by the other.
>
> `n = 2^20` is absent from
> [`results/delegation_full_marlin.png`](results/delegation_full_marlin.png): that figure
> inner-joins on `n` with `results_marlin_split.csv`, which has no `2^20` row.
> Adding one means running `bench_marlin_split` at `n = 2^20`, where its
> `D = 4n` sizing needs a `2^22` SRS.

### Verifier-side amortization cost

The local party's verification splits into three parts:

| part | stage | `n = 2^10` | `n = 2^18` | `n = 2^20` | scales with |
|------|-------|-----------:|-----------:|-----------:|-------------|
| **1** | statement prep (4 sparse commitments) | 0.92 ms | 0.89 ms | 1.13 ms | — |
| **1** | fold path verify                      | 0.65 ms | 0.68 ms | 0.70 ms | `log K` |
| **1** | promise discharge                     | 24.2 ms | 25.5 ms | 25.4 ms | `log K` |
|       | **proof of inclusion — subtotal**     | **25.8 ms** | **27.1 ms** | **27.3 ms** | `log K` |
| **2** | `RokP` verify                         | 4.13 ms | 4.30 ms | 4.35 ms | — |
| **3** | inner-lincheck verify                 | 3.21 ms | 3.36 ms | 3.37 ms | — |

(at `K = 128`; [`results/verifier_amortization.png`](results/verifier_amortization.png) plots
part 1, which is the term amortization adds per participating party — pass
`--with-rokp` to fold part 2 back in.)

Parts 2 and 3 are a fixed 7.3–7.7 ms tail, flat in both `n` and `K`. Part 1 is
independent of `n` too — the six curves agree to 15% for `K ≥ 2`, across a 1024×
range of `n` — and grows only with `log K`, at roughly 3.1 ms per doubling,
essentially all of it in the promise discharge. (The whole grid is measured in
one run, so that agreement is a within-session number; the ±15% is measurement
scatter on quantities of a few ms, not an `n`-dependence — it does not trend
with `n`.) Statement preparation is the client's own share: committing
to its four sparse leaf polynomials. It is *not* charged the `O(n)` evaluation
of `y_A, y_B, y_C` — that is the answer it is delegating.

### Is delegating worth it for the local party?

Both sides start from the same point: the moment Marlin's verifier samples `β`
and the claim `t(β)` — this repo's `R_P` statement `(α, β, y, η)` — is still to
be established. The baseline is the **Marlin inner sumcheck**, which
`marlin_ahp/inner_lincheck.rs` reimplements from arkworks-rs/marlin's
`prover_third_round`; `RokP::reduce` is deliberately *not* in it, since plain
Marlin never runs it ([`results/delegation_gain.png`](results/delegation_gain.png)):

| `n`    | Marlin inner sumcheck | delegate + verify (`K = 128`) | gain | break-even |
|--------|----------------------:|------------------------------:|-----:|-----------:|
| `2^10` |  27.7 ms              | 33.1 ms                       | 0.8× | `K = 32`   |
| `2^12` |  80.6 ms              | 33.3 ms                       | 2.4× | —          |
| `2^14` | 272.4 ms              | 32.8 ms                       | 8.3× | —          |
| `2^16` | 934.9 ms              | 33.3 ms                       | 28.1× | —         |
| `2^18` | 4041.8 ms             | 34.7 ms                       | 116× | —          |
| `2^20` | 11365.0 ms            | 35.0 ms                       | 325× | —          |

For reference, not plotted: running this construction *unamortized* costs
`RokP::reduce` + lincheck = 11.2 s at `n = 2^18`, so the encoding is 2.8× a
plain lincheck before the fold buys anything back (`p_local_rokp_ms` in the
CSV).

The 34.7 ms is the conservative reading, where the party checks the Server's
work itself. A party that only needs to *forward* the proof to a third-party
verifier pays statement preparation alone — 0.89 ms, a 4500× reduction — and the
~30 ms lands on the end verifier instead. That is the additive `O(log K)`
verifier overhead, i.e. a transfer of cost rather than a saving.

The figure splits the delegated cost in two, so the amortization-specific part
is separable from the fixed tail:

| part | at `n = 2^18`, `K = 128` |
|------|-------------------------:|
| **1** proof of inclusion — statement prep + fold path + promise discharge | 27.09 ms |
| **2** `RokP` verify | 4.30 ms |
| **3** folded lincheck verify | 3.36 ms |

Parts 2 and 3 are a fixed 7.7 ms tail, flat in both `n` and `K`; all of the
growth is in part 1's promise discharge.

### What amortization costs the verifying side

[`results/verifier_aggregate.png`](results/verifier_aggregate.png) is the same ledger read
across all `K` jobs at once: `K` independent lincheck verifies against `K`
proofs of inclusion plus a single `RokP` and a single folded lincheck (the
aggregate reading, where one auditor checks everything).

Amortization **loses** here, by 2.6× at `K = 1` rising to 8.1× at `K = 128`
(`n = 2^18`: 430 ms → 3475 ms). This is not a defect — it is the additive
`O(log K)` verifier overhead the construction trades for the Server's `O(K·s) →
O(K·log K + s)` saving, made explicit. One proof of inclusion costs 27.1 ms
against 3.4 ms for a plain lincheck verify, and that per-job ratio itself grows
with `log K`, so the gap widens slowly rather than converging.

In the true multi-verifier setting the `K` verifiers cannot communicate, so each
checks parts 2 and 3 for itself and the amortized total is `K · 34.7 ms` —
strictly worse again. The plotted reading is the generous one.

### How much of Marlin can be delegated at all

Everything above measures the inner sumcheck in isolation. Put it back in the
context of a whole proof ([`results/marlin_split.png`](results/marlin_split.png)): only the
witness-*independent* part is a claim about the public matrices, so only it can
be handed to a Server. From [`results/results_marlin_split.csv`](results/results_marlin_split.csv):

| `n`    | witness-dependent | witness-independent | ceiling |
|--------|------------------:|--------------------:|--------:|
| `2^10` | 58.7% / 62.5%     | 41.3% / 37.5%       | 1.70× / 1.60× |
| `2^12` | 55.8% / 60.6%     | 44.2% / 39.4%       | 1.79× / 1.65× |
| `2^14` | 54.5% / 61.6%     | 45.5% / 38.4%       | 1.83× / 1.62× |
| `2^16` | 54.3% / 60.1%     | 45.7% / 39.9%       | 1.84× / 1.66× |
| `2^18` | 54.8% / 61.4%     | 45.2% / 38.6%       | 1.82× / 1.63× |

(`no ZK / ZK`. Witness-independent counts round 3 **plus** `t(X)`, which sits
inside round 2 but depends only on `α` and the matrices — about 1% of the
prover. Zero-knowledge work is entirely witness-dependent, so it shrinks the
delegatable share by ~6 points.)

So the whole-proof speedup is capped at `(WD + WI)/WD ≈ 1.6–1.8×` no matter how
cheap verification becomes. [`results/delegation_full_marlin.png`](results/delegation_full_marlin.png)
plots the approach to that ceiling: at `n = 2^18` delegation reaches **1.81×**
against a ceiling of 1.82× — verification (34.7 ms) is negligible beside a 6.3 s
witness-dependent half — while at `n = 2^10` it decays from 1.55× to 1.20× as
`K` grows, because there the `log K` verification overhead is no longer small.

> The 121× in `results/delegation_gain.png` and the 1.8× here are not in conflict: the
> first is the speedup on the *inner sumcheck alone*, the second is what that
> becomes once the undelegatable half of the prover is included. The second is
> the number to quote for a full Marlin proof.

The `D = 4n` SRS this benchmark needs (for the ZK mask, degree `3n − 1`) does
**not** distort the split: both rounds carry one degree-`D` shifted commitment
and one degree-`D` batched opening, so the inflation is near-symmetric and
cancels in the ratio. Re-measuring at `D = 2n` (`--srs-mult 2 --zk off`) moves
the witness-independent share by ~1 point (42.7% vs 44.2% at `n = 2^12`; 44.9%
vs 45.5% at `n = 2^14`), and reproduces `p_local_lincheck_ms` to within 6–9%.

### Denser matrices are better for delegation

The split above is for `2n` non-zeros, the sparsest R1CS this harness builds.
Real circuits run denser. Sweeping `--densities` shows why it matters
([`results/marlin_density.png`](results/marlin_density.png), at `n = 2^14`):

| non-zeros | `\|K\|` | delegatable share | ceiling |
|-----------|--------:|------------------:|--------:|
| `2n`      | `2n`    | 44.9% / 41.0%     | 1.81× / 1.69× |
| `3n`      | `4n`    | 54.0% / 49.6%     | 2.18× / 1.99× |
| `4n`      | `4n`    | 54.3% / 49.8%     | 2.19× / 2.00× |
| `5n`      | `8n`    | 65.0% / 60.7%     | 2.86× / 2.55× |
| `6n`      | `8n`    | 65.3% / 61.2%     | 2.88× / 2.58× |
| `8n`      | `8n`    | 65.1% / 61.0%     | 2.86× / 2.56× |

(`no ZK / ZK`.) The witness-dependent half is FFTs and commitments over
`H`-sized polynomials and barely notices density — only the `Az`/`Bz` products
(0.6% of round 1 at `2n`, 2.5% at `8n`) and `t(X)` scale with it. The
witness-independent half is round 3 over `|K|`, which doubles with it. So
**the delegatable share climbs from ~45% to ~65% and the ceiling from 1.8× to
2.9×** as matrices densify.

The share is a **step function of `|K| = next_pow2(d·n)`, not of `d`**: `3n` and
`4n` are indistinguishable, as are `5n`, `6n` and `8n`. The plateaus are exactly
where the padding puts them, which is a useful sanity check on the measurement.

Practical reading: `2n` is the pessimistic corner of the parameter space. A
hand-written circuit at 3–5 non-zeros per constraint sits in the 54–65% band,
where delegating the witness-independent half is worth 2.2–2.9× rather than
1.8×.

> **Measurement caveat.** Absolute times in
> [`results/results_marlin_density.csv`](results/results_marlin_density.csv) and
> [`results/results_marlin_split.csv`](results/results_marlin_split.csv) drift by up to ~1.8×
> between blocks minutes apart — machine frequency state, not the code. At
> `n = 2^16`, `d = 3` and `d = 4` differ 1.87× in absolute time (8390 ms vs
> 4487 ms) yet give shares of 55.8% and 55.1%. Rounds 1-3 are measured
> milliseconds apart within one block, so the drift scales them uniformly and
> cancels in any within-row ratio. **Every figure and table above uses within-row
> ratios and is unaffected; do not compare absolute ms across rows.**

The local party's delegated cost is flat in `n`, so the gain is set entirely by
how expensive the local prove is. At `n = 2^10` the lincheck is already cheap
enough that the amortization overhead overtakes it at `K = 32`; from `n = 2^12`
up, delegation wins across the whole sweep. Note the gain *decreases* in `K`:
amortization is a win for the Server, and the local party pays a `log K` premium
for it.

---

## Status / caveats

- **This repository is still under active development.** Interfaces, module
  layout, and benchmark numbers are all subject to change without notice — pin
  a commit if you depend on any of it.
- `R_P` is intentionally left without an `is_satisfied` impl — it is the seam
  where different discharging SNARKs plug in (see `relations/p.rs`).
- SRS is sized for the largest `(n, K)` in a run; sizing logic and the
  degree-growth argument are documented in `examples/bench_csv.rs`.
- **Marlin's outer and inner sumchecks here use different α-normalizations.**
  Upstream weights `t` by the unnormalized `u_H(α, h_i)`, which is what keeps
  its verifier succinct; this crate's `R_P` uses `λ_i(α)` on both sides, because
  `P_M(α,β) = Σ M[i,j]λ_i(α)λ_j(β)` is the bivariate evaluation the delegation
  construction is about. The two are different functionals — `u_H(α,h_i) =
  |H|·λ_i(α)/h_i` is a per-`i` factor — so `outer_sumcheck` and
  `inner_lincheck` are each correct under their own convention but do **not**
  compose into one end-to-end verifiable proof. Costs are unaffected (identical
  `|K|`, degrees and operations), so the split measurements stand;
  `outer_sumcheck::repo_p_statement` converts at the seam. Full reasoning in
  that module's docs.
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



