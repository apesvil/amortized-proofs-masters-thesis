#!/usr/bin/env python3
"""
Plot Side 1 (K independent linchecks) vs Side 2 (amortize + 1 lincheck)
for both prover and verifier times.

Input CSV columns: n, k, side, prove_ms, verify_ms

`--side` splits the two rows into separate files; `--verifier-csv` replaces the
stale Side-2 verifier column with post-fix measurements (see
`patch_side2_verify`) and should be passed whenever a verifier row is rendered.

Usage:
    python scripts/plot_amortization.py results/results.csv
    python scripts/plot_amortization.py results/results.csv --out fig.pdf
    python scripts/plot_amortization.py results/results.csv --n 16
    python scripts/plot_amortization.py results/results.csv --side prover \\
        --out amortization_prover.pdf
    python scripts/plot_amortization.py results/results.csv --side verifier \\
        --verifier-csv results/results_verifier.csv --out amortization_verifier.pdf
"""

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt


def find_crossover(pivot, side1_col="side1", side2_col="side2"):
    """First K where side2 < side1. Returns None if no crossover."""
    cross = pivot[pivot[side2_col] < pivot[side1_col]]
    if cross.empty:
        return None
    return cross.index[0]


def plot_panel(ax, pivot, title, ylabel, side1_label, side2_label):
    ax.plot(pivot.index, pivot["side1"], "o-", label=side1_label, linewidth=1.5)
    ax.plot(pivot.index, pivot["side2"], "s-", label=side2_label, linewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("K (number of delegated jobs)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # Crossover annotation (only meaningful when side2 actually drops below side1)
    k_star = find_crossover(pivot)
    if k_star is not None:
        ax.axvline(k_star, color="gray", linestyle="--", alpha=0.6)
        y_at_star = pivot.loc[k_star, "side2"]
        ax.annotate(
            f"K* = {k_star}",
            (k_star, y_at_star),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            color="gray",
        )


def patch_side2_verify(df, path):
    """Replace the Side-2 `verify_ms` column with post-fix measurements.

    `results/results.csv`'s Side-2 verifier timings predate the square-and-multiply fix
    in `rok_pcc::evaluate_constraint` and overstate the verifier by up to ~14×
    at n = 2^18. `results/results_verifier.csv` measures the same four stages after the
    fix, so summing them reproduces exactly what `bench_csv` puts in
    `verify_ms` for Side 2 (statement prep is not part of that column).

    Side 1 is left alone: one lincheck verify never touched the naive
    exponentiation, so those numbers were always correct.
    """
    vdf = pd.read_csv(path)
    needed = {"n", "k", "v_fold_ms", "v_discharge_ms", "v_rokp_ms", "v_lincheck_ms"}
    missing = needed - set(vdf.columns)
    if missing:
        print(f"{path} is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    fixed = vdf[["n", "k"]].copy()
    fixed["side"] = "side2"
    fixed["fixed_ms"] = (vdf["v_fold_ms"] + vdf["v_discharge_ms"]
                         + vdf["v_rokp_ms"] + vdf["v_lincheck_ms"])

    out = df.merge(fixed, on=["n", "k", "side"], how="left")
    unmatched = out["fixed_ms"].isna() & (out["side"] == "side2")
    if unmatched.any():
        print(f"warning: {int(unmatched.sum())} Side-2 rows had no match in "
              f"{path} and keep their stale values", file=sys.stderr)
    out["verify_ms"] = out["fixed_ms"].fillna(out["verify_ms"])
    return out.drop(columns=["fixed_ms"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="path to CSV produced by examples/bench_csv")
    ap.add_argument("--out", default="results/amortization.pdf", help="output path")
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="restrict to a single n; default = one column per n in the CSV",
    )
    ap.add_argument(
        "--side",
        choices=("both", "prover", "verifier"),
        default="both",
        help="which row to render; 'prover' and 'verifier' give one file each",
    )
    ap.add_argument(
        "--verifier-csv",
        default=None,
        help="results/results_verifier.csv, to replace the stale Side-2 verify column "
             "(see patch_side2_verify). Strongly recommended with --side "
             "verifier or both.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    expected = {"n", "k", "side", "prove_ms", "verify_ms"}
    missing = expected - set(df.columns)
    if missing:
        print(f"CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    wants_verifier = args.side in ("both", "verifier")
    if args.verifier_csv:
        df = patch_side2_verify(df, args.verifier_csv)
    elif wants_verifier:
        print("warning: rendering verifier timings straight from "
              f"{args.csv}. Its Side-2 verify_ms predates the "
              "rok_pcc::evaluate_constraint fix and overstates the verifier by "
              "up to ~14x at n = 2^18. Pass --verifier-csv results/results_verifier.csv "
              "to use post-fix measurements.", file=sys.stderr)

    ns = [args.n] if args.n is not None else sorted(df["n"].unique())

    rows = ["prover", "verifier"] if args.side == "both" else [args.side]
    fig, axes = plt.subplots(
        len(rows), len(ns), figsize=(5 * len(ns), 4 * len(rows)),
        squeeze=False, sharex="col",
    )

    for col, n in enumerate(ns):
        sub = df[df["n"] == n]
        if sub.empty:
            print(f"warning: no data for n={n}", file=sys.stderr)
            continue

        # Render n as 2^i when it's a power of two (the common case).
        n_int = int(n)
        if n_int > 0 and (n_int & (n_int - 1)) == 0:
            n_label = f"$n = 2^{{{n_int.bit_length() - 1}}}$"
        else:
            n_label = f"n = {n_int}"

        for row, which in enumerate(rows):
            if which == "prover":
                plot_panel(
                    axes[row, col],
                    sub.pivot(index="k", columns="side", values="prove_ms"),
                    f"Server (Prover), {n_label}",
                    "server time (ms)",
                    side1_label="Side 1: K lincheck proofs",
                    side2_label="Side 2: amortize + 1 lincheck proof",
                )
            else:
                plot_panel(
                    axes[row, col],
                    sub.pivot(index="k", columns="side", values="verify_ms"),
                    f"Local party (Verifier), {n_label}",
                    "per-verifier time (ms)",
                    side1_label="Side 1: 1 lincheck verify",
                    side2_label="Side 2: 1 path + RokP + lincheck verify",
                )

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
