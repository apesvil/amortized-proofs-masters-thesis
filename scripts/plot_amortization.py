#!/usr/bin/env python3
"""
Plot Side 1 (K independent linchecks) vs Side 2 (amortize + 1 lincheck)
for both prover and verifier times.

Input CSV columns: n, k, side, prove_ms, verify_ms

Usage:
    python scripts/plot_amortization.py results.csv
    python scripts/plot_amortization.py results.csv --out fig.pdf
    python scripts/plot_amortization.py results.csv --n 16
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="path to CSV produced by examples/bench_csv")
    ap.add_argument("--out", default="amortization.pdf", help="output path")
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="restrict to a single n; default = one column per n in the CSV",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    expected = {"n", "k", "side", "prove_ms", "verify_ms"}
    missing = expected - set(df.columns)
    if missing:
        print(f"CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    ns = [args.n] if args.n is not None else sorted(df["n"].unique())

    # Layout: rows = {prover, verifier}, cols = ns.
    fig, axes = plt.subplots(
        2, len(ns), figsize=(5 * len(ns), 8), squeeze=False, sharex="col"
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

        pivot_prove = sub.pivot(index="k", columns="side", values="prove_ms")
        pivot_verify = sub.pivot(index="k", columns="side", values="verify_ms")

        plot_panel(
            axes[0, col],
            pivot_prove,
            f"Server (Prover), {n_label}",
            "server time (ms)",
            side1_label="Side 1: K lincheck proofs",
            side2_label="Side 2: amortize + 1 lincheck proof",
        )
        plot_panel(
            axes[1, col],
            pivot_verify,
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
