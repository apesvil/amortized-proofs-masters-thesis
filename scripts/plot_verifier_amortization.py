#!/usr/bin/env python3
"""
Figure 1 — verifier cost of the amortization part alone.

The local party's work splits at the point where the pipeline stops being
SNARK-agnostic:

    amortization      = statement preparation + fold path verify
                      + promise discharge + RokP verify
    folded-stmt check = 1 inner-lincheck verify on the root R_P claim

Everything in the first group reads only the SRS and n, so it is independent of
which SNARK discharges the final R_P claim; only the lincheck is Marlin-specific.
This figure plots the first group, excluding the lincheck entirely.

Left panel: amortization cost vs K, one line per n.
Right panel: stage composition for one n, so the O(n) statement-preparation term
and the log K part of the amortization are separable by eye.

Input: CSV from `cargo run --release --example bench_verifier_csv`.

Usage:
    python scripts/plot_verifier_amortization.py results_verifier.csv
    python scripts/plot_verifier_amortization.py results_verifier.csv --n 262144
    python scripts/plot_verifier_amortization.py results_verifier.csv --no-stmt-prep
"""

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt

# n is ordered, so it gets a sequential ramp (one hue, light -> dark).
SEQ_BLUE = ["#a8c7ee", "#7aa9e4", "#4d8bd9", "#2a78d6", "#17508f"]
# Categorical slots 1-4, fixed order, for the stage decomposition.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_MUTED = "#0b0b0b", "#52514e"

STAGES = [
    ("v_stmt_prep_ms", "statement prep (4 commitments)", CAT[0]),
    ("v_fold_ms", "fold path verify", CAT[1]),
    ("v_discharge_ms", "promise discharge", CAT[2]),
    ("v_rokp_ms", "RokP verify", CAT[3]),
]


def n_label(n):
    n = int(n)
    if n > 0 and (n & (n - 1)) == 0:
        return f"$n = 2^{{{n.bit_length() - 1}}}$"
    return f"$n = {n}$"


def k_label(k):
    k = int(k)
    if k > 0 and (k & (k - 1)) == 0:
        return f"$2^{{{k.bit_length() - 1}}}$"
    return str(k)


def style(ax):
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="CSV from examples/bench_verifier_csv")
    ap.add_argument("--out", default="verifier_amortization.pdf")
    ap.add_argument("--n", type=int, default=None,
                    help="n for the right-hand composition panel (default: largest)")
    ap.add_argument("--no-stmt-prep", action="store_true",
                    help="exclude statement preparation, leaving the K-dependent part only")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"n", "k", "v_stmt_prep_ms", "v_fold_ms", "v_discharge_ms",
              "v_rokp_ms", "v_amort_ms"}
    missing = needed - set(df.columns)
    if missing:
        print(f"CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df.copy()
    stages = STAGES[1:] if args.no_stmt_prep else STAGES
    df["amort"] = sum(df[c] for c, _, _ in stages)

    ns = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # -- (a) amortization cost vs K, one line per n --------------------------
    # The curves for different n coincide — that is the result, so they are not
    # end-labelled (five labels on one point is a smear); the legend carries
    # identity and the spread annotation carries the finding. Linear y against
    # log x makes the log K growth read as a straight line.
    ax = axes[0]
    spread = []
    for i, n in enumerate(ns):
        sub = df[df["n"] == n].sort_values("k")
        if sub.empty:
            continue
        color = SEQ_BLUE[i % len(SEQ_BLUE)]
        ax.plot(sub["k"], sub["amort"], "o-", color=color,
                linewidth=2, markersize=5, label=n_label(n))
        spread.append(sub.set_index("k")["amort"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$K$ (number of amortized jobs)")
    ax.set_ylabel("per-verifier time (ms)")
    ax.set_title("(a) Amortization only, lincheck excluded",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(bottom=0)
    style(ax)

    if len(spread) > 1:
        wide = pd.concat(spread, axis=1)
        worst = ((wide.max(axis=1) - wide.min(axis=1)) / wide.min(axis=1)).max()
        ax.annotate(
            f"independent of $n$: curves agree to {worst * 100:.0f}%\n"
            f"growth is in $\\log K$ alone",
            (0.97, 0.06), xycoords="axes fraction", ha="right",
            fontsize=8.5, color=INK_MUTED,
        )

    # -- (b) stage composition, one n ----------------------------------------
    n_star = args.n if args.n is not None else ns[-1]
    sub = df[df["n"] == n_star].sort_values("k")
    if sub.empty:
        print(f"warning: no data for n={n_star}", file=sys.stderr)
    ax = axes[1]
    x = list(range(len(sub)))
    bottom = [0.0] * len(sub)
    for col, label, color in stages:
        vals = sub[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, width=0.68,
               label=label, edgecolor="white", linewidth=1.0)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels([k_label(k) for k in sub["k"]])
    ax.set_xlabel("$K$")
    ax.set_ylabel("per-verifier time (ms)")
    ax.set_title(f"(b) Stage composition, {n_label(n_star)}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
