#!/usr/bin/env python3
"""
Figure 6 — delegating inside a *full* Marlin proof.

Earlier figures ask what the inner sumcheck alone costs to prove versus to
delegate. This one puts that saving back in context: a party producing a whole
Marlin proof can only delegate the witness-independent part, so its total is

    keep everything :  WD + WI                        (a full Marlin proof)
    delegate the WI :  WD + verify(1 + 2 + 3)         (amortized, per K)

where WD = rounds 1-2 less t, and WI = round 3 plus t (t is witness-independent
work sitting inside round 2).

Because WD is common to both, it caps the speedup at (WD + WI) / WD however
cheap verification becomes — the ceiling drawn in panel (b). That is a far
soberer number than the inner-sumcheck-only comparison in delegation_gain.pdf,
and it is the honest one for a full proof.

Two CSVs, deliberately:
  * results/results_marlin_split.csv  — WD and WI, measured at D = 4n
  * results/results_verifier.csv      — delegated verification cost, measured at D = 2n
Mixing them is safe: verification is D-insensitive (the only D-dependence left
is pow_usize(beta, D - d), O(log D) since the square-and-multiply fix), whereas
WI is not — which is exactly why WI is taken from the split CSV and not from
p_local_lincheck_ms.

Usage:
    python scripts/plot_delegation_full_marlin.py \
        results/results_marlin_split.csv results/results_verifier.csv
    python scripts/plot_delegation_full_marlin.py ... --zk 1
"""

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Categorical slots 1-3, matching the other delegation figures.
C_KEEP, C_DELEG, C_CEIL = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED = "#0b0b0b", "#52514e"


def n_label(n):
    n = int(n)
    if n > 0 and (n & (n - 1)) == 0:
        return f"$n = 2^{{{n.bit_length() - 1}}}$"
    return f"$n = {n}$"


def style(ax):
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("split_csv", help="CSV from examples/bench_marlin_split")
    ap.add_argument("verifier_csv", help="CSV from examples/bench_verifier_csv")
    ap.add_argument("--out", default="results/delegation_full_marlin.pdf")
    ap.add_argument("--zk", type=int, default=0, choices=(0, 1),
                    help="which ZK setting to plot (default 0, matching the "
                         "non-hiding KZG the rest of the crate uses)")
    args = ap.parse_args()

    split = pd.read_csv(args.split_csv)
    ver = pd.read_csv(args.verifier_csv)

    for name, df, needed in (
        (args.split_csv, split,
         {"n", "zk", "round1_ms", "round2_ms", "round2_t_ms", "round3_ms"}),
        (args.verifier_csv, ver,
         {"n", "k", "v_stmt_prep_ms", "v_fold_ms", "v_discharge_ms",
          "v_rokp_ms", "v_lincheck_ms"}),
    ):
        missing = needed - set(df.columns)
        if missing:
            print(f"{name} is missing columns: {missing}", file=sys.stderr)
            sys.exit(1)

    split = split[split["zk"] == args.zk].copy()
    if split.empty:
        print(f"no rows with zk={args.zk} in {args.split_csv}", file=sys.stderr)
        sys.exit(1)
    split["wd_ms"] = (split["round1_ms"] + split["round2_ms"]
                      - split["round2_t_ms"])
    split["wi_ms"] = split["round3_ms"] + split["round2_t_ms"]
    split["full_ms"] = split["wd_ms"] + split["wi_ms"]
    split["ceiling"] = split["full_ms"] / split["wd_ms"]

    ver = ver.copy()
    ver["client_ms"] = (ver["v_stmt_prep_ms"] + ver["v_fold_ms"]
                        + ver["v_discharge_ms"] + ver["v_rokp_ms"]
                        + ver["v_lincheck_ms"])

    merged = ver.merge(split[["n", "wd_ms", "wi_ms", "full_ms", "ceiling"]],
                       on="n", how="inner")
    if merged.empty:
        print("the two CSVs share no n values", file=sys.stderr)
        sys.exit(1)
    merged["delegated_ms"] = merged["wd_ms"] + merged["client_ms"]
    merged["speedup"] = merged["full_ms"] / merged["delegated_ms"]

    ns = sorted(merged["n"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # -- (a) absolute cost vs n, at the largest K -----------------------------
    #    Lines rather than bars: a bar on a log axis has an arbitrary baseline,
    #    and these span two decades.
    ax = axes[0]
    k_star = merged["k"].max()
    at_k = merged[merged["k"] == k_star].sort_values("n")
    ax.plot(at_k["n"], at_k["full_ms"], "o-", color=C_KEEP,
            linewidth=2, markersize=5, label="full Marlin proof (WD $+$ WI)")
    ax.plot(at_k["n"], at_k["delegated_ms"], "s-", color=C_DELEG,
            linewidth=2, markersize=5,
            label=f"WD $+$ verify amortized ($K = {int(k_star)}$)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel("local prover time (ms)")
    ax.set_title("(a) Cost of a whole proof", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    style(ax)

    # -- (b) speedup against the ceiling --------------------------------------
    ax = axes[1]
    for i, n in enumerate(ns):
        sub = merged[merged["n"] == n].sort_values("k")
        shade = 0.35 + 0.65 * (i / max(len(ns) - 1, 1))
        ax.plot(sub["k"], sub["speedup"], "o-", color=C_DELEG, alpha=shade,
                linewidth=2, markersize=4, label=n_label(n))
        ax.axhline(sub["ceiling"].iloc[0], color=C_CEIL, linewidth=1.0,
                   linestyle=":", alpha=shade)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.annotate("break-even", (merged["k"].max(), 1.0),
                textcoords="offset points", xytext=(-2, 5),
                fontsize=8, color=INK_MUTED, ha="right")
    # Label the ceiling once, on the line itself, and leave the explanation to
    # the caption — a two-line gloss here collides with everything.
    ax.annotate("ceiling $=$ (WD$+$WI)$/$WD", (merged["k"].max(),
                merged["ceiling"].max()),
                textcoords="offset points", xytext=(-2, 4),
                fontsize=8, color=C_CEIL, ha="right", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$K$")
    ax.set_ylabel("speedup from delegating ($\\times$)")
    ax.set_title("(b) Speedup, and what caps it", fontsize=10, color=INK,
                 loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=2)
    # Headroom above for the ceiling label, and room below the break-even rule
    # so the legend is not clipped by the axes.
    ax.set_ylim(bottom=0.90, top=ax.get_ylim()[1] * 1.12)
    style(ax)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
