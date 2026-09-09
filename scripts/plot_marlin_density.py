#!/usr/bin/env python3
"""
Figure 7 — how matrix density moves the witness-dependent / witness-independent
split of a Marlin prover.

Density `d` means `d·n` non-zeros across (A, B, C) jointly. The witness-
dependent half is dominated by FFTs and commitments over `H`-sized polynomials,
which do not care about density; the witness-independent half is round 3 over
the indexer domain `|K|`, which does. So the delegatable share should grow with
density — and with it the ceiling on any delegation speedup.

Note `|K| = next_pow2(d·n)`, so d = 3 and d = 4 share `|K| = 4n`, as do d = 5
and d = 6 at `8n`. Round 3 is therefore a step function of density, while the
Az/Bz products and t(X) track the true `d·n`. The `|K|` plateaus are marked.

Input: CSV from `cargo run --release --example bench_marlin_split` with several
`--densities`.

Usage:
    python scripts/plot_marlin_density.py results/results_marlin_density.csv
    python scripts/plot_marlin_density.py results/results_marlin_density.csv --n 65536
"""

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Categorical slots 1-4, fixed order.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ_BLUE = ["#a8c7ee", "#7aa9e4", "#4d8bd9", "#2a78d6", "#17508f"]
INK, INK_MUTED = "#0b0b0b", "#52514e"

BANDS = [
    ("round1_ms", "round 1 (witness)", CAT[0]),
    ("round2_wd_ms", "round 2 less $t$ (witness)", CAT[1]),
    ("round2_t_ms", "$t(X)$ (witness-independent)", CAT[3]),
    ("round3_ms", "round 3: inner sumcheck (witness-independent)", CAT[2]),
]


def n_label(n):
    n = int(n)
    if n > 0 and (n & (n - 1)) == 0:
        return f"$n = 2^{{{n.bit_length() - 1}}}$"
    return f"$n = {n}$"


def style(ax):
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="CSV from examples/bench_marlin_split")
    ap.add_argument("--out", default="results/marlin_density.pdf")
    ap.add_argument("--n", type=int, default=None,
                    help="n for the composition panel (default: largest)")
    ap.add_argument("--panels", choices=("both", "composition"), default="both",
                    help="'composition' emits panel (a) alone, for embedding")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"n", "density", "k_size", "zk", "round1_ms", "round2_ms",
              "round2_t_ms", "round3_ms"}
    missing = needed - set(df.columns)
    if missing:
        print(f"CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df.copy()
    df["round2_wd_ms"] = df["round2_ms"] - df["round2_t_ms"]
    df["total_ms"] = df["round1_ms"] + df["round2_ms"] + df["round3_ms"]
    df["wd_ms"] = df["round1_ms"] + df["round2_wd_ms"]
    df["wi_ms"] = df["round3_ms"] + df["round2_t_ms"]
    df["wi_pct"] = 100.0 * df["wi_ms"] / df["total_ms"]
    df["ceiling"] = df["total_ms"] / df["wd_ms"]

    ds = sorted(df["density"].unique())
    ns = sorted(df["n"].unique())
    n_star = args.n if args.n is not None else ns[-1]

    solo = args.panels == "composition"
    fig, axes = plt.subplots(1, 1 if solo else 2,
                             figsize=(6.0, 4.4) if solo else (11.5, 4.4),
                             squeeze=False)
    axes = axes[0]

    # -- (a) composition vs density, one n, paired by ZK setting -------------
    #    Stacked, and therefore linear: stacking on a log axis would break
    #    additivity. Same paired layout as `plot_marlin_split.py`.
    ax = axes[0]
    zks = sorted(df["zk"].unique())
    at_n = df[df["n"] == n_star]
    if at_n.empty:
        print(f"no rows for n={n_star}", file=sys.stderr)
        sys.exit(1)
    xs = list(range(len(ds)))
    width = 0.36
    offsets = {z: (i - (len(zks) - 1) / 2) * width for i, z in enumerate(zks)}
    for z in zks:
        sub = at_n[at_n["zk"] == z].set_index("density").reindex(ds)
        bottom = [0.0] * len(ds)
        pos = [x + offsets[z] for x in xs]
        for col, label, color in BANDS:
            vals = 100.0 * sub[col].to_numpy() / sub["total_ms"].to_numpy()
            ax.bar(pos, vals, width, bottom=bottom, color=color,
                   label=label if z == zks[0] else None,
                   edgecolor="white", linewidth=1.0)
            bottom = [b + v for b, v in zip(bottom, vals)]
        # Above the bars: under the axis these collide with the density labels.
        for x in xs:
            ax.annotate("ZK" if z else "no ZK", (x + offsets[z], 100),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7, color=INK_MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"${int(d)}n$" for d in ds])
    ax.set_xlabel("non-zeros in $(A, B, C)$")
    ax.set_ylabel("share of prover time (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    # No panel letter when it stands alone.
    ax.set_title(f"{'' if solo else '(a) '}Composition, {n_label(n_star)}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.10), ncol=2)
    style(ax)

    if solo:
        fig.tight_layout()
        fig.savefig(args.out)
        print(f"wrote {args.out}")
        return

    # -- (b) delegatable share vs density, all n -----------------------------
    ax = axes[1]
    for i, n in enumerate(ns):
        color = SEQ_BLUE[i % len(SEQ_BLUE)]
        for zk in sorted(df["zk"].unique()):
            s = df[(df["n"] == n) & (df["zk"] == zk)].sort_values("density")
            if s.empty:
                continue
            ax.plot(s["density"], s["wi_pct"], "o-" if not zk else "s--",
                    color=color, linewidth=2, markersize=5,
                    label=f"{n_label(n)}, {'ZK' if zk else 'no ZK'}")
    # Where |K| steps up: the plateaus that make round 3 a step function.
    steps = df.groupby("density")["k_size"].first()
    prev = None
    for d in ds:
        k = steps.loc[d]
        if prev is not None and k != prev:
            ax.axvline(d - 0.5, color=INK_MUTED, linewidth=0.8,
                       linestyle=":", alpha=0.6)
        prev = k
    ax.set_xticks(ds)
    ax.set_xticklabels([f"${int(d)}n$" for d in ds])
    ax.set_xlabel("non-zeros in $(A, B, C)$")
    ax.set_ylabel("delegatable share of the prover (%)")
    ax.set_title("(b) What fraction can be delegated", fontsize=10, color=INK,
                 loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", ncol=2)
    ax.annotate("dotted rules: $|K|$ doubles", (0.03, 0.94),
                xycoords="axes fraction", fontsize=8, color=INK_MUTED)
    style(ax)
    ax.grid(True, axis="both", alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
