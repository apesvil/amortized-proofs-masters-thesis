#!/usr/bin/env python3
"""
Figure 5 — how a Marlin proving time splits into witness-dependent and
witness-independent work.

    round 1   w-hat, z_A, z_B [, mask]     witness-dependent
    round 2   t, g_1, h_1                  witness-dependent, except t
    round 3   g_2, h_2                     witness-INdependent: the inner
                                           sumcheck, the only delegatable part

`t(X)` is broken out of round 2 because it depends only on alpha and the
matrices, never on the witness — so the witness-independent share is not simply
"round 3", and the figure shows it as its own band.

Both zero-knowledge settings are plotted. ZK work (the mask polynomial and the
v_H blinding) is entirely witness-dependent, so it moves the split; `zk = 0`
matches inner_lincheck's non-hiding KZG, `zk = 1` is what a deployment runs.

Input: CSV from `cargo run --release --example bench_marlin_split`.

Usage:
    python scripts/plot_marlin_split.py results/results_marlin_split.csv
"""

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Categorical slots 1-4, fixed order.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_MUTED = "#0b0b0b", "#52514e"

# (column, label, colour). Witness-dependent first, then the independent parts.
BANDS = [
    ("round1_ms", "round 1 (witness)", CAT[0]),
    ("round2_wd_ms", "round 2 less $t$ (witness)", CAT[1]),
    ("round2_t_ms", "$t(X)$ (witness-independent)", CAT[3]),
    ("round3_ms", "round 3: inner sumcheck (witness-independent)", CAT[2]),
]


def n_label(n):
    n = int(n)
    if n > 0 and (n & (n - 1)) == 0:
        return f"$2^{{{n.bit_length() - 1}}}$"
    return str(n)


def style(ax):
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="CSV from examples/bench_marlin_split")
    ap.add_argument("--out", default="results/marlin_split.pdf")
    ap.add_argument("--panels", choices=("both", "composition"), default="both",
                    help="'composition' emits panel (a) alone, for embedding")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"n", "zk", "round1_ms", "round2_ms", "round2_t_ms", "round3_ms"}
    missing = needed - set(df.columns)
    if missing:
        print(f"CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df.copy()
    df["round2_wd_ms"] = df["round2_ms"] - df["round2_t_ms"]
    df["total_ms"] = df["round1_ms"] + df["round2_ms"] + df["round3_ms"]
    # Witness-independent = round 3 plus t, which sits inside round 2.
    df["wi_ms"] = df["round3_ms"] + df["round2_t_ms"]
    df["wi_pct"] = 100.0 * df["wi_ms"] / df["total_ms"]

    ns = sorted(df["n"].unique())
    zks = sorted(df["zk"].unique())

    solo = args.panels == "composition"
    fig, axes = plt.subplots(1, 1 if solo else 2,
                             figsize=(6.0, 4.6) if solo else (11.5, 4.6),
                             squeeze=False)
    axes = axes[0]

    # -- (a) composition: stacked, and therefore linear. Stacking on a log axis
    #    would break additivity — segment lengths would stop summing to the bar.
    ax = axes[0]
    width = 0.36
    xs = list(range(len(ns)))
    offsets = {z: (i - (len(zks) - 1) / 2) * width for i, z in enumerate(zks)}
    for z in zks:
        sub = df[df["zk"] == z].set_index("n").reindex(ns)
        bottom = [0.0] * len(ns)
        pos = [x + offsets[z] for x in xs]
        for col, label, color in BANDS:
            vals = 100.0 * sub[col].to_numpy() / sub["total_ms"].to_numpy()
            ax.bar(pos, vals, width, bottom=bottom, color=color,
                   edgecolor="white", linewidth=1.0,
                   label=label if z == zks[0] else None)
            bottom = [b + v for b, v in zip(bottom, vals)]
        # Above the bars, not below: under the axis they collide with the n
        # tick labels once there are more than a couple of groups.
        for x in xs:
            ax.annotate("ZK" if z else "no ZK", (x + offsets[z], 100),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7, color=INK_MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([n_label(n) for n in ns])
    ax.set_xlabel("$n$ (constraints; $2n$ non-zeros)")
    ax.set_ylabel("share of prover time (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    # No panel letter when it stands alone.
    ax.set_title("Composition" if solo else "(a) Composition",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.10), ncol=2)
    style(ax)

    if solo:
        fig.tight_layout()
        fig.savefig(args.out)
        print(f"wrote {args.out}")
        return

    # -- (b) absolute: lines, not stacked bars, so a log axis is honest.
    ax = axes[1]
    for z in zks:
        sub = df[df["zk"] == z].sort_values("n")
        dash = "-" if not z else "--"
        tag = "ZK" if z else "no ZK"
        ax.plot(sub["n"], sub["round1_ms"] + sub["round2_wd_ms"], f"o{dash}",
                color=CAT[0], linewidth=2, markersize=5,
                label=f"witness-dependent, {tag}")
        ax.plot(sub["n"], sub["wi_ms"], f"s{dash}",
                color=CAT[2], linewidth=2, markersize=5,
                label=f"witness-independent, {tag}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel("prover time (ms)")
    ax.set_title("(b) Absolute", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    lo, hi = df["wi_pct"].min(), df["wi_pct"].max()
    ax.annotate(
        f"delegatable share: {lo:.0f}–{hi:.0f}%",
        (0.97, 0.06), xycoords="axes fraction", ha="right",
        fontsize=8.5, color=INK_MUTED,
    )
    style(ax)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
