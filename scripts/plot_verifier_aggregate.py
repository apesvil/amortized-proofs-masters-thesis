#!/usr/bin/env python3
"""
Figure 4 — total verification work: K plain linchecks vs one amortized proof.

    baseline   K independent lincheck verifies          =  K · lincheck
    amortized  K proofs of inclusion, one RokP, one      =  K · (1) + (2) + (3)
               verify of the folded lincheck

This is the *aggregate* reading, where a single party (an auditor, an
aggregator) checks all K jobs, so parts 2 and 3 are checked once. In the true
multi-verifier setting the K verifiers cannot communicate, so each checks 2 and
3 for itself and the amortized total is `K · (1 + 2 + 3)` — strictly worse than
what is plotted here. Both readings answer "what does amortization cost the
verifying side", and this one is the generous one.

Note `K = 1` is structurally different: there is no fold, so a "proof of
inclusion" is just the party's own four commitments.

Both curves are essentially independent of `n` (see the annotation), so one `n`
is plotted rather than five near-identical panels.

Input: CSV from `cargo run --release --example bench_verifier_csv`.

Usage:
    python scripts/plot_verifier_aggregate.py results/results_verifier.csv
    python scripts/plot_verifier_aggregate.py results/results_verifier.csv --n 16384
"""

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Categorical slots 1-2, matching the other figures: blue is the
# non-amortized baseline, orange the amortized path.
C_BASE, C_AMORT = "#2a78d6", "#eb6834"
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
    ap.add_argument("csv", help="CSV from examples/bench_verifier_csv")
    ap.add_argument("--out", default="results/verifier_aggregate.pdf")
    ap.add_argument("--n", type=int, default=None,
                    help="which n to plot (default: largest in the CSV)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"n", "k", "v_stmt_prep_ms", "v_fold_ms", "v_discharge_ms",
              "v_rokp_ms", "v_lincheck_ms"}
    missing = needed - set(df.columns)
    if missing:
        print(f"CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df.copy()
    df["inclusion_ms"] = (df["v_stmt_prep_ms"] + df["v_fold_ms"]
                          + df["v_discharge_ms"])
    df["baseline_ms"] = df["k"] * df["v_lincheck_ms"]
    df["amortized_ms"] = (df["k"] * df["inclusion_ms"]
                          + df["v_rokp_ms"] + df["v_lincheck_ms"])
    df["overhead"] = df["amortized_ms"] / df["baseline_ms"]

    ns = sorted(df["n"].unique())
    n_star = args.n if args.n is not None else ns[-1]
    sub = df[df["n"] == n_star].sort_values("k")
    if sub.empty:
        print(f"no data for n={n_star}", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # -- (a) absolute total verification work --------------------------------
    ax = axes[0]
    ax.plot(sub["k"], sub["baseline_ms"], "o-", color=C_BASE,
            linewidth=2, markersize=5, label="$K$ independent lincheck verifies")
    ax.plot(sub["k"], sub["amortized_ms"], "s-", color=C_AMORT,
            linewidth=2, markersize=5,
            label="$K$ inclusions $+$ 1 RokP $+$ 1 lincheck")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$K$ (number of jobs)")
    ax.set_ylabel("total verification time (ms)")
    ax.set_title(f"(a) Aggregate verification work, {n_label(n_star)}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.7)
    style(ax)

    # -- (b) overhead factor --------------------------------------------------
    ax = axes[1]
    ax.plot(sub["k"], sub["overhead"], "s-", color=C_AMORT,
            linewidth=2, markersize=5)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.annotate("break-even", (sub["k"].iloc[-1], 1.0),
                textcoords="offset points", xytext=(-2, 5),
                fontsize=8, color=INK_MUTED, ha="right")
    # The ratio is  I(K)/L + (R + L)/(K·L):  the fixed RokP + lincheck tail
    # washes out, leaving the per-job ratio I(K)/L. That is not an asymptote —
    # the inclusion cost I(K) itself grows with log K, so the ratio keeps
    # climbing. The rule marks where the per-job term sits at the largest K.
    last = sub.iloc[-1]
    per_job = last["inclusion_ms"] / last["v_lincheck_ms"]
    ax.axhline(per_job, color=C_AMORT, linewidth=1.0, linestyle=":", alpha=0.7)
    ax.annotate(
        f"per-job inclusion / lincheck verify $= {per_job:.1f}\\times$ "
        f"at $K = {int(last['k'])}$; grows as $\\log K$",
        (sub["k"].iloc[0], per_job), textcoords="offset points",
        xytext=(2, 5), fontsize=8, color=C_AMORT,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$K$ (number of jobs)")
    ax.set_ylabel("amortized $/$ baseline ($\\times$)")
    ax.set_title("(b) What amortization costs the verifying side",
                 fontsize=10, color=INK, loc="left")
    ax.set_ylim(bottom=0)
    style(ax)

    # Both curves are flat in n; say so rather than drawing five panels.
    at_max_k = df[df["k"] == df["k"].max()]
    if len(at_max_k) > 1:
        spread = ((at_max_k["amortized_ms"].max() - at_max_k["amortized_ms"].min())
                  / at_max_k["amortized_ms"].min())
        axes[1].annotate(
            f"independent of $n$: agree to {spread * 100:.0f}% across "
            f"$2^{{{int(ns[0]).bit_length() - 1}}}\\!-\\!"
            f"2^{{{int(ns[-1]).bit_length() - 1}}}$",
            (0.97, 0.06), xycoords="axes fraction", ha="right",
            fontsize=8.5, color=INK_MUTED,
        )

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
