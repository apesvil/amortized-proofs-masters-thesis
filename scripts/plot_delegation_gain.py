#!/usr/bin/env python3
"""
Figure 2 — does the local party gain by delegating?

Server-delegation scenario. A local party with one R_{A,B,C} job either

  (i)  keeps the work and proves it itself, with the Marlin inner sumcheck
       (InnerLincheck::prove) — arkworks-rs/marlin's `prover_third_round` plus
       that round's commit and opening work. It is not charged a separate
       statement-preparation term: the lincheck prover builds `f` over `K`
       regardless and `y = Σ f(κ)` falls out of that table.

  (ii) delegates: commits to its four sparse leaf polynomials, then verifies the
       amortized proof. It never computes `y_A, y_B, y_C` itself; that is what
       it delegates. Two depths are plotted:
         * end-to-end (1 + 2 + 3): proof of inclusion + RokP verify + one
           inner-lincheck verify on the folded R_P claim,
         * inclusion only (1): statement prep + fold path + promise discharge.
       The gap between them is the fixed RokP + lincheck tail, ~7.4 ms, which
       every party pays regardless of K.

Common starting point: both sides begin the moment the verifier samples β
(arkworks `verifier_second_round`), with the claim `t(β)` — this repo's R_P
statement (α, β, y, η) — still to be established. Neither side is charged for
the outer sumcheck, which is identical in both worlds. `RokP::reduce` is not in
the baseline: plain Marlin never runs it — it exists only to reduce a *folded*
root back to a single R_P claim.

Delegating is cheaper wherever the orange curve sits below the blue rule; where
it crosses, the crossing K is marked.

`--reference` adds the no-proof floor `ref_direct_eval_ms`: what a party pays to
just evaluate P_A, P_B, P_C at (α, β) itself, if it has nobody to convince.

Usage:
    python scripts/plot_delegation_gain.py results/results_verifier.csv
    python scripts/plot_delegation_gain.py results/results_verifier.csv --reference
"""

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Categorical slots 1-3.
C_LOCAL, C_DELEG, C_INCL = "#2a78d6", "#eb6834", "#1baf7a"
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
    ap.add_argument("--out", default="results/delegation_gain.pdf")
    ap.add_argument("--reference", action="store_true",
                    help="also draw the no-proof floor (direct evaluation)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"n", "k", "v_stmt_prep_ms", "v_fold_ms", "v_discharge_ms",
              "v_rokp_ms", "v_lincheck_ms", "p_local_lincheck_ms"}
    missing = needed - set(df.columns)
    if missing:
        print(f"CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df.copy()
    # Recomputed from the stages rather than read from v_total_ms, so the plot
    # stays correct if the grouping is ever redrawn. Part 1 (proof of inclusion)
    # carries statement prep: it is what you must do to have something to be
    # included, and at ~1 ms it is 4% of the part either way.
    df["inclusion_ms"] = (df["v_stmt_prep_ms"] + df["v_fold_ms"]
                          + df["v_discharge_ms"])
    df["client_ms"] = (df["inclusion_ms"] + df["v_rokp_ms"]
                       + df["v_lincheck_ms"])
    df["gain"] = df["p_local_lincheck_ms"] / df["client_ms"]

    ns = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(3.5 * len(ns), 4.0),
                             squeeze=False)

    for col, n in enumerate(ns):
        sub = df[df["n"] == n].sort_values("k")
        if sub.empty:
            continue

        # -- absolute cost to the local party --------------------------------
        ax = axes[0, col]
        local = sub["p_local_lincheck_ms"].iloc[0]
        ax.axhline(local, color=C_LOCAL, linewidth=2, linestyle="--",
                   label="prove locally: Marlin inner sumcheck")
        if args.reference and "ref_direct_eval_ms" in sub:
            # A floor, not a series: drawn in muted ink so it reads as a rule.
            ax.axhline(sub["ref_direct_eval_ms"].iloc[0], color=INK_MUTED,
                       linewidth=1.0, linestyle="-.",
                       label="evaluate it yourself (no proof)")
        ax.plot(sub["k"], sub["client_ms"], "o-", color=C_DELEG,
                linewidth=2, markersize=5,
                label="delegate: verify $(a) + (b) + (c)$ (end-to-end)")
        ax.plot(sub["k"], sub["inclusion_ms"], "s--", color=C_INCL,
                linewidth=1.8, markersize=5,
                label="delegate: verify $(a)$ (proof of inclusion)")
        if col == len(ns) - 1:
            ax.annotate("prove locally", (sub["k"].iloc[0], local),
                        textcoords="offset points", xytext=(2, 5),
                        fontsize=8, color=C_LOCAL)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("$K$")
        ax.set_ylabel("local-party time (ms)" if col == 0 else "")
        ax.set_title(n_label(n), fontsize=10, color=INK, loc="left")
        # Headroom, so the baseline rule does not sit on the top spine.
        ax.set_ylim(top=ax.get_ylim()[1] * 1.7)

        # The gain, written inside the gap it measures, at the widest K. When
        # the two lines nearly touch there is no gap to write in, so the label
        # goes above the upper of the pair instead.
        last = sub.iloc[-1]
        gain_last = last["gain"]
        text = (f"{gain_last:.0f}$\\times$" if gain_last >= 10
                else f"{gain_last:.1f}$\\times$")
        if 0.5 < gain_last < 2.0:
            anchor, offset, va = max(local, last["client_ms"]), (-6, 9), "bottom"
        else:
            anchor, offset, va = (local * last["client_ms"]) ** 0.5, (-6, 0), "center"
        ax.annotate(text, (last["k"], anchor), textcoords="offset points",
                    xytext=offset, fontsize=9, color=C_LOCAL,
                    ha="right", va=va)
        # The K where delegating stops paying off, if it does within the sweep
        # — the one thing the dropped ratio row said that the curves do not.
        lost = sub[sub["gain"] < 1.0]
        if not lost.empty:
            k_star = lost["k"].iloc[0]
            ax.axvline(k_star, color=INK_MUTED, linewidth=1.0,
                       linestyle=":", alpha=0.7)
            ax.annotate(f"break-even $K = {int(k_star)}$",
                        (k_star, ax.get_ylim()[1]),
                        textcoords="offset points", xytext=(-4, -6),
                        fontsize=8, color=INK_MUTED, ha="right", va="top")
        style(ax)

    # One legend for the whole figure: a per-panel box collides with the
    # baseline rule.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=8.5)

    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
