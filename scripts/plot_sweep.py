#!/usr/bin/env python3
"""
Publication-quality plots from GridEngine sweep CSV.

Usage:
    python plot_sweep.py grid_sweep.csv [output_dir]

Produces:
    1. Tracking error vs parameter  (per controller)
    2. Energy vs parameter
    3. Degradation tr(G)/Dim vs parameter
    4. Control effort vs parameter
    5. Error-effort trade-off  (Pareto-style)
    6. Breakdown map
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "Proportional":   "#2176AE",
    "AnisoAware":     "#E04040",
    "Pulsed":         "#57A773",
    "EventTriggered": "#E8871E",
}
MARKERS = {
    "Proportional":   "o",
    "AnisoAware":     "s",
    "Pulsed":         "^",
    "EventTriggered": "D",
}

def get_style(name):
    for key in COLORS:
        if key.lower() in name.lower():
            return COLORS[key], MARKERS[key]
    return "#888888", "x"


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_sweep.py <grid_sweep.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    param_col = df.columns[0]
    controllers = df["controller"].unique()

    # ── Figure 1: Tracking error ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for ctrl in controllers:
        sub = df[df["controller"] == ctrl].sort_values(param_col)
        c, m = get_style(ctrl)
        ax.plot(sub[param_col], sub["avg_x"], color=c, marker=m,
                markersize=4, linewidth=1.8, label=ctrl)
        # mark breakdowns
        bd = sub[sub["breakdown"] == 1]
        if len(bd):
            ax.scatter(bd[param_col], bd["avg_x"], color=c,
                       marker="x", s=60, zorder=5, linewidths=2)
    ax.set_xlabel(param_col.replace(".", " → "))
    ax.set_ylabel(r"Mean tracking error $\langle |x| \rangle$")
    ax.set_title("Tracking Error vs Control Parameter")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.savefig(os.path.join(out_dir, "fig1_error.pdf"))
    fig.savefig(os.path.join(out_dir, "fig1_error.png"))
    print(f"  -> fig1_error.pdf")
    plt.close(fig)

    # ── Figure 2: Energy ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for ctrl in controllers:
        sub = df[df["controller"] == ctrl].sort_values(param_col)
        c, m = get_style(ctrl)
        ax.plot(sub[param_col], sub["avg_E"], color=c, marker=m,
                markersize=4, linewidth=1.8, label=ctrl)
    ax.set_xlabel(param_col.replace(".", " → "))
    ax.set_ylabel(r"Mean energy $\langle E \rangle$")
    ax.set_title("Energy Accumulation vs Control Parameter")
    ax.legend(framealpha=0.9)
    fig.savefig(os.path.join(out_dir, "fig2_energy.pdf"))
    fig.savefig(os.path.join(out_dir, "fig2_energy.png"))
    print(f"  -> fig2_energy.pdf")
    plt.close(fig)

    # ── Figure 3: Degradation tr(G) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for ctrl in controllers:
        sub = df[df["controller"] == ctrl].sort_values(param_col)
        c, m = get_style(ctrl)
        ax.plot(sub[param_col], sub["avg_trG"], color=c, marker=m,
                markersize=4, linewidth=1.8, label=ctrl)
    ax.axhline(2.0, color="gray", ls="--", lw=0.8, label=r"$\mathrm{tr}(G) = \mathrm{Dim}$ (pristine)")
    ax.set_xlabel(param_col.replace(".", " → "))
    ax.set_ylabel(r"Mean $\mathrm{tr}(G)$")
    ax.set_title("Tensor Degradation vs Control Parameter")
    ax.legend(framealpha=0.9)
    fig.savefig(os.path.join(out_dir, "fig3_degradation.pdf"))
    fig.savefig(os.path.join(out_dir, "fig3_degradation.png"))
    print(f"  -> fig3_degradation.pdf")
    plt.close(fig)

    # ── Figure 4: Control effort ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for ctrl in controllers:
        sub = df[df["controller"] == ctrl].sort_values(param_col)
        c, m = get_style(ctrl)
        ax.plot(sub[param_col], sub["avg_effort"], color=c, marker=m,
                markersize=4, linewidth=1.8, label=ctrl)
    ax.set_xlabel(param_col.replace(".", " → "))
    ax.set_ylabel(r"Mean control effort $\langle |u| \rangle$")
    ax.set_title("Control Effort vs Parameter")
    ax.legend(framealpha=0.9)
    fig.savefig(os.path.join(out_dir, "fig4_effort.pdf"))
    fig.savefig(os.path.join(out_dir, "fig4_effort.png"))
    print(f"  -> fig4_effort.pdf")
    plt.close(fig)

    # ── Figure 5: Pareto — Error vs Effort ────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for ctrl in controllers:
        sub = df[df["controller"] == ctrl]
        ok = sub[sub["breakdown"] == 0]
        bd = sub[sub["breakdown"] == 1]
        c, m = get_style(ctrl)
        ax.scatter(ok["avg_effort"], ok["avg_x"], color=c, marker=m,
                   s=30, alpha=0.8, label=ctrl, edgecolors="none")
        if len(bd):
            ax.scatter(bd["avg_effort"], bd["avg_x"], color=c, marker="x",
                       s=50, alpha=0.6, linewidths=1.5)
    ax.set_xlabel(r"Mean control effort $\langle |u| \rangle$")
    ax.set_ylabel(r"Mean tracking error $\langle |x| \rangle$")
    ax.set_title("Error–Effort Trade-off (Pareto Front)")
    ax.legend(framealpha=0.9)
    fig.savefig(os.path.join(out_dir, "fig5_pareto.pdf"))
    fig.savefig(os.path.join(out_dir, "fig5_pareto.png"))
    print(f"  -> fig5_pareto.pdf")
    plt.close(fig)

    # ── Figure 6: Combined 2×2 summary ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    titles = [
        (r"$\langle |x| \rangle$", "avg_x"),
        (r"$\langle E \rangle$",   "avg_E"),
        (r"$\mathrm{tr}(G)$",      "avg_trG"),
        (r"$\langle |u| \rangle$", "avg_effort"),
    ]
    for idx, (ylabel, col) in enumerate(titles):
        ax = axes[idx // 2][idx % 2]
        for ctrl in controllers:
            sub = df[df["controller"] == ctrl].sort_values(param_col)
            c, m = get_style(ctrl)
            ax.plot(sub[param_col], sub[col], color=c, marker=m,
                    markersize=3, linewidth=1.5, label=ctrl)
        ax.set_ylabel(ylabel)
        if idx >= 2:
            ax.set_xlabel(param_col.replace(".", " → "))
        if idx == 0:
            ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle("Grid Sweep: Controller Comparison", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "fig6_summary.pdf"))
    fig.savefig(os.path.join(out_dir, "fig6_summary.png"))
    print(f"  -> fig6_summary.pdf")
    plt.close(fig)

    # ── Print best per controller ─────────────────────────────────────
    print("\n=== Best configurations (lowest avg_x, no breakdown) ===")
    ok = df[df["breakdown"] == 0]
    if len(ok):
        for ctrl in controllers:
            sub = ok[ok["controller"] == ctrl]
            if len(sub) == 0:
                print(f"  {ctrl}: ALL FAILED")
                continue
            best = sub.loc[sub["avg_x"].idxmin()]
            print(f"  {ctrl}: {param_col}={best[param_col]:.2f}  "
                  f"avg|x|={best['avg_x']:.4f}  "
                  f"avgE={best['avg_E']:.3f}  "
                  f"effort={best['avg_effort']:.4f}")
    else:
        print("  All configurations broke down!")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
