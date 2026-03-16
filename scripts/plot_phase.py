#!/usr/bin/env python3
"""
Publication-quality phase diagrams from 2D GridEngine sweep CSV.

Usage:
    python plot_phase.py grid_sweep2d.csv [output_dir]

Produces per-controller heatmaps of avg|x|, avg E, and a stability map.
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_phase.py <grid_sweep2d.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    p1_col, p2_col = df.columns[0], df.columns[1]
    controllers = df["controller"].unique()

    p1_label = p1_col.replace(".", " → ")
    p2_label = p2_col.replace(".", " → ")

    # ── Per-controller heatmaps: avg|x| ──────────────────────────────
    n_ctrl = len(controllers)
    fig, axes = plt.subplots(1, n_ctrl, figsize=(4.5 * n_ctrl, 4),
                             squeeze=False)
    axes = axes[0]

    vmin = df[df["breakdown"] == 0]["avg_x"].min() if len(df[df["breakdown"] == 0]) else 0
    vmax = min(df["avg_x"].quantile(0.95), 0.8)

    for idx, ctrl in enumerate(controllers):
        ax = axes[idx]
        sub = df[df["controller"] == ctrl]
        piv = sub.pivot_table(index=p2_col, columns=p1_col,
                              values="avg_x", aggfunc="first")
        im = ax.pcolormesh(piv.columns, piv.index, piv.values,
                           cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                           shading="auto")
        # Overlay breakdown contour
        bd = sub.pivot_table(index=p2_col, columns=p1_col,
                             values="breakdown", aggfunc="first")
        ax.contour(bd.columns, bd.index, bd.values.astype(float),
                   levels=[0.5], colors="black", linewidths=1.5,
                   linestyles="--")
        ax.set_xlabel(p1_label)
        if idx == 0:
            ax.set_ylabel(p2_label)
        else:
            ax.set_yticklabels([])
        ax.set_title(ctrl)

    fig.colorbar(im, ax=axes, label=r"$\langle |x| \rangle$",
                 shrink=0.85, pad=0.02)
    fig.suptitle("Phase Diagram: Tracking Error", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_error.pdf"))
    fig.savefig(os.path.join(out_dir, "phase_error.png"))
    print("  -> phase_error.pdf")
    plt.close(fig)

    # ── Per-controller heatmaps: avg E ───────────────────────────────
    fig, axes = plt.subplots(1, n_ctrl, figsize=(4.5 * n_ctrl, 4),
                             squeeze=False)
    axes = axes[0]
    vmax_e = df["avg_E"].quantile(0.95)

    for idx, ctrl in enumerate(controllers):
        ax = axes[idx]
        sub = df[df["controller"] == ctrl]
        piv = sub.pivot_table(index=p2_col, columns=p1_col,
                              values="avg_E", aggfunc="first")
        im = ax.pcolormesh(piv.columns, piv.index, piv.values,
                           cmap="hot", vmin=0, vmax=vmax_e, shading="auto")
        bd = sub.pivot_table(index=p2_col, columns=p1_col,
                             values="breakdown", aggfunc="first")
        ax.contour(bd.columns, bd.index, bd.values.astype(float),
                   levels=[0.5], colors="cyan", linewidths=1.5,
                   linestyles="--")
        ax.set_xlabel(p1_label)
        if idx == 0:
            ax.set_ylabel(p2_label)
        else:
            ax.set_yticklabels([])
        ax.set_title(ctrl)

    fig.colorbar(im, ax=axes, label=r"$\langle E \rangle$",
                 shrink=0.85, pad=0.02)
    fig.suptitle("Phase Diagram: Energy", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_energy.pdf"))
    fig.savefig(os.path.join(out_dir, "phase_energy.png"))
    print("  -> phase_energy.pdf")
    plt.close(fig)

    # ── Stability map: best controller at each point ─────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5))

    p1_vals = sorted(df[p1_col].unique())
    p2_vals = sorted(df[p2_col].unique())
    best_map = np.full((len(p2_vals), len(p1_vals)), np.nan)
    err_map  = np.full((len(p2_vals), len(p1_vals)), np.nan)

    ctrl_to_id = {c: i for i, c in enumerate(controllers)}

    for i, v2 in enumerate(p2_vals):
        for j, v1 in enumerate(p1_vals):
            pts = df[(df[p1_col] == v1) & (df[p2_col] == v2)]
            ok = pts[pts["breakdown"] == 0]
            if len(ok) > 0:
                best = ok.loc[ok["avg_x"].idxmin()]
                best_map[i, j] = ctrl_to_id[best["controller"]]
                err_map[i, j] = best["avg_x"]
            else:
                best_map[i, j] = -1

    from matplotlib.colors import ListedColormap
    colors = ["#2176AE", "#E04040", "#57A773", "#E8871E",
              "#8855CC", "#CC6688"]
    fail_color = "#333333"
    cmap_list = [fail_color] + colors[:n_ctrl]
    cmap = ListedColormap(cmap_list)

    plot_data = best_map.copy()
    plot_data[plot_data == -1] = -0.5
    for i, c in enumerate(controllers):
        plot_data[best_map == i] = i + 0.5

    im = ax.pcolormesh(p1_vals, p2_vals, plot_data,
                       cmap=cmap, vmin=-1, vmax=n_ctrl,
                       shading="auto")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=fail_color, label="Breakdown")]
    for i, c in enumerate(controllers):
        handles.append(Patch(facecolor=colors[i], label=c))
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_xlabel(p1_label)
    ax.set_ylabel(p2_label)
    ax.set_title("Stability Map: Best Controller per Region")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_stability.pdf"))
    fig.savefig(os.path.join(out_dir, "phase_stability.png"))
    print("  -> phase_stability.pdf")
    plt.close(fig)

    # ── Print summary ────────────────────────────────────────────────
    ok = df[df["breakdown"] == 0]
    print(f"\nTotal points: {len(df)}, OK: {len(ok)}, "
          f"Breakdown: {len(df) - len(ok)}")
    if len(ok):
        best = ok.loc[ok["avg_x"].idxmin()]
        print(f"Global best: {best['controller']}  "
              f"{p1_col}={best[p1_col]:.2f}  {p2_col}={best[p2_col]:.2f}  "
              f"avg|x|={best['avg_x']:.4f}")
    print(f"\nFigures saved to {out_dir}/")


if __name__ == "__main__":
    main()
