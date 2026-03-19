"""
Phase‑diagram atlas — generates heatmaps + stability maps for any 2D sweep CSV.

Usage:
    python plot_atlas.py <csv_file> <output_dir> [--title "Slice title"]
"""

import sys, os, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

NICE = {
    "controller.gain": "Control Intensity (gain)",
    "coupling.alpha":  "Coupling Strength (alpha)",
    "grid.gamma_diss": "Dissipation Rate (gamma)",
    "grid.noise_amp":  "Thermal Noise (sigma)",
    "grid.kappa_tau":  "Anisotropy Self-Protection (kappa_tau)",
    "relaxation.tau":  "Base Relaxation Time (tau_0)",
    "grid.D_E":        "Energy Diffusion (D_E)",
}

CTRL_COLORS = {
    "Proportional":   "#d62728",
    "AnisoAware":     "#2ca02c",
    "EventTriggered": "#1f77b4",
    "Pulsed":         "#ff7f0e",
}

def nice(name):
    return NICE.get(name, name)

def load(csv_path):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    p1_name, p2_name = cols[0], cols[1]
    return df, p1_name, p2_name

def make_heatmaps(df, p1, p2, metric, metric_label, outdir, prefix, title_extra=""):
    ctrls = df["controller"].unique()
    n = len(ctrls)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    fig.suptitle(f"{metric_label}{title_extra}", fontsize=14, fontweight="bold", y=1.02)

    vmin = df.loc[~df["breakdown"].astype(bool), metric].quantile(0.02) if metric in df.columns else 0
    vmax = df.loc[~df["breakdown"].astype(bool), metric].quantile(0.95) if metric in df.columns else 1

    for i, ctrl in enumerate(ctrls):
        ax = axes[0, i]
        sub = df[df["controller"] == ctrl].copy()
        piv = sub.pivot_table(index=p2, columns=p1, values=metric, aggfunc="mean")
        bd  = sub.pivot_table(index=p2, columns=p1, values="breakdown", aggfunc="max")

        X = piv.columns.values.astype(float)
        Y = piv.index.values.astype(float)
        Z = piv.values.astype(float)

        im = ax.pcolormesh(X, Y, Z, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                           shading="nearest")

        if bd is not None:
            Zb = bd.values.astype(float)
            ax.contour(X, Y, Zb, levels=[0.5], colors="black",
                       linewidths=1.5, linestyles="--")
            ax.contourf(X, Y, Zb, levels=[0.5, 1.5], colors=["red"],
                        alpha=0.15)

        ax.set_xlabel(nice(p1), fontsize=10)
        if i == 0:
            ax.set_ylabel(nice(p2), fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.set_title(ctrl, fontsize=11,
                     color=CTRL_COLORS.get(ctrl, "black"), fontweight="bold")

    fig.colorbar(im, ax=axes[0, -1], label=metric_label, shrink=0.85)
    fig.tight_layout()
    path = os.path.join(outdir, f"{prefix}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

def make_stability_map(df, p1, p2, outdir, prefix, title_extra=""):
    ctrls = sorted(df["controller"].unique())
    ctrl_to_id = {c: i for i, c in enumerate(ctrls)}
    n_ctrl = len(ctrls)

    p1_vals = sorted(df[p1].unique())
    p2_vals = sorted(df[p2].unique())
    best = np.full((len(p2_vals), len(p1_vals)), -1, dtype=int)

    for j, v2 in enumerate(p2_vals):
        for i, v1 in enumerate(p1_vals):
            sub = df[(np.isclose(df[p1], v1)) & (np.isclose(df[p2], v2))]
            ok = sub[~sub["breakdown"].astype(bool)]
            if ok.empty:
                best[j, i] = n_ctrl  # all breakdown
            else:
                winner = ok.loc[ok["avg_x"].idxmin(), "controller"]
                best[j, i] = ctrl_to_id[winner]

    colors = [CTRL_COLORS.get(c, "#999999") for c in ctrls] + ["#333333"]
    labels = ctrls + ["Breakdown"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n_ctrl + 1.5), cmap.N)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pcolormesh(p1_vals, p2_vals, best, cmap=cmap, norm=norm, shading="nearest")
    ax.set_xlabel(nice(p1), fontsize=11)
    ax.set_ylabel(nice(p2), fontsize=11)
    ax.set_title(f"Best Controller by Region{title_extra}",
                 fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    patches = [Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=patches, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="gray")

    fig.tight_layout()
    path = os.path.join(outdir, f"{prefix}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

def make_effort_map(df, p1, p2, outdir, prefix, title_extra=""):
    """Heatmap of avg control effort — shows energy cost landscape."""
    make_heatmaps(df, p1, p2, "avg_effort", f"Control Effort{title_extra}",
                  outdir, prefix, "")

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_atlas.py <csv> <outdir> [--title 'Title']")
        sys.exit(1)

    csv_path = sys.argv[1]
    outdir   = sys.argv[2]
    title_extra = ""
    if "--title" in sys.argv:
        idx = sys.argv.index("--title")
        if idx + 1 < len(sys.argv):
            title_extra = f"  —  {sys.argv[idx+1]}"

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    stem = pathlib.Path(csv_path).stem
    df, p1, p2 = load(csv_path)

    print(f"Loaded {len(df)} rows: {p1} x {p2}, "
          f"controllers: {df['controller'].unique().tolist()}")

    print("Generating error heatmaps...")
    make_heatmaps(df, p1, p2, "avg_x", f"Tracking Error (avg|x|){title_extra}",
                  outdir, f"{stem}_error", "")

    print("Generating energy heatmaps...")
    make_heatmaps(df, p1, p2, "avg_E", f"Energy Density (avg E){title_extra}",
                  outdir, f"{stem}_energy", "")

    print("Generating effort heatmaps...")
    make_heatmaps(df, p1, p2, "avg_effort", f"Control Effort{title_extra}",
                  outdir, f"{stem}_effort", "")

    print("Generating stability map...")
    make_stability_map(df, p1, p2, outdir, f"{stem}_stability", title_extra)

    print("Done.\n")

if __name__ == "__main__":
    main()
