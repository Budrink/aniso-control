"""
Fusion viability analysis — plots for E_target sweep and 2D phase diagram.
Shows the fundamental tokamak constraint: maintain temperature while controlling.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

FIG_DIR = Path("figures/fusion")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CTRL_COLORS = {
    "Proportional":   "#e74c3c",
    "AnisoAware":     "#2ecc71",
    "EventTriggered": "#3498db",
    "PID":            "#9b59b6",
}
CTRL_MARKERS = {
    "Proportional":   "o",
    "AnisoAware":     "s",
    "EventTriggered": "D",
    "PID":            "X",
}


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# ============================================================
#  1. Fusion Margin vs E_target (1D sweep)
# ============================================================
def plot_fusion_margin(csv_path="sweep_fusion.csv"):
    print("\n=== 1: Fusion Margin vs E_target ===")
    df = load(csv_path)
    pcol = df.columns[0]
    ctrls = df["controller"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Fusion margin
    ax = axes[0, 0]
    for ctrl in ctrls:
        d = df[df["controller"] == ctrl].sort_values(pcol)
        color = CTRL_COLORS.get(ctrl, "gray")
        marker = CTRL_MARKERS.get(ctrl, "o")
        ax.plot(d[pcol], d["fusion_margin"], color=color, marker=marker,
                ms=5, label=ctrl)
    ax.axhline(1.0, color="k", ls="--", lw=1.5, alpha=0.7, label="Fusion threshold")
    ax.fill_between(df[pcol].unique(), 0, 1, alpha=0.08, color="red")
    ax.set_xlabel("Target energy E_target")
    ax.set_ylabel("Fusion margin (E_core / E_target)")
    ax.set_title("(a) Fusion viability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # (b) Control effort
    ax = axes[0, 1]
    for ctrl in ctrls:
        d = df[df["controller"] == ctrl].sort_values(pcol)
        color = CTRL_COLORS.get(ctrl, "gray")
        marker = CTRL_MARKERS.get(ctrl, "o")
        ax.plot(d[pcol], d["avg_effort"], color=color, marker=marker,
                ms=5, label=ctrl)
    ax.set_xlabel("Target energy E_target")
    ax.set_ylabel("Average control effort |u|")
    ax.set_title("(b) Control cost")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Core energy vs target
    ax = axes[1, 0]
    targets = sorted(df[pcol].unique())
    ax.plot(targets, targets, "k--", lw=1.5, alpha=0.5, label="E = E_target (ideal)")
    for ctrl in ctrls:
        d = df[df["controller"] == ctrl].sort_values(pcol)
        color = CTRL_COLORS.get(ctrl, "gray")
        marker = CTRL_MARKERS.get(ctrl, "o")
        ax.plot(d[pcol], d["avg_E_core"], color=color, marker=marker,
                ms=5, label=ctrl)
    ax.set_xlabel("Target energy E_target")
    ax.set_ylabel("Actual core energy E_core")
    ax.set_title("(c) Temperature tracking")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Barrier anisotropy (barrier health)
    ax = axes[1, 1]
    for ctrl in ctrls:
        d = df[df["controller"] == ctrl].sort_values(pcol)
        color = CTRL_COLORS.get(ctrl, "gray")
        marker = CTRL_MARKERS.get(ctrl, "o")
        ax.plot(d[pcol], d["avg_barrier_aniso"], color=color, marker=marker,
                ms=5, label=ctrl)
    ax.set_xlabel("Target energy E_target")
    ax.set_ylabel("Barrier anisotropy")
    ax.set_title("(d) Barrier health")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fusion Constraint: Reactor must maintain E >= E_target",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fusion_margin.png", dpi=150)
    plt.close(fig)
    print("  saved fusion_margin.png")


# ============================================================
#  2. 2D Phase Diagram: E_target vs eta_ctrl
# ============================================================
def plot_fusion_phase(csv_path="sweep_fusion_2d.csv"):
    print("\n=== 2: Fusion Phase Diagram (E_target vs eta_ctrl) ===")
    df = load(csv_path)
    p1 = df.columns[0]  # E_target
    p2 = df.columns[1]  # grid.eta_ctrl

    ctrls = df["controller"].unique()
    v1 = sorted(df[p1].unique())
    v2 = sorted(df[p2].unique())
    n1, n2 = len(v1), len(v2)

    fig, axes = plt.subplots(1, len(ctrls), figsize=(5 * len(ctrls), 5))
    if len(ctrls) == 1:
        axes = [axes]

    for ax, ctrl in zip(axes, ctrls):
        sub = df[df["controller"] == ctrl]
        Z_fm = np.full((n2, n1), np.nan)
        Z_uh = np.full((n2, n1), np.nan)
        for _, row in sub.iterrows():
            i1 = v1.index(row[p1])
            i2 = v2.index(row[p2])
            Z_fm[i2, i1] = row["fusion_margin"]
            Z_uh[i2, i1] = row["underheat"]

        im = ax.imshow(Z_fm, origin="lower", aspect="auto",
                       extent=[v1[0], v1[-1], v2[0], v2[-1]],
                       cmap="RdYlGn", vmin=0.5, vmax=1.5)

        # Overlay underheat boundary
        if not np.all(np.isnan(Z_uh)):
            ax.contour(Z_uh, levels=[0.5], colors=["red"], linewidths=2,
                       extent=[v1[0], v1[-1], v2[0], v2[-1]])

        ax.set_xlabel("E_target")
        ax.set_ylabel("eta_ctrl (waste heat fraction)")
        ax.set_title(ctrl)
        plt.colorbar(im, ax=ax, label="Fusion margin", shrink=0.8)

    fig.suptitle("Fusion Viability Map: green=viable, red=underheat",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fusion_phase.png", dpi=150)
    plt.close(fig)
    print("  saved fusion_phase.png")


# ============================================================
#  3. Effort vs Fusion Margin scatter (Pareto front)
# ============================================================
def plot_pareto(csv_path="sweep_fusion.csv"):
    print("\n=== 3: Effort vs Fusion Margin (Pareto) ===")
    df = load(csv_path)
    pcol = df.columns[0]
    ctrls = df["controller"].unique()

    fig, ax = plt.subplots(figsize=(10, 7))
    for ctrl in ctrls:
        d = df[df["controller"] == ctrl]
        color = CTRL_COLORS.get(ctrl, "gray")
        marker = CTRL_MARKERS.get(ctrl, "o")
        sc = ax.scatter(d["avg_effort"], d["fusion_margin"],
                        c=d[pcol], cmap="viridis",
                        marker=marker, s=60, alpha=0.8, edgecolors=color,
                        linewidths=1.5, label=ctrl)

    ax.axhline(1.0, color="red", ls="--", lw=1.5, alpha=0.7,
               label="Fusion threshold")
    ax.fill_between([0, 5], 0, 1, alpha=0.06, color="red")
    ax.set_xlabel("Control effort |u|", fontsize=12)
    ax.set_ylabel("Fusion margin (E_core / E_target)", fontsize=12)
    ax.set_title("The Tokamak Dilemma: Effort vs Fusion Viability", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("E_target")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fusion_pareto.png", dpi=150)
    plt.close(fig)
    print("  saved fusion_pareto.png")


# ============================================================
#  4. Underheat + Disruption combined status
# ============================================================
def plot_outcome_map(csv_path="sweep_fusion_2d.csv"):
    print("\n=== 4: Outcome Map (underheat / stable / disruption) ===")
    df = load(csv_path)
    p1 = df.columns[0]
    p2 = df.columns[1]
    ctrls = df["controller"].unique()
    v1 = sorted(df[p1].unique())
    v2 = sorted(df[p2].unique())
    n1, n2 = len(v1), len(v2)

    cmap = mcolors.ListedColormap(["#e74c3c", "#f39c12", "#2ecc71"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, len(ctrls), figsize=(5 * len(ctrls), 5))
    if len(ctrls) == 1:
        axes = [axes]

    for ax, ctrl in zip(axes, ctrls):
        sub = df[df["controller"] == ctrl]
        Z = np.full((n2, n1), np.nan)
        for _, row in sub.iterrows():
            i1 = v1.index(row[p1])
            i2 = v2.index(row[p2])
            if row["disruption"]:
                Z[i2, i1] = 0    # disruption (red)
            elif row["underheat"]:
                Z[i2, i1] = 1    # underheat (orange)
            else:
                Z[i2, i1] = 2    # stable + hot enough (green)

        im = ax.imshow(Z, origin="lower", aspect="auto",
                       extent=[v1[0], v1[-1], v2[0], v2[-1]],
                       cmap=cmap, norm=norm)
        ax.set_xlabel("E_target")
        ax.set_ylabel("eta_ctrl")
        ax.set_title(ctrl)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Stable + Fusion OK"),
        Patch(facecolor="#f39c12", label="Stable but Underheat"),
        Patch(facecolor="#e74c3c", label="Disruption"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Reactor Outcome: Three failure modes",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fusion_outcome.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fusion_outcome.png")


# ============================================================
#  5. Heater Strategy Comparison
# ============================================================
def plot_heater_comparison(csv_target="sweep_fusion.csv",
                           csv_pulsed="sweep_fusion_pulsed.csv",
                           csv_adaptive="sweep_fusion_adaptive.csv"):
    print("\n=== 5: Heater Strategy Comparison ===")
    dfs = {}
    for name, path in [("Target", csv_target),
                        ("Pulsed", csv_pulsed),
                        ("Adaptive", csv_adaptive)]:
        try:
            dfs[name] = load(path)
        except FileNotFoundError:
            print(f"  skipping {name}: {path} not found")

    if len(dfs) < 2:
        print("  need at least 2 heater CSVs")
        return

    HEATER_STYLES = {
        "Target":   {"color": "#2ecc71", "marker": "o", "ls": "-"},
        "Pulsed":   {"color": "#e67e22", "marker": "s", "ls": "--"},
        "Adaptive": {"color": "#9b59b6", "marker": "D", "ls": "-."},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # For target heater: x-axis is E_target, values vary
    # For pulsed/adaptive: values are constant (E_target doesn't affect heater)
    # Plot everything, but for pulsed/adaptive show as horizontal lines

    ctrls_to_show = ["Proportional", "AnisoAware"]

    # (a) Barrier health vs avg_E — the key comparison
    ax = axes[0, 0]
    for hname, df in dfs.items():
        st = HEATER_STYLES[hname]
        for ctrl in ctrls_to_show:
            d = df[df["controller"] == ctrl]
            ccolor = CTRL_COLORS.get(ctrl, "gray")
            ax.scatter(d["avg_E"].astype(float), d["avg_barrier_aniso"].astype(float),
                       color=ccolor, marker=st["marker"], s=50, alpha=0.8,
                       edgecolors="k", linewidths=0.5,
                       label=f"{ctrl} + {hname}")
    ax.set_xlabel("Average energy E")
    ax.set_ylabel("Barrier anisotropy")
    ax.set_title("(a) Barrier health vs Energy (lower=healthier at same E)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (b) Control effort vs avg_E
    ax = axes[0, 1]
    for hname, df in dfs.items():
        st = HEATER_STYLES[hname]
        for ctrl in ctrls_to_show:
            d = df[df["controller"] == ctrl]
            ccolor = CTRL_COLORS.get(ctrl, "gray")
            ax.scatter(d["avg_E"].astype(float), d["avg_effort"].astype(float),
                       color=ccolor, marker=st["marker"], s=50, alpha=0.8,
                       edgecolors="k", linewidths=0.5,
                       label=f"{ctrl} + {hname}")
    ax.set_xlabel("Average energy E")
    ax.set_ylabel("Control effort |u|")
    ax.set_title("(b) Control cost vs Energy")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (c) State error vs avg_E
    ax = axes[1, 0]
    for hname, df in dfs.items():
        st = HEATER_STYLES[hname]
        for ctrl in ctrls_to_show:
            d = df[df["controller"] == ctrl]
            ccolor = CTRL_COLORS.get(ctrl, "gray")
            ax.scatter(d["avg_E"].astype(float), d["avg_x"].astype(float),
                       color=ccolor, marker=st["marker"], s=50, alpha=0.8,
                       edgecolors="k", linewidths=0.5,
                       label=f"{ctrl} + {hname}")
    ax.set_xlabel("Average energy E")
    ax.set_ylabel("State error |x|")
    ax.set_title("(c) Stability vs Energy")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (d) Bar chart: at comparable energy, barrier health per strategy
    ax = axes[1, 1]
    # For target heater at E_target=2.5 (closest to adaptive's avg_E)
    bars_data = []
    for ctrl in ctrls_to_show:
        for hname, df in dfs.items():
            d = df[df["controller"] == ctrl]
            if hname == "Target":
                pcol = d.columns[0]
                closest = d.iloc[(d["avg_E"].astype(float) - 2.5).abs().argsort()[:1]]
                if not closest.empty:
                    bars_data.append((f"{ctrl}\n{hname}",
                                     float(closest["avg_barrier_aniso"].values[0]),
                                     float(closest["avg_effort"].values[0]),
                                     float(closest["avg_E"].values[0])))
            else:
                row = d.iloc[0]
                bars_data.append((f"{ctrl}\n{hname}",
                                  float(row["avg_barrier_aniso"]),
                                  float(row["avg_effort"]),
                                  float(row["avg_E"])))

    if bars_data:
        labels = [b[0] for b in bars_data]
        barrier_vals = [b[1] for b in bars_data]
        colors = []
        for b in bars_data:
            if "Proportional" in b[0]:
                colors.append("#e74c3c")
            else:
                colors.append("#2ecc71")
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, barrier_vals, color=colors, alpha=0.8, edgecolor="k")
        for i, b in enumerate(bars_data):
            ax.text(i, barrier_vals[i] + 0.05,
                    f"E={b[3]:.1f}\nu={b[2]:.2f}",
                    ha="center", fontsize=7, va="bottom")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Barrier anisotropy")
        ax.set_title("(d) Barrier degradation by strategy")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Heater Strategy: Target vs Pulsed vs Adaptive Pulsed",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "heater_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved heater_comparison.png")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

    plot_fusion_margin("sweep_fusion.csv")
    plot_pareto("sweep_fusion.csv")
    plot_fusion_phase("sweep_fusion_2d.csv")
    plot_outcome_map("sweep_fusion_2d.csv")
    plot_heater_comparison()

    print("\nAll fusion plots saved to", FIG_DIR)
