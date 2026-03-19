"""
Overnight sweep analysis — run after all sweeps complete.
Generates publication-quality plots for nonlinear regimes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from matplotlib.patches import Patch

FIG_DIR = Path("figures/overnight")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CTRL_COLORS = {
    "Proportional":   "#e74c3c",
    "AnisoAware":     "#2ecc71",
    "EventTriggered": "#3498db",
    "PID":            "#9b59b6",
}


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# ============================================================
#  A. Disruption Phase Diagram (2D)
# ============================================================
def plot_disruption_map(csv_path="overnight_disruption_map.csv"):
    print("\n=== A: Disruption Phase Diagram ===")
    df = load(csv_path)
    p1 = df.columns[0]  # heater.power
    p2 = df.columns[1]  # gamma_diss

    ctrls = df["controller"].unique()
    v1 = sorted(df[p1].unique())
    v2 = sorted(df[p2].unique())
    n1, n2 = len(v1), len(v2)

    for metric, label, cmap, vmin, vmax in [
        ("avg_E",             "Energy",        "inferno",   None, None),
        ("avg_x",             "Tracking |x|",  "magma",     None, None),
        ("avg_barrier_aniso", "Barrier Aniso",  "viridis",   None, None),
        ("avg_wall_flux",     "Wall Flux",      "hot",       None, None),
        ("avg_confinement",   "Confinement",    "coolwarm",  None, None),
    ]:
        fig, axes = plt.subplots(1, len(ctrls), figsize=(5*len(ctrls), 4.5))
        if len(ctrls) == 1:
            axes = [axes]

        all_vals = []
        for ci, ctrl in enumerate(ctrls):
            sub = df[df["controller"] == ctrl]
            grid = np.full((n2, n1), np.nan)
            for _, row in sub.iterrows():
                i1 = v1.index(row[p1])
                i2 = v2.index(row[p2])
                grid[i2, i1] = row[metric] if metric in row else 0
            all_vals.append(grid)

        global_min = min(np.nanmin(g) for g in all_vals)
        global_max = max(np.nanmax(g) for g in all_vals)

        for ci, ctrl in enumerate(ctrls):
            ax = axes[ci]
            im = ax.imshow(all_vals[ci], origin="lower", aspect="auto",
                           extent=[v1[0], v1[-1], v2[0], v2[-1]],
                           cmap=cmap, vmin=global_min, vmax=global_max)
            # Overlay disruption boundary
            sub = df[df["controller"] == ctrl]
            disrupt_pts = sub[sub["disruption"] == 1]
            if not disrupt_pts.empty:
                ax.scatter(disrupt_pts[p1], disrupt_pts[p2],
                           marker="x", c="white", s=20, alpha=0.7)
            breakdown_pts = sub[sub["breakdown"] == 1]
            if not breakdown_pts.empty:
                ax.scatter(breakdown_pts[p1], breakdown_pts[p2],
                           marker="X", c="red", s=30, alpha=0.9, edgecolors="white", linewidths=0.5)
            ax.set_xlabel("Heater Power")
            ax.set_ylabel("gamma_diss" if ci == 0 else "")
            ax.set_title(ctrl)
            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(f"Disruption Map: {label}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = f"disruption_map_{metric.replace('avg_','')}"
        fig.savefig(FIG_DIR / f"{fname}.png", dpi=150)
        plt.close(fig)
        print(f"  saved {fname}.png")

    # Stability comparison: overlay all controllers
    fig, ax = plt.subplots(figsize=(8, 6))
    for ctrl in ctrls:
        sub = df[df["controller"] == ctrl]
        bd = sub[sub["breakdown"] == 1]
        ok = sub[sub["breakdown"] == 0]
        color = CTRL_COLORS.get(ctrl, "gray")
        ax.scatter(ok[p1], ok[p2], c=color, marker="o", s=15, alpha=0.5, label=f"{ctrl} OK")
        ax.scatter(bd[p1], bd[p2], c=color, marker="X", s=40, alpha=0.9,
                   edgecolors="black", linewidths=0.5, label=f"{ctrl} FAIL")
    ax.set_xlabel("Heater Power")
    ax.set_ylabel("gamma_diss (cooling rate)")
    ax.set_title("Disruption Boundary: Controller Comparison")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "disruption_boundary.png", dpi=150)
    plt.close(fig)
    print("  saved disruption_boundary.png")


# ============================================================
#  B. Thermal Runaway (1D)
# ============================================================
def plot_thermal_runaway(csv_path="overnight_thermal_runaway.csv"):
    print("\n=== B: Thermal Runaway ===")
    df = load(csv_path)
    pcol = df.columns[0]  # resolution.l0

    metrics = [
        ("avg_E",             "Energy",        "Energy vs Resolution Coupling"),
        ("avg_effort",        "Control Effort", "Effort vs Resolution Coupling"),
        ("avg_x",             "Tracking |x|",  "Tracking Error vs Resolution Coupling"),
        ("avg_wall_flux",     "Wall Flux",      "Wall Flux vs Resolution Coupling"),
        ("avg_confinement",   "Confinement",    "Confinement vs Resolution Coupling"),
        ("avg_barrier_aniso", "Barrier Aniso",  "Barrier Strength vs Resolution Coupling"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ctrls = df["controller"].unique()

    for mi, (col, ylabel, title) in enumerate(metrics):
        ax = axes.flat[mi]
        for ctrl in ctrls:
            s = df[df["controller"] == ctrl].sort_values(pcol)
            color = CTRL_COLORS.get(ctrl, "gray")
            # Mark breakdown points
            ok = s[s["breakdown"] == 0]
            bd = s[s["breakdown"] == 1]
            ax.plot(ok[pcol], ok[col], color=color, marker="o", ms=4, label=ctrl)
            if not bd.empty:
                ax.scatter(bd[pcol], bd[col], color=color, marker="X", s=60,
                           edgecolors="black", linewidths=0.5, zorder=5)
        ax.set_xlabel("Resolution coupling l0")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if mi == 0:
            ax.legend()

    fig.suptitle("Thermal Runaway: Sweep l0 (aggressive melt regime)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "thermal_runaway.png", dpi=150)
    plt.close(fig)
    print("  saved thermal_runaway.png")

    # Energy ratio plot: all controllers vs AnisoAware
    fig, ax = plt.subplots(figsize=(10, 6))
    aniso = df[df["controller"] == "AnisoAware"].sort_values(pcol)
    if not aniso.empty:
        for ctrl in ctrls:
            if ctrl == "AnisoAware":
                continue
            other = df[df["controller"] == ctrl].sort_values(pcol)
            if other.empty:
                continue
            merged = pd.merge(other[[pcol, "avg_E", "avg_effort", "breakdown"]],
                              aniso[[pcol, "avg_E", "avg_effort", "breakdown"]],
                              on=pcol, suffixes=("_other", "_aniso"))
            ratio_E = merged["avg_E_other"] / merged["avg_E_aniso"].clip(lower=0.01)
            color = CTRL_COLORS.get(ctrl, "gray")
            marker = "X" if ctrl == "PID" else "o" if ctrl == "Proportional" else "s"
            ax.plot(merged[pcol], ratio_E, color=color, marker=marker, ms=4,
                    label=f"{ctrl} / AnisoAware")
        ax.axhline(1, color="k", ls=":", alpha=0.5)
        ax.set_xlabel("Resolution coupling l0")
        ax.set_ylabel("Energy Ratio vs AnisoAware")
        ax.set_title("Controller Performance Divergence (Energy)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thermal_divergence.png", dpi=150)
    plt.close(fig)
    print("  saved thermal_divergence.png")


# ============================================================
#  C. Pulsed Critical (2D)
# ============================================================
def plot_pulsed_critical(csv_path="overnight_pulsed_critical.csv"):
    print("\n=== C: Pulsed Heater in Critical Zone ===")
    df = load(csv_path)
    p1 = df.columns[0]  # heater.power
    p2 = df.columns[1]  # heater.duty
    ctrls = df["controller"].unique()

    v1 = sorted(df[p1].unique())
    v2 = sorted(df[p2].unique())

    # Stability map: green=ok, red=disrupted
    fig, axes = plt.subplots(1, len(ctrls), figsize=(5*len(ctrls), 5))
    if len(ctrls) == 1:
        axes = [axes]

    for ci, ctrl in enumerate(ctrls):
        ax = axes[ci]
        sub = df[df["controller"] == ctrl]
        grid = np.full((len(v2), len(v1)), np.nan)
        for _, row in sub.iterrows():
            i1 = v1.index(row[p1])
            i2 = v2.index(row[p2])
            # Color by status: 0=ok, 1=disruption, 2=breakdown
            val = 0
            if row.get("disruption", 0) == 1:
                val = 1
            if row.get("breakdown", 0) == 1:
                val = 2
            grid[i2, i1] = val

        cmap = mcolors.ListedColormap(["#2ecc71", "#f39c12", "#e74c3c"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       extent=[v1[0], v1[-1], v2[0], v2[-1]],
                       cmap=cmap, norm=norm)
        ax.set_xlabel("Heater Power")
        ax.set_ylabel("Duty cycle" if ci == 0 else "")
        ax.set_title(ctrl)

    legend_elements = [Patch(facecolor="#2ecc71", label="OK"),
                       Patch(facecolor="#f39c12", label="Disruption risk"),
                       Patch(facecolor="#e74c3c", label="Breakdown")]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=8)
    fig.suptitle("Pulsed Heater: Stability Map (power vs duty)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "pulsed_stability_map.png", dpi=150)
    plt.close(fig)
    print("  saved pulsed_stability_map.png")

    # Energy heatmap
    for metric, label in [("avg_E", "Energy"), ("avg_barrier_aniso", "Barrier")]:
        fig, axes2 = plt.subplots(1, len(ctrls), figsize=(5*len(ctrls), 4.5))
        if len(ctrls) == 1:
            axes2 = [axes2]

        all_grids = []
        for ctrl in ctrls:
            sub = df[df["controller"] == ctrl]
            grid = np.full((len(v2), len(v1)), np.nan)
            for _, row in sub.iterrows():
                i1 = v1.index(row[p1])
                i2 = v2.index(row[p2])
                grid[i2, i1] = row[metric]
            all_grids.append(grid)
        gmin = min(np.nanmin(g) for g in all_grids)
        gmax = max(np.nanmax(g) for g in all_grids)

        for ci, ctrl in enumerate(ctrls):
            ax = axes2[ci]
            im = ax.imshow(all_grids[ci], origin="lower", aspect="auto",
                           extent=[v1[0], v1[-1], v2[0], v2[-1]],
                           cmap="inferno", vmin=gmin, vmax=gmax)
            ax.set_xlabel("Heater Power")
            ax.set_ylabel("Duty cycle" if ci == 0 else "")
            ax.set_title(ctrl)
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"Pulsed Heater: {label}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = f"pulsed_critical_{metric.replace('avg_','')}"
        fig.savefig(FIG_DIR / f"{fname}.png", dpi=150)
        plt.close(fig)
        print(f"  saved {fname}.png")


# ============================================================
#  D. Landau Phase Transition (1D)
# ============================================================
def plot_landau(csv_path="overnight_landau.csv"):
    print("\n=== D: Landau Phase Transition ===")
    df = load(csv_path)
    pcol = df.columns[0]
    ctrls = df["controller"].unique()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metrics = [
        ("avg_E",             "Energy"),
        ("avg_effort",        "Control Effort"),
        ("avg_x",             "Tracking |x|"),
        ("avg_barrier_aniso", "Barrier Aniso"),
        ("avg_confinement",   "Confinement"),
        ("avg_wall_flux",     "Wall Flux"),
    ]
    for mi, (col, ylabel) in enumerate(metrics):
        ax = axes.flat[mi]
        for ctrl in ctrls:
            s = df[df["controller"] == ctrl].sort_values(pcol)
            color = CTRL_COLORS.get(ctrl, "gray")
            ok = s[s["breakdown"] == 0]
            bd = s[s["breakdown"] == 1]
            ax.plot(ok[pcol], ok[col], color=color, marker="o", ms=3, label=ctrl)
            if not bd.empty:
                ax.scatter(bd[pcol], bd[col], color=color, marker="X", s=50,
                           edgecolors="black", linewidths=0.5)
        ax.set_xlabel("Heater Power")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if mi == 0:
            ax.legend()

    fig.suptitle("Landau Phase Transition: Heater Power Sweep", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "landau_transition.png", dpi=150)
    plt.close(fig)
    print("  saved landau_transition.png")


# ============================================================
#  E. Adaptive Critical (2D)
# ============================================================
def plot_adaptive_critical(csv_path="overnight_adaptive_critical.csv"):
    print("\n=== E: Adaptive Heater Phase Diagram ===")
    df = load(csv_path)
    p1 = df.columns[0]  # heater.power
    p2 = df.columns[1]  # resolution.l0
    ctrls = df["controller"].unique()

    v1 = sorted(df[p1].unique())
    v2 = sorted(df[p2].unique())

    # Stability map
    fig, axes = plt.subplots(1, len(ctrls), figsize=(5*len(ctrls), 5))
    if len(ctrls) == 1:
        axes = [axes]
    for ci, ctrl in enumerate(ctrls):
        ax = axes[ci]
        sub = df[df["controller"] == ctrl]
        grid = np.full((len(v2), len(v1)), np.nan)
        for _, row in sub.iterrows():
            i1 = v1.index(row[p1])
            i2 = v2.index(row[p2])
            grid[i2, i1] = row["avg_E"]
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       extent=[v1[0], v1[-1], v2[0], v2[-1]],
                       cmap="inferno")
        # Overlay disruptions
        disrupt = sub[sub["disruption"] == 1]
        if not disrupt.empty:
            ax.scatter(disrupt[p1], disrupt[p2], marker="x", c="cyan", s=20, alpha=0.7)
        breakdown = sub[sub["breakdown"] == 1]
        if not breakdown.empty:
            ax.scatter(breakdown[p1], breakdown[p2], marker="X", c="red", s=30,
                       edgecolors="white", linewidths=0.5)
        ax.set_xlabel("Heater Power")
        ax.set_ylabel("Resolution l0" if ci == 0 else "")
        ax.set_title(ctrl)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Energy")

    fig.suptitle("Adaptive Heater: Energy Map (power vs l0)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "adaptive_critical_energy.png", dpi=150)
    plt.close(fig)
    print("  saved adaptive_critical_energy.png")

    # Confinement efficiency
    fig, axes = plt.subplots(1, len(ctrls), figsize=(5*len(ctrls), 5))
    if len(ctrls) == 1:
        axes = [axes]
    for ci, ctrl in enumerate(ctrls):
        ax = axes[ci]
        sub = df[df["controller"] == ctrl]
        grid = np.full((len(v2), len(v1)), np.nan)
        for _, row in sub.iterrows():
            i1 = v1.index(row[p1])
            i2 = v2.index(row[p2])
            e = max(row["avg_E"], 0.01)
            grid[i2, i1] = row.get("avg_confinement", 0) / e
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       extent=[v1[0], v1[-1], v2[0], v2[-1]],
                       cmap="RdYlGn")
        ax.set_xlabel("Heater Power")
        ax.set_ylabel("Resolution l0" if ci == 0 else "")
        ax.set_title(ctrl)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Conf/Energy")

    fig.suptitle("Adaptive Heater: Confinement Efficiency", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "adaptive_critical_efficiency.png", dpi=150)
    plt.close(fig)
    print("  saved adaptive_critical_efficiency.png")


# ============================================================
def main():
    files = {
        "disruption_map":    ("overnight_disruption_map.csv",    plot_disruption_map),
        "thermal_runaway":   ("overnight_thermal_runaway.csv",   plot_thermal_runaway),
        "pulsed_critical":   ("overnight_pulsed_critical.csv",   plot_pulsed_critical),
        "landau":            ("overnight_landau.csv",            plot_landau),
        "adaptive_critical": ("overnight_adaptive_critical.csv", plot_adaptive_critical),
    }

    for name, (csv_path, plot_fn) in files.items():
        try:
            plot_fn(csv_path)
        except FileNotFoundError:
            print(f"\n  SKIP {name}: {csv_path} not found yet")
        except Exception as e:
            print(f"\n  ERROR {name}: {e}")

    print(f"\n\nAll overnight plots saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
