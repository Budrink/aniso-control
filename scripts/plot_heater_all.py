"""
Comprehensive heater strategy analysis.
Compares all 6 strategies: Constant, Pulsed, EventDriven (local),
GlobalEvent, AdaptivePulsed — across 3 controllers.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("figures/heater")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CTRL_COLORS = {
    "Proportional":   "#e74c3c",
    "AnisoAware":     "#2ecc71",
    "EventTriggered": "#3498db",
}


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def plot_1d(df, pcol, xlabel, prefix, title=""):
    ctrls = df["controller"].unique()
    metrics = [
        ("avg_E",            "Energy"),
        ("avg_effort",       "Effort"),
        ("avg_x",            "Tracking |x|"),
        ("avg_wall_flux",    "Wall Flux"),
        ("avg_confinement",  "Confinement"),
        ("avg_barrier_aniso","Barrier Aniso"),
    ]
    for col, ylabel in metrics:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for ctrl in ctrls:
            s = df[df["controller"] == ctrl].sort_values(pcol)
            ax.plot(s[pcol], s[col], color=CTRL_COLORS.get(ctrl, "gray"),
                    marker="o", ms=4, label=ctrl)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {ylabel}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{prefix}_{col.replace('avg_','')}.png", dpi=150)
        plt.close(fig)


def extract_point(df, pcol, pval, ctrl):
    row = df[(df["controller"] == ctrl) &
             (np.isclose(df[pcol], pval, atol=0.01))]
    return row.iloc[0] if not row.empty else None


def grand_comparison():
    """Load all CSVs and build a grand comparison bar chart."""
    # Load data
    try:
        df_pulsed = load("sweep_heater_pulsed_hot.csv")
        df_event  = load("sweep_heater_event_hot.csv")
        df_global = load("sweep_heater_global.csv")
        df_adapt  = load("sweep_heater_adaptive.csv")
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    pcol_p = df_pulsed.columns[0]
    pcol_e = df_event.columns[0]
    pcol_g = df_global.columns[0]
    pcol_a = df_adapt.columns[0]

    strategies = [
        ("Constant\n(P=3.0)",        df_pulsed, pcol_p, 1.0),
        ("Pulsed 50%\n(P=3.0)",      df_pulsed, pcol_p, 0.50),
        ("Pulsed 20%\n(P=3.0)",      df_pulsed, pcol_p, 0.20),
        ("Event local\nt=0.10",      df_event,  pcol_e, 0.10),
        ("Global event\nt=2.0",      df_global, pcol_g, 2.00),
        ("Global event\nt=2.25",     df_global, pcol_g, 2.25),
        ("Adaptive\ndmin=0.10",      df_adapt,  pcol_a, 0.10),
        ("Adaptive\ndmin=0.30",      df_adapt,  pcol_a, 0.30),
    ]

    ctrls = ["Proportional", "AnisoAware", "EventTriggered"]
    metrics = [
        ("avg_E",             "Energy"),
        ("avg_effort",        "Effort"),
        ("avg_x",             "Tracking |x|"),
        ("avg_wall_flux",     "Wall Flux"),
        ("avg_confinement",   "Confinement"),
        ("avg_barrier_aniso", "Barrier Aniso"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    x_labels = [s[0] for s in strategies]
    x_pos = np.arange(len(strategies))
    width = 0.25

    for mi, (col, mlabel) in enumerate(metrics):
        ax = axes.flat[mi]
        for ci, ctrl in enumerate(ctrls):
            vals = []
            for _, df, pcol, pval in strategies:
                r = extract_point(df, pcol, pval, ctrl)
                vals.append(r[col] if r is not None and col in r.index else 0)
            ax.bar(x_pos + ci * width, vals, width,
                   label=ctrl if mi == 0 else "",
                   color=CTRL_COLORS.get(ctrl, "gray"), alpha=0.85)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=35, ha="right")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.grid(True, alpha=0.2, axis="y")

    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Grand Heater Strategy Comparison (all P=3.0)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "grand_all.png", dpi=150)
    plt.close(fig)
    print("saved grand_all.png")


def energy_savings_chart():
    """Bar chart: % energy savings vs constant heater for each strategy."""
    try:
        df_pulsed = load("sweep_heater_pulsed_hot.csv")
        df_event  = load("sweep_heater_event_hot.csv")
        df_global = load("sweep_heater_global.csv")
        df_adapt  = load("sweep_heater_adaptive.csv")
    except Exception as e:
        print(f"Error: {e}")
        return

    pcol_p = df_pulsed.columns[0]
    pcol_e = df_event.columns[0]
    pcol_g = df_global.columns[0]
    pcol_a = df_adapt.columns[0]

    ctrls = ["Proportional", "AnisoAware", "EventTriggered"]

    # Baseline: constant heater at P=3.0 (= pulsed duty=1.0)
    base = {}
    for ctrl in ctrls:
        r = extract_point(df_pulsed, pcol_p, 1.0, ctrl)
        base[ctrl] = r["avg_E"] if r is not None else 1

    strategies_pts = [
        ("Pulsed\nduty=50%",  df_pulsed, pcol_p, 0.50),
        ("Pulsed\nduty=20%",  df_pulsed, pcol_p, 0.20),
        ("Pulsed\nduty=10%",  df_pulsed, pcol_p, 0.10),
        ("Event local\nt=0.10", df_event, pcol_e, 0.10),
        ("Global event\nt=2.0", df_global, pcol_g, 2.00),
        ("Adaptive\ndmin=0.10", df_adapt, pcol_a, 0.10),
        ("Adaptive\ndmin=0.20", df_adapt, pcol_a, 0.20),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(strategies_pts))
    width = 0.25

    for ci, ctrl in enumerate(ctrls):
        pcts = []
        for _, df, pcol, pval in strategies_pts:
            r = extract_point(df, pcol, pval, ctrl)
            e = r["avg_E"] if r is not None else base[ctrl]
            pcts.append((1 - e / base[ctrl]) * 100)
        ax.bar(x_pos + ci * width, pcts, width,
               label=ctrl, color=CTRL_COLORS.get(ctrl, "gray"), alpha=0.85)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([s[0] for s in strategies_pts], fontsize=9)
    ax.set_ylabel("Energy savings vs constant heater (%)")
    ax.set_title("Energy Savings by Heater Strategy (P=3.0)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="k", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "savings_all.png", dpi=150)
    plt.close(fig)
    print("saved savings_all.png")


def confinement_efficiency():
    """Confinement per unit energy for each strategy."""
    try:
        df_pulsed = load("sweep_heater_pulsed_hot.csv")
        df_global = load("sweep_heater_global.csv")
        df_adapt  = load("sweep_heater_adaptive.csv")
    except Exception as e:
        print(f"Error: {e}")
        return

    pcol_p = df_pulsed.columns[0]
    pcol_g = df_global.columns[0]
    pcol_a = df_adapt.columns[0]

    ctrls = ["Proportional", "AnisoAware", "EventTriggered"]

    strategies_pts = [
        ("Constant",      df_pulsed, pcol_p, 1.0),
        ("Pulsed 50%",    df_pulsed, pcol_p, 0.50),
        ("Pulsed 20%",    df_pulsed, pcol_p, 0.20),
        ("Global t=2.0",  df_global, pcol_g, 2.00),
        ("Adaptive 0.10", df_adapt,  pcol_a, 0.10),
        ("Adaptive 0.30", df_adapt,  pcol_a, 0.30),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(strategies_pts))
    width = 0.25

    for ci, ctrl in enumerate(ctrls):
        effs = []
        for _, df, pcol, pval in strategies_pts:
            r = extract_point(df, pcol, pval, ctrl)
            if r is not None and "avg_confinement" in r.index:
                effs.append(r["avg_confinement"] / max(r["avg_E"], 0.01))
            else:
                effs.append(0)
        ax.bar(x_pos + ci * width, effs, width,
               label=ctrl, color=CTRL_COLORS.get(ctrl, "gray"), alpha=0.85)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([s[0] for s in strategies_pts], fontsize=9)
    ax.set_ylabel("Confinement / Energy")
    ax.set_title("Confinement Efficiency (higher = better)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confinement_efficiency.png", dpi=150)
    plt.close(fig)
    print("saved confinement_efficiency.png")


def global_event_phase_transition():
    """Show sharp phase transition in global event heater."""
    try:
        df = load("sweep_heater_global.csv")
    except Exception as e:
        print(f"Error: {e}")
        return
    pcol = df.columns[0]
    ctrls = df["controller"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for col, ylabel, ax in [
        ("avg_E", "Energy", axes[0]),
        ("avg_confinement", "Confinement", axes[1]),
        ("avg_barrier_aniso", "Barrier Aniso", axes[2]),
    ]:
        for ctrl in ctrls:
            s = df[df["controller"] == ctrl].sort_values(pcol)
            ax.plot(s[pcol], s[col], color=CTRL_COLORS.get(ctrl, "gray"),
                    marker="o", ms=5, label=ctrl, linewidth=2)
        ax.set_xlabel("Trigger threshold (barrier aniso)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(2.0, color="purple", ls="--", alpha=0.5, label="transition")

    fig.suptitle("Global Event Heater: Phase Transition (P=3.0)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "global_phase_transition.png", dpi=150)
    plt.close(fig)
    print("saved global_phase_transition.png")


def adaptive_vs_fixed():
    """Compare adaptive pulsed vs fixed pulsed at equivalent average duty."""
    try:
        df_fixed = load("sweep_heater_pulsed_hot.csv")
        df_adapt = load("sweep_heater_adaptive.csv")
    except Exception as e:
        print(f"Error: {e}")
        return

    pcol_f = df_fixed.columns[0]
    pcol_a = df_adapt.columns[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for col, ylabel, ax in [
        ("avg_E",             "Energy", axes[0]),
        ("avg_confinement",   "Confinement", axes[1]),
        ("avg_barrier_aniso", "Barrier Aniso", axes[2]),
    ]:
        for ctrl in ["AnisoAware", "Proportional"]:
            sf = df_fixed[df_fixed["controller"] == ctrl].sort_values(pcol_f)
            sa = df_adapt[df_adapt["controller"] == ctrl].sort_values(pcol_a)
            c = CTRL_COLORS.get(ctrl, "gray")
            ax.plot(sf[pcol_f], sf[col], color=c, ls="--", marker="s", ms=3,
                    label=f"{ctrl} fixed")
            ax.plot(sa[pcol_a], sa[col], color=c, ls="-", marker="o", ms=4,
                    label=f"{ctrl} adaptive")
        ax.set_xlabel("Duty / Duty_min")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Adaptive vs Fixed Pulsed Heater (P=3.0)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "adaptive_vs_fixed.png", dpi=150)
    plt.close(fig)
    print("saved adaptive_vs_fixed.png")


def print_summary():
    try:
        df_pulsed = load("sweep_heater_pulsed_hot.csv")
        df_global = load("sweep_heater_global.csv")
        df_adapt  = load("sweep_heater_adaptive.csv")
    except Exception as e:
        print(f"Error: {e}")
        return

    pcol_p = df_pulsed.columns[0]
    pcol_g = df_global.columns[0]
    pcol_a = df_adapt.columns[0]

    ctrls = ["Proportional", "AnisoAware", "EventTriggered"]

    # Constant baseline
    print("\n" + "="*80)
    print("  SUMMARY: All strategies at P=3.0")
    print("="*80)

    print(f"\n{'Strategy':<30s}", end="")
    for ctrl in ctrls:
        print(f"  {ctrl:>14s} E  conf  eff", end="")
    print()
    print("-" * 110)

    pts = [
        ("Constant (duty=1.0)",  df_pulsed, pcol_p, 1.00),
        ("Pulsed duty=0.50",     df_pulsed, pcol_p, 0.50),
        ("Pulsed duty=0.20",     df_pulsed, pcol_p, 0.20),
        ("Pulsed duty=0.10",     df_pulsed, pcol_p, 0.10),
        ("GlobalEvent t=1.75",   df_global, pcol_g, 1.75),
        ("GlobalEvent t=2.00",   df_global, pcol_g, 2.00),
        ("GlobalEvent t=2.25",   df_global, pcol_g, 2.25),
        ("Adaptive dmin=0.10",   df_adapt,  pcol_a, 0.10),
        ("Adaptive dmin=0.20",   df_adapt,  pcol_a, 0.20),
        ("Adaptive dmin=0.30",   df_adapt,  pcol_a, 0.30),
    ]

    for label, df, pcol, pval in pts:
        print(f"{label:<30s}", end="")
        for ctrl in ctrls:
            r = extract_point(df, pcol, pval, ctrl)
            if r is not None:
                e = r["avg_E"]
                c = r.get("avg_confinement", 0)
                eff = c / max(e, 0.01)
                print(f"  {e:7.3f}  {c:5.2f} {eff:5.3f}", end="")
            else:
                print(f"  {'---':>18s}", end="")
        print()


def main():
    print("Generating individual sweep plots...")

    # Individual 1D plots
    try:
        df = load("sweep_heater_global.csv")
        plot_1d(df, df.columns[0], "Trigger (barrier aniso)",
                "global", "Global Event Heater P=3.0")
        print("  global event plots done")
    except Exception as e:
        print(f"  global: {e}")

    try:
        df = load("sweep_heater_adaptive.csv")
        plot_1d(df, df.columns[0], "Duty min",
                "adaptive", "Adaptive Pulsed Heater P=3.0")
        print("  adaptive plots done")
    except Exception as e:
        print(f"  adaptive: {e}")

    # Comparison plots
    print("\nGenerating comparison plots...")
    grand_comparison()
    energy_savings_chart()
    confinement_efficiency()
    global_event_phase_transition()
    adaptive_vs_fixed()
    print_summary()

    print(f"\nAll plots saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
