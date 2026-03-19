"""
Analyze and plot heater strategy sweeps.
Compares: Pulsed heater (duty sweep) vs Event-Driven heater (trigger sweep)
in two power regimes: low (0.5) and high (3.0).
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
CTRL_LS = {
    "Proportional":   "-",
    "AnisoAware":     "-",
    "EventTriggered": "--",
}


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def plot_1d_sweep(df, param_col, param_label, prefix, title_extra=""):
    ctrls = df["controller"].unique()
    metrics = [
        ("avg_E",              "Avg Energy"),
        ("avg_effort",         "Avg Control Effort"),
        ("avg_x",              "Avg Tracking Error |x|"),
        ("avg_wall_flux",      "Avg Wall Flux"),
        ("avg_confinement",    "Avg Confinement Ratio"),
        ("avg_barrier_aniso",  "Avg Barrier Anisotropy"),
    ]

    for col, ylabel in metrics:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for ctrl in ctrls:
            sub = df[df["controller"] == ctrl].sort_values(param_col)
            color = CTRL_COLORS.get(ctrl, "gray")
            ls = CTRL_LS.get(ctrl, "-")
            ax.plot(sub[param_col], sub[col], color=color, ls=ls,
                    marker="o", ms=4, label=ctrl)
        ax.set_xlabel(param_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {param_label}{title_extra}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"{prefix}_{col.replace('avg_','')}"
        fig.savefig(FIG_DIR / f"{fname}.png", dpi=150)
        plt.close(fig)
        print(f"  saved {fname}.png")


def plot_pulsed_comparison(df_lo, df_hi):
    """Side-by-side pulsed heater comparison: low power vs high power."""
    pcol = df_lo.columns[0]
    ctrls = df_lo["controller"].unique()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics = [
        ("avg_E",            "Energy"),
        ("avg_effort",       "Control Effort"),
        ("avg_x",            "Tracking Error"),
        ("avg_wall_flux",    "Wall Flux"),
        ("avg_confinement",  "Confinement"),
        ("avg_barrier_aniso","Barrier Aniso"),
    ]

    for mi, (col, label) in enumerate(metrics):
        ax = axes.flat[mi]
        for ctrl in ctrls:
            color = CTRL_COLORS.get(ctrl, "gray")
            sub_lo = df_lo[df_lo["controller"] == ctrl].sort_values(pcol)
            sub_hi = df_hi[df_hi["controller"] == ctrl].sort_values(pcol)
            ax.plot(sub_lo[pcol], sub_lo[col], color=color, ls="--", alpha=0.6,
                    marker="s", ms=3, label=f"{ctrl} (P=0.5)")
            ax.plot(sub_hi[pcol], sub_hi[col], color=color, ls="-",
                    marker="o", ms=4, label=f"{ctrl} (P=3.0)")
        ax.set_xlabel("Duty cycle")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if mi == 0:
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Pulsed Heater: Low Power (P=0.5) vs High Power (P=3.0)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "pulsed_power_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved pulsed_power_comparison.png")


def plot_energy_savings(df_pulsed_hi, df_event_hi):
    """Show energy savings from pulsed/event heating vs constant (duty=1.0)."""
    pcol_p = df_pulsed_hi.columns[0]
    pcol_e = df_event_hi.columns[0]
    ctrls = df_pulsed_hi["controller"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pulsed: % energy reduction vs duty=1.0
    ax = axes[0]
    for ctrl in ctrls:
        sub = df_pulsed_hi[df_pulsed_hi["controller"] == ctrl].sort_values(pcol_p)
        base_E = sub[sub[pcol_p] >= 0.99]["avg_E"].values
        if len(base_E) == 0:
            continue
        base_E = base_E[0]
        pct = (1.0 - sub["avg_E"] / base_E) * 100
        color = CTRL_COLORS.get(ctrl, "gray")
        ax.plot(sub[pcol_p], pct, color=color, marker="o", ms=4, label=ctrl)
    ax.axhline(0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("Duty cycle")
    ax.set_ylabel("Energy reduction vs constant (%)")
    ax.set_title("Pulsed Heater (P=3.0): Energy Savings")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Event: % energy reduction vs highest trigger
    ax = axes[1]
    for ctrl in ctrls:
        sub = df_event_hi[df_event_hi["controller"] == ctrl].sort_values(pcol_e)
        base_E = sub[sub[pcol_e] >= sub[pcol_e].max() - 0.001]["avg_E"].values
        if len(base_E) == 0:
            continue
        base_E = base_E[0]
        pct = (1.0 - sub["avg_E"] / base_E) * 100
        color = CTRL_COLORS.get(ctrl, "gray")
        ax.plot(sub[pcol_e], pct, color=color, marker="o", ms=4, label=ctrl)
    ax.axhline(0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("Trigger threshold")
    ax.set_ylabel("Energy reduction vs max trigger (%)")
    ax.set_title("Event-Driven Heater (P=3.0): Energy Savings")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "energy_savings_hot.png", dpi=150)
    plt.close(fig)
    print("  saved energy_savings_hot.png")


def plot_grand_comparison(df_pulsed_lo, df_pulsed_hi, df_event_lo, df_event_hi):
    """
    Grand comparison: at each strategy's optimal point, show metrics
    for all three controllers.
    """
    def best_row(df, pcol, ctrl="AnisoAware"):
        sub = df[df["controller"] == ctrl].sort_values(pcol)
        if sub.empty or "avg_confinement" not in sub.columns:
            return sub.iloc[0] if not sub.empty else None
        score = sub["avg_confinement"] / (sub["avg_E"] + 0.01)
        return sub.loc[score.idxmax()]

    pcol_p = df_pulsed_lo.columns[0]
    pcol_e = df_event_lo.columns[0]

    strategies = {
        "Constant\n(P=0.5)":     (df_pulsed_lo, pcol_p, 1.0),
        "Pulsed 50%\n(P=0.5)":   (df_pulsed_lo, pcol_p, 0.50),
        "Pulsed 20%\n(P=0.5)":   (df_pulsed_lo, pcol_p, 0.20),
        "Constant\n(P=3.0)":     (df_pulsed_hi, pcol_p, 1.0),
        "Pulsed 50%\n(P=3.0)":   (df_pulsed_hi, pcol_p, 0.50),
        "Pulsed 20%\n(P=3.0)":   (df_pulsed_hi, pcol_p, 0.20),
        "Event t=0.1\n(P=3.0)":  (df_event_hi,  pcol_e, 0.10),
        "Event t=0.3\n(P=3.0)":  (df_event_hi,  pcol_e, 0.30),
    }

    metric_cols  = ["avg_E", "avg_effort", "avg_x", "avg_wall_flux",
                    "avg_confinement", "avg_barrier_aniso"]
    metric_names = ["Energy", "Effort", "|x|", "Wall Flux",
                    "Confinement", "Barrier"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ctrls_list = ["Proportional", "AnisoAware", "EventTriggered"]

    x_labels = list(strategies.keys())
    x_pos = np.arange(len(x_labels))
    width = 0.25

    for mi, (col, mlabel) in enumerate(zip(metric_cols, metric_names)):
        ax = axes.flat[mi]
        for ci, ctrl in enumerate(ctrls_list):
            vals = []
            for sname, (df, pcol, pval) in strategies.items():
                row = df[(df["controller"] == ctrl) &
                         (np.isclose(df[pcol], pval, atol=0.01))]
                vals.append(row[col].values[0] if not row.empty and col in row.columns else 0)
            color = CTRL_COLORS.get(ctrl, "gray")
            ax.bar(x_pos + ci * width, vals, width,
                   label=ctrl if mi == 0 else "", color=color, alpha=0.85)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.grid(True, alpha=0.2, axis="y")

    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Grand Heater Strategy Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "grand_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved grand_comparison.png")


def print_table(df, pcol, label):
    ctrls = df["controller"].unique()
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'='*100}")
    header = f"{'Param':>8s}"
    for ctrl in ctrls:
        header += f"  | {ctrl:>14s} E   effort  conf  barrier wallflux"
    print(header)
    print("-" * (8 + len(ctrls) * 58))

    for val in sorted(df[pcol].unique()):
        line = f"{val:8.3f}"
        for ctrl in ctrls:
            row = df[(df[pcol] == val) & (df["controller"] == ctrl)]
            if row.empty:
                line += f"  | {'---':>50s}"
                continue
            r = row.iloc[0]
            line += (f"  | {r['avg_E']:7.3f}  {r['avg_effort']:6.3f}  "
                     f"{r.get('avg_confinement',0):5.2f}  "
                     f"{r.get('avg_barrier_aniso',0):5.3f}  "
                     f"{r.get('avg_wall_flux',0):7.4f}")
        print(line)


def print_key_findings(df_pulsed_hi, df_pulsed_lo):
    pcol = df_pulsed_hi.columns[0]
    print("\n" + "="*80)
    print("  KEY FINDINGS")
    print("="*80)

    for label, df in [("P=3.0", df_pulsed_hi), ("P=0.5", df_pulsed_lo)]:
        ctrls = df["controller"].unique()
        for ctrl in ctrls:
            sub = df[df["controller"] == ctrl].sort_values(pcol)
            E_full = sub[sub[pcol] >= 0.99]["avg_E"].values[0]
            E_min  = sub["avg_E"].min()
            duty_min = sub.loc[sub["avg_E"].idxmin(), pcol]
            pct = (1.0 - E_min / E_full) * 100
            print(f"  {label} {ctrl:>15s}: duty={duty_min:.2f} -> "
                  f"E={E_min:.3f} (vs {E_full:.3f} constant, {pct:.1f}% savings)")

    # Confinement efficiency: confinement per unit energy
    print("\n  Confinement efficiency (confinement / energy):")
    for label, df in [("P=3.0", df_pulsed_hi), ("P=0.5", df_pulsed_lo)]:
        for ctrl in ["AnisoAware", "Proportional"]:
            sub = df[df["controller"] == ctrl].sort_values(pcol)
            eff = sub["avg_confinement"] / sub["avg_E"]
            best_idx = eff.idxmax()
            best_duty = sub.loc[best_idx, pcol]
            best_eff = eff.loc[best_idx]
            full_eff = (sub[sub[pcol] >= 0.99]["avg_confinement"].values[0] /
                        sub[sub[pcol] >= 0.99]["avg_E"].values[0])
            improvement = (best_eff / full_eff - 1) * 100
            print(f"    {label} {ctrl:>15s}: best at duty={best_duty:.2f}, "
                  f"eff={best_eff:.4f} vs {full_eff:.4f} constant "
                  f"({improvement:+.1f}%)")


def main():
    files = {
        "pulsed_lo":  "sweep_heater_pulsed.csv",
        "event_lo":   "sweep_heater_event.csv",
        "pulsed_hi":  "sweep_heater_pulsed_hot.csv",
        "event_hi":   "sweep_heater_event_hot.csv",
    }

    dfs = {}
    for key, path in files.items():
        try:
            dfs[key] = load_csv(path)
            print(f"Loaded {path}: {len(dfs[key])} rows")
        except Exception as e:
            print(f"Could not load {path}: {e}")

    # Print tables
    if "pulsed_lo" in dfs:
        print_table(dfs["pulsed_lo"], dfs["pulsed_lo"].columns[0],
                    "Pulsed Heater (P=0.5), sweep duty")
    if "pulsed_hi" in dfs:
        print_table(dfs["pulsed_hi"], dfs["pulsed_hi"].columns[0],
                    "Pulsed Heater (P=3.0), sweep duty")
    if "event_lo" in dfs:
        print_table(dfs["event_lo"], dfs["event_lo"].columns[0],
                    "Event-Driven Heater (P=0.5), sweep trigger")
    if "event_hi" in dfs:
        print_table(dfs["event_hi"], dfs["event_hi"].columns[0],
                    "Event-Driven Heater (P=3.0), sweep trigger")

    # Key findings
    if "pulsed_hi" in dfs and "pulsed_lo" in dfs:
        print_key_findings(dfs["pulsed_hi"], dfs["pulsed_lo"])

    # Plots
    print("\nGenerating plots...")

    if "pulsed_lo" in dfs:
        plot_1d_sweep(dfs["pulsed_lo"], dfs["pulsed_lo"].columns[0],
                      "Duty cycle", "pulsed_lo", " (P=0.5)")
    if "pulsed_hi" in dfs:
        plot_1d_sweep(dfs["pulsed_hi"], dfs["pulsed_hi"].columns[0],
                      "Duty cycle", "pulsed_hi", " (P=3.0)")
    if "event_hi" in dfs:
        plot_1d_sweep(dfs["event_hi"], dfs["event_hi"].columns[0],
                      "Trigger threshold", "event_hi", " (P=3.0)")

    if "pulsed_lo" in dfs and "pulsed_hi" in dfs:
        plot_pulsed_comparison(dfs["pulsed_lo"], dfs["pulsed_hi"])

    if "pulsed_hi" in dfs and "event_hi" in dfs:
        plot_energy_savings(dfs["pulsed_hi"], dfs["event_hi"])

    if all(k in dfs for k in ["pulsed_lo", "pulsed_hi", "event_lo", "event_hi"]):
        plot_grand_comparison(dfs["pulsed_lo"], dfs["pulsed_hi"],
                              dfs["event_lo"], dfs["event_hi"])

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
