"""Compare pulsed heating at different duty cycles vs constant baseline.
Key plot: metrics vs avg_E (Pareto curves) — if pulsed gives better
controllability at same energy, points lie below the constant line."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = "figures/fusion"
os.makedirs(out_dir, exist_ok=True)

dc = pd.read_csv("sweep_compare_const.csv")
dp5 = pd.read_csv("sweep_duty_p5.csv")
dp3 = pd.read_csv("sweep_duty_p3.csv")

dc["avg_power"] = dc[dc.columns[0]].astype(float)
dp5["duty"] = dp5[dp5.columns[0]].astype(float)
dp5["avg_power"] = 20.0 * dp5["duty"]
dp3["duty"] = dp3[dp3.columns[0]].astype(float)
dp3["avg_power"] = 20.0 * dp3["duty"]

ctrls = ["Proportional", "AnisoAware"]
ctrl_colors = {"Proportional": "#e74c3c", "AnisoAware": "#2ecc71"}

# ── Figure 1: Pareto curves (metrics vs avg_E) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Pulsed Duty Cycle Optimization: Metrics vs Average Energy\n"
             "Peak power=20, comparing period=3s and 5s vs constant",
             fontsize=13, fontweight="bold")

metrics = [
    (axes[0, 0], "avg_x", "State Error |x|", False),
    (axes[0, 1], "avg_barrier_aniso", "Barrier Anisotropy", False),
    (axes[1, 0], "avg_effort", "Control Effort", False),
    (axes[1, 1], "fusion_margin", "Fusion Margin (E_core/E_target)", True),
]

for ax, col_name, ylabel, show_threshold in metrics:
    for ctrl in ctrls:
        col = ctrl_colors[ctrl]
        cc = dc[dc["controller"] == ctrl].sort_values("avg_E")
        c5 = dp5[dp5["controller"] == ctrl].sort_values("avg_E")
        c3 = dp3[dp3["controller"] == ctrl].sort_values("avg_E")

        ax.plot(cc["avg_E"], cc[col_name], "-", color=col, linewidth=2.5,
                label=f"{ctrl} Constant", alpha=0.9)
        ax.plot(c5["avg_E"], c5[col_name], "--s", color=col, markersize=4,
                label=f"{ctrl} Pulsed T=5s", alpha=0.7)
        ax.plot(c3["avg_E"], c3[col_name], ":D", color=col, markersize=4,
                label=f"{ctrl} Pulsed T=3s", alpha=0.7)

    if show_threshold:
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="Fusion viable")
    ax.axvline(2.0, color="gray", ls="--", alpha=0.3, label="E_target" if ax == axes[0,0] else "")
    ax.set_xlabel("Average Energy")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fn1 = os.path.join(out_dir, "duty_pareto.png")
plt.savefig(fn1, dpi=150)
print(f"Saved {fn1}")

# ── Figure 2: Duty-specific view ──
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("Effect of Duty Cycle on Controllability\n"
              "Peak power=20, period=5s (solid) and 3s (dashed)",
              fontsize=13, fontweight="bold")

duty_metrics = [
    (axes2[0, 0], "avg_x", "State Error |x|"),
    (axes2[0, 1], "avg_barrier_aniso", "Barrier Anisotropy"),
    (axes2[1, 0], "avg_effort", "Control Effort"),
    (axes2[1, 1], "avg_E", "Average Energy"),
]

for ax, col_name, ylabel in duty_metrics:
    for ctrl in ctrls:
        col = ctrl_colors[ctrl]
        c5 = dp5[dp5["controller"] == ctrl].sort_values("duty")
        c3 = dp3[dp3["controller"] == ctrl].sort_values("duty")
        ax.plot(c5["duty"], c5[col_name], "-o", color=col, markersize=4,
                label=f"{ctrl} T=5s")
        ax.plot(c3["duty"], c3[col_name], "--s", color=col, markersize=4,
                label=f"{ctrl} T=3s", alpha=0.7)
    ax.set_xlabel("Duty Cycle")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fn2 = os.path.join(out_dir, "duty_sweep.png")
plt.savefig(fn2, dpi=150)
print(f"Saved {fn2}")

# ── Figure 3: Controllability efficiency ──
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Controllability Efficiency: State Error per Unit Energy\n"
              "Lower = better control at given operating temperature",
              fontsize=13, fontweight="bold")

for idx, ctrl in enumerate(ctrls):
    ax = axes3[idx]
    col = ctrl_colors[ctrl]

    cc = dc[dc["controller"] == ctrl].sort_values("avg_E")
    c5 = dp5[dp5["controller"] == ctrl].sort_values("avg_E")
    c3 = dp3[dp3["controller"] == ctrl].sort_values("avg_E")

    eff_c = cc["avg_x"] / cc["avg_E"]
    eff_5 = c5["avg_x"] / c5["avg_E"]
    eff_3 = c3["avg_x"] / c3["avg_E"]

    ax.plot(cc["avg_E"], eff_c, "-", color=col, linewidth=2.5, label="Constant")
    ax.plot(c5["avg_E"], eff_5, "--s", color=col, markersize=4, label="Pulsed T=5s")
    ax.plot(c3["avg_E"], eff_3, ":D", color=col, markersize=4, label="Pulsed T=3s")

    ax.set_xlabel("Average Energy")
    ax.set_ylabel("|x| / E (lower = better)")
    ax.set_title(ctrl)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fn3 = os.path.join(out_dir, "duty_efficiency.png")
plt.savefig(fn3, dpi=150)
print(f"Saved {fn3}")

# ── Print key comparisons ──
print("\n=== Controllability at avg_E ~ 6.0 (E_target=2.0, 3x margin) ===")
target_E = 6.0
for ctrl in ctrls:
    print(f"\n{ctrl}:")
    for name, df in [("Constant", dc), ("Pulsed T=5s", dp5), ("Pulsed T=3s", dp3)]:
        sub = df[df["controller"] == ctrl]
        idx = (sub["avg_E"] - target_E).abs().idxmin()
        row = sub.loc[idx]
        duty_str = f"duty={row['duty']:.2f}" if "duty" in row else "N/A"
        print(f"  {name:>14s} (E={row['avg_E']:.1f}, {duty_str}): "
              f"|x|={row['avg_x']:.4f}  barrier={row['avg_barrier_aniso']:.3f}  "
              f"effort={row['avg_effort']:.3f}  fm={row['fusion_margin']:.2f}")

print("\n=== Controllability at avg_E ~ 4.0 ===")
target_E = 4.0
for ctrl in ctrls:
    print(f"\n{ctrl}:")
    for name, df in [("Constant", dc), ("Pulsed T=5s", dp5), ("Pulsed T=3s", dp3)]:
        sub = df[df["controller"] == ctrl]
        idx = (sub["avg_E"] - target_E).abs().idxmin()
        row = sub.loc[idx]
        duty_str = f"duty={row['duty']:.2f}" if "duty" in row else "N/A"
        print(f"  {name:>14s} (E={row['avg_E']:.1f}, {duty_str}): "
              f"|x|={row['avg_x']:.4f}  barrier={row['avg_barrier_aniso']:.3f}  "
              f"effort={row['avg_effort']:.3f}  fm={row['fusion_margin']:.2f}")
