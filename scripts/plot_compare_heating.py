"""Compare constant vs pulsed heating strategies at matched average power."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = "figures/fusion"
os.makedirs(out_dir, exist_ok=True)

dc = pd.read_csv("sweep_compare_const.csv")
dp_fast = pd.read_csv("sweep_compare_pulsed.csv")
dp_slow = pd.read_csv("sweep_compare_pulsed_slow.csv")

dc["avg_power"] = dc[dc.columns[0]].astype(float)
dp_fast["avg_power"] = dp_fast[dp_fast.columns[0]].astype(float) * 0.5
dp_slow["avg_power"] = dp_slow[dp_slow.columns[0]].astype(float) * 0.5

strategies = [
    ("Constant", dc, "-", "o"),
    ("Pulsed T=1.5s", dp_fast, "--", "s"),
    ("Pulsed T=10s", dp_slow, ":", "D"),
]

ctrl_colors = {"Proportional": "#e74c3c", "AnisoAware": "#2ecc71"}
ctrls = ["Proportional", "AnisoAware"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Heating Strategy Comparison: Constant vs Pulsed (duty=0.5)\n"
             "Matched average power, Landau energy response", fontsize=13, fontweight="bold")

metrics = [
    (axes[0, 0], "avg_E", "Average Energy", True),
    (axes[0, 1], "avg_barrier_aniso", "Barrier Anisotropy", False),
    (axes[1, 0], "avg_effort", "Control Effort", False),
    (axes[1, 1], "avg_x", "State Error |x|", False),
]

for ax, col_name, ylabel, show_target in metrics:
    for ctrl in ctrls:
        base_color = ctrl_colors[ctrl]
        for sname, df, ls, mk in strategies:
            sub = df[df["controller"] == ctrl].sort_values("avg_power")
            label = f"{ctrl} ({sname})"
            alpha = 1.0 if "Constant" in sname else 0.7
            ax.plot(sub["avg_power"], sub[col_name], linestyle=ls, marker=mk,
                    color=base_color, label=label, markersize=4, alpha=alpha)
    if show_target:
        ax.axhline(2.0, color="gray", ls=":", alpha=0.5, label="E_target")
    ax.set_xlabel("Average Heater Power")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

plt.tight_layout()
fn = os.path.join(out_dir, "const_vs_pulsed.png")
plt.savefig(fn, dpi=150)
print(f"Saved {fn}")

# Relative difference summary
print("\n=== Relative Differences at avg_power = 5, 10, 15, 20 ===")
target_powers = [5, 10, 15, 20]
print(f"{'Ctrl':>14s} {'Strategy':>16s} {'Power':>6s} {'dE%':>7s} {'dBarr%':>7s} {'dEff%':>7s} {'d|x|%':>7s}")
for ctrl in ctrls:
    cc = dc[dc["controller"] == ctrl].set_index(dc.columns[0])
    for sname, df, _, _ in strategies[1:]:
        sub = df[df["controller"] == ctrl]
        sub = sub.assign(avg_pow=sub["avg_power"])
        for tp in target_powers:
            c_row = cc.loc[cc.index.astype(float).values == tp]
            p_row = sub[np.isclose(sub["avg_pow"], tp, atol=0.5)]
            if len(c_row) == 0 or len(p_row) == 0:
                continue
            for metric in ["avg_E", "avg_barrier_aniso", "avg_effort", "avg_x"]:
                vc = float(c_row[metric].iloc[0])
                vp = float(p_row[metric].iloc[0])
                pct = (vp - vc) / abs(vc) * 100 if abs(vc) > 1e-10 else 0
                if metric == "avg_E":
                    print(f"{ctrl:>14s} {sname:>16s} {tp:6d}", end="")
                print(f" {pct:+7.2f}%", end="")
            print()

# Summary bar chart
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Barrier Health Comparison at Matched Average Power", fontsize=13, fontweight="bold")

for idx, ctrl in enumerate(ctrls):
    ax = axes2[idx]
    for sname, df, _, _ in strategies:
        sub = df[df["controller"] == ctrl].sort_values("avg_power")
        ax.plot(sub["avg_power"], sub["avg_barrier_aniso"], label=sname, linewidth=2)
    ax.set_xlabel("Average Heater Power")
    ax.set_ylabel("Barrier Anisotropy")
    ax.set_title(ctrl)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fn2 = os.path.join(out_dir, "barrier_comparison.png")
plt.savefig(fn2, dpi=150)
print(f"Saved {fn2}")
