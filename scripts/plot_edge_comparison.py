"""Decisive experiment: pulsed heating raises disruption threshold
by providing periodic observation windows."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

out_dir = "figures/fusion"
os.makedirs(out_dir, exist_ok=True)

dc = pd.read_csv("sweep_edge5_const.csv")
dp = pd.read_csv("sweep_edge5_pulsed.csv")

dc["avg_power"] = dc[dc.columns[0]].astype(float)
dp["avg_power"] = dp[dp.columns[0]].astype(float) * 0.7

ctrls = ["Proportional", "AnisoAware"]
ctrl_colors = {"Proportional": "#e74c3c", "AnisoAware": "#2ecc71"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Decisive Experiment: Pulsed Heating Raises Disruption Threshold\n"
             "E-dependent observation noise (beta=2.0): hot plasma blinds the controller",
             fontsize=13, fontweight="bold")

# 1. State error vs avg_E — showing failure threshold
ax = axes[0, 0]
for ctrl in ctrls:
    col = ctrl_colors[ctrl]
    cc = dc[dc["controller"] == ctrl].sort_values("avg_E")
    cp = dp[dp["controller"] == ctrl].sort_values("avg_E")
    ax.plot(cc["avg_E"], cc["avg_x"], "-o", color=col, markersize=4, label=f"{ctrl} Constant")
    ax.plot(cp["avg_E"], cp["avg_x"], "--s", color=col, markersize=4, label=f"{ctrl} Pulsed T=2s", alpha=0.7)
ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="Failure threshold")
ax.set_xlabel("Average Energy")
ax.set_ylabel("State Error |x|")
ax.set_title("State Error vs Energy")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Control effort vs avg_E
ax = axes[0, 1]
for ctrl in ctrls:
    col = ctrl_colors[ctrl]
    cc = dc[dc["controller"] == ctrl].sort_values("avg_E")
    cp = dp[dp["controller"] == ctrl].sort_values("avg_E")
    ax.plot(cc["avg_E"], cc["avg_effort"], "-o", color=col, markersize=4, label=f"{ctrl} Constant")
    ax.plot(cp["avg_E"], cp["avg_effort"], "--s", color=col, markersize=4, label=f"{ctrl} Pulsed", alpha=0.7)
ax.set_xlabel("Average Energy")
ax.set_ylabel("Control Effort")
ax.set_title("Effort vs Energy")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Stability map
ax = axes[1, 0]
for ctrl in ctrls:
    col = ctrl_colors[ctrl]
    for label, df, marker, offset in [("Constant", dc, "o", -0.15), ("Pulsed", dp, "s", 0.15)]:
        sub = df[df["controller"] == ctrl].sort_values("avg_E")
        ok = sub[sub["disruption"].astype(int) == 0]
        fail = sub[sub["disruption"].astype(int) == 1]
        y_val = (1 if ctrl == "AnisoAware" else 0) + offset
        if len(ok) > 0:
            ax.scatter(ok["avg_E"], [y_val]*len(ok), c="green", marker=marker, s=40, alpha=0.8,
                      edgecolors="black", linewidths=0.5)
        if len(fail) > 0:
            ax.scatter(fail["avg_E"], [y_val]*len(fail), c="red", marker=marker, s=40, alpha=0.8,
                      edgecolors="black", linewidths=0.5)

ax.set_yticks([0, 1])
ax.set_yticklabels(["Proportional", "AnisoAware"])
ax.set_xlabel("Average Energy")
ax.set_title("Stability Map (green=OK, red=FAIL)")
handles = [mpatches.Patch(color="green", label="Stable"),
           mpatches.Patch(color="red", label="Disruption"),
           plt.Line2D([0],[0], marker="o", color="gray", label="Constant", markersize=6, linestyle="none"),
           plt.Line2D([0],[0], marker="s", color="gray", label="Pulsed", markersize=6, linestyle="none")]
ax.legend(handles=handles, fontsize=8)
ax.grid(True, alpha=0.3, axis="x")
ax.set_ylim(-0.5, 1.5)

# 4. Summary: failure threshold comparison
ax = axes[1, 1]
bar_data = {}
for ctrl in ctrls:
    for label, df in [("Constant", dc), ("Pulsed", dp)]:
        sub = df[df["controller"] == ctrl].sort_values("avg_E")
        ok = sub[sub["disruption"].astype(int) == 0]
        if len(ok) > 0:
            bar_data[(ctrl, label)] = ok["avg_E"].max()
        else:
            bar_data[(ctrl, label)] = 0

x_pos = [0, 1, 3, 4]
heights = [bar_data[("Proportional", "Constant")],
           bar_data[("Proportional", "Pulsed")],
           bar_data[("AnisoAware", "Constant")],
           bar_data[("AnisoAware", "Pulsed")]]
colors = ["#e74c3c", "#c0392b", "#2ecc71", "#27ae60"]
labels = ["P Const", "P Pulsed", "Aniso Const", "Aniso Pulsed"]
bars = ax.bar(x_pos, heights, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=15, fontsize=9)
ax.set_ylabel("Max Sustainable Energy (avg_E)")
ax.set_title("Disruption Threshold Comparison")
ax.axhline(2.0, color="gray", ls=":", alpha=0.5, label="E_target")
for bar, h in zip(bars, heights):
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.1f}", ha="center", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fn = os.path.join(out_dir, "decisive_experiment.png")
plt.savefig(fn, dpi=150)
print(f"Saved {fn}")

# Print summary
print("\n=== Disruption Threshold Summary ===")
for ctrl in ctrls:
    for label, df in [("Constant", dc), ("Pulsed", dp)]:
        sub = df[df["controller"] == ctrl].sort_values("avg_E")
        ok = sub[sub["disruption"].astype(int) == 0]
        fail = sub[sub["disruption"].astype(int) == 1]
        max_ok_E = ok["avg_E"].max() if len(ok) > 0 else 0
        min_fail_E = fail["avg_E"].min() if len(fail) > 0 else float("inf")
        print(f"  {ctrl:>14s} {label:>8s}: stable up to E={max_ok_E:.1f}, "
              f"fails at E={min_fail_E:.1f}")
