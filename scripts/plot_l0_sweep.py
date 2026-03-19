#!/usr/bin/env python3
"""Plot 1D l0 sweep results: energy, effort, confinement vs resolution coupling."""
import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

csv_path = sys.argv[1] if len(sys.argv) > 1 else "sweep_crit5.csv"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "figures/critical"
os.makedirs(out_dir, exist_ok=True)
base = os.path.splitext(os.path.basename(csv_path))[0]

df = pd.read_csv(csv_path)
p = df.columns[0]
controllers = df["controller"].unique()
colors = {"Proportional": "#e74c3c", "AnisoAware": "#2ecc71", "EventTriggered": "#3498db"}

def plot_metric(metric, ylabel, title, fname, logy=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ctrl in controllers:
        sub = df[df["controller"] == ctrl].sort_values(p)
        ax.plot(sub[p], sub[metric], "-o", ms=3, lw=1.8,
                color=colors.get(ctrl, "gray"), label=ctrl)
    ax.set_xlabel(r"Resolution coupling $\ell_0$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname))
    print(f"  -> {fname}")
    plt.close(fig)

plot_metric("avg_E", r"$\langle E \rangle$", "Mean Energy vs Resolution Coupling",
            f"{base}_energy.png", logy=True)
plot_metric("avg_effort", r"$\langle |u| \rangle$", "Control Effort vs Resolution Coupling",
            f"{base}_effort.png")
plot_metric("avg_x", r"$\langle |x| \rangle$", "Tracking Error vs Resolution Coupling",
            f"{base}_tracking.png")
plot_metric("avg_wall_flux", "Wall flux", "Energy Loss to Wall vs Resolution Coupling",
            f"{base}_wallflux.png", logy=True)
plot_metric("avg_confinement", "Confinement ratio", "Confinement Quality vs Resolution Coupling",
            f"{base}_confinement.png")
plot_metric("avg_barrier_aniso", "Barrier anisotropy", "Barrier Deformation vs Resolution Coupling",
            f"{base}_barrier.png")

# Combined: energy ratio P/A
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

P = df[df["controller"] == "Proportional"].sort_values(p).set_index(p)
A = df[df["controller"] == "AnisoAware"].sort_values(p).set_index(p)

l0_vals = P.index

ax1.plot(l0_vals, P["avg_E"], "-o", ms=3, lw=1.8, color="#e74c3c", label="Proportional")
ax1.plot(l0_vals, A["avg_E"], "-s", ms=3, lw=1.8, color="#2ecc71", label="AnisoAware")
ax1.fill_between(l0_vals, A["avg_E"], P["avg_E"], alpha=0.15, color="#e74c3c")
ax1.set_xlabel(r"Resolution coupling $\ell_0$")
ax1.set_ylabel(r"$\langle E \rangle$")
ax1.set_title("Energy: waste heat feedback")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.annotate(f"{P['avg_E'].iloc[-1]/A['avg_E'].iloc[-1]:.1f}x",
             xy=(l0_vals[-1], (P["avg_E"].iloc[-1] + A["avg_E"].iloc[-1])/2),
             fontsize=14, fontweight="bold", color="#e74c3c", ha="right")

eff_save = (1 - A["avg_effort"] / P["avg_effort"]) * 100
wf_save = (1 - A["avg_wall_flux"] / P["avg_wall_flux"]) * 100

ax2.plot(l0_vals, eff_save, "-o", ms=3, lw=1.8, color="#2ecc71", label="Effort saving")
ax2.plot(l0_vals, wf_save, "-s", ms=3, lw=1.8, color="#3498db", label="Wall flux saving")
ax2.set_xlabel(r"Resolution coupling $\ell_0$")
ax2.set_ylabel("AnisoAware advantage (%)")
ax2.set_title("AnisoAware vs Proportional: savings")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=-5)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, f"{base}_comparison.png"))
print(f"  -> {base}_comparison.png")
plt.close(fig)

print(f"\nAll figures saved to {out_dir}/")
