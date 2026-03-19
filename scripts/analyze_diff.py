#!/usr/bin/env python3
"""Analyze controller performance differences from a 2D sweep CSV."""
import sys
import pandas as pd
import numpy as np

csv_path = sys.argv[1] if len(sys.argv) > 1 else "sweep_crit3.csv"
df = pd.read_csv(csv_path)
p1, p2 = df.columns[0], df.columns[1]

prop = df[df["controller"] == "Proportional"].set_index([p1, p2])
aniso = df[df["controller"] == "AnisoAware"].set_index([p1, p2])
event = df[df["controller"] == "EventTriggered"].set_index([p1, p2])

common = prop.index.intersection(aniso.index)
prop = prop.loc[common]
aniso = aniso.loc[common]

r_effort = prop["avg_effort"] / aniso["avg_effort"].clip(lower=1e-6)
r_wf = prop["avg_wall_flux"] / aniso["avg_wall_flux"].clip(lower=1e-6)
r_conf = aniso["avg_confinement"] / prop["avg_confinement"].clip(lower=1e-6)
r_x = prop["avg_x"] / aniso["avg_x"].clip(lower=1e-6)

print("=== Proportional vs AnisoAware ===")
print(f"Effort ratio (P/A):       min={r_effort.min():.3f}  max={r_effort.max():.3f}  mean={r_effort.mean():.3f}")
print(f"Wall flux ratio (P/A):    min={r_wf.min():.3f}  max={r_wf.max():.3f}  mean={r_wf.mean():.3f}")
print(f"Confinement ratio (A/P):  min={r_conf.min():.3f}  max={r_conf.max():.3f}  mean={r_conf.mean():.3f}")
print(f"Tracking ratio (P/A):     min={r_x.min():.3f}  max={r_x.max():.3f}  mean={r_x.mean():.3f}")
print()

# Effort savings
effort_save = (1 - aniso["avg_effort"] / prop["avg_effort"]) * 100
print(f"AnisoAware effort savings: min={effort_save.min():.1f}%  max={effort_save.max():.1f}%  mean={effort_save.mean():.1f}%")

# Wall flux savings
wf_save = (1 - aniso["avg_wall_flux"] / prop["avg_wall_flux"]) * 100
print(f"AnisoAware wall flux savings: min={wf_save.min():.1f}%  max={wf_save.max():.1f}%  mean={wf_save.mean():.1f}%")
print()

# Top 10 biggest effort advantage
top = r_effort.sort_values(ascending=False).head(10)
print("Top 10 points with biggest effort advantage (P/A ratio):")
for idx, val in top.items():
    pr = prop.loc[idx]
    ar = aniso.loc[idx]
    es = (1 - ar["avg_effort"] / pr["avg_effort"]) * 100
    print(f"  power={idx[0]:.1f} diss={idx[1]:.1f}: "
          f"P_eff={pr['avg_effort']:.4f} A_eff={ar['avg_effort']:.4f} save={es:.1f}% | "
          f"P_x={pr['avg_x']:.4f} A_x={ar['avg_x']:.4f} | "
          f"P_wf={pr['avg_wall_flux']:.5f} A_wf={ar['avg_wall_flux']:.5f} | "
          f"P_E={pr['avg_E']:.3f} A_E={ar['avg_E']:.3f}")

# Top 5 biggest wall flux advantage
print()
top_wf = r_wf.sort_values(ascending=False).head(5)
print("Top 5 points with biggest wall flux advantage (P/A ratio):")
for idx, val in top_wf.items():
    pr = prop.loc[idx]
    ar = aniso.loc[idx]
    ws = (1 - ar["avg_wall_flux"] / pr["avg_wall_flux"]) * 100
    print(f"  power={idx[0]:.1f} diss={idx[1]:.1f}: "
          f"P_wf={pr['avg_wall_flux']:.5f} A_wf={ar['avg_wall_flux']:.5f} save={ws:.1f}% | "
          f"P_conf={pr['avg_confinement']:.2f} A_conf={ar['avg_confinement']:.2f}")

# Summary: find the "most critical" operating point (high E, marginal confinement)
print()
print("=== CRITICAL OPERATING POINT CANDIDATES ===")
# Points with lowest confinement for Proportional
low_conf_p = prop.sort_values("avg_confinement").head(5)
print("Lowest confinement (Proportional):")
for idx in low_conf_p.index:
    pr = prop.loc[idx]
    ar = aniso.loc[idx]
    print(f"  power={idx[0]:.1f} diss={idx[1]:.1f}: "
          f"P_conf={pr['avg_confinement']:.2f} A_conf={ar['avg_confinement']:.2f} | "
          f"P_barr={pr['avg_barrier_aniso']:.3f} A_barr={ar['avg_barrier_aniso']:.3f} | "
          f"P_E={pr['avg_E']:.2f} A_E={ar['avg_E']:.2f}")
