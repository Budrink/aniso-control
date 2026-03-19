#!/usr/bin/env python3
"""Analyze 1D sweep with controller comparison."""
import sys
import pandas as pd

csv_path = sys.argv[1] if len(sys.argv) > 1 else "sweep_crit5.csv"
df = pd.read_csv(csv_path)
p1 = df.columns[0]

controllers = df["controller"].unique()
print(f"Sweep parameter: {p1}")
print(f"Controllers: {list(controllers)}")
print()

# Extract per-controller series
ctrl_data = {}
for c in controllers:
    ctrl_data[c] = df[df["controller"] == c].set_index(p1)

# Compare key metrics
if "Proportional" in ctrl_data and "AnisoAware" in ctrl_data:
    P = ctrl_data["Proportional"]
    A = ctrl_data["AnisoAware"]

    print(f"{'l0':>6}  {'P_x':>8} {'A_x':>8} {'P_eff':>8} {'A_eff':>8}"
          f"  {'P_E':>8} {'A_E':>8}  {'E_ratio':>8}"
          f"  {'P_wf':>8} {'A_wf':>8}  {'P_conf':>8} {'A_conf':>8}"
          f"  {'P_barr':>8} {'A_barr':>8}")
    print("-" * 150)

    for idx in P.index:
        p = P.loc[idx]
        a = A.loc[idx]
        e_rat = p["avg_E"] / max(a["avg_E"], 1e-6)
        print(f"{idx:6.2f}  {p['avg_x']:8.4f} {a['avg_x']:8.4f} "
              f"{p['avg_effort']:8.4f} {a['avg_effort']:8.4f}  "
              f"{p['avg_E']:8.3f} {a['avg_E']:8.3f}  {e_rat:8.2f}  "
              f"{p['avg_wall_flux']:8.5f} {a['avg_wall_flux']:8.5f}  "
              f"{p['avg_confinement']:8.2f} {a['avg_confinement']:8.2f}  "
              f"{p['avg_barrier_aniso']:8.4f} {a['avg_barrier_aniso']:8.4f}")

    # Summary at key points
    print()
    for l0_val in [0.01, 0.15, 0.30, 0.45, 0.59]:
        if l0_val not in P.index:
            closest = min(P.index, key=lambda x: abs(x - l0_val))
            l0_val = closest
        p = P.loc[l0_val]
        a = A.loc[l0_val]
        e_rat = p["avg_E"] / max(a["avg_E"], 1e-6)
        eff_save = (1 - a["avg_effort"] / p["avg_effort"]) * 100
        wf_save = (1 - a["avg_wall_flux"] / p["avg_wall_flux"]) * 100
        print(f"l0={l0_val:.2f}: Energy P/A ratio = {e_rat:.2f}x | "
              f"Effort saving = {eff_save:.1f}% | "
              f"Wall flux saving = {wf_save:.1f}% | "
              f"P_conf={p['avg_confinement']:.2f} A_conf={a['avg_confinement']:.2f} | "
              f"P_barr={p['avg_barrier_aniso']:.3f} A_barr={a['avg_barrier_aniso']:.3f}")
