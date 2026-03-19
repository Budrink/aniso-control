#!/usr/bin/env python3
"""
Analyze 2D sweep results to find the critical boundary and
identify the "critically unstable" operating point.

Usage:
    python scripts/plot_critical.py sweep_critical_find.csv [output_dir]
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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


def make_pivot(df, ctrl, p1, p2, value_col, aggfunc="first"):
    sub = df[df["controller"] == ctrl]
    return sub.pivot_table(index=p2, columns=p1,
                           values=value_col, aggfunc=aggfunc)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_critical.py <sweep2d.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    p1, p2 = df.columns[0], df.columns[1]
    controllers = df["controller"].unique()
    n_ctrl = len(controllers)

    p1_label = p1.replace(".", " → ").replace("_", " ")
    p2_label = p2.replace(".", " → ").replace("_", " ")

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # ── 1. Stability map: breakdown + disruption ──────────────────────
    fig, axes = plt.subplots(1, n_ctrl, figsize=(4.8 * n_ctrl, 4.2),
                             squeeze=False)
    axes = axes[0]

    cmap_stab = ListedColormap(["#2ecc71", "#f39c12", "#e74c3c"])
    norm_stab = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_stab.N)

    for idx, ctrl in enumerate(controllers):
        ax = axes[idx]
        sub = df[df["controller"] == ctrl]

        status = sub["breakdown"].astype(int) + sub["disruption"].astype(int)
        sub = sub.copy()
        sub["status"] = status.values

        piv = sub.pivot_table(index=p2, columns=p1,
                              values="status", aggfunc="first")
        im = ax.pcolormesh(piv.columns, piv.index, piv.values,
                           cmap=cmap_stab, norm=norm_stab, shading="auto")
        ax.set_xlabel(p1_label)
        if idx == 0:
            ax.set_ylabel(p2_label)
        else:
            ax.set_yticklabels([])
        ax.set_title(ctrl)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#2ecc71", label="Stable"),
        Patch(facecolor="#f39c12", label="Disruption risk"),
        Patch(facecolor="#e74c3c", label="Breakdown"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Stability Map", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_stability.png"))
    print(f"  -> {base_name}_stability.png")
    plt.close(fig)

    # ── 2. Tracking error heatmap ────────────────────────────────────
    fig, axes = plt.subplots(1, n_ctrl, figsize=(4.8 * n_ctrl, 4.2),
                             squeeze=False)
    axes = axes[0]
    vmax_x = min(df["avg_x"].quantile(0.92), 1.0)

    for idx, ctrl in enumerate(controllers):
        ax = axes[idx]
        piv = make_pivot(df, ctrl, p1, p2, "avg_x")
        im = ax.pcolormesh(piv.columns, piv.index, piv.values,
                           cmap="RdYlGn_r", vmin=0, vmax=vmax_x,
                           shading="auto")
        bd = make_pivot(df, ctrl, p1, p2, "breakdown")
        ax.contour(bd.columns, bd.index, bd.values.astype(float),
                   levels=[0.5], colors="black", linewidths=1.5,
                   linestyles="--")
        ax.set_xlabel(p1_label)
        if idx == 0: ax.set_ylabel(p2_label)
        else: ax.set_yticklabels([])
        ax.set_title(ctrl)

    fig.colorbar(im, ax=axes, label=r"$\langle |x| \rangle$",
                 shrink=0.85, pad=0.02)
    fig.suptitle("Tracking Error", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_avg_x.png"))
    print(f"  -> {base_name}_avg_x.png")
    plt.close(fig)

    # ── 3. Energy heatmap ────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_ctrl, figsize=(4.8 * n_ctrl, 4.2),
                             squeeze=False)
    axes = axes[0]
    vmax_e = df["avg_E"].quantile(0.92)

    for idx, ctrl in enumerate(controllers):
        ax = axes[idx]
        piv = make_pivot(df, ctrl, p1, p2, "avg_E")
        im = ax.pcolormesh(piv.columns, piv.index, piv.values,
                           cmap="hot", vmin=0, vmax=vmax_e, shading="auto")
        ax.set_xlabel(p1_label)
        if idx == 0: ax.set_ylabel(p2_label)
        else: ax.set_yticklabels([])
        ax.set_title(ctrl)

    fig.colorbar(im, ax=axes, label=r"$\langle E \rangle$",
                 shrink=0.85, pad=0.02)
    fig.suptitle("Mean Energy", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_avg_E.png"))
    print(f"  -> {base_name}_avg_E.png")
    plt.close(fig)

    # ── 4. Control effort heatmap ────────────────────────────────────
    fig, axes = plt.subplots(1, n_ctrl, figsize=(4.8 * n_ctrl, 4.2),
                             squeeze=False)
    axes = axes[0]
    vmax_u = df["avg_effort"].quantile(0.92)

    for idx, ctrl in enumerate(controllers):
        ax = axes[idx]
        piv = make_pivot(df, ctrl, p1, p2, "avg_effort")
        im = ax.pcolormesh(piv.columns, piv.index, piv.values,
                           cmap="YlOrRd", vmin=0, vmax=vmax_u,
                           shading="auto")
        ax.set_xlabel(p1_label)
        if idx == 0: ax.set_ylabel(p2_label)
        else: ax.set_yticklabels([])
        ax.set_title(ctrl)

    fig.colorbar(im, ax=axes, label=r"$\langle |u| \rangle$",
                 shrink=0.85, pad=0.02)
    fig.suptitle("Control Effort", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_avg_effort.png"))
    print(f"  -> {base_name}_avg_effort.png")
    plt.close(fig)

    # ── 5. Barrier anisotropy ────────────────────────────────────────
    if "avg_barrier_aniso" in df.columns:
        fig, axes = plt.subplots(1, n_ctrl, figsize=(4.8 * n_ctrl, 4.2),
                                 squeeze=False)
        axes = axes[0]
        vmax_b = max(df["avg_barrier_aniso"].quantile(0.92), 0.1)

        for idx, ctrl in enumerate(controllers):
            ax = axes[idx]
            piv = make_pivot(df, ctrl, p1, p2, "avg_barrier_aniso")
            im = ax.pcolormesh(piv.columns, piv.index, piv.values,
                               cmap="viridis", vmin=0, vmax=vmax_b,
                               shading="auto")
            ax.set_xlabel(p1_label)
            if idx == 0: ax.set_ylabel(p2_label)
            else: ax.set_yticklabels([])
            ax.set_title(ctrl)

        fig.colorbar(im, ax=axes, label="barrier anisotropy",
                     shrink=0.85, pad=0.02)
        fig.suptitle("Barrier Strength (ring anisotropy)", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base_name}_barrier.png"))
        print(f"  -> {base_name}_barrier.png")
        plt.close(fig)

    # ── 6. Confinement ratio ─────────────────────────────────────────
    if "avg_confinement" in df.columns:
        fig, axes = plt.subplots(1, n_ctrl, figsize=(4.8 * n_ctrl, 4.2),
                                 squeeze=False)
        axes = axes[0]
        vmax_c = min(df["avg_confinement"].quantile(0.95), 20)

        for idx, ctrl in enumerate(controllers):
            ax = axes[idx]
            piv = make_pivot(df, ctrl, p1, p2, "avg_confinement")
            piv = piv.clip(upper=vmax_c)
            im = ax.pcolormesh(piv.columns, piv.index, piv.values,
                               cmap="coolwarm", vmin=0, vmax=vmax_c,
                               shading="auto")
            ax.set_xlabel(p1_label)
            if idx == 0: ax.set_ylabel(p2_label)
            else: ax.set_yticklabels([])
            ax.set_title(ctrl)

        fig.colorbar(im, ax=axes, label="confinement ratio",
                     shrink=0.85, pad=0.02)
        fig.suptitle("Confinement Quality", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base_name}_confinement.png"))
        print(f"  -> {base_name}_confinement.png")
        plt.close(fig)

    # ── 7. Find critical boundary ────────────────────────────────────
    print("\n" + "=" * 60)
    print("CRITICAL BOUNDARY ANALYSIS")
    print("=" * 60)

    p1_vals = sorted(df[p1].unique())
    p2_vals = sorted(df[p2].unique())

    for ctrl in controllers:
        sub = df[df["controller"] == ctrl]
        ok = sub[sub["breakdown"] == 0]
        fail = sub[sub["breakdown"] == 1]

        print(f"\n--- {ctrl} ---")
        print(f"  Stable: {len(ok)} / {len(sub)}")

        if len(ok) > 0:
            best = ok.loc[ok["avg_x"].idxmin()]
            print(f"  Best stable:  {p1}={best[p1]:.2f}  {p2}={best[p2]:.2f}  "
                  f"avg|x|={best['avg_x']:.4f}  avg_E={best['avg_E']:.4f}  "
                  f"effort={best['avg_effort']:.4f}")

        # Find boundary: stable points where a neighbor is unstable
        boundary_pts = []
        for _, row in ok.iterrows():
            v1, v2 = row[p1], row[p2]
            step1 = p1_vals[1] - p1_vals[0] if len(p1_vals) > 1 else 1
            step2 = p2_vals[1] - p2_vals[0] if len(p2_vals) > 1 else 1
            neighbors = [
                (v1 + step1, v2), (v1 - step1, v2),
                (v1, v2 + step2), (v1, v2 - step2),
            ]
            for nv1, nv2 in neighbors:
                nb = fail[(abs(fail[p1] - nv1) < step1 * 0.1) &
                          (abs(fail[p2] - nv2) < step2 * 0.1)]
                if len(nb) > 0:
                    boundary_pts.append(row)
                    break

        if boundary_pts:
            bdf = pd.DataFrame(boundary_pts)
            most_critical = bdf.loc[bdf["avg_x"].idxmax()]
            print(f"  Critical pt:  {p1}={most_critical[p1]:.2f}  "
                  f"{p2}={most_critical[p2]:.2f}  "
                  f"avg|x|={most_critical['avg_x']:.4f}  "
                  f"avg_E={most_critical['avg_E']:.4f}  "
                  f"effort={most_critical['avg_effort']:.4f}")
            if "avg_confinement" in bdf.columns:
                print(f"               confinement={most_critical['avg_confinement']:.2f}  "
                      f"barrier={most_critical['avg_barrier_aniso']:.4f}  "
                      f"wall_flux={most_critical['avg_wall_flux']:.6f}")

    # Find THE critical point: where Proportional fails but others survive
    print("\n" + "=" * 60)
    print("DIFFERENTIAL STABILITY ANALYSIS")
    print("=" * 60)

    if n_ctrl >= 2:
        c0 = controllers[0]  # Proportional (baseline)
        for ctrl in controllers[1:]:
            sub0 = df[df["controller"] == c0].set_index([p1, p2])
            sub1 = df[df["controller"] == ctrl].set_index([p1, p2])

            common = sub0.index.intersection(sub1.index)
            diff = sub0.loc[common]
            diff_ctrl = sub1.loc[common]

            where_c0_fails_c1_ok = (diff["breakdown"] == 1) & \
                                   (diff_ctrl["breakdown"] == 0)

            advantage_pts = diff_ctrl[where_c0_fails_c1_ok]
            if len(advantage_pts) > 0:
                print(f"\n{ctrl} survives where {c0} fails: "
                      f"{len(advantage_pts)} points")
                best = advantage_pts.loc[advantage_pts["avg_x"].idxmax()]
                print(f"  Most dramatic: {p1}={best.name[0]:.2f}  "
                      f"{p2}={best.name[1]:.2f}")
                print(f"  {ctrl}: avg|x|={best['avg_x']:.4f}  "
                      f"effort={best['avg_effort']:.4f}")
            else:
                print(f"\nNo points where {ctrl} survives but {c0} fails.")

            # Where both survive but ctrl is much better
            both_ok = (diff["breakdown"] == 0) & \
                      (diff_ctrl["breakdown"] == 0)
            if both_ok.sum() > 0:
                d0 = diff[both_ok]
                d1 = diff_ctrl[both_ok]
                ratio = d0["avg_x"] / d1["avg_x"].clip(lower=1e-6)
                if ratio.max() > 1.5:
                    best_idx = ratio.idxmax()
                    print(f"  Best advantage ({c0}/{ctrl} error ratio): "
                          f"{ratio.max():.2f}x at "
                          f"{p1}={best_idx[0]:.2f} {p2}={best_idx[1]:.2f}")

    print(f"\nFigures saved to {out_dir}/")


if __name__ == "__main__":
    main()
