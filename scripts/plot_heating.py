"""
Publication-quality phase diagrams for heating + control sweeps.
Reads 2D sweep CSVs and generates heatmaps + stability maps.
"""
import sys, os, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Patch

NICE = {
    'heater.power':    'Heater Power',
    'controller.gain': 'Controller Gain',
    'grid.gamma_diss': 'Dissipation rate',
    'grid.eta_ctrl':   'Controller waste heat (eta)',
    'grid.D_E':        'Energy diffusion D_E',
    'grid.noise_amp':  'Noise amplitude',
    'coupling.alpha':  'Coupling alpha',
    'relaxation.tau':  'Relaxation tau',
    'grid.kappa_tau':  'Kappa-tau',
}

CTRL_COLORS = {
    'Proportional':   '#2196F3',
    'AnisoAware':     '#4CAF50',
    'EventTriggered': '#FF9800',
}

def nice(name):
    return NICE.get(name, name)

def load_2d(path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    p1_name, p2_name = cols[0], cols[1]
    return df, p1_name, p2_name

def pivot(df, p1, p2, metric, ctrl):
    sub = df[df['controller'] == ctrl]
    if sub.empty:
        return None, None, None
    piv = sub.pivot_table(index=p2, columns=p1, values=metric, aggfunc='first')
    return piv.columns.values, piv.index.values, piv.values

def plot_heatmaps(df, p1, p2, title_prefix, out_prefix):
    controllers = df['controller'].unique()
    metrics = [
        ('avg_x',     'Avg |x| (error)',    'hot_r',    None),
        ('avg_E',     'Avg Energy',          'YlOrRd',   None),
        ('avg_aniso', 'Avg Anisotropy',      'viridis',  None),
        ('avg_effort','Avg |u| (effort)',    'PuBu',     None),
    ]
    
    for metric_name, metric_label, cmap, vmax in metrics:
        if metric_name not in df.columns:
            continue
        n_ctrl = len(controllers)
        fig, axes = plt.subplots(1, n_ctrl, figsize=(5*n_ctrl + 1, 4.5),
                                 squeeze=False)
        fig.suptitle(f'{title_prefix}: {metric_label}', fontsize=13, y=1.02)
        
        vmin_all = df[metric_name].min()
        vmax_all = df[metric_name].quantile(0.95) if vmax is None else vmax
        
        for ci, ctrl in enumerate(controllers):
            ax = axes[0, ci]
            x_vals, y_vals, Z = pivot(df, p1, p2, metric_name, ctrl)
            if Z is None:
                continue
            
            # Mark breakdown cells
            _, _, Z_bd = pivot(df, p1, p2, 'breakdown', ctrl)
            
            im = ax.pcolormesh(x_vals, y_vals, Z,
                               cmap=cmap, vmin=vmin_all, vmax=vmax_all,
                               shading='nearest')
            
            if Z_bd is not None:
                for iy in range(Z_bd.shape[0]):
                    for ix in range(Z_bd.shape[1]):
                        if Z_bd[iy, ix] > 0.5:
                            ax.plot(x_vals[ix], y_vals[iy], 'kx',
                                    ms=5, mew=1.2, alpha=0.7)
            
            ax.set_xlabel(nice(p1))
            ax.set_ylabel(nice(p2))
            ax.set_title(ctrl, fontsize=11)
        
        fig.colorbar(im, ax=axes[0, -1], shrink=0.85, label=metric_label)
        fig.tight_layout()
        fname = f'{out_prefix}_{metric_name}'
        fig.savefig(fname + '.png', dpi=180, bbox_inches='tight')
        fig.savefig(fname + '.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {fname}.png')

def plot_stability_map(df, p1, p2, title_prefix, out_prefix):
    controllers = df['controller'].unique()
    
    # For each (p1, p2) find best controller (lowest avg_x among non-breakdown)
    groups = df.groupby([p1, p2])
    
    records = []
    for (v1, v2), grp in groups:
        ok = grp[grp['breakdown'] == 0]
        if ok.empty:
            records.append({'p1': v1, 'p2': v2, 'best': 'BREAKDOWN', 'best_x': 999})
        else:
            best_row = ok.loc[ok['avg_x'].idxmin()]
            records.append({'p1': v1, 'p2': v2,
                           'best': best_row['controller'],
                           'best_x': best_row['avg_x']})
    
    rdf = pd.DataFrame(records)
    all_labels = list(controllers) + ['BREAKDOWN']
    label_to_int = {l: i for i, l in enumerate(all_labels)}
    rdf['best_int'] = rdf['best'].map(label_to_int)
    
    piv = rdf.pivot_table(index='p2', columns='p1', values='best_int', aggfunc='first')
    x_vals = piv.columns.values
    y_vals = piv.index.values
    Z = piv.values
    
    colors = [CTRL_COLORS.get(c, '#999') for c in controllers] + ['#333333']
    cmap = LinearSegmentedColormap.from_list('ctrl', colors, N=len(all_labels))
    bounds = list(range(len(all_labels) + 1))
    norm = BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pcolormesh(x_vals, y_vals, Z, cmap=cmap, norm=norm, shading='nearest')
    ax.set_xlabel(nice(p1), fontsize=11)
    ax.set_ylabel(nice(p2), fontsize=11)
    ax.set_title(f'{title_prefix}: Best Controller Map', fontsize=12)
    
    patches = [Patch(color=colors[i], label=all_labels[i]) for i in range(len(all_labels))]
    ax.legend(handles=patches, loc='upper right', fontsize=8, framealpha=0.9)
    
    fig.tight_layout()
    fname = f'{out_prefix}_stability'
    fig.savefig(fname + '.png', dpi=180, bbox_inches='tight')
    fig.savefig(fname + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {fname}.png')

def plot_aniso_map(df, p1, p2, title_prefix, out_prefix):
    """Anisotropy in tail — barrier survival map."""
    controllers = df['controller'].unique()
    if 'tail_aniso' not in df.columns:
        return
    
    n_ctrl = len(controllers)
    fig, axes = plt.subplots(1, n_ctrl, figsize=(5*n_ctrl + 1, 4.5), squeeze=False)
    fig.suptitle(f'{title_prefix}: Barrier Strength (tail anisotropy)', fontsize=13, y=1.02)
    
    vmax = df['tail_aniso'].quantile(0.95)
    
    for ci, ctrl in enumerate(controllers):
        ax = axes[0, ci]
        x_vals, y_vals, Z = pivot(df, p1, p2, 'tail_aniso', ctrl)
        if Z is None:
            continue
        _, _, Z_bd = pivot(df, p1, p2, 'breakdown', ctrl)
        
        im = ax.pcolormesh(x_vals, y_vals, Z,
                           cmap='magma', vmin=0, vmax=max(vmax, 0.01),
                           shading='nearest')
        if Z_bd is not None:
            for iy in range(Z_bd.shape[0]):
                for ix in range(Z_bd.shape[1]):
                    if Z_bd[iy, ix] > 0.5:
                        ax.plot(x_vals[ix], y_vals[iy], 'wx', ms=5, mew=1.2, alpha=0.7)
        
        ax.set_xlabel(nice(p1))
        ax.set_ylabel(nice(p2))
        ax.set_title(ctrl, fontsize=11)
    
    fig.colorbar(im, ax=axes[0, -1], shrink=0.85, label='Tail anisotropy')
    fig.tight_layout()
    fname = f'{out_prefix}_barrier'
    fig.savefig(fname + '.png', dpi=180, bbox_inches='tight')
    fig.savefig(fname + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {fname}.png')


def process_csv(csv_path):
    df, p1, p2 = load_2d(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(os.path.dirname(csv_path), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, base)
    
    title = f'{nice(p1)} x {nice(p2)}'
    n_bd = df['breakdown'].sum()
    n_total = len(df)
    print(f'\n=== {base}: {nice(p1)} x {nice(p2)} ===')
    print(f'    {n_total} points, {n_bd} breakdowns ({100*n_bd/n_total:.0f}%)')
    
    plot_heatmaps(df, p1, p2, title, out_prefix)
    plot_stability_map(df, p1, p2, title, out_prefix)
    plot_aniso_map(df, p1, p2, title, out_prefix)


if __name__ == '__main__':
    csv_dir = sys.argv[1] if len(sys.argv) > 1 else r'C:\Aniso\build'
    csvs = sorted(glob.glob(os.path.join(csv_dir, 'sweep_*.csv')))
    if not csvs:
        print(f'No sweep_*.csv found in {csv_dir}')
        sys.exit(1)
    
    print(f'Found {len(csvs)} CSV files')
    for csv_path in csvs:
        process_csv(csv_path)
    print('\nDone!')
