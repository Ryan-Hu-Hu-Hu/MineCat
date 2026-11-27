"""
Figure 6 - UMAP projections of three MineCat (deep) representations (rows: the fused
input representation, the K_M task-expert output and the k_cat task-expert output),
coloured by measured K_M, k_cat and EC class.

The representations are the fold-0 model's activations on held-out pairs, reduced by PCA
and embedded with UMAP (n_neighbors 25, min_dist 0.10, cosine, random_state 0).

Source data: figures/source_data/fig6_representation_umap.csv.gz
             (layer, pair_id, umap_1, umap_2, km_log10, kcat_log10, ec_class)
Output:      figures/main/fig6_representation_umap.{png,pdf}
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_MAIN, src  # noqa: E402
from plotting_style import savefig, use_arial  # noqa: E402

use_arial()
LAYERS = [('input', 'Concatenate layer'),
          ('km_expert', '$K_M$ head'),
          ('kcat_expert', '$k_{cat}$ head')]
COL_TITLES = ['$K_M$', '$k_{cat}$', 'EC class']
EC_NAMES = {1: 'EC 1 Oxidoreductase', 2: 'EC 2 Transferase', 3: 'EC 3 Hydrolase',
            4: 'EC 4 Lyase', 5: 'EC 5 Isomerase', 6: 'EC 6 Ligase', 7: 'EC 7 Translocase'}
EC_COLORS = {1: '#E63946', 2: '#457B9D', 3: '#2A9D8F', 4: '#E8833A',
             5: '#8E44AD', 6: '#E9C46A', 7: '#264653'}


def ec_top(s):
    try:
        return int(str(s).strip().split('.')[0])
    except (ValueError, IndexError):
        return -1


def main():
    coords = pd.read_csv(src('fig6_representation_umap.csv.gz'))

    fig, axes = plt.subplots(len(LAYERS), 3, figsize=(24, 8 * len(LAYERS)),
                             gridspec_kw=dict(wspace=0.34, hspace=0.18))
    _PLET = iter("ABCDEFGHIJKLMNOPQRS")
    col_mappable = {}
    for ri, (key, lname) in enumerate(LAYERS):
        c = coords[coords.layer == key]
        Y = c[['umap_1', 'umap_2']].to_numpy()
        km, kcat, ecc = c['km_log10'].to_numpy(), c['kcat_log10'].to_numpy(), c['ec_class'].to_numpy()
        # K_M, k_cat continuous
        for ci, (vals, tkey, cmap, clab) in enumerate([
                (km, 'km', 'viridis', r'$\log_{10}\,K_M$'),
                (kcat, 'kcat', 'plasma', r'$\log_{10}\,k_{cat}$')]):
            ax = axes[ri, ci]
            ax.text(-0.05, 1.07, next(_PLET), transform=ax.transAxes, fontsize=34, fontweight="bold", va="bottom", ha="right", zorder=40)
            m = np.isfinite(vals)
            ax.scatter(Y[~m, 0], Y[~m, 1], s=3, c='#e2e2e2', linewidths=0, rasterized=True)
            lo, hi = np.nanpercentile(vals[m], [2, 98])
            sc = ax.scatter(Y[m, 0], Y[m, 1], s=5, c=vals[m], cmap=cmap, vmin=lo, vmax=hi,
                            linewidths=0, alpha=0.8, rasterized=True)
            # one shared colorbar per column (same scale across rows); drawn at the
            # bottom of the column after layout.
            if ci not in col_mappable:
                col_mappable[ci] = (sc, clab)
        # EC class
        axc = axes[ri, 2]
        axc.text(-0.05, 1.07, next(_PLET), transform=axc.transAxes, fontsize=34, fontweight="bold", va="bottom", ha="right", zorder=40)
        for cls in range(1, 8):
            m = ecc == cls
            if m.sum():
                axc.scatter(Y[m, 0], Y[m, 1], s=5, c=EC_COLORS[cls], linewidths=0,
                            alpha=0.7, rasterized=True)
        # EC-agreement annotation removed (reported in the caption/manuscript)
        for ci in range(3):
            axes[ri, ci].set_xlabel('UMAP-1', fontsize=20, fontweight='bold')
            axes[ri, ci].set_ylabel('UMAP-2', fontsize=20, fontweight='bold')
            axes[ri, ci].tick_params(labelsize=15)
            for s in ('top', 'right'):
                axes[ri, ci].spines[s].set_visible(False)
    ec_handles = [Line2D([0], [0], marker='o', ls='', ms=14, mfc=EC_COLORS[c],
                         mec='none', label=EC_NAMES[c]) for c in range(1, 8)]
    # column headers (x titles) on top of the panel row
    for ci in range(3):
        axes[0, ci].set_title(COL_TITLES[ci], fontsize=38, fontweight='bold', pad=16)
    # layer row labels on the far left (replaces per-panel titles)
    fig.subplots_adjust(left=0.11, bottom=0.11)
    for ri, (_, lname) in enumerate(LAYERS):
        pos = axes[ri, 0].get_position()
        fig.text(0.028, pos.y0 + pos.height / 2, lname, rotation=90, va='center', ha='center',
                 fontsize=40, fontweight='bold')
    # one shared horizontal colorbar at the bottom of the K_M and k_cat columns
    fig.canvas.draw()
    last = len(LAYERS) - 1
    for ci, (sc, clab) in col_mappable.items():
        p = axes[last, ci].get_position()
        cax = fig.add_axes([p.x0, 0.058, p.width, 0.011])
        cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
        cb.set_label(clab, fontsize=23, fontweight='bold'); cb.ax.tick_params(labelsize=19)
    # EC-class legend at the bottom of its column
    p2 = axes[last, 2].get_position()
    fig.legend(handles=ec_handles, loc='upper center', bbox_to_anchor=(p2.x0 + p2.width / 2, 0.082),
               ncol=2, fontsize=19, title='EC class',
               title_fontproperties={'weight': 'bold', 'size': 20},
               frameon=True, labelspacing=0.5, columnspacing=1.2, handletextpad=0.4)
    # (verbose methods caption removed — belongs in the figure legend text, not on the figure)
    savefig(fig, OUT_MAIN, 'fig6_representation_umap')


if __name__ == '__main__':
    main()
