"""
Figure S10 - substrate chemical-space maps for the three substrate encoders MineCat uses
(rows: MolGNet, MolT5, in-house SMILES Transformer; columns: median K_M, median k_cat,
modal EC class of the enzymes acting on each substrate). One point per unique substrate.

Embeddings were standardized, reduced to 50 PCA components and embedded with UMAP
(n_neighbors 30, min_dist 0.25, cosine, random_state 0).

Source data: figures/source_data/figS10_substrate_umap.csv.gz
             (encoder, umap_1, umap_2, median_km_log10, median_kcat_log10, ec_class)
Output:      figures/supplementary/figS10_substrate_space.{png,pdf}
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_SUPP, src  # noqa: E402
from plotting_style import savefig, use_arial  # noqa: E402

use_arial()
matplotlib.rcParams['mathtext.it'] = 'Arial:italic:bold'
ENCODERS = ['MolGNet (graph)', 'MolT5 (chem-language)', 'In-house SMILES-Transformer']
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
    coords = pd.read_csv(src('figS10_substrate_umap.csv.gz'))
    rows = len(ENCODERS)
    fig, axes = plt.subplots(rows, 3, figsize=(24, 8.2 * rows),
                             gridspec_kw=dict(wspace=0.34, hspace=0.20))
    _PLET = iter("ABCDEFGHIJKLMNOPQRS")
    shared = {}
    mappable = {}
    for ri, ename in enumerate(ENCODERS):
        c = coords[coords.encoder == ename]
        Y = c[['umap_1', 'umap_2']].to_numpy()
        km, kcat, ecc = (c[k].to_numpy() for k in ('median_km_log10', 'median_kcat_log10', 'ec_class'))
        for ci, (vals, cmap, clab, cat) in enumerate([
                (km, 'viridis', r'$\log_{10}\,K_M$ (M)', False),
                (kcat, 'plasma', r'$\log_{10}\,k_{cat}$ (s$^{-1}$)', False),
                (ecc, None, 'EC class', True)]):
            ax = axes[ri, ci]
            ax.text(-0.05, 1.05, next(_PLET), transform=ax.transAxes, fontsize=34, fontweight="bold", va="bottom", ha="right", zorder=40)
            if cat:
                for cls in range(1, 8):
                    m = vals == cls
                    if m.sum():
                        ax.scatter(Y[m, 0], Y[m, 1], s=8, c=EC_COLORS[cls], linewidths=0,
                                   alpha=0.7, rasterized=True)
            else:
                m = np.isfinite(vals)
                ax.scatter(Y[~m, 0], Y[~m, 1], s=5, c='#dddddd', linewidths=0, rasterized=True)
                if ci not in shared:
                    shared[ci] = (cmap, *np.nanpercentile(vals[m], [2, 98]), clab)
                cmap_, lo, hi, clab_ = shared[ci]
                mappable[ci] = ax.scatter(Y[m, 0], Y[m, 1], s=9, c=vals[m], cmap=cmap_, vmin=lo, vmax=hi,
                                          linewidths=0, alpha=0.85, rasterized=True)
            ax.set_xlabel('UMAP-1', fontsize=24, fontweight='bold')
            ax.set_ylabel('UMAP-2', fontsize=24, fontweight='bold')
            ax.tick_params(labelsize=18)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if ri == 0:
                ax.set_title(['$K_M$', '$k_{cat}$', 'EC class'][ci], fontsize=38, fontweight='bold', pad=16)
    fig.subplots_adjust(left=0.11, bottom=0.11)
    for ri, ename in enumerate(ENCODERS):
        pos = axes[ri, 0].get_position()
        fig.text(0.028, pos.y0 + pos.height / 2, ename.split(' (')[0].replace('In-house ', '').replace('SMILES-Transformer', 'SMILES-Transformer'), rotation=90, va='center', ha='center',
                 fontsize=36, fontweight='bold')
    fig.canvas.draw()
    last = rows - 1
    for ci in (0, 1):
        clab_ = shared[ci][3]
        p = axes[last, ci].get_position()
        cax = fig.add_axes([p.x0, 0.058, p.width, 0.011])
        cb = fig.colorbar(mappable[ci], cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=23)
        if ci == 1:
            cb.set_ticks([-2, -1, 0, 1, 2])
            cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        fig.text(p.x0 + p.width / 2, 0.030, clab_, ha='center', va='center',
                 fontsize=27, fontweight='bold')
    h = [Line2D([0], [0], marker='o', ls='', mfc=EC_COLORS[c], mec='none', ms=16,
                label=EC_NAMES[c]) for c in range(1, 8)]
    p2 = axes[last, 2].get_position()
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(p2.x0 + p2.width / 2, 0.082),
               ncol=2, fontsize=19, title='EC class',
               title_fontproperties={'weight': 'bold', 'size': 20},
               frameon=True, labelspacing=0.5, columnspacing=1.2, handletextpad=0.4)
    savefig(fig, OUT_SUPP, 'figS10_substrate_space')


if __name__ == '__main__':
    main()
