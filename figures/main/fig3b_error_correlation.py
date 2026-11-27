"""
Figure 3B - Pearson correlation of per-pair prediction errors (predicted - measured,
log10) between MineCat views and the published predictors, on the pairs every model
predicts. Lower correlation = more complementary errors.

Source data: figures/source_data/error_correlation_{km,kcat}.csv  (figures/make_source_data.py)
Output:      figures/main/fig3b_error_correlation.{png,pdf}
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_MAIN, src  # noqa: E402
from plotting_style import savefig, use_arial  # noqa: E402

use_arial()
TASKS = [('km', '$K_M$'), ('kcat', '$k_{cat}$')]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(24, 9.5), gridspec_kw=dict(wspace=0.66))
    for ax, (task, tlab) in zip(axes, TASKS):
        # per-panel letters omitted: the Km/kcat titles identify each heatmap and
        # the Fig 4 composite supplies the a/b group letters.
        corr = pd.read_csv(src(f'error_correlation_{task}.csv'), index_col=0)
        labels = list(corr.columns)
        M = corr.values
        off_diag = M[~np.eye(len(M), dtype=bool)]
        vmin = np.floor(off_diag.min() * 20) / 20      # tighten scale to reveal structure
        im = ax.imshow(M, cmap='RdYlBu_r', vmin=vmin, vmax=1.0, aspect='equal')
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=26, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=26, fontweight='bold')
        # highlight the two MineCat components (ensemble members)
        for i, lab in enumerate(labels):
            if lab.startswith('MineCat'):
                ax.get_xticklabels()[i].set_color('#7A1128')
                ax.get_yticklabels()[i].set_color('#7A1128')
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = M[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if v > vmin + 0.75 * (1 - vmin) else '#222')
        ax.set_title(tlab, fontsize=34, fontweight='bold', pad=26)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=24)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cb.set_label('PCC (errors)', fontsize=27, fontweight='bold')
    savefig(fig, OUT_MAIN, 'fig3b_error_correlation')
    for task, _ in TASKS:
        c = pd.read_csv(src(f'error_correlation_{task}.csv'), index_col=0)
        print(f"{task}: deep-tree error r = {c.loc['MineCat (deep)', 'MineCat (tree)']:.3f}")


if __name__ == '__main__':
    main()
