"""
Figure 4 (and Figure S9) - generalization to remote homologs.

Heatmaps of test R2 (left) and SCC (right) for K_M (top) and k_cat (bottom) as the maximum
global sequence identity between a test enzyme and any training enzyme is lowered from 40%
to 10%. Test rows passing each cutoff are pooled over the 10 folds.
  Figure 4 : MineCat (ensemble) + the nine published predictors
  Figure S9: the three ensemble members (tree, deep, composition)

Source data: figures/source_data/seqid_metrics.csv (figures/make_source_data.py, from
             results/cluster_cv/*oof_predictions.csv.gz + test_train_identity.csv.gz)
Output:      figures/main/fig4_sequence_identity.{png,pdf}
             figures/supplementary/figS9_member_sequence_identity.{png,pdf}
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_MAIN, OUT_SUPP, src  # noqa: E402
from plotting_style import savefig  # noqa: E402

DISPLAY = {'MineCat (ensemble)': 'MineCat (ens)', 'MineCat (tree)': 'MineCat (tree)',
           'MineCat (composition)': 'MineCat (comp)', 'MineCat (deep)': 'MineCat (deep)'}
MEMBERS = ('MineCat (tree)', 'MineCat (deep)', 'MineCat (composition)')
MODEL_ORDER = ['MineCat (ensemble)', 'MineCat (tree)', 'MineCat (composition)', 'MineCat (deep)',
               'CataPro', 'KmPred', 'UniKP', 'EF-UniKP', 'KinForm', 'CatPred', 'MPEK',
               'PreTKcat', 'DLTKcat']


def draw_heatmap(tbl, model_subset, out_dir, name):
    """fig1-style 2x2 of HEATMAPS: rows = K_m (top) / k_cat (bottom), cols = R²/SCC.
    Each heatmap: y = models (MineCat first), x = seq-id cutoff, cell = metric value.
    K_M/k_cat appear as row headers (left) and R²/SCC as column headers (top),
    matching Fig 3 / Fig 4."""
    from matplotlib.ticker import FormatStrFormatter
    # Force Arial for every text element.
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Arial'
    matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'

    CELL_FS, XT_FS, YT_FS = 17, 23, 21
    COLHEAD_FS, ROWHEAD_FS, PLET_FS = 36, 36, 38
    CBAR_FS, XLAB_FS = 19, 36

    tasks = [('km', '$K_M$'), ('kcat', '$k_{cat}$')]
    cols = [('R2', 'R²'), ('SCC', 'SCC')]
    cutoffs = sorted(tbl['cutoff'].unique(), reverse=True)     # 40 -> 10
    present = [m for m in model_subset if m in set(tbl.model)]
    # MineCat first, then benchmarks by Km R² at the 40% cutoff (desc)
    def r40(m):
        s = tbl[(tbl.task == 'km') & (tbl.model == m) & (tbl.cutoff == cutoffs[0])]
        return float(s['R2'].iloc[0]) if not s.empty else -9
    mc = [m for m in present if str(m).startswith('MineCat')]
    bench = sorted([m for m in present if not str(m).startswith('MineCat')], key=r40, reverse=True)
    models = mc + bench
    ylabels = [DISPLAY.get(m, m) for m in models]
    fig, axes = plt.subplots(2, 2, figsize=(21, 13), gridspec_kw=dict(wspace=0.16, hspace=0.22))
    _PL = iter('ABCDEFGH')
    for ri, (task, tlab) in enumerate(tasks):
        for ci, (metric, mlab) in enumerate(cols):
            ax = axes[ri, ci]
            M = np.full((len(models), len(cutoffs)), np.nan)
            for i, m in enumerate(models):
                for j, c in enumerate(cutoffs):
                    s = tbl[(tbl.task == task) & (tbl.model == m) & (tbl.cutoff == c)]
                    if not s.empty:
                        M[i, j] = float(s[metric].iloc[0])
            vmin, vmax = np.nanmin(M), np.nanmax(M)
            im = ax.imshow(M, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            for i in range(len(models)):
                for j in range(len(cutoffs)):
                    if np.isfinite(M[i, j]):
                        fr = (M[i, j] - vmin) / (vmax - vmin + 1e-9)
                        ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                                fontsize=CELL_FS, fontweight='bold',
                                color='white' if fr < 0.45 else '#111')
            plet_x = -0.20 if ci == 0 else -0.02   # A/C sit further left of the model labels
            ax.text(plet_x, 1.03, next(_PL), transform=ax.transAxes, fontsize=PLET_FS,
                    fontweight='bold', fontfamily='Arial', va='bottom', ha='right')
            ax.set_xticks(range(len(cutoffs)))
            ax.set_xticklabels([str(c) for c in cutoffs], fontsize=XT_FS, fontweight='bold')
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(ylabels if ci == 0 else [''] * len(models),
                               fontsize=YT_FS, fontweight='bold')
            if ci == 0:
                for i, m in enumerate(models):
                    if str(m).startswith('MineCat'):
                        ax.get_yticklabels()[i].set_color('#7A1128')
            # column header (R²/SCC) on the top row only, like Fig 3/4
            if ri == 0:
                ax.set_title(mlab, fontsize=COLHEAD_FS, fontweight='bold', pad=14)
            cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
            cb.ax.tick_params(labelsize=CBAR_FS)
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.text(0.5, 0.04, 'Max train/test sequence identity cutoff (%)',
             ha='center', fontsize=XLAB_FS, fontweight='bold')
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.11, top=0.92)
    # row headers (K_M / k_cat) as vertical text to the LEFT of each row, like Fig 3/4
    fig.canvas.draw()
    for ri, (task, tlab) in enumerate(tasks):
        p = axes[ri, 0].get_position()
        fig.text(0.045, 0.5 * (p.y0 + p.y1), tlab, rotation=90,
                 va='center', ha='center', fontsize=ROWHEAD_FS, fontweight='bold')
    savefig(fig, out_dir, name)


def main():
    tbl = pd.read_csv(src('seqid_metrics.csv'))
    present = [m for m in MODEL_ORDER if m in set(tbl.model)]
    draw_heatmap(tbl, [m for m in present if m not in MEMBERS], OUT_MAIN, 'fig4_sequence_identity')
    draw_heatmap(tbl, [m for m in MEMBERS if m in present], OUT_SUPP, 'figS9_member_sequence_identity')


if __name__ == '__main__':
    main()
