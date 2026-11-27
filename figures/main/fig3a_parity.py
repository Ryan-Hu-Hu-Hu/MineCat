"""
Figure 3A - predicted vs measured parity plots for the four MineCat views (columns:
ensemble, tree, deep, composition; rows: K_M, k_cat). All pooled out-of-fold predictions
are scattered (colour = point density); R2 / PCC are the mean over the 10 test folds.

Source data: results/cluster_cv/oof_predictions.csv.gz
Output:      figures/main/fig3a_parity.{png,pdf}  (combined into Figure 3 by fig3_parity_error_correlation.py)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_MAIN, load_predictions  # noqa: E402
from plotting_style import savefig, use_arial  # noqa: E402

use_arial()
MODELS = [('MineCat (ens)', 'MineCat (ensemble)'),
          ('MineCat (tree)', 'MineCat (tree)'),
          ('MineCat (deep)', 'MineCat (deep)'),
          ('MineCat (comp)', 'MineCat (composition)')]
KM_C = '#1D6A96'
KCAT_C = '#E8833A'
# TASKS = [('km', '$K_M$', KM_C, r'$\log_{10}\,K_M$ (M)'),
#          ('kcat', '$k_{cat}$', KCAT_C, r'$\log_{10}\,k_{cat}$ (s$^{-1}$)')]
TASKS = [('km', '$K_M$', KM_C, r'$\log_{10}\,K_M$'),
         ('kcat', '$k_{cat}$', KCAT_C, r'$\log_{10}\,k_{cat}$')]


def density_color(x, y):
    xy = np.vstack([x, y])
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(len(x), size=min(4000, len(x)), replace=False)
    kde = gaussian_kde(xy[:, fit_idx])
    z = kde(xy)
    return z, z.argsort()


def main():
    preds = load_predictions('cluster', include_benchmarks=False)
    dfs = {name: preds[key] for name, key in MODELS}
    fig = plt.figure(figsize=(29, 14))
    outer = GridSpec(2, len(MODELS), figure=fig, wspace=0.26, hspace=0.28)
    rows = []
    scatter_axes = {}   # (ri, ci) -> scatter ax, for one centered x-label per row
    for ri, (task, tlab, color, axlab) in enumerate(TASKS):
        tcol, pcol, mcol = f'{task}_true_log10', f'{task}_pred_log10', f'has_{task}'
        for ci, (mname, _) in enumerate(MODELS):
            df = dfs[mname]
            d = df[df[mcol].astype(bool) & np.isfinite(df[pcol]) & np.isfinite(df[tcol])]
            yt, yp = d[tcol].values, d[pcol].values
            # metrics = MEAN over the 10 held-out folds (matches the benchmark-figure
            # convention / paper headline); points scattered are all pooled OOF preds.
            r2s, rmses, pccs = [], [], []
            for _, g in d.groupby('fold'):
                if len(g) > 2 and g[tcol].nunique() > 1:
                    r2s.append(r2_score(g[tcol], g[pcol]))
                    rmses.append(np.sqrt(mean_squared_error(g[tcol], g[pcol])))
                    pccs.append(pearsonr(g[tcol], g[pcol])[0])
            r2, rmse, pcc = np.mean(r2s), np.mean(rmses), np.mean(pccs)
            rows.append(dict(task=task, model=mname, n=len(yt), n_folds=len(r2s),
                             R2=r2, RMSE=rmse, PCC=pcc))

            gs = outer[ri, ci].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                                           wspace=0.04, hspace=0.04)
            ax = fig.add_subplot(gs[1, 0])
            scatter_axes[(ri, ci)] = ax
            # per-panel letters omitted: each scatter is titled, and the Fig 4
            # composite (make_fig4_combined.py) supplies the a/b group letters.
            axtop = fig.add_subplot(gs[0, 0], sharex=ax)
            axr = fig.add_subplot(gs[1, 1], sharey=ax)

            z, order = density_color(yt, yp)
            ax.scatter(yt[order], yp[order], c=z[order], s=5,
                       cmap='viridis',
                       linewidths=0, alpha=0.85, rasterized=True)
            lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
            pad = 0.04 * (hi - lo); lim = (lo - pad, hi + pad)
            ax.plot(lim, lim, ls='--', color='#333', lw=1.5, zorder=5)
            ax.set_xlim(lim); ax.set_ylim(lim)
            # x-label: drawn once per row, centered (added after the loop).
            # y-label (Predicted ...): only on the leftmost column, one per row.
            if ci == 0:
                ax.set_ylabel(f'Predicted {axlab}', fontsize=34, fontweight='bold')
            ax.tick_params(labelsize=27)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.text(0.04, 0.96,
                    f'R² = {r2:.3f}\nPCC = {pcc:.3f}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=24,
                    fontweight='bold', color=color,
                    bbox=dict(boxstyle='round',
                              fc='white', ec=color, alpha=0.9))
            bins = np.linspace(lim[0], lim[1], 45)
            axtop.hist(yt, bins=bins, color=color, alpha=0.75, edgecolor='white', linewidth=0.3)
            axr.hist(yp, bins=bins, orientation='horizontal', color=color, alpha=0.75,
                     edgecolor='white', linewidth=0.3)
            for a in (axtop, axr):
                a.axis('off')
            # title only on the top row (column header), like Fig 3
            if ri == 0:
                axtop.set_title(mname, fontsize=34, fontweight='bold', pad=6)
    # fig.text(0.5, 0.065, 'R² / PCC / RMSE are the mean over the 10 held-out ClusterCV folds '
    #          '(matching the benchmark figure); scattered points are all pooled out-of-fold predictions.',
    #          ha='center', fontsize=13, style='italic', color='#444')
    fig.subplots_adjust(bottom=0.11)
    # one centered "Measured ..." x-label per row
    fig.canvas.draw()
    for ri, (task, tlab, color, axlab) in enumerate(TASKS):
        axL = scatter_axes[(ri, 0)]; axR = scatter_axes[(ri, len(MODELS) - 1)]
        pL = axL.get_position(); pR = axR.get_position()
        xc = 0.5 * (pL.x0 + pR.x1)
        yb = pL.y0 - 0.072
        fig.text(xc, yb, f'Measured {axlab}', ha='center', va='center',
                 fontsize=34, fontweight='bold')
    savefig(fig, OUT_MAIN, 'fig3a_parity')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
