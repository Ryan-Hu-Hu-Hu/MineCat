"""
Figure 2 - all-metric radar charts, 4 panels: {K_M, k_cat} x {with / without reaction conditions}.

Same design as fig_radar_metrics.py (each spoke drawn on its OWN real scale; the tick
labels along a spoke are the actual metric values, outward = better, MAE/RMSE reversed),
but the models are no longer auto-selected per task. Instead every model from the fig1
violin panel is shown, split by its env-awareness group, so the K_m and k_cat radars for a
given group contain the SAME models in the SAME legend order. The group's MineCat variant
(ensemble for w/ env, ensemble-no-env for w/o env) is the filled hero.

Source data: figures/source_data/benchmark_metrics.csv (10-fold cluster-CV means)
Output:      figures/main/fig2_benchmark_radar.{png,pdf}
Run:         python figures/main/fig2_benchmark_radar.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_MAIN, src  # noqa: E402
from plotting_style import COLORS, savefig, use_arial  # noqa: E402

use_arial()
TABLE = src('benchmark_metrics.csv')
METRICS = [('R2_mean', 'R²', True), ('PCC_mean', 'PCC', True), ('SCC_mean', 'SCC', True),
           ('MAE_mean', 'MAE', False), ('RMSE_mean', 'RMSE', False)]  # higher_is_better?
RINGS = [0.0, 0.25, 0.5, 0.75, 1.0]
MC_COLOR = '#3A0CA3'   # filled hero (MineCat variant of the group)

TASKS = [('km', '$K_M$'), ('kcat', '$k_{cat}$')]
GROUP_ORDER = ['w/ env', 'w/o env']
# MineCat-first, then external benchmarks by K_m R2 descending; identical order for both tasks
GROUPS = {
    # tree/deep moved to the supplementary members radar (fig_radar_metrics.py);
    # main radar shows the deployed ensemble vs external benchmarks only.
    'w/ env':  ['MineCat (ensemble)',
                'EF-UniKP', 'PreTKcat', 'MPEK', 'DLTKcat'],
    'w/o env': ['MineCat (no env)', 'UniKP', 'KinForm', 'CataPro', 'KmPred', 'CatPred'],
}
HERO = {'w/ env': 'MineCat (ensemble)', 'w/o env': 'MineCat (no env)'}
LEGEND_LABEL = {
    'MineCat (ensemble)': 'MineCat (ens)',
    'MineCat (tree)': 'MineCat (tree)',
    'MineCat (deep)': 'MineCat (deep)',
    'MineCat (no env)': 'MineCat (ens) (no env)',
}

# Enlarged fonts (per request: bigger spoke labels + bigger real-value tick numbers)
SPOKE_FS = 31
TICK_NUM_FS = 23
TITLE_FS = 32
LEGEND_FS = 30
GROUP_FS = 50     # row labels (w/ env, w/o env)
COLHEAD_FS = 50   # column headers (K_M, k_cat)
PLET_FS = 44      # panel letters (A, B, C, D)


def main():
    df = pd.read_csv(TABLE)
    tdf = {task: df[df.task == task].set_index('model') for task, _ in TASKS}

    # Rows = env group (w/ env, w/o env); cols = task (K_m, k_cat). This puts the two panels
    # that share a model set side-by-side, so ONE legend per row (at the right) covers both.
    fig, axes = plt.subplots(2, 2, figsize=(24, 20), subplot_kw=dict(polar=True))
    _PLET = iter("ABCDEFGHIJKLMNOPQRS")
    ang = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])

    for r, group in enumerate(GROUP_ORDER):
        show = GROUPS[group]
        hero = HERO[group]
        for c, (task, tlab) in enumerate(TASKS):
            ax = axes[r, c]
            ax.text(-0.02, 1.08, next(_PLET), transform=ax.transAxes, fontsize=PLET_FS, fontweight="bold", fontfamily="Arial", va="bottom", ha="right", zorder=40)
            t = tdf[task]

            # per-axis real range from the models in THIS panel. Extra padding on the
            # "better" (outer) end so the best model's vertex pulls inward and the outermost
            # tick number clears the filled hero area; small padding on the "worse" end.
            rng = {}
            for col, _, hib in METRICS:
                v = t.loc[show, col].astype(float)
                lo, hi = float(v.min()), float(v.max())
                span = hi - lo if hi > lo else abs(hi) + 1e-6
                pad_better, pad_worse = 0.17 * span, 0.06 * span
                if hib:   # higher is better -> outer end is hi
                    rng[col] = (lo - pad_worse, hi + pad_better, hib)
                else:     # lower is better  -> outer end is lo
                    rng[col] = (lo - pad_better, hi + pad_worse, hib)

            def radius(col, val):
                lo, hi, hib = rng[col]
                frac = (val - lo) / (hi - lo) if hib else (hi - val) / (hi - lo)
                return float(np.clip(frac, 0, 1))

            for m in show:
                vals = [radius(col, float(t.loc[m, col])) for col, _, _ in METRICS]
                vc = np.concatenate([vals, vals[:1]])
                is_hero = m == hero
                color = MC_COLOR if is_hero else COLORS.get(m, '#888888')
                ax.plot(ang_c, vc, color=color, lw=3.6 if is_hero else 1.9,
                        ls='-' if is_hero else '--',
                        label=LEGEND_LABEL.get(m, m), zorder=5 if is_hero else 3)
                if is_hero:
                    ax.fill(ang_c, vc, color=color, alpha=0.13, zorder=2)

            ax.set_xticks(ang)
            ax.set_xticklabels([lab for _, lab, _ in METRICS],
                               fontsize=SPOKE_FS, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.set_yticks(RINGS)
            ax.set_yticklabels([])
            ax.tick_params(axis='x', pad=44)
            ax.grid(True, ls='--', alpha=0.5)

            # real-value tick labels along each spoke
            for a, (col, _, hib) in zip(ang, METRICS):
                lo, hi, _ = rng[col]
                for fr in RINGS[1:]:
                    real = lo + fr * (hi - lo) if hib else hi - fr * (hi - lo)
                    ax.text(a, fr, f'{real:.2f}', ha='center', va='center',
                            fontsize=TICK_NUM_FS, color='#111', fontweight='bold',
                            clip_on=False, zorder=6,
                            bbox=dict(boxstyle='round,pad=0.14', fc='white',
                                      ec='none', alpha=0.90))

        # one legend per env-group row, as a horizontal strip BELOW the row's two panels
        handles, labels = axes[r, 1].get_legend_handles_labels()
        leg_y = 0.530 if r == 0 else 0.050
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.545, leg_y),
                   ncol=len(labels),
                   fontsize=LEGEND_FS, prop={'weight': 'bold', 'size': LEGEND_FS},
                   frameon=True, columnspacing=1.4, handlelength=2.2)
        # env-group label as vertical text to the LEFT of the row (closer to panels A / C)
        row_y = 0.744 if r == 0 else 0.276
        fig.text(0.090, row_y, group, rotation=90, va='center', ha='center',
                 fontsize=GROUP_FS, fontweight='bold')

    # task name as a single column header per column (top), like the row labels
    for c, (task, tlab) in enumerate(TASKS):
        col_x = 0.329 if c == 0 else 0.761
        fig.text(col_x, 0.975, tlab, va='center', ha='center',
                 fontsize=COLHEAD_FS, fontweight='bold')

    fig.subplots_adjust(wspace=0.14, hspace=0.50, left=0.14, right=0.95,
                        top=0.90, bottom=0.12)
    savefig(fig, OUT_MAIN, 'fig2_benchmark_radar')


if __name__ == '__main__':
    main()
