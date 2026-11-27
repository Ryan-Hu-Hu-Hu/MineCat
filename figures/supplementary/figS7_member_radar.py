"""
Figure S7 - all-metric radar for the three MineCat ensemble members (tree, deep,
composition), K_M (left) and k_cat (right); each spoke on its own real-value scale
(MAE / RMSE reversed so outward = better).

Source data: figures/source_data/benchmark_metrics.csv
Output:      figures/supplementary/figS7_member_radar.{png,pdf}
Run:         python figures/supplementary/figS7_member_radar.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_SUPP, src  # noqa: E402
from plotting_style import savefig, use_arial  # noqa: E402

use_arial()
matplotlib.rcParams['mathtext.bf'] = 'Arial:italic:bold'
TABLE = src('benchmark_metrics.csv')
METRICS = [('R2_mean', 'R²', True), ('PCC_mean', 'PCC', True), ('SCC_mean', 'SCC', True),
           ('MAE_mean', 'MAE', False), ('RMSE_mean', 'RMSE', False)]  # higher_is_better?
# Three MineCat ensemble members (the deployed ensemble itself is the Fig 3 hero,
# so this supplementary radar contrasts the members that feed the blend).
MEMBERS = [('MineCat (tree)', 'MineCat (tree)', '#C44E52'),
           ('MineCat (deep)', 'MineCat (deep)', '#4C72B0'),
           ('MineCat (composition)', 'MineCat (comp)', '#2A9D8F')]
TASKS = [('km', r'$\mathbf{K_M}$'), ('kcat', r'$\mathbf{k_{cat}}$')]
RINGS = [0.0, 0.25, 0.5, 0.75, 1.0]     # radial ring fractions


SPOKE_FS, TICK_NUM_FS, COLHEAD_FS, PLET_FS, LEGEND_FS = 30, 22, 40, 40, 28


def main():
    df = pd.read_csv(TABLE)
    df = df[df.model != 'MineCat (no env)']
    fig, axes = plt.subplots(1, 2, figsize=(22, 12.5), subplot_kw=dict(polar=True))
    ang = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    _PLET = iter("AB")
    handles = labels = None

    for ci, (ax, (task, tlab)) in enumerate(zip(axes, TASKS)):
        ax.text(-0.10, 1.10, next(_PLET), transform=ax.transAxes, fontsize=PLET_FS,
                fontweight='bold', fontfamily='Arial', va='bottom', ha='right', zorder=40)
        t = df[df.task == task].copy().set_index('model')
        show = [key for key, _, _ in MEMBERS]

        rng = {}
        for col, _, hib in METRICS:
            v = t.loc[show, col].astype(float)
            lo, hi = float(v.min()), float(v.max())
            pad = 0.10 * (hi - lo if hi > lo else abs(hi) + 1e-6)
            rng[col] = (lo - pad, hi + pad, hib)

        def radius(col, val):
            lo, hi, hib = rng[col]
            frac = (val - lo) / (hi - lo) if hib else (hi - val) / (hi - lo)
            return float(np.clip(frac, 0, 1))

        for key, disp, mcol in MEMBERS:
            vals = [radius(col, float(t.loc[key, col])) for col, _, _ in METRICS]
            vc = np.concatenate([vals, vals[:1]])
            ax.plot(ang_c, vc, color=mcol, lw=3.6, ls='-', marker='o', ms=9,
                    label=disp, zorder=4)

        ax.set_xticks(ang)
        ax.set_xticklabels([lab for _, lab, _ in METRICS], fontsize=SPOKE_FS, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_yticks(RINGS)
        ax.set_yticklabels([])
        ax.tick_params(axis='x', pad=44)
        ax.grid(True, ls='--', alpha=0.5)

        for a, (col, _, hib) in zip(ang, METRICS):
            lo, hi, _ = rng[col]
            for fr in RINGS[1:]:
                real = lo + fr * (hi - lo) if hib else hi - fr * (hi - lo)
                ax.text(a, fr, f'{real:.2f}', ha='center', va='center', fontsize=TICK_NUM_FS,
                        color='#111', fontweight='bold', clip_on=False, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.90))
        handles, labels = ax.get_legend_handles_labels()

    fig.canvas.draw()
    for ci, (task, tlab) in enumerate(TASKS):
        pos = axes[ci].get_position()
        fig.text(0.5 * (pos.x0 + pos.x1) + (-0.07 if ci == 0 else 0.07), 0.955, tlab, va='center', ha='center',
                 fontsize=COLHEAD_FS, fontweight='bold')
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.045), ncol=len(labels),
               fontsize=LEGEND_FS, prop={'weight': 'bold', 'size': LEGEND_FS},
               frameon=True, columnspacing=1.6, handlelength=2.4)
    fig.subplots_adjust(wspace=0.45, left=0.05, right=0.97, top=0.84, bottom=0.14)
    savefig(fig, OUT_SUPP, 'figS7_member_radar')


if __name__ == '__main__':
    main()
