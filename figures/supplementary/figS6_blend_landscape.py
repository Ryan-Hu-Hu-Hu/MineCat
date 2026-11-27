"""
Figure S6 - three-way ensemble blend-weight landscape. Pooled out-of-fold R2 over the
(w_tree, w_deep, w_comp) simplex (grid step 0.02) for K_M and k_cat, drawn as a ternary
filled contour; the star marks the grid optimum.

Source data: figures/source_data/blend_landscape_{km,kcat}.csv (figures/make_source_data.py)
Output:      figures/supplementary/figS6_blend_landscape.{png,pdf}
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_SUPP, src  # noqa: E402
from plotting_style import savefig, use_arial  # noqa: E402

use_arial()
A = np.array([0.0, 0.0]); B = np.array([1.0, 0.0]); C = np.array([0.5, np.sqrt(3) / 2])  # tree/deep/comp


def bary(w):
    return w[0] * A + w[1] * B + w[2] * C


def panel(ax, task, label):
    land = pd.read_csv(src(f'blend_landscape_{task}.csv'))
    grid = land[['w_tree', 'w_deep', 'w_comp']].to_numpy(float)
    r2 = land['pooled_R2'].to_numpy(float)
    pts = np.array([bary(w) for w in grid])
    vmax = r2.max(); vmin = max(vmax - 0.12, r2.min())  # tighten scale to reveal the ridge
    # continuous ternary spectrum (filled contour) over the simplex grid
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1])
    sc = ax.tricontourf(triang, np.clip(r2, vmin, vmax), levels=60, cmap='viridis',
                        vmin=vmin, vmax=vmax, antialiased=True)
    sc.set_edgecolor('face')             # avoid white seams between contour bands
    # triangle outline + corners
    tri = np.array([A, B, C, A]); ax.plot(tri[:, 0], tri[:, 1], color='#333', lw=1.5)
    for corner, lab, dx, dy, ha in [(A, 'Tree', -0.00, -0.06, 'right'), (B, 'Deep', 0.00, -0.06, 'left'),
                                    (C, 'Comp', 0.0, 0.03, 'center')]:
        ax.text(corner[0] + dx, corner[1] + dy, lab, fontsize=26, fontweight='bold', ha=ha, va='center')
    istar = int(np.argmax(r2)); pstar = bary(grid[istar])
    ax.scatter(*pstar, marker='*', s=620, color='#E63946', edgecolor='white', linewidth=1.3,
               zorder=6, label=f'grid opt  w={grid[istar][0]:.2f}/{grid[istar][1]:.2f}/{grid[istar][2]:.2f}\nR²={r2[istar]:.3f}')
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_ylim(-0.10, C[1] + 0.13)
    ax.set_title(f'{label}', fontsize=32, fontweight='bold', pad=-6)
    # on-figure legend removed (per request); star = grid optimum, open circle =
    # LOFO-mean weights — described in the figure caption.
    cb = plt.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.05, pad=0.005, shrink=0.8)
    cb.locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=18)
    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    cb.set_label('Pooled R²', fontsize=22, fontweight='bold')
    return dict(task=task, w_tree=grid[istar][0], w_deep=grid[istar][1], w_comp=grid[istar][2],
                r2_star=float(r2[istar]))


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9.6), gridspec_kw=dict(wspace=0.34))
    _PLET = iter("ABCDEFGHIJKLMNOPQRS")
    ax1.text(-0.05, 1.00, next(_PLET), transform=ax1.transAxes, fontsize=36, fontweight="bold", va="bottom", ha="right", zorder=40)
    ax2.text(-0.05, 1.00, next(_PLET), transform=ax2.transAxes, fontsize=36, fontweight="bold", va="bottom", ha="right", zorder=40)
    rows = [panel(ax1, 'km', '$K_m$'), panel(ax2, 'kcat', '$k_{cat}$')]
    savefig(fig, OUT_SUPP, 'figS6_blend_landscape')
    for r in rows:
        print(f"{r['task']:5s} grid optimum w(t/d/c)={r['w_tree']:.2f}/{r['w_deep']:.2f}/{r['w_comp']:.2f} "
              f"R2*={r['r2_star']:.3f}")


if __name__ == '__main__':
    main()
