"""
Figure 5 - reconstruction fidelity of the MineCat (deep) auxiliary decoders, in aggregate
(220 held-out fold-0 test complexes, fold-0 model) and on one curated held-out complex
(pair_id 22137; 10-fold model average for the structure panels).

  A  |predicted - true| distogram bin, aggregate          B  IFP decoder scores, aggregate
  C  distogram of the curated pair (true / predicted / |error|)
  D  per-residue expected distance to the ligand rendered on the chai-1 structure (PyMOL):
     predicted / ground truth / |error|, with pocket close-ups
  E  IFP heatmaps (predicted / ground truth / agreement)  F  IFP residues on the pocket (PyMOL)
  G  cross-attention entropy, aggregate                   H  substrate-to-protein attention, curated pair
  I  predicted vs true residue-ligand distance            J  EGNN pocket-denoising RMSD trajectory

Source data: figures/source_data/fig5_reconstruction/
    panel_data.npz        arrays for panels A, C, E, G-J
    summary_metrics.json  aggregate scores (bin accuracy, distance MAE, IFP acc/F1/AUC, rho, AUC)
    pymol_*.png           PyMOL renders for panels D and F
The panel data were produced by running the 10 MineCat (deep) checkpoints over the fold-0
test set (model internals: distogram logits, IFP logits, cross-attention weights, EGNN
denoising trajectory); the PyMOL renders colour the chai-1 structure by those values.

Output: figures/main/fig5_reconstruction.{png,pdf}
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_MAIN, src  # noqa: E402
from plotting_style import use_arial  # noqa: E402

use_arial()
AX_FS, TK_FS, LET_FS, TITLE_FS = 28, 21, 40, 27
DIST_CMAP = 'plasma_r'     # sequential: bright = near the ligand
ERR_CMAP = 'Reds'
CONTACT_A = 8.0            # contact threshold (A) marked in panel I
IFP_TYPES = ['H-bond', 'Hydrophobic', 'Aromatic', 'Pos. charge', 'Neg. charge', 'General']
DATA = src('fig5_reconstruction')


def load():
    z = np.load(os.path.join(DATA, 'panel_data.npz'))
    with open(os.path.join(DATA, 'summary_metrics.json')) as f:
        m = json.load(f)
    png = lambda n: os.path.join(DATA, n)  # noqa: E731
    return dict(
        dbin=z['dbin'], bin_acc=m['bin_acc'], ang_mae=m['ang_mae'],
        ifp_acc=m['ifp_acc'], ifp_f1=m['ifp_f1'], ifp_auc=m['ifp_auc'],
        ent_A=z['ent_A'], ent_B=z['ent_B'],
        ex_adp=(z['adp_true_bins'], z['adp_pred_bins'], z['adp_mask']), ex_attn=z['attention'],
        td=z['true_distance'], pdv=z['pred_distance'], sp=m['sp'], auc=m['auc'],
        rmsds=list(z['denoise_rmsd']), vmin=m['vmin'], vmax=m['vmax'], emax=m['emax'],
        paths=dict(p_pred=png('pymol_predicted.png'), p_true=png('pymol_ground_truth.png'),
                   p_err=png('pymol_error.png'), pz_pred=png('pymol_predicted_pocket.png'),
                   pz_true=png('pymol_ground_truth_pocket.png'), pz_err=png('pymol_error_pocket.png')),
        Pm=z['ifp_pred'], Tm=z['ifp_true'], agree=z['ifp_agreement'], rn=list(z['ifp_residues']),
        p_ifp3d=png('pymol_ifp_pocket.png'))


_PEND = []


def _letter(ax, s, dx=-0.12, row=0):
    """Register a panel letter; placed after layout so all letters in a row share one y.
    dx=None -> fixed left-margin x; otherwise dx is in axes-width units left/right of ax."""
    _PEND.append((ax, s, dx, row))


def _pocket_inset(ax, png):
    """Zoomed-pocket inset (lower right of a cartoon panel) with a large label."""
    axin = ax.inset_axes([0.62, 0.0, 0.40, 0.40])
    axin.imshow(plt.imread(png))
    axin.set_xticks([]); axin.set_yticks([])
    for sp in axin.spines.values():
        sp.set_edgecolor('#333'); sp.set_linewidth(1.6)
    axin.set_title('Pocket', fontsize=30, fontweight='bold', pad=2)


def _despine(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def _cbar(fig, ax, cmap, vmin, vmax, label):
    """Compact horizontal colorbar for the narrow cartoon panels."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, location='bottom', aspect=22, shrink=0.82)
    cb.set_label(label, fontsize=32, fontweight='bold')
    cb.ax.tick_params(labelsize=27)


def draw(C):
    fig = plt.figure(figsize=(23, 41.0))
    outer = fig.add_gridspec(5, 1, height_ratios=[1.30, 1.95, 2.05, 1.10, 1.10], hspace=0.30,
                             left=0.075, right=0.985)
    _PEND.clear()

    # ============ Row 0: A hist | B IFP scores | C distogram (true / pred / error) ============
    g1 = outer[0].subgridspec(1, 5, width_ratios=[1.0, 0.45, 1.28, 0.30, 3.6], wspace=0.12)
    axA = fig.add_subplot(g1[0, 0]); _letter(axA, 'A', None, 0)
    axA.hist(C['dbin'], bins=np.arange(0, 17) - 0.5, color='#457B9D', edgecolor='white')
    axA.set_xlabel('Distance bin', fontsize=AX_FS, fontweight='bold')
    axA.set_ylabel('Pair count', fontsize=AX_FS, fontweight='bold')
    axA.text(0.98, 0.97, f"bin acc = {C['bin_acc']:.3f}\nMAE = {C['ang_mae']:.2f} Å",
             transform=axA.transAxes, ha='right', va='top', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbb'))
    axA.tick_params(labelsize=TK_FS); _despine(axA)

    axB = fig.add_subplot(g1[0, 2]); _letter(axB, 'B', -0.34, 0)
    vals = [C['ifp_acc'], C['ifp_f1'], C['ifp_auc']]
    axB.bar([0, 1, 2], vals, color=['#2A9D8F', '#E9C46A', '#E76F51'], edgecolor='white', width=0.9)
    axB.set_xticks([0, 1, 2]); axB.set_xticklabels(['Accuracy', 'F1', 'ROC-AUC'], fontweight='bold')
    axB.set_xlim(-0.55, 2.55)
    for i, v in enumerate(vals):
        axB.text(i, v + 0.012, f'{v:.3f}', ha='center', fontsize=22, fontweight='bold')
    axB.set_ylim(0, 1.1); axB.set_ylabel('Score', fontsize=AX_FS, fontweight='bold')
    axB.tick_params(axis='y', labelsize=TK_FS); axB.tick_params(axis='x', labelsize=20); _despine(axB)

    # C: distogram of the curated pair: True / Pred / Error (agreement-style)
    g1c = g1[0, 4].subgridspec(3, 2, width_ratios=[1.0, 0.025], wspace=0.04, hspace=0.20)
    axC_full = fig.add_subplot(g1[0, 4]); axC_full.axis('off'); _letter(axC_full, 'C', -0.04, 0)
    axCt = fig.add_subplot(g1c[0, 0]); axCp = fig.add_subplot(g1c[1, 0], sharex=axCt)
    axCe = fig.add_subplot(g1c[2, 0], sharex=axCt)
    tb0, pb0, m0mask = C['ex_adp']
    Lp = np.where(m0mask.any(1))[0]; Ln = np.where(m0mask.any(0))[0]
    st = tb0[np.ix_(Lp, Ln)].astype(float).T; spr = pb0[np.ix_(Lp, Ln)].astype(float).T
    vmn = float(min(st.min(), spr.min())); vmx = float(max(st.max(), spr.max()))
    imt = axCt.imshow(st, aspect='auto', cmap='viridis', vmin=vmn, vmax=vmx)
    axCp.imshow(spr, aspect='auto', cmap='viridis', vmin=vmn, vmax=vmx)
    d = np.abs(st - spr)
    imE = axCe.imshow(d, aspect='auto', cmap='Blues', vmin=0, vmax=float(d.max()))
    for ax_, lab_ in ((axCt, 'True'), (axCp, 'Pred'), (axCe, 'Error')):
        ax_.set_ylabel(lab_, fontsize=24, fontweight='bold', labelpad=4)
        ax_.tick_params(labelsize=TK_FS)
    axC_full.text(-0.115, 0.5, 'Ligand atom', transform=axC_full.transAxes, rotation=90,
                  ha='center', va='center', fontsize=AX_FS, fontweight='bold')
    axCe.set_xlabel('Protein residue', fontsize=AX_FS, fontweight='bold')
    axCe.set_xticks([0, 50, 100, 200]); axCe.tick_params(axis='x', labelsize=20)
    plt.setp(axCt.get_xticklabels(), visible=False); plt.setp(axCp.get_xticklabels(), visible=False)
    cax = fig.add_subplot(g1c[0:2, 1])
    cbC = fig.colorbar(imt, cax=cax)
    cbC.set_label('Distance bin', fontsize=AX_FS, fontweight='bold'); cbC.ax.tick_params(labelsize=TK_FS - 2)
    caxE = fig.add_subplot(g1c[2, 1])
    cbE = fig.colorbar(imE, cax=caxE)
    cbE.set_label('|Error| bin', fontsize=AX_FS, fontweight='bold'); cbE.ax.tick_params(labelsize=TK_FS - 2)

    # ============ Row 1: D PyMOL cartoons (pred / true / error), full width ============
    g1d = outer[1].subgridspec(1, 3, wspace=0.05)
    cart = [(C['paths']['p_pred'], C['paths']['pz_pred'], 'Predicted', DIST_CMAP, C['vmin'], C['vmax'], 'Distance (Å)'),
            (C['paths']['p_true'], C['paths']['pz_true'], 'Ground truth', DIST_CMAP, C['vmin'], C['vmax'], 'Distance (Å)'),
            (C['paths']['p_err'], C['paths']['pz_err'], 'Error', ERR_CMAP, 0.0, C['emax'], 'Error (Å)')]
    for j, (png, pngz, ttl, cmp, lo, hi, lab) in enumerate(cart):
        axc = fig.add_subplot(g1d[0, j])
        if j == 0:
            axD_full = fig.add_subplot(outer[1]); axD_full.axis('off'); _letter(axD_full, 'D', None, 1)
        axc.imshow(plt.imread(png)); axc.axis('off'); axc.set_anchor('N')
        axc.set_title(ttl, fontsize=40, fontweight='bold', pad=10)
        _pocket_inset(axc, pngz)
        _cbar(fig, axc, cmp, lo, hi, lab)

    # ============ Row 2: E IFP heatmaps (larger) | F docking ============
    g2 = outer[2].subgridspec(1, 3, width_ratios=[0.04, 2.2, 1.7], wspace=0.22)
    g2L = g2[0, 1].subgridspec(3, 1, hspace=0.50)
    for row, (M, ttl, cmap, discrete) in enumerate([
            (C['Pm'], 'Predicted', 'Blues', False),
            (C['Tm'], 'Ground truth', 'Blues', False),
            (C['agree'], 'Agreement', None, True)]):
        ax = fig.add_subplot(g2L[row, 0])
        Mt = M.transpose(1, 0, 2) if discrete else M.T
        if discrete:
            ax.imshow(Mt, aspect='auto')
        else:
            im = ax.imshow(Mt, aspect='auto', cmap=cmap, vmin=0, vmax=1)
            cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.ax.tick_params(labelsize=20)
        ax.set_yticks(range(6)); ax.set_yticklabels(IFP_TYPES, fontsize=21, fontweight='bold')
        ax.set_xticks(range(len(C['rn'])))
        if row == 2:
            ax.set_xticklabels(C['rn'], rotation=0, ha='center', fontsize=24, fontweight='bold')
        else:
            ax.set_xticklabels([''] * len(C['rn']))
        ax.set_title(ttl, fontsize=TITLE_FS + 2, fontweight='bold', pad=5)
        if row == 0:
            _letter(ax, 'E', None, 2)
    axF = fig.add_subplot(g2[0, 2]); _letter(axF, 'F', -0.02, 2)
    axF.imshow(plt.imread(C['p_ifp3d'])); axF.axis('off')
    axF.legend(handles=[Patch(color=[0.16, 0.62, 0.56], label='Correct'),
                        Patch(color=[0.90, 0.49, 0.13], label='Missed'),
                        Patch(color=[0.79, 0.36, 0.37], label='False positive')],
               loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=22,
               prop={'weight': 'bold', 'size': 22}, frameon=True)

    # ============ Cross-attention row: G entropy | H attention ============
    g3 = outer[3].subgridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.30)
    axG = fig.add_subplot(g3[0, 0]); _letter(axG, 'G', None, 3)
    axG.hist(C['ent_A'], bins=30, alpha=0.72, label='Substrate to protein', color='#457B9D')
    axG.hist(C['ent_B'], bins=30, alpha=0.72, label='Protein to substrate', color='#E76F51')
    axG.set_xlabel('Mean attention entropy (nats)', fontsize=AX_FS, fontweight='bold')
    axG.set_ylabel('Sample count', fontsize=AX_FS, fontweight='bold')
    axG.tick_params(labelsize=TK_FS); axG.legend(prop={'weight': 'bold', 'size': 24}); _despine(axG)
    g3h = g3[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.04)
    axH = fig.add_subplot(g3h[0, 0]); _letter(axH, 'H', -0.20, 3)
    imH = axH.imshow(C['ex_attn'], aspect='auto', cmap='magma')
    axH.set_xlabel('Protein residue', fontsize=AX_FS, fontweight='bold')
    axH.set_ylabel('Substrate atom', fontsize=AX_FS, fontweight='bold')
    axH.tick_params(labelsize=TK_FS)
    cbH = fig.colorbar(imH, cax=fig.add_subplot(g3h[0, 1]))
    cbH.set_label('Attention weight', fontsize=AX_FS, fontweight='bold'); cbH.ax.tick_params(labelsize=TK_FS - 2)

    # ============ Distance row: I scatter | J rmsd ============
    g4 = outer[4].subgridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.30)
    axI = fig.add_subplot(g4[0, 0]); _letter(axI, 'I', None, 4)
    axI.scatter(C['td'], C['pdv'], s=16, color='#1D6A96', alpha=0.6, edgecolor='none')
    lim = [0, max(C['td'].max(), C['pdv'].max()) * 1.02]
    axI.plot(lim, lim, '--', color='#888', lw=1.4)
    axI.axvline(CONTACT_A, color='#E63946', ls=':', lw=1.6); axI.axhline(CONTACT_A, color='#E63946', ls=':', lw=1.6)
    axI.set_xlabel('True distance (Å)', fontsize=AX_FS, fontweight='bold')
    axI.set_ylabel('Predicted distance (Å)', fontsize=AX_FS, fontweight='bold')
    axI.text(0.04, 0.96, f"ρ = {C['sp']:.2f}\nAUC = {C['auc']:.2f}", transform=axI.transAxes,
             ha='left', va='top', fontsize=TK_FS + 1, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbb'))
    axI.tick_params(labelsize=TK_FS); _despine(axI); axI.grid(True, ls='--', alpha=0.4); axI.set_axisbelow(True)
    g4j = g4[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.04)
    axJ = fig.add_subplot(g4j[0, 0]); _letter(axJ, 'J', -0.20, 4)
    axJ.plot(range(len(C['rmsds'])), C['rmsds'], 'o-', color='#E63946', lw=2.6, ms=7)
    axJ.set_ylim(0, max(C['rmsds']) * 1.35)
    axJ.set_xlabel('Denoising step', fontsize=AX_FS, fontweight='bold')
    axJ.set_ylabel('Pocket RMSD (Å)', fontsize=AX_FS, fontweight='bold')
    axJ.tick_params(labelsize=TK_FS); _despine(axJ); axJ.grid(True, ls='--', alpha=0.5); axJ.set_axisbelow(True)

    fig.canvas.draw()
    row_y = {}
    for ax_, let_, dx_, row_ in _PEND:
        row_y[row_] = max(row_y.get(row_, 0.0), ax_.get_position().y1)
    for ax_, let_, dx_, row_ in _PEND:
        pos = ax_.get_position()
        if dx_ is None:
            fig.text(0.008, row_y[row_] + 0.004, let_, fontsize=LET_FS, fontweight='bold',
                     fontfamily='Arial', va='bottom', ha='left', zorder=50)
        else:
            fig.text(pos.x0 + dx_ * pos.width, row_y[row_] + 0.004, let_, fontsize=LET_FS,
                     fontweight='bold', fontfamily='Arial', va='bottom', ha='right', zorder=50)
    fig.savefig(os.path.join(OUT_MAIN, 'fig5_reconstruction.png'), dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUT_MAIN, 'fig5_reconstruction.pdf'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved fig5_reconstruction')


if __name__ == '__main__':
    draw(load())
