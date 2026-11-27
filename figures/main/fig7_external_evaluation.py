"""
Figure 7 and Figure S12 - generalization to independent external datasets.

Figure 7
  A-F  per-fold Spearman correlation (mean +/- fold SD) of MineCat views/blends and the
       published predictors on six external test sets that do not overlap the training data
       (HXKm K_M, CatPred40 K_M, IECata k_cat/K_M, EnzyExtract K_M / k_cat / k_cat/K_M);
       every model contributes one prediction per fold model (10 fold models).
  G-K  UMAP of internal (grey) vs external (coloured) pairs in the feature space of each view
       (deep: fused input, K_M head, k_cat head; composition; tree).
Figure S12 (feature-shift mechanism)
  A    blend-weight simplex: optimal external weights per dataset/task vs the internal weights
  B    domain-classifier AUC (internal vs external) in each member's feature space
  C, D per-decile R2 and locally optimal blend weights vs learned-space novelty
  E-H  calibration slope, |bias|, model variance and RMSE vs learned-space novelty

Source data: figures/source_data/fig7_external/
    leaderboard_per_fold_scc.csv  blend_weight_sweep.csv  member_domain_shift.csv
    ood_crossover.csv  ood_calibration.csv  domain_auc_by_space.csv  umap_coords.csv.gz
Output: figures/main/fig7_external_evaluation.{png,pdf}
        figures/supplementary/figS12_feature_shift.{png,pdf}
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_MAIN, OUT_SUPP, src  # noqa: E402
from plotting_style import use_arial  # noqa: E402

use_arial()
SRC = src("fig7_external")
LB_YTICK_FS, LB_XTICK_FS, LB_VALUE_FS = 16, 16, 14   # leaderboard panels A-F (as published)
TLAB = {"km": r"$K_m$", "kcat": r"$k_{cat}$", "kcatkm": r"$k_{cat}/K_m$"}
LB_COL = {"MineCat-3way[naive13]": "#E8833A", "MineCat-3way[compheavy]": "#7A1128", "MineCat-comp": "#2A9D8F",
          "MineCat-deep": "#4C72B0", "MineCat-tree": "#C44E52", "MineCat-3way[internal]": "#8172B3",
          "MineCat-2way[internal]": "#937860"}
LB_DISP = {"MineCat-3way[naive13]": "MineCat (naive)", "MineCat-3way[compheavy]": "MineCat (comp-heavy)",
           "MineCat-comp": "MineCat (comp)", "MineCat-deep": "MineCat (deep)", "MineCat-tree": "MineCat (tree)",
           "MineCat-3way[internal]": "MineCat (3-way, int-w)", "MineCat-2way[internal]": "MineCat (2-way, int-w)"}
LB_LEG = ["MineCat-3way[naive13]", "MineCat-3way[compheavy]", "MineCat-comp", "MineCat-deep",
          "MineCat-tree", "MineCat-3way[internal]", "MineCat-2way[internal]"]
EXTC = {"HXKm": "#1b9e77", "CatPred40": "#d95f02", "iecata": "#7570b3", "enzyextract": "#e7298a"}
PTS = {("HXKm", "km"): ("o", "#4C72B0"), ("CatPred40", "km"): ("s", "#55A868"),
       ("iecata", "kcatkm"): ("^", "#8172B3"), ("enzyextract", "km"): ("D", "#C44E52"),
       ("enzyextract", "kcat"): ("P", "#CCB974"), ("enzyextract", "kcatkm"): ("v", "#64B5CD")}
MCOL = {"tree": "#c1443c", "deep": "#e0922f", "comp": "#2A9D8F"}
Av = np.array([0., 0.]); Bv = np.array([1., 0.]); Cv = np.array([0.5, np.sqrt(3) / 2])
_LET = iter("abcdefghijklmnopqrstuvwxyz")


def L(ax, dx=-0.015, dy=1.05):
    ax.text(dx, dy, f"{next(_LET)}", transform=ax.transAxes, fontsize=34, fontweight="bold",
            va="bottom", ha="right")


DS_DISP = {"HXKm": "HXKm", "CatPred40": "CatPred40", "iecata": "IECata", "enzyextract": "Enzyextract"}


def _set_letters(s):
    global _LET
    _LET = iter(s)


def bary(w): return w[0] * Av + w[1] * Bv + w[2] * Cv


def main():
    df_lb = pd.read_csv(f"{SRC}/leaderboard_per_fold_scc.csv")
    sw = pd.read_csv(f"{SRC}/blend_weight_sweep.csv")
    sf = sw[sw.method == "blend_self"].copy(); sf["w3"] = sf["w"].apply(lambda s: [float(x) for x in s.split("/")])
    msh = pd.read_csv(f"{SRC}/member_domain_shift.csv")
    cal = pd.read_csv(f"{SRC}/ood_calibration.csv")
    cross = pd.read_csv(f"{SRC}/ood_crossover.csv")
    um = pd.read_csv(f"{SRC}/umap_coords.csv.gz")

    # ===== Fig 9 (main): A leaderboard (a-f) + B UMAP (g-k) =====
    fig = plt.figure(figsize=(23, 28.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.8, 2.4], hspace=0.16,
                             top=0.96, bottom=0.04, left=0.05, right=0.985)
    gsA = outer[0].subgridspec(2, 3, hspace=0.24, wspace=0.58)
    # 2x3 grid: G,H,(legend) on top; I,J,K below. G/I and H/J column-aligned.
    gsB = outer[1].subgridspec(2, 3, hspace=0.30, wspace=0.05)
    _set_letters("ABCDEFGHIJK")          # A-F (leaderboard) then G-K (UMAP)
    # ===== supplementary quantification figure (panels a-h) =====
    fig2 = plt.figure(figsize=(30, 15.5))
    gsC = fig2.add_gridspec(2, 4, hspace=0.62, wspace=0.42,
                            top=0.95, bottom=0.12, left=0.05, right=0.985)

    # ---------- A: full leaderboard ----------
    combos = [["HXKm", "km"], ["CatPred40", "km"], ["iecata", "kcatkm"],
              ["enzyextract", "km"], ["enzyextract", "kcat"], ["enzyextract", "kcatkm"]]
    lb_bottom = []
    for idx, (ds, task) in enumerate(combos):
        ax = fig.add_subplot(gsA[idx // 3, idx % 3])
        sub = df_lb[(df_lb.dataset == ds) & (df_lb.task == task)].sort_values("SCC").reset_index(drop=True)
        y = np.arange(len(sub)); colors = [LB_COL.get(m, "#9fb3c8") for m in sub.model]
        ax.barh(y, sub.SCC, xerr=sub.SCC_std, color=colors, edgecolor="black", linewidth=0.6,
                error_kw=dict(ecolor="#444", lw=1.1, capsize=2.5))
        ax.set_yticks(y); ax.set_yticklabels([LB_DISP.get(m, m) for m in sub.model], fontsize=LB_YTICK_FS, fontweight="bold")
        top = sub.SCC.max(); top_std = sub.loc[sub.SCC.idxmax(), "SCC_std"]
        for yi, r in sub.iterrows():
            ax.text(r.SCC + r.SCC_std + 0.008, yi, f"{r.SCC:.3f}", va="center", fontsize=LB_VALUE_FS, fontweight="bold")
            if r.model in LB_COL:
                ax.get_yticklabels()[yi].set_color(LB_COL[r.model])
        ax.axvspan(top - top_std, top + 0.2, color="#fff3cd", alpha=0.35, zorder=0)
        ax.axvline(top, ls="--", color="#555", lw=0.9, alpha=0.6); ax.set_xlim(0, top + top_std + 0.12)
        ax.set_title(f"{DS_DISP.get(ds, ds)} · {TLAB[task]}  (n={int(sub.n.max()):,})", fontsize=21, fontweight="bold")
        if idx // 3 == 1:
            ax.set_xlabel("per-fold SCC (mean ± fold σ)", fontsize=18.5, fontweight="bold")
        ax.tick_params(axis="x", labelsize=LB_XTICK_FS); ax.spines[["top", "right"]].set_visible(False)
        L(ax, dx=-0.26)
        if idx // 3 == 1:
            lb_bottom.append(ax)
    lb_handles = [Patch(fc=LB_COL[m], label=LB_DISP[m]) for m in LB_LEG] + [Patch(fc="#9fb3c8", label="Benchmarks")]

    # ---------- B: UMAP ----------
    # 5 panels: deep member == its fused concatenated layer, so merged into one (no duplicate)
    umap_panels = [("deep_input", "MineCat (deep) — concatenated"),
                   ("deep_km_expert", r"MineCat (deep) — $K_m$ head"),
                   ("deep_kcat_expert", r"MineCat (deep) — $k_{cat}$ head"),
                   ("comp", "MineCat (comp)"), ("tree", "MineCat (tree)")]
    # G,H share the top UMAP row (half width each); I,J,K fill the bottom row
    UMAP_SCALE = 0.80                     # fraction of cell width used by each square
    umap_cells = [gsB[0, 0], gsB[0, 1], gsB[0, 2], gsB[1, 0], gsB[1, 1]]
    umap_top_axes = []
    for i, (panel, title) in enumerate(umap_panels):
        cell = umap_cells[i].subgridspec(1, 2, width_ratios=[UMAP_SCALE, 1 - UMAP_SCALE], wspace=0)
        ax = fig.add_subplot(cell[0, 0])
        ax.set_box_aspect(1)              # all UMAP panels same square size
        if i < 3:
            umap_top_axes.append(ax)
        sub_um = um[um.space == panel]
        if len(sub_um):
            emb = sub_um[["umap_1", "umap_2"]].to_numpy(); lab = sub_um["dataset"].astype(str).to_numpy()
            m = lab == "internal"
            ax.scatter(emb[m, 0], emb[m, 1], s=4, c="#d6d6d6", alpha=0.40, linewidths=0,
                       label="internal", rasterized=True)
            # draw larger external sets first and SUBSAMPLE so no single dataset
            # (e.g. enzyextract, ~20k pts) buries the others or the internal manifold
            rng = np.random.default_rng(0)
            order = sorted(["HXKm", "CatPred40", "iecata", "enzyextract"],
                           key=lambda d: -(lab == d).sum())
            for ds in order:
                md = np.where(lab == ds)[0]
                if md.size == 0:
                    continue
                if md.size > 1500:
                    md = rng.choice(md, 1500, replace=False)
                ax.scatter(emb[md, 0], emb[md, 1], s=7, c=EXTC[ds], alpha=0.6,
                           linewidths=0, label=DS_DISP.get(ds, ds), rasterized=True)
        else:
            ax.text(0.5, 0.5, f"{panel}\n(coords pending)", ha="center", va="center", transform=ax.transAxes, color="red")
        ax.set_title(title, fontsize=22, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([]); L(ax, dy=1.0 if i >= 3 else 1.05)

    # model legend (A-F) centered in the band between the D-F row and the G/H row,
    # placed slightly closer to D-F and enlarged.
    fig.canvas.draw()
    df_bottom = min(a.get_position().y0 for a in lb_bottom) - 0.028   # below the x-labels
    gh_top = max(a.get_position().y1 for a in umap_top_axes) + 0.020  # above the G/H titles
    leg_y = 0.70 * df_bottom + 0.30 * gh_top
    fig.legend(handles=lb_handles, loc="center", ncol=8, frameon=True,
               bbox_to_anchor=(0.5, leg_y), prop={"weight": "bold", "size": 20})
    # shared dataset legend in the empty top-right UMAP cell (right of H, above K)
    gk_handles = [Line2D([0], [0], marker="o", ls="", ms=20, mfc="#d6d6d6", mec="none", label="Internal")]
    gk_handles += [Line2D([0], [0], marker="o", ls="", ms=20, mfc=EXTC[ds], mec="none",
                          label=DS_DISP.get(ds, ds)) for ds in ["enzyextract", "iecata", "CatPred40", "HXKm"]]
    ax_leg = fig.add_subplot(gsB[1, 2]); ax_leg.axis("off")
    ax_leg.legend(handles=gk_handles, loc="center left", bbox_to_anchor=(0.0, 0.5), ncol=1, frameon=True,
                  prop={"weight": "bold", "size": 26}, labelspacing=0.9, handletextpad=0.6)

    fig.savefig(f"{OUT_MAIN}/fig7_external_evaluation.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT_MAIN}/fig7_external_evaluation.pdf", bbox_inches="tight", facecolor="white")
    print("saved fig7_external_evaluation")

    # ---------- C: quantification (S14) ----------
    dss = ["HXKm", "CatPred40", "iecata", "enzyextract"]
    DS_LBL = ["HXKm", "CatPred40", "IECata", "Enzyextract"]
    MEMBERS = ["tree", "deep", "comp"]
    MEM_LBL = {"tree": "Tree", "deep": "Deep", "comp": "Comp"}
    _set_letters("ABCDEFGH")
    PEND2 = []
    axS = fig2.add_subplot(gsC[0, 0]); PEND2.append((axS, next(_LET), 0))
    tri = np.array([Av, Bv, Cv, Av]); axS.plot(tri[:, 0], tri[:, 1], color="#333", lw=1.6, zorder=1)
    for f in np.arange(0.2, 1.0, 0.2):
        axS.plot(*zip((1 - f) * Av + f * Bv, (1 - f) * Av + f * Cv), color="#ddd", lw=0.7, zorder=0)
        axS.plot(*zip((1 - f) * Bv + f * Av, (1 - f) * Bv + f * Cv), color="#ddd", lw=0.7, zorder=0)
        axS.plot(*zip((1 - f) * Cv + f * Av, (1 - f) * Cv + f * Bv), color="#ddd", lw=0.7, zorder=0)
    for corner, lab, dx, dy, ha in [(Av, "tree", -0.03, -0.05, "right"), (Bv, "deep", 0.03, -0.05, "left"),
                                    (Cv, "comp", 0.0, 0.04, "center")]:
        axS.text(corner[0] + dx, corner[1] + dy, lab, fontsize=30, fontweight="bold", ha=ha, va="center")
    for _, r in sf.iterrows():
        mk, col = PTS.get((r.dataset, r.task), ("o", "#1D6A96"))
        axS.scatter(*bary(r.w3), s=420, marker=mk, color=col, edgecolor="black", linewidth=1.0, zorder=4)
    INT_W = {"km": [0.71, 0.25, 0.04], "kcat": [0.57, 0.27, 0.17]}
    for w in INT_W.values():
        axS.scatter(*bary(w), s=900, marker="*", color="#111", edgecolor="white", linewidth=1.3, zorder=7)
    hleg = [Line2D([0], [0], marker="*", color="w", markerfacecolor="#111", markeredgecolor="white",
                   markersize=24, label="Internal")]
    hleg += [Line2D([0], [0], marker=mk, color="w", markerfacecolor=col, markeredgecolor="black",
                    markersize=17, label=f"{DS_LBL[dss.index(ds)]}·{TLAB[tk]}") for (ds, tk), (mk, col) in PTS.items()]
    axS.legend(handles=hleg, fontsize=21, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2,
               frameon=True, prop={"weight": "bold", "size": 21})
    axS.set_xlim(-0.12, 1.10); axS.set_ylim(-0.02, 1.02); axS.set_aspect("equal"); axS.axis("off")

    axd = fig2.add_subplot(gsC[0, 1]); PEND2.append((axd, next(_LET), 0)); x = np.arange(len(dss)); bw = 0.26
    for j, mm in enumerate(MEMBERS):
        axd.bar(x + (j - 1) * bw, [msh[(msh.space == mm) & (msh.ds == d)].domain_AUC.iloc[0] for d in dss],
                bw, color=MCOL[mm], edgecolor="black", linewidth=0.5, label=MEM_LBL[mm])
    axd.axhline(0.5, ls=":", color="#777"); axd.set_xticks(x)
    axd.set_xticklabels(DS_LBL, rotation=22, fontsize=22, fontweight="bold")
    axd.tick_params(axis="y", labelsize=22)
    axd.set_ylabel("Domain AUC (int↔ext)", fontsize=26, fontweight="bold"); axd.set_ylim(0.5, 1.0)
    axd.spines[["top", "right"]].set_visible(False)

    axo = fig2.add_subplot(gsC[0, 2]); PEND2.append((axo, next(_LET), 0))
    for mm in MEMBERS:
        axo.plot(cross["ood"], cross[f"R2_{mm}"], "-o", color=MCOL[mm], label=MEM_LBL[mm], lw=3.2, ms=13)
    axo.set_xlabel("Learned-space OOD decile", fontsize=26, fontweight="bold")
    axo.set_ylabel("Per-decile R²", fontsize=26, fontweight="bold")
    axo.tick_params(labelsize=22); axo.spines[["top", "right"]].set_visible(False)
    axo.legend(prop={"weight": "bold", "size": 22}, loc="lower left", frameon=True)

    axp = fig2.add_subplot(gsC[0, 3]); PEND2.append((axp, next(_LET), 0))
    if {"w_tree", "w_deep", "w_comp"}.issubset(cross.columns):
        axp.stackplot(cross["ood"], cross["w_tree"], cross["w_deep"], cross["w_comp"],
                      colors=[MCOL[m] for m in MEMBERS], labels=[MEM_LBL[m] for m in MEMBERS])
        axp.set_ylim(0, 1)
    axp.set_xlabel("Learned-space OOD decile", fontsize=26, fontweight="bold")
    axp.set_ylabel("Local optimal weight", fontsize=26, fontweight="bold")
    axp.tick_params(labelsize=22); axp.spines[["top", "right"]].set_visible(False)

    def linep(cell, ycol, ylab, absval=False, hln=None):
        ax = fig2.add_subplot(cell)
        for mm in MEMBERS:
            v = cal[f"{ycol}_{mm}"]; v = v.abs() if absval else v
            ax.plot(cal["ood"], v, "-o", color=MCOL[mm], label=MEM_LBL[mm], lw=3.2, ms=13)
        if hln is not None:
            ax.axhline(hln, ls=":", color="#777")
        ax.set_xlabel("Learned-space OOD (z)", fontsize=26, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=26, fontweight="bold")
        ax.tick_params(labelsize=22)
        ax.spines[["top", "right"]].set_visible(False)
        return ax
    for cell, ycol, ylab, absv, hl, let in [
            (gsC[1, 0], "slope", "Calibration slope (1=ideal)", False, 1.0, "E"),
            (gsC[1, 1], "bias", "|Bias|", True, None, "F"),
            (gsC[1, 2], "var", "Model variance", False, None, "G"),
            (gsC[1, 3], "rmse", r"RMSE (log$_{10}$ $K_m$)", False, None, "H")]:
        ax_ = linep(cell, ycol, ylab, absval=absv, hln=hl)
        ax_.legend(prop={"weight": "bold", "size": 22}, loc="upper left", frameon=True)
        PEND2.append((ax_, let, 1))

    shared_h = [Patch(fc=MCOL[m], label=MEM_LBL[m]) for m in MEMBERS]
    for ax_ in (axd, axp):
        ax_.legend(handles=shared_h, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
                   frameon=True, prop={"weight": "bold", "size": 22}, handletextpad=0.5)
    fig2.canvas.draw()
    row_top = {}
    for ax_, let_, row_ in PEND2:
        row_top[row_] = max(row_top.get(row_, 0.0), ax_.get_position().y1)
    for ax_, let_, row_ in PEND2:
        pos = ax_.get_position()
        fig2.text(pos.x0 - 0.02, row_top[row_] + 0.02, let_, fontsize=42, fontweight="bold",
                  va="bottom", ha="right")
    fig2.savefig(f"{OUT_SUPP}/figS12_feature_shift.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig2.savefig(f"{OUT_SUPP}/figS12_feature_shift.pdf", bbox_inches="tight", facecolor="white")
    print("saved figS12_feature_shift")


if __name__ == "__main__":
    main()
