"""
Figure S13 - in-distribution reference for MineCat (ensemble): fold-mean test R2 under the
leakage-free 40%-identity cluster split vs a random 10-fold split, for K_M and k_cat.

Source data: figures/source_data/cluster_vs_random.csv (figures/make_source_data.py)
Output:      figures/supplementary/figS13_cluster_vs_random.{png,pdf}
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_SUPP, src  # noqa: E402
from plotting_style import savefig, use_arial  # noqa: E402

use_arial()
_T = pd.read_csv(src("cluster_vs_random.csv")).round(3)
CLUSTER = dict(zip(_T[_T.split == "cluster"].task, _T[_T.split == "cluster"].R2_mean))  # 40% seq-id (OOD)
RANDOM = dict(zip(_T[_T.split == "random"].task, _T[_T.split == "random"].R2_mean))     # in-distribution
C_CLU, C_RND = "#1D6A96", "#E8833A"


def main():
    tasks = ["km", "kcat"]
    labels = [r"$K_M$", r"$k_{cat}$"]
    x = np.arange(len(tasks)); w = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 7))
    b1 = ax.bar(x - w / 2, [CLUSTER[t] for t in tasks], w, color=C_CLU,
                edgecolor="black", linewidth=0.8, label="Cluster split")
    b2 = ax.bar(x + w / 2, [RANDOM[t] for t in tasks], w, color=C_RND,
                edgecolor="black", linewidth=0.8, label="Random split")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                    f"{r.get_height():.3f}", ha="center", va="bottom",
                    fontsize=22, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=30, fontweight="bold")
    ax.set_ylabel("Test R²", fontsize=28, fontweight="bold")
    ax.set_ylim(0, 0.95)
    ax.tick_params(axis="y", labelsize=22)
    ax.legend(fontsize=24, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              prop={"weight": "bold", "size": 24}, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.45); ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, OUT_SUPP, "figS13_cluster_vs_random")


if __name__ == "__main__":
    main()
