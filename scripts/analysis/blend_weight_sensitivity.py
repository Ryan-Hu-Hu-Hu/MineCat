"""Do the optimal blend weights depend on sequence identity to the training set?

Accuracy clearly does: for K_M the fold-pooled R2 of MineCat (ensemble) falls from ~0.49
for test enzymes with >=40% global identity to a training enzyme down to ~0.35 below 10%.
This script asks the separate question of whether the *blend* should change with identity.

For each identity bin it reports
  * the in-bin optimum over the (tree, deep, composition) simplex (an upper bound: those
    weights are fitted on the same rows they are scored on),
  * a leave-one-fold-out refit inside the bin (weights for fold f fitted on the bin's other
    folds), which is the honest estimate of what an identity-aware schedule would buy, and
  * the shipped default weights.

Result on the released out-of-fold predictions: the LOFO refit ranges from -0.017 to +0.006
R2 against the default weights (mean -0.002, i.e. slightly worse on average), and the
composition weight shows no trend with identity, so the package deliberately offers no
identity-dependent schedule. What does change the optimum is
a change of data provenance - see minecat.ensemble.WEIGHT_PRESETS and Figure S12.

    python scripts/analysis/blend_weight_sensitivity.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from minecat import config  # noqa: E402
from minecat.ensemble import DEFAULT_WEIGHTS, best_weights  # noqa: E402

BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 101)]
MIN_ROWS = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(config.RESULTS_DIR, "cluster_cv"))
    ap.add_argument("--out", default=None, help="optional CSV output")
    args = ap.parse_args()

    oof = pd.read_csv(os.path.join(args.results, "oof_predictions.csv.gz"))
    ident = pd.read_csv(os.path.join(args.results, "test_train_identity.csv.gz"))
    m = oof.merge(ident, on="pair_id")
    gid = m["max_global_identity_to_train"].to_numpy()

    rows = []
    for task in config.TASKS:
        d = m[m[f"{task}_true_log10"].notna()]
        g = gid[d.index.to_numpy()]
        w0 = DEFAULT_WEIGHTS[task]
        print(f"\n=== {task}: optimal (tree/deep/comp) by identity to training set ===")
        print("  identity      n   in-bin optimum        R2_opt  R2_LOFO  R2_default")
        for lo, hi in BINS:
            s = d[(g >= lo) & (g < hi)]
            if len(s) < MIN_ROWS:
                print(f"  {lo:3d}-{hi:3d}%  {len(s):6d}   (too few rows)")
                continue
            y = s[f"{task}_true_log10"].to_numpy()
            tr, dp, cp = (s[f"{task}_{k}"].to_numpy() for k in ("tree", "deep", "comp"))
            folds = s["fold"].to_numpy()
            w, r2_opt = best_weights(y, tr, dp, cp)

            pred = np.empty(len(s))
            for f in np.unique(folds):
                te = folds == f
                other = ~te
                if other.sum() < 100 or te.sum() < 10:      # too small to refit: use default
                    wf = (w0["tree"], w0["deep"], w0["comp"])
                else:
                    wf, _ = best_weights(y[other], tr[other], dp[other], cp[other])
                pred[te] = wf[0] * tr[te] + wf[1] * dp[te] + wf[2] * cp[te]
            r2_lofo = r2_score(y, pred)
            r2_def = r2_score(y, w0["tree"] * tr + w0["deep"] * dp + w0["comp"] * cp)
            print(f"  {lo:3d}-{hi:3d}%  {len(s):6d}   {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"
                  f"          {r2_opt:.4f}   {r2_lofo:.4f}   {r2_def:.4f}")
            rows.append(dict(task=task, identity_low=lo, identity_high=hi, n=len(s),
                             w_tree=w[0], w_deep=w[1], w_comp=w[2], R2_in_bin_optimum=r2_opt,
                             R2_lofo_refit=r2_lofo, R2_default_weights=r2_def))
    out = pd.DataFrame(rows).round(4)
    gain = (out.R2_lofo_refit - out.R2_default_weights)
    print(f"\nLOFO refit vs default weights: {gain.min():+.4f} to {gain.max():+.4f} R2 "
          f"(mean {gain.mean():+.4f}) -> no identity-dependent schedule is warranted.")
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
