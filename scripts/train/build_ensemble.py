"""Blend the three MineCat views into MineCat (ensemble), leave-one-fold-out.

Reads the out-of-fold predictions written by train_tree.py, train_deep.py and
train_composition.py, chooses simplex weights (tree, deep, composition) for each test
fold on the other nine folds, and reports fold-mean metrics for every view.

    python scripts/train/build_ensemble.py
    # reproduce the paper numbers from the shipped out-of-fold predictions:
    python scripts/train/build_ensemble.py --from-shipped

Outputs (--out, default results/ensemble/):
    oof_predictions.csv          pair_id, fold, per-task truth / tree / deep / comp / ensemble
    ensemble_weights_{task}.csv  per-fold LOFO weights
    cv_summary.csv               fold-mean +/- SD metrics per view and task
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from minecat import config  # noqa: E402
from minecat.data import METRIC_KEYS, load_dataset, per_fold_metrics, summarize  # noqa: E402
from minecat.ensemble import lofo_blend  # noqa: E402

SHIPPED = os.path.join(config.RESULTS_DIR, "cluster_cv", "oof_predictions.csv.gz")


def member_predictions(task, args, df):
    """DataFrame [pair_id, fold, y, tree, deep, comp] for the labelled rows of `task`."""
    if args.from_shipped:
        s = pd.read_csv(SHIPPED)
        out = s[["pair_id", "fold", f"{task}_true_log10", f"{task}_tree",
                 f"{task}_deep", f"{task}_comp"]].copy()
        out.columns = ["pair_id", "fold", "y", "tree", "deep", "comp"]
        return out.dropna()
    R = config.RESULTS_DIR
    tree = pd.read_csv(os.path.join(R, "tree", f"oof_predictions_{task}.csv"))
    comp = pd.read_csv(os.path.join(R, "composition", f"oof_predictions_{task}.csv"))
    deep = pd.read_csv(os.path.join(R, "deep", "oof_predictions.csv"))
    out = pd.DataFrame({"pair_id": df["pair_id"], "fold": df["fold"],
                        "y": df[f"{task}_log10"].where(df[f"has_{task}"].astype(bool))})
    for name, t in (("tree", tree), ("deep", deep), ("comp", comp)):
        out = out.merge(t[["pair_id", f"{task}_pred_log10"]].rename(
            columns={f"{task}_pred_log10": name}), on="pair_id")
    return out.dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DATASET_CSV)
    ap.add_argument("--from-shipped", action="store_true",
                    help=f"use the released out-of-fold predictions ({SHIPPED})")
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "ensemble"))
    ap.add_argument("--step", type=float, default=0.05, help="simplex grid step")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load_dataset(args.dataset)

    merged, summary = None, []
    for task in config.TASKS:
        m = member_predictions(task, args, df)
        y, folds = m["y"].to_numpy(), m["fold"].to_numpy()
        tree, deep, comp = (m[c].to_numpy() for c in ("tree", "deep", "comp"))
        m["ensemble"], weights = lofo_blend(folds, y, tree, deep, comp, step=args.step)
        w = pd.DataFrame([(f, *v) for f, v in sorted(weights.items())],
                         columns=["fold", "w_tree", "w_deep", "w_comp"])
        w.to_csv(os.path.join(args.out, f"ensemble_weights_{task}.csv"), index=False)
        print(f"\n[{task}] n={len(m)}  mean LOFO weights tree/deep/comp = "
              f"{w.w_tree.mean():.3f}/{w.w_deep.mean():.3f}/{w.w_comp.mean():.3f}")
        for view in ("ensemble", "tree", "deep", "comp"):
            s = summarize(per_fold_metrics(folds, y, m[view].to_numpy()))
            summary.append({"task": task, "view": view,
                            **{f"{k}_mean": s[k][0] for k in METRIC_KEYS},
                            **{f"{k}_sd": s[k][1] for k in METRIC_KEYS}})
            print(f"  {view:9s} " + "  ".join(f"{k}={s[k][0]:.4f}+/-{s[k][1]:.3f}" for k in METRIC_KEYS))
        cols = m.rename(columns={c: f"{task}_{c}" for c in ("y", "tree", "deep", "comp", "ensemble")})
        cols = cols.rename(columns={f"{task}_y": f"{task}_true_log10"})
        merged = cols if merged is None else merged.merge(cols, on=["pair_id", "fold"], how="outer")
    merged.sort_values("pair_id").to_csv(os.path.join(args.out, "oof_predictions.csv"), index=False)
    pd.DataFrame(summary).round(4).to_csv(os.path.join(args.out, "cv_summary.csv"), index=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
