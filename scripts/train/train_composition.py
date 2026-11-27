"""Train MineCat (composition) with 10-fold cluster cross-validation.

Hand-crafted, identity-independent features (2,782-d):
    protein descriptors AAC + CKSAAP + CTD (567) | MACCS (167) | Morgan r=2, 2048 bits

Model: ExtraTreesRegressor(n_estimators=1000, random_state=42) per fold, trained on the
8 training folds (no environment input, no stacking). Composition features cannot encode
protein identity, which makes this view degrade gracefully under distribution shift.

    python scripts/train/train_composition.py --task km
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from minecat import config  # noqa: E402
from minecat.data import (METRIC_KEYS, cluster_splits, denormalize, load_dataset,  # noqa: E402
                          load_feature, regression_metrics, summarize, task_labels)

COMPOSITION_BLOCKS = ("protein_desc", "maccs", "morgan")


def composition_features(feature_dir=None):
    return np.concatenate([load_feature(k, feature_dir) for k in COMPOSITION_BLOCKS],
                          axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=config.TASKS, required=True)
    ap.add_argument("--dataset", default=config.DATASET_CSV)
    ap.add_argument("--features", default=config.FEATURE_DIR)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "composition"))
    ap.add_argument("--ckpt-dir", default=os.path.join(config.WEIGHTS_DIR, "composition"))
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--n-estimators", type=int, default=1000)
    args = ap.parse_args()
    task = args.task
    ckpt_dir = os.path.join(args.ckpt_dir, task)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    df = load_dataset(args.dataset)
    y_log, y_norm, valid, lo, hi = task_labels(df, task)
    X = composition_features(args.features)[valid]
    folds = df["fold"].to_numpy()[valid]
    print(f"[composition/{task}] rows={len(valid)} features={X.shape[1]}")

    oof = np.full(len(df), np.nan)
    fold_metrics = []
    for f, (tr, va, te) in enumerate(cluster_splits(folds)):
        model = ExtraTreesRegressor(n_estimators=args.n_estimators, random_state=42, n_jobs=args.n_jobs)
        model.fit(X[tr], y_norm[tr])
        pred = denormalize(model.predict(X[te]), lo, hi)
        oof[valid[te]] = pred
        m = regression_metrics(y_log[te], pred)
        fold_metrics.append(m)
        joblib.dump(model, os.path.join(ckpt_dir, f"fold{f}.joblib"))
        print(f"  fold {f}: " + "  ".join(f"{k}={m[k]:.4f}" for k in METRIC_KEYS), flush=True)

    s = summarize(fold_metrics)
    print("mean +/- SD: " + "  ".join(f"{k}={s[k][0]:.4f}+/-{s[k][1]:.4f}" for k in METRIC_KEYS))
    pd.DataFrame({"pair_id": df["pair_id"], "fold": df["fold"],
                  f"{task}_pred_log10": oof}).to_csv(
        os.path.join(args.out, f"oof_predictions_{task}.csv"), index=False)
    pd.DataFrame([{"fold": i, **m} for i, m in enumerate(fold_metrics)]).to_csv(
        os.path.join(args.out, f"fold_metrics_{task}.csv"), index=False)


if __name__ == "__main__":
    main()
