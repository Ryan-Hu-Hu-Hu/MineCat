"""Train MineCat (tree) with 10-fold cluster cross-validation.

Features (4,864-d, row-aligned frozen embeddings):
    ProtT5-XL mean (1024) | in-house SMILES Transformer (1024) | MolT5-large (1024)
    | MolGNet pooled (768) | EC2Vec (1024)

Per test fold f (validation fold v = (f + 1) % 10, training = the other 8 folds):
    base   = ExtraTrees(features)                    -> fit on training folds
    first  = ExtraTrees([features, pH, T])           -> fit on training folds
    second = LinearRegression([base, first])         -> fit on the validation fold
    prediction = second([base(x), first(x, env)])    (the env-aware stacking cascade)

ExtraTreesRegressor(n_estimators=1000, random_state=42), labels min-max normalized to
[-1, 1] per task. NOTE: fully grown 1,000-tree forests on 4,866-d inputs are large
(~5-9 GB per fold on disk); use --ckpt-dir to place them on a big disk.

    python scripts/train/train_tree.py --task km
    python scripts/train/train_tree.py --task kcat --n-jobs 32
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from minecat import config  # noqa: E402
from minecat.data import (METRIC_KEYS, cluster_splits, denormalize, environment_matrix,  # noqa: E402
                          load_dataset, load_feature, regression_metrics, summarize, task_labels)

TREE_BLOCKS = ("prot_t5", "smiles_trfm", "molt5", "molgnet_pooled", "ec2vec")
RANDOM_STATE = 42


def tree_features(feature_dir=None):
    return np.concatenate([load_feature(k, feature_dir) for k in TREE_BLOCKS], axis=1).astype(np.float32)


def cascade_predict(ckpt, X, env):
    """Full stacking cascade (used for the reported cross-validation results)."""
    base = ckpt["base"].predict(X).reshape(-1, 1)
    first = ckpt["first"].predict(np.concatenate([X, env], axis=1)).reshape(-1, 1)
    return ckpt["second"].predict(np.concatenate([base, first], axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=config.TASKS, required=True)
    ap.add_argument("--dataset", default=config.DATASET_CSV)
    ap.add_argument("--features", default=config.FEATURE_DIR)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "tree"))
    ap.add_argument("--ckpt-dir", default=os.path.join(config.WEIGHTS_DIR, "tree"))
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--n-estimators", type=int, default=1000)
    args = ap.parse_args()
    task = args.task
    ckpt_dir = os.path.join(args.ckpt_dir, task)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    df = load_dataset(args.dataset)
    y_log, y_norm, valid, lo, hi = task_labels(df, task)
    X = tree_features(args.features)[valid]
    env = environment_matrix(df)[valid]
    folds = df["fold"].to_numpy()[valid]
    print(f"[tree/{task}] rows={len(valid)} features={X.shape[1]} label range=[{lo:.3f}, {hi:.3f}]")

    oof = np.full(len(df), np.nan)
    fold_metrics = []
    for f, (tr, va, te) in enumerate(cluster_splits(folds)):
        et = lambda: ExtraTreesRegressor(n_estimators=args.n_estimators,  # noqa: E731
                                         random_state=RANDOM_STATE, n_jobs=args.n_jobs)
        base = et().fit(X[tr], y_norm[tr])
        first = et().fit(np.concatenate([X[tr], env[tr]], axis=1), y_norm[tr])
        stack_in = np.stack([base.predict(X[va]),
                             first.predict(np.concatenate([X[va], env[va]], axis=1))], axis=1)
        second = LinearRegression().fit(stack_in, y_norm[va])
        ckpt = {"base": base, "first": first, "second": second}
        pred = denormalize(cascade_predict(ckpt, X[te], env[te]), lo, hi)
        oof[valid[te]] = pred
        m = regression_metrics(y_log[te], pred)
        fold_metrics.append(m)
        joblib.dump(ckpt, os.path.join(ckpt_dir, f"fold{f}.joblib"))
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
