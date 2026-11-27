"""Dataset loading, label normalization, cluster cross-validation splits and metrics.

The protocol is shared by all three MineCat views (and by every benchmark model in the
paper), so that out-of-fold predictions are directly comparable:

* labels are log10(K_M [M]) and log10(k_cat [s^-1]); rows may carry one or both;
* labels are min-max normalized to [-1, 1] over all labelled rows of a task;
* fold f is the test fold, fold (f + 1) % 10 is the validation fold (used for early
  stopping of the deep model and for the tree stacking layer), the other 8 folds train.
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config

METRIC_KEYS = ["R2", "PCC", "SCC", "MAE", "RMSE"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def load_dataset(csv_path=None):
    """Load the MineCat dataset (48,891 enzyme-substrate-condition rows).

    Columns used by the code: pair_id, EC, Sequence, Smiles, km_log10, kcat_log10,
    has_km, has_kcat, ph, temp_c, fold.
    """
    return pd.read_csv(csv_path or config.DATASET_CSV)


def load_feature(key, feature_dir=None, dtype=np.float32):
    """Load one row-aligned feature array (or dict / list for token & structure features)."""
    obj = load_pickle(config.feature_path(key, feature_dir))
    if isinstance(obj, (dict, list)):
        return obj
    return np.asarray(obj, dtype=dtype)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def normalize_log10(log_vals):
    """Min-max normalize a log10 array (NaN = absent) to [-1, 1].

    Returns (norm, mask, lo, hi); absent rows get norm 0 and mask 0. Labels are stored as
    float32 (as in the original training run) before normalization.
    """
    log_vals = np.asarray(log_vals, dtype=np.float32).astype(np.float64)
    valid = np.isfinite(log_vals)
    lo, hi = float(log_vals[valid].min()), float(log_vals[valid].max())
    norm = np.zeros(len(log_vals), dtype=np.float32)
    if hi - lo > 1e-8:
        norm[valid] = (2.0 * (log_vals[valid] - lo) / (hi - lo) - 1.0).astype(np.float32)
    return norm, valid.astype(np.float32), lo, hi


def denormalize(y, lo, hi):
    return (np.asarray(y) + 1.0) / 2.0 * (hi - lo) + lo


def task_labels(df, task):
    """Per-task labelled rows for the tree models.

    Returns (labels_log10[valid], y_norm[valid], valid_idx, lo, hi).
    """
    col, mask_col = {"km": ("km_log10", "has_km"), "kcat": ("kcat_log10", "has_kcat")}[task]
    full = df[col].to_numpy(dtype=np.float64)
    valid_idx = np.where(df[mask_col].to_numpy().astype(bool) & np.isfinite(full))[0]
    y = full[valid_idx].astype(np.float32)
    lo, hi = float(y.min()), float(y.max())
    y_norm = (2.0 * (y - lo) / (hi - lo) - 1.0).astype(np.float32)
    return y, y_norm, valid_idx, lo, hi


def label_ranges(df):
    """(lo, hi) normalization bounds per task, needed to de-normalize model outputs."""
    out = {}
    for task, col in (("km", "km_log10"), ("kcat", "kcat_log10")):
        v = df[col].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        out[task] = (float(v.min()), float(v.max()))
    return out


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cluster_splits(folds, n_folds=config.N_FOLDS, verbose=True):
    """Rotating-validation cluster splits: test = f, val = (f + 1) % n, train = rest."""
    folds = np.asarray(folds)
    splits = []
    for f in range(n_folds):
        v = (f + 1) % n_folds
        te = np.where(folds == f)[0]
        va = np.where(folds == v)[0]
        tr = np.where((folds != f) & (folds != v))[0]
        splits.append((tr, va, te))
        if verbose:
            print(f"  fold {f} (val = fold {v}): train={len(tr)} val={len(va)} test={len(te)}")
    return splits


def environment_matrix(df, ph_fill=None, temp_fill=None):
    """(N, 2) [pH, temperature degC] with missing values filled by the dataset medians."""
    ph = df["ph"].to_numpy(dtype=np.float32)
    tc = df["temp_c"].to_numpy(dtype=np.float32)
    ph_fill = np.nanmedian(ph) if ph_fill is None else ph_fill
    temp_fill = np.nanmedian(tc) if temp_fill is None else temp_fill
    return np.stack([np.where(np.isfinite(ph), ph, ph_fill),
                     np.where(np.isfinite(tc), tc, temp_fill)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 2 or np.unique(y_true).size < 2:
        return {k: float("nan") for k in METRIC_KEYS}
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "PCC": float(pearsonr(y_true, y_pred)[0]),
        "SCC": float(spearmanr(y_true, y_pred)[0]),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def per_fold_metrics(folds, y_true, y_pred):
    """List of metric dicts, one per fold (the paper reports the mean +/- SD over folds)."""
    folds = np.asarray(folds)
    return [regression_metrics(np.asarray(y_true)[folds == f], np.asarray(y_pred)[folds == f])
            for f in np.unique(folds)]


def summarize(fold_metrics):
    return {k: (float(np.nanmean([m[k] for m in fold_metrics])),
                float(np.nanstd([m[k] for m in fold_metrics]))) for k in METRIC_KEYS}
