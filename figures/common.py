"""Paths and loaders shared by the figure scripts.

Inputs:
  results/cluster_cv/oof_predictions.csv.gz            MineCat views, 10-fold cluster CV
  results/cluster_cv/benchmark_oof_predictions.csv.gz  9 published predictors, same folds
  results/random_cv/oof_predictions.csv.gz             MineCat views, random 10-fold CV
  figures/source_data/                                 per-figure tables (see figures/README.md)
"""
import os
import sys

import numpy as np
import pandas as pd

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(FIG_DIR)
SOURCE = os.path.join(FIG_DIR, "source_data")
OUT_MAIN = os.path.join(FIG_DIR, "main")
OUT_SUPP = os.path.join(FIG_DIR, "supplementary")
RESULTS = os.path.join(REPO, "results")
sys.path.insert(0, FIG_DIR)

MINECAT_VIEWS = {            # display name -> column stem in oof_predictions.csv.gz
    "MineCat (ensemble)": "ensemble",
    "MineCat (tree)": "tree",
    "MineCat (deep)": "deep",
    "MineCat (composition)": "comp",
    "MineCat (no env)": "ensemble_noenv",
}
BENCHMARKS = ["EF-UniKP", "UniKP", "PreTKcat", "KinForm", "CataPro", "MPEK", "KmPred",
              "CatPred", "DLTKcat"]
# whether a predictor encodes reaction conditions (pH / temperature)
ENV_GROUP = {"MineCat (ensemble)": "w/ env", "MineCat (tree)": "w/ env",
             "MineCat (deep)": "w/ env", "MineCat (composition)": "w/ env",
             "MineCat (no env)": "w/o env", "EF-UniKP": "w/ env", "PreTKcat": "w/ env",
             "MPEK": "w/ env", "DLTKcat": "w/ env", "UniKP": "w/o env", "KinForm": "w/o env",
             "CataPro": "w/o env", "KmPred": "w/o env", "CatPred": "w/o env"}


def src(*parts):
    return os.path.join(SOURCE, *parts)


def load_predictions(split="cluster", include_benchmarks=True):
    """{model name: DataFrame[pair_id, fold, has_km, has_kcat, {task}_true_log10, {task}_pred_log10]}."""
    oof = pd.read_csv(os.path.join(RESULTS, f"{split}_cv", "oof_predictions.csv.gz"))
    if split == "random":
        oof = oof.rename(columns={"fold_random": "fold"})
    base = oof[["pair_id", "fold", "km_true_log10", "kcat_true_log10"]].copy()
    base["has_km"] = np.isfinite(base["km_true_log10"]).astype(int)
    base["has_kcat"] = np.isfinite(base["kcat_true_log10"]).astype(int)
    out = {}
    for name, stem in MINECAT_VIEWS.items():
        if f"km_{stem}" not in oof.columns:
            continue
        d = base.copy()
        d["km_pred_log10"], d["kcat_pred_log10"] = oof[f"km_{stem}"], oof[f"kcat_{stem}"]
        out[name] = d
    if include_benchmarks and split == "cluster":
        b = pd.read_csv(os.path.join(RESULTS, "cluster_cv", "benchmark_oof_predictions.csv.gz"))
        assert (b["pair_id"].values == base["pair_id"].values).all()
        for name in BENCHMARKS:
            d = base.copy()
            d["km_pred_log10"], d["kcat_pred_log10"] = b[f"{name}_km"], b[f"{name}_kcat"]
            out[name] = d
    return out
