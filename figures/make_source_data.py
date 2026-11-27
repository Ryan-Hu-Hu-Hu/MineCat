"""Derive the figure source-data tables from the released cross-validation results.

    python figures/make_source_data.py

Writes into figures/source_data/:
    benchmark_metrics.csv          Fig 2, Fig S7   fold-mean +/- SD of R2/PCC/SCC/MAE/RMSE
    error_correlation_{task}.csv   Fig 3B          Pearson correlation of prediction errors
    seqid_metrics.csv              Fig 4, Fig S9   pooled metrics vs. test-train identity cutoff
    blend_landscape_{task}.csv     Fig S6          pooled R2 over the (tree, deep, comp) simplex
    cluster_vs_random.csv          Fig S13         ensemble R2, cluster vs random split
Figures 5-7 and S10-S12 use precomputed source data shipped in source_data/ (model
internals, UMAP coordinates and external benchmarks; see figures/README.md).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ENV_GROUP, RESULTS, load_predictions, src  # noqa: E402

TASKS = ("km", "kcat")
METRICS = ["R2", "PCC", "SCC", "MAE", "RMSE"]
CUTOFFS = [40, 35, 30, 25, 20, 15, 10]


def metrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    if len(yt) < 3 or len(np.unique(yt)) < 2:
        return {m: np.nan for m in METRICS}
    return {"R2": r2_score(yt, yp), "PCC": pearsonr(yt, yp)[0],
            "SCC": spearmanr(yt, yp).correlation, "MAE": mean_absolute_error(yt, yp),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp)))}


def labelled(d, task):
    t, p = f"{task}_true_log10", f"{task}_pred_log10"
    return d[d[f"has_{task}"].astype(bool) & np.isfinite(d[p]) & np.isfinite(d[t])]


def benchmark_metrics(preds):
    """MineCat views: recomputed per fold from the out-of-fold predictions. Published
    predictors: the per-fold test metrics logged when each was retrained on our folds
    (results/cluster_cv/benchmark_fold_metrics.csv; identical to recomputing them from
    benchmark_oof_predictions.csv.gz up to 4-decimal rounding, except DLTKcat k_cat whose
    prediction dump differs from its training log on folds 0-6)."""
    logged = pd.read_csv(os.path.join(RESULTS, "cluster_cv", "benchmark_fold_metrics.csv"))
    rows = []
    for task in TASKS:
        for name, d in preds.items():
            lg = logged[(logged.model == name) & (logged.task == task)]
            if len(lg):
                per_fold = lg[METRICS].to_dict("records")
            else:
                d = labelled(d, task)
                per_fold = [metrics(g[f"{task}_true_log10"], g[f"{task}_pred_log10"])
                            for _, g in d.groupby("fold")]
            row = {"task": task, "model": name, "env_group": ENV_GROUP[name]}
            for m in METRICS:
                v = [pf[m] for pf in per_fold]
                row[f"{m}_mean"], row[f"{m}_std"] = round(float(np.mean(v)), 4), round(float(np.std(v)), 4)
            rows.append(row)
    return pd.DataFrame(rows)


ERROR_CORR_ORDER = ["MineCat (ensemble)", "MineCat (tree)", "MineCat (deep)", "MineCat (composition)",
                    "CataPro", "KmPred", "UniKP", "EF-UniKP", "KinForm", "CatPred", "MPEK",
                    "PreTKcat", "DLTKcat"]


def error_correlation(preds, task):
    names = ERROR_CORR_ORDER
    errs = {}
    for name in names:
        d = labelled(preds[name], task)
        errs[name] = pd.Series((d[f"{task}_pred_log10"] - d[f"{task}_true_log10"]).values,
                               index=d["pair_id"].values)
    short = {"MineCat (ensemble)": "MineCat (ens)", "MineCat (composition)": "MineCat (comp)"}
    ef = pd.DataFrame(errs).dropna().rename(columns=short)
    return ef.corr(method="pearson")


def seqid_metrics(preds):
    ident = pd.read_csv(os.path.join(RESULTS, "cluster_cv", "test_train_identity.csv.gz"))
    gid = ident["max_global_identity_to_train"].to_numpy()
    rows = []
    for task in TASKS:
        for cutoff in CUTOFFS:
            sel_id = np.ones(len(gid), bool) if cutoff == 40 else gid < cutoff
            for name, d in preds.items():
                if name == "MineCat (no env)":
                    continue
                assert (d["pair_id"].values == ident["pair_id"].values).all()
                t, p = d[f"{task}_true_log10"].values, d[f"{task}_pred_log10"].values
                sel = sel_id & d[f"has_{task}"].astype(bool).values & np.isfinite(p) & np.isfinite(t)
                rows.append({**metrics(t[sel], p[sel]), "task": task, "cutoff": cutoff,
                             "model": name, "n": int(sel.sum())})
    return pd.DataFrame(rows)


def blend_landscape(preds, task, step=0.02):
    d = {k: labelled(preds[k], task) for k in ("MineCat (tree)", "MineCat (deep)", "MineCat (composition)")}
    y = d["MineCat (tree)"][f"{task}_true_log10"].to_numpy()
    tr, dp, cp = (d[k][f"{task}_pred_log10"].to_numpy() for k in d)
    n = int(round(1 / step))
    grid = [(i * step, j * step, (n - i - j) * step) for i in range(n + 1) for j in range(n + 1 - i)]
    return pd.DataFrame([{"w_tree": round(a, 4), "w_deep": round(b, 4), "w_comp": round(c, 4),
                          "pooled_R2": r2_score(y, a * tr + b * dp + c * cp)} for a, b, c in grid])


def cluster_vs_random(cluster, random):
    rows = []
    for split, preds in (("cluster", cluster), ("random", random)):
        d = preds["MineCat (ensemble)"]
        for task in TASKS:
            dd = labelled(d, task)
            v = [r2_score(g[f"{task}_true_log10"], g[f"{task}_pred_log10"]) for _, g in dd.groupby("fold")]
            rows.append({"split": split, "task": task, "R2_mean": float(np.mean(v)), "R2_std": float(np.std(v))})
    return pd.DataFrame(rows)


def main():
    os.makedirs(src(), exist_ok=True)
    preds = load_predictions("cluster")
    benchmark_metrics(preds).to_csv(src("benchmark_metrics.csv"), index=False)
    for task in TASKS:
        error_correlation(preds, task).to_csv(src(f"error_correlation_{task}.csv"))
        blend_landscape(preds, task).to_csv(src(f"blend_landscape_{task}.csv"), index=False)
    seqid_metrics(preds).to_csv(src("seqid_metrics.csv"), index=False)
    cluster_vs_random(preds, load_predictions("random", include_benchmarks=False)).round(4).to_csv(
        src("cluster_vs_random.csv"), index=False)
    print(f"source data -> {src()}")


if __name__ == "__main__":
    main()
