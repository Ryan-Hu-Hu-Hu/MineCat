# Cross-validation results

Every file is keyed by `pair_id` (row of `data/minecat_dataset.csv.gz`). Predictions are
log<sub>10</sub> K<sub>M</sub> [M] and log<sub>10</sub> k<sub>cat</sub> [s<sup>-1</sup>]; each row's
prediction comes from the fold model for which that row was held out.

## `cluster_cv/` (10-fold, 40%-identity cluster split - the paper's main evaluation)

| file | content |
|---|---|
| `oof_predictions.csv.gz` | `{task}_true_log10`, and out-of-fold predictions of `{task}_tree`, `{task}_deep`, `{task}_comp`, `{task}_ensemble` (3-way LOFO blend) and `{task}_ensemble_noenv` (condition-free ensemble) |
| `ensemble_weights_{km,kcat}.csv` | leave-one-fold-out blend weights (tree, deep, composition) per test fold |
| `benchmark_oof_predictions.csv.gz` | out-of-fold predictions of nine published predictors retrained on the same folds (EF-UniKP, UniKP, PreTKcat, KinForm, CataPro, MPEK, KmPred, CatPred, DLTKcat) |
| `benchmark_fold_metrics.csv` | per-fold test metrics logged when the published predictors were retrained (used for Figure 2) |
| `test_train_identity.csv.gz` | global sequence identity of each test enzyme to its fold's training set (`scripts/analysis/test_train_identity.py`; Figure 4) |

`python scripts/train/build_ensemble.py --from-shipped` recomputes the LOFO weights and the
headline metrics (K<sub>M</sub> R<sup>2</sup> 0.4114, k<sub>cat</sub> R<sup>2</sup> 0.2267) from `oof_predictions.csv.gz`.

## `random_cv/`

`oof_predictions.csv.gz`: the same views under a random 10-fold split (`fold_random`), the
in-distribution reference of Figure S13.
