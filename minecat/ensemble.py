"""Three-view convex blend (tree / deep / composition) in log10 space.

Weights live on the 2-simplex (w_tree + w_deep + w_comp = 1, all >= 0, grid step 0.05,
231 points). They are chosen LEAVE-ONE-FOLD-OUT: the weights applied to test fold f
maximize pooled R^2 on the out-of-fold predictions of the other nine folds, so no
held-out pair informs its own blend weights.

Reference weights (mean over the 10 LOFO fits, cluster split; see results/):
    K_M  : tree 0.705 / deep 0.250 / composition 0.045
    k_cat: tree 0.565 / deep 0.265 / composition 0.170

Two presets are provided (WEIGHT_PRESETS), because the optimal weights depend on whether
the target pairs come from the same curation pipeline as the training corpus:

  "internal" (default)  the LOFO weights above. Appropriate for pairs drawn from the same
                        kind of source as the training data (BRENDA / SABIO-RK-derived
                        measurements, featurized by this repository's scripts).
  "external"            composition-heavy weights measured on four independent datasets
                        (HXKm, CatPred40, IECata, EnzyExtract). There the tree member's
                        optimal weight collapses to 0.00 and composition takes 0.60-0.75,
                        worth +0.06 to +0.21 R2 over the internal weights
                        (figures/source_data/fig7_external/blend_weight_sweep.csv).
                        The cause is feature-distribution shift, not novelty as such: the
                        domain classifier separates internal from external pairs almost
                        perfectly in the tree and deep feature spaces (AUC 0.91-0.98) but
                        much less so for composition descriptors (AUC 0.62-0.87), which
                        cannot encode protein identity.

NOTE ON SEQUENCE IDENTITY. Accuracy depends strongly on the MMseqs2 global identity between
a test enzyme and the training set (K_M R2 falls from ~0.49 at >=40% to ~0.35 below 10%),
but the optimal *weights* do not: refitting them per identity bin inside the cluster split
changes R2 by -0.017 to +0.006 (mean -0.002, i.e. no gain), and the composition weight
shows no trend with identity
(scripts/analysis/blend_weight_sensitivity.py reproduces the table). Likewise, on external
data the composition-heavy optimum holds at every novelty decile rather than ramping in.
So there is deliberately no identity-dependent weight schedule: pick the preset by data
provenance, not by sequence identity.
"""
import numpy as np
from sklearn.metrics import r2_score

MEMBERS = ("tree", "deep", "comp")

# mean LOFO weights from the 10-fold cluster cross-validation (results/ensemble_weights_*.csv)
DEFAULT_WEIGHTS = {
    "km": {"tree": 0.705, "deep": 0.250, "comp": 0.045},
    "kcat": {"tree": 0.565, "deep": 0.265, "comp": 0.170},
}

# leave-one-dataset-out optima over the four external datasets (blend_weight_sweep.csv);
# k_cat / (k_cat/K_M) are covered by EnzyExtract only, so they carry more uncertainty.
EXTERNAL_WEIGHTS = {
    "km": {"tree": 0.00, "deep": 0.30, "comp": 0.70},
    "kcat": {"tree": 0.00, "deep": 0.35, "comp": 0.65},
}

WEIGHT_PRESETS = {"internal": DEFAULT_WEIGHTS, "external": EXTERNAL_WEIGHTS}


def parse_weights(spec, task):
    """Resolve a weight specification for one task.

    `spec` is a preset name from WEIGHT_PRESETS, or three non-negative numbers
    "tree,deep,comp" (renormalized to sum to 1), e.g. "0.0,0.3,0.7".
    """
    if spec in WEIGHT_PRESETS:
        return dict(WEIGHT_PRESETS[spec][task])
    parts = [p for p in str(spec).replace("/", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--weights must be one of {sorted(WEIGHT_PRESETS)} or "
                         f"three numbers 'tree,deep,comp'; got {spec!r}")
    vals = [float(p) for p in parts]
    if min(vals) < 0 or sum(vals) <= 0:
        raise ValueError(f"blend weights must be non-negative and not all zero; got {vals}")
    total = sum(vals)
    return dict(zip(MEMBERS, [v / total for v in vals]))


def simplex_grid(step=0.05):
    n = int(round(1.0 / step))
    return np.array([(i * step, j * step, (n - i - j) * step)
                     for i in range(n + 1) for j in range(n + 1 - i)])


def best_weights(y, tree, deep, comp, step=0.05):
    """Grid search the simplex for the weights maximizing pooled R^2."""
    best_w, best_r = (1 / 3, 1 / 3, 1 / 3), -np.inf
    for wt, wd, wc in simplex_grid(step):
        r = r2_score(y, wt * tree + wd * deep + wc * comp)
        if r > best_r:
            best_r, best_w = r, (wt, wd, wc)
    return best_w, best_r


def lofo_blend(folds, y, tree, deep, comp, step=0.05):
    """Leave-one-fold-out blend. Returns (oof_blend, {fold: (w_tree, w_deep, w_comp)})."""
    folds = np.asarray(folds)
    out = np.empty(len(y), dtype=np.float64)
    weights = {}
    for f in np.unique(folds):
        te, other = folds == f, folds != f
        w, _ = best_weights(y[other], tree[other], deep[other], comp[other], step)
        out[te] = w[0] * tree[te] + w[1] * deep[te] + w[2] * comp[te]
        weights[int(f)] = w
    return out, weights


def blend(tree, deep, comp, weights):
    """Apply a {'tree','deep','comp'} weight dict to member predictions (log10)."""
    return (weights["tree"] * np.asarray(tree) + weights["deep"] * np.asarray(deep)
            + weights["comp"] * np.asarray(comp))
