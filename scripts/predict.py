"""Predict K_M and k_cat for new enzyme-substrate pairs with MineCat (ensemble).

Input CSV columns:
    Sequence, Smiles            required
    EC                          recommended (EC2Vec; blank -> zero vector)
    ph, temp_c                  optional assay conditions (both needed for the env arm)

Workflow:
    # 1) (optional, recommended) cofold the complexes with chai-1
    python scripts/data/fold_structures.py --csv pairs.csv --out structures/
    # 2) features (+ MolGNet in its own environment)
    python scripts/data/build_features.py --csv pairs.csv --structures structures/ --out pairs_features/
    python scripts/data/build_molgnet_features.py --csv pairs.csv --out pairs_features/
    # 3) predict
    python scripts/predict.py --csv pairs.csv --features pairs_features/ --out predictions.csv

Each view averages its 10 cross-validation fold models (log10 space); the views are then
blended with the cross-validated ensemble weights. Rows with both pH and temperature use
the environment-aware arm (deep pH/T encoding, tree stacking cascade); other rows use the
environment-free arm (zeroed pH/T encoding, tree base model). Without a chai-1 structure the
deep pocket modality is masked.

Blend weights (--blend, --blend-km, --blend-kcat) take a preset or an explicit
"tree,deep,comp" triple:
  internal  (default) the leave-one-fold-out weights of the cluster cross-validation;
                      use for pairs from the same kind of source as the training corpus.
  external            composition-heavy weights measured on four independent datasets,
                      where the tree member's optimum collapses to 0 (+0.06 to +0.21 R2
                      there). Use when the pairs come from a different curation pipeline.
See minecat/ensemble.py for provenance. Sequence identity to the training set changes
accuracy a great deal but not the optimal weights, so it does not select the preset.

Expected weights layout (see weights/README.md):
    weights/deep/fold{0-9}.pt
    weights/tree/{km,kcat}/fold{0-9}.joblib           {'base', 'first', 'second'}
    weights/composition/{km,kcat}/fold{0-9}.joblib
    weights/model_meta.json                           label ranges + condition medians

Output columns: log10 K_M [M], log10 k_cat [s^-1], log10 k_cat/K_M [M^-1 s^-1], the
across-fold SD of each ensemble prediction, and every view's own prediction.
"""
import argparse
import gc
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minecat import config  # noqa: E402
from minecat.data import denormalize, environment_matrix, load_feature  # noqa: E402
from minecat.deep_dataset import (MineCatDeepDataset, collate, drop_virtual_node,  # noqa: E402
                                  encode_ph_temp, model_inputs, to_device)
from minecat.ensemble import WEIGHT_PRESETS, parse_weights  # noqa: E402
from minecat.models import MineCatDeep  # noqa: E402

TREE_BLOCKS = ("prot_t5", "smiles_trfm", "molt5", "molgnet_pooled", "ec2vec")
COMP_BLOCKS = ("protein_desc", "maccs", "morgan")


def env_arm(df):
    if {"ph", "temp_c"} <= set(df.columns):
        return (np.isfinite(pd.to_numeric(df["ph"], errors="coerce"))
                & np.isfinite(pd.to_numeric(df["temp_c"], errors="coerce"))).to_numpy()
    return np.zeros(len(df), dtype=bool)


@torch.no_grad()
def predict_deep(df, feat_dir, wdir, meta, folds, use_env, device, batch_size):
    F = {k: load_feature(k, feat_dir) for k in
         ("molt5", "esm2_mean", "ec2vec", "smiles_trfm", "molgnet_pooled")}
    res = load_feature("esm2_residue", feat_dir)
    atoms = {s: drop_virtual_node(v) for s, v in load_feature("molgnet_atoms", feat_dir).items()}
    n = len(df)
    struct = {}
    for k in ("adp", "ifp", "pocket"):
        p = config.feature_path(k, feat_dir)
        struct[k] = load_feature(k, feat_dir) if os.path.exists(p) else [None] * n
    ph = pd.to_numeric(df.get("ph", pd.Series([np.nan] * n)), errors="coerce").to_numpy(np.float32)
    tc = pd.to_numeric(df.get("temp_c", pd.Series([np.nan] * n)), errors="coerce").to_numpy(np.float32)
    ph_temp = encode_ph_temp(ph, tc, meta["ph_median"], meta["temp_median"])
    ph_temp[~use_env] = 0.0                       # environment-free arm
    ds = MineCatDeepDataset(df["Sequence"].tolist(), df["Smiles"].tolist(), F["molt5"],
                            F["esm2_mean"], F["ec2vec"], ph_temp, F["smiles_trfm"],
                            F["molgnet_pooled"], res, atoms, struct["adp"], struct["ifp"],
                            struct["pocket"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    (lo, hi), (klo, khi) = meta["label_range"]["km"], meta["label_range"]["kcat"]
    km, kc = np.zeros((len(folds), n)), np.zeros((len(folds), n))
    for j, f in enumerate(folds):
        model = MineCatDeep().to(device)
        model.load_state_dict(torch.load(os.path.join(wdir, "deep", f"fold{f}.pt"), map_location=device))
        model.eval()
        ptr = 0
        for batch in loader:
            k1, k2, *_ = model(*model_inputs(to_device(batch, device)))
            b = k1.shape[0]
            km[j, ptr:ptr + b] = denormalize(k1.cpu().numpy(), lo, hi)
            kc[j, ptr:ptr + b] = denormalize(k2.cpu().numpy(), klo, khi)
            ptr += b
        print(f"  deep fold {f} done", flush=True)
        del model
    return {"km": km, "kcat": kc}


def predict_trees(X, env, use_env, wdir, meta, folds, stage, kind):
    out = {}
    for task in config.TASKS:
        lo, hi = meta["label_range"][task]
        preds = np.zeros((len(folds), len(X)))
        for j, f in enumerate(folds):
            ck = joblib.load(os.path.join(wdir, kind, task, f"fold{f}.joblib"))
            if kind == "composition":
                p = ck.predict(X)
            else:
                base = ck["base"].predict(X)
                first = ck["first"].predict(np.concatenate([X, env], axis=1))
                env_p = (ck["second"].predict(np.stack([base, first], axis=1))
                         if stage == "cascade" else first)
                p = np.where(use_env, env_p, base)
            preds[j] = denormalize(p, lo, hi)
            del ck
            gc.collect()
            print(f"  {kind} {task} fold {f} done", flush=True)
        out[task] = preds
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=config.WEIGHTS_DIR,
                    help="directory holding the trained fold models (see weights/README.md)")
    ap.add_argument("--folds", default=",".join(map(str, range(config.N_FOLDS))),
                    help="fold models to average (default: all 10)")
    ap.add_argument("--tree-stage", choices=["cascade", "first"], default="cascade",
                    help="env arm of the tree: full stacking cascade (as in cross-validation) "
                         "or the env-aware ExtraTrees alone")
    ap.add_argument("--blend", default="internal",
                    help=f"ensemble blend weights for both tasks: {sorted(WEIGHT_PRESETS)} "
                         f"or 'tree,deep,comp' (e.g. '0.0,0.3,0.7')")
    ap.add_argument("--blend-km", default=None, help="override --blend for K_M")
    ap.add_argument("--blend-kcat", default=None, help="override --blend for k_cat")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    folds = [int(x) for x in args.folds.split(",")]
    with open(os.path.join(args.weights, "model_meta.json")) as f:
        meta = json.load(f)
    use_env = env_arm(df)
    blend = {t: parse_weights(getattr(args, f"blend_{t}") or args.blend, t)
             for t in config.TASKS}
    print(f"{len(df)} pairs ({int(use_env.sum())} with assay conditions), folds={folds}")
    for t in config.TASKS:
        w = blend[t]
        print(f"  {t:5s} blend weights tree/deep/comp = "
              f"{w['tree']:.3f}/{w['deep']:.3f}/{w['comp']:.3f}")

    env = (environment_matrix(df, meta["ph_median"], meta["temp_median"]) if use_env.any()
           else np.tile([meta["ph_median"], meta["temp_median"]], (len(df), 1)).astype(np.float32))
    X_tree = np.concatenate([load_feature(k, args.features) for k in TREE_BLOCKS], axis=1)
    X_comp = np.concatenate([load_feature(k, args.features) for k in COMP_BLOCKS], axis=1)

    print("[deep]")
    deep = predict_deep(df, args.features, args.weights, meta, folds, use_env,
                        torch.device(args.device), args.batch_size)
    print("[tree]")
    tree = predict_trees(X_tree, env, use_env, args.weights, meta, folds, args.tree_stage, "tree")
    print("[composition]")
    comp = predict_trees(X_comp, None, use_env, args.weights, meta, folds, None, "composition")

    out = pd.DataFrame(index=df.index)
    for c in ("pair_id", "complex_id", "name", "id"):
        if c in df.columns:
            out[c] = df[c]
    ens = {}
    for task in config.TASKS:
        w = blend[task]
        per_fold = w["tree"] * tree[task] + w["deep"] * deep[task] + w["comp"] * comp[task]
        ens[task] = per_fold.mean(0)
        out[f"{task}_log10"] = ens[task]
        out[f"{task}_log10_sd"] = per_fold.std(0)
        for name, arr in (("tree", tree), ("deep", deep), ("comp", comp)):
            out[f"{task}_log10_{name}"] = arr[task].mean(0)
    out["kcat_km_log10"] = ens["kcat"] - ens["km"]
    out["environment_arm"] = np.where(use_env, "env", "no_env")
    for task in config.TASKS:
        w = blend[task]
        out[f"{task}_blend_weights"] = f"{w['tree']:.3f}/{w['deep']:.3f}/{w['comp']:.3f}"
    out.round(4).to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
