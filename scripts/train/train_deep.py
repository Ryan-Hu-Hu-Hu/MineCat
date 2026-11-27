"""Train MineCat (deep) with 10-fold cluster cross-validation.

Protocol (as in the paper): for test fold f, fold (f + 1) % 10 is the validation fold
(early stopping, LR schedule) and the remaining 8 folds train. Each fold:

  * AdamW (lr 1e-4, weight decay 0.01), batch 128, dropout 0.4, co-modal dropout 0.10
  * 10-epoch linear warm-up, then ReduceLROnPlateau on the validation K_M loss
    (factor 0.5, patience 15); gradient clipping at 0.5
  * loss = Huber(K_M) + 0.7 * Huber(k_cat)
           + lambda(epoch) * [distogram CE + IFP BCE + pocket-denoising MSE],
    lambda(epoch) = max(0, 1 - epoch / 200)
  * early stopping on the validation K_M loss (patience 25, at most 400 epochs)

Outputs:
    weights/deep/fold{f}.pt                best-validation state_dict per fold (--ckpt-dir)
    results/deep/logs/training_log.tsv     per-epoch metrics (--out)
    results/deep/oof_predictions.csv       out-of-fold log10 predictions

    python scripts/train/train_deep.py                       # all folds
    python scripts/train/train_deep.py --folds 0,1           # a subset (e.g. per GPU)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from minecat import config  # noqa: E402
from minecat.data import (cluster_splits, denormalize, load_dataset, load_feature,  # noqa: E402
                          normalize_log10, regression_metrics)
from minecat.deep_dataset import (MineCatDeepDataset, collate, drop_virtual_node,  # noqa: E402
                                  encode_ph_temp, model_inputs, to_device)
from minecat.models import MineCatDeep  # noqa: E402
from minecat.models.deep import IFP_DIM, N_INT_TYPES  # noqa: E402

HP = dict(lr=1e-4, weight_decay=0.01, dropout=0.4, dim=64, nhead=4, batch_size=128,
          max_epochs=400, patience=25, warmup_epochs=10, lam_kcat=0.7, modality_dropout=0.10,
          aux_anneal_epochs=200, grad_clip=0.5)


def load_inputs(df, feature_dir):
    print("Loading feature store ...")
    F = {k: load_feature(k, feature_dir) for k in
         ("molt5", "esm2_mean", "ec2vec", "smiles_trfm", "molgnet_pooled")}
    F["esm2_residue"] = load_feature("esm2_residue", feature_dir)
    F["molgnet_atoms"] = {s: drop_virtual_node(v)
                          for s, v in load_feature("molgnet_atoms", feature_dir).items()}
    for k in ("adp", "ifp", "pocket"):
        F[k] = load_feature(k, feature_dir)
    print(f"  pockets available: {sum(p is not None for p in F['pocket'])}/{len(df)}")
    return F


def make_dataset(df, F, ph_temp, km_norm, km_mask, kc_norm, kc_mask):
    return MineCatDeepDataset(
        df["Sequence"].tolist(), df["Smiles"].tolist(), F["molt5"], F["esm2_mean"],
        F["ec2vec"], ph_temp, F["smiles_trfm"], F["molgnet_pooled"], F["esm2_residue"],
        F["molgnet_atoms"], F["adp"], F["ifp"], F["pocket"], km_norm, km_mask, kc_norm, kc_mask)


def aux_lambda(epoch):
    return max(0.0, 1.0 - epoch / HP["aux_anneal_epochs"])


def train_epoch(model, loader, opt, epoch, device, huber, ce):
    model.train()
    lam = aux_lambda(epoch)
    for batch in loader:
        b = to_device(batch, device)
        opt.zero_grad()
        km_p, kc_p, adp, ifp_logits, denoise, noises = model(
            *model_inputs(b), modality_dropout=HP["modality_dropout"])
        kmv = b["km_mask"].bool()
        if not kmv.any():
            continue
        loss = huber(km_p[kmv], b["km"][kmv])
        kcv = b["kcat_mask"].bool()
        if kcv.any():
            l_kc = huber(kc_p[kcv], b["kcat"][kcv])
            if torch.isfinite(l_kc):
                loss = loss + HP["lam_kcat"] * l_kc
        if lam > 0:
            B, P, L, K = adp.shape
            adp_ce = ce(adp.reshape(-1, K), b["adp_bins"].reshape(-1)).reshape(B, P, L)
            loss = loss + lam * (adp_ce * b["adp_mask"]).sum() / b["adp_mask"].sum().clamp(min=1)
            m = b["ifp_mask"].unsqueeze(-1).expand(-1, -1, N_INT_TYPES).reshape(-1, IFP_DIM)
            bce = nn.functional.binary_cross_entropy_with_logits(ifp_logits, b["ifp"], reduction="none")
            loss = loss + lam * (bce * m).sum() / m.sum().clamp(min=1)
            dn = [((d - n) ** 2 * pm.unsqueeze(-1)).sum() / pm.sum().clamp(1)
                  for d, n, pm in zip(denoise, noises, b["pocket_mask"])
                  if d is not None and n is not None]
            if dn:
                loss = loss + lam * torch.stack(dn).mean()
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=HP["grad_clip"])
        opt.step()


@torch.no_grad()
def predict(model, loader, device):
    """Return normalized (km, kcat, km_label, km_mask, kcat_label, kcat_mask, index)."""
    model.eval()
    cols = {k: [] for k in ("km_p", "kc_p", "km", "km_mask", "kcat", "kcat_mask", "index")}
    for batch in loader:
        b = to_device(batch, device)
        km_p, kc_p, *_ = model(*model_inputs(b))
        for k, v in (("km_p", km_p), ("kc_p", kc_p), ("km", b["km"]), ("km_mask", b["km_mask"]),
                     ("kcat", b["kcat"]), ("kcat_mask", b["kcat_mask"]), ("index", b["index"])):
            cols[k].append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in cols.items()}


def evaluate(out, ranges, huber_fn):
    kmv, kcv = out["km_mask"] > 0, out["kcat_mask"] > 0
    loss = float(huber_fn(torch.tensor(out["km_p"][kmv]), torch.tensor(out["km"][kmv]))) if kmv.any() else float("nan")
    (lo, hi), (klo, khi) = ranges
    km = regression_metrics(denormalize(out["km"][kmv], lo, hi), denormalize(out["km_p"][kmv], lo, hi))
    kc = regression_metrics(denormalize(out["kcat"][kcv], klo, khi), denormalize(out["kc_p"][kcv], klo, khi))
    return loss, km, kc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DATASET_CSV)
    ap.add_argument("--features", default=config.FEATURE_DIR)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "deep"))
    ap.add_argument("--ckpt-dir", default=os.path.join(config.WEIGHTS_DIR, "deep"))
    ap.add_argument("--folds", default=",".join(map(str, range(config.N_FOLDS))))
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=HP["max_epochs"])
    args = ap.parse_args()
    HP["max_epochs"] = args.max_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)

    df = load_dataset(args.dataset)
    F = load_inputs(df, args.features)
    km_norm, km_mask, lo, hi = normalize_log10(df["km_log10"].to_numpy())
    kc_norm, kc_mask, klo, khi = normalize_log10(df["kcat_log10"].to_numpy())
    ranges = ((lo, hi), (klo, khi))
    print(f"N={len(df)}  K_M rows={int(km_mask.sum())}  k_cat rows={int(kc_mask.sum())}")
    ph, tc = df["ph"].to_numpy(np.float32), df["temp_c"].to_numpy(np.float32)
    splits = cluster_splits(df["fold"].to_numpy())
    huber = nn.HuberLoss(delta=1.0)
    ce = nn.CrossEntropyLoss(reduction="none")

    log_path = os.path.join(args.out, "logs", "training_log.tsv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("fold\tepoch\tseconds\tlr\tval_loss\t" + "\t".join(
                f"{t}_{s}_{m}" for s in ("val", "test") for t in ("km", "kcat")
                for m in ("R2", "PCC", "SCC")) + "\n")

    oof_path = os.path.join(args.out, "oof_predictions.csv")
    oof = (pd.read_csv(oof_path) if os.path.exists(oof_path) else pd.DataFrame({
        "pair_id": df["pair_id"], "fold": df["fold"], "km_pred_log10": np.nan,
        "kcat_pred_log10": np.nan}))

    for fold in [int(x) for x in args.folds.split(",")]:
        tr, va, te = splits[fold]
        print(f"\n===== fold {fold}: train={len(tr)} val={len(va)} test={len(te)} =====")
        ph_temp = encode_ph_temp(ph, tc, float(np.nanmedian(ph[tr])), float(np.nanmedian(tc[tr])))
        ds = make_dataset(df, F, ph_temp, km_norm, km_mask, kc_norm, kc_mask)
        def loader(idx, shuffle):
            return DataLoader(Subset(ds, idx), batch_size=HP["batch_size"], shuffle=shuffle,
                              collate_fn=collate, num_workers=args.num_workers)
        tr_ld, va_ld, te_ld = loader(tr, True), loader(va, False), loader(te, False)

        model = MineCatDeep(dim=HP["dim"], nhead=HP["nhead"], dropout=HP["dropout"]).to(device)
        opt = optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
        warm = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0,
                                           total_iters=HP["warmup_epochs"])
        plateau = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=15,
                                                       factor=0.5, min_lr=1e-7)
        best, best_state, wait, t0 = float("inf"), None, 0, time.time()
        for epoch in range(1, HP["max_epochs"] + 1):
            train_epoch(model, tr_ld, opt, epoch, device, huber, ce)
            v_loss, v_km, v_kc = evaluate(predict(model, va_ld, device), ranges, huber)
            _, t_km, t_kc = evaluate(predict(model, te_ld, device), ranges, huber)
            if epoch <= HP["warmup_epochs"]:
                warm.step()
            else:
                plateau.step(v_loss)
            with open(log_path, "a") as f:
                vals = [v[m] for v in (v_km, v_kc, t_km, t_kc) for m in ("R2", "PCC", "SCC")]
                f.write(f"{fold}\t{epoch}\t{time.time() - t0:.0f}\t{opt.param_groups[0]['lr']:.2e}\t"
                        f"{v_loss:.5f}\t" + "\t".join(f"{x:.4f}" for x in vals) + "\n")
            if epoch == 1 or epoch % 10 == 0:
                print(f"  ep{epoch:3d} val_loss={v_loss:.4f} val R2 km={v_km['R2']:.3f} "
                      f"| test R2 km={t_km['R2']:.3f} kcat={t_kc['R2']:.3f}", flush=True)
            if v_loss < best:
                best, wait = v_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if wait >= HP["patience"] or math.isnan(v_loss):
                break
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(args.ckpt_dir, f"fold{fold}.pt"))
        out = predict(model, te_ld, device)
        _, t_km, t_kc = evaluate(out, ranges, huber)
        print(f"  fold {fold} best-val model: test R2 K_M={t_km['R2']:.4f}  k_cat={t_kc['R2']:.4f}")
        rows = out["index"]
        oof.loc[rows, "km_pred_log10"] = denormalize(out["km_p"], lo, hi)
        oof.loc[rows, "kcat_pred_log10"] = denormalize(out["kc_p"], klo, khi)
        oof.to_csv(oof_path, index=False)
        ds = model = None  # release memory before the next fold
    print(f"\nDone. Out-of-fold predictions -> {oof_path}")


if __name__ == "__main__":
    main()
