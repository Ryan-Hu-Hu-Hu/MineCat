"""Build the MineCat feature store for a CSV of enzyme-substrate pairs.

Works for the training dataset and for new pairs alike: every feature is computed from
(Sequence, Smiles, EC) and, optionally, a chai-1 cofold of the complex, then written
row-aligned to the CSV (see minecat/config.py FEATURE_FILES for names and shapes).

Required CSV columns: Sequence, Smiles, EC   (EC may be blank -> zero EC2Vec vector)
Optional:             complex_id             (sub-directory name of the chai-1 output;
                                              default: row index)

MolGNet features need a separate environment (torch_geometric + MolGNet weights); build
them with scripts/data/build_molgnet_features.py into the same --out directory.

Examples
  # training set (structures folded with scripts/data/fold_structures.py)
  python scripts/data/build_features.py --csv data/minecat_dataset.csv.gz \
      --structures structures/ --out features/
  # new pairs, without structures (the pocket modality is then masked)
  python scripts/data/build_features.py --csv data/examples/example_pairs.csv --out my_features/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from minecat import config  # noqa: E402
from minecat.data import save_pickle  # noqa: E402
from minecat.features import composition, ec, protein, structure, substrate  # noqa: E402

ALL_GROUPS = ["esm2", "prot_t5", "molt5", "smiles_trfm", "composition", "ec2vec", "structure"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="output feature directory")
    ap.add_argument("--structures", default=None,
                    help="directory with <complex_id>/chai1_fold/pred.model_idx_0.cif "
                         "(written by scripts/data/fold_structures.py)")
    ap.add_argument("--cif-name", default="chai1_fold/pred.model_idx_0.cif",
                    help="CIF path inside each complex directory")
    ap.add_argument("--only", default=",".join(ALL_GROUPS),
                    help="comma list of feature groups to build: " + ",".join(ALL_GROUPS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    seqs = df["Sequence"].astype(str).tolist()
    smis = df["Smiles"].astype(str).tolist()
    groups = [g.strip() for g in args.only.split(",") if g.strip()]
    os.makedirs(args.out, exist_ok=True)
    out = lambda key: config.feature_path(key, args.out)  # noqa: E731
    print(f"{len(df)} rows, {len(set(seqs))} unique sequences, {len(set(smis))} unique SMILES")

    if "esm2" in groups:
        print("[ESM2-35M layer 11]")
        means, residues = protein.esm2_layer11(seqs, device=args.device)
        save_pickle(np.stack([means[s] for s in seqs]).astype(np.float32), out("esm2_mean"))
        save_pickle(residues, out("esm2_residue"))
    if "prot_t5" in groups:
        print("[ProtT5-XL]")
        t5 = protein.prot_t5_mean(seqs, device=args.device)
        save_pickle(np.stack([t5[s] for s in seqs]).astype(np.float32), out("prot_t5"))
    if "molt5" in groups:
        print("[MolT5-large]")
        mt = substrate.molt5_large(smis, device=args.device)
        save_pickle(np.stack([mt[s] for s in smis]).astype(np.float32), out("molt5"))
    if "smiles_trfm" in groups:
        print("[in-house SMILES Transformer]")
        st = substrate.smiles_transformer(smis, device=args.device)
        save_pickle(np.stack([st[s] for s in smis]).astype(np.float32), out("smiles_trfm"))
    if "composition" in groups:
        print("[composition descriptors + fingerprints]")
        desc = composition.protein_descriptors(seqs)
        maccs, morgan = composition.fingerprints(smis)
        save_pickle(np.stack([desc[s] for s in seqs]).astype(np.float32), out("protein_desc"))
        save_pickle(np.stack([maccs[s] for s in smis]).astype(np.float32), out("maccs"))
        save_pickle(np.stack([morgan[s] for s in smis]).astype(np.float32), out("morgan"))
    if "ec2vec" in groups:
        print("[EC2Vec]")
        ecs = df["EC"].fillna("").astype(str).tolist() if "EC" in df.columns else [""] * len(df)
        arr = ec.ec2vec(ecs)
        print(f"  EC2Vec found for {(np.abs(arr).sum(1) > 0).sum()}/{len(arr)} rows")
        save_pickle(arr, out("ec2vec"))
    if "structure" in groups:
        print("[structure: distances / IFP / pocket]")
        ids = (df["complex_id"].astype(str).tolist() if "complex_id" in df.columns
               else [str(i) for i in range(len(df))])
        cache, adp, ifp, pocket = {}, [], [], []
        for cid in ids:
            if cid not in cache:
                cif = (os.path.join(args.structures, cid, args.cif_name)
                       if args.structures else None)
                cache[cid] = structure.structure_features(cif)
            a, f, p = cache[cid]
            adp.append(a); ifp.append(f); pocket.append(p)
        n_ok = sum(x is not None for x in pocket)
        print(f"  structures available for {n_ok}/{len(ids)} rows")
        save_pickle(adp, out("adp")); save_pickle(ifp, out("ifp")); save_pickle(pocket, out("pocket"))
    print(f"Feature store -> {args.out}")


if __name__ == "__main__":
    main()
