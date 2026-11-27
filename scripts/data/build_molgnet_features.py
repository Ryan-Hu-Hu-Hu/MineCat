"""MolGNet substrate-graph embeddings (pooled 768-d + per-atom tokens).

MolGNet is the pretrained molecular graph transformer distributed with PreTKcat
(https://github.com/MrVincentCai/PreTKcat). It is not redistributed here. Set it up once:

    git clone https://github.com/MrVincentCai/PreTKcat external/PreTKcat
    # needs: torch, torch_geometric, rdkit (a separate conda env is easiest)

Then:
    python scripts/data/build_molgnet_features.py --csv data/minecat_dataset.csv.gz \
        --out features/ --pretkcat external/PreTKcat

Writes (into --out):
    molgnet_pooled.pkl  (N, 768) float32   global-add-pool of the atom embeddings
    molgnet_atoms.pkl   {smiles: (n_atoms + 1, 768) float16}   per-atom tokens; the last row
                        is MolGNet's virtual node (dropped by the deep model at load time)
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

FEATURE_NAMES = {"pooled": "molgnet_pooled.pkl", "atoms": "molgnet_atoms.pkl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretkcat", default=os.path.join("external", "PreTKcat"),
                    help="path to a PreTKcat checkout (provides MPG_util + MolGNet.pt)")
    ap.add_argument("--ckpt", default=None, help="MolGNet.pt (default: <pretkcat>/pretrained_model/MolGNet.pt)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.pretkcat, "Pretrained_Model"))
    from MPG_util.graph_bert import MolGT                     # noqa: E402
    from MPG_util.mol2graph import mol_to_graph_data_dic      # noqa: E402
    from rdkit import Chem                                    # noqa: E402
    from torch_geometric.nn import global_add_pool            # noqa: E402

    ckpt = args.ckpt or os.path.join(args.pretkcat, "pretrained_model", "MolGNet.pt")
    enc = MolGT(5, 768, 12, 3, 0.5)
    enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
    enc.eval()

    smiles = pd.read_csv(args.csv)["Smiles"].astype(str).tolist()
    uniq = list(dict.fromkeys(smiles))
    print(f"{len(smiles)} rows, {len(uniq)} unique SMILES")
    pooled, atoms, failed = {}, {}, []
    with torch.no_grad():
        for i, smi in enumerate(uniq):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                failed.append(smi)
                continue
            data = mol_to_graph_data_dic(mol)
            h = enc(data)
            pooled[smi] = global_add_pool(h, torch.zeros(data.x.size(0), dtype=torch.long))[0].numpy().astype(np.float32)
            atoms[smi] = h.numpy().astype(np.float16)
            if i % 500 == 0:
                print(f"  {i + 1}/{len(uniq)}", flush=True)
    if failed:
        print(f"WARNING: {len(failed)} SMILES could not be parsed -> zero vectors / single zero token")
    zero_tok = np.zeros((1, 768), np.float16)
    arr = np.stack([pooled.get(s, np.zeros(768, np.float32)) for s in smiles]).astype(np.float32)
    atoms_all = {s: atoms.get(s, zero_tok) for s in uniq}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, FEATURE_NAMES["pooled"]), "wb") as f:
        pickle.dump(arr, f, protocol=4)
    with open(os.path.join(args.out, FEATURE_NAMES["atoms"]), "wb") as f:
        pickle.dump(atoms_all, f, protocol=4)
    print(f"saved MolGNet features -> {args.out}")


if __name__ == "__main__":
    main()
