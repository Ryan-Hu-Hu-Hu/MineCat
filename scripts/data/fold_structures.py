"""Cofold enzyme-substrate complexes with chai-1 (structure input for MineCat deep).

One complex per unique (Sequence, Smiles) pair, identified by `complex_id` (or the row
index if the CSV has no such column). Output layout, read by build_features.py:

    <out>/<complex_id>/<complex_id>.fasta
    <out>/<complex_id>/chai1_fold/pred.model_idx_0.cif   (+ the other samples / scores)

Settings match the training structures: `chai-lab fold --num-diffn-timesteps 80`
(chai-lab 0.6.1, single-sequence mode, default 5 diffusion samples; model_idx_0 is used).
Idempotent: complexes with an existing CIF are skipped.

    pip install chai_lab==0.6.1
    python scripts/data/fold_structures.py --csv data/examples/example_pairs.csv --out structures/
"""
import argparse
import os
import subprocess

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diffusion-steps", type=int, default=80)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per complex")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ids = (df["complex_id"].astype(str) if "complex_id" in df.columns
           else pd.Series([str(i) for i in range(len(df))]))
    todo = dict(zip(ids, zip(df["Sequence"].astype(str), df["Smiles"].astype(str))))
    print(f"{len(todo)} complexes")
    ok = fail = 0
    for cid, (seq, smi) in todo.items():
        d = os.path.join(args.out, cid)
        cif = os.path.join(d, "chai1_fold", "pred.model_idx_0.cif")
        if os.path.exists(cif):
            ok += 1
            continue
        os.makedirs(d, exist_ok=True)
        fasta = os.path.join(d, f"{cid}.fasta")
        with open(fasta, "w") as f:
            f.write(f">protein|{cid}-protein\n{seq}\n>ligand|{cid}-ligand\n{smi}\n")
        cmd = ["chai-lab", "fold", "--num-diffn-timesteps", str(args.diffusion_steps),
               "--device", args.device, fasta, os.path.join(d, "chai1_fold") + "/"]
        try:
            subprocess.run(cmd, check=True, timeout=args.timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            ok += os.path.exists(cif)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            fail += 1
            print(f"  FAILED {cid}: {type(e).__name__}")
    print(f"done: {ok} folded, {fail} failed")


if __name__ == "__main__":
    main()
