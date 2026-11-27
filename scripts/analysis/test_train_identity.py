"""Global sequence identity of every test enzyme to its fold's training set.

For test fold k (training = folds other than k and its validation fold (k + 1) % 10), each
unique test sequence is searched against the unique training sequences with MMseqs2 at
maximum sensitivity, and its GLOBAL identity is

    global_id = 100 * max_over_train(nident / qlen),   nident = fident * alnlen

i.e. the fraction of the whole test protein that is identical to its best training hit.
(Local alignment identity would over-count enzymes that share only a short catalytic
domain.) Rows with global_id < c are "novel at c%"; at 40% every test row qualifies by
construction of the cluster split. Used for Figure 4 and Figure S9.

    python scripts/analysis/test_train_identity.py --mmseqs /path/to/mmseqs \
        --out results/cluster_cv/test_train_identity.csv.gz
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from minecat import config  # noqa: E402
from minecat.data import load_dataset  # noqa: E402


def max_global_identity(test_seqs, train_seqs, mmseqs, threads):
    with tempfile.TemporaryDirectory() as tmp:
        te, tr, m8 = (os.path.join(tmp, n) for n in ("test.fa", "train.fa", "hits.m8"))
        with open(te, "w") as f:
            f.writelines(f">q{i}\n{s}\n" for i, s in enumerate(test_seqs))
        with open(tr, "w") as f:
            f.writelines(f">t{i}\n{s}\n" for i, s in enumerate(train_seqs))
        subprocess.run([mmseqs, "easy-search", te, tr, m8, os.path.join(tmp, "mm"),
                        "--min-seq-id", "0.0", "-c", "0.0", "--alignment-mode", "3",
                        "-s", "7.5", "-e", "1000", "--max-seqs", "20000",
                        "--format-output", "query,fident,alnlen,qlen",
                        "--threads", str(threads), "-v", "1"], check=True)
        best = np.zeros(len(test_seqs))
        if os.path.exists(m8):
            for line in open(m8):
                q, fident, alnlen, qlen = line.split("\t")
                gid = 100.0 * float(fident) * float(alnlen) / float(qlen) if float(qlen) > 0 else 0.0
                k = int(q[1:])
                best[k] = max(best[k], min(gid, 100.0))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DATASET_CSV)
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "cluster_cv",
                                                  "test_train_identity.csv.gz"))
    args = ap.parse_args()
    df = load_dataset(args.dataset)
    seqs, folds = df["Sequence"].astype(str).to_numpy(), df["fold"].to_numpy()
    gid = np.full(len(df), np.nan)
    for k in range(config.N_FOLDS):
        val = (k + 1) % config.N_FOLDS
        te = np.where(folds == k)[0]
        train_uniq = sorted(set(seqs[(folds != k) & (folds != val)]))
        test_uniq = sorted(set(seqs[te]))
        best = dict(zip(test_uniq, max_global_identity(test_uniq, train_uniq, args.mmseqs, args.threads)))
        gid[te] = [best[s] for s in seqs[te]]
        print(f"fold {k}: median global identity {np.median(gid[te]):.1f}%")
    pd.DataFrame({"pair_id": df["pair_id"], "max_global_identity_to_train": gid}).to_csv(
        args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
