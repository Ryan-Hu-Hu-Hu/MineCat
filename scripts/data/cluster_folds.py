"""Leakage-free 10-fold split: MMseqs2 clustering at 40% sequence identity.

Unique sequences are clustered with `mmseqs easy-cluster --min-seq-id 0.4`; whole clusters
are then assigned to 10 folds by greedy bin packing (largest cluster first, into the
currently lightest fold), so every enzyme family stays inside one fold and the folds hold
a similar number of rows. The shipped dataset already contains this `fold` column; the
script is provided to reproduce it or to split a new dataset.

    python scripts/data/cluster_folds.py --csv my_dataset.csv --out my_dataset_folds.csv \
        --mmseqs /path/to/mmseqs
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd


def mmseqs_clusters(seqs, mmseqs, identity, threads):
    with tempfile.TemporaryDirectory() as work:
        fasta = os.path.join(work, "seqs.fasta")
        with open(fasta, "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">seq_{i}\n{s}\n")
        prefix, tmp = os.path.join(work, "clu"), os.path.join(work, "tmp")
        os.makedirs(tmp)
        cmd = [mmseqs, "easy-cluster", fasta, prefix, tmp, "--min-seq-id", str(identity),
               "--cov-mode", "0", "-c", "0.0", "--threads", str(threads)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        rep = {}
        with open(prefix + "_cluster.tsv") as f:
            for line in f:
                r, m = line.rstrip("\n").split("\t")
                rep[m] = r
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--identity", type=float, default=0.4)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    uniq = sorted(set(df["Sequence"].astype(str)))
    rep = mmseqs_clusters(uniq, args.mmseqs, args.identity, args.threads)
    seq_to_cluster = {uniq[int(m.split("_")[1])]: r for m, r in rep.items()}
    for i, s in enumerate(uniq):
        seq_to_cluster.setdefault(s, f"seq_{i}")
    clusters = df["Sequence"].astype(str).map(seq_to_cluster)
    sizes = clusters.groupby(clusters).size().sort_values(ascending=False)  # rows per cluster
    print(f"{len(uniq)} sequences -> {len(sizes)} clusters (largest: {sizes.iloc[0]} rows)")

    load = np.zeros(args.n_folds, dtype=np.int64)
    assign = {}
    for clu, n in sizes.items():
        f = int(np.argmin(load))
        assign[clu] = f
        load[f] += n
    df["fold"] = clusters.map(assign).astype(int)
    print(df.groupby("fold").size().to_string())
    df.to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
