"""
prepare_corpus.py — turn the raw ChEMBL chemreps dump into a clean SMILES pretraining
corpus for our own SMILES Transformer.

Input : chembl_37_chemreps.txt.gz  (TSV: chembl_id, canonical_smiles, standard_inchi,
        inchi_key), from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/
Steps : keep the canonical_smiles column, RDKit-canonicalize, drop invalid / too-long
        (> max_atoms_tokens), keep the largest fragment (strip salts/counterions),
        de-duplicate. Writes one canonical SMILES per line.

Output: one canonical SMILES per line (2,757,061 SMILES for ChEMBL 37)

Run: python scripts/smiles_transformer/prepare_corpus.py --raw chembl_37_chemreps.txt.gz \
         --out smiles_corpus/chembl37_smiles.txt
"""
import gzip
import os
import sys

from rdkit import Chem
from rdkit import RDLogger

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from minecat.smiles_transformer.tokenizer import split  # noqa: E402

RDLogger.DisableLog('rdApp.*')

MAX_TOKENS = 218      # leave room for SOS/EOS within seq_len=220


def largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) <= 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='chembl_XX_chemreps.txt.gz')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    RAW, OUT = args.raw, args.out
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    seen = set()
    kept = skipped = 0
    with gzip.open(RAW, 'rt') as f, open(OUT, 'w') as out:
        f.readline()  # header: chembl_id\tcanonical_smiles\t...
        for ln, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                skipped += 1; continue
            smi = parts[1]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                skipped += 1; continue
            mol = largest_fragment(mol)
            try:
                can = Chem.MolToSmiles(mol)
            except Exception:
                skipped += 1; continue
            if not can or can in seen:
                skipped += 1; continue
            if len(split(can).split()) > MAX_TOKENS:
                skipped += 1; continue
            seen.add(can)
            out.write(can + '\n')
            kept += 1
            if (kept % 200000) == 0:
                print(f"  kept {kept:,} (scanned {ln + 1:,})", flush=True)
    print(f"DONE  kept {kept:,}  skipped {skipped:,}  -> {OUT}")


if __name__ == '__main__':
    main()
