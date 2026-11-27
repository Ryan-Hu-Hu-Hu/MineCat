"""
build_vocab.py — build our SMILES vocabulary from the prepared corpus.

Tokenizes each SMILES with the atom-aware `split`, then keeps tokens with frequency
>= min_freq (default 500, matching the reference recipe). Writes vocab.json
(47 tokens for ChEMBL 37).

Run: python scripts/smiles_transformer/build_vocab.py --corpus smiles_corpus/chembl37_smiles.txt \
         --out smiles_corpus/vocab.json
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from minecat.smiles_transformer.tokenizer import split, WordVocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', required=True, help='vocab.json')
    ap.add_argument('--min_freq', type=int, default=500)
    args = ap.parse_args()

    def token_lines():
        with open(args.corpus) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield split(line).split()

    vocab = WordVocab(token_lines(), min_freq=args.min_freq)
    print(f"VOCAB SIZE: {len(vocab)}  (min_freq={args.min_freq})")
    print("tokens:", vocab.itos)
    vocab.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
