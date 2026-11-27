"""
dataset.py — denoising-translation dataset for pretraining the SMILES Transformer so the
ENCODER actually learns a molecular representation.

Each item returns (src_ids, tgt_ids):
  tgt = the CANONICAL SMILES (as stored in the corpus)          -> decoder target
  src = a RANDOMIZED SMILES of the SAME molecule (RDKit doRandom),
        with a fraction of tokens replaced by <mask>            -> encoder input

Because src (randomized + masked) is a DIFFERENT string from tgt (canonical), the
encoder-decoder cannot trivially copy source->output through the attention memory
(the failure mode of both the reference copy objective and a naive src==tgt AR setup,
which collapse to ppl~1 and leave the encoder untrained). To translate atom-reordered
/ corrupted SMILES into the canonical form the model must encode the actual molecule.

If RDKit randomization fails, src falls back to the canonical form (still masked), so
masking alone prevents a pure copy in that rare case.
"""
import random
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit import RDLogger

from .tokenizer import split, SOS, EOS, PAD, MASK

RDLogger.DisableLog('rdApp.*')


class Seq2seqDataset(Dataset):
    def __init__(self, smiles, vocab, seq_len=220, p_mask=0.15):
        self.smiles = smiles          # canonical SMILES (corpus)
        self.vocab = vocab
        self.seq_len = seq_len
        self.p_mask = p_mask

    def __len__(self):
        return len(self.smiles)

    def _randomized_tokens(self, canonical):
        try:
            mol = Chem.MolFromSmiles(canonical)
            if mol is not None:
                rsm = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                toks = split(rsm).split()
                if 0 < len(toks) <= self.seq_len - 2:
                    return toks
        except Exception:
            pass
        return split(canonical).split()

    def _pad_ids(self, ids):
        ids = ids[:self.seq_len]
        ids += [PAD] * (self.seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, item):
        canonical = self.smiles[item]
        tgt_toks = split(canonical).split()
        src_toks = self._randomized_tokens(canonical)

        src_ids = [self.vocab.stoi.get(t, self.vocab.unk_index) for t in src_toks]
        # BART-style token masking on the ENCODER input only
        if self.p_mask > 0:
            src_ids = [MASK if random.random() < self.p_mask else t for t in src_ids]
        src_ids = [SOS] + src_ids + [EOS]

        tgt_ids = [SOS] + [self.vocab.stoi.get(t, self.vocab.unk_index) for t in tgt_toks] + [EOS]

        return self._pad_ids(src_ids), self._pad_ids(tgt_ids)
