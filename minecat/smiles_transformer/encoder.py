"""Embed SMILES with the pretrained in-house SMILES Transformer (1,024-d per molecule).

Tokenization mirrors the reference SMILES-Transformer recipe: atom-aware split, long
SMILES truncated to the first 109 + last 109 tokens, <sos>/<eos>, padded to 220.
"""
import numpy as np
import torch

from .. import config
from .model import TrfmSeq2seq
from .tokenizer import EOS, PAD, SOS, UNK, WordVocab, split

SEQ_LEN = 220


def smiles_to_ids(smiles, vocab, seq_len=SEQ_LEN):
    toks = split(smiles).split()
    if len(toks) > seq_len - 2:
        toks = toks[:109] + toks[-109:]
    ids = [SOS] + [vocab.stoi.get(t, UNK) for t in toks] + [EOS]
    return ids + [PAD] * (seq_len - len(ids))


class SmilesEncoder:
    def __init__(self, ckpt=None, vocab=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vocab = WordVocab.load(vocab or config.SMILES_TRANSFORMER_VOCAB)
        ck = torch.load(ckpt or config.SMILES_TRANSFORMER_CKPT, map_location=self.device)
        hp = ck["args"]
        self.model = TrfmSeq2seq(ck["vocab_size"], hp["hidden"], ck["vocab_size"],
                                 hp["n_layer"]).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    @torch.no_grad()
    def embed(self, smiles_list, batch_size=100):
        """Return {smiles: (1024,) float32} for the unique SMILES in `smiles_list`."""
        uniq = list(dict.fromkeys(smiles_list))
        out = {}
        for i in range(0, len(uniq), batch_size):
            chunk = uniq[i:i + batch_size]
            ids = torch.tensor([smiles_to_ids(s, self.vocab) for s in chunk], device=self.device)
            feats = self.model.encode(ids.t())          # (B, 1024) numpy
            for s, v in zip(chunk, feats):
                out[s] = v.astype(np.float32)
        return out
