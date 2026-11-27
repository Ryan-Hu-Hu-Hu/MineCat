"""Protein language-model embeddings.

* ESM2-35M (facebook/esm2_t12_35M_UR50D), hidden state of transformer block 11 (of 12):
  per-residue (L, 480) float16 tokens for the deep model's cross-attention / distogram,
  and their mean (480,) as a fused modality.
* ProtT5-XL-UniRef50 encoder (Rostlab/prot_t5_xl_half_uniref50-enc): mean over residues
  (1024,) for MineCat (tree). Sequences longer than 1,000 residues keep the first and last
  500 residues (UniKP-style truncation), rare amino acids U/Z/O/B are mapped to X.
"""
import re

import numpy as np
import torch

ESM2_MODEL = "facebook/esm2_t12_35M_UR50D"
ESM2_LAYER = 11
PROT_T5_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
MAX_LEN = 1000


def _device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


@torch.no_grad()
def esm2_layer11(sequences, device=None, batch_size=16, verbose=True):
    """Return ({seq: (480,) float32 mean}, {seq: (L, 480) float16 per-residue})."""
    from transformers import AutoTokenizer, EsmModel
    dev = _device(device)
    tok = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = EsmModel.from_pretrained(ESM2_MODEL).to(dev).eval()
    uniq = list(dict.fromkeys(sequences))
    means, residues = {}, {}
    for i in range(0, len(uniq), batch_size):
        chunk = uniq[i:i + batch_size]
        enc = tok([s[:MAX_LEN] for s in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN + 2)
        hs = model(input_ids=enc["input_ids"].to(dev),
                   attention_mask=enc["attention_mask"].to(dev),
                   output_hidden_states=True).hidden_states[ESM2_LAYER]
        for b, s in enumerate(chunk):
            L = int(enc["attention_mask"][b].sum()) - 2          # strip <cls>/<eos>
            emb = hs[b, 1:L + 1].float().cpu().numpy().astype(np.float16)
            residues[s] = emb
            means[s] = emb.astype(np.float32).mean(0)
        if verbose and (i // batch_size) % 50 == 0:
            print(f"  ESM2-35M L11: {i + len(chunk)}/{len(uniq)}", flush=True)
    return means, residues


def _truncate(seq, max_len=MAX_LEN):
    if len(seq) <= max_len:
        return seq
    half = max_len // 2
    return seq[:half] + seq[-half:]


@torch.no_grad()
def prot_t5_mean(sequences, device=None, batch_size=8, verbose=True):
    """Return {seq: (1024,) float32} ProtT5-XL mean embeddings (keyed by the input sequence)."""
    from transformers import T5EncoderModel, T5Tokenizer
    dev = _device(device)
    tok = T5Tokenizer.from_pretrained(PROT_T5_MODEL, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROT_T5_MODEL).to(dev).eval()
    uniq = list(dict.fromkeys(sequences))
    out = {}
    for i in range(0, len(uniq), batch_size):
        chunk = uniq[i:i + batch_size]
        proc = [" ".join(re.sub(r"[UZOB]", "X", _truncate(s))) for s in chunk]
        enc = tok(proc, add_special_tokens=True, padding="longest", return_tensors="pt").to(dev)
        h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        for b, s in enumerate(chunk):
            L = max(int(enc["attention_mask"][b].sum()) - 1, 1)  # exclude </s>
            out[s] = h[b, :L].mean(0).float().cpu().numpy()
        if verbose and (i // batch_size) % 100 == 0:
            print(f"  ProtT5-XL: {i + len(chunk)}/{len(uniq)}", flush=True)
    return out
