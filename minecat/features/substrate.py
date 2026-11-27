"""Substrate embeddings.

* MolT5-large (laituan245/molt5-large-smiles2caption) encoder, mean over tokens
  EXCLUDING the trailing </s> token -> (1024,).
* In-house SMILES Transformer -> (1024,)  (see minecat.smiles_transformer).
* MolGNet (pretrained molecular graph transformer, as distributed with PreTKcat) ->
  pooled (768,) + per-atom tokens (n_atoms + 1 virtual node, 768). MolGNet needs
  torch_geometric and the MolGNet code/weights, so it runs as a separate script:
  scripts/data/build_molgnet_features.py.
"""
import torch

MOLT5_MODEL = "laituan245/molt5-large-smiles2caption"


@torch.no_grad()
def molt5_large(smiles_list, device=None, verbose=True):
    """Return {smiles: (1024,) float32}."""
    from transformers import T5EncoderModel, T5Tokenizer
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = T5Tokenizer.from_pretrained(MOLT5_MODEL, model_max_length=512)
    model = T5EncoderModel.from_pretrained(MOLT5_MODEL).to(dev).eval()
    uniq = list(dict.fromkeys(smiles_list))
    out = {}
    for i, smi in enumerate(uniq):
        enc = tok(smi, return_tensors="pt", truncation=True, max_length=512, padding=False).to(dev)
        h = model(**enc).last_hidden_state[0][:-1]                # drop trailing </s>
        out[smi] = h.mean(0).float().cpu().numpy()
        if verbose and i % 500 == 0:
            print(f"  MolT5-large: {i + 1}/{len(uniq)}", flush=True)
    return out


def smiles_transformer(smiles_list, ckpt=None, vocab=None, device=None):
    """Return {smiles: (1024,) float32} from the in-house SMILES Transformer."""
    from ..smiles_transformer import SmilesEncoder
    return SmilesEncoder(ckpt, vocab, device).embed(smiles_list)
