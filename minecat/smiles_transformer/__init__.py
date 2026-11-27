"""In-house SMILES Transformer (denoising seq2seq autoencoder pretrained on ChEMBL 37).

`SmilesEncoder` turns SMILES strings into the 1,024-d embedding used by MineCat."""
from .encoder import SmilesEncoder

__all__ = ["SmilesEncoder"]
