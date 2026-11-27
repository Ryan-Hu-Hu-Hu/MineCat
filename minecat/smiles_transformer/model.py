"""
model.py — our own SMILES Transformer (TrfmSeq2seq), an in-house reimplementation of
the published SMILES-Transformer autoencoder architecture (Honda et al. 2019), the
same architecture used by UniKP. Trained from scratch on ChEMBL 37
(scripts/smiles_transformer/pretrain.py), so code, vocabulary and weights are all in-house.

Architecture (identical to the reference so the 1024-d embedding convention matches):
  - token embedding (vocab -> hidden=256)
  - sinusoidal positional encoding
  - torch.nn.Transformer(d_model=256, nhead=4, num_encoder_layers=4,
                         num_decoder_layers=4, dim_feedforward=256)  [seq-first (T,B,H)]
  - linear head -> vocab, log-softmax

Training objective: teacher-forced AUTOREGRESSIVE reconstruction (shifted decoder input
+ causal mask), NOT the reference's degenerate `trfm(embedded, embedded)` copy objective
(which collapses to ppl~1 and leaves the encoder untrained). The architecture/params are
identical to the reference; only the decoding is corrected so the encoder actually learns.

Embedding extraction `encode()` returns concat[mean, max, first-token(last enc layer),
first-token(penultimate enc layer)] over the encoder stack -> 4*256 = 1024-d, exactly
matching the reference feature layout.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, H)

    def forward(self, x):
        # x: (T, B, H) -> add PE broadcast over batch
        x = x + self.pe[:, :x.size(0)].transpose(0, 1)
        return self.dropout(x)


class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(
            d_model=hidden_size, nhead=4,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src, tgt, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        """Proper (teacher-forced) autoregressive decoding.
        src: (S, B) full source sequence -> encoder memory.
        tgt: (T, B) shifted decoder input (starts with SOS), causal-masked so the
             decoder cannot see the token it must predict and is forced to rely on
             the encoder memory (this is what makes the ENCODER actually learn).
        Returns log-probs (T, B, V)."""
        src_emb = self.pe(self.embed(src))          # (S, B, H)
        tgt_emb = self.pe(self.embed(tgt))          # (T, B, H)
        hidden = self.trfm(
            src_emb, tgt_emb, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask)  # (T, B, H)
        return F.log_softmax(self.out(hidden), dim=2)      # (T, B, V)

    @torch.no_grad()
    def _encode(self, src):
        # src: (T, B) -> (B, 4H) numpy
        output = self.pe(self.embed(src))           # (T, B, H)
        enc = self.trfm.encoder
        for i in range(enc.num_layers - 1):
            output = enc.layers[i](output)
        penult = output.detach().cpu().numpy()
        output = enc.layers[-1](output)
        if enc.norm is not None:
            output = enc.norm(output)
        output = output.detach().cpu().numpy()
        return np.hstack([output.mean(axis=0), output.max(axis=0),
                          output[0, :, :], penult[0, :, :]])  # (B, 4H)

    @torch.no_grad()
    def encode(self, src, chunk=100):
        """src: (T, B). Returns (B, 4H) numpy, chunked to bound memory."""
        B = src.shape[1]
        if B <= chunk:
            return self._encode(src)
        outs = [self._encode(src[:, s:s + chunk]) for s in range(0, B, chunk)]
        return np.concatenate(outs, axis=0)
