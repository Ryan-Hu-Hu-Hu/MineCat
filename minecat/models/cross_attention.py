"""Bidirectional substrate <-> protein cross-attention.

Direction A: substrate atom tokens attend to protein residue tokens.
Direction B: protein residue tokens attend to substrate atom tokens.
Each direction has its own Q/K/V projections, a residual + LayerNorm attention block,
a position-wise feed-forward block, and learnable attention pooling.
"""

import torch
from torch import nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared building block
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over a padded token sequence.

    Computes per-token scalar scores via a single linear projection,
    applies softmax with padding mask, then takes a weighted sum.

    Args:
        dim : token embedding dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, N, dim)
        mask   : (B, N) float, 1=valid, 0=padded
        returns: (B, dim)
        """
        scores  = self.score(tokens).squeeze(-1)          # (B, N)
        scores  = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)               # (B, N)
        weights = torch.nan_to_num(weights, nan=0.0)      # guard all-masked rows
        return (tokens * weights.unsqueeze(-1)).sum(dim=1) # (B, dim)


def _make_ff(dim: int, dropout: float) -> nn.Sequential:
    """Position-wise feed-forward: dim → 2*dim → dim."""
    return nn.Sequential(
        nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
    )


def _safe_mha(attn_module, q, k, v, key_padding_mask, need_weights: bool = False):
    """
    MHA wrapper that prevents NaN attention weights when all keys are masked.

    Root cause: if every key in a batch sample is masked (key_padding_mask[b] = all True),
    softmax(all -inf) = NaN attention weights.  In backward, PyTorch computes
    grad_V = attn_weights^T @ grad_out.  Even if grad_out = 0 (via nan_to_num guard),
    NaN × 0 = NaN in IEEE 754, so NaN gradients propagate and corrupt all weights.

    Fix: for samples with all-masked keys, force-unmask the first key so softmax is
    always over at least one valid position.  Then zero the output for those samples
    (their attention result is meaningless but gradient-safe).

    Args:
        need_weights : if True, return (out, attn_weights) instead of just out.
                       attn_weights are averaged over heads (average_attn_weights=True).
                       Enabling this disables flash-attention; use only for diagnostics.
    """
    if key_padding_mask is None:
        out, weights = attn_module(q, k, v, need_weights=need_weights,
                                   average_attn_weights=True)
        return (out, weights) if need_weights else out

    # Identify samples with all keys masked
    all_masked = key_padding_mask.all(dim=-1)   # (B,) bool
    if not all_masked.any():
        out, weights = attn_module(q, k, v, key_padding_mask=key_padding_mask,
                                   need_weights=need_weights, average_attn_weights=True)
        return (out, weights) if need_weights else out

    # Force-unmask first key for all-masked samples → stable softmax
    safe_mask = key_padding_mask.clone()
    safe_mask[all_masked, 0] = False
    out, weights = attn_module(q, k, v, key_padding_mask=safe_mask,
                               need_weights=need_weights, average_attn_weights=True)
    # Zero the output for all-masked samples (semantically no-op, gradient-safe)
    valid = (~all_masked).float().to(q.device)   # (B,)
    out = out * valid.view(-1, 1, 1)             # broadcast (B,1,1)
    if need_weights and weights is not None:
        weights = weights * valid.view(-1, 1, 1)
    return (out, weights) if need_weights else out


# ─────────────────────────────────────────────────────────────────────────────
# Bidirectional cross-attention: Sub ↔ Protein
# ─────────────────────────────────────────────────────────────────────────────

class BidirectionalCrossAttention(nn.Module):
    """
    Two cross-attention directions between substrate atom tokens and protein
    residue tokens, with fully separate Q, K, V projections per direction.

    Direction A (Sub→Pro):
      - Sub tokens are queries; Pro tokens are keys/values.
      - Updated sub tokens are pooled with AttentionPooling.
      - Returns pooled_A : (B, dim)

    Direction B (Pro→Sub):
      - Pro tokens are queries; Sub tokens are keys/values.
      - Updated pro tokens are pooled with AttentionPooling.
      - Returns pooled_B : (B, dim)

    Args:
        dim     : common projected token dimension
        nhead   : number of attention heads (must divide dim)
        dropout : dropout probability

    Returns:
        (pooled_A, pooled_B) — each (B, dim)
    """

    def __init__(self, dim: int, nhead: int, dropout: float):
        super().__init__()

        # ── Direction A: Sub → Pro ────────────────────────────────────────────
        self.proj_q_A    = nn.Linear(dim, dim)   # sub queries
        self.proj_k_A    = nn.Linear(dim, dim)   # pro keys
        self.proj_v_A    = nn.Linear(dim, dim)   # pro values
        self.attn_A      = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True)
        self.norm_attn_A = nn.LayerNorm(dim)
        self.ff_A        = _make_ff(dim, dropout)
        self.norm_ff_A   = nn.LayerNorm(dim)
        self.pool_A      = AttentionPooling(dim)
        self.norm_out_A  = nn.LayerNorm(dim)

        # ── Direction B: Pro → Sub ────────────────────────────────────────────
        self.proj_q_B    = nn.Linear(dim, dim)   # pro queries
        self.proj_k_B    = nn.Linear(dim, dim)   # sub keys
        self.proj_v_B    = nn.Linear(dim, dim)   # sub values
        self.attn_B      = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True)
        self.norm_attn_B = nn.LayerNorm(dim)
        self.ff_B        = _make_ff(dim, dropout)
        self.norm_ff_B   = nn.LayerNorm(dim)
        self.pool_B      = AttentionPooling(dim)
        self.norm_out_B  = nn.LayerNorm(dim)

    def forward(
        self,
        sub_tokens: torch.Tensor,   # (B, N_sub, dim)
        pro_tokens: torch.Tensor,   # (B, N_pro, dim)
        sub_mask:   torch.Tensor,   # (B, N_sub) float, 1=valid
        pro_mask:   torch.Tensor,   # (B, N_pro) float, 1=valid
    ):
        """Returns (pooled_sub_to_pro, pooled_pro_to_sub), each (B, dim)."""

        # ── Direction A: sub attends to pro ───────────────────────────────────
        q_A = self.proj_q_A(sub_tokens)
        k_A = self.proj_k_A(pro_tokens)
        v_A = self.proj_v_A(pro_tokens)
        out_A     = _safe_mha(self.attn_A, q_A, k_A, v_A, pro_mask == 0)
        out_A     = torch.nan_to_num(out_A, nan=0.0, posinf=0.0, neginf=0.0)
        sub_out   = torch.nan_to_num(
            self.norm_attn_A(out_A + sub_tokens), nan=0.0, posinf=0.0, neginf=0.0)
        sub_out   = torch.nan_to_num(
            self.norm_ff_A(self.ff_A(sub_out) + sub_out), nan=0.0, posinf=0.0, neginf=0.0)
        pooled_A  = self.norm_out_A(self.pool_A(sub_out, sub_mask))

        # ── Direction B: pro attends to sub ───────────────────────────────────
        q_B = self.proj_q_B(pro_tokens)
        k_B = self.proj_k_B(sub_tokens)
        v_B = self.proj_v_B(sub_tokens)
        out_B     = _safe_mha(self.attn_B, q_B, k_B, v_B, sub_mask == 0)
        out_B     = torch.nan_to_num(out_B, nan=0.0, posinf=0.0, neginf=0.0)
        pro_out   = torch.nan_to_num(
            self.norm_attn_B(out_B + pro_tokens), nan=0.0, posinf=0.0, neginf=0.0)
        pro_out   = torch.nan_to_num(
            self.norm_ff_B(self.ff_B(pro_out) + pro_out), nan=0.0, posinf=0.0, neginf=0.0)
        pooled_B  = self.norm_out_B(self.pool_B(pro_out, pro_mask))

        return pooled_A, pooled_B
