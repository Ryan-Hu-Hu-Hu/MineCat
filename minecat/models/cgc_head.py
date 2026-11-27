"""Customized Gate Control (CGC) multi-task regression head for K_M and k_cat.

One CGC layer with three experts (K_M-specific, k_cat-specific, shared), each a stack of
3 x [Linear -> LayerNorm -> GELU -> Dropout] operating on the fused representation.
Each task routes between its own expert and the shared expert with a 2-way softmax gate
(PLE/CGC, Tang et al., RecSys 2020), followed by a linear task head.

    head = CGCHead(total_dim=4512, dropout=0.4)
    km_pred, kcat_pred = head(fused)          # fused: (B, total_dim)
"""

import torch
from torch import nn
import torch.nn.functional as F


class CGCHead(nn.Module):
    def __init__(self, total_dim: int, dropout: float = 0.0):
        super().__init__()

        def _expert(d: int):
            return nn.Sequential(
                nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
            )

        self.expert_km     = _expert(total_dim)
        self.expert_kcat   = _expert(total_dim)
        self.expert_shared = _expert(total_dim)

        # Per-task gating: softmax over [task-specific expert, shared expert]
        self.gate_km   = nn.Linear(total_dim, 2)
        self.gate_kcat = nn.Linear(total_dim, 2)

        self.head_km   = nn.Linear(total_dim, 1)
        self.head_kcat = nn.Linear(total_dim, 1)

    def forward(self, cat_vector: torch.Tensor):
        """
        Args:
            cat_vector : (B, total_dim)

        Returns:
            km_pred   : (B,)
            kcat_pred : (B,)
        """
        e_km     = self.expert_km(cat_vector)
        e_kcat   = self.expert_kcat(cat_vector)
        e_shared = self.expert_shared(cat_vector)

        w_km   = F.softmax(self.gate_km(cat_vector),   dim=-1)   # (B, 2)
        out_km = w_km[:, 0:1] * e_km + w_km[:, 1:2] * e_shared  # (B, total_dim)

        w_kcat   = F.softmax(self.gate_kcat(cat_vector), dim=-1)  # (B, 2)
        out_kcat = w_kcat[:, 0:1] * e_kcat + w_kcat[:, 1:2] * e_shared

        km_pred   = self.head_km(out_km).squeeze(-1)
        kcat_pred = self.head_kcat(out_kcat).squeeze(-1)

        return km_pred, kcat_pred
