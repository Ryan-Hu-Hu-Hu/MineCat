"""E(n)-equivariant graph neural network for the chai-1 binding pocket.

Based on Satorras et al., "E(n) Equivariant Graph Neural Networks" (ICML 2021); edges are
built from a distance cutoff with a k-nearest-neighbour cap, and messages are aggregated
with torch.index_add_ (no torch_scatter dependency).

    egnn = EGNN(node_feat_dim=14, hidden_dim=128, num_layers=3, cutoff=8.0)
    pooled, per_atom = egnn(node_feats, positions, node_mask)
"""

import torch
from torch import nn


class EGNNLayer(nn.Module):
    """
    Single E(n)-equivariant GNN layer.
    Updates node features h and positions x equivariantly.

    Args:
        node_dim  : input node feature dimension
        hidden_dim: output and message hidden dimension
        edge_dim  : edge attribute dimension (default 1 = squared dist only)
        dropout   : dropout on message and update networks
    """

    def __init__(self, node_dim: int, hidden_dim: int,
                 edge_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        # Edge message MLP: (h_i, h_j, sq_dist, edge_attr) → message
        self.phi_e = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Coordinate update weight: message → scalar weight for position update
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),   # bounded weight prevents coordinate explosion
        )
        # Node update MLP: (h_i, aggregated messages) → h_i'
        self.phi_h = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Gate on node update (stabilises training)
        self.gate  = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor,
                node_mask: torch.Tensor):
        """
        Args:
            h          : (N, node_dim)   — node features
            x          : (N, 3)          — atom positions
            edge_index : (2, E)          — directed edges (src, dst)
            edge_attr  : (E, edge_dim)   — edge attributes
            node_mask  : (N,)            — 1 = valid atom, 0 = padding

        Returns:
            h_new : (N, hidden_dim)
            x_new : (N, 3)  — updated positions (equivariant)
        """
        src, dst = edge_index   # each shape (E,)
        N        = h.size(0)

        h_src = h[src]   # (E, node_dim)
        h_dst = h[dst]   # (E, node_dim)
        x_src = x[src]   # (E, 3)
        x_dst = x[dst]   # (E, 3)

        # Relative displacement and squared distance
        diffs   = x_dst - x_src               # (E, 3) — directed
        sq_dist = (diffs ** 2).sum(-1, keepdim=True)   # (E, 1)

        # Edge messages
        edge_in  = torch.cat([h_src, h_dst, sq_dist, edge_attr], dim=-1)
        messages = self.phi_e(edge_in)          # (E, hidden_dim)
        messages = self.dropout(messages)

        # Equivariant position update: aggregate weighted displacement
        coord_weight = self.phi_x(messages)     # (E, 1)
        weighted_diff = diffs * coord_weight    # (E, 3)

        x_agg = torch.zeros(N, 3, device=h.device, dtype=h.dtype)
        x_agg.index_add_(0, dst, weighted_diff)
        x_new = x + x_agg * node_mask.unsqueeze(-1)  # masked: don't move padding

        # Aggregated messages per destination node
        m_agg = torch.zeros(N, messages.size(1), device=h.device, dtype=h.dtype)
        m_agg.index_add_(0, dst, messages)

        # Node update with gating
        h_cat  = torch.cat([h, m_agg], dim=-1)
        h_upd  = self.phi_h(h_cat)
        g      = self.gate(h_cat)
        h_new  = h_upd * g                      # (N, hidden_dim)

        # Zero out padding node features
        h_new  = h_new * node_mask.unsqueeze(-1)

        return h_new, x_new


class EGNN(nn.Module):
    """
    E(n)-equivariant GNN for 3D pocket encoding.

    Constructs kNN edges from atom positions, applies EGNNLayers,
    then pools node features to a single pocket embedding.

    Args:
        node_feat_dim : raw input node feature dim (e.g. 14)
        hidden_dim    : internal hidden dim (per EGNN layer)
        num_layers    : number of EGNNLayer message-passing steps
        dropout       : dropout rate
        cutoff        : Å cutoff for edge construction (atoms farther
                        than this do not have a direct edge)
        max_neighbors : max edges per node (kNN cap; -1 = no cap)
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.0,
                 cutoff: float = 8.0, max_neighbors: int = 16):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.cutoff        = cutoff
        self.max_neighbors = max_neighbors

        # Initial node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
        )

        # EGNN layers: first layer takes hidden_dim input; subsequent reuse hidden_dim
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, edge_dim=1, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Pooling projection (after aggregation)
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def build_edges(self, positions: torch.Tensor, mask: torch.Tensor):
        """
        Build directed edge index from pairwise distances.
        Edges connect atoms i→j where dist(i,j) <= cutoff AND mask[i] AND mask[j].

        Args:
            positions : (N, 3)
            mask      : (N,) bool/float — 1 = valid atom

        Returns:
            edge_index : (2, E) long
            edge_attr  : (E, 1) float — squared distances
        """
        N   = positions.size(0)
        dev = positions.device

        # Pairwise squared distances
        diff    = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
        sq_dist = (diff ** 2).sum(-1)                              # (N, N)

        # Valid pairs: within cutoff AND both valid atoms AND not self-loop
        valid_pair = mask.unsqueeze(0) * mask.unsqueeze(1)         # (N, N)
        within_cut = (sq_dist <= self.cutoff ** 2)
        no_self    = ~torch.eye(N, dtype=torch.bool, device=dev)
        edge_mask  = within_cut & no_self & (valid_pair > 0)

        # kNN cap
        if self.max_neighbors > 0:
            # For each dst node, keep only max_neighbors nearest src nodes
            sq_with_inf = sq_dist.clone()
            sq_with_inf[~edge_mask] = float('inf')
            _, top_k_idx = sq_with_inf.topk(
                min(self.max_neighbors, N - 1), dim=1, largest=False)
            knn_mask = torch.zeros(N, N, dtype=torch.bool, device=dev)
            row_idx  = torch.arange(N, device=dev).unsqueeze(1).expand_as(top_k_idx)
            knn_mask[row_idx, top_k_idx] = True
            edge_mask = edge_mask & knn_mask

        dst_idx, src_idx = edge_mask.nonzero(as_tuple=True)
        edge_index = torch.stack([src_idx, dst_idx], dim=0)   # (2, E)
        edge_attr  = sq_dist[src_idx, dst_idx].unsqueeze(-1)  # (E, 1)

        return edge_index, edge_attr

    def forward(self, node_feats: torch.Tensor, positions: torch.Tensor,
                node_mask: torch.Tensor):
        """
        Args:
            node_feats : (N, node_feat_dim) — raw node features
            positions  : (N, 3)             — 3D coordinates
            node_mask  : (N,)               — 1=valid, 0=padding

        Returns:
            pooled   : (hidden_dim,)         — masked mean pooled embedding
            per_atom : (N, hidden_dim)       — per-node embeddings after all layers
        """
        h = self.node_embedding(node_feats)    # (N, hidden_dim)
        h = h * node_mask.unsqueeze(-1)        # zero padding

        x = positions.clone()                  # (N, 3)

        # Build edges once from initial positions (positions shift is small, rebuild
        # each layer would be more accurate but expensive)
        edge_index, edge_attr = self.build_edges(positions, node_mask)

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr, node_mask)

        per_atom = h   # (N, hidden_dim)

        # Masked mean pooling
        valid_count = node_mask.sum().clamp(min=1.0)
        pooled      = (h * node_mask.unsqueeze(-1)).sum(0) / valid_count  # (hidden_dim,)
        pooled      = self.pool_proj(pooled)

        return pooled, per_atom
