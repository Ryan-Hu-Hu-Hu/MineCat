"""MineCat (deep): multi-modal K_M / k_cat regressor.

Eight modalities are concatenated into one fused vector (total_dim = 4,512) and fed to a
CGC multi-task head (K_M and k_cat):

    idx  modality                                   width   how it enters the head
    [0]  MolGNet pooled substrate graph              768    raw (frozen embedding)
    [1]  MolT5-large SMILES embedding               1024    raw
    [2]  ESM2-35M (layer 11) mean-pooled sequence    480    raw
    [3]  substrate <-> protein cross-attention        64    computed (see below)
    [4]  EC2Vec enzyme-function embedding           1024    raw
    [5]  chai-1 binding-pocket EGNN                   64    computed
    [6]  pH / temperature RBF encoding                64    projected
    [7]  in-house SMILES-Transformer embedding      1024    raw

Frozen pretrained embeddings are fed RAW (no learned projection bottleneck); only the token
streams used by cross-attention and the distogram decoder are projected to `dim` = 64.

Structure-aware auxiliary decoders (training-time losses, annealed to zero over 200 epochs):
  * ADP distogram:  per (residue, ligand-atom) pair, 17 distance bins (2-22 A + far),
                    bilinear in the projected protein / substrate tokens (cross-entropy);
  * IFP:            64 pocket residues x 6 interaction types from the fused vector (BCE);
  * pocket denoise: EGNN predicts the Gaussian noise added to pocket coordinates (MSE).

forward() returns (km_pred, kcat_pred, adp_logits, ifp_logits, denoise_pred, noise_added);
predictions are in the [-1, 1] normalized label space.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .cgc_head import CGCHead
from .cross_attention import BidirectionalCrossAttention
from .egnn import EGNN

MOLGNET_DIM = 768
MOLT5_DIM = 1024
EC2VEC_DIM = 1024
SMILES_TRFM_DIM = 1024
ESM2_DIM = 480
PH_TEMP_DIM = 64            # 32 pH + 32 temperature Gaussian RBFs

ADP_N_BINS = 17
IFP_MAX_RES = 64
N_INT_TYPES = 6
IFP_DIM = IFP_MAX_RES * N_INT_TYPES
POCKET_NODE_DIM = 14


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class MineCatDeep(nn.Module):
    def __init__(self, dim: int = 64, nhead: int = 4, dropout: float = 0.4,
                 esm2_dim: int = ESM2_DIM, adp_n_bins: int = ADP_N_BINS,
                 ifp_dim: int = IFP_DIM, egnn_hidden: int = 128, egnn_layers: int = 3,
                 denoise_sigma: float = 0.5):
        super().__init__()
        self.projection_dim = dim
        self.dropout_p = dropout
        self.adp_n_bins = adp_n_bins
        self.denoise_sigma = denoise_sigma

        total_dim = (MOLGNET_DIM + MOLT5_DIM + esm2_dim + EC2VEC_DIM + SMILES_TRFM_DIM
                     + 3 * dim)
        self.total_dim = total_dim

        # token projections for cross-attention + distogram decoder
        self.sub_token_proj = nn.Sequential(nn.Linear(MOLGNET_DIM, dim), nn.LayerNorm(dim))
        # MolT5 projection: used ONLY as a broadcast added to substrate atom tokens
        self.smiles_proj = nn.Sequential(
            nn.Linear(MOLT5_DIM, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, dim), nn.LayerNorm(dim))
        self.pro_token_proj = nn.Sequential(
            nn.Linear(esm2_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, dim), nn.LayerNorm(dim))
        # learnable per-residue relevance gate
        self.relevance_gate = nn.Linear(dim, 1)

        # [3] bidirectional cross-attention
        self.cross_attn = BidirectionalCrossAttention(dim, nhead, dropout)
        self.cross_attn_proj = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))

        # [5] binding-pocket EGNN (+ coordinate-denoising head)
        self.pocket_egnn = EGNN(node_feat_dim=POCKET_NODE_DIM, hidden_dim=egnn_hidden,
                                num_layers=egnn_layers, dropout=dropout,
                                cutoff=8.0, max_neighbors=16)
        self.pocket_proj = nn.Sequential(
            nn.Linear(egnn_hidden, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.denoise_head = nn.Sequential(
            nn.Linear(egnn_hidden, egnn_hidden // 2), nn.SiLU(),
            nn.Linear(egnn_hidden // 2, 3))

        # [6] pH / temperature
        self.ph_temp_proj = nn.Sequential(nn.Linear(PH_TEMP_DIM, dim), nn.LayerNorm(dim))

        # ADP distogram decoder (bilinear per distance bin)
        adp_hidden = max(16, dim // 4)
        self.adp_hidden = adp_hidden
        self.adp_pro_proj = nn.Linear(dim, adp_n_bins * adp_hidden)
        self.adp_sub_proj = nn.Linear(dim, adp_n_bins * adp_hidden)
        self._adp_scale = adp_hidden ** -0.5

        # IFP decoder
        ifp_hidden = max(256, total_dim // 2)
        self.ifp_decoder = nn.Sequential(
            nn.Linear(total_dim, ifp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ifp_hidden, ifp_dim))

        # multi-task regression head
        self.cgc_head = CGCHead(total_dim=total_dim, dropout=dropout)

    # ------------------------------------------------------------------
    def _drop(self, x):
        return F.dropout(x, self.dropout_p, training=self.training)

    def _run_egnn(self, feats_list, pos_list, mask_list):
        pooled, denoise, noises = [], [], []
        for feats, pos, mask in zip(feats_list, pos_list, mask_list):
            if self.training and self.denoise_sigma > 0:
                noise = torch.randn_like(pos) * self.denoise_sigma * mask.unsqueeze(-1)
                pos_in = pos + noise
                noises.append(noise)
            else:
                pos_in = pos
                noises.append(None)
            centroid = (pos_in * mask.unsqueeze(-1)).sum(0) / mask.sum().clamp(1)
            p, per_atom = self.pocket_egnn(feats, pos_in - centroid, mask)
            pooled.append(p)
            denoise.append(self.denoise_head(per_atom) if self.training else None)
        return torch.stack(pooled), denoise, noises

    @staticmethod
    def _modality_dropout(vecs, p, training):
        """Co-modal dropout: each modality is zeroed independently per sample."""
        if not training or p <= 0.0:
            return vecs
        B = vecs[0].size(0)
        return [v * torch.bernoulli(torch.full((B, 1), 1 - p, device=v.device)) for v in vecs]

    # ------------------------------------------------------------------
    def fuse(self, molgnet_pooled, molgnet_atoms, atom_mask, molt5, esm2_mean,
             esm2_residues, residue_mask, ec2vec, ph_temp, smiles_trfm,
             pocket_feats, pocket_pos, pocket_mask, modality_dropout=0.0):
        """Build the fused representation. Returns (fused, adp_logits, denoise, noises)."""
        ph_temp_vec = self._drop(self.ph_temp_proj(ph_temp))

        sub_tokens = self._drop(self.sub_token_proj(_clean(molgnet_atoms)))
        smiles_vec = self._drop(self.smiles_proj(_clean(molt5)))
        pro_tokens = self._drop(self.pro_token_proj(_clean(esm2_residues)))
        pro_tokens = pro_tokens * torch.sigmoid(self.relevance_gate(pro_tokens))

        a2p, p2a = self.cross_attn(sub_tokens + smiles_vec.unsqueeze(1), pro_tokens,
                                   atom_mask, residue_mask)
        cross_vec = self._drop(self.cross_attn_proj(torch.cat([a2p, p2a], dim=1)))

        # full-length distogram: (B, L_res, N_atoms, n_bins)
        B, Lp, Ln = pro_tokens.size(0), pro_tokens.size(1), sub_tokens.size(1)
        pro_h = self.adp_pro_proj(pro_tokens).view(B, Lp, self.adp_n_bins, self.adp_hidden)
        sub_h = self.adp_sub_proj(sub_tokens).view(B, Ln, self.adp_n_bins, self.adp_hidden)
        adp_logits = torch.einsum("bpkh,blkh->bplk", pro_h, sub_h) * self._adp_scale

        pocket_raw, denoise, noises = self._run_egnn(pocket_feats, pocket_pos, pocket_mask)
        pocket_vec = self._drop(self.pocket_proj(pocket_raw))

        vecs = [_clean(molgnet_pooled), _clean(molt5), _clean(esm2_mean), cross_vec,
                _clean(ec2vec), pocket_vec, ph_temp_vec, _clean(smiles_trfm)]
        vecs = self._modality_dropout(vecs, modality_dropout, self.training)
        fused = _clean(torch.cat(vecs, dim=1))
        return fused, adp_logits, denoise, noises

    def forward(self, molgnet_pooled, molgnet_atoms, atom_mask, molt5, esm2_mean,
                esm2_residues, residue_mask, ec2vec, ph_temp, smiles_trfm,
                pocket_feats, pocket_pos, pocket_mask, modality_dropout=0.0):
        fused, adp_logits, denoise, noises = self.fuse(
            molgnet_pooled, molgnet_atoms, atom_mask, molt5, esm2_mean, esm2_residues,
            residue_mask, ec2vec, ph_temp, smiles_trfm, pocket_feats, pocket_pos,
            pocket_mask, modality_dropout)
        ifp_logits = self.ifp_decoder(fused)
        km_pred, kcat_pred = self.cgc_head(fused)
        return _clean(km_pred), _clean(kcat_pred), adp_logits, ifp_logits, denoise, noises
