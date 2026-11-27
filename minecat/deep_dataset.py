"""PyTorch dataset / collate for MineCat (deep).

Each item bundles the per-row modalities; `collate` pads protein residues and substrate
atoms to the longest member of the mini-batch, and pads the full-length distogram target
to the same (max_L x max_N) grid so the auxiliary loss covers the whole complex.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .models.deep import ADP_N_BINS, IFP_MAX_RES, N_INT_TYPES, POCKET_NODE_DIM

# distogram bins: 16 intervals over [2, 22) A + 1 "far" bin
ADP_EDGES = np.linspace(2.0, 22.0, ADP_N_BINS, dtype=np.float32)

# pH / temperature Gaussian radial-basis encoding
PH_CENTERS = np.linspace(0.0, 14.0, 32, dtype=np.float32)
PH_SIGMA = 0.5
T_CENTERS = np.linspace(0.0, 100.0, 32, dtype=np.float32)
T_SIGMA = 4.0


def _rbf(values, centers, sigma):
    return np.exp(-0.5 * ((values[:, None] - centers[None, :]) / sigma) ** 2)


def encode_ph_temp(ph, temp, ph_fill, temp_fill):
    """(N, 64) RBF encoding; missing values are filled with the training-fold medians."""
    ph = np.where(np.isfinite(ph), ph, ph_fill).astype(np.float32)
    temp = np.where(np.isfinite(temp), temp, temp_fill).astype(np.float32)
    return np.concatenate([_rbf(ph, PH_CENTERS, PH_SIGMA),
                           _rbf(temp, T_CENTERS, T_SIGMA)], axis=1).astype(np.float32)


def bin_distances(dist):
    """Calpha-ligand distance matrix -> (bin class int64, mask float32) at full size."""
    if dist is None:
        return torch.zeros((1, 1), dtype=torch.int64), torch.zeros((1, 1), dtype=torch.float32)
    d = np.clip(np.asarray(dist, dtype=np.float32), 2.0, None)
    bins = (np.digitize(d, ADP_EDGES) - 1).clip(0, ADP_N_BINS - 1).astype(np.int64)
    return torch.from_numpy(bins), torch.from_numpy(np.ones(bins.shape, dtype=np.float32))


def pad_ifp(ifp):
    out = np.zeros((IFP_MAX_RES, N_INT_TYPES), dtype=np.float32)
    mask = np.zeros(IFP_MAX_RES, dtype=np.float32)
    if ifp is not None:
        r = min(ifp.shape[0], IFP_MAX_RES)
        out[:r] = ifp[:r]
        mask[:r] = 1.0
    return torch.from_numpy(out.reshape(-1)), torch.from_numpy(mask)


def drop_virtual_node(atoms):
    """MolGNet appends one virtual (graph-level) node; drop it so tokens align 1:1 with
    the ligand heavy atoms of the chai-1 structure."""
    atoms = np.asarray(atoms, dtype=np.float32)
    return atoms[:-1] if atoms.shape[0] > 1 else atoms


class MineCatDeepDataset(Dataset):
    """Row-aligned inputs for MineCatDeep.

    Args (N = number of rows):
        sequences, smiles       : lists of length N (keys into the residue / atom dicts)
        molt5, esm2_mean, ec2vec, smiles_trfm, molgnet_pooled : (N, D) arrays
        ph_temp                 : (N, 64) RBF encoding
        esm2_residues           : dict {sequence: (L, 480)}
        molgnet_atoms           : dict {smiles: (n_atoms, 768)}  (virtual node already dropped)
        adp, ifp, pocket        : lists of length N (entries may be None)
        km_norm, km_mask, kcat_norm, kcat_mask : (N,) normalized labels and presence masks
    """

    def __init__(self, sequences, smiles, molt5, esm2_mean, ec2vec, ph_temp, smiles_trfm,
                 molgnet_pooled, esm2_residues, molgnet_atoms, adp, ifp, pocket,
                 km_norm=None, km_mask=None, kcat_norm=None, kcat_mask=None):
        n = len(sequences)
        self.sequences, self.smiles = sequences, smiles
        self.molt5 = np.asarray(molt5, dtype=np.float32)
        self.esm2_mean = np.asarray(esm2_mean, dtype=np.float32)
        self.ec2vec = np.asarray(ec2vec, dtype=np.float32)
        self.ph_temp = np.asarray(ph_temp, dtype=np.float32)
        self.smiles_trfm = np.asarray(smiles_trfm, dtype=np.float32)
        self.molgnet_pooled = np.asarray(molgnet_pooled, dtype=np.float32)
        self.esm2_residues, self.molgnet_atoms = esm2_residues, molgnet_atoms
        self.adp, self.ifp, self.pocket = adp, ifp, pocket
        zeros = np.zeros(n, dtype=np.float32)
        self.km_norm = zeros if km_norm is None else np.asarray(km_norm, dtype=np.float32)
        self.km_mask = zeros if km_mask is None else np.asarray(km_mask, dtype=np.float32)
        self.kcat_norm = zeros if kcat_norm is None else np.asarray(kcat_norm, dtype=np.float32)
        self.kcat_mask = zeros if kcat_mask is None else np.asarray(kcat_mask, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        f32 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
        residues = f32(self.esm2_residues[self.sequences[i]])
        atoms = f32(self.molgnet_atoms[self.smiles[i]])
        adp_bins, adp_mask = bin_distances(self.adp[i])
        ifp_vec, ifp_mask = pad_ifp(self.ifp[i])
        p = self.pocket[i]
        if p is not None:
            pk_feats = f32(p["node_feats"])
            pk_pos = f32(p["positions"])
            pk_mask = torch.ones(pk_feats.size(0))
        else:   # structure-free row: a single masked dummy atom
            pk_feats, pk_pos, pk_mask = torch.zeros(1, POCKET_NODE_DIM), torch.zeros(1, 3), torch.zeros(1)
        return dict(
            residues=residues, atoms=atoms, molgnet_pooled=f32(self.molgnet_pooled[i]),
            molt5=f32(self.molt5[i]), esm2_mean=f32(self.esm2_mean[i]),
            ec2vec=f32(self.ec2vec[i]), ph_temp=f32(self.ph_temp[i]),
            smiles_trfm=f32(self.smiles_trfm[i]),
            adp_bins=adp_bins, adp_mask=adp_mask, ifp=ifp_vec, ifp_mask=ifp_mask,
            pocket_feats=pk_feats, pocket_pos=pk_pos, pocket_mask=pk_mask,
            km=torch.tensor(self.km_norm[i]), km_mask=torch.tensor(self.km_mask[i]),
            kcat=torch.tensor(self.kcat_norm[i]), kcat_mask=torch.tensor(self.kcat_mask[i]),
            index=torch.tensor(i, dtype=torch.long))


def collate(items):
    B = len(items)
    max_L = max(it["residues"].size(0) for it in items)
    max_N = max(it["atoms"].size(0) for it in items)
    res = torch.zeros(B, max_L, items[0]["residues"].size(1))
    res_mask = torch.zeros(B, max_L)
    atoms = torch.zeros(B, max_N, items[0]["atoms"].size(1))
    atom_mask = torch.zeros(B, max_N)
    adp_bins = torch.zeros(B, max_L, max_N, dtype=torch.int64)
    adp_mask = torch.zeros(B, max_L, max_N)
    for b, it in enumerate(items):
        L, N = it["residues"].size(0), it["atoms"].size(0)
        res[b, :L], res_mask[b, :L] = it["residues"], 1.0
        atoms[b, :N], atom_mask[b, :N] = it["atoms"], 1.0
        p = min(it["adp_bins"].size(0), max_L)
        q = min(it["adp_bins"].size(1), max_N)
        adp_bins[b, :p, :q] = it["adp_bins"][:p, :q]
        adp_mask[b, :p, :q] = it["adp_mask"][:p, :q]
    stack = lambda k: torch.stack([it[k] for it in items])  # noqa: E731
    return dict(
        residues=res, residue_mask=res_mask, atoms=atoms, atom_mask=atom_mask,
        adp_bins=adp_bins, adp_mask=adp_mask,
        pocket_feats=[it["pocket_feats"] for it in items],
        pocket_pos=[it["pocket_pos"] for it in items],
        pocket_mask=[it["pocket_mask"] for it in items],
        **{k: stack(k) for k in ("molgnet_pooled", "molt5", "esm2_mean", "ec2vec", "ph_temp",
                                 "smiles_trfm", "ifp", "ifp_mask", "km", "km_mask", "kcat",
                                 "kcat_mask", "index")})


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = [x.to(device) for x in v] if isinstance(v, list) else v.to(device)
    return out


def model_inputs(batch):
    """Positional arguments for MineCatDeep.forward from a collated batch."""
    return (batch["molgnet_pooled"], batch["atoms"], batch["atom_mask"], batch["molt5"],
            batch["esm2_mean"], batch["residues"], batch["residue_mask"], batch["ec2vec"],
            batch["ph_temp"], batch["smiles_trfm"], batch["pocket_feats"],
            batch["pocket_pos"], batch["pocket_mask"])
