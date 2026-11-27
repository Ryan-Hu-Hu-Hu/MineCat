"""Hand-crafted, identity-independent descriptors for MineCat (composition).

protein : amino-acid composition (AAC, 20) + composition of k-spaced amino-acid pairs
          (CKSAAP, k = 0, 400) + composition/transition/distribution over 7
          physicochemical groupings (CTD, 147)                       -> 567-d
substrate: MACCS keys (167) + Morgan fingerprint (radius 2, 2048 bits) -> 2,215-d
"""
from itertools import product as iproduct

import numpy as np


def protein_descriptors(sequences):
    """Return {seq: (567,) float32} = AAC (20) + CKSAAP k=0 (400) + CTD (7 properties x 21 = 147)."""
    AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
    AA_PAIRS = [a + b for a, b in iproduct(AA_LIST, repeat=2)]  # 400 pairs

    CTD_GROUPS = {
        'hydrophobicity':     ({'A', 'G', 'V', 'S', 'T', 'P'},
                               {'D', 'E', 'H', 'K', 'R', 'N', 'Q'},
                               {'C', 'L', 'I', 'M', 'F', 'Y', 'W'}),
        'normalized_vdw':     ({'G', 'A', 'S', 'T', 'P', 'D', 'C'},
                               {'N', 'V', 'E', 'Q', 'I', 'L'},
                               {'M', 'H', 'K', 'F', 'R', 'Y', 'W'}),
        'polarity':           ({'L', 'I', 'F', 'W', 'C', 'M', 'V', 'Y'},
                               {'P', 'A', 'T', 'G', 'S'},
                               {'H', 'Q', 'R', 'K', 'N', 'E', 'D'}),
        'charge':             ({'K', 'R'},
                               {'A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'},
                               {'D', 'E'}),
        'secondary_struct':   ({'E', 'A', 'L', 'M', 'Q', 'K', 'R', 'H'},
                               {'V', 'I', 'Y', 'C', 'W', 'F', 'T'},
                               {'G', 'N', 'P', 'S', 'D'}),
        'solvent_access':     ({'A', 'L', 'F', 'C', 'G', 'I', 'V', 'W'},
                               {'R', 'K', 'Q', 'E', 'N', 'D'},
                               {'M', 'S', 'P', 'T', 'H', 'Y'}),
        'polarizability':     ({'G', 'A', 'S', 'D', 'T'},
                               {'C', 'P', 'N', 'V', 'E', 'Q', 'I', 'L'},
                               {'K', 'M', 'H', 'F', 'R', 'Y', 'W'}),
    }

    def compute_aac(seq):
        seq = seq.upper()
        n = max(len(seq), 1)
        return np.array([seq.count(aa) / n for aa in AA_LIST], dtype=np.float32)

    def compute_cksaap(seq, k=0):
        seq = seq.upper()
        counts = {p: 0 for p in AA_PAIRS}
        n = 0
        for i in range(len(seq) - k - 1):
            p = seq[i] + seq[i + k + 1]
            if p in counts:
                counts[p] += 1
                n += 1
        n = max(n, 1)
        return np.array([counts[p] / n for p in AA_PAIRS], dtype=np.float32)

    def seq_to_group_seq(seq, groups):
        g = []
        for aa in seq.upper():
            if   aa in groups[0]: g.append(0)
            elif aa in groups[1]: g.append(1)
            elif aa in groups[2]: g.append(2)
        return g

    def ctdc_one(g_seq):
        n = max(len(g_seq), 1)
        return np.array([g_seq.count(i) / n for i in range(3)], dtype=np.float32)

    def ctdt_one(g_seq):
        n = max(len(g_seq) - 1, 1)
        t = np.zeros(3, dtype=np.float32)
        for i in range(len(g_seq) - 1):
            a, b = min(g_seq[i], g_seq[i+1]), max(g_seq[i], g_seq[i+1])
            if   (a, b) == (0, 1): t[0] += 1
            elif (a, b) == (0, 2): t[1] += 1
            elif (a, b) == (1, 2): t[2] += 1
        return t / n

    def ctdd_one(g_seq):
        n = max(len(g_seq), 1)
        result = []
        for grp in range(3):
            positions = [i / n for i, v in enumerate(g_seq) if v == grp]
            if not positions:
                result.extend([0.0] * 5)
            else:
                percentiles = [0, 25, 50, 75, 100]
                result.extend([float(np.percentile(positions, p)) for p in percentiles])
        return np.array(result, dtype=np.float32)

    def compute_ctdc(seq):
        feats = []
        for prop, (g0, g1, g2) in CTD_GROUPS.items():
            g_seq = seq_to_group_seq(seq, (g0, g1, g2))
            feats.append(ctdc_one(g_seq))   # 3
            feats.append(ctdt_one(g_seq))   # 3
            feats.append(ctdd_one(g_seq))   # 15
        return np.concatenate(feats)   # 7 × 21 = 147

    result = {}
    for seq in dict.fromkeys(sequences):
        aac  = compute_aac(seq)       # 20
        ksaap = compute_cksaap(seq)   # 400
        ctdc  = compute_ctdc(seq)     # 147
        desc  = np.concatenate([aac, ksaap, ctdc]).astype(np.float32)
        result[seq] = desc

    return result


def fingerprints(smiles_list):
    """Return ({smiles: MACCS (167,)}, {smiles: Morgan r=2 (2048,)}) as float32 bit vectors."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem

    maccs_dict  = {}
    morgan_dict = {}

    for smi in dict.fromkeys(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  WARNING: cannot parse SMILES for fingerprints: {smi[:60]}")
            maccs_dict[smi]  = np.zeros(167, dtype=np.float32)
            morgan_dict[smi] = np.zeros(2048, dtype=np.float32)
            continue

        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_dict[smi] = np.array(maccs, dtype=np.float32)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        morgan_dict[smi] = np.array(fp, dtype=np.float32)

    return maccs_dict, morgan_dict
