"""Structure features from a chai-1 enzyme-substrate cofold (mmCIF).

For each complex we derive three inputs for MineCat (deep):

* Calpha-ligand distance matrix (n_residues x n_ligand_heavy_atoms), binned into a
  17-class distogram target for the ADP auxiliary decoder;
* interaction fingerprint (IFP): up to 64 residues whose Calpha lies within 8 A of the
  ligand centroid x 6 interaction types (H-bond, hydrophobic, aromatic, positive,
  negative, general contact);
* binding-pocket graph for the EGNN: ligand heavy atoms (<= 64) + backbone/C-beta atoms
  of up to 64 residues within 8 A of the ligand, 14-d one-hot node features.

The defaults (protein chain "A", ligand residue "LIG2") match chai-1 output; ESMFold-style
outputs typically name the ligand "LIG".
"""
import os

import numpy as np

PROTEIN_CHAIN = os.environ.get("PROTEIN_CHAIN", "A")
LIGAND_COMP = os.environ.get("LIGAND_COMP", "LIG2")
POCKET_CUTOFF = 8.0
MAX_PRO_RES = 64
MAX_LIG_ATOMS = 64
IFP_MAX_RES = 64
N_INT_TYPES = 6
AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}
POS_CHARGED = {"ARG", "LYS", "HIS"}
NEG_CHARGED = {"ASP", "GLU"}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
ELECTRONEG = {"N", "O", "S"}
PROT_ATOM_IDX = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}
LIG_ELEM_IDX = {"C": 5, "N": 6, "O": 7, "S": 8, "P": 9}
HALOGEN_IDX = 10
OTHER_IDX = 11
NODE_FEAT_DIM = 14


def _parse_cif_columns(cif_path):
    """Parse _atom_site loop from CIF and return per-atom dicts."""
    atoms = []  # list of dicts with fields
    col_names = []
    in_atom   = False
    col_group = col_type_sym = col_atom_id = col_comp_id = -1
    col_asym  = col_seq_id   = col_x = col_y = col_z = -1

    with open(cif_path, 'r') as fh:
        for raw_line in fh:
            line = raw_line.rstrip('\n')

            if line.strip() == 'loop_':
                col_names = []
                in_atom   = False
                col_group = col_type_sym = col_atom_id = col_comp_id = -1
                col_asym  = col_seq_id   = col_x = col_y = col_z = -1
                continue

            if line.startswith('_atom_site.'):
                field = line.strip().split('.', 1)[1]
                col_names.append(field)
                in_atom = True
                idx = len(col_names) - 1
                if   field == 'group_PDB':     col_group    = idx
                elif field == 'type_symbol':   col_type_sym = idx
                elif field == 'label_atom_id': col_atom_id  = idx
                elif field == 'label_comp_id': col_comp_id  = idx
                elif field == 'label_asym_id': col_asym     = idx
                elif field == 'label_seq_id':  col_seq_id   = idx
                elif field == 'Cartn_x':       col_x        = idx
                elif field == 'Cartn_y':       col_y        = idx
                elif field == 'Cartn_z':       col_z        = idx
                continue

            if not in_atom:
                continue
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue

            parts = line.split()
            need  = max(col_x, col_y, col_z) + 1 if col_x >= 0 else 0
            if len(parts) < need:
                continue

            try:
                x = float(parts[col_x]) if col_x >= 0 else 0.0
                y = float(parts[col_y]) if col_y >= 0 else 0.0
                z = float(parts[col_z]) if col_z >= 0 else 0.0
            except (ValueError, IndexError):
                continue

            atoms.append({
                'grp':    parts[col_group]    if col_group    >= 0 else '',
                'elem':   parts[col_type_sym] if col_type_sym >= 0 else 'C',
                'atom':   parts[col_atom_id]  if col_atom_id  >= 0 else '',
                'comp':   parts[col_comp_id]  if col_comp_id  >= 0 else '',
                'asym':   parts[col_asym]     if col_asym     >= 0 else '',
                'seq_id': parts[col_seq_id]   if col_seq_id   >= 0 else '0',
                'x': x, 'y': y, 'z': z,
            })

    return atoms


def parse_cif(cif_path, protein_chain=PROTEIN_CHAIN, ligand_comp=LIGAND_COMP):
    """
    Returns:
      ca_coords   : (L, 3) float32 — Cα positions
      lig_coords  : (M, 3) float32 — ligand heavy-atom positions
      residues    : dict {seq_id (int): {'name': str, 'atoms': {atom_name: (elem, x, y, z)}}}
      lig_atoms   : list of (elem, x, y, z)
    """
    raw = _parse_cif_columns(cif_path)

    ca_list    = []
    lig_list   = []
    residues   = {}
    lig_atoms  = []

    for a in raw:
        grp  = a['grp']
        elem = a['elem'].upper()
        atom = a['atom']
        comp = a['comp']
        asym = a['asym']
        x, y, z = a['x'], a['y'], a['z']

        if grp == 'ATOM' and asym == protein_chain:
            # Cα for ADP
            if atom == 'CA':
                ca_list.append([x, y, z])

            # All atoms for IFP
            try:
                res_num = int(a['seq_id'])
            except ValueError:
                continue
            if res_num not in residues:
                residues[res_num] = {'name': comp, 'atoms': {}}
            residues[res_num]['atoms'][atom] = (elem, x, y, z)

        elif grp == 'HETATM' and comp == ligand_comp and elem != 'H':
            lig_list.append([x, y, z])
            lig_atoms.append((elem, x, y, z))

    ca_coords  = np.array(ca_list,  dtype=np.float32) if ca_list  else np.zeros((0, 3), np.float32)
    lig_coords = np.array(lig_list, dtype=np.float32) if lig_list else np.zeros((0, 3), np.float32)

    return ca_coords, lig_coords, residues, lig_atoms


def compute_adp(ca_coords, lig_coords):
    """Calpha-ligand heavy-atom distance matrix (N_res, N_lig) float32."""
    if len(ca_coords) == 0 or len(lig_coords) == 0:
        return None
    diff = ca_coords[:, None, :] - lig_coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1)).astype(np.float32)


def compute_ifp(residues, lig_atoms):
    """IFP (IFP_MAX_RES, N_INT_TYPES) float32."""
    if not residues or not lig_atoms:
        return None

    lig_xyz   = np.array([[x, y, z] for (_, x, y, z) in lig_atoms], dtype=np.float32)
    lig_elems = [e for (e, _, _, _) in lig_atoms]
    lig_centroid = lig_xyz.mean(axis=0)

    ca_dists = []
    for seq_id, res_data in residues.items():
        if 'CA' not in res_data['atoms']:
            continue
        _, cax, cay, caz = res_data['atoms']['CA']
        ca = np.array([cax, cay, caz], dtype=np.float32)
        d  = float(np.linalg.norm(ca - lig_centroid))
        ca_dists.append((d, seq_id))

    ca_dists.sort()
    active_ids = [s for (d, s) in ca_dists if d <= POCKET_CUTOFF][:IFP_MAX_RES]

    if not active_ids:
        return None

    ifp = np.zeros((IFP_MAX_RES, N_INT_TYPES), dtype=np.float32)
    lig_no_mask = np.array([1 if e in ELECTRONEG else 0 for e in lig_elems])
    lig_c_mask  = np.array([1 if e == 'C' else 0 for e in lig_elems])

    for slot, seq_id in enumerate(active_ids):
        res_data = residues[seq_id]
        res_name = res_data.get('name', 'UNK').upper()
        atoms    = res_data['atoms']

        res_xyz   = np.array([[x, y, z] for (_, x, y, z) in atoms.values()], dtype=np.float32)
        res_elems = [e.upper() for (e, _, _, _) in atoms.values()]
        res_names = list(atoms.keys())

        if len(res_xyz) == 0:
            continue

        diff_all = res_xyz[:, None, :] - lig_xyz[None, :, :]
        dists    = np.sqrt((diff_all ** 2).sum(-1))

        min_dist = dists.min() if dists.size > 0 else 999.0
        ifp[slot, 5] = 1.0 if min_dist < 4.0 else 0.0

        for ai, (elem, atom_name) in enumerate(zip(res_elems, res_names)):
            if elem in ELECTRONEG and lig_no_mask.any():
                if dists[ai][lig_no_mask.astype(bool)].min() < 3.5:
                    ifp[slot, 0] = 1.0
                    break

        if lig_c_mask.any():
            for ai, (elem, atom_name) in enumerate(zip(res_elems, res_names)):
                if elem == 'C' and atom_name not in BACKBONE_ATOMS:
                    if dists[ai][lig_c_mask.astype(bool)].min() < 4.5:
                        ifp[slot, 1] = 1.0
                        break

        if res_name in AROMATIC_RES and 'CA' in atoms:
            _, cax, cay, caz = atoms['CA']
            ca_arr = np.array([cax, cay, caz], dtype=np.float32)
            ca_lig = np.sqrt(((ca_arr - lig_xyz) ** 2).sum(-1)).min()
            if ca_lig < 8.0:
                ifp[slot, 2] = 1.0

        if res_name in POS_CHARGED:
            for ai, (elem, atom_name) in enumerate(zip(res_elems, res_names)):
                if elem == 'N' and atom_name not in BACKBONE_ATOMS and lig_no_mask.any():
                    if dists[ai][lig_no_mask.astype(bool)].min() < 4.5:
                        ifp[slot, 3] = 1.0
                        break

        if res_name in NEG_CHARGED:
            for ai, (elem, atom_name) in enumerate(zip(res_elems, res_names)):
                if elem == 'O' and atom_name not in BACKBONE_ATOMS:
                    if dists[ai].min() < 4.5:
                        ifp[slot, 4] = 1.0
                        break

    return ifp


def compute_pocket(residues, lig_atoms):
    """Pocket coords for EGNN: {'node_feats': float16(N,14), 'positions': float32(N,3)}."""
    if not residues or not lig_atoms:
        return None

    lig_xyz   = np.array([[x, y, z] for (_, x, y, z) in lig_atoms[:MAX_LIG_ATOMS]], dtype=np.float32)
    lig_elems = [e for (e, _, _, _) in lig_atoms[:MAX_LIG_ATOMS]]

    ca_dists = []
    for seq_id, res_data in residues.items():
        if 'CA' not in res_data['atoms']:
            continue
        _, cax, cay, caz = res_data['atoms']['CA']
        ca = np.array([cax, cay, caz], dtype=np.float32)
        d  = float(np.min(np.linalg.norm(lig_xyz - ca[None, :], axis=1)))
        if d <= POCKET_CUTOFF:
            ca_dists.append((d, seq_id))

    ca_dists.sort()
    active_ids = [s for (_, s) in ca_dists[:MAX_PRO_RES]]

    if not active_ids:
        return None

    pro_positions = []
    pro_feats     = []
    for seq_id in active_ids:
        atoms = residues[seq_id]['atoms']
        for atom_name, (elem, x, y, z) in atoms.items():
            if atom_name not in PROT_ATOM_IDX:
                continue
            feat = np.zeros(NODE_FEAT_DIM, dtype=np.float16)
            feat[PROT_ATOM_IDX[atom_name]] = 1.0
            feat[-1] = 0.0
            pro_feats.append(feat)
            pro_positions.append([x, y, z])

    if not pro_positions:
        return None

    lig_feats = []
    for elem, (x, y, z) in zip(lig_elems, lig_xyz.tolist()):
        feat = np.zeros(NODE_FEAT_DIM, dtype=np.float16)
        if elem in LIG_ELEM_IDX:
            feat[LIG_ELEM_IDX[elem]] = 1.0
        elif elem in {'F', 'CL', 'BR', 'I'}:
            feat[HALOGEN_IDX] = 1.0
        else:
            feat[OTHER_IDX] = 1.0
        feat[-1] = 1.0
        lig_feats.append(feat)

    all_feats = lig_feats + pro_feats
    all_pos   = lig_xyz.tolist() + pro_positions

    return {
        'node_feats': np.array(all_feats, dtype=np.float16),
        'positions':  np.array(all_pos,   dtype=np.float32),
    }


def structure_features(cif_path, protein_chain=PROTEIN_CHAIN, ligand_comp=LIGAND_COMP):
    """Return (distances, ifp, pocket) for one complex; each is None if unavailable."""
    if cif_path is None or not os.path.exists(cif_path):
        return None, None, None
    ca, lig, residues, lig_atoms = parse_cif(cif_path, protein_chain, ligand_comp)
    if len(ca) == 0 or len(lig) == 0:
        return None, None, None
    return compute_adp(ca, lig), compute_ifp(residues, lig_atoms), compute_pocket(residues, lig_atoms)
