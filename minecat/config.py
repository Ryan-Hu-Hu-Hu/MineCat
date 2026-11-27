"""Central paths and constants.

Every path can be overridden with an environment variable so the same code runs on any
machine without edits:

    MINECAT_DATASET   dataset CSV                 (default: data/minecat_dataset.csv.gz)
    MINECAT_FEATURES  feature store directory     (default: features/)
    MINECAT_WEIGHTS   trained model directory     (default: weights/)
    MINECAT_RESULTS   cross-validation outputs    (default: results/)
    MINECAT_EC2VEC    EC2Vec lookup table (CSV)   (default: external/EC2Vec/Embedding_Results/embedded_EC_number.csv)
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, default):
    return os.environ.get(name, default)


DATASET_CSV = _env("MINECAT_DATASET", os.path.join(REPO_ROOT, "data", "minecat_dataset.csv.gz"))
FEATURE_DIR = _env("MINECAT_FEATURES", os.path.join(REPO_ROOT, "features"))
WEIGHTS_DIR = _env("MINECAT_WEIGHTS", os.path.join(REPO_ROOT, "weights"))
RESULTS_DIR = _env("MINECAT_RESULTS", os.path.join(REPO_ROOT, "results"))
EC2VEC_CSV = _env("MINECAT_EC2VEC", os.path.join(
    REPO_ROOT, "external", "EC2Vec", "Embedding_Results", "embedded_EC_number.csv"))

SMILES_TRANSFORMER_CKPT = os.path.join(WEIGHTS_DIR, "smiles_transformer", "smiles_transformer.pt")
SMILES_TRANSFORMER_VOCAB = os.path.join(WEIGHTS_DIR, "smiles_transformer", "vocab.json")

# ---------------------------------------------------------------------------
# Cross-validation protocol
# ---------------------------------------------------------------------------
N_FOLDS = 10            # MMseqs2 40%-identity clusters packed into 10 folds
SEQ_IDENTITY = 0.4
MAX_PROTEIN_LEN = 1000  # sequences are truncated to 1,000 residues for ESM2
TASKS = ("km", "kcat")

# ---------------------------------------------------------------------------
# Feature store file names (all row-aligned to the dataset CSV unless noted)
# ---------------------------------------------------------------------------
FEATURE_FILES = {
    # protein
    "esm2_mean":        "esm2_35m_l11_mean.pkl",         # (N, 480)
    "esm2_residue":     "esm2_35m_l11_residue.pkl",      # dict {sequence: (L, 480) float16}
    "prot_t5":          "prot_t5_xl_mean.pkl",           # (N, 1024)
    "protein_desc":     "protein_descriptors.pkl",       # (N, 567)  AAC + CKSAAP + CTD
    # substrate
    "molt5":            "molt5_large.pkl",               # (N, 1024)
    "molgnet_pooled":   "molgnet_pooled.pkl",            # (N, 768)
    "molgnet_atoms":    "molgnet_atoms.pkl",             # dict {smiles: (n_atoms + 1, 768)}
    "smiles_trfm":      "smiles_transformer.pkl",        # (N, 1024) in-house encoder
    "maccs":            "maccs.pkl",                     # (N, 167)
    "morgan":           "morgan.pkl",                    # (N, 2048)
    # enzyme function
    "ec2vec":           "ec2vec.pkl",                    # (N, 1024)
    # structure (chai-1 cofold; list of length N, entry None if unavailable)
    "adp":              "structure_distances.pkl",       # (n_res, n_lig_atoms) Calpha-ligand distances
    "ifp":              "structure_ifp.pkl",             # (64, 6) interaction fingerprint
    "pocket":           "structure_pocket.pkl",          # {'node_feats': (n, 14), 'positions': (n, 3)}
}


def feature_path(key, feature_dir=None):
    return os.path.join(feature_dir or FEATURE_DIR, FEATURE_FILES[key])
