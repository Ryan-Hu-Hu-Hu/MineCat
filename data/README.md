# Data

## `minecat_dataset.csv.gz` (48,891 rows)

Merged K<sub>M</sub> / k<sub>cat</sub> corpus assembled from the CataPro and OpenEnzymeDB predictor
datasets (which ultimately derive from BRENDA and SABIO-RK). One row per enzyme-substrate pair
and reaction condition: wild-type pairs with several distinct BRENDA assay conditions are
expanded to one row per (pH, temperature) condition, so rows of the same pair share sequence,
substrate and structure but differ in labels and conditions.

| column | description |
|---|---|
| `pair_id` | row index (0 ... 48,890); key used by every result file |
| `complex_id` | index of the unique (Sequence, Smiles) complex (42,761); names the chai-1 structure directory |
| `EC`, `EnzymeType`, `Organism`, `UniProtID` | enzyme annotation (`EnzymeType` = `wild` or the mutation) |
| `Substrate`, `Smiles` | substrate name and SMILES |
| `Sequence` | protein sequence (<= 1,000 residues) |
| `Km(M)`, `kcat(s^-1)` | measured values |
| `km_log10`, `kcat_log10` | log<sub>10</sub> labels used for training |
| `has_km`, `has_kcat` | label presence (46,944 rows with K<sub>M</sub>, 30,068 with k<sub>cat</sub>) |
| `ph`, `temp_c` | assay pH and temperature (deg C) |
| `cond_src` | provenance of the condition: exact BRENDA match (`brenda_exact`), per-condition expansion (`brenda_expand`), back-off to (EC, substrate) / (EC, organism) / EC level (`brenda_ec_sub`, `brenda_ec_org`, `brenda_ec`), earlier K<sub>M</sub> annotation (`km_prev`) or dataset median (`median`) |
| `fold` | cluster fold 0-9 (MMseqs2 40% identity clusters, greedy size-balanced; `scripts/data/cluster_folds.py`) |

Unique: 19,307 sequences, 4,405 substrates, 3,031 EC numbers.

Cross-validation protocol: test fold *f*, validation fold (*f* + 1) mod 10 (early stopping of the
deep model, stacking layer of the tree model), training = the remaining eight folds.

## `random_split_folds.csv`

`pair_id`, `fold_random`: the random 10-fold split used for the in-distribution
reference (Figure S13).

## `examples/example_pairs.csv`

Four example inputs for `scripts/predict.py` (human carbonic anhydrase II with CO<sub>2</sub> at
pH 8.8 / 0 deg C, two dataset pairs with conditions, and one pair without conditions).

## Features and structures

Pre-computed features (~11 GB, dominated by per-residue ESM2 embeddings) and the chai-1
structures are not stored in the repository; regenerate them with `scripts/data/` (see the main
README). The featurizers in `minecat/features/` reproduce the training features exactly (verified
on random rows: ESM2, ProtT5, MolT5, SMILES Transformer, MolGNet, descriptors, fingerprints,
EC2Vec and all structure features match the stored training inputs to <= 6e-8).
