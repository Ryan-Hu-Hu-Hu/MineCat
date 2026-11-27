# MineCat

MineCat predicts the enzyme kinetic parameters **K<sub>M</sub>** (Michaelis constant) and
**k<sub>cat</sub>** (turnover number) for an enzyme-substrate pair, and is evaluated under a
leakage-free split in which no enzyme family (MMseqs2 cluster at 40% sequence identity) is
shared between training and test data.

MineCat combines three complementary views, blended in log<sub>10</sub> space:

| View | Model | Inputs |
|---|---|---|
| **MineCat (tree)** | Extremely Randomized Trees + pH/temperature stacking cascade | frozen ProtT5-XL, in-house SMILES Transformer, MolT5-large, MolGNet and EC2Vec embeddings (4,864-d) |
| **MineCat (deep)** | multi-modal network (8 modalities, bidirectional substrate-protein cross-attention, CGC multi-task head) with structure-reconstruction auxiliary decoders (distogram, interaction fingerprint, pocket denoising) | MolGNet atom tokens, MolT5, SMILES Transformer, ESM2-35M residues, EC2Vec, chai-1 binding-pocket EGNN, pH/temperature |
| **MineCat (composition)** | Extremely Randomized Trees | identity-independent descriptors: AAC + CKSAAP + CTD (protein), MACCS + Morgan (substrate) |

Blend weights (tree / deep / composition) are chosen leave-one-fold-out on the cluster
split: K<sub>M</sub> 0.705 / 0.250 / 0.045 and k<sub>cat</sub> 0.565 / 0.265 / 0.170.

## Results (10-fold cluster cross-validation, 48,891 pairs)

Fold-mean test R<sup>2</sup> (SD over folds); full metrics in `results/` and Figure 2.

| Model | K<sub>M</sub> R<sup>2</sup> | k<sub>cat</sub> R<sup>2</sup> |
|---|---|---|
| **MineCat (ensemble)** | **0.411** (0.060) | **0.227** (0.049) |
| MineCat (tree) | 0.406 (0.057) | 0.208 (0.045) |
| MineCat (deep) | 0.364 (0.066) | 0.139 (0.078) |
| MineCat (composition) | 0.313 (0.065) | 0.131 (0.042) |
| EF-UniKP (best published predictor, retrained on the same folds) | 0.379 (0.059) | 0.189 (0.056) |

For reference, a conventional random 10-fold split gives K<sub>M</sub> R<sup>2</sup> 0.672 and
k<sub>cat</sub> R<sup>2</sup> 0.678 for the same ensemble (Figure S13).

## Repository layout

```
minecat/                     Python package
  models/                    MineCatDeep, cross-attention, EGNN, CGC head
  features/                  featurizers: ESM2 / ProtT5, MolT5 / SMILES Transformer,
                             composition descriptors, EC2Vec, chai-1 structure features
  smiles_transformer/        in-house SMILES Transformer (model, tokenizer, encoder)
  data.py, deep_dataset.py   dataset, labels, cluster splits, metrics; deep-model batching
  ensemble.py                leave-one-fold-out simplex blend
scripts/
  data/                      cluster_folds.py, fold_structures.py (chai-1),
                             build_features.py, build_molgnet_features.py
  smiles_transformer/        prepare_corpus.py, build_vocab.py, pretrain.py (ChEMBL 37)
  train/                     train_tree.py, train_deep.py, train_composition.py, build_ensemble.py
  analysis/                  test_train_identity.py (remote-homology analysis)
  predict.py                 K_M / k_cat prediction for new pairs
data/                        dataset (48,891 rows with folds), random-split folds, example input
results/                     out-of-fold predictions of all MineCat views and 9 published predictors
weights/                     model metadata + SMILES Transformer weights (other weights: see weights/README.md)
figures/                     figure scripts, source data and rendered figures (see figures/README.md)
```

## Installation

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate minecat
pip install -e .                          # optional (editable install keeps the repo-relative default paths)
```

A CUDA GPU is recommended for feature extraction and the deep model; the tree models are CPU-only.
Two external resources are needed for featurization and are **not** redistributed here:

* **EC2Vec** (GPL-3.0): `git clone https://github.com/MengLiu90/EC2Vec external/EC2Vec`
  (or set `MINECAT_EC2VEC` to its `Embedding_Results/embedded_EC_number.csv`).
* **MolGNet** (distributed with PreTKcat): `git clone https://github.com/MrVincentCai/PreTKcat external/PreTKcat`;
  run `scripts/data/build_molgnet_features.py` in an environment with `torch_geometric`.

Structure features use **chai-1** (`pip install chai_lab==0.6.1`).

## Quick start

**Reproduce the paper's numbers and figures** (no GPU, no model weights needed):

```bash
python scripts/train/build_ensemble.py --from-shipped    # LOFO blend from the released out-of-fold predictions
bash figures/make_all_figures.sh                          # regenerates Figures 2-7 and S6, S7, S9-S13
```

**Predict K<sub>M</sub> / k<sub>cat</sub> for new pairs** (requires the trained weights, see `weights/README.md`).
Input: a CSV with `Sequence`, `Smiles`, `EC`, and optionally `ph`, `temp_c` (e.g. `data/examples/example_pairs.csv`).

```bash
python scripts/data/fold_structures.py --csv pairs.csv --out structures/          # chai-1 cofold (optional)
python scripts/data/build_features.py  --csv pairs.csv --structures structures/ --out pairs_features/
python scripts/data/build_molgnet_features.py --csv pairs.csv --out pairs_features/ --pretkcat external/PreTKcat
python scripts/predict.py --csv pairs.csv --features pairs_features/ --out predictions.csv
```

`predictions.csv` contains log<sub>10</sub> K<sub>M</sub> [M], log<sub>10</sub> k<sub>cat</sub> [s<sup>-1</sup>],
log<sub>10</sub> k<sub>cat</sub>/K<sub>M</sub>, the spread across the 10 fold models, and each view's own prediction.
Rows with both pH and temperature use the condition-aware arm; other rows the condition-free arm.

### Choosing the blend weights

`--blend` (and per-task `--blend-km` / `--blend-kcat`) selects how the three views are mixed:

| value | tree / deep / composition | when to use |
|---|---|---|
| `internal` (default) | K<sub>M</sub> 0.705 / 0.250 / 0.045, k<sub>cat</sub> 0.565 / 0.265 / 0.170 | pairs from the same kind of source as the training corpus (BRENDA / SABIO-RK-style measurements featurized with these scripts) |
| `external` | K<sub>M</sub> 0.00 / 0.30 / 0.70, k<sub>cat</sub> 0.00 / 0.35 / 0.65 | pairs from a different curation pipeline |
| `"0.2,0.3,0.5"` | explicit, renormalized to sum to 1 | custom |

The `external` preset is measured, not guessed: on four independent datasets (HXKm,
CatPred40, IECata, EnzyExtract) the tree member's optimal weight collapses to 0.00 and
composition takes 0.60-0.75, worth **+0.06 to +0.21 R<sup>2</sup>** over the internal weights.
The cause is feature-distribution shift - a domain classifier separates internal from
external pairs almost perfectly in the tree and deep feature spaces (AUC 0.91-0.98) but far
less well for composition descriptors (AUC 0.62-0.87), which cannot encode protein identity.

**Sequence identity does not select the preset.** Identity to the training set strongly
affects *accuracy* (K<sub>M</sub> R<sup>2</sup> ~0.49 at >=40% global identity, ~0.35 below 10%), but not the
optimal *weights*: refitting them per identity bin inside the cluster split changes R<sup>2</sup> by
-0.017 to +0.006 (mean -0.002, i.e. no gain), and the composition weight shows no trend with
identity. On external data the composition-heavy optimum likewise holds at every novelty
decile rather than ramping in. Reproduce with:

```bash
python scripts/analysis/blend_weight_sensitivity.py
```

**Train from scratch** (10-fold cluster CV; each tree fold takes roughly an hour on ~30 CPU cores, each deep fold several hours on one GPU):

```bash
python scripts/data/fold_structures.py --csv data/minecat_dataset.csv.gz --out structures/
python scripts/data/build_features.py  --csv data/minecat_dataset.csv.gz --structures structures/ --out features/
python scripts/data/build_molgnet_features.py --csv data/minecat_dataset.csv.gz --out features/
python scripts/train/train_tree.py --task km;          python scripts/train/train_tree.py --task kcat
python scripts/train/train_composition.py --task km;   python scripts/train/train_composition.py --task kcat
python scripts/train/train_deep.py
python scripts/train/build_ensemble.py
```

Optional: retrain the SMILES Transformer on ChEMBL 37 with `scripts/smiles_transformer/`
(the released weights are in `weights/smiles_transformer/`).

All paths can be redirected with `MINECAT_DATASET`, `MINECAT_FEATURES`, `MINECAT_WEIGHTS`,
`MINECAT_RESULTS` and `MINECAT_EC2VEC` (see `minecat/config.py`).

## Data

`data/minecat_dataset.csv.gz` merges K<sub>M</sub> and k<sub>cat</sub> records from the CataPro
and OpenEnzymeDB predictor datasets (which derive from BRENDA and SABIO-RK), with reaction
conditions and the 10 cluster folds; see `data/README.md`.

## Citation

If you use MineCat, please cite the accompanying article (citation to be added on publication).

## License

Code and the in-house SMILES Transformer weights: MIT (see `LICENSE`). Third-party models used
for featurization (ESM2, ProtT5, MolT5, MolGNet, EC2Vec, chai-1) are subject to their own licenses.
