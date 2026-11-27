# Model weights

| path | content | size | in git |
|---|---|---|---|
| `model_meta.json` | label normalization ranges and pH / temperature medians used at inference | 1 KB | yes |
| `smiles_transformer/smiles_transformer.pt`, `vocab.json` | in-house SMILES Transformer (ChEMBL 37, 47-token vocabulary) | 22 MB | yes |
| `deep/fold{0-9}.pt` | MineCat (deep) state_dicts, one per cluster fold | 10 x 730 MB | no |
| `tree/{km,kcat}/fold{0-9}.joblib` | MineCat (tree): dict with `base`, `first` (pH/T-aware) ExtraTrees and the `second` linear stacker | ~146 GB in total | no |
| `composition/{km,kcat}/fold{0-9}.joblib` | MineCat (composition) ExtraTrees | ~71 GB in total | no |

The fold models are trained by `scripts/train/` directly into this layout, and
`scripts/predict.py` averages the ten fold models of each view. The large checkpoints are
distributed separately (download link to be added); place them here or point
`MINECAT_WEIGHTS` to their directory.

The tree checkpoints are scikit-learn 1.7.2 pickles; load them with the same scikit-learn
version. Fully grown 1,000-tree forests are large; `predict.py --folds 0,1,2` uses a subset of
fold models if memory or disk is limited.
