# Figures and source data

Every data-driven figure of the paper is regenerated from `results/` and `source_data/` by
the scripts below (`bash figures/make_all_figures.sh`). Rendered PNG (300 dpi) and PDF files
sit next to each script. The outputs were checked against the images in the manuscript:
all are pixel-identical except Figure S12, whose 2 x 4 panel arrangement is re-created here.

Style: `plotting_style.py` + `figure_style.yaml` (Arial, bold labels, white background).
`common.py` holds paths and the loader for the out-of-fold prediction files.
`make_source_data.py` derives the tables marked (*) from `results/`.

## Main figures

| Figure | Script | Source data |
|---|---|---|
| 1 Framework overview | schematic (not code-generated) | - |
| 2 Benchmark radar (K<sub>M</sub>, k<sub>cat</sub> x with / without conditions) | `main/fig2_benchmark_radar.py` | `benchmark_metrics.csv` (*) |
| 3 Parity plots (A) and error correlation (B) | `main/fig3_parity_error_correlation.py` (runs `fig3a_parity.py`, `fig3b_error_correlation.py`) | `results/cluster_cv/oof_predictions.csv.gz`; `error_correlation_{km,kcat}.csv` (*) |
| 4 Generalization vs test-train sequence identity | `main/fig4_sequence_identity.py` | `seqid_metrics.csv` (*) |
| 5 Auxiliary-decoder reconstruction | `main/fig5_reconstruction.py` | `fig5_reconstruction/` |
| 6 UMAP of deep representations | `main/fig6_representation_umap.py` | `fig6_representation_umap.csv.gz` |
| 7 External datasets (leaderboard A-F, UMAP G-K) | `main/fig7_external_evaluation.py` | `fig7_external/` |
| 8 MineCat-guided RcALAS engineering | schematic + wet-lab data (not code-generated) | - |

## Supplementary figures

| Figure | Script | Source data |
|---|---|---|
| S1-S3 Architectures of the three views | schematics | - |
| S4, S5, S8 Dataset composition, preparation and clustering workflows | schematics | `data/minecat_dataset.csv.gz` |
| S6 Blend-weight landscape | `supplementary/figS6_blend_landscape.py` | `blend_landscape_{km,kcat}.csv` (*) |
| S7 Member radar | `supplementary/figS7_member_radar.py` | `benchmark_metrics.csv` (*) |
| S9 Member sequence-identity heatmaps | written by `main/fig4_sequence_identity.py` | `seqid_metrics.csv` (*) |
| S10 Substrate chemical space | `supplementary/figS10_substrate_space.py` | `figS10_substrate_umap.csv.gz` |
| S11 Protein / EC space | `supplementary/figS11_protein_ec_space.py` | `figS11_protein_ec_umap.csv.gz` |
| S12 Feature-shift mechanism | written by `main/fig7_external_evaluation.py` | `fig7_external/` |
| S13 Cluster vs random split | `supplementary/figS13_cluster_vs_random.py` | `cluster_vs_random.csv` (*) |
| S14-S17 Sequence alignment, SDS-PAGE, in vivo 5-ALA production, purification | wet-lab | - |

## Tables

`source_data/tables/` holds every table as CSV, plus the per-pair data behind Table 4:

| Table | File(s) |
|---|---|
| 1 Modality ablation (Delta R<sup>2</sup>) | `table1_modality_ablation.csv` |
| 2 Per-EC-class performance | `table2_per_ec_class.csv` |
| 3 Reaction-condition stratification | `table3_condition_stratification.csv` |
| 4 Case-study ranking (13 models) | `table4_case_study_ranking.csv`; `table4_carbonic_anhydrase_per_variant.csv` (predictions + measured specific activity, 11 variants); `table4_alas_benchmark_per_pair.csv` (predictions + measured K<sub>M</sub> / k<sub>cat</sub>, 28 pairs, `novel` flag for Table S2) |
| S1 Datasets | `tableS1_datasets.csv` |
| S2 ALAS benchmark, novel-only | `tableS2_alas_novel_only.csv` |
| S3 Strains and plasmids | `tableS3_strains_plasmids.csv` |

`blend_weight_vs_identity.csv` is not used by a figure: it records the optimal blend
weights per test-train sequence-identity bin (`scripts/analysis/blend_weight_sensitivity.py`),
the evidence behind MineCat offering provenance-based rather than identity-based blend presets.

## Notes on the precomputed source data

* `fig5_reconstruction/`: outputs of the ten MineCat (deep) checkpoints on 220 held-out
  fold-0 complexes (distogram bins, IFP probabilities, cross-attention weights) and on the
  curated held-out complex `pair_id` 22137 (per-residue expected distances, IFP, attention,
  EGNN denoising trajectory), plus the PyMOL renders of panels D and F.
* `fig6_representation_umap.csv.gz`: fold-0 model activations (fused input, K<sub>M</sub> and
  k<sub>cat</sub> expert outputs) on held-out pairs, PCA-reduced, UMAP (n_neighbors 25,
  min_dist 0.10, cosine, random_state 0).
* `figS10_*`, `figS11_*`: one point per unique substrate / sequence / EC number; standardized
  embeddings, 50 PCA components, UMAP (n_neighbors 30, min_dist 0.25, cosine, random_state 0).
* `fig7_external/`: external evaluation on HXKm, CatPred40, IECata and EnzyExtract (pairs
  overlapping the training data removed): per-fold SCC leaderboard, blend-weight sweeps,
  domain-classifier AUCs, OOD-binned calibration / crossover statistics, and UMAP coordinates.
* `benchmark_metrics.csv`: MineCat rows recomputed from the out-of-fold predictions; the
  published predictors use their per-fold training logs (`results/cluster_cv/benchmark_fold_metrics.csv`).
  The two agree to 4-decimal rounding except DLTKcat k<sub>cat</sub>, whose prediction dump
  differs from its training log on folds 0-6 (R<sup>2</sup> -0.185 from predictions vs -0.193 logged).
