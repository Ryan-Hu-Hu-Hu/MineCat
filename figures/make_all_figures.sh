#!/usr/bin/env bash
# Regenerate every data-driven figure of the paper from results/ and figures/source_data/.
set -euo pipefail
cd "$(dirname "$0")/.."
python figures/make_source_data.py
for f in figures/main/fig2_benchmark_radar.py \
         figures/main/fig3_parity_error_correlation.py \
         figures/main/fig4_sequence_identity.py \
         figures/main/fig5_reconstruction.py \
         figures/main/fig6_representation_umap.py \
         figures/main/fig7_external_evaluation.py \
         figures/supplementary/figS6_blend_landscape.py \
         figures/supplementary/figS7_member_radar.py \
         figures/supplementary/figS10_substrate_space.py \
         figures/supplementary/figS11_protein_ec_space.py \
         figures/supplementary/figS13_cluster_vs_random.py; do
  echo "== $f"; python "$f"
done
