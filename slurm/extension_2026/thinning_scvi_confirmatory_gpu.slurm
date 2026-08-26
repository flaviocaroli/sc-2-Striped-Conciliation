#!/bin/bash
#SBATCH --job-name=thin-cmp-conf-scvi
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3159436/sc2/logs/paper_extension_2026/thin-cmp-conf-scvi_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/paper_extension_2026/thin-cmp-conf-scvi_%A_%a.err

set -Eeuo pipefail

REPO=/home/3159436/sc2/code
PY=/home/3159436/.conda/envs/sc2-p2-scvi/bin/python

cd "$REPO"

[[ "$(git rev-parse HEAD)" == "${SC2_EXPECTED_COMMIT}" ]] || exit 20

export PYTHONNOUSERSITE=1

env \
  -u PYTHONPATH \
  -u PYTHONHOME \
  SC2_THINNING_CONFIRMATORY_PROTOCOL_SHA="${SC2_THINNING_CONFIRMATORY_PROTOCOL_SHA}" \
  "$PY" \
  scripts/analysis/evaluate_thinning_scvi_confirmatory.py \
  --protocol configs/extension_2026/thinning_comparator_confirmatory_protocol_v1.json \
  --task-id "${SLURM_ARRAY_TASK_ID}"
