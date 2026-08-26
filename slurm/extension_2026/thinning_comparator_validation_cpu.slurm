#!/bin/bash
#SBATCH --job-name=thin-cmp-val-det
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3159436/sc2/logs/paper_extension_2026/thin-cmp-val-det_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/paper_extension_2026/thin-cmp-val-det_%A_%a.err

set -Eeuo pipefail

REPO=/home/3159436/sc2/code
PY=/home/3159436/.conda/envs/sc2-data/bin/python

cd "$REPO"

[[ "$(git rev-parse HEAD)" == "${SC2_EXPECTED_COMMIT}" ]] || exit 20

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src:${REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"$PY" \
  scripts/analysis/evaluate_thinning_comparator_validation.py \
  --protocol configs/extension_2026/thinning_comparator_validation_protocol_v1.json \
  --task-id "${SLURM_ARRAY_TASK_ID}"
