#!/bin/bash
#SBATCH --job-name=scfoundation-final
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3159436/sc2/logs/paper_extension_2026/scfoundation-final_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/paper_extension_2026/scfoundation-final_%A_%a.err

set -Eeuo pipefail

REPO=/home/3159436/sc2/code
PY=/home/3159436/.conda/envs/sc2-scfoundation/bin/python

PROTOCOL="${REPO}/configs/extension_2026/scfoundation_final_protocol_v1.json"
RUNNER="${REPO}/scripts/eval/run_scfoundation_final_task.py"

cd "$REPO"

[[ "$(git rev-parse HEAD)" == "${SC2_EXPECTED_COMMIT}" ]] || {
    echo "Git commit mismatch" >&2
    exit 1
}

env \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u LD_LIBRARY_PATH \
  SC2_SCFOUNDATION_FINAL_PROTOCOL_SHA="${SC2_SCFOUNDATION_FINAL_PROTOCOL_SHA}" \
  "$PY" "$RUNNER" \
    --protocol "$PROTOCOL" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
