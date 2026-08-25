#!/bin/bash
#SBATCH --job-name=sc2-baron-lodo
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3159436/sc2/logs/paper_extension_2026/sc2-baron-lodo_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/paper_extension_2026/sc2-baron-lodo_%A_%a.err

set -Eeuo pipefail

REPO=/home/3159436/sc2/code
PY=/home/3159436/.conda/envs/sc2-data/bin/python
PROTOCOL="${REPO}/configs/extension_2026/baron_lodo_classifier_protocol_v1.json"
RUNNER="${REPO}/scripts/analysis/run_baron_lodo_classifier_task.py"
OUTPUT=/home/3159436/sc2/outputs/paper_extension_2026/baron_lodo_v1/classification_v1

cd "$REPO"

[[ "$(git rev-parse HEAD)" == "${SC2_EXPECTED_COMMIT}" ]] || {
    echo "Commit mismatch" >&2
    exit 1
}

ACTUAL="$(
    sha256sum "$PROTOCOL" |
    awk '{print $1}'
)"

[[ "$ACTUAL" == "${SC2_BARON_CLASSIFIER_PROTOCOL_SHA}" ]] || {
    echo "Protocol SHA mismatch" >&2
    exit 1
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

env \
    -u PYTHONHOME \
    -u LD_LIBRARY_PATH \
    PYTHONPATH="${REPO}/src:${REPO}" \
    SC2_BARON_CLASSIFIER_PROTOCOL_SHA="${SC2_BARON_CLASSIFIER_PROTOCOL_SHA}" \
    "$PY" "$RUNNER" \
      --protocol "$PROTOCOL" \
      --task-id "${SLURM_ARRAY_TASK_ID}" \
      --output-root "$OUTPUT"
