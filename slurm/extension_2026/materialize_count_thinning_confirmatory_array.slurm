#!/bin/bash
#SBATCH --job-name=sc2-thin-cmat
#SBATCH --output=/home/3159436/sc2/logs/paper_extension_2026/sc2-thin-cmat_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/paper_extension_2026/sc2-thin-cmat_%A_%a.err

set -Eeuo pipefail
umask 027

SC2_ROOT=/home/3159436/sc2
REPO="${SC2_ROOT}/code"

SC2_DATA_PY=/home/3159436/.conda/envs/sc2-data/bin/python

PROTOCOL="${REPO}/configs/extension_2026/count_thinning_confirmatory_protocol_v1.json"
SCRIPT="${REPO}/scripts/data/materialize_count_thinning_confirmatory.py"

: "${SC2_EXPECTED_COMMIT:?SC2_EXPECTED_COMMIT is required}"
: "${SC2_PHASE1C_PROTOCOL_SHA:?SC2_PHASE1C_PROTOCOL_SHA is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "$REPO"

ACTUAL_COMMIT="$(git rev-parse HEAD)"

if [[ "$ACTUAL_COMMIT" != "$SC2_EXPECTED_COMMIT" ]]; then
    echo "COMMIT_MISMATCH"
    echo "expected=${SC2_EXPECTED_COMMIT}"
    echo "actual=${ACTUAL_COMMIT}"
    exit 1
fi

ACTUAL_PROTOCOL_SHA="$(
    sha256sum "$PROTOCOL" \
    | awk '{print $1}'
)"

if [[ "$ACTUAL_PROTOCOL_SHA" != "$SC2_PHASE1C_PROTOCOL_SHA" ]]; then
    echo "PROTOCOL_SHA_MISMATCH"
    echo "expected=${SC2_PHASE1C_PROTOCOL_SHA}"
    echo "actual=${ACTUAL_PROTOCOL_SHA}"
    exit 1
fi

case "$SLURM_ARRAY_TASK_ID" in
    0)
        DATASET=internal_test
        ;;
    1)
        DATASET=baron_pancreas
        ;;
    2)
        DATASET=zheng68k
        ;;
    *)
        echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

OUTPUT_DIR="$(
    env \
        -u PYTHONPATH \
        -u PYTHONHOME \
        -u LD_LIBRARY_PATH \
        "$SC2_DATA_PY" - \
        "$PROTOCOL" \
        "$DATASET" \
        <<'PY'
import json
import sys

p = json.load(open(sys.argv[1]))
dataset = sys.argv[2]

print(
    p["materialization"]
     ["datasets"]
     [dataset]
     ["output_dir"]
)
PY
)"

echo "============================================================"
echo "PHASE1C_CONFIRMATORY_MATERIALIZATION_TASK"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "DATASET=${DATASET}"
echo "COMMIT=${ACTUAL_COMMIT}"
echo "PROTOCOL_SHA256=${ACTUAL_PROTOCOL_SHA}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "============================================================"

env \
    -u PYTHONPATH \
    -u PYTHONHOME \
    -u LD_LIBRARY_PATH \
    PYTHONPATH="${REPO}/src:${REPO}" \
    "$SC2_DATA_PY" \
    "$SCRIPT" \
    --protocol "$PROTOCOL" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR"

echo "PHASE1C_CONFIRMATORY_MATERIALIZATION_TASK=PASS"
