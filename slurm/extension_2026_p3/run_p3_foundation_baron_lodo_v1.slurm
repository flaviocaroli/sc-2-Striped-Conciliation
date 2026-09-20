#!/bin/bash
#SBATCH --job-name=p3-found-lodo
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_baron_lodo_v1/lodo_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_baron_lodo_v1/lodo_%A_%a.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2-data/bin/python

RUNNER="${REPO}/scripts/analysis/run_baron_foundation_lodo_v1.py"
PROTOCOL="${REPO}/configs/extension_2026_p3/p3_foundation_baron_representation_protocol_v1.json"

OUT="${SC2}/outputs/paper_extension_2026_p3/foundation_baron_representation_v1/lodo_v1"

: "${EXPECTED_JOB_HEAD:?}"
: "${EXPECTED_RUNNER_SHA:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test "$(sha256sum "${RUNNER}" | awk '{print $1}')" \
    = "${EXPECTED_RUNNER_SHA}"

test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" \
    = "${EXPECTED_PROTOCOL_SHA}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${REPO}"

export SC2_FOUNDATION_REP_PROTOCOL_SHA="${EXPECTED_PROTOCOL_SHA}"

"${PY}" "${RUNNER}" \
    --protocol "${PROTOCOL}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --output-root "${OUT}"

echo "P3_FOUNDATION_BARON_LODO_TASK=PASS"
