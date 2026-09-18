#!/bin/bash
#SBATCH --job-name=p3-scf-val
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/3159436/sc2/logs/p3_scfoundation_validation_v1/scf_val_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/p3_scfoundation_validation_v1/scf_val_%A_%a.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2-scfoundation/bin/python

RUNNER="${REPO}/scripts/eval/run_scfoundation_p3_16k_validation_v1.py"
PROTOCOL="${REPO}/configs/extension_2026_p3/p3_scfoundation_16k_validation_protocol_v1.json"

: "${EXPECTED_JOB_HEAD:?}"
: "${EXPECTED_RUNNER_SHA:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = \
    "${EXPECTED_JOB_HEAD}"

test -z "$(git status --porcelain=v1 -uall)"

test "$(
    sha256sum "${RUNNER}" |
    awk '{print $1}'
)" = "${EXPECTED_RUNNER_SHA}"

test "$(
    sha256sum "${PROTOCOL}" |
    awk '{print $1}'
)" = "${EXPECTED_PROTOCOL_SHA}"

export PYTHONPATH="${REPO}/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "P3 scFOUNDATION FULL VALIDATION"
echo "============================================================"
echo "ARRAY_JOB=${SLURM_ARRAY_JOB_ID}"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "HOST=$(hostname)"

nvidia-smi \
    --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv,noheader

"${PY}" "${RUNNER}" \
    --protocol "${PROTOCOL}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"

echo "P3_SCF_FULL_VALIDATION_TASK=PASS"
