#!/bin/bash
#SBATCH --job-name=p3-thin-cmat
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/3159436/sc2/logs/p3_count_thinning_confirmatory_materialization_v1/task_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/p3_count_thinning_confirmatory_materialization_v1/task_%A_%a.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"
PY=/home/3159436/.conda/envs/sc2-data/bin/python

PROTOCOL="${REPO}/configs/extension_2026_p3/p3_16k_count_thinning_confirmatory_materialization_v1.json"

SCRIPT="${REPO}/scripts/data/materialize_p3_16k_count_thinning_confirmatory_v1.py"

: "${EXPECTED_JOB_HEAD:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"
: "${EXPECTED_SCRIPT_SHA:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test "$(
    sha256sum "${PROTOCOL}" |
    awk '{print $1}'
)" = "${EXPECTED_PROTOCOL_SHA}"

test "$(
    sha256sum "${SCRIPT}" |
    awk '{print $1}'
)" = "${EXPECTED_SCRIPT_SHA}"

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${REPO}"

echo "============================================================"
echo "P3 CONFIRMATORY THINNING MATERIALIZATION"
echo "============================================================"
echo "ARRAY_JOB=${SLURM_ARRAY_JOB_ID}"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "HEAD=${EXPECTED_JOB_HEAD}"
echo "HOST=$(hostname)"

"${PY}" -u "${SCRIPT}" \
    --protocol "${PROTOCOL}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"

echo "P3_CONFIRMATORY_THINNING_TASK=PASS"
