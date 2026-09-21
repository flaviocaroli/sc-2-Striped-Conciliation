#!/bin/bash
#SBATCH --job-name=p3-thin-src
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_count_thinning_confirmatory_sources_v1/job_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_count_thinning_confirmatory_sources_v1/job_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2-data/bin/python

SCRIPT="${REPO}/scripts/data/prepare_p3_16k_count_thinning_confirmatory_sources_v1.py"

OUT="/scratch/3159436/sc2_p3/outputs/count_thinning_v1/confirmatory_count_views_v1"

: "${EXPECTED_JOB_HEAD:?}"
: "${EXPECTED_SCRIPT_SHA:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test "$(
    sha256sum "${SCRIPT}" |
    awk '{print $1}'
)" = "${EXPECTED_SCRIPT_SHA}"

test ! -e "${OUT}"

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${REPO}"

echo "============================================================"
echo "P3 16K CONFIRMATORY THINNING COUNT SOURCES"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HEAD=${EXPECTED_JOB_HEAD}"
echo "HOST=$(hostname)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${PY}" -u "${SCRIPT}" \
    --output-dir "${OUT}"

test -s "${OUT}/count_views_receipt.json"
test -s "${OUT}/SHA256SUMS.txt"

(
    cd "${OUT}"
    sha256sum -c SHA256SUMS.txt
)

echo "P3_THINNING_CONFIRMATORY_SOURCES_JOB=PASS"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
