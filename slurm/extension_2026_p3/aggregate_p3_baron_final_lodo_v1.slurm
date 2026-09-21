#!/bin/bash
#SBATCH --job-name=p3-baron-lodo-agg
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_baron_final_lodo_v1/aggregate_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_baron_final_lodo_v1/aggregate_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"
PY=/home/3159436/.conda/envs/sc2-data/bin/python

PROTOCOL="${REPO}/configs/extension_2026_p3/p3_baron_final_expression_lodo_protocol_v1.json"
AGGREGATOR="${REPO}/scripts/analysis/aggregate_p3_baron_final_lodo_v1.py"

OUT="${SC2}/outputs/paper_extension_2026_p3/baron_lodo_final_v1"

: "${EXPECTED_JOB_HEAD:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"
: "${EXPECTED_AGGREGATOR_SHA:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test "$(
    sha256sum "${PROTOCOL}" |
    awk '{print $1}'
)" = "${EXPECTED_PROTOCOL_SHA}"

test "$(
    sha256sum "${AGGREGATOR}" |
    awk '{print $1}'
)" = "${EXPECTED_AGGREGATOR_SHA}"

test ! -e "${OUT}"

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${REPO}"

"${PY}" -u "${AGGREGATOR}" \
    --protocol "${PROTOCOL}" \
    --output-dir "${OUT}"

(
    cd "${OUT}"
    sha256sum -c SHA256SUMS.txt
)

echo "P3_BARON_FINAL_LODO_AGGREGATOR_JOB=PASS"
