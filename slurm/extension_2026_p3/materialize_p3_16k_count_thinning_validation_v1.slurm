#!/bin/bash
#SBATCH --job-name=p3-thin-valmat
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_count_thinning_validation_v1/materialize_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_count_thinning_validation_v1/materialize_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2-data/bin/python

PROTOCOL="${REPO}/configs/extension_2026_p3/p3_16k_count_thinning_validation_materialization_v1.json"

SCRIPT="${REPO}/scripts/data/materialize_count_thinning_panels_p3_16k.py"

FINAL="/scratch/3159436/sc2_p3/outputs/count_thinning_v1/validation_panels_v1"
TEMP="${FINAL}.tmp_${SLURM_JOB_ID}"

: "${EXPECTED_JOB_HEAD:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"
: "${EXPECTED_SCRIPT_SHA:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" \
    = "${EXPECTED_PROTOCOL_SHA}"

test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" \
    = "${EXPECTED_SCRIPT_SHA}"

test ! -e "${FINAL}"

rm -rf "${TEMP}"

export PYTHONPATH="${REPO}/src:${REPO}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

"${PY}" "${SCRIPT}" \
    --protocol "${PROTOCOL}" \
    --output-dir "${TEMP}"

(
    cd "${TEMP}"
    sha256sum -c SHA256SUMS.txt
)

"${PY}" - "${TEMP}/validation_qc_summary.json" <<'PY'
import json
import sys
from pathlib import Path

q = json.loads(
    Path(sys.argv[1]).read_text()
)

assert q["scope"] == "internal_validation_only"
assert q["panel_count"] == 15
assert q["all_count_bounds_pass"] is True
assert q["all_integer_counts_pass"] is True
assert q["all_zero_library_cells"] == 0
assert q["all_full_target_alignment_exact"] is True
assert q["model_outcomes_inspected"] is False
assert q["test_or_external_materialized"] is False

assert q["source_identity"]["n_cells"] == 2500
assert q["source_identity"]["n_genes"] == 16384

print("P3_THINNING_MATERIALIZATION_QC=PASS")
PY

mv "${TEMP}" "${FINAL}"

echo "FINAL=${FINAL}"
echo "P3_16K_THINNING_VALIDATION_MATERIALIZATION=PASS"
