#!/bin/bash
#SBATCH --job-name=sc2-thin-val
#SBATCH --output=/home/3159436/sc2/logs/paper_extension_2026/sc2-thin-val_%j.out
#SBATCH --error=/home/3159436/sc2/logs/paper_extension_2026/sc2-thin-val_%j.err

set -Eeuo pipefail
umask 027

: "${SC2_EXPECTED_COMMIT:?SC2_EXPECTED_COMMIT is required}"

SC2_ROOT=/home/3159436/sc2
REPO="${SC2_ROOT}/code"
SC2_DATA_PY=/home/3159436/.conda/envs/sc2-data/bin/python

PROTOCOL="${REPO}/configs/extension_2026/count_thinning_protocol_v1.json"

FINAL_ROOT="${SC2_ROOT}/outputs/paper_extension_2026/count_thinning_v1/validation"
TEMP_ROOT="${FINAL_ROOT}.tmp_${SLURM_JOB_ID}"

cleanup () {
    if [[ -d "$TEMP_ROOT" ]]; then
        rm -rf "$TEMP_ROOT"
    fi
}

trap cleanup ERR INT TERM

cd "$REPO"

CURRENT_COMMIT="$(git rev-parse HEAD)"

if [[ "$CURRENT_COMMIT" != "$SC2_EXPECTED_COMMIT" ]]; then
    echo "ERROR: Git commit changed before job execution."
    echo "expected=${SC2_EXPECTED_COMMIT}"
    echo "actual=${CURRENT_COMMIT}"
    exit 20
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: repository is dirty in materialization job"
    git status --short
    exit 21
fi

if [[ -e "$FINAL_ROOT" ]]; then
    echo "ERROR: final validation thinning root already exists:"
    echo "$FINAL_ROOT"
    exit 22
fi

rm -rf "$TEMP_ROOT"

export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

echo "========================================"
echo "SC2 COUNT THINNING — VALIDATION ONLY"
echo "job=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "commit=${CURRENT_COMMIT}"
echo "protocol=${PROTOCOL}"
echo "temporary=${TEMP_ROOT}"
echo "final=${FINAL_ROOT}"
echo "========================================"

"$SC2_DATA_PY" --version

"$SC2_DATA_PY" \
    scripts/data/materialize_count_thinning_panels.py \
    --protocol "$PROTOCOL" \
    --output-dir "$TEMP_ROOT"

(
    cd "$TEMP_ROOT"
    sha256sum -c SHA256SUMS.txt
)

"$SC2_DATA_PY" - "$TEMP_ROOT/validation_qc_summary.json" <<'PY'
import json
import sys
from pathlib import Path

qc = json.loads(
    Path(sys.argv[1]).read_text()
)

assert qc["scope"] == "internal_validation_only"
assert qc["panel_count"] == 15
assert qc["all_count_bounds_pass"] is True
assert qc["all_integer_counts_pass"] is True
assert qc["all_zero_library_cells"] == 0
assert qc["all_full_target_alignment_exact"] is True
assert qc["model_outcomes_inspected"] is False
assert qc["test_or_external_materialized"] is False

print("SLURM_QC_GATE=PASS")
PY

mv "$TEMP_ROOT" "$FINAL_ROOT"

trap - ERR INT TERM

echo "FINAL_ROOT=${FINAL_ROOT}"
echo "VALIDATION_ONLY_MATERIALIZATION=PASS"
