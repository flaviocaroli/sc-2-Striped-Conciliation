#!/bin/bash
#SBATCH --job-name=p3-thin-select
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_count_thinning_validation_inference_v1/select_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_count_thinning_validation_inference_v1/select_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2-data/bin/python

PROTOCOL="${REPO}/configs/extension_2026_p3/p3_16k_count_thinning_validation_inference_v1.json"

SELECTOR="${REPO}/scripts/eval/select_p3_16k_count_thinning_thresholds_v1.py"

OUT="${SC2}/outputs/paper_extension_2026_p3/count_thinning_v1/validation_threshold_selection_v1"

: "${EXPECTED_JOB_HEAD:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"
: "${EXPECTED_SELECTOR_SHA:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" \
    = "${EXPECTED_PROTOCOL_SHA}"

test "$(sha256sum "${SELECTOR}" | awk '{print $1}')" \
    = "${EXPECTED_SELECTOR_SHA}"

test ! -e "${OUT}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${REPO}"

"${PY}" "${SELECTOR}" \
    --protocol "${PROTOCOL}" \
    --output-dir "${OUT}"

(
    cd "${OUT}"
    sha256sum -c SHA256SUMS.txt
)

"${PY}" - "${OUT}/selected_thresholds.json" <<'PY'
import json
import math
import sys
from pathlib import Path

p = json.loads(
    Path(sys.argv[1]).read_text()
)

assert p["status"] == "PASS"
assert p["scientific_result"] is True

assert p["selection_data"] == \
    "internal_validation_only"

assert p["validation_evaluations"] == 75
assert p["runs_per_q"] == 25
assert p["model_seeds"] == 5
assert p["thinning_replicates"] == 5

assert p["runtime_placeholder_used_for_selection"] is False

assert p["internal_test_used"] is False
assert p["baron_used"] is False
assert p["zheng68k_used"] is False

s = p["selected_thresholds"]

assert set(s) == {
    "0.50",
    "0.70",
    "0.85",
}

for q, x in s.items():

    threshold = float(
        x["threshold"]
    )

    assert math.isfinite(threshold)

    assert 0.0 <= threshold <= 1.0

    # Must be on the frozen 0.005 grid.
    assert abs(
        threshold * 200
        - round(threshold * 200)
    ) < 1e-8

    assert int(x["n_runs"]) == 25

    assert float(
        x["max_true_zero_fill"]
    ) <= 0.020000000001

print(
    "P3_THINNING_SELECTED_THRESHOLDS_QC=PASS"
)

print(
    json.dumps(
        s,
        indent=2,
        sort_keys=True,
    )
)
PY

echo "P3_THINNING_THRESHOLD_SELECTION_JOB=PASS"
