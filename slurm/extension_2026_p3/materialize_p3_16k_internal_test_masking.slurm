#!/bin/bash
#SBATCH --job-name=sc2-p3-test16k-mat
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3159436/sc2/logs/p3_internal_test_masking_v1/test16k_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_internal_test_masking_v1/test16k_%j.err

set -Eeuo pipefail
umask 027

REPO=/home/3159436/sc2/code
PY=/home/3159436/.conda/envs/sc2-data/bin/python

PROTOCOL=/home/3159436/sc2/code/configs/extension_2026_p3/p3_16k_internal_test_masking_protocol_v1.json
SCRIPT=/home/3159436/sc2/code/scripts/data/materialize_p3_16k_internal_test_masking.py

TARGET=/home/3159436/sc2/data/census_curated_pretrain_pilot250k_g16384/benchmarks/test_masking_v1
TMP=/home/3159436/sc2/data/census_curated_pretrain_pilot250k_g16384/benchmarks/test_masking_v1.tmp

: "${EXPECTED_HEAD:?EXPECTED_HEAD missing}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

unset PYTHONPATH
unset LD_LIBRARY_PATH
unset PYTHONHOME

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

test ! -e "${TARGET}"
test ! -e "${TMP}"

echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HEAD=$(git rev-parse HEAD)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${PY}" "${SCRIPT}"     --protocol "${PROTOCOL}"     --output-root "${TMP}"

for RATE in 15 30 50; do
    test -s "${TMP}/test_mask${RATE}.npz"
done

test -s "${TMP}/test_panels_manifest.json"

"${PY}" - "${TMP}" <<'PYQC'
from pathlib import Path
import json
import sys
import numpy as np

root = Path(sys.argv[1])

for pct in (15, 30, 50):
    path = root / f"test_mask{pct}.npz"

    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["x"])
        y = np.asarray(z["y"])
        m = np.asarray(z["synthetic_mask"], dtype=bool)

        assert x.shape == (5000, 16384)
        assert y.shape == x.shape
        assert m.shape == x.shape

        assert np.all(x[m] == 0.0)
        assert np.all(y[m] > 0.0)
        assert np.array_equal(x[~m], y[~m])

manifest = json.loads(
    (root / "test_panels_manifest.json").read_text()
)

assert manifest["status"] == "PASS"
assert manifest["model_inference_performed"] is False
assert manifest["threshold_retuning_performed"] is False

print("TEST_PANEL_SCIENTIFIC_QC=PASS")
PYQC

mv "${TMP}" "${TARGET}"

echo "TEST_PANEL_PROMOTION=PASS"
echo "TARGET=${TARGET}"

sha256sum     "${TARGET}/test_mask15.npz"     "${TARGET}/test_mask30.npz"     "${TARGET}/test_mask50.npz"     "${TARGET}/test_panels_manifest.json"

echo "P3_INTERNAL_TEST_PANEL_JOB=PASS"
echo "MODEL_INFERENCE_PERFORMED=false"
echo "BARON_ZHENG_USED=false"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
