#!/bin/bash
#SBATCH --job-name=sc2-p3-zheng16k-mat
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_11_zheng_materialization_v1/zheng16k_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_11_zheng_materialization_v1/zheng16k_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"
DATA="${SC2}/data"

PY=/home/3159436/.conda/envs/sc2-data/bin/python

P3="${DATA}/census_curated_pretrain_pilot250k_g16384"
P2="${DATA}/external_benchmarks/zheng68k_v1"

BUILDER="${REPO}/scripts/data/prepare_external_repair_benchmarks_p3_16k.py"
PROTOCOL="${REPO}/configs/extension_2026_p3/p3_16k_zheng_masking_protocol_v1.json"

TARGET="${DATA}/external_benchmarks/zheng68k_g16384_v1"
TMP="${TARGET}.tmp"

: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

unset PYTHONPATH
unset LD_LIBRARY_PATH
unset PYTHONHOME

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

readarray -t INFO < <(
"${PY}" - "${PROTOCOL}" <<'PYINFO'
import json
import sys

x = json.load(open(sys.argv[1]))
print(x["raw_root"])
print(x["raw_archive"])
PYINFO
)

test "${#INFO[@]}" -eq 2

RAW_ROOT="${INFO[0]}"
RAW_ARCHIVE="${INFO[1]}"

test -d "${RAW_ROOT}"
test -s "${RAW_ARCHIVE}"

test -s "${RAW_ROOT}/extracted/filtered_matrices_mex/hg19/genes.tsv"
test -s "${RAW_ROOT}/extracted/filtered_matrices_mex/hg19/barcodes.tsv"
test -s "${RAW_ROOT}/extracted/filtered_matrices_mex/hg19/matrix.mtx"

test ! -e "${TARGET}"
test ! -e "${TMP}"

echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HEAD=$(git rev-parse HEAD)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${PY}" "${BUILDER}" \
    --dataset zheng68k \
    --raw-root "${RAW_ROOT}" \
    --raw-archive "${RAW_ARCHIVE}" \
    --vocabulary "${P3}/gene_vocabulary.parquet" \
    --gene-stats "${P3}/train_gene_stats.parquet" \
    --output-dir "${TMP}" \
    --n-cells 5000 \
    --seed 20260729

for RATE in 15 30 50; do
    test -s "${TMP}/mask${RATE}.npz"
done

test -s "${TMP}/selected_cells.csv"
test -s "${TMP}/benchmark_metadata.json"
test -s "${TMP}/benchmark_panels.sha256"

"${PY}" - "${TMP}" "${P2}" <<'PYQC'
from pathlib import Path
import numpy as np
import pandas as pd
import sys

new = Path(sys.argv[1])
old = Path(sys.argv[2])

old_cells = pd.read_csv(old / "selected_cells.csv")
new_cells = pd.read_csv(new / "selected_cells.csv")

assert len(old_cells) == len(new_cells) == 5000
assert list(old_cells.columns) == list(new_cells.columns)
assert old_cells.astype(str).equals(new_cells.astype(str))

targets = []
availability = []

for pct, rate in ((15, 0.15), (30, 0.30), (50, 0.50)):
    with np.load(new / f"mask{pct}.npz", allow_pickle=False) as z:
        x = np.asarray(z["x"])
        y = np.asarray(z["y"])
        m = np.asarray(z["synthetic_mask"], dtype=bool)
        a = np.asarray(z["available_gene_mask"], dtype=bool)

        assert x.shape == (5000, 16384)
        assert y.shape == (5000, 16384)
        assert m.shape == (5000, 16384)
        assert a.shape == (16384,)

        assert 0 < int(a.sum()) <= 16384

        assert np.all(m[:, ~a] == 0)
        assert np.all(x[:, ~a] == 0.0)
        assert np.all(y[:, ~a] == 0.0)

        assert np.all(x[m] == 0.0)
        assert np.all(y[m] > 0.0)
        assert np.array_equal(x[~m], y[~m])

        positive = ((y > 0.0) & a[None, :]).sum(axis=1)

        expected = np.maximum(
            1,
            np.rint(rate * positive).astype(np.int64),
        )

        assert np.array_equal(
            expected,
            m.sum(axis=1).astype(np.int64),
        )

        targets.append(y)
        availability.append(a)

assert np.array_equal(targets[0], targets[1])
assert np.array_equal(targets[0], targets[2])

assert np.array_equal(availability[0], availability[1])
assert np.array_equal(availability[0], availability[2])

print("ZHENG_CELLS=5000")
print("ZHENG_GENES=16384")
print("ZHENG_AVAILABLE_GENES=", int(availability[0].sum()))
print("P2_P3_ZHENG_CELL_IDENTITY_EXACT=true")
print("ZHENG_16K_SCIENTIFIC_QC=PASS")
PYQC

mv "${TMP}" "${TARGET}"

echo
echo "================ ZHENG PANEL HASHES ================"

sha256sum \
    "${TARGET}/mask15.npz" \
    "${TARGET}/mask30.npz" \
    "${TARGET}/mask50.npz" \
    "${TARGET}/selected_cells.csv" \
    "${TARGET}/benchmark_metadata.json" \
    "${TARGET}/benchmark_panels.sha256"

echo
echo "ZHENG_16K_PROMOTION=PASS"
echo "P3_11_ZHENG_MATERIALIZATION_JOB=PASS"
echo "MODEL_INFERENCE_PERFORMED=false"
echo "THRESHOLD_RETUNING=false"
