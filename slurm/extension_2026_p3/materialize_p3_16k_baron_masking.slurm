#!/bin/bash
#SBATCH --job-name=sc2-p3-baron16k-mat
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_10_baron_materialization_v1/baron16k_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_10_baron_materialization_v1/baron16k_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"
DATA="${SC2}/data"

PY=/home/3159436/.conda/envs/sc2-data/bin/python

P3="${DATA}/census_curated_pretrain_pilot250k_g16384"
P2_BARON="${DATA}/external_benchmarks/baron_pancreas_v1"

VOCAB="${P3}/gene_vocabulary.parquet"
STATS="${P3}/train_gene_stats.parquet"

BUILDER="${REPO}/scripts/data/prepare_external_repair_benchmarks_p3_16k.py"
PROTOCOL="${REPO}/configs/extension_2026_p3/p3_16k_baron_masking_protocol_v1.json"

TARGET="${DATA}/external_benchmarks/baron_pancreas_g16384_v1"
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

readarray -t RAWINFO < <(
"${PY}" - "${PROTOCOL}" <<'PYRAW'
import json
import sys

p = json.load(open(sys.argv[1]))
print(p["raw_root"])
print(p["raw_archive"])
PYRAW
)

test "${#RAWINFO[@]}" -eq 2

RAW_ROOT="${RAWINFO[0]}"
RAW_ARCHIVE="${RAWINFO[1]}"

test -d "${RAW_ROOT}"
test -s "${RAW_ARCHIVE}"

test ! -e "${TARGET}"
test ! -e "${TMP}"

echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HEAD=$(git rev-parse HEAD)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "RAW_ROOT=${RAW_ROOT}"
echo "RAW_ARCHIVE=${RAW_ARCHIVE}"

"${PY}" "${BUILDER}" \
    --dataset baron_pancreas \
    --raw-root "${RAW_ROOT}" \
    --raw-archive "${RAW_ARCHIVE}" \
    --vocabulary "${VOCAB}" \
    --gene-stats "${STATS}" \
    --output-dir "${TMP}" \
    --n-cells 5000 \
    --seed 20260729

for RATE in 15 30 50; do
    test -s "${TMP}/mask${RATE}.npz"
done

test -s "${TMP}/selected_cells.csv"
test -s "${TMP}/benchmark_metadata.json"
test -s "${TMP}/benchmark_panels.sha256"

"${PY}" - \
    "${TMP}" \
    "${P2_BARON}" <<'PYQC'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

new = Path(sys.argv[1])
old = Path(sys.argv[2])

# Exact same external cell selection as frozen 4K benchmark.
a = pd.read_csv(old / "selected_cells.csv")
b = pd.read_csv(new / "selected_cells.csv")

assert list(a.columns) == list(b.columns)
assert len(a) == len(b) == 5000
assert a.astype(str).equals(b.astype(str))

targets = []
availability = []

for pct, rate in ((15, 0.15), (30, 0.30), (50, 0.50)):
    path = new / f"mask{pct}.npz"

    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["x"])
        y = np.asarray(z["y"])
        mask = np.asarray(z["synthetic_mask"], dtype=bool)
        available = np.asarray(
            z["available_gene_mask"],
            dtype=bool,
        )

        assert x.shape == (5000, 16384)
        assert y.shape == (5000, 16384)
        assert mask.shape == (5000, 16384)
        assert available.shape == (16384,)

        assert 0 < int(available.sum()) <= 16384

        assert int(np.asarray(z["seed"]).item()) == 20260729

        assert np.all(mask[:, ~available] == 0)
        assert np.all(y[:, ~available] == 0.0)
        assert np.all(x[:, ~available] == 0.0)

        assert np.all(y[mask] > 0.0)
        assert np.all(x[mask] == 0.0)
        assert np.array_equal(x[~mask], y[~mask])

        npos = ((y > 0.0) & available[None, :]).sum(axis=1)

        expected = np.maximum(
            1,
            np.rint(rate * npos).astype(np.int64),
        )

        actual = mask.sum(axis=1).astype(np.int64)

        assert np.array_equal(expected, actual)

        targets.append(y)
        availability.append(available)

# Same clean target and availability for 15/30/50 panels.
assert np.array_equal(targets[0], targets[1])
assert np.array_equal(targets[0], targets[2])
assert np.array_equal(availability[0], availability[1])
assert np.array_equal(availability[0], availability[2])

meta = json.loads(
    (new / "benchmark_metadata.json").read_text()
)

assert int(meta["n_genes"]) == 16384
assert int(meta["seed"]) == 20260729

print("BARON_CELLS=5000")
print("BARON_GENES=16384")
print("BARON_AVAILABLE_GENES=", int(availability[0].sum()))
print("P2_P3_BARON_CELL_IDENTITY_EXACT=true")
print("BARON_16K_SCIENTIFIC_QC=PASS")
PYQC

mv "${TMP}" "${TARGET}"

echo
echo "================ FROZEN PANEL HASHES ================"

sha256sum \
    "${TARGET}/mask15.npz" \
    "${TARGET}/mask30.npz" \
    "${TARGET}/mask50.npz" \
    "${TARGET}/selected_cells.csv" \
    "${TARGET}/benchmark_metadata.json" \
    "${TARGET}/benchmark_panels.sha256"

echo
echo "BARON_16K_PROMOTION=PASS"
echo "P3_10_BARON_MATERIALIZATION_JOB=PASS"
echo "MODEL_INFERENCE_PERFORMED=false"
echo "THRESHOLD_RETUNING=false"
echo "ZHENG_USED=false"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
