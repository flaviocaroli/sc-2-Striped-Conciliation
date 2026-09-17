#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3159436/sc2/logs/p3_9_internal_test_v1/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_9_internal_test_v1/%x_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2_mamba/bin/python
PREFIX=/home/3159436/.conda/envs/sc2_mamba
OVERLAY="${SC2}/overlays/mamba22_git_torch25_cu124"

PANELS="${SC2}/data/census_curated_pretrain_pilot250k_g16384/benchmarks/test_masking_v1"
OUTROOT="${SC2}/outputs/paper_extension_2026_p3/p3_9_internal_test_v1"

: "${SEED:?SEED missing}"
: "${CONFIG:?CONFIG missing}"
: "${CHECKPOINT:?CHECKPOINT missing}"
: "${EXPECTED_CKPT_SHA:?EXPECTED_CKPT_SHA missing}"
: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"

case "${SEED}" in
    20260728|20260729|20260730|20260731|20260801) ;;
    *)
        echo "STOP: invalid seed ${SEED}" >&2
        exit 1
        ;;
esac

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test -s "${CONFIG}"
test -s "${CHECKPOINT}"

ACTUAL_SHA=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
test "${ACTUAL_SHA}" = "${EXPECTED_CKPT_SHA}"

# Test panels are frozen exact inputs.
test "$(
    sha256sum "${PANELS}/test_mask15.npz" |
    awk '{print $1}'
)" = "84a5bf413c0bcdf192fb18f43326871ff3fb4c9f700c31a9ce0e321f9c8eb33c"

test "$(
    sha256sum "${PANELS}/test_mask30.npz" |
    awk '{print $1}'
)" = "fc804ea6496660c950bd3ada3930e8879df8a96c59883adba06312cb63e60533"

test "$(
    sha256sum "${PANELS}/test_mask50.npz" |
    awk '{print $1}'
)" = "5392e7fc9b24cb5a1a017ab5e537dd9017d6ee36b6ec5a0ae9011b5d38c47628"

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${OVERLAY}:${REPO}/src"

SITE=$(
    "${PY}" - <<'PYENV'
import site
paths = site.getsitepackages()
assert paths
print(paths[0])
PYENV
)

TORCH_LIB="${SITE}/torch/lib"
NVIDIA_ROOT="${SITE}/nvidia"

test -d "${TORCH_LIB}"
test -d "${NVIDIA_ROOT}"

NVIDIA_LIB_PATHS=$(
    find "${NVIDIA_ROOT}" \
        -type d \
        -name lib \
        -printf '%p\n' |
    sort |
    paste -sd: -
)

test -n "${NVIDIA_LIB_PATHS}"

export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${TORCH_LIB}:${PREFIX}/lib"

echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "SEED=${SEED}"
echo "CONFIG=${CONFIG}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "CHECKPOINT_SHA256=${ACTUAL_SHA}"
echo "HEAD=$(git rev-parse HEAD)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${PY}" - <<'PYGATE'
import ctypes

ctypes.CDLL("libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("libnvrtc.so.12", mode=ctypes.RTLD_GLOBAL)

import torch
import mamba_ssm
import causal_conv1d
import causal_conv1d_cuda
import selective_scan_cuda
import sc2.eval.evaluate_continuous_repair

assert torch.cuda.is_available()

gpu = torch.cuda.get_device_name(0)

assert "H200" in gpu.upper(), gpu
assert torch.__version__ == "2.5.0+cu124"
assert mamba_ssm.__version__ == "2.2.4"
assert causal_conv1d.__version__ == "1.4.0"

print("GPU =", gpu)
print("P3_9_H200_RUNTIME_GATE=PASS")
PYGATE

RUNROOT="${OUTROOT}/s${SEED}"

test ! -e "${RUNROOT}" || {
    echo "STOP: output already exists: ${RUNROOT}" >&2
    exit 1
}

mkdir -p "${RUNROOT}"

for RATE in 15 30 50; do

    case "${RATE}" in
        15) THRESHOLD=0.60 ;;
        30) THRESHOLD=0.57 ;;
        50) THRESHOLD=0.52 ;;
    esac

    BENCHMARK="${PANELS}/test_mask${RATE}.npz"
    DEST="${RUNROOT}/mask${RATE}"

    echo
    echo "============================================================"
    echo "INTERNAL TEST SEED=${SEED} MASK=${RATE} THRESHOLD=${THRESHOLD}"
    echo "============================================================"

    "${PY}" -u \
        -m sc2.eval.evaluate_continuous_repair \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --benchmark "${BENCHMARK}" \
        --output-dir "${DEST}" \
        --threshold "${THRESHOLD}"

    test -s "${DEST}/summary.json"
    test -s "${DEST}/summary.csv"
    test -s "${DEST}/threshold_sweep.csv"
    test -s "${DEST}/risk_coverage.csv"

    "${PY}" - \
        "${DEST}/summary.json" \
        "${CHECKPOINT}" \
        "${BENCHMARK}" \
        "${THRESHOLD}" <<'PYQC'
import json
import math
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
checkpoint = str(Path(sys.argv[2]))
benchmark = str(Path(sys.argv[3]))
expected_threshold = float(sys.argv[4])

assert summary["checkpoint"] == checkpoint
assert summary["benchmark"] == benchmark

assert int(summary["n_available_genes"]) == 16384
assert float(summary["available_gene_fraction"]) == 1.0

assert summary["threshold_source"] == "frozen_external_threshold"

assert math.isclose(
    float(summary["threshold"]),
    expected_threshold,
    rel_tol=0.0,
    abs_tol=1e-12,
)

assert float(summary["observed_nonzero_changed_fraction"]) == 0.0
assert float(summary["observed_nonzero_mse"]) == 0.0
assert float(summary["observed_nonzero_mae"]) == 0.0
assert float(summary["observed_nonzero_max_abs_error"]) == 0.0

for key in (
    "masked_mse",
    "masked_mae",
    "sample_spearman",
    "gene_spearman",
    "prediction_sd",
    "target_sd",
    "sd_ratio",
    "gate_auroc",
    "gate_auprc",
    "gate_brier",
    "gate_ece",
    "threshold_recall",
    "threshold_precision",
    "threshold_true_zero_fill",
):
    value = float(summary[key])
    assert math.isfinite(value), (key, value)

# IMPORTANT:
# true-zero fill on test is an observed outcome.
# It is NOT used to reject/retune the frozen threshold.
print("TEST_SUMMARY_QC=PASS")
PYQC

    echo "P3_9_TEST_PANEL_PASS=mask${RATE}"

done

echo
echo "P3_9_INTERNAL_TEST_JOB=PASS"
echo "THRESHOLD_RETUNING=false"
echo "BARON_ZHENG_USED=false"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
