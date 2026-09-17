#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_8_validation_v1/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_8_validation_v1/%x_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2_mamba/bin/python
PREFIX=/home/3159436/.conda/envs/sc2_mamba
OVERLAY="${SC2}/overlays/mamba22_git_torch25_cu124"

PANEL_ROOT="${SC2}/data/census_curated_pretrain_pilot250k_g16384/benchmarks/validation_masking_v1"
OUTROOT="${SC2}/outputs/paper_extension_2026_p3/p3_8_validation_v1"

: "${FAMILY:?FAMILY missing}"
: "${SEED:?SEED missing}"
: "${CONFIG:?CONFIG missing}"
: "${CHECKPOINT:?CHECKPOINT missing}"
: "${EXPECTED_CKPT_SHA:?EXPECTED_CKPT_SHA missing}"
: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"

case "${FAMILY}" in
    small|large) ;;
    *)
        echo "STOP: invalid FAMILY=${FAMILY}" >&2
        exit 1
        ;;
esac

case "${SEED}" in
    20260728|20260729|20260730|20260731|20260801) ;;
    *)
        echo "STOP: invalid SEED=${SEED}" >&2
        exit 1
        ;;
esac

test -f "${CONFIG}"
test -s "${CHECKPOINT}"

ACTUAL_CKPT_SHA=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')

test "${ACTUAL_CKPT_SHA}" = "${EXPECTED_CKPT_SHA}"

cd "${REPO}"

ACTUAL_HEAD=$(git rev-parse HEAD)
test "${ACTUAL_HEAD}" = "${EXPECTED_JOB_HEAD}"

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
echo "FAMILY=${FAMILY}"
echo "SEED=${SEED}"
echo "CONFIG=${CONFIG}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "CHECKPOINT_SHA256=${ACTUAL_CKPT_SHA}"
echo "HEAD=${ACTUAL_HEAD}"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${PY}" - <<'PYGATE'
import ctypes

ctypes.CDLL(
    "libcudart.so.12",
    mode=ctypes.RTLD_GLOBAL,
)

ctypes.CDLL(
    "libnvrtc.so.12",
    mode=ctypes.RTLD_GLOBAL,
)

import numpy
import pandas
import scipy
import sklearn
import torch
import yaml

import sc2
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
print("NUMPY =", numpy.__version__)
print("PANDAS =", pandas.__version__)
print("SCIPY =", scipy.__version__)
print("SKLEARN =", sklearn.__version__)
print("TORCH =", torch.__version__)
print("MAMBA_SSM =", mamba_ssm.__version__)
print("CAUSAL_CONV1D =", causal_conv1d.__version__)
print("P3_8_H200_RUNTIME_GATE=PASS")
PYGATE

# Exact frozen panel hashes again on the compute node.
test "$(
    sha256sum "${PANEL_ROOT}/validation_mask15.npz" |
    awk '{print $1}'
)" = "285fd6d68216902385c3ecfadc138d9d30ae095cc740ad2fcb58d9c2418730be"

test "$(
    sha256sum "${PANEL_ROOT}/validation_mask30.npz" |
    awk '{print $1}'
)" = "c17b8ed48050dbc805816d0eb4e090c4b6777a2ac034d66a1b0f2c5048e842a4"

test "$(
    sha256sum "${PANEL_ROOT}/validation_mask50.npz" |
    awk '{print $1}'
)" = "0fecceb37296fadd361f8cc293f2d0fd5ad7c65668b19a9d455615cdc0d41f51"

echo "COMPUTE_PANEL_HASH_GATE=PASS"

RUNROOT="${OUTROOT}/${FAMILY}/s${SEED}"

test ! -e "${RUNROOT}" || {
    echo "STOP: output already exists: ${RUNROOT}" >&2
    exit 1
}

mkdir -p "${RUNROOT}"

for RATE in 15 30 50; do

    BENCHMARK="${PANEL_ROOT}/validation_mask${RATE}.npz"
    DEST="${RUNROOT}/mask${RATE}"

    echo
    echo "============================================================"
    echo "VALIDATION FAMILY=${FAMILY} SEED=${SEED} MASK=${RATE}"
    echo "============================================================"

    "${PY}" -u \
        -m sc2.eval.evaluate_continuous_repair \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --benchmark "${BENCHMARK}" \
        --output-dir "${DEST}"

    test -s "${DEST}/summary.json"
    test -s "${DEST}/summary.csv"
    test -s "${DEST}/threshold_sweep.csv"
    test -s "${DEST}/risk_coverage.csv"

    "${PY}" - \
        "${DEST}/summary.json" \
        "${CHECKPOINT}" \
        "${BENCHMARK}" <<'PYQC'
import json
import math
from pathlib import Path
import sys

summary_path = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]))
benchmark = str(Path(sys.argv[3]))

s = json.loads(summary_path.read_text())

assert s["checkpoint"] == checkpoint
assert s["benchmark"] == benchmark
assert int(s["n_available_genes"]) == 16384
assert float(s["available_gene_fraction"]) == 1.0

assert s["threshold_source"] == "selected_on_this_validation_set"

assert float(s["threshold_true_zero_fill"]) <= 0.020000000001

assert float(s["observed_nonzero_changed_fraction"]) == 0.0
assert float(s["observed_nonzero_mse"]) == 0.0
assert float(s["observed_nonzero_mae"]) == 0.0
assert float(s["observed_nonzero_max_abs_error"]) == 0.0

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
    "gate_prevalence",
    "threshold",
    "threshold_recall",
    "threshold_precision",
    "threshold_true_zero_fill",
):
    value = float(s[key])
    assert math.isfinite(value), (key, value)

print("SUMMARY_QC=PASS")
PYQC

    echo "P3_8_PANEL_PASS=mask${RATE}"

done

echo
echo "P3_8_VALIDATION_JOB=PASS"
echo "TEST_BARON_ZHENG_USED=false"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
