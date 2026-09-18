#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3159436/sc2/logs/p3_10_baron_normalized_v1/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_10_baron_normalized_v1/%x_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2_mamba/bin/python
PREFIX=/home/3159436/.conda/envs/sc2_mamba
OVERLAY="${SC2}/overlays/mamba22_git_torch25_cu124"

PANELS="${SC2}/data/external_benchmarks/baron_pancreas_g16384_v1"
OUTROOT="${SC2}/outputs/paper_extension_2026_p3/p3_10_baron_normalized_v1"

: "${SEED:?}"
: "${CONFIG:?}"
: "${CHECKPOINT:?}"
: "${EXPECTED_CKPT_SHA:?}"
: "${EXPECTED_JOB_HEAD:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test "$(sha256sum "${PANELS}/mask15.npz" | awk '{print $1}')" = \
"8c90c598f5732e081cf87a69ccdbd5763ad2fb7bb207509b614d4a2da8655199"

test "$(sha256sum "${PANELS}/mask30.npz" | awk '{print $1}')" = \
"26c3959a2a557ad8e19aa56a6f0217998c8fa892f234e43429e42254c9a4e7f0"

test "$(sha256sum "${PANELS}/mask50.npz" | awk '{print $1}')" = \
"96a33461cece231780d62bc0967c293eb50286df68aefcea4112394fe80031da"

test "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" = \
"${EXPECTED_CKPT_SHA}"

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${OVERLAY}:${REPO}/src"

SITE=$(
    "${PY}" - <<'PYENV'
import site
print(site.getsitepackages()[0])
PYENV
)

TORCH_LIB="${SITE}/torch/lib"
NVIDIA_ROOT="${SITE}/nvidia"

NVIDIA_LIB_PATHS=$(
    find "${NVIDIA_ROOT}" \
        -type d \
        -name lib \
        -printf '%p\n' |
    sort |
    paste -sd: -
)

export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${TORCH_LIB}:${PREFIX}/lib"

"${PY}" - <<'PYGATE'
import ctypes
ctypes.CDLL("libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("libnvrtc.so.12", mode=ctypes.RTLD_GLOBAL)

import torch
import mamba_ssm
import causal_conv1d
import causal_conv1d_cuda
import selective_scan_cuda

assert torch.cuda.is_available()
assert "H200" in torch.cuda.get_device_name(0).upper()

print("GPU =", torch.cuda.get_device_name(0))
print("P3_10_BARON_RUNTIME_GATE=PASS")
PYGATE

RUNROOT="${OUTROOT}/s${SEED}"

test ! -e "${RUNROOT}"
mkdir -p "${RUNROOT}"

for RATE in 15 30 50; do

    case "${RATE}" in
        15) THRESHOLD=0.60 ;;
        30) THRESHOLD=0.57 ;;
        50) THRESHOLD=0.52 ;;
    esac

    DEST="${RUNROOT}/mask${RATE}"

    "${PY}" -u \
        -m sc2.eval.evaluate_continuous_repair \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --benchmark "${PANELS}/mask${RATE}.npz" \
        --output-dir "${DEST}" \
        --threshold "${THRESHOLD}"

    test -s "${DEST}/summary.json"

    "${PY}" - \
        "${DEST}/summary.json" \
        "${THRESHOLD}" <<'PYQC'
import json
import math
import sys

s = json.load(open(sys.argv[1]))
threshold = float(sys.argv[2])

assert s["threshold_source"] == "frozen_external_threshold"

assert math.isclose(
    float(s["threshold"]),
    threshold,
    rel_tol=0.0,
    abs_tol=1e-12,
)

assert int(s["n_available_genes"]) == 13762

assert math.isclose(
    float(s["available_gene_fraction"]),
    13762 / 16384,
    rel_tol=0.0,
    abs_tol=1e-12,
)

assert float(s["observed_nonzero_changed_fraction"]) == 0.0

for key in (
    "masked_mse",
    "masked_mae",
    "sample_spearman",
    "gene_spearman",
    "sd_ratio",
    "gate_auroc",
    "gate_auprc",
    "threshold_recall",
    "threshold_precision",
    "threshold_true_zero_fill",
):
    assert math.isfinite(float(s[key])), key

print("BARON_SUMMARY_QC=PASS")
PYQC

    echo "P3_10_BARON_PANEL_PASS=mask${RATE}"

done

echo "P3_10_BARON_JOB=PASS"
echo "THRESHOLD_RETUNING=false"
echo "ZHENG_USED=false"
