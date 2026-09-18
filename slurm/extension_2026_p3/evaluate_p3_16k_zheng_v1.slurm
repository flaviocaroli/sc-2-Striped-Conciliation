#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3159436/sc2/logs/p3_11_zheng_normalized_v1/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_11_zheng_normalized_v1/%x_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2_mamba/bin/python
PREFIX=/home/3159436/.conda/envs/sc2_mamba
OVERLAY="${SC2}/overlays/mamba22_git_torch25_cu124"

PANELS="${SC2}/data/external_benchmarks/zheng68k_g16384_v1"
RECEIPT="${REPO}/configs/extension_2026_p3/p3_16k_zheng_panels_receipt_v1.json"

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/p3_11_zheng_normalized_v1"

: "${SEED:?}"
: "${CONFIG:?}"
: "${CHECKPOINT:?}"
: "${EXPECTED_CKPT_SHA:?}"
: "${EXPECTED_JOB_HEAD:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

readarray -t INFO < <(
"${PY}" - "${RECEIPT}" <<'PYINFO'
import json
import sys

x = json.load(open(sys.argv[1]))

print(x["panel_sha256"]["mask15"])
print(x["panel_sha256"]["mask30"])
print(x["panel_sha256"]["mask50"])
print(x["available_genes"])
PYINFO
)

test "${#INFO[@]}" -eq 4

H15="${INFO[0]}"
H30="${INFO[1]}"
H50="${INFO[2]}"
AVAILABLE="${INFO[3]}"

test "$(sha256sum "${PANELS}/mask15.npz" | awk '{print $1}')" = "${H15}"
test "$(sha256sum "${PANELS}/mask30.npz" | awk '{print $1}')" = "${H30}"
test "$(sha256sum "${PANELS}/mask50.npz" | awk '{print $1}')" = "${H50}"

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
print("P3_11_ZHENG_RUNTIME_GATE=PASS")
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
        "${THRESHOLD}" \
        "${AVAILABLE}" <<'PYQC'
import json
import math
import sys

s = json.load(open(sys.argv[1]))
threshold = float(sys.argv[2])
available = int(sys.argv[3])

assert s["threshold_source"] == "frozen_external_threshold"

assert math.isclose(
    float(s["threshold"]),
    threshold,
    rel_tol=0.0,
    abs_tol=1e-12,
)

assert int(s["n_available_genes"]) == available

assert math.isclose(
    float(s["available_gene_fraction"]),
    available / 16384,
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

print("ZHENG_SUMMARY_QC=PASS")
PYQC

    echo "P3_11_ZHENG_PANEL_PASS=mask${RATE}"

done

echo "P3_11_ZHENG_JOB=PASS"
echo "THRESHOLD_RETUNING=false"
