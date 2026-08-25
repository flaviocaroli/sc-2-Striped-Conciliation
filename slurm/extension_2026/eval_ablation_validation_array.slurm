#!/bin/bash
#SBATCH --job-name=sc2-abl5-val
#SBATCH --gres=gpu:1
#SBATCH --output=/home/3159436/sc2/logs/paper_extension_2026/sc2-abl5-val_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/paper_extension_2026/sc2-abl5-val_%A_%a.err

set -Eeuo pipefail

: "${SC2_EXPECTED_COMMIT:?SC2_EXPECTED_COMMIT is required}"
: "${SC2_ABLATION_VALIDATION_PROTOCOL_SHA:?SC2_ABLATION_VALIDATION_PROTOCOL_SHA is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

REPO=/home/3159436/sc2/code
PROTOCOL="${REPO}/configs/extension_2026/ablation_validation_protocol_v1.json"
OVERLAY=/home/3159436/sc2/overlays/mamba22_git_torch25_cu124

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2_mamba

cd "${REPO}"

CURRENT_COMMIT="$(git rev-parse HEAD)"

if [[ "${CURRENT_COMMIT}" != "${SC2_EXPECTED_COMMIT}" ]]; then
    echo "ERROR: Git commit mismatch."
    echo "CURRENT=${CURRENT_COMMIT}"
    echo "EXPECTED=${SC2_EXPECTED_COMMIT}"
    exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: repository is dirty."
    git status --short
    exit 2
fi

ACTUAL_PROTOCOL_SHA="$(
    sha256sum "${PROTOCOL}" | awk '{print $1}'
)"

if [[ "${ACTUAL_PROTOCOL_SHA}" != "${SC2_ABLATION_VALIDATION_PROTOCOL_SHA}" ]]; then
    echo "ERROR: Phase-1B protocol SHA mismatch."
    echo "ACTUAL=${ACTUAL_PROTOCOL_SHA}"
    echo "EXPECTED=${SC2_ABLATION_VALIDATION_PROTOCOL_SHA}"
    exit 2
fi

export PYTHONPATH="${OVERLAY}:${REPO}/src:${REPO}:${PYTHONPATH:-}"

TORCH_LIB="$(
python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"

NVIDIA_ROOT="$(
python - <<'PY'
from pathlib import Path
import site

for root in site.getsitepackages():
    p = Path(root) / "nvidia"
    if p.is_dir():
        print(p)
        break
PY
)"

NVIDIA_LIB_PATHS="$(
    find "${NVIDIA_ROOT}" \
        -type d \
        -name lib \
        -printf '%p\n' \
    | sort \
    | paste -sd: -
)"

export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import ctypes
import torch

import causal_conv1d
import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda

ctypes.CDLL("libcudart.so.12")
ctypes.CDLL("libnvrtc.so.12")

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable inside H200 job."
    )

if torch.cuda.device_count() < 1:
    raise SystemExit(
        "No CUDA device visible."
    )

print("BARON_LODO_GPU_RUNTIME=PASS")
print("TORCH=" + str(torch.__version__))
print("TORCH_CUDA=" + str(torch.version.cuda))
print("GPU=" + torch.cuda.get_device_name(0))
print("CUDA_VISIBLE_DEVICES=" + str(
    __import__("os").environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
))
PY

echo "========================================"
echo "SC2 THINNING VALIDATION INFERENCE"
echo "job=${SLURM_JOB_ID}"
echo "array_job=${SLURM_ARRAY_JOB_ID:-none}"
echo "task=${SLURM_ARRAY_TASK_ID}"
echo "host=$(hostname)"
echo "commit=${CURRENT_COMMIT}"
echo "protocol=${PROTOCOL}"
echo "protocol_sha=${ACTUAL_PROTOCOL_SHA}"
echo "========================================"

python -u \
    scripts/eval/run_ablation_validation_task.py \
    --protocol "${PROTOCOL}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --output-root "/home/3159436/sc2/outputs/paper_extension_2026/ablation5_v1/validation_eval_v1"
