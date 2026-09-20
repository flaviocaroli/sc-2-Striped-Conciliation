#!/bin/bash
#SBATCH --job-name=p3-thin-vinf
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-24
#SBATCH --output=/home/3159436/sc2/logs/p3_count_thinning_validation_inference_v1/validation_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/p3_count_thinning_validation_inference_v1/validation_%A_%a.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2_mamba/bin/python
PREFIX=/home/3159436/.conda/envs/sc2_mamba

OVERLAY="${SC2}/overlays/mamba22_git_torch25_cu124"

PROTOCOL="${REPO}/configs/extension_2026_p3/p3_16k_count_thinning_validation_inference_v1.json"
RUNNER="${REPO}/scripts/eval/run_p3_16k_count_thinning_validation_task_v1.py"

: "${EXPECTED_JOB_HEAD:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"
: "${EXPECTED_RUNNER_SHA:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" \
    = "${EXPECTED_PROTOCOL_SHA}"

test "$(sha256sum "${RUNNER}" | awk '{print $1}')" \
    = "${EXPECTED_RUNNER_SHA}"

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export PYTHONPATH="${OVERLAY}:${REPO}/src:${REPO}"

SITE=$(
    "${PY}" - <<'PY'
import site
x = site.getsitepackages()
assert x
print(x[0])
PY
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

"${PY}" - <<'PY'
import ctypes
import torch

ctypes.CDLL(
    "libcudart.so.12",
    mode=ctypes.RTLD_GLOBAL,
)

ctypes.CDLL(
    "libnvrtc.so.12",
    mode=ctypes.RTLD_GLOBAL,
)

import causal_conv1d
import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1

gpu = torch.cuda.get_device_name(0)

assert "H200" in gpu.upper(), gpu
assert torch.__version__ == "2.5.0+cu124"
assert mamba_ssm.__version__ == "2.2.4"
assert causal_conv1d.__version__ == "1.4.0"

print("P3_THINNING_GPU_RUNTIME=PASS")
print("GPU=" + gpu)
PY

export SC2_P3_THIN_PROTOCOL_SHA="${EXPECTED_PROTOCOL_SHA}"

echo "============================================================"
echo "P3 16K COUNT-THINNING VALIDATION"
echo "============================================================"

echo "ARRAY_JOB=${SLURM_ARRAY_JOB_ID}"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "HOST=$(hostname)"
echo "HEAD=${EXPECTED_JOB_HEAD}"

"${PY}" -u "${RUNNER}" \
    --protocol "${PROTOCOL}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"

echo "P3_THINNING_GPU_TASK=PASS"
