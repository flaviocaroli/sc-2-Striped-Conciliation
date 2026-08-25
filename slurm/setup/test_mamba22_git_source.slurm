#!/bin/bash
#SBATCH --job-name=sc2-mamba22-git
#SBATCH --partition=medium_gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2_mamba

module load cuda/12.8

export CUDA_HOME=/software/cuda/12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS=2

OVERLAY=/home/3159436/sc2/overlays/mamba22_git_torch25_cu124

BUILD_ROOT="${TMPDIR:-/tmp}/sc2-mamba-src-${SLURM_JOB_ID}"

rm -rf "${BUILD_ROOT}"

if [[ -e "${OVERLAY}" ]]; then
    STAMP=$(date +%Y%m%d_%H%M%S)

    mkdir -p \
      /home/3159436/sc2/archive/mamba_overlays

    mv \
      "${OVERLAY}" \
      "/home/3159436/sc2/archive/mamba_overlays/mamba22_git_torch25_cu124_${STAMP}"
fi

mkdir -p \
  "${OVERLAY}" \
  "${BUILD_ROOT}"

export PYTHONPATH="${OVERLAY}:${PYTHONPATH:-}"

cleanup() {
    rm -rf "${BUILD_ROOT}"
}

trap cleanup EXIT

echo "========================================"
echo "SC2 OFFICIAL MAMBA GIT-SOURCE TEST"
echo "job=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "overlay=${OVERLAY}"
echo "build_root=${BUILD_ROOT}"
echo "========================================"

echo
echo "===== BASE ENVIRONMENT ====="

echo "python=$(which python)"
echo "git=$(command -v git)"
echo "nvcc=$(command -v nvcc)"
echo "CUDA_HOME=${CUDA_HOME}"

python --version
nvcc --version
gcc --version | head -1

python - <<'PY'
import torch

print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())
print("device =", torch.cuda.get_device_name(0))
print(
    "cxx11_abi =",
    torch._C._GLIBCXX_USE_CXX11_ABI,
)

assert torch.__version__.startswith("2.5.0")
assert torch.version.cuda == "12.4"
assert torch.cuda.is_available()

print("BASE_H200_ENV=PASS")
PY

echo
echo "===== RUNTIME PYTHON DEPENDENCIES ====="

python -m pip install \
  --target "${OVERLAY}" \
  --upgrade \
  --no-cache-dir \
  "einops==0.8.1" \
  "transformers==4.46.3"

echo
echo "===== CLONE CAUSAL-CONV1D v1.4.0 ====="

git clone \
  --depth 1 \
  --branch v1.4.0 \
  --single-branch \
  https://github.com/Dao-AILab/causal-conv1d.git \
  "${BUILD_ROOT}/causal-conv1d"

echo
echo "===== VERIFY CAUSAL SOURCE ====="

test -f \
  "${BUILD_ROOT}/causal-conv1d/csrc/causal_conv1d.cpp"

test -f \
  "${BUILD_ROOT}/causal-conv1d/csrc/causal_conv1d_fwd.cu"

test -f \
  "${BUILD_ROOT}/causal-conv1d/csrc/causal_conv1d_bwd.cu"

test -f \
  "${BUILD_ROOT}/causal-conv1d/csrc/causal_conv1d_update.cu"

find \
  "${BUILD_ROOT}/causal-conv1d/csrc" \
  -maxdepth 1 \
  -type f \
  -printf '%f\n' \
  | sort

echo "CAUSAL_SOURCE_TREE=PASS"

echo
echo "===== BUILD CAUSAL-CONV1D FROM GIT ====="

CAUSAL_CONV1D_FORCE_BUILD=TRUE \
python -m pip install \
  --target "${OVERLAY}" \
  --no-build-isolation \
  --no-deps \
  --no-cache-dir \
  "${BUILD_ROOT}/causal-conv1d"

echo
echo "===== VERIFY CAUSAL-CONV1D ====="

python - <<'PY'
import causal_conv1d
import causal_conv1d_cuda

print(
    "causal_conv1d =",
    causal_conv1d.__version__,
)

print(
    "causal_conv1d_cuda =",
    causal_conv1d_cuda.__file__,
)

print("CAUSAL_CONV1D_CUDA=PASS")
PY

echo
echo "===== CLONE MAMBA v2.2.4 ====="

git clone \
  --depth 1 \
  --branch v2.2.4 \
  --single-branch \
  https://github.com/state-spaces/mamba.git \
  "${BUILD_ROOT}/mamba"

echo
echo "===== VERIFY MAMBA SOURCE ====="

test -f \
  "${BUILD_ROOT}/mamba/csrc/selective_scan/selective_scan.cpp"

test -f \
  "${BUILD_ROOT}/mamba/csrc/selective_scan/selective_scan_fwd_fp32.cu"

test -f \
  "${BUILD_ROOT}/mamba/csrc/selective_scan/selective_scan_bwd_fp32_real.cu"

echo "MAMBA_SOURCE_TREE=PASS"

echo
echo "===== BUILD MAMBA FROM GIT ====="

MAMBA_FORCE_BUILD=TRUE \
python -m pip install \
  --target "${OVERLAY}" \
  --no-build-isolation \
  --no-deps \
  --no-cache-dir \
  "${BUILD_ROOT}/mamba"

echo
echo "===== VERIFY FUSED IMPORTS ====="

python - <<'PY'
import importlib

modules = [
    "causal_conv1d",
    "causal_conv1d_cuda",
    "mamba_ssm",
    "selective_scan_cuda",
]

for name in modules:
    module = importlib.import_module(name)

    print(
        f"{name:25s}",
        "PASS",
        getattr(module, "__version__", ""),
    )

print("FUSED_IMPORTS=PASS")
PY

echo
echo "===== EXTENSION LOCATIONS ====="

python - <<'PY'
import causal_conv1d_cuda
import selective_scan_cuda

print(
    "causal_conv1d_cuda =",
    causal_conv1d_cuda.__file__,
)

print(
    "selective_scan_cuda =",
    selective_scan_cuda.__file__,
)
PY

echo
echo "===== REAL 4096-POSITION H200 TEST ====="

python - <<'PY'
import json
import time

import torch
from mamba_ssm import Mamba

torch.manual_seed(20260728)

device = torch.device("cuda")

model = Mamba(
    d_model=128,
    d_state=4,
    d_conv=4,
    expand=1,
    use_fast_path=True,
).to(device)

x = torch.randn(
    2,
    4096,
    128,
    device=device,
    dtype=torch.float32,
    requires_grad=True,
)

# Warm-up
with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    warm = model(x)
    warm_loss = warm.float().square().mean()

warm_loss.backward()

model.zero_grad(set_to_none=True)
x.grad = None

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

started = time.perf_counter()

with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    y = model(x)
    loss = y.float().square().mean()

loss.backward()

torch.cuda.synchronize()

elapsed = time.perf_counter() - started

result = {
    "shape": list(y.shape),
    "seconds_forward_backward": elapsed,
    "loss": float(loss.detach()),
    "peak_gpu_memory_gb": (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    ),
    "input_gradient_finite": bool(
        torch.isfinite(x.grad).all()
    ),
}

print(
    json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )
)

assert y.shape == x.shape
assert result["input_gradient_finite"]

print("MAMBA22_H200_FUSED=PASS")
PY

echo
echo "===== ENVIRONMENT RECORD ====="

mkdir -p \
  /home/3159436/sc2/outputs/environment

{
    echo "date=$(date --iso-8601=seconds)"
    echo "base_environment=sc2_mamba"
    echo "torch=2.5.0+cu124"
    echo "cuda_compiler=12.8"
    echo "causal_conv1d=1.4.0"
    echo "mamba_ssm=2.2.4"
    echo "overlay=${OVERLAY}"
} > \
  /home/3159436/sc2/outputs/environment/mamba22_git_h200.txt

echo
echo "OFFICIAL_MAMBA_FEASIBILITY=PASS"
