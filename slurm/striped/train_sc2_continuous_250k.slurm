#!/bin/bash
#SBATCH --job-name=sc2-cont-250k
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SC2_ENV_NAME:-sc2}"

REPO=/home/3159436/sc2/code
cd "${REPO}"

# Optional package overlay. For the official Mamba runtime this points to the
# isolated mamba_ssm / causal_conv1d installation.
if [[ -n "${SC2_PYTHON_OVERLAY:-}" ]]; then
  export PYTHONPATH="${SC2_PYTHON_OVERLAY}:${REPO}/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
fi

# Official fused CUDA extensions need more than libcudart.  In particular,
# cuDNN SDPA may require NVRTC at runtime, so expose every NVIDIA wheel lib/
# directory from the active Conda environment.
if [[ "${SC2_ENABLE_FUSED_RUNTIME:-0}" == "1" ]]; then
  TORCH_LIB=$(
    python - <<'PYTORCHLIB'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PYTORCHLIB
  )

  NVIDIA_ROOT=$(
    python - <<'PYNVIDIA'
from pathlib import Path
import site

for root in site.getsitepackages():
    candidate = Path(root) / "nvidia"
    if candidate.is_dir():
        print(candidate)
        break
PYNVIDIA
  )

  if [[ -z "${NVIDIA_ROOT}" ]]; then
    echo "ERROR: NVIDIA Python package root not found"
    exit 1
  fi

  NVIDIA_LIB_PATHS=$(
    find "${NVIDIA_ROOT}" \
      -type d \
      -name lib \
      -printf '%p\n' \
    | sort \
    | paste -sd: -
  )

  if [[ -z "${NVIDIA_LIB_PATHS}" ]]; then
    echo "ERROR: no NVIDIA runtime lib directories found"
    exit 1
  fi

  export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

  echo "torch_lib=${TORCH_LIB}"
  echo "nvidia_root=${NVIDIA_ROOT}"
  echo "nvidia_runtime_libs=${NVIDIA_LIB_PATHS}"

  python - <<'PYRUNTIME'
import ctypes

ctypes.CDLL("libcudart.so.12")
ctypes.CDLL("libnvrtc.so.12")

print("cuda_runtime_libraries=PASS")
PYRUNTIME
fi

# Fail before training if this job explicitly requires the fused Mamba stack.
if [[ "${SC2_REQUIRE_FUSED_IMPORTS:-0}" == "1" ]]; then
  python - <<'PYFUSED'
import causal_conv1d
import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda

print("causal_conv1d =", causal_conv1d.__version__)
print("mamba_ssm =", mamba_ssm.__version__)
print("FUSED_IMPORTS=PASS")
PYFUSED
fi

DATA=/home/3159436/sc2/data/census_curated_pretrain_pilot250k
EXPECTED_MANIFEST_SHA=2e5adfb5558113ad570651e57cabee78362ae70ac27c96db7e353eea51055b85

ACTUAL_MANIFEST_SHA=$(sha256sum "${DATA}/shards.parquet" | awk '{print $1}')

if [[ "${ACTUAL_MANIFEST_SHA}" != "${EXPECTED_MANIFEST_SHA}" ]]; then
  echo "manifest_hash_mismatch"
  echo "expected=${EXPECTED_MANIFEST_SHA}"
  echo "actual=${ACTUAL_MANIFEST_SHA}"
  exit 1
fi

echo "manifest_hash=PASS"

python scripts/data/validate_shards.py \
  --manifest "${DATA}/shards.parquet"

TRAIN_CONFIG=${SC2_TRAIN_CONFIG:-configs/train/striped/sc2_continuous_250k.yaml}

echo "========================================"
echo "SC2 TRAINING JOB"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "hostname=$(hostname)"
echo "train_config=${TRAIN_CONFIG}"
echo "git_commit=$(git rev-parse HEAD)"
echo "start_time=$(date --iso-8601=seconds)"
echo "========================================"

python - "${TRAIN_CONFIG}" <<'PYCFG'
import sys
import yaml

with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)

print("resolved_run_name =", cfg["run_name"])
print("resolved_total_steps =", cfg["train"]["total_steps"])
print("resolved_batch_size =", cfg["train"]["batch_size"])
print("resolved_grad_accum =", cfg["train"]["grad_accum_steps"])
print("resolved_log_every =", cfg["train"]["log_every_steps"])
print("resolved_save_every =", cfg["train"]["save_every_steps"])
print(
    "resolved_gradient_checkpointing =",
    cfg["model"].get("gradient_checkpointing"),
)
PYCFG

python -u -m sc2.train.train_sc2_striped_full_continuous \
  --config "${TRAIN_CONFIG}" \
  --paths configs/paths/continuous_hpc.yaml \
  --resume auto
