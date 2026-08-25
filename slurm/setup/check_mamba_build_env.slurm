#!/bin/bash
#SBATCH --job-name=sc2-mamba-check
#SBATCH --partition=debug_gpuh200
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:14:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2

echo "host=$(hostname)"
echo "python=$(which python)"
python --version

python - <<'PY'
import os
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("CUDA_HOME", CUDA_HOME)
print("cxx11_abi", torch._C._GLIBCXX_USE_CXX11_ABI)
print("PATH", os.environ.get("PATH"))
PY

echo "nvcc_path=$(command -v nvcc || true)"
nvcc --version || true
nvidia-smi
