#!/bin/bash
#SBATCH --job-name=probe_cuda
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate sc2_mamba
set -u

echo "HOSTNAME=$(hostname)"
echo "USER=$(whoami)"
echo "PATH=$PATH"

echo
echo "=== GPU visibility ==="
nvidia-smi || true

echo
echo "=== CUDA modules ==="
module avail cuda 2>&1 || true

echo
echo "=== nvcc ==="
which nvcc || true
nvcc --version || true

echo
echo "=== torch check ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
PY