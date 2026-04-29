#!/bin/bash
#SBATCH --job-name=mk_mamba_env
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
set -u

# Load your cluster CUDA toolkit module if needed.
# Replace with the actual module name shown by `module avail cuda`.
module avail cuda || true
# module load cuda/12.4

which nvcc || true
nvcc --version || true

conda create -y -n sc2_mamba python=3.10
conda activate sc2_mamba

pip install --upgrade pip setuptools wheel packaging ninja

# Install a stable PyTorch build from the official wheel index.
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
PY

pip install "causal-conv1d>=1.4.0" --no-build-isolation
pip install "mamba-ssm[causal-conv1d]" --no-build-isolation

python - <<'PY'
import torch
from mamba_ssm import Mamba
print("mamba import ok")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
PY