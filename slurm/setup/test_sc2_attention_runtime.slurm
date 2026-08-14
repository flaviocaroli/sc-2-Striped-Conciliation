#!/bin/bash
#SBATCH --job-name=sc2-attn-runtime
#SBATCH --partition=debug_gpuh200
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:14:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2_mamba

REPO=/home/3159436/sc2/code
OVERLAY=/home/3159436/sc2/overlays/mamba22_git_torch25_cu124

TORCH_LIB=/home/3159436/.conda/envs/sc2_mamba/lib/python3.10/site-packages/torch/lib
NVIDIA_ROOT=/home/3159436/.conda/envs/sc2_mamba/lib/python3.10/site-packages/nvidia

NVIDIA_LIB_PATHS=$(
  find "${NVIDIA_ROOT}" \
    -type d \
    -name lib \
    -printf '%p\n' \
  | sort \
  | paste -sd: -
)

export PYTHONPATH="${OVERLAY}:${REPO}/src:${PYTHONPATH:-}"

export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

echo "========================================"
echo "SC2 ATTENTION RUNTIME TEST"
echo "job=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "========================================"

python - <<'PY'
import ctypes
import torch

ctypes.CDLL("libnvrtc.so.12")

print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("gpu =", torch.cuda.get_device_name(0))
print("cudnn =", torch.backends.cudnn.version())

print("NVRTC_LOAD=PASS")
PY

echo
echo "===== PYTORCH MULTIHEAD ATTENTION ====="

python - <<'PY'
import torch
from torch import nn

torch.manual_seed(20260728)

device = torch.device("cuda")

attn = nn.MultiheadAttention(
    embed_dim=128,
    num_heads=4,
    dropout=0.0,
    batch_first=True,
).to(device)

x = torch.randn(
    2,
    256,
    128,
    device=device,
    requires_grad=True,
)

with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    y, _ = attn(
        x,
        x,
        x,
        need_weights=False,
    )

    loss = y.float().square().mean()

loss.backward()

torch.cuda.synchronize()

print("shape =", tuple(y.shape))
print(
    "gradient_finite =",
    bool(torch.isfinite(x.grad).all()),
)

assert torch.isfinite(x.grad).all()

print("MHA_SDPA=PASS")
PY

echo
echo "===== SC2 ROUTED ATTENTION ====="

python - <<'PY'
import torch

from sc2.models.striped.routed_attention_checkpoint import (
    RoutedTopKAttentionCheckpoint,
)

torch.manual_seed(20260728)

device = torch.device("cuda")

block = RoutedTopKAttentionCheckpoint(
    d_model=128,
    n_heads=4,
    top_k=256,
    learned_fraction=0.5,
    dropout=0.0,
    expression_score_weight=0.25,
    nonzero_bonus=0.05,
    marker_prior_weight=0.0,
).to(device)

hidden = torch.randn(
    2,
    4096,
    128,
    device=device,
    requires_grad=True,
)

expression = torch.rand(
    2,
    4096,
    device=device,
)

with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    output = block(
        hidden,
        score_source=expression,
        marker_prior=None,
    )

    loss = output.float().square().mean()

loss.backward()

torch.cuda.synchronize()

print(
    "shape =",
    tuple(output.shape),
)

print(
    "gradient_finite =",
    bool(
        torch.isfinite(
            hidden.grad
        ).all()
    ),
)

assert torch.isfinite(
    hidden.grad
).all()

print(
    "SC2_ROUTED_ATTENTION=PASS"
)
PY
