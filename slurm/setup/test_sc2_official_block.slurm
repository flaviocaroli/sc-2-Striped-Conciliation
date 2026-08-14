#!/bin/bash
#SBATCH --job-name=sc2-official-block
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
CUDART_LIB=/home/3159436/.conda/envs/sc2_mamba/lib/python3.10/site-packages/nvidia/cuda_runtime/lib

export PYTHONPATH="${OVERLAY}:${REPO}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CUDART_LIB}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import json
import time

import torch

from sc2.models.striped.native_mamba_block import (
    NativeStripedMambaBlock,
)

print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("gpu =", torch.cuda.get_device_name(0))

assert torch.cuda.is_available()

torch.manual_seed(20260728)

block = NativeStripedMambaBlock(
    d_model=128,
    d_state=4,
    d_conv=4,
    expand=1,
    dropout=0.0,
    bidirectional=True,
    merge_mode="gate",
    backend="official",
).cuda()

print(
    "adapter_backend =",
    block.backend,
)

print(
    "inner_class =",
    type(block.block).__name__,
)

assert block.backend == "official"

x = torch.randn(
    2,
    4096,
    128,
    device="cuda",
    dtype=torch.float32,
    requires_grad=True,
)

# Warm-up
with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    y = block(x)
    loss = y.float().square().mean()

loss.backward()
torch.cuda.synchronize()

assert y.shape == x.shape
assert torch.isfinite(x.grad).all()

print("SC2_OFFICIAL_BLOCK_WARMUP=PASS")

block.zero_grad(set_to_none=True)
x.grad = None

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

started = time.perf_counter()

with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    y = block(x)
    loss = y.float().square().mean()

loss.backward()
torch.cuda.synchronize()

seconds = (
    time.perf_counter()
    - started
)

result = {
    "shape": list(y.shape),
    "seconds_forward_backward": seconds,
    "peak_gpu_memory_gb": (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    ),
    "gradient_finite": bool(
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

assert result["gradient_finite"]

print("SC2_OFFICIAL_BLOCK=PASS")
PY
