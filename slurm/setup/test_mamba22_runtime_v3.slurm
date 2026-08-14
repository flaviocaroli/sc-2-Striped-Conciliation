#!/bin/bash
#SBATCH --job-name=sc2-mamba-rt3
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

OVERLAY=/home/3159436/sc2/overlays/mamba22_git_torch25_cu124

TORCH_LIB=/home/3159436/.conda/envs/sc2_mamba/lib/python3.10/site-packages/torch/lib

CUDART_LIB=/home/3159436/.conda/envs/sc2_mamba/lib/python3.10/site-packages/nvidia/cuda_runtime/lib

export PYTHONPATH="${OVERLAY}:${PYTHONPATH:-}"

export LD_LIBRARY_PATH="${CUDART_LIB}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

echo "========================================"
echo "SC2 MAMBA RUNTIME TEST V3"
echo "job=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "overlay=${OVERLAY}"
echo "torch_lib=${TORCH_LIB}"
echo "cudart_lib=${CUDART_LIB}"
echo "========================================"

echo
echo "===== GPU ====="

nvidia-smi \
  --query-gpu=name,driver_version,memory.total \
  --format=csv

echo
echo "===== TORCH ====="

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

assert torch.cuda.is_available()

print("TORCH_H200=PASS")
PY

echo
echo "===== FUSED IMPORTS ====="

python - <<'PY'
import torch

import causal_conv1d
import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda

print(
    "causal_conv1d =",
    causal_conv1d.__version__,
)

print(
    "causal_conv1d_cuda =",
    causal_conv1d_cuda.__file__,
)

print(
    "mamba_ssm =",
    mamba_ssm.__version__,
)

print(
    "selective_scan_cuda =",
    selective_scan_cuda.__file__,
)

print("FUSED_IMPORTS_RUNTIME=PASS")
PY

echo
echo "===== LINKER CHECK ====="

for SO in \
  "${OVERLAY}"/causal_conv1d_cuda*.so \
  "${OVERLAY}"/selective_scan_cuda*.so
do
    echo
    echo "FILE=${SO}"

    ldd "${SO}" \
      | grep -E \
        'libcudart|libc10|libtorch|not found' \
      || true

    if ldd "${SO}" | grep -q 'not found'; then
        echo "ERROR: unresolved shared library"
        exit 20
    fi
done

echo
echo "DYNAMIC_LINKING=PASS"

echo
echo "===== 4096-POSITION WARMUP ====="

python - <<'PY'
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

with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    y = model(x)
    loss = y.float().square().mean()

loss.backward()
torch.cuda.synchronize()

print("shape =", tuple(y.shape))
print("loss =", float(loss.detach()))
print(
    "gradient_finite =",
    bool(torch.isfinite(x.grad).all()),
)

assert y.shape == x.shape
assert torch.isfinite(x.grad).all()

print("MAMBA_WARMUP=PASS")
PY

echo
echo "===== TIMED RUNS ====="

python - <<'PY'
import json
import statistics
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

print(
    "parameters =",
    sum(
        p.numel()
        for p in model.parameters()
    ),
)

times = []

for run in range(5):
    x = torch.randn(
        2,
        4096,
        128,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    model.zero_grad(set_to_none=True)

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

    seconds = time.perf_counter() - started

    record = {
        "run": run + 1,
        "seconds": seconds,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        ),
        "loss": float(loss.detach()),
        "gradient_finite": bool(
            torch.isfinite(x.grad).all()
        ),
    }

    print(
        json.dumps(
            record,
            sort_keys=True,
        )
    )

    assert record["gradient_finite"]

    times.append(seconds)

summary = {
    "runs": len(times),
    "mean_seconds": statistics.mean(times),
    "median_seconds": statistics.median(times),
    "min_seconds": min(times),
    "max_seconds": max(times),
}

print(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
)

assert max(times) < 60.0

print("MAMBA22_4096_H200=PASS")
print("OFFICIAL_MAMBA_RUNTIME=PASS")
PY
