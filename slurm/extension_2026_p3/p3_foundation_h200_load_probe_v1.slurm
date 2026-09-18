#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_16k_resource_preflight_v1/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_16k_resource_preflight_v1/%x_%j.err

set -Eeuo pipefail

: "${MODE:?}"
: "${EXPECTED_JOB_HEAD:?}"

REPO=/home/3159436/sc2/code

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"
echo "MODE=${MODE}"

nvidia-smi \
    --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv,noheader

if [ "${MODE}" = "hardware" ]; then
    echo "FOUNDATION_H200_CAPACITY_PROBE=PASS"
    exit 0
fi

: "${MODEL:?}"
: "${MODEL_DIR:?}"
: "${MODEL_PY:?}"

test -x "${MODEL_PY}"
test -d "${MODEL_DIR}"

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"${MODEL_PY}" - "${MODEL}" "${MODEL_DIR}" <<'PY'
from pathlib import Path
import sys

model_name = sys.argv[1]
model_dir = Path(sys.argv[2])

import torch
from transformers import AutoModel

assert torch.cuda.is_available()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

free_before, total = torch.cuda.mem_get_info()

model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype="auto",
)

model = model.cuda()
model.eval()

torch.cuda.synchronize()

free_after, _ = torch.cuda.mem_get_info()

params = sum(p.numel() for p in model.parameters())

print("MODEL=", model_name)
print("MODEL_DIR=", model_dir)
print("PARAMETERS=", params)
print("GPU_TOTAL_GIB=", total / 1024**3)
print("GPU_FREE_BEFORE_GIB=", free_before / 1024**3)
print("GPU_FREE_AFTER_LOAD_GIB=", free_after / 1024**3)
print(
    "PEAK_ALLOCATED_GIB=",
    torch.cuda.max_memory_allocated() / 1024**3,
)
print(
    "PEAK_RESERVED_GIB=",
    torch.cuda.max_memory_reserved() / 1024**3,
)

print("FULL_NATIVE_FORWARD_MEMORY_MEASURED=false")
print("FOUNDATION_LOCAL_MODEL_LOAD_PROBE=PASS")
PY
