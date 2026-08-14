#!/bin/bash
#SBATCH --job-name=sc2-p1-eval
#SBATCH --gres=gpu:1
#SBATCH --output=/home/3159436/sc2/logs/sc2-p1-eval_%j.out
#SBATCH --error=/home/3159436/sc2/logs/sc2-p1-eval_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SC2_ENV_NAME:-sc2_mamba}"

REPO=/home/3159436/sc2/code
cd "${REPO}"

if [[ -n "${SC2_PYTHON_OVERLAY:-}" ]]; then
  export PYTHONPATH="${SC2_PYTHON_OVERLAY}:${REPO}/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
fi

TORCH_LIB=$(
python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)

NVIDIA_ROOT=$(
python - <<'PY'
from pathlib import Path
import site
for root in site.getsitepackages():
    p = Path(root) / "nvidia"
    if p.is_dir():
        print(p)
        break
PY
)

NVIDIA_LIB_PATHS=$(
find "${NVIDIA_ROOT}" -type d -name lib -printf '%p\n' | sort | paste -sd: -
)

export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import ctypes
import causal_conv1d
import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda
ctypes.CDLL("libcudart.so.12")
ctypes.CDLL("libnvrtc.so.12")
print("EVAL_RUNTIME=PASS")
PY


echo "===== EVALUATION INPUTS ====="
for name in SC2_EVAL_CONFIG SC2_CHECKPOINT SC2_BENCHMARK SC2_OUTPUT_DIR; do
  value="${!name:-}"
  if [[ -z "${value}" ]]; then
    echo "ERROR: ${name} is empty"
    exit 2
  fi
  echo "${name}=${value}"
done

[[ -f "${SC2_EVAL_CONFIG}" ]] || {
  echo "ERROR: eval config not found: ${SC2_EVAL_CONFIG}"
  exit 2
}

[[ -f "${SC2_CHECKPOINT}" ]] || {
  echo "ERROR: checkpoint not found: ${SC2_CHECKPOINT}"
  exit 2
}

[[ -f "${SC2_BENCHMARK}" ]] || {
  echo "ERROR: benchmark not found: ${SC2_BENCHMARK}"
  exit 2
}

python -u -m sc2.eval.evaluate_continuous_repair \
  --config "${SC2_EVAL_CONFIG}" \
  --checkpoint "${SC2_CHECKPOINT}" \
  --benchmark "${SC2_BENCHMARK}" \
  --output-dir "${SC2_OUTPUT_DIR}"
