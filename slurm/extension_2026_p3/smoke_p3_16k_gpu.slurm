#!/bin/bash
#SBATCH --job-name=sc2-p3-smoke16k
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/3159436/sc2_p3/logs/gpu_smoke16k/sc2-p3-smoke16k-%A_%a.out
#SBATCH --error=/scratch/3159436/sc2_p3/logs/gpu_smoke16k/sc2-p3-smoke16k-%A_%a.err

set -Eeuo pipefail
umask 027

REPO=/home/3159436/sc2/code
PY=/home/3159436/.conda/envs/sc2_mamba/bin/python
OVERLAY=/home/3159436/sc2/overlays/mamba22_git_torch25_cu124

SCRIPT="${REPO}/scripts/extension_2026_p3/smoke_p3_16k_gpu.py"

SMALL="${REPO}/configs/train/extension_2026_p3/architecture_audit/sc2_p3_16k_small_archaudit_s20260728.yaml"
LARGE="${REPO}/configs/train/extension_2026_p3/architecture_audit/sc2_p3_16k_large_archaudit_s20260728.yaml"

EXPECTED_SMALL_SHA=dff3a11a3274e9221d6e4e82e83fd1e01c92ad860b08fda150bc939f4826fe32
EXPECTED_LARGE_SHA=44e88855a53e9c554fd4b486dad46ad14b1af345ea0d58824e8f8771ac6f4c40

sha() {
    sha256sum "$1" | awk '{print $1}'
}

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        VARIANT=small
        CONFIG="${SMALL}"
        EXPECTED_SHA="${EXPECTED_SMALL_SHA}"
        ;;
    1)
        VARIANT=large
        CONFIG="${LARGE}"
        EXPECTED_SHA="${EXPECTED_LARGE_SHA}"
        ;;
    *)
        echo "Unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
        exit 1
        ;;
esac

test -f "${CONFIG}"
test "$(sha "${CONFIG}")" = "${EXPECTED_SHA}"
test -x "${SCRIPT}"
test -x "${PY}"
test -d "${OVERLAY}"

unset PYTHONPATH || true

export PYTHONPATH="${OVERLAY}:${REPO}/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"

echo "============================================================"
echo "P3 16K H200 SMOKE"
echo "============================================================"
echo "HOST=$(hostname)"
echo "JOB=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "VARIANT=${VARIANT}"
echo "CONFIG=${CONFIG}"
echo "CONFIG_SHA256=$(sha "${CONFIG}")"
echo "GIT_HEAD=$(git rev-parse HEAD)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${PY}" - <<'PY'
import torch
import numpy
import mamba_ssm
import causal_conv1d
import sc2

print("TORCH =", torch.__version__)
print("TORCH_CUDA =", torch.version.cuda)
print("CUDA_AVAILABLE =", torch.cuda.is_available())
print("DEVICE_COUNT =", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

print("DEVICE =", torch.cuda.get_device_name(0))
print("NUMPY =", numpy.__version__, numpy.__file__)
print("MAMBA_SSM =", mamba_ssm.__version__, mamba_ssm.__file__)
print("CAUSAL_CONV1D =", causal_conv1d.__version__, causal_conv1d.__file__)
print("SC2 =", sc2.__version__, sc2.__file__)
print("GPU_ENV_GATE=PASS")
PY

"${PY}" \
    "${SCRIPT}" \
    --config "${CONFIG}" \
    --variant "${VARIANT}" \
    --batch-size 1 \
    --seed 20260728

echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "P3_16K_H200_SMOKE_WRAPPER=PASS"
