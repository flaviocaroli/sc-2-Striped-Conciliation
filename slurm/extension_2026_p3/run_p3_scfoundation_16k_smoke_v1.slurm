#!/bin/bash
#SBATCH --job-name=p3fm-scf16k-smoke
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/scfoundation_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/scfoundation_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2-scfoundation/bin/python

PROTOCOL="${REPO}/configs/extension_2026_p3/p3_scfoundation_16k_validation_smoke_protocol_v1.json"
RUNNER="${REPO}/scripts/eval/run_scfoundation_final_task.py"

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/scfoundation"
RUNOUT="${OUTROOT}/run"
GPU_LOG="${OUTROOT}/gpu_memory_mib.tsv"

: "${EXPECTED_JOB_HEAD:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test -x "${PY}"
test -s "${PROTOCOL}"
test -s "${RUNNER}"
test ! -e "${RUNOUT}"

mkdir -p "${OUTROOT}"

unset PYTHONHOME
unset LD_LIBRARY_PATH

export PYTHONPATH="${REPO}/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"

nvidia-smi \
    --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv,noheader

"${PY}" - <<'PY'
import torch

print("TORCH_VERSION=", torch.__version__)
print("CUDA_BUILD=", torch.version.cuda)
print("CUDA_AVAILABLE=", torch.cuda.is_available())

assert torch.cuda.is_available()

print("GPU_NAME=", torch.cuda.get_device_name(0))
PY

(
    while true; do

        printf '%s\t' "$(date -u +%s.%N)"

        nvidia-smi \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits |
            head -n1

        sleep 0.25

    done
) > "${GPU_LOG}" &

SAMPLER=$!

cleanup() {
    kill "${SAMPLER}" 2>/dev/null || true
    wait "${SAMPLER}" 2>/dev/null || true
}

trap cleanup EXIT

START=$(date +%s)

"${PY}" "${RUNNER}" \
    --protocol "${PROTOCOL}" \
    --task-id 0

END=$(date +%s)

cleanup
trap - EXIT

PEAK=$(
    awk '
        NF >= 2 {
            gsub(/[^0-9.]/, "", $2)
            if (($2 + 0) > max) max = $2 + 0
        }
        END {
            printf "%.0f", max + 0
        }
    ' "${GPU_LOG}"
)

echo "SCFOUNDATION_RUNTIME_SECONDS=$((END - START))"
echo "SCFOUNDATION_PEAK_GPU_MEMORY_MIB=${PEAK}"

test -d "${RUNOUT}"

echo
echo "OUTPUT FILES:"

find "${RUNOUT}" \
    -type f \
    -printf '%p|%s bytes\n' |
    sort

COUNT=$(
    find "${RUNOUT}" \
        -type f |
    wc -l
)

test "${COUNT}" -gt 0

echo "P3_SCF_SOURCE_FEATURE_SPACE=16384"
echo "P3_SCF_MAPPED_GENES=14382"
echo "P3_SCF_DATASET=INTERNAL_VALIDATION"
echo "P3_SCF_SMOKE_CELLS=16"
echo "P3_SCF_SCIENTIFIC_RESULT=false"
echo "P3_SCF_CUSTOM_DECODER=false"
echo "P3_SCF_AD_HOC_INVERSE=false"
echo "P3_SCF_16K_VALIDATION_SMOKE=PASS"
