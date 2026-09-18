#!/bin/bash
#SBATCH --job-name=p3fm-scf16k-v3
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/scfoundation_v3_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/scfoundation_v3_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2-scfoundation/bin/python

RUNNER="${REPO}/scripts/eval/run_scfoundation_p3_16k_smoke_v2.py"

# Existing protocol already passed benchmark-SHA validation.
PROTOCOL="${REPO}/configs/extension_2026_p3/p3_scfoundation_16k_validation_smoke_protocol_v1.json"

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/scfoundation"
RUNOUT="${OUTROOT}/run"
GPU_LOG="${OUTROOT}/gpu_memory_v3_mib.tsv"

: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"
: "${EXPECTED_PROTOCOL_SHA:?EXPECTED_PROTOCOL_SHA missing}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test -x "${PY}"
test -s "${RUNNER}"
test -s "${PROTOCOL}"

ACTUAL_PROTOCOL_SHA=$(
    sha256sum "${PROTOCOL}" |
    awk '{print $1}'
)

echo "EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA}"
echo "ACTUAL_PROTOCOL_SHA=${ACTUAL_PROTOCOL_SHA}"

test "${ACTUAL_PROTOCOL_SHA}" = \
    "${EXPECTED_PROTOCOL_SHA}"

# Prior jobs failed before producing scientific output.
# Never overwrite a real output.
test ! -e "${RUNOUT}" || {
    echo "STOP: existing run output ${RUNOUT}" >&2
    exit 1
}

rm -f "${GPU_LOG}"

unset PYTHONHOME
unset LD_LIBRARY_PATH

export PYTHONPATH="${REPO}/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "scFoundation P3 16K SMOKE V3"
echo "============================================================"
echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"
echo "SOURCE_GENES=16384"
echo "MAPPED_GENES=14382"
echo "NATIVE_GENES=19264"
echo "CELLS=16"

nvidia-smi \
    --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv,noheader

"${PY}" - <<'PY'
import torch

assert torch.cuda.is_available()

print("TORCH=", torch.__version__)
print("CUDA_BUILD=", torch.version.cuda)
print("GPU=", torch.cuda.get_device_name(0))
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

            if (($2 + 0) > max) {
                max = $2 + 0
            }
        }

        END {
            printf "%.0f", max + 0
        }
    ' "${GPU_LOG}"
)

echo "SCFOUNDATION_RUNTIME_SECONDS=$((END - START))"
echo "SCFOUNDATION_PEAK_GPU_MEMORY_MIB=${PEAK}"

test -d "${RUNOUT}"

COUNT=$(
    find "${RUNOUT}" \
        -type f |
    wc -l
)

test "${COUNT}" -gt 0

find "${RUNOUT}" \
    -type f \
    -printf '%p|%s bytes\n' |
    sort

echo "P3_SCF_SOURCE_FEATURE_SPACE=16384"
echo "P3_SCF_MAPPED_GENES=14382"
echo "P3_SCF_NATIVE_GENES=19264"
echo "P3_SCF_CELLS=16"
echo "P3_SCF_SCIENTIFIC_RESULT=false"
echo "P3_SCF_CUSTOM_DECODER=false"
echo "P3_SCF_AD_HOC_INVERSE=false"
echo "P3_SCF_PROTOCOL_REWRITTEN=false"
echo "P3_SCF_P2_RUNNER_MODIFIED=false"
echo "P3_SCF_16K_SMOKE_V3=PASS"
