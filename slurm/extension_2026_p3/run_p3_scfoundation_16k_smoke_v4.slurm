#!/bin/bash
#SBATCH --job-name=p3fm-scf16k-v4
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/scfoundation_v4_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/scfoundation_v4_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2-scfoundation/bin/python

RUNNER_REL="scripts/eval/run_scfoundation_p3_16k_smoke_v2.py"
PROTOCOL_REL="configs/extension_2026_p3/p3_scfoundation_16k_validation_smoke_protocol_v1.json"

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/scfoundation"
RUNOUT="${OUTROOT}/run"
GPU_LOG="${OUTROOT}/gpu_memory_v4_mib.tsv"

: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"
: "${EXPECTED_PROTOCOL_SHA:?EXPECTED_PROTOCOL_SHA missing}"

cd "${REPO}"

ACTUAL_HEAD=$(git rev-parse HEAD)

echo "EXPECTED_JOB_HEAD=${EXPECTED_JOB_HEAD}"
echo "ACTUAL_HEAD=${ACTUAL_HEAD}"

test "${ACTUAL_HEAD}" = "${EXPECTED_JOB_HEAD}"

# Do not depend on mutable working-tree copies.
git cat-file -e "${EXPECTED_JOB_HEAD}^{commit}"

TMPROOT=$(
    mktemp -d \
        "/tmp/sc2-scf-p3-${SLURM_JOB_ID}.XXXXXX"
)

SAMPLER=""

cleanup() {
    if [ -n "${SAMPLER}" ]; then
        kill "${SAMPLER}" 2>/dev/null || true
        wait "${SAMPLER}" 2>/dev/null || true
    fi

    rm -rf "${TMPROOT}"
}

trap cleanup EXIT

RUNNER="${TMPROOT}/runner.py"
PROTOCOL="${TMPROOT}/protocol.json"

git show \
    "${EXPECTED_JOB_HEAD}:${RUNNER_REL}" \
    > "${RUNNER}"

git show \
    "${EXPECTED_JOB_HEAD}:${PROTOCOL_REL}" \
    > "${PROTOCOL}"

test -s "${RUNNER}"
test -s "${PROTOCOL}"

ACTUAL_PROTOCOL_SHA=$(
    sha256sum "${PROTOCOL}" |
    awk '{print $1}'
)

echo "EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA}"
echo "MATERIALIZED_PROTOCOL_SHA=${ACTUAL_PROTOCOL_SHA}"

test "${ACTUAL_PROTOCOL_SHA}" = \
    "${EXPECTED_PROTOCOL_SHA}"

"${PY}" -m py_compile "${RUNNER}"

# Previous jobs never produced scientific RUNOUT.
test ! -e "${RUNOUT}" || {
    echo "STOP: existing scientific run output ${RUNOUT}" >&2
    exit 1
}

test ! -e "${GPU_LOG}" || {
    echo "STOP: existing v4 GPU log ${GPU_LOG}" >&2
    exit 1
}

mkdir -p "${OUTROOT}"

unset PYTHONHOME
unset LD_LIBRARY_PATH

export PYTHONPATH="${REPO}/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "scFoundation P3 16K SMOKE V4"
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

START=$(date +%s)

"${PY}" "${RUNNER}" \
    --protocol "${PROTOCOL}" \
    --task-id 0

END=$(date +%s)

kill "${SAMPLER}" 2>/dev/null || true
wait "${SAMPLER}" 2>/dev/null || true
SAMPLER=""

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
echo "P3_SCF_RUNNER_SOURCE=GIT_OBJECT"
echo "P3_SCF_PROTOCOL_SOURCE=GIT_OBJECT"
echo "P3_SCF_16K_SMOKE_V4=PASS"
