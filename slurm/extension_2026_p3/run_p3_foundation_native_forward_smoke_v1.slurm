#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/native_forward_v1_%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/native_forward_v1_%x_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

INPUT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/native_forward_input_v1/validation_first16_raw_counts.npz"
OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/native_gpu_forward_v1"

RUNNER="${REPO}/scripts/extension_2026_p3/run_foundation_native_forward_smoke_v1.py"

OVERLAY="${SC2}/overlays/mamba22_git_torch25_cu124"

: "${MODEL:?MODEL missing}"
: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"
: "${EXPECTED_INPUT_SHA:?EXPECTED_INPUT_SHA missing}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = \
    "${EXPECTED_JOB_HEAD}"

test -z "$(git status --porcelain=v1 -uall)"

test -s "${RUNNER}"
test -s "${INPUT}"

test "$(
    sha256sum "${INPUT}" |
    awk '{print $1}'
)" = "${EXPECTED_INPUT_SHA}"

case "${MODEL}" in

    Geneformer)

        PY=/home/3159436/.conda/envs/sc2-p3-geneformer/bin/python

        SOURCE="${SC2}/external/foundation_models/p3_16k/sources/Geneformer"
        CKPT="${SC2}/external/foundation_models/p3_16k/checkpoints/Geneformer-V2-104M"

        test "$(
            git -C "${SOURCE}" rev-parse HEAD
        )" = \
            "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"

        export PYTHONPATH="${REPO}/src"
        ;;

    GeneMamba)

        PY=/home/3159436/.conda/envs/sc2-p3-genemamba/bin/python

        SOURCE="${SC2}/external/foundation_models/p3_16k/sources/GeneMamba"
        CKPT="${SC2}/external/foundation_models/p3_16k/checkpoints/GeneMamba"

        test "$(
            git -C "${SOURCE}" rev-parse HEAD
        )" = \
            "22f571a8d48256894de05e9d87ea9ec6caa338c0"

        # Exact environment combination that already passed.
        export PYTHONPATH="${OVERLAY}:${CKPT}:${SOURCE}:${REPO}/src"
        ;;

    scPRINT)

        PY=/home/3159436/.conda/envs/sc2-p3-scprint/bin/python

        SOURCE="${SC2}/external/foundation_models/p3_16k/sources/scPRINT"
        CKPT="${SC2}/external/foundation_models/p3_16k/checkpoints/scPRINT/medium-v1.5.ckpt"

        test "$(
            git -C "${SOURCE}" rev-parse HEAD
        )" = \
            "e7c60766cbff58bb7280a1c8fe4b7461b1af1911"

        export PYTHONPATH="${REPO}/src"
        ;;

    *)

        echo "Unknown MODEL=${MODEL}" >&2
        exit 1
        ;;

esac

test -x "${PY}"
test -e "${SOURCE}"
test -e "${CKPT}"

OUT="${OUTROOT}/${MODEL}"

test ! -e "${OUT}"

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "P3 FOUNDATION NATIVE GPU FORWARD"
echo "============================================================"
echo "MODEL=${MODEL}"
echo "JOB=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "SOURCE_FEATURE_SPACE=16384"
echo "SCIENTIFIC_RESULT=false"

nvidia-smi \
    --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv,noheader

"${PY}" "${RUNNER}" \
    --model "${MODEL}" \
    --input-bundle "${INPUT}" \
    --checkpoint "${CKPT}" \
    --source "${SOURCE}" \
    --output-dir "${OUT}"

test -s "${OUT}/smoke_receipt.json"

echo "MODEL=${MODEL}"
echo "P3_FOUNDATION_NATIVE_GPU_FORWARD_JOB=PASS"
