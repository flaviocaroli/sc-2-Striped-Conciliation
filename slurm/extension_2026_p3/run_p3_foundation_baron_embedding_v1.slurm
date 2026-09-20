#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_baron_embeddings_v1/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_baron_embeddings_v1/%x_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

RUNNER="${REPO}/scripts/extension_2026_p3/run_foundation_baron_embeddings_v1.py"
PROTOCOL="${REPO}/configs/extension_2026_p3/p3_foundation_baron_representation_protocol_v1.json"

OVERLAY="${SC2}/overlays/mamba22_git_torch25_cu124"

: "${MODEL:?}"
: "${EXPECTED_JOB_HEAD:?}"
: "${EXPECTED_RUNNER_SHA:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test "$(sha256sum "${RUNNER}" | awk '{print $1}')" \
    = "${EXPECTED_RUNNER_SHA}"

test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" \
    = "${EXPECTED_PROTOCOL_SHA}"

case "${MODEL}" in

    Geneformer)

        PY=/home/3159436/.conda/envs/sc2-p3-geneformer/bin/python

        export PYTHONPATH="${REPO}/src:${REPO}"
        ;;

    GeneMamba)

        PY=/home/3159436/.conda/envs/sc2-p3-genemamba/bin/python

        CKPT="${SC2}/external/foundation_models/p3_16k/checkpoints/GeneMamba"
        SOURCE="${SC2}/external/foundation_models/p3_16k/sources/GeneMamba"

        export PYTHONPATH="${OVERLAY}:${CKPT}:${SOURCE}:${REPO}/src:${REPO}"
        ;;

    *)

        echo "STOP: invalid MODEL=${MODEL}" >&2
        exit 1
        ;;

esac

test -x "${PY}"

unset PYTHONHOME

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "P3 BARON FOUNDATION EMBEDDING"
echo "============================================================"

echo "MODEL=${MODEL}"
echo "JOB=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"

nvidia-smi \
    --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv,noheader

"${PY}" "${RUNNER}" \
    --protocol "${PROTOCOL}" \
    --model "${MODEL}"

echo "P3_${MODEL}_BARON_EMBEDDING_JOB=PASS"
