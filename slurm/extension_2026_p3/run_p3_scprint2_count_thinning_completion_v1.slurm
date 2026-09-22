#!/bin/bash
#SBATCH --job-name=p3-scp2-final
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_scprint2_count_thinning_completion_v1/scprint2_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/p3_scprint2_count_thinning_completion_v1/scprint2_%A_%a.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

ENV=/home/3159436/.conda/envs/sc2-p3-scprint2
SOURCE=/home/3159436/sc2/external/foundation_models/p3_reconstruction_qualification_v1/scPRINT-2

PROTOCOL="${REPO}/configs/extension_2026_p3/p3_scprint_count_thinning_protocol_v2.json"
AMENDMENT="${REPO}/configs/extension_2026_p3/p3_scprint2_count_thinning_completion_amendment_v1.json"
RUNNER="${REPO}/scripts/eval/run_p3_scprint2_count_thinning_completion_v1.py"

: "${PINNED_HEAD:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"
: "${EXPECTED_AMENDMENT_SHA:?}"

test "$(
    git -C "${REPO}" rev-parse HEAD
)" = "${PINNED_HEAD}"

test "$(
    sha256sum "${PROTOCOL}" |
    awk '{print $1}'
)" = "${EXPECTED_PROTOCOL_SHA}"

test "$(
    sha256sum "${AMENDMENT}" |
    awk '{print $1}'
)" = "${EXPECTED_AMENDMENT_SHA}"

"${ENV}/bin/lamin" connect \
    anonymous/lamin_storage

PYTHONPATH="${REPO}/src:${REPO}:${SOURCE}:${PYTHONPATH:-}" \
EXPECTED_PROTOCOL_SHA="${EXPECTED_PROTOCOL_SHA}" \
EXPECTED_AMENDMENT_SHA="${EXPECTED_AMENDMENT_SHA}" \
"${ENV}/bin/python" \
    "${RUNNER}" \
    --protocol "${PROTOCOL}" \
    --amendment "${AMENDMENT}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
