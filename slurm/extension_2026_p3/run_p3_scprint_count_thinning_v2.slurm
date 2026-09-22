#!/bin/bash
#SBATCH --job-name=p3-scprint-ct2
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-17
#SBATCH --output=/home/3159436/sc2/logs/p3_scprint_count_thinning_v2/scprint_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/p3_scprint_count_thinning_v2/scprint_%A_%a.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PROTOCOL="${REPO}/configs/extension_2026_p3/p3_scprint_count_thinning_protocol_v2.json"
RUNNER="${REPO}/scripts/eval/run_p3_scprint_count_thinning_task_v2.py"

: "${PINNED_HEAD:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"

test "$(
    git -C "${REPO}" rev-parse HEAD
)" = "${PINNED_HEAD}"

test "$(
    sha256sum "${PROTOCOL}" |
    awk '{print $1}'
)" = "${EXPECTED_PROTOCOL_SHA}"

if [ "${SLURM_ARRAY_TASK_ID}" -lt 9 ]; then

    ENV=/home/3159436/.conda/envs/sc2-p3-scprint
    SOURCE=/home/3159436/sc2/external/foundation_models/p3_16k/sources/scPRINT

    "${ENV}/bin/lamin" connect \
        anonymous/sc2p3scprintv1

    VERSION="$(
    "${ENV}/bin/python" - <<'PY'
import importlib.metadata
print(
    importlib.metadata.version(
        "scDataLoader"
    )
)
PY
    )"

    test "${VERSION}" = "2.1.1"

else

    ENV=/home/3159436/.conda/envs/sc2-p3-scprint2
    SOURCE=/home/3159436/sc2/external/foundation_models/p3_reconstruction_qualification_v1/scPRINT-2

    "${ENV}/bin/lamin" connect \
        anonymous/lamin_storage

fi

PYTHONPATH="${REPO}/src:${REPO}:${SOURCE}:${PYTHONPATH:-}" \
EXPECTED_PROTOCOL_SHA="${EXPECTED_PROTOCOL_SHA}" \
"${ENV}/bin/python" \
    "${RUNNER}" \
    --protocol "${PROTOCOL}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
