#!/bin/bash
#SBATCH --job-name=scfoundation-smoke
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3159436/sc2/logs/paper_extension_2026/scfoundation-smoke_%j.out
#SBATCH --error=/home/3159436/sc2/logs/paper_extension_2026/scfoundation-smoke_%j.err

set -Eeuo pipefail

REPO=/home/3159436/sc2/code
PY=/home/3159436/.conda/envs/sc2-scfoundation/bin/python

PROTOCOL="${REPO}/configs/extension_2026/scfoundation_decoder_smoke_protocol_v1.json"
RUNNER="${REPO}/scripts/eval/run_scfoundation_decoder_smoke.py"

cd "$REPO"

[[ "$(git rev-parse HEAD)" == "${SC2_EXPECTED_COMMIT}" ]] || {
    echo "Git commit mismatch" >&2
    exit 1
}

env \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u LD_LIBRARY_PATH \
  SC2_SCFOUNDATION_SMOKE_PROTOCOL_SHA="${SC2_SCFOUNDATION_SMOKE_PROTOCOL_SHA}" \
  "$PY" "$RUNNER" \
    --protocol "$PROTOCOL"
