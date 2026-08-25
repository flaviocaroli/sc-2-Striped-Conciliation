#!/bin/bash
#SBATCH --job-name=sc2-abl5
#SBATCH --partition=short_gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3159436/sc2/logs/paper_extension_2026/sc2-abl5_%A_%a.out
#SBATCH --error=/home/3159436/sc2/logs/paper_extension_2026/sc2-abl5_%A_%a.err

set -Eeuo pipefail

REPO=/home/3159436/sc2/code
TASKS="${REPO}/configs/extension_2026/ablation_train_tasks_v1.tsv"
LAUNCHER="${REPO}/slurm/striped/train_sc2_continuous_250k.slurm"
OVERLAY=/home/3159436/sc2/overlays/mamba22_git_torch25_cu124

cd "$REPO"

[[ "$(git rev-parse HEAD)" == "${SC2_EXPECTED_COMMIT}" ]] || {
    echo "Git commit mismatch" >&2
    exit 1
}

ROW="$(
    awk -F '\t' \
      -v id="${SLURM_ARRAY_TASK_ID}" \
      'NR>1 && $1==id {print; exit}' \
      "$TASKS"
)"

[[ -n "$ROW" ]] || {
    echo "No task mapping" >&2
    exit 1
}

IFS=$'\t' read -r TASK_ID VARIANT SEED RUN_NAME CONFIG <<< "$ROW"

[[ -f "$CONFIG" ]] || {
    echo "Missing config: $CONFIG" >&2
    exit 1
}

echo "TASK_ID=$TASK_ID"
echo "VARIANT=$VARIANT"
echo "SEED=$SEED"
echo "RUN_NAME=$RUN_NAME"
echo "CONFIG=$CONFIG"

export SC2_ENV_NAME=sc2_mamba
export SC2_PYTHON_OVERLAY="$OVERLAY"
export SC2_ENABLE_FUSED_RUNTIME=1
export SC2_REQUIRE_FUSED_IMPORTS=1
export SC2_TRAIN_CONFIG="$CONFIG"

exec bash "$LAUNCHER"
