#!/bin/bash
#SBATCH --job-name=sc2medium_train
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --signal=TERM@900
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail

CONFIG_PATH="configs/train/striped/sc2_medium_lung_large.yaml"
PATHS_CONFIG="configs/paths/hpc.yaml"

source ~/.bashrc || true
eval "$(conda shell.bash hook)"
conda activate sc2

export SC2_DATA_ROOT=/home/3159436/sc2/data
export SC2_OUTPUT_ROOT=/home/3159436/sc2/outputs
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /home/3159436/sc2/logs /home/3159436/sc2/outputs
cd /home/3159436/sc2/code

python -u -m sc2.train.train_sc2_striped_medium \
  --config "${CONFIG_PATH}" \
  --paths "${PATHS_CONFIG}" \
  --auto-resume &

PY_PID=$!

_term() {
  echo "SLURM signal received; forwarding TERM to python pid ${PY_PID}" >&2
  kill -TERM "${PY_PID}" 2>/dev/null || true
  wait "${PY_PID}" || true
  exit 0
}

trap _term TERM USR1 INT
wait "${PY_PID}"
