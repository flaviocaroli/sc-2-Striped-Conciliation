#!/bin/bash
#SBATCH --job-name=sc2_dropcal
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

export BASHRCSOURCED=${BASHRCSOURCED:-0}
set -eo pipefail

CONFIG_PATH="${1:?Usage: sbatch eval_dropout_calibration.slurm CONFIG_PATH}"
PATHS_CONFIG="configs/paths/hpc.yaml"

eval "$(conda shell.bash hook)"
conda activate sc2

set -u

export SC2_DATA_ROOT=/home/3159436/sc2/data
export SC2_OUTPUT_ROOT=/home/3159436/sc2/outputs
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p \
  /home/3159436/sc2/logs \
  /home/3159436/sc2/outputs

cd /home/3159436/sc2/code

python -u \
  -m sc2.eval.benchmarks.evaluate_dropout_calibration \
  --config "${CONFIG_PATH}" \
  --paths "${PATHS_CONFIG}"
