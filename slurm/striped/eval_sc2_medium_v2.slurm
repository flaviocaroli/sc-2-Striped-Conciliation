#!/bin/bash
#SBATCH --job-name=sc2medv2_eval
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

export BASHRCSOURCED=${BASHRCSOURCED:-0}
set -eo pipefail

CONFIG_PATH="configs/eval/striped/sc2_medium_v2_lung_eval.yaml"
PATHS_CONFIG="configs/paths/hpc.yaml"

eval "$(conda shell.bash hook)"
conda activate sc2
set -u

export SC2_DATA_ROOT=/home/3159436/sc2/data
export SC2_OUTPUT_ROOT=/home/3159436/sc2/outputs
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /home/3159436/sc2/logs /home/3159436/sc2/outputs
cd /home/3159436/sc2/code

python -u -m sc2.eval.evaluate_sc2_mamba_bridge \
  --config "${CONFIG_PATH}" \
  --paths "${PATHS_CONFIG}"
