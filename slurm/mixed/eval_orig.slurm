#!/bin/bash
#SBATCH --job-name=sc2orig_eval
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --exclude=gnode16
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail

CONFIG_PATH="configs/eval/mixed/sc2_native_mamba_original_like_lung_large_eval.yaml"
PATHS_CONFIG="configs/paths/hpc.yaml"

eval "$(conda shell.bash hook)"
conda activate sc2

export SC2_DATA_ROOT=/home/3159436/sc2/data
export SC2_OUTPUT_ROOT=/home/3159436/sc2/outputs
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}

mkdir -p /home/3159436/sc2/logs /home/3159436/sc2/outputs

cd /home/3159436/sc2/code

python -m sc2.eval.evaluate_sc2_mamba_bridge \
  --config "${CONFIG_PATH}" \
  --paths "${PATHS_CONFIG}"