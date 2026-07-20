#!/bin/bash
#SBATCH --job-name=sc2full_mask
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

export BASHRCSOURCED=${BASHRCSOURCED:-0}
set -eo pipefail
CONFIG_PATH="${1:?Usage: sbatch eval_sc2_striped_full_masked.slurm CONFIG_PATH}"
PATHS_CONFIG="configs/paths/hpc.yaml"
eval "$(conda shell.bash hook)"
conda activate sc2
set -u
export SC2_DATA_ROOT=/home/3159436/sc2/data
export SC2_OUTPUT_ROOT=/home/3159436/sc2/outputs
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
cd /home/3159436/sc2/code
python -u -m sc2.eval.benchmarks.evaluate_sc2_striped_full_masked \
  --config "${CONFIG_PATH}" \
  --paths "${PATHS_CONFIG}"
