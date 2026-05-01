#!/bin/bash
#SBATCH --job-name=sc2mamba
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate sc2
set -u

export SC2_DATA_ROOT=/home/3159436/sc2/data
export SC2_OUTPUT_ROOT=/home/3159436/sc2/outputs

cd /home/3159436/sc2/code

python -m sc2.train.train_sc2_mamba_bridge \
  --config configs/train/mixed/sc2_mamba_bridge_lung_large.yaml \
  --paths configs/paths/hpc.yaml