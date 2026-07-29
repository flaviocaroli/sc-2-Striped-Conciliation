#!/bin/bash
#SBATCH --job-name=sc2-cont-250k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2
cd /home/3159436/sc2/code
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}

python scripts/data/validate_shards.py \
  --manifest /home/3159436/sc2/data/census_curated_pretrain_pilot250k/shards.parquet \
  --verify-hashes

python -m sc2.train.train_sc2_striped_full_continuous \
  --config configs/train/striped/sc2_continuous_250k.yaml \
  --paths configs/paths/continuous_hpc.yaml \
  --resume auto
