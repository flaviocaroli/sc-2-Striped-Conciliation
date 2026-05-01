#!/bin/bash
#SBATCH --job-name=census_large
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate sc2
set -u

export SC2_DATA_ROOT=/home/3159436/sc2/data

cd /home/3159436/sc2/code

python preprocess/census/export_census_large_lung_pilot.py \
  --max-datasets 12 \
  --cells-per-dataset 2500