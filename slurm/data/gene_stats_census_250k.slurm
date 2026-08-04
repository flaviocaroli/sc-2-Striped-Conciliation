#!/bin/bash
#SBATCH --job-name=sc2-gstats-250k
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2-data
cd /home/3159436/sc2/code
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}
DATA=/home/3159436/sc2/data/census_curated_pretrain_pilot250k

python scripts/data/compute_census_train_gene_stats.py \
  --config configs/data/census_curated_pretrain_pilot250k.yaml \
  --selected-cells "${DATA}/selected_cells.parquet" \
  --output "${DATA}/train_gene_stats.parquet" \
  --state-dir "${DATA}/gene_stats_state" \
  --chunk-size 1024 \
  --save-every-chunks 10 \
  --resume
