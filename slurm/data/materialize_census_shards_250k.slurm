#!/bin/bash
#SBATCH --job-name=sc2-shards-250k
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2-data
cd /home/3159436/sc2/code
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}
DATA=/home/3159436/sc2/data/census_curated_pretrain_pilot250k

python scripts/data/materialize_census_shards.py \
  --config configs/data/census_curated_pretrain_pilot250k.yaml \
  --selected-cells "${DATA}/selected_cells.parquet" \
  --vocabulary "${DATA}/gene_vocabulary.parquet" \
  --output-root "${DATA}/shards" \
  --cells-per-shard 25000 \
  --fetch-chunk-size 5000 \
  --resume
