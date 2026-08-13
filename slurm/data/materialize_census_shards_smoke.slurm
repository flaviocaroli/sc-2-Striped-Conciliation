#!/bin/bash
#SBATCH --job-name=sc2-shards-smoke
#SBATCH --partition=debug_cpu
#SBATCH --qos=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:14:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2-data

export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

cd /home/3159436/sc2/code

DATA=/home/3159436/sc2/data/census_curated_pretrain_pilot250k
SMOKE=${DATA}/materialization_smoke

python scripts/data/materialize_census_shards.py \
  --config configs/data/census_curated_pretrain_pilot250k.yaml \
  --selected-cells "${SMOKE}/selected_cells_smoke.parquet" \
  --vocabulary "${DATA}/gene_vocabulary.parquet" \
  --output-root "${SMOKE}/shards" \
  --cells-per-shard 100 \
  --fetch-chunk-size 100 \
  --resume
