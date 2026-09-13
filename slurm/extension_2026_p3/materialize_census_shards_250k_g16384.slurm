#!/bin/bash
#SBATCH --job-name=sc2-p3-shards16k
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/3159436/sc2_p3/logs/materialize_g16384/%x-%j.out
#SBATCH --error=/scratch/3159436/sc2_p3/logs/materialize_g16384/%x-%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2-data
cd /home/3159436/sc2/code
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

DATA=/scratch/3159436/sc2_p3/data/census_curated_pretrain_pilot250k_g16384

python scripts/data/materialize_census_shards.py \
  --config configs/data/census_curated_pretrain_pilot250k_p3_g16384.yaml \
  --selected-cells "${DATA}/selected_cells.parquet" \
  --vocabulary "${DATA}/gene_vocabulary.parquet" \
  --output-root "${DATA}/shards" \
  --cells-per-shard 25000 \
  --fetch-chunk-size 5000 \
  --resume
