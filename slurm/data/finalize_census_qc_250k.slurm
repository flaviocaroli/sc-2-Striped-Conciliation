#!/bin/bash
#SBATCH --job-name=sc2-qc-250k
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2-data
cd /home/3159436/sc2/code
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}
DATA=/home/3159436/sc2/data/census_curated_pretrain_pilot250k

python scripts/data/finalize_census_plan_qc.py \
  --config configs/data/census_curated_pretrain_pilot250k.yaml \
  --planned-cells "${DATA}/plan/planned_cells.parquet" \
  --output "${DATA}/selected_cells.parquet" \
  --qc-output "${DATA}/candidate_cell_qc.parquet" \
  --chunk-size 10000
