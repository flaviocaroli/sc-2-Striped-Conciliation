#!/bin/bash
#SBATCH --job-name=sc2-plan250k
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2-data

export REPO=/home/3159436/sc2/code
export DATA=/home/3159436/sc2/data/census_curated_pretrain_pilot250k
export CACHE=${DATA}/registry_review/census_datasets_2025-01-30.parquet
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

cd "${REPO}"
mkdir -p /home/3159436/sc2/logs

echo "start_time=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "cache=${CACHE}"

python scripts/data/build_census_corpus.py \
  --config configs/data/census_curated_pretrain_pilot250k.yaml \
  --data-root /home/3159436/sc2/data \
  --datasets-cache "${CACHE}" \
  --plan-only

echo "end_time=$(date --iso-8601=seconds)"
