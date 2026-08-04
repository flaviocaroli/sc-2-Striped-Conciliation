#!/bin/bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2-data
cd /home/3159436/sc2/code
mkdir -p /home/3159436/sc2/logs
JOB_ID=$(sbatch --parsable slurm/data/gene_stats_census_250k.slurm)
echo "${JOB_ID}" > /home/3159436/sc2/logs/sc2-gstats-250k.latest_job_id
echo "gene_stats_job=${JOB_ID}"
echo "stdout=/home/3159436/sc2/logs/sc2-gstats-250k_${JOB_ID}.out"
echo "stderr=/home/3159436/sc2/logs/sc2-gstats-250k_${JOB_ID}.err"
