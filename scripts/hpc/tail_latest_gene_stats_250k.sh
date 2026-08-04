#!/bin/bash
set -euo pipefail
ID_FILE=/home/3159436/sc2/logs/sc2-gstats-250k.latest_job_id
if [[ ! -s "${ID_FILE}" ]]; then
  echo "No saved gene-statistics job ID. Submit with scripts/hpc/submit_gene_stats_250k.sh" >&2
  exit 1
fi
JOB_ID=$(cat "${ID_FILE}")
echo "gene_stats_job=${JOB_ID}"
squeue -j "${JOB_ID}" || true
tail -f "/home/3159436/sc2/logs/sc2-gstats-250k_${JOB_ID}.out"
