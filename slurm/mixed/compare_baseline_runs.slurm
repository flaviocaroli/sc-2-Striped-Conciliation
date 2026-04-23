#!/bin/bash
#SBATCH --job-name=compare_runs
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate sc2
set -u

cd /home/3159436/sc2/code

python -m sc2.eval.compare_baseline_runs \
  --sc-baseline-metrics /home/3159436/sc2/outputs/census_autoencoder_multids_lung/metrics.json \
  --bulk-baseline-metrics /home/3159436/sc2/outputs/archs4_denoising_autoencoder_pilot/metrics.json \
  --mixed-overall-csv /home/3159436/sc2/outputs/evals/bulk_sc_mixed_autoencoder_pilot_eval/overall_by_modality_split.csv \
  --output-csv /home/3159436/sc2/outputs/evals/bulk_sc_mixed_autoencoder_pilot_eval/baseline_comparison.csv \
  --output-json /home/3159436/sc2/outputs/evals/bulk_sc_mixed_autoencoder_pilot_eval/baseline_comparison.json