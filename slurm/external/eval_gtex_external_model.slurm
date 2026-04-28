#!/bin/bash
#SBATCH --job-name=gtex_eval
#SBATCH --partition=medium_gpu
#SBATCH --gres=gpu:GA100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -eo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: sbatch $0 <config_path>"
  exit 1
fi

CONFIG_PATH="$1"

eval "$(conda shell.bash hook)"
conda activate sc2
set -u

export SC2_DATA_ROOT=/home/3159436/sc2/data
export SC2_OUTPUT_ROOT=/home/3159436/sc2/outputs

cd /home/3159436/sc2/code

python -m sc2.eval.evaluate_bulk_model_on_gtex \
  --config "${CONFIG_PATH}" \
  --paths configs/paths/hpc.yaml