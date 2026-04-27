#!/bin/bash
#SBATCH --job-name=archs4_lgevl
#SBATCH --partition=medium_gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate sc2
set -u

export SC2_DATA_ROOT=/home/3159436/sc2/data
export SC2_OUTPUT_ROOT=/home/3159436/sc2/outputs

cd /home/3159436/sc2/code

python -m sc2.eval.evaluate_archs4_denoising_autoencoder \
  --config configs/eval/bulk/archs4_lung_large_denoising_autoencoder_eval.yaml \
  --paths configs/paths/hpc.yaml