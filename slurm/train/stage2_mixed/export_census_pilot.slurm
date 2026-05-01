#!/bin/bash
#SBATCH --job-name=export_census_pilot
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail

source ~/.bashrc
conda activate sc2

export SC2_DATA_ROOT=/home/3159436/sc2/data

cd /home/3159436/sc2/code
python preprocess/export_census_pilot.py