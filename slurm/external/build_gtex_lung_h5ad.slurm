#!/bin/bash
#SBATCH --job-name=gtex_build
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate sc2
set -u

export SC2_DATA_ROOT=/home/3159436/sc2/data

cd /home/3159436/sc2/code

python preprocess/gtex/build_gtex_lung_h5ad.py \
  --expression-tsv /home/3159436/sc2/data/raw/gtex/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.gct.gz \
  --sample-attributes-tsv /home/3159436/sc2/data/raw/gtex/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt \
  --output-h5ad /home/3159436/sc2/data/raw/gtex/gtex_lung.h5ad