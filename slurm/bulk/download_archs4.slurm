#!/bin/bash
#SBATCH --job-name=dl_archs4
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail

mkdir -p /home/3159436/sc2/data/raw/archs4
cd /home/3159436/sc2/data/raw/archs4

wget -c https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.5.h5 -O human_gene_v2.5.h5