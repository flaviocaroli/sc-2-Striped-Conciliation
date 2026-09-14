#!/bin/bash
#SBATCH --job-name=sc2-p3-val16k
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/3159436/sc2_p3/logs/validation_masking_v1/sc2-p3-val16k-%j.out
#SBATCH --error=/scratch/3159436/sc2_p3/logs/validation_masking_v1/sc2-p3-val16k-%j.err

set -Eeuo pipefail

exec "/home/3159436/.conda/envs/sc2-data/bin/python"   "/home/3159436/sc2/code/scripts/data/materialize_p3_16k_validation_masking.py"   --protocol "/home/3159436/sc2/code/configs/extension_2026_p3/p3_16k_validation_masking_protocol_v1.json"   --identity "/home/3159436/sc2/outputs/paper_extension_2026_p3/validation_masking_v1/identity/validation_identity_v1.tsv"   --output-root "/scratch/3159436/sc2_p3/data/census_curated_pretrain_pilot250k_g16384/benchmarks/validation_masking_v1"   --protocol-sha "60f67518cafd4be7b47732eec7a4779a8e617cb402f89109afabe29ea2b2f785"   --identity-sha "53d415b17cd45e6fb9b61f0f2069ded13df44ccbd1b810b2322a869341c1c38e"   --manifest-sha "a73839f78c26684bdc1d1f2a01543816a1549c0082af3d47955104dd94e8dcf8"
