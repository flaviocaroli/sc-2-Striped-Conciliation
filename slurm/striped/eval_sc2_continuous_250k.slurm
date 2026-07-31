#!/bin/bash
#SBATCH --job-name=sc2-eval-250k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/3159436/sc2/logs/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc2
cd /home/3159436/sc2/code
export PYTHONPATH=/home/3159436/sc2/code/src:${PYTHONPATH:-}

DATA=/home/3159436/sc2/data/census_curated_pretrain_pilot250k
CHECKPOINT=${SC2_CHECKPOINT:-/home/3159436/sc2/outputs/sc2_continuous_250k_sc_only_v1/checkpoints/last.pt}
VALIDATION_NPZ=${SC2_VALIDATION_NPZ:-${DATA}/benchmarks/validation_mask15.npz}
TEST_NPZ=${SC2_TEST_NPZ:-${DATA}/benchmarks/test_mask15.npz}
VALIDATION_OUT=${SC2_VALIDATION_OUT:-/home/3159436/sc2/outputs/evals/sc2_continuous_250k_validation}
TEST_OUT=${SC2_TEST_OUT:-/home/3159436/sc2/outputs/evals/sc2_continuous_250k_test}

python -m sc2.eval.evaluate_continuous_repair \
  --config configs/eval/striped/sc2_continuous_eval.yaml \
  --checkpoint "${CHECKPOINT}" \
  --benchmark "${VALIDATION_NPZ}" \
  --output-dir "${VALIDATION_OUT}"

THRESHOLD=$(python - "${VALIDATION_OUT}/summary.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))['threshold'])
PY
)

python -m sc2.eval.evaluate_continuous_repair \
  --config configs/eval/striped/sc2_continuous_eval.yaml \
  --checkpoint "${CHECKPOINT}" \
  --benchmark "${TEST_NPZ}" \
  --output-dir "${TEST_OUT}" \
  --threshold "${THRESHOLD}"
