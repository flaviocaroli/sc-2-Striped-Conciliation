#!/bin/bash
set -euo pipefail

cd /home/3159436/sc2/code

RUN_DIR=/home/3159436/sc2/outputs/sc2_full20x4_hurdle_pilot
mkdir -p "${RUN_DIR}"
JOB_RECORD="${RUN_DIR}/evaluation_jobs_$(date +%Y%m%d_%H%M%S).txt"

for TAG in best_any last; do
    INTERNAL_JOB=$(sbatch --parsable \
      -p medium_gpuh200 \
      --gres=gpu:1 \
      --time=06:00:00 \
      slurm/striped/eval_sc2_striped_full.slurm \
      configs/eval/striped/sc2_full20x4_hurdle_${TAG}_eval.yaml)

    MASKED_JOB=$(sbatch --parsable \
      -p medium_gpuh200 \
      --gres=gpu:1 \
      --time=06:00:00 \
      slurm/benchmarks/eval_sc2_striped_full_masked.slurm \
      configs/benchmarks/sc_reconstruction/sc2_full20x4_hurdle_${TAG}_masked.yaml)

    DROPOUT_VAL_JOB=$(sbatch --parsable \
      -p medium_gpuh200 \
      --gres=gpu:1 \
      --time=06:00:00 \
      slurm/benchmarks/eval_sc2_striped_full_dropout.slurm \
      configs/benchmarks/sc_reconstruction/sc2_full20x4_hurdle_${TAG}_dropout_val.yaml)

    DROPOUT_TEST_JOB=$(sbatch --parsable \
      --dependency=afterok:${DROPOUT_VAL_JOB} \
      -p medium_gpuh200 \
      --gres=gpu:1 \
      --time=06:00:00 \
      slurm/benchmarks/eval_sc2_striped_full_dropout.slurm \
      configs/benchmarks/sc_reconstruction/sc2_full20x4_hurdle_${TAG}_dropout_test.yaml)

    LINE="tag=${TAG} internal=${INTERNAL_JOB} masked=${MASKED_JOB} dropout_val=${DROPOUT_VAL_JOB} dropout_test=${DROPOUT_TEST_JOB}"
    echo "${LINE}" | tee -a "${JOB_RECORD}"
done

echo "job_record=${JOB_RECORD}"
