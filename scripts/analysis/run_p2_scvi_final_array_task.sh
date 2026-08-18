#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?}"
: "${P2_SCVI_FINAL_PRERESULT_COMMIT:?}"

TASK="${SLURM_ARRAY_TASK_ID}"

if ! [[ "${TASK}" =~ ^[0-9]+$ ]]; then
    echo "Invalid noninteger task: ${TASK}" >&2
    exit 2
fi

if (( TASK < 0 || TASK > 26 )); then
    echo "Invalid array task: ${TASK}" >&2
    exit 2
fi

DATASETS=(
    internal_test
    zheng68k
    baron_pancreas
)

SEEDS=(
    20260728
    20260729
    20260730
)

MASKS=(
    15
    30
    50
)

DATASET_INDEX=$(( TASK / 9 ))
REM=$(( TASK % 9 ))
SEED_INDEX=$(( REM / 3 ))
MASK_INDEX=$(( REM % 3 ))

DATASET="${DATASETS[$DATASET_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
MASK="${MASKS[$MASK_INDEX]}"

if [[ "${P2_SCVI_ARRAY_DRY_RUN:-0}" == "1" ]]; then
    printf \
        'ARRAY_MAPPING task=%s dataset=%s seed=%s mask=%s\n' \
        "${TASK}" \
        "${DATASET}" \
        "${SEED}" \
        "${MASK}"
    exit 0
fi

SC2_ROOT=/home/3159436/sc2
REPO=${SC2_ROOT}/code
DATA=${SC2_ROOT}/data
OUTPUTS=${SC2_ROOT}/outputs

SCVI_PY=/home/3159436/.conda/envs/sc2-p2-scvi/bin/python

CORPUS=${DATA}/census_curated_pretrain_pilot250k

PROTOCOL=${REPO}/configs/eval/striped/sc2_p2_comparator_protocol.json
PRETEST=${REPO}/configs/eval/striped/sc2_p2_pretest_selection_receipt.json
SELECTION=${REPO}/configs/eval/striped/sc2_p2_scvi_validation_selection_receipt.json
COUNT_RECEIPT=${REPO}/configs/eval/striped/sc2_p2_scvi_final_count_view_receipt.json

COUNT_ROOT=${OUTPUTS}/analysis/p2_scvi_final_count_views_v1
FINAL_ROOT=${OUTPUTS}/evals/p2_scvi_final_v1

case "${DATASET}" in
    internal_test)
        BENCHMARK="${CORPUS}/benchmarks/test_mask${MASK}.npz"
        ;;
    zheng68k)
        BENCHMARK="${DATA}/external_benchmarks/zheng68k_v1/mask${MASK}.npz"
        ;;
    baron_pancreas)
        BENCHMARK="${DATA}/external_benchmarks/baron_pancreas_v1/mask${MASK}.npz"
        ;;
    *)
        echo "Unexpected dataset=${DATASET}" >&2
        exit 3
        ;;
esac

OUTDIR="${FINAL_ROOT}/${DATASET}/seed${SEED}/mask${MASK}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = \
     "${P2_SCVI_FINAL_PRERESULT_COMMIT}"

test "$(git rev-parse origin/p1-official-mamba)" = \
     "${P2_SCVI_FINAL_PRERESULT_COMMIT}"

test -f "${COUNT_RECEIPT}"
test -f "${BENCHMARK}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src:${REPO}"

"${SCVI_PY}" \
    -u \
    -m scripts.analysis.evaluate_p2_scvi_final \
    --dataset "${DATASET}" \
    --benchmark "${BENCHMARK}" \
    --mask-percent "${MASK}" \
    --seed "${SEED}" \
    --protocol "${PROTOCOL}" \
    --pretest-receipt "${PRETEST}" \
    --scvi-selection-receipt "${SELECTION}" \
    --count-view-receipt "${COUNT_RECEIPT}" \
    --count-view-root "${COUNT_ROOT}" \
    --output-dir "${OUTDIR}"
