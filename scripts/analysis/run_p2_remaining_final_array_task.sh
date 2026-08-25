#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?}"
: "${P2_REMAINING_FINAL_COMMIT:?}"

TASK="${SLURM_ARRAY_TASK_ID}"

if ! [[ "${TASK}" =~ ^[0-9]+$ ]]; then
    echo "Invalid task=${TASK}" >&2
    exit 2
fi

if (( TASK < 0 || TASK > 14 )); then
    echo "Task outside 0..14: ${TASK}" >&2
    exit 2
fi

METHODS=(
    positive_train_mean
    positive_train_median
    truncated_low_rank
    knn
    alra
)

DATASETS=(
    internal_test
    zheng68k
    baron_pancreas
)

METHOD_INDEX=$(( TASK / 3 ))
DATASET_INDEX=$(( TASK % 3 ))

METHOD="${METHODS[$METHOD_INDEX]}"
DATASET="${DATASETS[$DATASET_INDEX]}"

if [[ "${P2_REMAINING_DRY_RUN:-0}" == "1" ]]; then
    printf \
      'TASK_MAPPING task=%s method=%s dataset=%s\n' \
      "${TASK}" \
      "${METHOD}" \
      "${DATASET}"
    exit 0
fi

SC2_ROOT=/home/3159436/sc2
REPO=${SC2_ROOT}/code
DATA=${SC2_ROOT}/data
OUTPUTS=${SC2_ROOT}/outputs

PY=/home/3159436/.conda/envs/sc2-data/bin/python

CORPUS=${DATA}/census_curated_pretrain_pilot250k

PROTOCOL=${REPO}/configs/eval/striped/sc2_p2_comparator_protocol.json
PRETEST=${REPO}/configs/eval/striped/sc2_p2_pretest_selection_receipt.json

TRAIN_STATS=${OUTPUTS}/analysis/p2_train_positive_stats_v1/positive_train_statistics.npz

ALRA_ROOT=${SC2_ROOT}/tools/p2-alra
ALRA_SOURCE=${ALRA_ROOT}/src/ALRA
ALRA_RLIB=${ALRA_ROOT}/Rlib
ALRA_R_SCRIPT=${REPO}/scripts/analysis/run_p2_alra.R
RSCRIPT=/usr/bin/Rscript

FINAL_ROOT=${OUTPUTS}/evals/p2_remaining_comparators_final_v1

case "${METHOD}" in

    positive_train_mean|positive_train_median)
        SELECTION_RECEIPT=${REPO}/configs/eval/striped/sc2_p2_baseline_validation_receipt.json
        ;;

    truncated_low_rank)
        SELECTION_RECEIPT=${REPO}/configs/eval/striped/sc2_p2_low_rank_validation_receipt.json
        ;;

    knn)
        SELECTION_RECEIPT=${REPO}/configs/eval/striped/sc2_p2_knn_validation_receipt.json
        ;;

    alra)
        SELECTION_RECEIPT=${REPO}/configs/eval/striped/sc2_p2_alra_validation_receipt.json
        ;;

    *)
        exit 3
        ;;
esac

cd "${REPO}"

test "$(git rev-parse HEAD)" = \
     "${P2_REMAINING_FINAL_COMMIT}"

test "$(git rev-parse origin/p1-official-mamba)" = \
     "${P2_REMAINING_FINAL_COMMIT}"

FINAL_DIR="${FINAL_ROOT}/${METHOD}/${DATASET}"
TASK_TMP="${FINAL_ROOT}/.task_${TASK}_${METHOD}_${DATASET}.tmp"

test ! -e "${FINAL_DIR}"
test ! -e "${TASK_TMP}"

mkdir -p "${TASK_TMP}"

cleanup_on_error () {
    rc=$?

    if (( rc != 0 )); then
        echo \
          "TASK_FAILED task=${TASK} method=${METHOD} dataset=${DATASET} tmp=${TASK_TMP}" \
          >&2
    fi

    exit "${rc}"
}

trap cleanup_on_error EXIT

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src:${REPO}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

for MASK in 15 30 50; do

    case "${DATASET}" in

        internal_test)
            BENCHMARK="${CORPUS}/benchmarks/test_mask${MASK}.npz"

            case "${MASK}" in
                15)
                    BENCH_SHA=eb9d3d63f42bc30fb5233f0f39271f792b43535a6be4f46b690cc188ed99c3d7
                    ;;
                30)
                    BENCH_SHA=ca59d9407830484360c476865c1b339b0ed62b8c4e0a75e8ca5f07acbc1d7ecf
                    ;;
                50)
                    BENCH_SHA=e1a75d8b27eda99ccf0ba08857fad385708ca0dd5cbdc0166ded9f1607b7a4d2
                    ;;
            esac
            ;;

        zheng68k)
            BENCHMARK="${DATA}/external_benchmarks/zheng68k_v1/mask${MASK}.npz"

            case "${MASK}" in
                15)
                    BENCH_SHA=35490be0d1a83b46517d2d5792a2c0a1bc9c5b3f800f29aeb90447d11d8a6ecc
                    ;;
                30)
                    BENCH_SHA=ba6b8e6d2698048faac52e6b384a795bd429ec3d58bb385b1fc769ad50da7871
                    ;;
                50)
                    BENCH_SHA=2313651071d4c2d69053a51e21ba477d3f2c3c146172dc187ac809af4590f040
                    ;;
            esac
            ;;

        baron_pancreas)
            BENCHMARK="${DATA}/external_benchmarks/baron_pancreas_v1/mask${MASK}.npz"

            case "${MASK}" in
                15)
                    BENCH_SHA=20dd3ef3454c91fad9ef121518827df4b64ced1adb5d1b405e38ab0f6acc498b
                    ;;
                30)
                    BENCH_SHA=edc1422a764c1aa40a9028401df6629c771fc7f3809ff5e9a7443be6fabd1e3e
                    ;;
                50)
                    BENCH_SHA=7654fcc78719327f73fcbe46bf69beaf866e32b9331134f2aec1bc4964b2d792
                    ;;
            esac
            ;;

        *)
            exit 4
            ;;
    esac

    OUT="${TASK_TMP}/mask${MASK}"

    "${PY}" \
        -u \
        -m scripts.analysis.evaluate_p2_remaining_final \
        --method "${METHOD}" \
        --dataset "${DATASET}" \
        --mask-percent "${MASK}" \
        --benchmark "${BENCHMARK}" \
        --benchmark-sha256 "${BENCH_SHA}" \
        --protocol "${PROTOCOL}" \
        --pretest-receipt "${PRETEST}" \
        --selection-receipt "${SELECTION_RECEIPT}" \
        --train-stats "${TRAIN_STATS}" \
        --alra-r-script "${ALRA_R_SCRIPT}" \
        --alra-rlib "${ALRA_RLIB}" \
        --alra-source "${ALRA_SOURCE}" \
        --rscript "${RSCRIPT}" \
        --output-dir "${OUT}"

    (
        cd "${OUT}"
        sha256sum -c SHA256SUMS.txt >/dev/null
    )

done

mkdir -p \
    "${FINAL_ROOT}/${METHOD}"

mv \
    "${TASK_TMP}" \
    "${FINAL_DIR}"

trap - EXIT

echo \
  "P2_REMAINING_FINAL_TASK=PASS task=${TASK} method=${METHOD} dataset=${DATASET}"
