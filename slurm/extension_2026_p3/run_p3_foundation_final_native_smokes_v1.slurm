#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/final_native_%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/final_native_%x_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

INPUT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/native_forward_input_v1/validation_first16_raw_counts.npz"

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/native_gpu_forward_v1"

: "${MODEL:?}"
: "${EXPECTED_JOB_HEAD:?}"
: "${EXPECTED_INPUT_SHA:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = \
    "${EXPECTED_JOB_HEAD}"

test -z "$(git status --porcelain=v1 -uall)"

test "$(
    sha256sum "${INPUT}" |
    awk '{print $1}'
)" = "${EXPECTED_INPUT_SHA}"

case "${MODEL}" in

    Geneformer)

        PY=/home/3159436/.conda/envs/sc2-p3-geneformer/bin/python

        SCRIPT="${REPO}/scripts/extension_2026_p3/run_geneformer_native_smoke_final_v1.py"

        SOURCE="${SC2}/external/foundation_models/p3_16k/sources/Geneformer"

        CKPT="${SC2}/external/foundation_models/p3_16k/checkpoints/Geneformer-V2-104M"

        GF_AUX="${SC2}/external/foundation_models/p3_16k/geneformer_aux_v1/geneformer"

        export GENEFORMER_GENE_MEDIAN_FILE="${GF_AUX}/gene_median_dictionary_gc104M.pkl"
        export GENEFORMER_TOKEN_DICTIONARY_FILE="${GF_AUX}/token_dictionary_gc104M.pkl"
        export GENEFORMER_GENE_MAPPING_FILE="${GF_AUX}/ensembl_mapping_dict_gc104M.pkl"

        ;;

    scPRINT)

        test "${ALLOW_SCPRINT:-0}" = "1"

        PY=/home/3159436/.conda/envs/sc2-p3-scprint/bin/python

        LAMIN=/home/3159436/.conda/envs/sc2-p3-scprint/bin/lamin

        SCRIPT="${REPO}/scripts/extension_2026_p3/run_scprint_native_smoke_final_v1.py"

        SOURCE="${SC2}/external/foundation_models/p3_16k/sources/scPRINT"

        CKPT="${SC2}/external/foundation_models/p3_16k/checkpoints/scPRINT/medium-v1.5.ckpt"

        "${LAMIN}" connect \
            anonymous/sc2p3scprintv1

        "${PY}" - <<'PY'
from scdataloader.utils import load_genes

human = load_genes(
    ["NCBITaxon:9606"]
)

mouse = load_genes(
    ["NCBITaxon:10090"]
)

assert len(human) == 85720
assert len(mouse) == 78009

print(
    "SCPRINT_RUNTIME_REGISTRY=PASS"
)
PY

        ;;

    *)

        echo "STOP: unsupported model ${MODEL}" >&2
        exit 1
        ;;
esac

test -x "${PY}"
test -s "${SCRIPT}"
test -e "${SOURCE}"
test -e "${CKPT}"

OUT="${OUTROOT}/${MODEL}"

test ! -e "${OUT}" || {
    echo "STOP: output exists ${OUT}" >&2
    exit 1
}

unset PYTHONHOME

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "P3 FOUNDATION FINAL NATIVE SMOKE"
echo "============================================================"
echo "MODEL=${MODEL}"
echo "JOB=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "SCIENTIFIC_RESULT=false"

nvidia-smi \
    --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv,noheader

"${PY}" "${SCRIPT}" \
    --input-bundle "${INPUT}" \
    --checkpoint "${CKPT}" \
    --source "${SOURCE}" \
    --output-dir "${OUT}"

test -s "${OUT}/smoke_receipt.json"

echo "MODEL=${MODEL}"
echo "P3_FOUNDATION_FINAL_NATIVE_SMOKE=PASS"
