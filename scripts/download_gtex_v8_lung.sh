#!/bin/bash
set -euo pipefail

DATA_ROOT="${SC2_DATA_ROOT:-/home/3159436/sc2/data}"
GTEX_RAW_DIR="${DATA_ROOT}/raw/gtex"
mkdir -p "${GTEX_RAW_DIR}"

SAMPLE_ATTR_FILE="${GTEX_RAW_DIR}/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
EXPR_FILE="${GTEX_RAW_DIR}/GTEx_Analysis_v10_gene_tpm_non_lcm.gct.gz"
OUTPUT_H5AD="${GTEX_RAW_DIR}/gtex_lung.h5ad"

# Current public GTEx bucket patterns inferred from Adult GTEx V10 public collections.
SAMPLE_ATTR_CANDIDATES=(
  "https://storage.googleapis.com/adult-gtex/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
)

EXPR_CANDIDATES=(
  "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm_non_lcm.gct.gz"
  "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_2022-06-06_v10_RNASeQCv2.4.2_gene_tpm_non_lcm.gct.gz"
  "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm_non_lcm.gct"
  "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_2022-06-06_v10_RNASeQCv2.4.2_gene_tpm_non_lcm.gct"
)

pick_working_url() {
  local kind="$1"
  shift
  for url in "$@"; do
    echo "Checking ${kind}: ${url}"
    if curl -I -L --fail --silent --show-error "${url}" >/dev/null; then
      echo "${url}"
      return 0
    fi
  done
  return 1
}

echo "Resolving GTEx sample-attributes URL..."
SAMPLE_ATTR_URL="$(pick_working_url sample_attributes "${SAMPLE_ATTR_CANDIDATES[@]}")" || {
  echo "Could not find a working GTEx sample-attributes URL."
  exit 1
}
echo "Using sample attributes: ${SAMPLE_ATTR_URL}"

echo "Resolving GTEx expression URL..."
EXPR_URL="$(pick_working_url expression_matrix "${EXPR_CANDIDATES[@]}")" || {
  echo "Could not find a working GTEx expression URL."
  echo "Tried:"
  printf '  %s\n' "${EXPR_CANDIDATES[@]}"
  exit 1
}
echo "Using expression matrix: ${EXPR_URL}"

echo "Downloading GTEx sample attributes..."
curl -L --fail --retry 3 --retry-delay 5 -o "${SAMPLE_ATTR_FILE}" "${SAMPLE_ATTR_URL}"

echo "Downloading GTEx expression matrix..."
curl -L --fail --retry 3 --retry-delay 5 -o "${EXPR_FILE}" "${EXPR_URL}"

echo "Downloaded files:"
ls -lh "${SAMPLE_ATTR_FILE}" "${EXPR_FILE}"

echo "Building GTEx lung h5ad..."
python preprocess/gtex/build_gtex_lung_h5ad.py \
  --expression-tsv "${EXPR_FILE}" \
  --sample-attributes-tsv "${SAMPLE_ATTR_FILE}" \
  --output-h5ad "${OUTPUT_H5AD}"

echo "Built GTEx lung h5ad:"
ls -lh "${OUTPUT_H5AD}"