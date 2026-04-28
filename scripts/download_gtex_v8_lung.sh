#!/bin/bash
set -euo pipefail

DATA_ROOT="${SC2_DATA_ROOT:-/home/3159436/sc2/data}"
GTEX_RAW_DIR="${DATA_ROOT}/raw/gtex"
mkdir -p "${GTEX_RAW_DIR}"

# These can be overridden at submit time if GTEx changes the paths again.
SAMPLE_ATTR_URL="${SAMPLE_ATTR_URL:-https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt}"
EXPR_URL="${EXPR_URL:-https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz}"

SAMPLE_ATTR_FILE="${GTEX_RAW_DIR}/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
EXPR_FILE="${GTEX_RAW_DIR}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
OUTPUT_H5AD="${GTEX_RAW_DIR}/gtex_lung.h5ad"

echo "Checking GTEx URLs..."
curl -I -L --fail "${SAMPLE_ATTR_URL}" | head
curl -I -L --fail "${EXPR_URL}" | head

echo "Downloading GTEx sample attributes..."
curl -L --fail --retry 3 --retry-delay 5 -o "${SAMPLE_ATTR_FILE}" "${SAMPLE_ATTR_URL}"

echo "Downloading GTEx gene TPM matrix..."
curl -L --fail --retry 3 --retry-delay 5 -o "${EXPR_FILE}" "${EXPR_URL}"

echo "Files downloaded:"
ls -lh "${SAMPLE_ATTR_FILE}" "${EXPR_FILE}"

echo "Building GTEx lung h5ad..."
python preprocess/gtex/build_gtex_lung_h5ad.py \
  --expression-tsv "${EXPR_FILE}" \
  --sample-attributes-tsv "${SAMPLE_ATTR_FILE}" \
  --output-h5ad "${OUTPUT_H5AD}"

echo "Done."
ls -lh "${OUTPUT_H5AD}"