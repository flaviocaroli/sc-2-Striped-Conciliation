#!/bin/bash
set -euo pipefail

DATA_ROOT="${SC2_DATA_ROOT:-/home/3159436/sc2/data}"
GTEX_RAW_DIR="${DATA_ROOT}/raw/gtex"
mkdir -p "${GTEX_RAW_DIR}"

SAMPLE_ATTR_URL="https://storage.googleapis.com/adult-gtex/annotations/v11/metadata-files/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt"
EXPR_URL="https://storage.googleapis.com/adult-gtex/bulk-gex/v11/rna-seq/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.gct.gz"

SAMPLE_ATTR_FILE="${GTEX_RAW_DIR}/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt"
EXPR_FILE="${GTEX_RAW_DIR}/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.gct.gz"
OUTPUT_H5AD="${GTEX_RAW_DIR}/gtex_lung.h5ad"

echo "Downloading GTEx V11 sample attributes..."
curl -L --fail --retry 3 --retry-delay 5 -o "${SAMPLE_ATTR_FILE}" "${SAMPLE_ATTR_URL}"

echo "Downloading GTEx V11 gene TPM matrix..."
curl -L --fail --retry 3 --retry-delay 5 -o "${EXPR_FILE}" "${EXPR_URL}"

echo "Downloaded files:"
ls -lh "${SAMPLE_ATTR_FILE}" "${EXPR_FILE}"

echo "Building GTEx lung h5ad..."
python preprocess/gtex/build_gtex_lung_h5ad.py \
  --expression-tsv "${EXPR_FILE}" \
  --sample-attributes-tsv "${SAMPLE_ATTR_FILE}" \
  --output-h5ad "${OUTPUT_H5AD}"

echo "Done."
ls -lh "${OUTPUT_H5AD}"