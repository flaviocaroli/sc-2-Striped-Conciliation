#!/bin/bash
#SBATCH --job-name=sc2-p3-mean16k
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/3159436/sc2_p3/logs/gene_mean16k/sc2-p3-mean16k-%j.out
#SBATCH --error=/scratch/3159436/sc2_p3/logs/gene_mean16k/sc2-p3-mean16k-%j.err

set -Eeuo pipefail
umask 027

SC2_ROOT=/home/3159436/sc2
REPO="${SC2_ROOT}/code"

P3=/scratch/3159436/sc2_p3/data/census_curated_pretrain_pilot250k_g16384

PY=/home/3159436/.conda/envs/sc2-data/bin/python

MANIFEST="${P3}/shards.parquet"
OUTPUT="${P3}/gene_mean_log1p.npy"
BUILDER="${REPO}/scripts/data/compute_gene_mean_from_shards.py"

EXPECTED_MANIFEST_SHA=a73839f78c26684bdc1d1f2a01543816a1549c0082af3d47955104dd94e8dcf8

sha() {
    sha256sum "$1" | awk '{print $1}'
}

echo "============================================================"
echo "SC2 P3 16K TRAIN-ONLY GENE MEAN"
echo "============================================================"
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "MANIFEST=${MANIFEST}"
echo "OUTPUT=${OUTPUT}"
echo "GIT_HEAD=$(git -C "${REPO}" rev-parse HEAD)"

test -f "${MANIFEST}"
test "$(sha "${MANIFEST}")" = "${EXPECTED_MANIFEST_SHA}"
test -f "${BUILDER}"
test ! -e "${OUTPUT}"

echo
echo "============================================================"
echo "ENVIRONMENT"
echo "============================================================"

env \
    -u PYTHONPATH \
    -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 \
    "${PY}" - <<'PY'
import sys
import numpy
import scipy
import pandas

print("PYTHON =", sys.executable)
print("NUMPY =", numpy.__version__, numpy.__file__)
print("SCIPY =", scipy.__version__, scipy.__file__)
print("PANDAS =", pandas.__version__, pandas.__file__)
PY

echo
echo "============================================================"
echo "BUILD"
echo "============================================================"

env \
    -u PYTHONPATH \
    -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${REPO}/src" \
    "${PY}" \
    "${BUILDER}" \
    --manifest "${MANIFEST}" \
    --output "${OUTPUT}" \
    --modality sc

echo
echo "============================================================"
echo "POST-BUILD QC"
echo "============================================================"

env \
    -u PYTHONPATH \
    -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 \
    "${PY}" - <<'PY'
from pathlib import Path
import numpy as np

path = Path(
    "/scratch/3159436/sc2_p3/data/"
    "census_curated_pretrain_pilot250k_g16384/"
    "gene_mean_log1p.npy"
)

x = np.load(path, mmap_mode="r")

assert x.shape == (16384,), x.shape
assert x.dtype == np.float32, x.dtype
assert bool(np.isfinite(x).all())
assert bool((x >= 0).all())

print("GENE_MEAN_SHAPE =", x.shape)
print("GENE_MEAN_DTYPE =", x.dtype)
print("GENE_MEAN_FINITE =", bool(np.isfinite(x).all()))
print("GENE_MEAN_NONNEGATIVE =", bool((x >= 0).all()))
print("GENE_MEAN_MIN =", float(x.min()))
print("GENE_MEAN_MAX =", float(x.max()))
print("GENE_MEAN_MEAN =", float(x.mean()))
print("GENE_MEAN_STD =", float(x.std()))
print("GENE_MEAN_QC=PASS")
PY

echo
echo "GENE_MEAN_SHA256=$(sha "${OUTPUT}")"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "P3_16K_GENE_MEAN_JOB=PASS"
