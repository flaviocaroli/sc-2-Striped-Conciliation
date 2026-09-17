#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_7_production_home/%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_7_production_home/%x_%j.err

set -Eeuo pipefail
umask 027

REPO=/home/3159436/sc2/code
PY=/home/3159436/.conda/envs/sc2_mamba/bin/python
PREFIX=/home/3159436/.conda/envs/sc2_mamba
OVERLAY=/home/3159436/sc2/overlays/mamba22_git_torch25_cu124
MANIFEST=/home/3159436/sc2/data/census_curated_pretrain_pilot250k_g16384/train_shards.parquet
MEAN=/home/3159436/sc2/data/census_curated_pretrain_pilot250k_g16384/gene_mean_log1p.npy
PATHS_CONFIG=/home/3159436/sc2/code/configs/paths/continuous_hpc.yaml

: "${CONFIG:?CONFIG environment variable is required}"
: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD is required}"

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Exact environment layout proven by H200 preflight 661470.
export PYTHONPATH="${OVERLAY}:${REPO}/src"

SITE=$(
    "${PY}" - <<'PYENV'
import site

paths = site.getsitepackages()
assert paths
print(paths[0])
PYENV
)

TORCH_LIB="${SITE}/torch/lib"
NVIDIA_ROOT="${SITE}/nvidia"

test -d "${TORCH_LIB}"
test -d "${NVIDIA_ROOT}"

NVIDIA_LIB_PATHS=$(
    find "${NVIDIA_ROOT}"         -type d         -name lib         -printf '%p\n' |
    sort |
    paste -sd: -
)

test -n "${NVIDIA_LIB_PATHS}"

export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${TORCH_LIB}:${PREFIX}/lib"

cd "${REPO}"

ACTUAL_HEAD=$(git rev-parse HEAD)

echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HEAD=${ACTUAL_HEAD}"
echo "EXPECTED_JOB_HEAD=${EXPECTED_JOB_HEAD}"
echo "CONFIG=${CONFIG}"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

test "${ACTUAL_HEAD}" = "${EXPECTED_JOB_HEAD}"

"${PY}" -     "${CONFIG}"     "${MANIFEST}"     "${MEAN}"     "${REPO}" <<'PYGATE'
import ctypes
from pathlib import Path
import sys

ctypes.CDLL(
    "libcudart.so.12",
    mode=ctypes.RTLD_GLOBAL,
)

ctypes.CDLL(
    "libnvrtc.so.12",
    mode=ctypes.RTLD_GLOBAL,
)

import numpy as np
import torch
import yaml

import sc2
import mamba_ssm
import causal_conv1d
import causal_conv1d_cuda
import selective_scan_cuda

from sc2.data.shard_manifest import load_manifest

config_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
mean_path = Path(sys.argv[3])
repo = Path(sys.argv[4]).resolve()

assert torch.cuda.is_available()

gpu = torch.cuda.get_device_name(0)

assert "H200" in gpu.upper(), gpu
assert torch.__version__ == "2.5.0+cu124"
assert mamba_ssm.__version__ == "2.2.4"
assert causal_conv1d.__version__ == "1.4.0"

assert str(Path(sc2.__file__).resolve()).startswith(
    str(repo / "src" / "sc2")
)

cfg = yaml.safe_load(config_path.read_text())

assert cfg["data"]["manifest"] == str(manifest_path)
assert cfg["data"]["gene_mean_log1p"] == str(mean_path)

batch = int(cfg["train"]["batch_size"])
accum = int(cfg["train"]["grad_accum_steps"])
steps = int(cfg["train"]["total_steps"])

assert steps == 5000
assert batch * accum == 32

records, _ = load_manifest(manifest_path)

assert len(records) == 8
assert all(r.split == "train" for r in records)
assert all(r.modality == "sc" for r in records)
assert all(int(r.n_genes) == 16384 for r in records)
assert sum(int(r.n_rows) for r in records) == 200000

for record in records:
    assert str(record.path).startswith("/home/3159436/")
    assert Path(record.path).is_dir()

mean = np.load(mean_path)

assert mean.shape == (16384,)
assert mean.dtype == np.float32
assert np.isfinite(mean).all()

print("GPU =", gpu)
print("RUN =", cfg["run_name"])
print("TORCH =", torch.__version__)
print("MAMBA_SSM =", mamba_ssm.__version__)
print("CAUSAL_CONV1D =", causal_conv1d.__version__)
print("BATCH =", batch)
print("ACCUM =", accum)
print("EFFECTIVE_BATCH =", batch * accum)
print("TOTAL_STEPS =", steps)
print("CUDA_RUNTIME_LIBRARIES=PASS")
print("FUSED_EXTENSIONS=PASS")
print("TRAIN_ONLY_RUNTIME_GATE=PASS")
PYGATE

START=$(date +%s)

"${PY}" -u     -m sc2.train.train_sc2_striped_full_continuous     --config "${CONFIG}"     --paths "${PATHS_CONFIG}"     --fresh

END=$(date +%s)
ELAPSED=$((END - START))

echo "PRODUCTION_WALL_SECONDS=${ELAPSED}"
echo "P3_PRODUCTION_JOB=PASS"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
