#!/bin/bash
#SBATCH --job-name=p3fm-env-repair-v3
#SBATCH --partition=compute
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/repair_v3_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/repair_v3_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

ENVROOT=/home/3159436/.conda/envs
BASE="${ENVROOT}/sc2_mamba"

SRCROOT="${SC2}/external/foundation_models/p3_16k/sources"
CKPTROOT="${SC2}/external/foundation_models/p3_16k/checkpoints"

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/environments"

: "${EXPECTED_JOB_HEAD:?}"
: "${CONDA_BIN:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test -x "${BASE}/bin/python"

export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Keep package/build caches off HOME.
export PIP_CACHE_DIR="/tmp/sc2-pip-${SLURM_JOB_ID}"
export CONDA_PKGS_DIRS="/tmp/sc2-conda-${SLURM_JOB_ID}"

mkdir -p \
    "${PIP_CACHE_DIR}" \
    "${CONDA_PKGS_DIRS}"

cleanup() {
    rm -rf \
        "${PIP_CACHE_DIR}" \
        "${CONDA_PKGS_DIRS}"
}

trap cleanup EXIT


valid_env() {
    ENV="$1"

    test -x "${ENV}/bin/python" || return 1

    "${ENV}/bin/python" - <<'PY' >/dev/null 2>&1
import sys
import torch
import numpy

assert sys.version_info[:2] == (3, 10)
PY
}


ensure_env() {
    ENV="$1"

    if valid_env "${ENV}"; then
        echo "SALVAGE_ENV=${ENV}"
        return
    fi

    echo "RECREATE_ENV=${ENV}"

    rm -rf -- "${ENV}"

    "${CONDA_BIN}" create \
        --yes \
        --prefix "${ENV}" \
        --clone "${BASE}"

    valid_env "${ENV}"
}


receipt() {
    MODEL="$1"
    ENV="$2"
    SOURCE="$3"
    ASSET="$4"

    OUT="${OUTROOT}/${MODEL}"

    mkdir -p "${OUT}"

    PYTHONPATH="${REPO}/src:${SOURCE}:${ASSET}" \
    "${ENV}/bin/python" - \
        "${MODEL}" \
        "${ENV}" \
        "${SOURCE}" \
        "${ASSET}" \
        "${OUT}/environment_receipt_v3.json" \
        "${SLURM_JOB_ID}" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import importlib
import json
import subprocess
import sys

model = sys.argv[1]
env = Path(sys.argv[2])
source = Path(sys.argv[3])
asset = Path(sys.argv[4])
destination = Path(sys.argv[5])
job = int(sys.argv[6])

versions = {}

for name in (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "sc2",
):
    m = importlib.import_module(name)

    versions[name] = getattr(
        m,
        "__version__",
        None,
    )

    print(
        "IMPORT_PASS",
        model,
        name,
        versions[name],
    )

if model == "Geneformer":

    m = importlib.import_module("geneformer")

    external = {
        "geneformer":
            getattr(m, "__version__", None),
    }

elif model == "GeneMamba":

    importlib.import_module(
        "configuration_genemamba"
    )

    importlib.import_module(
        "modeling_genemamba"
    )

    m = importlib.import_module(
        "mamba_ssm"
    )

    external = {
        "mamba_ssm":
            getattr(m, "__version__", None),
    }

elif model == "scPRINT":

    m = importlib.import_module("scprint")

    external = {
        "scprint":
            getattr(m, "__version__", None),
    }

else:
    raise ValueError(model)

payload = {
    "schema":
        "sc2-p3-foundation-environment-receipt-v3",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "model":
        model,

    "slurm_job_id":
        job,

    "environment":
        str(env),

    "source":
        str(source),

    "asset":
        str(asset),

    "versions":
        versions,

    "external":
        external,

    "scientific_inference":
        False,

    "pip_freeze":
        subprocess.check_output(
            [
                str(env / "bin/python"),
                "-m",
                "pip",
                "freeze",
            ],
            text=True,
        ).splitlines(),

    "status":
        "PASS",
}

destination.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print(
    "ENV_RECEIPT=",
    destination,
)

print(
    "P3_FOUNDATION_ENVIRONMENT=PASS",
    model,
)
PY
}


echo "============================================================"
echo "1. GENEFORMER"
echo "============================================================"

GF_ENV="${ENVROOT}/sc2-p3-geneformer"
GF_SRC="${SRCROOT}/Geneformer"
GF_ASSET="${CKPTROOT}/Geneformer-V2-104M"

ensure_env "${GF_ENV}"

GF_PY="${GF_ENV}/bin/python"

# Install package normally only if missing.
# This operates on the external source tree, not the SC2 repo.
if ! "${GF_PY}" -c 'import geneformer' >/dev/null 2>&1; then

    "${GF_PY}" -m pip install \
        --no-cache-dir \
        "${GF_SRC}"
fi

PYTHONPATH="${REPO}/src" \
"${GF_PY}" - "${GF_ASSET}" <<'PY'
import sys
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(
    sys.argv[1],
    local_files_only=True,
)

print(
    "GENEFORMER_CONFIG=",
    type(cfg).__name__,
)

print("GENEFORMER_LOAD_INTERFACE=PASS")
PY

receipt \
    Geneformer \
    "${GF_ENV}" \
    "${GF_SRC}" \
    "${GF_ASSET}"


echo
echo "============================================================"
echo "2. GENEMAMBA"
echo "============================================================"

GM_ENV="${ENVROOT}/sc2-p3-genemamba"
GM_SRC="${SRCROOT}/GeneMamba"
GM_ASSET="${CKPTROOT}/GeneMamba"

ensure_env "${GM_ENV}"

GM_PY="${GM_ENV}/bin/python"

# GeneMamba's repo has a broad frozen environment requirements
# file. For inference we install the package itself without
# replacing the already-working SC2 CUDA/PyTorch stack.
"${GM_PY}" -m pip install \
    --no-cache-dir \
    --no-deps \
    "${GM_SRC}"

# sc2_mamba normally already supplies mamba-ssm.
# Only build it if genuinely absent.
if ! "${GM_PY}" -c 'import mamba_ssm' >/dev/null 2>&1; then

    "${GM_PY}" -m pip install \
        --no-cache-dir \
        --no-build-isolation \
        'mamba-ssm==2.2.2'
fi

PYTHONPATH="${GM_ASSET}:${GM_SRC}:${REPO}/src" \
"${GM_PY}" - "${GM_ASSET}" <<'PY'
import sys

import torch
import transformers
import mamba_ssm

from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(
    sys.argv[1],
    trust_remote_code=True,
    local_files_only=True,
)

print(
    "GENEMAMBA_CONFIG=",
    type(cfg).__name__,
)

print(
    "GENEMAMBA_TORCH=",
    torch.__version__,
)

print(
    "GENEMAMBA_MAMBA_SSM=",
    getattr(mamba_ssm, "__version__", "NA"),
)

print("GENEMAMBA_LOAD_INTERFACE=PASS")
PY

receipt \
    GeneMamba \
    "${GM_ENV}" \
    "${GM_SRC}" \
    "${GM_ASSET}"


echo
echo "============================================================"
echo "3. scPRINT"
echo "============================================================"

SP_ENV="${ENVROOT}/sc2-p3-scprint"
SP_SRC="${SRCROOT}/scPRINT"
SP_ASSET="${CKPTROOT}/scPRINT/medium-v1.5.ckpt"

ensure_env "${SP_ENV}"

SP_PY="${SP_ENV}/bin/python"

# Job 662339 already installed scPRINT successfully.
# Reinstall only if the package itself is absent.
if ! "${SP_PY}" -c 'import scprint' >/dev/null 2>&1; then

    "${SP_PY}" -m pip install \
        --no-cache-dir \
        "${SP_SRC}"
fi

PYTHONPATH="${REPO}/src" \
"${SP_PY}" - <<'PY'
import scprint
import sc2
import torch

print(
    "SCPRINT_VERSION=",
    getattr(scprint, "__version__", "NA"),
)

print(
    "SCPRINT_TORCH=",
    torch.__version__,
)

print("SCPRINT_LOAD_INTERFACE=PASS")
PY

receipt \
    scPRINT \
    "${SP_ENV}" \
    "${SP_SRC}" \
    "${SP_ASSET}"


echo
echo "============================================================"
echo "4. ENVIRONMENT QC"
echo "============================================================"

du -sh \
    "${GF_ENV}" \
    "${GM_ENV}" \
    "${SP_ENV}"

quota -s 2>/dev/null || true

# Critical: environment building must not dirty SC2 repository.
cd "${REPO}"

test -z "$(git status --porcelain=v1 -uall)" || {
    echo "SC2_REPOSITORY_BECAME_DIRTY"
    git status --short
    exit 1
}

echo "SC2_REPOSITORY_REMAINED_CLEAN=true"
echo "P3_FOUNDATION_ENV_REPAIR_V3=PASS"
