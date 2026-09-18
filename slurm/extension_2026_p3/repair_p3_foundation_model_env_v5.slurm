#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/envrepair_v5_%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/envrepair_v5_%x_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

SRCROOT="${SC2}/external/foundation_models/p3_16k/sources"
CKPTROOT="${SC2}/external/foundation_models/p3_16k/checkpoints"
ENVROOT=/home/3159436/.conda/envs

OVERLAY="${SC2}/overlays/mamba22_git_torch25_cu124"

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/environment_repair_v5"

: "${MODEL:?MODEL missing}"
: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR="/tmp/sc2-pip-${SLURM_JOB_ID}"

mkdir -p \
    "${PIP_CACHE_DIR}" \
    "${OUTROOT}/${MODEL}"

cleanup() {
    rm -rf "${PIP_CACHE_DIR}"
}

trap cleanup EXIT

echo "============================================================"
echo "MODEL=${MODEL}"
echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"
echo "============================================================"

if [ "${MODEL}" = "Geneformer" ]; then

    ENV="${ENVROOT}/sc2-p3-geneformer"
    PY="${ENV}/bin/python"

    SOURCE="${SRCROOT}/Geneformer"
    ASSET="${CKPTROOT}/Geneformer-V2-104M"

    test -x "${PY}"
    test -d "${SOURCE}"
    test -s "${ASSET}/model.safetensors"

    # Derive the Transformers version from the pinned checkpoint.
    # We do not guess a package version.
    TF_VERSION=$(
        "${PY}" - "${ASSET}" <<'PY'
from pathlib import Path
import json
import re
import sys

root = Path(sys.argv[1])

found = []

for name in (
    "config.json",
    "generation_config.json",
):
    p = root / name

    if not p.is_file():
        continue

    x = json.loads(p.read_text())

    v = x.get("transformers_version")

    if isinstance(v, str) and v:
        found.append(
            (
                name,
                v,
            )
        )

if not found:
    raise RuntimeError(
        "No transformers_version in pinned checkpoint metadata"
    )

print(
    found[0][1]
)
PY
    )

    echo "GENEFORMER_PINNED_TRANSFORMERS=${TF_VERSION}"

    case "${TF_VERSION}" in
        4.*)
            ;;
        *)
            echo "Unexpected pinned Transformers version ${TF_VERSION}" >&2
            exit 1
            ;;
    esac

    "${PY}" -m pip install \
        --no-cache-dir \
        "transformers==${TF_VERSION}"

    PYTHONPATH="${REPO}/src" \
    "${PY}" - "${ASSET}" <<'PY'
import sys

import geneformer
import transformers

from transformers import AutoConfig
from transformers import SpecialTokensMixin

cfg = AutoConfig.from_pretrained(
    sys.argv[1],
    local_files_only=True,
)

print(
    "GENEFORMER_TRANSFORMERS=",
    transformers.__version__,
)

print(
    "GENEFORMER_CONFIG_CLASS=",
    type(cfg).__name__,
)

print(
    "GENEFORMER_MAX_POSITION_EMBEDDINGS=",
    cfg.max_position_embeddings,
)

print("GENEFORMER_ENV_GATE=PASS")
PY

elif [ "${MODEL}" = "GeneMamba" ]; then

    ENV="${ENVROOT}/sc2-p3-genemamba"
    PY="${ENV}/bin/python"

    SOURCE="${SRCROOT}/GeneMamba"
    ASSET="${CKPTROOT}/GeneMamba"

    test -x "${PY}"
    test -d "${SOURCE}"
    test -s "${ASSET}/model.safetensors"

    # Do not rebuild mamba-ssm unless necessary.
    # First use the already-proven SC2 Mamba stack / overlay.
    PYTHONPATH="${OVERLAY}:${ASSET}:${SOURCE}:${REPO}/src" \
    "${PY}" - "${ASSET}" <<'PY'
import importlib.metadata as md
import sys

import torch
import mamba_ssm

from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(
    sys.argv[1],
    trust_remote_code=True,
    local_files_only=True,
)

try:
    version = md.version("mamba-ssm")
except Exception:
    version = getattr(
        mamba_ssm,
        "__version__",
        "unknown",
    )

print(
    "GENEMAMBA_CONFIG_CLASS=",
    type(cfg).__name__,
)

print(
    "GENEMAMBA_MAMBA_SSM=",
    version,
)

print(
    "GENEMAMBA_TORCH=",
    torch.__version__,
)

print("GENEMAMBA_ENV_GATE=PASS")
PY

elif [ "${MODEL}" = "scPRINT" ]; then

    ENV="${ENVROOT}/sc2-p3-scprint"
    PY="${ENV}/bin/python"

    SOURCE="${SRCROOT}/scPRINT"
    ASSET="${CKPTROOT}/scPRINT/medium-v1.5.ckpt"

    test -x "${PY}"
    test -d "${SOURCE}"
    test -s "${ASSET}"

    # scPRINT already installed successfully in the previous wave.
    # Do not impose unrelated generic packages.
    PYTHONPATH="${REPO}/src" \
    "${PY}" - <<'PY'
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

print("SCPRINT_ENV_GATE=PASS")
PY

else
    echo "Unknown MODEL=${MODEL}" >&2
    exit 1
fi

# ------------------------------------------------------------
# Receipt
# ------------------------------------------------------------

RECEIPT="${OUTROOT}/${MODEL}/environment_receipt_v5.json"

PYTHONPATH="${OVERLAY}:${SOURCE}:${ASSET}:${REPO}/src" \
"${PY}" - \
    "${MODEL}" \
    "${ENV}" \
    "${SOURCE}" \
    "${ASSET}" \
    "${RECEIPT}" \
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
receipt = Path(sys.argv[5])
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

external = {}

if model == "Geneformer":
    import geneformer
    import transformers

    external = {
        "geneformer":
            getattr(geneformer, "__version__", None),
        "transformers":
            transformers.__version__,
    }

elif model == "GeneMamba":
    import mamba_ssm

    try:
        import importlib.metadata as md
        mamba_version = md.version("mamba-ssm")
    except Exception:
        mamba_version = getattr(
            mamba_ssm,
            "__version__",
            "unknown",
        )

    external = {
        "mamba_ssm":
            mamba_version,
    }

elif model == "scPRINT":
    import scprint

    external = {
        "scprint":
            getattr(scprint, "__version__", None),
    }

payload = {
    "schema":
        "sc2-p3-foundation-environment-repair-v5",

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

receipt.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print("RECEIPT=", receipt)
print("P3_FOUNDATION_ENV_REPAIR_V5=PASS")
PY

cd "${REPO}"

test -z "$(git status --porcelain=v1 -uall)"

du -sh "${ENV}"

echo "ENVIRONMENT_REPAIR_JOB=PASS"
