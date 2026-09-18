#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/env_%x_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/env_%x_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

BASE_ENV=/home/3159436/.conda/envs/sc2_mamba
ENVROOT=/home/3159436/.conda/envs

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/environments"

: "${MODEL:?}"
: "${SOURCE:?}"
: "${ASSET:?}"
: "${ENVNAME:?}"
: "${CONDA_BIN:?}"
: "${EXPECTED_JOB_HEAD:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test -d "${SOURCE}"
test -e "${ASSET}"
test -x "${BASE_ENV}/bin/python"
test -x "${CONDA_BIN}"

ENV="${ENVROOT}/${ENVNAME}"
OUT="${OUTROOT}/${MODEL}"

test ! -e "${ENV}"
test ! -e "${OUT}"

mkdir -p "${OUT}"

export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1

echo "MODEL=${MODEL}"
echo "SOURCE=${SOURCE}"
echo "ASSET=${ASSET}"
echo "ENV=${ENV}"
echo "JOB=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"

"${CONDA_BIN}" create \
    --yes \
    --prefix "${ENV}" \
    --clone "${BASE_ENV}"

PY="${ENV}/bin/python"

"${PY}" -m pip install \
    --no-cache-dir \
    --upgrade \
    setuptools \
    wheel

if [ "${MODEL}" = "Geneformer" ]; then

    test -s "${SOURCE}/requirements.txt"

    "${PY}" -m pip install \
        --no-cache-dir \
        -r "${SOURCE}/requirements.txt"

    "${PY}" -m pip install \
        --no-cache-dir \
        --no-deps \
        -e "${SOURCE}"

elif [ "${MODEL}" = "GeneMamba" ]; then

    test -s "${SOURCE}/requirements.txt"

    "${PY}" -m pip install \
        --no-cache-dir \
        -r "${SOURCE}/requirements.txt"

    "${PY}" -m pip install \
        --no-cache-dir \
        --no-deps \
        -e "${SOURCE}"

elif [ "${MODEL}" = "scPRINT" ]; then

    test -s "${SOURCE}/pyproject.toml"

    "${PY}" -m pip install \
        --no-cache-dir \
        -e "${SOURCE}"

else
    echo "Unsupported MODEL=${MODEL}" >&2
    exit 1
fi

# Current SC2 source is required by the P3 preparation/evaluation
# scripts, but dependencies are not allowed to rewrite the
# external-model environment.
"${PY}" -m pip install \
    --no-cache-dir \
    --no-deps \
    -e "${REPO}"

"${PY}" - \
    "${MODEL}" \
    "${SOURCE}" \
    "${ASSET}" \
    "${ENV}" \
    "${OUT}" \
    "${SLURM_JOB_ID}" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import importlib
import json
import subprocess
import sys

model = sys.argv[1]
source = Path(sys.argv[2])
asset = Path(sys.argv[3])
env = Path(sys.argv[4])
out = Path(sys.argv[5])
job = int(sys.argv[6])

for name in (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "transformers",
    "sc2",
):
    module = importlib.import_module(name)

    print(
        "IMPORT_PASS",
        name,
        getattr(module, "__version__", "NA"),
    )

if model == "Geneformer":

    module = importlib.import_module("geneformer")

    print(
        "EXTERNAL_IMPORT_PASS",
        "geneformer",
        getattr(module, "__version__", "NA"),
    )

elif model == "GeneMamba":

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        asset,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(
        "EXTERNAL_CONFIG_PASS",
        type(config).__name__,
    )

elif model == "scPRINT":

    module = importlib.import_module("scprint")

    print(
        "EXTERNAL_IMPORT_PASS",
        "scprint",
        getattr(module, "__version__", "NA"),
    )

else:
    raise ValueError(model)

import torch

receipt = {
    "schema":
        "sc2-p3-foundation-environment-receipt-v1",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "model":
        model,

    "job":
        job,

    "environment":
        str(env),

    "source":
        str(source),

    "asset":
        str(asset),

    "python":
        sys.version,

    "torch":
        torch.__version__,

    "cuda_build":
        torch.version.cuda,

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

    "scientific_inference":
        False,

    "status":
        "PASS",
}

p = out / "environment_receipt_v1.json"

p.write_text(
    json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print("ENV_RECEIPT=", p)
print("P3_FOUNDATION_ENVIRONMENT=PASS")
PY

du -sh "${ENV}"

echo "MODEL=${MODEL}"
echo "P3_FOUNDATION_ENV_JOB=PASS"
