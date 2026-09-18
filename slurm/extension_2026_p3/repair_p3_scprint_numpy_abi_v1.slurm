#!/bin/bash
#SBATCH --job-name=p3fm-scprint-numpy
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/scprint_numpy_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/scprint_numpy_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

ENV=/home/3159436/.conda/envs/sc2-p3-scprint
PY="${ENV}/bin/python"

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/environment_repair_v6/scPRINT"
RECEIPT="${OUTROOT}/numpy_abi_receipt_v1.json"

: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -x "${PY}"
test ! -e "${RECEIPT}"

mkdir -p "${OUTROOT}"

export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR="/tmp/sc2-pip-${SLURM_JOB_ID}"

cleanup() {
    rm -rf "${PIP_CACHE_DIR}"
}

trap cleanup EXIT

mkdir -p "${PIP_CACHE_DIR}"

echo "============================================================"
echo "scPRINT NUMPY ABI REPAIR"
echo "============================================================"
echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"

"${PY}" - <<'PY'
import numpy
import torch

print("NUMPY_BEFORE=", numpy.__version__)
print("TORCH=", torch.__version__)
PY

# Torch 2.2.0 was compiled against the NumPy 1.x C ABI.
"${PY}" -m pip install \
    --no-cache-dir \
    'numpy==1.26.4'

"${PY}" -m pip check

PYTHONPATH="${REPO}/src" \
"${PY}" - \
    "${RECEIPT}" \
    "${SLURM_JOB_ID}" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys

import numpy as np
import torch
import scprint
import sc2

receipt = Path(sys.argv[1])
job = int(sys.argv[2])

assert np.__version__ == "1.26.4"

# Explicitly test the bridge that emitted the previous warning.
a = np.arange(
    12,
    dtype=np.float32,
).reshape(3, 4)

t = torch.from_numpy(a)

assert t.shape == (3, 4)
assert t.dtype == torch.float32

b = t.numpy()

assert np.array_equal(a, b)

# Also test torch -> NumPy from a newly allocated tensor.
c = torch.arange(
    8,
    dtype=torch.float32,
).reshape(2, 4)

d = c.numpy()

assert d.shape == (2, 4)
assert np.allclose(
    d,
    np.arange(8, dtype=np.float32).reshape(2, 4),
)

payload = {
    "schema":
        "sc2-p3-scprint-numpy-abi-repair-v1",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "slurm_job_id":
        job,

    "numpy":
        np.__version__,

    "torch":
        torch.__version__,

    "scprint":
        getattr(scprint, "__version__", None),

    "numpy_to_torch_roundtrip":
        True,

    "torch_to_numpy_roundtrip":
        True,

    "scientific_inference":
        False,

    "pip_freeze":
        subprocess.check_output(
            [
                sys.executable,
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

print("NUMPY_AFTER=", np.__version__)
print("TORCH=", torch.__version__)
print("SCPRINT=", getattr(scprint, "__version__", "NA"))
print("NUMPY_TO_TORCH_ROUNDTRIP=PASS")
print("TORCH_TO_NUMPY_ROUNDTRIP=PASS")
print("RECEIPT=", receipt)
print("P3_SCPRINT_NUMPY_ABI_REPAIR=PASS")
PY
