#!/bin/bash
#SBATCH --job-name=p3-foundation-audit
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_readiness_v1/foundation_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_readiness_v1/foundation_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"
OUT="${SC2}/outputs/paper_extension_2026_p3/foundation_readiness_v1"

PY=/home/3159436/.conda/envs/sc2-data/bin/python

: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test ! -e "${OUT}"
mkdir -p "${OUT}"

unset PYTHONPATH
unset LD_LIBRARY_PATH
unset PYTHONHOME

export PYTHONNOUSERSITE=1

"${PY}" - \
    "${REPO}" \
    "${SC2}" \
    "${OUT}" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys

repo = Path(sys.argv[1])
sc2 = Path(sys.argv[2])
out = Path(sys.argv[3])

vocab = (
    sc2
    / "data"
    / "census_curated_pretrain_pilot250k_g16384"
    / "gene_vocabulary.parquet"
)

assert vocab.is_file()

import pandas as pd

v = pd.read_parquet(vocab)

assert len(v) == 16384

patterns = (
    "scfoundation",
    "scgpt",
    "geneformer",
    "scprint",
    "genemamba",
    "mambacell",
    "sclong",
    "regformer",
)

config_files = []

for base in (
    repo / "configs" / "extension_2026",
    repo / "configs" / "eval" / "extension_2026",
):
    if not base.exists():
        continue

    for p in base.glob("*.json"):
        low = p.name.lower()
        if any(x in low for x in patterns):
            config_files.append(p)

config_files = sorted(set(config_files))

script_files = []

for p in repo.rglob("*.py"):
    low = p.name.lower()

    if any(x in low for x in patterns):
        script_files.append(p)

script_files = sorted(set(script_files))

absolute_refs = []
sha_tokens = []
interesting_fields = []


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, val in obj.items():
            new_path = f"{path}.{k}" if path else str(k)

            if any(
                t in str(k).lower()
                for t in (
                    "checkpoint",
                    "commit",
                    "source",
                    "repo",
                    "vocab",
                    "gene",
                    "environment",
                    "model",
                )
            ):
                if isinstance(val, (str, int, float, bool)):
                    interesting_fields.append(
                        {
                            "path": new_path,
                            "value": val,
                        }
                    )

            walk(val, new_path)

    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            walk(val, f"{path}[{i}]")

    elif isinstance(obj, str):
        if obj.startswith("/"):
            absolute_refs.append(obj)

        if re.fullmatch(r"[0-9a-fA-F]{64}", obj):
            sha_tokens.append(obj.lower())


for p in config_files:
    try:
        walk(json.loads(p.read_text()), p.name)
    except Exception:
        pass

ref_status = []

for raw in sorted(set(absolute_refs)):
    p = Path(raw)

    entry = {
        "path": raw,
        "exists": p.exists(),
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
    }

    if p.is_file():
        try:
            entry["size_bytes"] = p.stat().st_size
        except Exception:
            pass

    ref_status.append(entry)

env_root = Path("/home/3159436/.conda/envs")

envs = []

if env_root.is_dir():
    for p in sorted(env_root.iterdir()):
        if not p.is_dir():
            continue

        low = p.name.lower()

        if any(
            x in low
            for x in (
                "foundation",
                "scgpt",
                "geneformer",
                "print",
                "mamba",
            )
        ):
            envs.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "python_exists": (p / "bin/python").is_file(),
                }
            )

report = {
    "schema": "sc2-p3-foundation-readiness-audit-v1",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "p3_vocabulary": str(vocab),
    "p3_vocabulary_genes": 16384,

    "existing_foundation_config_files": [
        str(x) for x in config_files
    ],

    "existing_foundation_scripts": [
        str(x) for x in script_files
    ],

    "referenced_absolute_paths": ref_status,

    "existing_sha256_tokens": sorted(set(sha_tokens)),

    "interesting_frozen_fields": interesting_fields,

    "candidate_conda_environments": envs,

    "scientific_inference_performed": False,

    "next":
        "Use pinned P2 scFoundation/scGPT evidence plus this inventory "
        "to construct the P3 16K compatibility mapping and validation smoke.",

    "status": "AUDIT_COMPLETE",
}

destination = out / "foundation_readiness_audit_v1.json"

destination.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)

print("P3_VOCAB_GENES=16384")
print("FOUNDATION_CONFIG_FILES=", len(config_files))
print("FOUNDATION_SCRIPT_FILES=", len(script_files))
print("FOUNDATION_REFERENCED_PATHS=", len(ref_status))
print("FOUNDATION_ENV_CANDIDATES=", len(envs))
print("FOUNDATION_SCIENTIFIC_INFERENCE=false")
print("P3_FOUNDATION_READINESS_AUDIT=PASS")
print("OUTPUT=", destination)
PY
