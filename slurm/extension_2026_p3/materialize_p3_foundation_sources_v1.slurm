#!/bin/bash
#SBATCH --job-name=p3fm-materialize
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_materialization_v1/materialize_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_materialization_v1/materialize_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2-data/bin/python

FMROOT="${SC2}/external/foundation_models/p3_16k"
SRCROOT="${FMROOT}/sources"
CKPTROOT="${FMROOT}/checkpoints"

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_materialization_v1"

: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test ! -e "${FMROOT}"
test ! -e "${OUTROOT}"

mkdir -p \
    "${SRCROOT}" \
    "${CKPTROOT}" \
    "${OUTROOT}"

clone_exact() {
    NAME="$1"
    URL="$2"
    REV="$3"

    DEST="${SRCROOT}/${NAME}"

    echo
    echo "CLONE=${NAME}"
    echo "URL=${URL}"
    echo "REV=${REV}"

    mkdir "${DEST}"

    git -C "${DEST}" init -q
    git -C "${DEST}" remote add origin "${URL}"

    GIT_LFS_SKIP_SMUDGE=1 \
    git -C "${DEST}" fetch \
        --depth=1 \
        origin \
        "${REV}"

    GIT_LFS_SKIP_SMUDGE=1 \
    git -C "${DEST}" checkout \
        --detach \
        FETCH_HEAD

    ACTUAL=$(git -C "${DEST}" rev-parse HEAD)

    echo "ACTUAL=${ACTUAL}"

    test "${ACTUAL}" = "${REV}"
}

hf_download() {
    REPO_ID="$1"
    REV="$2"
    FILE="$3"
    DEST="$4"

    URL="https://huggingface.co/${REPO_ID}/resolve/${REV}/${FILE}?download=true"

    mkdir -p "$(dirname "${DEST}")"

    echo "HF_DOWNLOAD=${REPO_ID}@${REV}:${FILE}"

    curl \
        --fail \
        --location \
        --retry 4 \
        --retry-delay 2 \
        --connect-timeout 30 \
        --output "${DEST}.part" \
        "${URL}"

    test -s "${DEST}.part"

    mv "${DEST}.part" "${DEST}"
}

echo "============================================================"
echo "FOUNDATION MATERIALIZATION"
echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"
echo "HEAD=$(git rev-parse HEAD)"
echo "============================================================"

# ------------------------------------------------------------
# 1. Geneformer
# ------------------------------------------------------------

clone_exact \
    Geneformer \
    https://huggingface.co/ctheodoris/Geneformer \
    1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5

GF="${CKPTROOT}/Geneformer-V2-104M"

hf_download \
    ctheodoris/Geneformer \
    1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5 \
    Geneformer-V2-104M/model.safetensors \
    "${GF}/model.safetensors"

hf_download \
    ctheodoris/Geneformer \
    1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5 \
    Geneformer-V2-104M/config.json \
    "${GF}/config.json"

hf_download \
    ctheodoris/Geneformer \
    1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5 \
    Geneformer-V2-104M/generation_config.json \
    "${GF}/generation_config.json"

# Download tokenizer/mapping dictionaries from the same pinned HF revision,
# but only top-level small pkl assets.
"${PY}" - "${GF}" <<'PY'
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
import json
import sys

dest = Path(sys.argv[1])

repo = "ctheodoris/Geneformer"
rev = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"

api = f"https://huggingface.co/api/models/{repo}?blobs=true"

req = Request(
    api,
    headers={"User-Agent": "sc2-p3-geneformer-materializer"},
)

with urlopen(req, timeout=60) as response:
    obj = json.load(response)

selected = []

for item in obj.get("siblings", []):
    path = str(item.get("rfilename", ""))
    size = item.get("size")

    if "/" in path:
        continue

    low = path.lower()

    if not low.endswith(".pkl"):
        continue

    if not any(
        token in low
        for token in (
            "dictionary",
            "mapping",
            "median",
            "token",
            "gene",
        )
    ):
        continue

    if isinstance(size, int) and size > 100 * 1024 * 1024:
        continue

    selected.append(path)

for path in sorted(selected):
    url = (
        f"https://huggingface.co/{repo}/resolve/{rev}/"
        + quote(path)
        + "?download=true"
    )

    target = dest / path

    req = Request(
        url,
        headers={"User-Agent": "sc2-p3-geneformer-materializer"},
    )

    with urlopen(req, timeout=120) as response:
        target.write_bytes(response.read())

    print("GENEFORMER_ASSET=", target)

print("GENEFORMER_AUX_ASSET_COUNT=", len(selected))
PY

test "$(stat -c %s "${GF}/model.safetensors")" -eq 417571156

# ------------------------------------------------------------
# 2. GeneMamba
# ------------------------------------------------------------

clone_exact \
    GeneMamba \
    https://github.com/MineSelf2016/GeneMamba.git \
    22f571a8d48256894de05e9d87ea9ec6caa338c0

GM="${CKPTROOT}/GeneMamba"

for FILE in \
    config.json \
    configuration_genemamba.py \
    modeling_genemamba.py \
    modeling_outputs.py \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    model.safetensors
do
    hf_download \
        mineself2016/GeneMamba \
        dac10b05b4834fdb7a2fddbf47eec6acf6e9aee4 \
        "${FILE}" \
        "${GM}/${FILE}"
done

test "$(stat -c %s "${GM}/model.safetensors")" -eq 262998656

# ------------------------------------------------------------
# 3. scPRINT
# ------------------------------------------------------------

clone_exact \
    scPRINT \
    https://github.com/cantinilab/scPRINT.git \
    e7c60766cbff58bb7280a1c8fe4b7461b1af1911

SP="${CKPTROOT}/scPRINT"

hf_download \
    jkobject/scPRINT \
    8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed \
    medium-v1.5.ckpt \
    "${SP}/medium-v1.5.ckpt"

test "$(stat -c %s "${SP}/medium-v1.5.ckpt")" -eq 221742980

# ------------------------------------------------------------
# 4. scPRINT-2 source audit
# ------------------------------------------------------------

clone_exact \
    scPRINT-2 \
    https://github.com/cantinilab/scPRINT-2.git \
    9dbc57acb66ec45cf0a8676740473e0328887188

# ------------------------------------------------------------
# 5. scLong source
# ------------------------------------------------------------

clone_exact \
    scLong \
    https://github.com/BaiDing1234/scLong.git \
    41b72021540918e4386c4f2264351d7a6cbeed99

# ------------------------------------------------------------
# 6. RegFormer source
# ------------------------------------------------------------

clone_exact \
    RegFormer \
    https://github.com/BGIResearch/RegFormer.git \
    26de67939de93a9157b3ee99cc346fb532e92ba5

# ------------------------------------------------------------
# 7. Current scGPT source for new compatibility audit only
# ------------------------------------------------------------

clone_exact \
    scGPT-current \
    https://github.com/bowang-lab/scGPT.git \
    cebd6fae655b9c585a4807daa3ac31bb764f06b4

# ------------------------------------------------------------
# 8. Write immutable receipt
# ------------------------------------------------------------

"${PY}" - \
    "${FMROOT}" \
    "${OUTROOT}/foundation_materialization_receipt_v1.json" \
    "${EXPECTED_JOB_HEAD}" \
    "${SLURM_JOB_ID}" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

root = Path(sys.argv[1])
out = Path(sys.argv[2])
head = sys.argv[3]
job = int(sys.argv[4])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


sources = {}

for p in sorted((root / "sources").iterdir()):
    if not p.is_dir():
        continue

    commit = subprocess.check_output(
        ["git", "-C", str(p), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    sources[p.name] = {
        "path": str(p),
        "commit": commit,
    }


weights = {}

for p in sorted((root / "checkpoints").rglob("*")):
    if not p.is_file():
        continue

    if p.suffix.lower() not in {
        ".ckpt",
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".json",
        ".pkl",
    }:
        continue

    rel = str(p.relative_to(root))

    weights[rel] = {
        "path": str(p),
        "bytes": p.stat().st_size,
        "sha256": sha(p),
    }


requirements = {}

for source_name, source in sources.items():
    base = Path(source["path"])

    files = []

    for candidate in (
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "environment.yml",
        "environment.yaml",
        "sclong.yml",
    ):
        p = base / candidate

        if p.is_file():
            files.append(
                {
                    "path": str(p),
                    "sha256": sha(p),
                }
            )

    requirements[source_name] = files


payload = {
    "schema":
        "sc2-p3-foundation-materialization-receipt-v1",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "sc2_git_head": head,
    "slurm_job_id": job,

    "p3_source_feature_space_genes": 16384,

    "sources": sources,
    "assets": weights,
    "environment_spec_files": requirements,

    "scientific_inference_performed": False,

    "status": "PASS",
}

out.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)

print("FOUNDATION_SOURCE_COUNT=", len(sources))
print("FOUNDATION_ASSET_COUNT=", len(weights))
print("FOUNDATION_MATERIALIZATION_RECEIPT_SHA256=", sha(out))
print("P3_FOUNDATION_MATERIALIZATION=PASS")
PY

echo
echo "================ MATERIALIZED SIZE ================"
du -sh "${FMROOT}"

echo
echo "================ QUOTA ================"
quota -s 2>/dev/null || true

echo
echo "P3_FOUNDATION_SOURCE_JOB=PASS"
