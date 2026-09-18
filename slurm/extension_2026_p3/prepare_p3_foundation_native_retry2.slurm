#!/bin/bash
#SBATCH --job-name=p3fm-native-prep-r2
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/native_retry2_prep_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_native_smokes_v1/native_retry2_prep_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

GF_PY=/home/3159436/.conda/envs/sc2-p3-geneformer/bin/python
SP_PY=/home/3159436/.conda/envs/sc2-p3-scprint/bin/python
LAMIN=/home/3159436/.conda/envs/sc2-p3-scprint/bin/lamin

GF_REV=1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5
GF_AUX="${SC2}/external/foundation_models/p3_16k/geneformer_aux_v1"

LAMIN_PARENT="${SC2}/external/foundation_models/p3_16k"
LAMIN_STORAGE="${LAMIN_PARENT}/scprint_lamin_v1"
LAMIN_READY="${LAMIN_PARENT}/scprint_lamin_v1.READY"
LAMIN_NAME=sc2p3scprintv1

RECEIPT="${SC2}/outputs/paper_extension_2026_p3/foundation_native_smokes_v1/environment_repair_v7/native_retry2_prerequisites.json"

: "${EXPECTED_JOB_HEAD:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

mkdir -p \
    "${GF_AUX}" \
    "$(dirname "${RECEIPT}")"

echo "============================================================"
echo "GENEFORMER OFFICIAL V2 AUXILIARY ASSETS"
echo "============================================================"

"${GF_PY}" - \
    "${GF_AUX}" \
    "${GF_REV}" <<'PY'
from pathlib import Path
import hashlib
import json
import pickle
import sys

from huggingface_hub import hf_hub_download

root = Path(sys.argv[1])
revision = sys.argv[2]

repo = "ctheodoris/Geneformer"

names = [
    "geneformer/gene_median_dictionary_gc104M.pkl",
    "geneformer/token_dictionary_gc104M.pkl",
    "geneformer/ensembl_mapping_dict_gc104M.pkl",
]

for name in names:

    p = Path(
        hf_hub_download(
            repo_id=repo,
            filename=name,
            revision=revision,
            local_dir=root,
        )
    )

    raw = p.read_bytes()

    assert not raw.startswith(
        b"version https://git-lfs.github.com/spec/v1"
    ), p

    assert len(raw) > 100, (
        p,
        len(raw),
    )

    with p.open("rb") as f:
        obj = pickle.load(f)

    assert isinstance(obj, dict), (
        p,
        type(obj),
    )

    h = hashlib.sha256(raw).hexdigest()

    print(
        "GENEFORMER_ASSET",
        name,
        "BYTES=",
        len(raw),
        "ENTRIES=",
        len(obj),
        "SHA256=",
        h,
    )

print("GENEFORMER_AUXILIARY_ASSET_GATE=PASS")
PY

test -s "${GF_AUX}/geneformer/gene_median_dictionary_gc104M.pkl"
test -s "${GF_AUX}/geneformer/token_dictionary_gc104M.pkl"
test -s "${GF_AUX}/geneformer/ensembl_mapping_dict_gc104M.pkl"

echo
echo "============================================================"
echo "scPRINT LAMIN / BIONTY"
echo "============================================================"

test -x "${LAMIN}"

if [ ! -e "${LAMIN_READY}" ]; then

    # Dedicated SC2 instance only. Never touch unrelated storage.
    test ! -e "${LAMIN_STORAGE}" || {
        echo "STOP: Lamin storage exists without READY marker" >&2
        exit 1
    }

    cd "${LAMIN_PARENT}"

    HELP=$(
        "${LAMIN}" init --help 2>&1
    )

    if printf '%s\n' "${HELP}" |
       grep -q -- '--modules'; then

        "${LAMIN}" init \
            --storage "${LAMIN_STORAGE}" \
            --name "${LAMIN_NAME}" \
            --modules bionty

    elif printf '%s\n' "${HELP}" |
         grep -q -- '--schema'; then

        "${LAMIN}" init \
            --storage "${LAMIN_STORAGE}" \
            --name "${LAMIN_NAME}" \
            --schema bionty

    else
        echo "STOP: installed lamin exposes neither --modules nor --schema" >&2
        exit 1
    fi

    "${SP_PY}" - <<'PY'
from scdataloader.utils import (
    load_genes,
    populate_my_ontology,
)

populate_my_ontology(
    organisms=[
        "NCBITaxon:9606",
    ],
    sex=[
        "PATO:0000384",
        "PATO:0000383",
    ],
    celltypes=None,
    ethnicities=None,
    assays=None,
    tissues=None,
    diseases=None,
    dev_stages=None,
)

genes = load_genes(
    ["NCBITaxon:9606"]
)

assert len(genes) > 0

print(
    "SCPRINT_HUMAN_ONTOLOGY_GENES=",
    len(genes),
)

print(
    "SCPRINT_LAMIN_ONTOLOGY_GATE=PASS"
)
PY

    printf '%s\n' \
        "READY" \
        > "${LAMIN_READY}"

else

    "${SP_PY}" - <<'PY'
from scdataloader.utils import load_genes

genes = load_genes(
    ["NCBITaxon:9606"]
)

assert len(genes) > 0

print(
    "EXISTING_SCPRINT_HUMAN_ONTOLOGY_GENES=",
    len(genes),
)

print(
    "EXISTING_SCPRINT_LAMIN_GATE=PASS"
)
PY

fi

cd "${REPO}"

"${SP_PY}" - \
    "${GF_AUX}" \
    "${LAMIN_STORAGE}" \
    "${LAMIN_READY}" \
    "${RECEIPT}" \
    "${SLURM_JOB_ID}" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

gf = Path(sys.argv[1])
lamin = Path(sys.argv[2])
ready = Path(sys.argv[3])
receipt = Path(sys.argv[4])
job = int(sys.argv[5])

assets = {}

for name in (
    "gene_median_dictionary_gc104M.pkl",
    "token_dictionary_gc104M.pkl",
    "ensembl_mapping_dict_gc104M.pkl",
):
    p = gf / "geneformer" / name

    assert p.is_file()

    assets[name] = {
        "path": str(p),
        "bytes": p.stat().st_size,
        "sha256":
            hashlib.sha256(
                p.read_bytes()
            ).hexdigest(),
    }

assert ready.is_file()

payload = {
    "schema":
        "sc2-p3-foundation-native-retry2-prerequisites-v1",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "slurm_job_id":
        job,

    "geneformer_revision":
        "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5",

    "geneformer_assets":
        assets,

    "scprint_lamin_storage":
        str(lamin),

    "scprint_lamin_ready":
        True,

    "scientific_result":
        False,

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
print("FOUNDATION_NATIVE_RETRY2_PREREQUISITES=PASS")
PY
