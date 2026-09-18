#!/bin/bash
#SBATCH --job-name=p3fm-remote-audit
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_16k_resource_preflight_v1/remote_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_16k_resource_preflight_v1/remote_%j.err

set -Eeuo pipefail

SC2=/home/3159436/sc2
REPO="${SC2}/code"
OUT="${SC2}/outputs/paper_extension_2026_p3/foundation_16k_resource_preflight_v1/remote_asset_audit_v1.json"

PY=/home/3159436/.conda/envs/sc2-data/bin/python

: "${EXPECTED_JOB_HEAD:?}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

unset PYTHONPATH
unset LD_LIBRARY_PATH
unset PYTHONHOME

export PYTHONNOUSERSITE=1

"${PY}" - "${OUT}" <<'PY'
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import json
import subprocess
import sys

out = sys.argv[1]

hf_targets = {
    "Geneformer": {
        "repo": "ctheodoris/Geneformer",
        "wanted": [
            "Geneformer-V2-104M/",
            "Geneformer-V2-316M/",
        ],
    },
    "GeneMamba": {
        "repo": "mineself2016/GeneMamba",
        "wanted": [""],
    },
    "scPRINT": {
        "repo": "jkobject/scPRINT",
        "wanted": [
            "medium-v1",
            "medium-v1.5",
            "small-v2",
        ],
    },
}

github_targets = {
    "scPRINT": "https://github.com/cantinilab/scPRINT.git",
    "scPRINT-2": "https://github.com/cantinilab/scPRINT-2.git",
    "GeneMamba": "https://github.com/MineSelf2016/GeneMamba.git",
    "scLong": "https://github.com/BaiDing1234/scLong.git",
    "RegFormer": "https://github.com/BGIResearch/RegFormer.git",
    "scGPT": "https://github.com/bowang-lab/scGPT.git",
}

result = {
    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "downloads_performed": False,

    "huggingface": {},
    "github": {},
}

for name, spec in hf_targets.items():
    repo = spec["repo"]

    url = (
        "https://huggingface.co/api/models/"
        + repo
        + "?blobs=true"
    )

    try:
        req = Request(
            url,
            headers={"User-Agent": "sc2-p3-resource-audit"},
        )

        with urlopen(req, timeout=30) as response:
            obj = json.load(response)

        files = []

        for item in obj.get("siblings", []):
            path = str(item.get("rfilename", ""))
            size = item.get("size")

            if any(
                token == "" or token in path
                for token in spec["wanted"]
            ):
                files.append({
                    "path": path,
                    "size_bytes": size,
                })

        result["huggingface"][name] = {
            "repo": repo,
            "revision": obj.get("sha"),
            "files": files,
            "status": "PASS",
        }

        print(
            "HF_REMOTE_AUDIT_PASS",
            name,
            repo,
            "revision=",
            obj.get("sha"),
        )

        for f in files:
            print(
                "HF_FILE",
                name,
                f["path"],
                f["size_bytes"],
            )

    except Exception as exc:
        result["huggingface"][name] = {
            "repo": repo,
            "status": "ERROR",
            "error": repr(exc),
        }

        print(
            "HF_REMOTE_AUDIT_ERROR",
            name,
            repr(exc),
        )

for name, url in github_targets.items():
    try:
        r = subprocess.run(
            ["git", "ls-remote", url, "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )

        if r.returncode == 0 and r.stdout.strip():
            sha = r.stdout.split()[0]

            result["github"][name] = {
                "url": url,
                "head": sha,
                "status": "PASS",
            }

            print(
                "GITHUB_REMOTE_AUDIT_PASS",
                name,
                sha,
            )

        else:
            raise RuntimeError(r.stderr.strip())

    except Exception as exc:
        result["github"][name] = {
            "url": url,
            "status": "ERROR",
            "error": repr(exc),
        }

        print(
            "GITHUB_REMOTE_AUDIT_ERROR",
            name,
            repr(exc),
        )

with open(out, "w") as f:
    json.dump(
        result,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

print("REMOTE_CHECKPOINT_DOWNLOADS=false")
print("FOUNDATION_REMOTE_ASSET_AUDIT=PASS")
PY
