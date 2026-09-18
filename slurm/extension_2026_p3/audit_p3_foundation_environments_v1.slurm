#!/bin/bash
#SBATCH --job-name=p3fm-env-audit
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/home/3159436/sc2/logs/p3_foundation_environment_audit_v1/audit_%j.out
#SBATCH --error=/home/3159436/sc2/logs/p3_foundation_environment_audit_v1/audit_%j.err

set -Eeuo pipefail
umask 027

SC2=/home/3159436/sc2
REPO="${SC2}/code"

PY=/home/3159436/.conda/envs/sc2-data/bin/python

FMROOT="${SC2}/external/foundation_models/p3_16k"
SRCROOT="${FMROOT}/sources"
CKPTROOT="${FMROOT}/checkpoints"

OUTROOT="${SC2}/outputs/paper_extension_2026_p3/foundation_environment_audit_v1"

SCF_RUNNER="${REPO}/scripts/eval/run_scfoundation_final_task.py"
SCF_SMOKE="${REPO}/scripts/eval/run_scfoundation_decoder_smoke.py"
SCF_PROTOCOL="${REPO}/configs/extension_2026/scfoundation_final_protocol_v1.json"

: "${EXPECTED_JOB_HEAD:?EXPECTED_JOB_HEAD missing}"

cd "${REPO}"

test "$(git rev-parse HEAD)" = "${EXPECTED_JOB_HEAD}"
test -z "$(git status --porcelain=v1 -uall)"

test ! -e "${OUTROOT}"
mkdir -p "${OUTROOT}"

unset PYTHONPATH
unset LD_LIBRARY_PATH
unset PYTHONHOME

export PYTHONNOUSERSITE=1

"${PY}" - \
    "${SRCROOT}" \
    "${CKPTROOT}" \
    "${OUTROOT}" \
    "${SCF_RUNNER}" \
    "${SCF_SMOKE}" \
    "${SCF_PROTOCOL}" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import sys

srcroot = Path(sys.argv[1])
ckptroot = Path(sys.argv[2])
outroot = Path(sys.argv[3])

scf_runner = Path(sys.argv[4])
scf_smoke = Path(sys.argv[5])
scf_protocol = Path(sys.argv[6])

models = [
    "Geneformer",
    "GeneMamba",
    "scPRINT",
    "scPRINT-2",
    "scLong",
    "RegFormer",
    "scGPT-current",
]


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def dependency_files(root):
    names = []

    patterns = (
        "requirements*.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "environment*.yml",
        "environment*.yaml",
        "conda*.yml",
        "conda*.yaml",
    )

    for pattern in patterns:
        for p in root.glob(pattern):
            if p.is_file():
                names.append(p)

    return sorted(set(names))


def text_excerpt(path, max_chars=50000):
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return text[:max_chars]


def interesting_lines(root):
    patterns = (
        "checkpoint",
        "pretrain",
        "huggingface",
        "download",
        "denois",
        "reconstruct",
        "decoder",
        "embedding",
        "token",
        "vocab",
        "max_len",
        "max_length",
        "4096",
        "19264",
        "gene",
    )

    hits = []

    candidates = []

    for filename in (
        "README.md",
        "README.rst",
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
    ):
        p = root / filename

        if p.is_file():
            candidates.append(p)

    for p in root.rglob("*.py"):
        try:
            if len(p.relative_to(root).parts) <= 3:
                candidates.append(p)
        except Exception:
            pass

    candidates = sorted(set(candidates))

    for p in candidates:
        try:
            lines = p.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines, 1):
            low = line.lower()

            if any(token in low for token in patterns):
                hits.append({
                    "file": str(p),
                    "line": i,
                    "text": line[:500],
                })

    return hits[:4000]


def python_requires(files):
    result = []

    regexes = (
        re.compile(
            r'requires-python\s*=\s*["\']([^"\']+)'
        ),
        re.compile(
            r'python_requires\s*=\s*["\']([^"\']+)'
        ),
        re.compile(
            r'python\s*[=<>!~]+\s*([0-9][^,\s]*)',
            re.I,
        ),
    )

    for p in files:
        text = p.read_text(
            encoding="utf-8",
            errors="replace",
        )

        for regex in regexes:
            for m in regex.finditer(text):
                result.append({
                    "file": str(p),
                    "value": m.group(1),
                })

    return result


def source_info(name):
    root = srcroot / name

    assert root.is_dir(), root

    deps = dependency_files(root)

    return {
        "source_root": str(root),

        "git_head":
            __import__("subprocess")
            .check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            )
            .strip(),

        "dependency_files": [
            {
                "path": str(p),
                "sha256": sha(p),
                "content": text_excerpt(p),
            }
            for p in deps
        ],

        "python_requirements":
            python_requires(deps),

        "interesting_lines":
            interesting_lines(root),
    }


report = {
    "schema":
        "sc2-p3-foundation-environment-interface-audit-v1",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "source_feature_space_genes": 16384,

    "models": {
        name: source_info(name)
        for name in models
    },

    "checkpoints": {},
}

for p in sorted(ckptroot.rglob("*")):
    if not p.is_file():
        continue

    report["checkpoints"][
        str(p.relative_to(ckptroot))
    ] = {
        "path": str(p),
        "bytes": p.stat().st_size,
        "gib": p.stat().st_size / 1024**3,
        "sha256": sha(p),
    }

# ------------------------------------------------------------
# scFoundation exact runner context.
# We need this to build the 16K-specific wrapper without
# guessing what was previously hard-coded for 4096.
# ------------------------------------------------------------

def code_context(path):
    lines = path.read_text().splitlines()

    needles = (
        "4096",
        "4006",
        "19264",
        "mapping",
        "vocab",
        "availability",
        "panel",
        "checkpoint",
        "decoder",
        "predict",
        "output",
    )

    selected = set()

    for i, line in enumerate(lines):
        if any(x.lower() in line.lower() for x in needles):
            for j in range(
                max(0, i - 4),
                min(len(lines), i + 5),
            ):
                selected.add(j)

    result = []

    previous = None

    for i in sorted(selected):
        if previous is not None and i != previous + 1:
            result.append({
                "line": None,
                "text": "...",
            })

        result.append({
            "line": i + 1,
            "text": lines[i],
        })

        previous = i

    return result


report["scFoundation"] = {
    "runner": str(scf_runner),
    "runner_sha256": sha(scf_runner),
    "runner_context": code_context(scf_runner),

    "smoke": str(scf_smoke),
    "smoke_sha256": sha(scf_smoke),
    "smoke_context": code_context(scf_smoke),

    "p2_protocol": str(scf_protocol),
    "p2_protocol_sha256": sha(scf_protocol),

    "existing_environment":
        "/home/3159436/.conda/envs/sc2-scfoundation",

    "p3_mapping": {
        "source_genes": 16384,
        "mapped_genes": 14382,
        "unmapped_genes": 2002,
    },
}

destination = (
    outroot
    / "foundation_environment_interface_audit_v1.json"
)

destination.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

# Human-readable requirements summary
with (
    outroot
    / "environment_requirements_summary.txt"
).open("w") as f:

    for name in models:
        x = report["models"][name]

        f.write("=" * 80 + "\n")
        f.write(name + "\n")
        f.write("=" * 80 + "\n")

        f.write(
            "git_head="
            + x["git_head"]
            + "\n"
        )

        for r in x["python_requirements"]:
            f.write(
                "python_requirement="
                + str(r)
                + "\n"
            )

        for dep in x["dependency_files"]:
            f.write(
                "dependency_file="
                + dep["path"]
                + "\n"
            )

        f.write("\n")

print("FOUNDATION_SOURCE_MODELS_AUDITED=", len(models))
print("FOUNDATION_CHECKPOINT_FILES=", len(report["checkpoints"]))
print("SCFOUNDATION_P3_MAPPED_GENES=14382")
print("SCFOUNDATION_SOURCE_GENES=16384")
print("FOUNDATION_ENVIRONMENT_INTERFACE_AUDIT=PASS")
print("OUTPUT=", destination)
PY

echo
echo "================ REQUIREMENTS SUMMARY ================"

cat \
"${OUTROOT}/environment_requirements_summary.txt"

echo
echo "P3_FOUNDATION_ENVIRONMENT_AUDIT_JOB=PASS"
