#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def qlabel(q):
    return f"q{int(round(float(q) * 100)):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--task-id", required=True, type=int)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    protocol_path = Path(args.protocol).resolve()
    p = json.loads(protocol_path.read_text())

    expected_sha = os.environ.get("SC2_BARON_LODO_PROTOCOL_SHA")
    if expected_sha and sha(protocol_path) != expected_sha:
        raise RuntimeError("Protocol SHA mismatch")

    output_root = Path(args.output_root).resolve()

    if output_root != Path(p["output_root"]).resolve():
        raise RuntimeError("Output root differs from frozen protocol")

    matches = [
        x for x in p["export_tasks"]
        if int(x["task_id"]) == args.task_id
    ]

    if len(matches) != 1:
        raise RuntimeError("Unknown export task")

    task = matches[0]
    checkpoint = Path(task["checkpoint"])

    if not checkpoint.is_file():
        raise RuntimeError("Checkpoint missing")

    if sha(checkpoint) != task["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint SHA mismatch")

    for run in task["runs"]:
        if not Path(run["panel"]).is_file():
            raise RuntimeError(f"Missing panel: {run['panel']}")

        if not Path(run["reference_summary"]).is_file():
            raise RuntimeError(
                f"Missing frozen reference summary: "
                f"{run['reference_summary']}"
            )

    if args.check_only:
        print(
            f"BARON_LODO_EXPORT_CHECK=PASS "
            f"TASK={args.task_id} "
            f"MODEL_SEED={task['model_seed']} "
            f"RUNS={len(task['runs'])}"
        )
        return

    task_name = (
        f"task_{args.task_id:02d}_"
        f"model{task['model_seed']}"
    )

    final = output_root / task_name
    temp = output_root / f".tmp_{task_name}_{os.getpid()}"

    if final.exists() or temp.exists():
        raise RuntimeError("Output collision")

    output_root.mkdir(parents=True, exist_ok=True)
    temp.mkdir()

    records = []

    try:
        for run in task["runs"]:
            rep = int(run["replicate"])
            q = float(run["q"])
            threshold = float(run["threshold"])

            panel = Path(run["panel"])

            if sha(panel) != run["panel_sha256"]:
                raise RuntimeError(
                    f"Panel SHA mismatch: {panel}"
                )

            qdir = temp / f"rep{rep:02d}" / qlabel(q)
            evaldir = qdir / "eval"
            raw = qdir / "raw_arrays.npz"

            qdir.mkdir(parents=True)

            cmd = [
                sys.executable,
                "-m",
                "sc2.eval.export_continuous_repair_arrays",
                "--config",
                p["eval_config"]["path"],
                "--checkpoint",
                str(checkpoint),
                "--benchmark",
                str(panel),
                "--output-dir",
                str(evaldir),
                "--threshold",
                str(threshold),
                "--array-output",
                str(raw),
            ]

            subprocess.run(cmd, check=True)

            new_summary = json.loads(
                (evaldir / "summary.json").read_text()
            )

            ref_summary = json.loads(
                Path(run["reference_summary"]).read_text()
            )

            if (
                new_summary["threshold_source"]
                != "frozen_external_threshold"
            ):
                raise RuntimeError(
                    "Automatic threshold selection occurred"
                )

            if not math.isclose(
                float(new_summary["threshold"]),
                threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError("Threshold mismatch")

            for key in (
                "masked_mse",
                "masked_mae",
                "gene_spearman",
                "sample_spearman",
                "threshold_recall",
                "threshold_precision",
                "threshold_true_zero_fill",
            ):
                a = float(new_summary[key])
                b = float(ref_summary[key])

                if not math.isclose(
                    a, b,
                    rel_tol=1e-7,
                    abs_tol=1e-7,
                ):
                    raise RuntimeError(
                        f"Export rerun differs from frozen "
                        f"result for {key}: {a} vs {b}"
                    )

            with np.load(
                panel,
                allow_pickle=False,
            ) as d:
                x = np.asarray(
                    d["x"],
                    dtype=np.float32,
                )
                available = np.asarray(
                    d["available_gene_mask"],
                    dtype=bool,
                )

            with np.load(
                raw,
                allow_pickle=False,
            ) as d:
                expected = np.asarray(
                    d["expected_repair"],
                    dtype=np.float32,
                )
                probability = np.asarray(
                    d["dropout_probability"],
                    dtype=np.float32,
                )

            if expected.shape != x.shape:
                raise RuntimeError("Expected-repair shape mismatch")

            if probability.shape != x.shape:
                raise RuntimeError("Probability shape mismatch")

            if not np.isfinite(expected).all():
                raise RuntimeError("Nonfinite expected repair")

            if not np.isfinite(probability).all():
                raise RuntimeError("Nonfinite probability")

            eps = float(p["selective_repair"]["zero_threshold"])

            zero_candidate = (
                (np.abs(x) <= eps)
                & available[None, :]
            )

            selected = (
                zero_candidate
                & (probability >= threshold)
            )

            repaired = x.copy()
            repaired[selected] = expected[selected]

            observed = np.abs(x) > eps

            if not np.array_equal(
                repaired[observed],
                x[observed],
            ):
                raise RuntimeError(
                    "Observed nonzero preservation failed"
                )

            not_selected = zero_candidate & ~selected

            if not np.array_equal(
                repaired[not_selected],
                x[not_selected],
            ):
                raise RuntimeError(
                    "Unselected zero was modified"
                )

            matrix = qdir / "selective_reconstruction.npy"

            np.save(
                matrix,
                repaired.astype(
                    np.float32,
                    copy=False,
                ),
                allow_pickle=False,
            )

            raw.unlink()

            metadata = {
                "model_seed": int(task["model_seed"]),
                "replicate": rep,
                "q": q,
                "threshold": threshold,
                "panel": str(panel),
                "panel_sha256": run["panel_sha256"],
                "n_selected_repairs": int(selected.sum()),
                "n_zero_candidates": int(zero_candidate.sum()),
                "selected_fraction": float(
                    selected.sum()
                    / max(int(zero_candidate.sum()), 1)
                ),
                "observed_nonzero_preserved_exactly": True,
                "matrix": str(matrix.name),
                "matrix_sha256": sha(matrix),
                "definition": (
                    "copy observed nonzeros; for available "
                    "zero entries repair with expected_repair "
                    "iff dropout_probability >= frozen threshold"
                ),
            }

            (
                qdir / "representation.json"
            ).write_text(
                json.dumps(
                    metadata,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            records.append(metadata)

        if len(records) != 15:
            raise RuntimeError(
                f"Expected 15 exports, got {len(records)}"
            )

        (
            temp / "task_manifest.json"
        ).write_text(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "model_seed": int(task["model_seed"]),
                    "runs": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        files = sorted(
            [
                f
                for f in temp.rglob("*")
                if (
                    f.is_file()
                    and f.name != "SHA256SUMS.txt"
                )
            ],
            key=lambda f: str(f.relative_to(temp)),
        )

        (
            temp / "SHA256SUMS.txt"
        ).write_text(
            "".join(
                f"{sha(f)}  {f.relative_to(temp)}\n"
                for f in files
            )
        )

        os.replace(temp, final)

    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print("BARON_LODO_SC2_EXPORT_TASK=PASS")
    print(f"TASK_ID={args.task_id}")
    print(f"MODEL_SEED={task['model_seed']}")
    print("EXPORTED_REPRESENTATIONS=15")
    print(f"OUTPUT={final}")


if __name__ == "__main__":
    main()
