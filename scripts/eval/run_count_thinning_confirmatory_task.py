#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL_ID = "sc2-count-thinning-confirmatory-inference-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def q_label(q: float) -> str:
    return f"q{int(round(float(q) * 100)):03d}"


def manifest_records(path: Path) -> dict[str, str]:
    result = {}

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        digest, name = line.split(None, 1)
        name = name.strip()

        if name in result:
            raise RuntimeError(
                f"Duplicate manifest entry: {name}"
            )

        result[name] = digest

    return result


def verify_bound_file(
    path: Path,
    expected_sha: str,
    label: str,
) -> None:
    if not path.is_file():
        raise RuntimeError(
            f"Missing {label}: {path}"
        )

    actual = sha256_file(path)

    if actual != expected_sha:
        raise RuntimeError(
            f"{label} SHA mismatch: "
            f"expected={expected_sha} "
            f"actual={actual} "
            f"path={path}"
        )


def load_protocols(
    inference_protocol_path: Path,
):
    ip = json.loads(
        inference_protocol_path.read_text()
    )

    if ip.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(
            "Unexpected inference protocol_id."
        )

    scientific_path = Path(
        ip["scientific_protocol"]["path"]
    )

    verify_bound_file(
        scientific_path,
        ip["scientific_protocol"]["sha256"],
        "scientific protocol",
    )

    scientific = json.loads(
        scientific_path.read_text()
    )

    receipt_path = Path(
        ip["panel_receipt"]["path"]
    )

    verify_bound_file(
        receipt_path,
        ip["panel_receipt"]["sha256"],
        "panel receipt",
    )

    receipt = json.loads(
        receipt_path.read_text()
    )

    threshold_path = Path(
        ip["threshold_config"]["path"]
    )

    verify_bound_file(
        threshold_path,
        ip["threshold_config"]["sha256"],
        "threshold config",
    )

    thresholds = json.loads(
        threshold_path.read_text()
    )

    return ip, scientific, receipt, thresholds


def resolve_task(
    *,
    inference_protocol_path: Path,
    task_id: int,
    verify_panel_payloads: bool = True,
) -> dict[str, Any]:

    ip, scientific, receipt, threshold_config = (
        load_protocols(
            inference_protocol_path
        )
    )

    tasks = (
        scientific[
            "confirmatory_inference"
        ][
            "task_layout"
        ][
            "tasks"
        ]
    )

    matches = [
        x
        for x in tasks
        if int(x["task_id"]) == task_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one task_id={task_id}; "
            f"found {len(matches)}"
        )

    task = matches[0]

    model_seed = int(
        task["model_seed"]
    )

    dataset = str(
        task["dataset"]
    )

    replicate = int(
        task["replicate"]
    )

    checkpoint_info = (
        scientific[
            "confirmatory_inference"
        ][
            "model_checkpoints"
        ][str(model_seed)]
    )

    checkpoint = Path(
        checkpoint_info["path"]
    )

    verify_bound_file(
        checkpoint,
        checkpoint_info["sha256"],
        "model checkpoint",
    )

    dataset_spec = (
        scientific[
            "materialization"
        ][
            "datasets"
        ][dataset]
    )

    receipt_dataset = (
        receipt[
            "datasets"
        ][dataset]
    )

    panel_root = Path(
        receipt_dataset["root"]
    )

    manifest_path = Path(
        receipt_dataset[
            "bundle_manifest_path"
        ]
    )

    verify_bound_file(
        manifest_path,
        receipt_dataset[
            "bundle_manifest_sha256"
        ],
        "panel manifest",
    )

    manifest = manifest_records(
        manifest_path
    )

    primary_thresholds = (
        scientific[
            "confirmatory_inference"
        ][
            "primary_thresholds"
        ]
    )

    frozen_thresholds = (
        threshold_config[
            "thresholds"
        ]
    )

    evaluations = []

    for q in (0.85, 0.70, 0.50):

        matches = [
            item
            for item in dataset_spec[
                "panel_specs"
            ]
            if (
                int(item["replicate"])
                == replicate
                and math.isclose(
                    float(item["q"]),
                    q,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        ]

        if len(matches) != 1:
            raise RuntimeError(
                "Panel-spec resolution failed "
                f"dataset={dataset} "
                f"replicate={replicate} q={q}"
            )

        spec = matches[0]
        thinning_seed = int(spec["seed"])

        filename = (
            f"{dataset}_{q_label(q)}_"
            f"rep{replicate:02d}_"
            f"seed{thinning_seed}.npz"
        )

        if filename not in manifest:
            raise RuntimeError(
                f"Panel absent from manifest: {filename}"
            )

        panel = panel_root / filename

        if not panel.is_file():
            raise RuntimeError(
                f"Panel missing: {panel}"
            )

        if verify_panel_payloads:

            if sha256_file(panel) != manifest[filename]:
                raise RuntimeError(
                    f"Panel SHA mismatch: {panel}"
                )

            with np.load(
                panel,
                allow_pickle=False,
            ) as d:

                if str(d["dataset"].item()) != dataset:
                    raise RuntimeError(
                        "Panel dataset provenance mismatch."
                    )

                if int(
                    d["thinning_replicate"].item()
                ) != replicate:
                    raise RuntimeError(
                        "Panel replicate provenance mismatch."
                    )

                if int(
                    d["thinning_seed"].item()
                ) != thinning_seed:
                    raise RuntimeError(
                        "Panel seed provenance mismatch."
                    )

                if not math.isclose(
                    float(
                        d["thinning_q"].item()
                    ),
                    q,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise RuntimeError(
                        "Panel q provenance mismatch."
                    )

        primary = float(
            primary_thresholds[f"{q:.2f}"]
        )

        frozen = float(
            frozen_thresholds[f"{q:.2f}"]
        )

        if not math.isclose(
            primary,
            frozen,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Primary threshold differs from "
                "frozen validation-selected threshold."
            )

        evaluations.append(
            {
                "q": q,
                "q_label": q_label(q),
                "threshold": primary,
                "panel_path": panel,
                "panel_sha256": manifest[filename],
                "thinning_seed": thinning_seed,
            }
        )

    return {
        "inference_protocol":
            ip,

        "scientific_protocol":
            scientific,

        "task_id":
            task_id,

        "model_seed":
            model_seed,

        "dataset":
            dataset,

        "replicate":
            replicate,

        "checkpoint_path":
            checkpoint,

        "checkpoint_sha256":
            checkpoint_info["sha256"],

        "evaluations":
            evaluations,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--protocol",
        required=True,
    )

    p.add_argument(
        "--task-id",
        type=int,
        required=True,
    )

    p.add_argument(
        "--output-root",
        required=True,
    )

    return p.parse_args()


def main() -> None:

    args = parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    resolved = resolve_task(
        inference_protocol_path=
            protocol_path,
        task_id=args.task_id,
        verify_panel_payloads=True,
    )

    ip = resolved[
        "inference_protocol"
    ]

    expected_output_root = Path(
        ip["output"]["root"]
    ).resolve()

    output_root = Path(
        args.output_root
    ).resolve()

    if output_root != expected_output_root:
        raise RuntimeError(
            "Output root differs from "
            "frozen protocol."
        )

    evaluator_config = Path(
        ip["evaluator_config"]["path"]
    )

    verify_bound_file(
        evaluator_config,
        ip["evaluator_config"]["sha256"],
        "evaluator config",
    )

    verify_bound_file(
        Path(ip["evaluator"]["path"]),
        ip["evaluator"]["sha256"],
        "evaluator",
    )

    task_id = resolved["task_id"]
    model_seed = resolved["model_seed"]
    dataset = resolved["dataset"]
    replicate = resolved["replicate"]

    task_name = (
        f"task_{task_id:02d}"
        f"_model{model_seed}"
        f"_{dataset}"
        f"_rep{replicate:02d}"
    )

    final_dir = output_root / task_name

    if final_dir.exists():
        raise RuntimeError(
            f"Final task output already exists: "
            f"{final_dir}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir = (
        output_root
        / f".tmp_{task_name}_{os.getpid()}"
    )

    if temp_dir.exists():
        raise RuntimeError(
            f"Temporary task output exists: {temp_dir}"
        )

    temp_dir.mkdir()

    task_records = []

    try:
        for evaluation in resolved[
            "evaluations"
        ]:

            q = float(evaluation["q"])
            q_dir = (
                temp_dir
                / evaluation["q_label"]
            )

            q_dir.mkdir()

            threshold = float(
                evaluation["threshold"]
            )

            command = [
                sys.executable,
                "-m",
                "sc2.eval.evaluate_continuous_repair",
                "--config",
                str(evaluator_config),
                "--checkpoint",
                str(
                    resolved[
                        "checkpoint_path"
                    ]
                ),
                "--benchmark",
                str(
                    evaluation[
                        "panel_path"
                    ]
                ),
                "--output-dir",
                str(q_dir),
                "--threshold",
                str(threshold),
            ]

            print(
                "RUNNING_EVALUATION="
                + json.dumps(
                    {
                        "task_id": task_id,
                        "model_seed": model_seed,
                        "dataset": dataset,
                        "replicate": replicate,
                        "q": q,
                        "threshold": threshold,
                        "benchmark":
                            str(
                                evaluation[
                                    "panel_path"
                                ]
                            ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            subprocess.run(
                command,
                check=True,
            )

            required = [
                q_dir / "summary.json",
                q_dir / "summary.csv",
                q_dir / "threshold_sweep.csv",
                q_dir / "risk_coverage.csv",
            ]

            for path in required:
                if not path.is_file():
                    raise RuntimeError(
                        f"Missing evaluator output: {path}"
                    )

            summary = json.loads(
                (
                    q_dir
                    / "summary.json"
                ).read_text()
            )

            if (
                summary["threshold_source"]
                != "frozen_external_threshold"
            ):
                raise RuntimeError(
                    "Automatic threshold selection "
                    "was used in confirmatory inference."
                )

            if not math.isclose(
                float(summary["threshold"]),
                threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Evaluator threshold mismatch."
                )

            sweep = pd.read_csv(
                q_dir
                / "threshold_sweep.csv"
            )

            if len(sweep) != 201:
                raise RuntimeError(
                    f"Expected 201 threshold rows, "
                    f"got {len(sweep)}"
                )

            if set(sweep.columns) != {
                "threshold",
                "recall",
                "precision",
                "true_zero_fill",
                "selected",
            }:
                raise RuntimeError(
                    "Unexpected threshold sweep columns."
                )

            risk = pd.read_csv(
                q_dir
                / "risk_coverage.csv"
            )

            if len(risk) != 100:
                raise RuntimeError(
                    f"Expected 100 risk rows, got {len(risk)}"
                )

            task_records.append(
                {
                    "task_id":
                        task_id,

                    "model_seed":
                        model_seed,

                    "dataset":
                        dataset,

                    "replicate":
                        replicate,

                    "q":
                        q,

                    "threshold":
                        threshold,

                    "threshold_source":
                        summary[
                            "threshold_source"
                        ],

                    "benchmark":
                        str(
                            evaluation[
                                "panel_path"
                            ]
                        ),

                    "benchmark_sha256":
                        evaluation[
                            "panel_sha256"
                        ],

                    "thinning_seed":
                        evaluation[
                            "thinning_seed"
                        ],

                    "checkpoint":
                        str(
                            resolved[
                                "checkpoint_path"
                            ]
                        ),

                    "checkpoint_sha256":
                        resolved[
                            "checkpoint_sha256"
                        ],

                    # Preserve the complete evaluator output
                    # without assuming metric field names here.
                    "evaluator_summary":
                        summary,

                    "output_subdir":
                        evaluation[
                            "q_label"
                        ],
                }
            )

        if len(task_records) != 3:
            raise RuntimeError(
                "Expected exactly 3 q evaluations."
            )

        task_manifest = {
            "schema_version":
                "sc2-count-thinning-confirmatory-task-v1",

            "task_id":
                task_id,

            "model_seed":
                model_seed,

            "dataset":
                dataset,

            "replicate":
                replicate,

            "inference_protocol_path":
                str(protocol_path),

            "inference_protocol_sha256":
                sha256_file(protocol_path),

            "per_panel_threshold_selection_used":
                False,

            "old_transfer_threshold_gpu_rerun":
                False,

            "evaluations":
                task_records,
        }

        write_json(
            temp_dir
            / "task_manifest.json",
            task_manifest,
        )

        files = sorted(
            [
                p
                for p in temp_dir.rglob("*")
                if (
                    p.is_file()
                    and p.name
                    != "SHA256SUMS.txt"
                )
            ],
            key=lambda x:
                str(
                    x.relative_to(
                        temp_dir
                    )
                ),
        )

        manifest_path = (
            temp_dir
            / "SHA256SUMS.txt"
        )

        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  "
                f"{path.relative_to(temp_dir)}\n"
                for path in files
            )
        )

        for line in (
            manifest_path
            .read_text()
            .splitlines()
        ):
            digest, rel = line.split(
                None,
                1,
            )

            if (
                sha256_file(
                    temp_dir
                    / rel.strip()
                )
                != digest
            ):
                raise RuntimeError(
                    "Task manifest self-check failed."
                )

        if final_dir.exists():
            raise RuntimeError(
                "Final directory appeared "
                "before atomic finalization."
            )

        os.replace(
            temp_dir,
            final_dir,
        )

    except Exception:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )
        raise

    print(
        "PHASE1C_CONFIRMATORY_INFERENCE_TASK=PASS"
    )
    print(f"TASK_ID={task_id}")
    print(f"MODEL_SEED={model_seed}")
    print(f"DATASET={dataset}")
    print(f"REPLICATE={replicate}")
    print("EVALUATIONS=3")
    print(f"OUTPUT={final_dir}")


if __name__ == "__main__":
    main()
