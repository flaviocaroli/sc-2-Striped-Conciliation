#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run one frozen SC2 validation-thinning task: "
            "one model seed x one replicate, all q levels."
        )
    )
    p.add_argument("--protocol", required=True)
    p.add_argument("--task-id", required=True, type=int)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def q_label(q: float) -> str:
    return f"q{int(round(q * 100)):03d}"


def find_panel(
    panel_receipt: dict,
    *,
    q: float,
    replicate: int,
) -> dict:
    matches = [
        item
        for item in panel_receipt["panel_files"]
        if abs(float(item["q"]) - q) < 1.0e-12
        and int(item["replicate"]) == replicate
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one panel for q={q}, "
            f"replicate={replicate}; found {len(matches)}"
        )

    return matches[0]


def verify_sweep(
    path: Path,
    direct_threshold: float,
) -> None:
    sweep = pd.read_csv(path)

    required = {
        "threshold",
        "recall",
        "precision",
        "true_zero_fill",
        "selected",
    }

    if set(sweep.columns) != required:
        raise RuntimeError(
            f"Unexpected threshold-sweep columns: "
            f"{list(sweep.columns)}"
        )

    if len(sweep) != 201:
        raise RuntimeError(
            f"Expected 201 thresholds, got {len(sweep)}"
        )

    expected = np.linspace(
        0.0,
        1.0,
        201,
    )

    if not np.allclose(
        sweep["threshold"].to_numpy(
            dtype=np.float64
        ),
        expected,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Threshold grid differs from frozen 0:0.005:1 grid."
        )

    if not np.any(
        np.isclose(
            expected,
            direct_threshold,
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise RuntimeError(
            "Direct-transfer threshold is not on frozen grid."
        )


def write_sha_manifest(root: Path) -> None:
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name != "SHA256SUMS.txt"
    )

    lines = []

    for path in files:
        rel = path.relative_to(root)
        lines.append(
            f"{sha256_file(path)}  {rel.as_posix()}"
        )

    (
        root
        / "SHA256SUMS.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    protocol = json.loads(
        protocol_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        protocol["protocol_id"]
        != "sc2-count-thinning-validation-inference-v1"
    ):
        raise RuntimeError(
            "Wrong Phase-1B protocol."
        )

    tasks = protocol["task_layout"]["tasks"]

    matches = [
        task
        for task in tasks
        if int(task["task_id"]) == args.task_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Unknown task_id={args.task_id}"
        )

    task = matches[0]

    model_seed = int(
        task["model_seed"]
    )

    replicate = int(
        task["replicate"]
    )

    model_info = (
        protocol["models"][
            str(model_seed)
        ]
    )

    checkpoint = Path(
        model_info["checkpoint_path"]
    )

    if not checkpoint.is_file():
        raise RuntimeError(
            f"Checkpoint missing: {checkpoint}"
        )

    checkpoint_sha = sha256_file(
        checkpoint
    )

    if (
        checkpoint_sha
        != model_info["checkpoint_sha256"]
    ):
        raise RuntimeError(
            "Checkpoint SHA mismatch."
        )

    eval_config = Path(
        protocol["evaluator"]["config_path"]
    )

    if (
        sha256_file(eval_config)
        != protocol["evaluator"]["config_sha256"]
    ):
        raise RuntimeError(
            "Eval-config SHA mismatch."
        )

    panel_receipt_path = Path(
        protocol["panels"]["receipt_path"]
    )

    if (
        sha256_file(panel_receipt_path)
        != protocol["panels"]["receipt_sha256"]
    ):
        raise RuntimeError(
            "Panel-receipt SHA mismatch."
        )

    panel_receipt = json.loads(
        panel_receipt_path.read_text(
            encoding="utf-8"
        )
    )

    panel_root = Path(
        protocol["panels"]["root"]
    )

    output_root = Path(
        protocol["output"]["root"]
    )

    task_name = (
        f"task_{args.task_id:02d}"
        f"_model{model_seed}"
        f"_rep{replicate:02d}"
    )

    final_root = (
        output_root
        / task_name
    )

    job_token = (
        os.environ.get(
            "SLURM_JOB_ID",
            "manual",
        )
        + "_"
        + os.environ.get(
            "SLURM_ARRAY_TASK_ID",
            str(args.task_id),
        )
    )

    temp_root = (
        output_root
        / (
            f".tmp_{task_name}_"
            f"{job_token}"
        )
    )

    if final_root.exists():
        raise RuntimeError(
            f"Final task output already exists: "
            f"{final_root}"
        )

    if temp_root.exists():
        raise RuntimeError(
            f"Temporary task output already exists: "
            f"{temp_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_root.mkdir(
        parents=False,
        exist_ok=False,
    )

    task_records = []

    try:
        for q in protocol[
            "task_layout"
        ][
            "q_order"
        ]:
            q = float(q)

            panel_info = find_panel(
                panel_receipt,
                q=q,
                replicate=replicate,
            )

            benchmark = (
                panel_root
                / panel_info["file"]
            )

            if not benchmark.is_file():
                raise RuntimeError(
                    f"Panel missing: {benchmark}"
                )

            if (
                sha256_file(benchmark)
                != panel_info["sha256"]
            ):
                raise RuntimeError(
                    f"Panel SHA mismatch: {benchmark}"
                )

            threshold = float(
                protocol[
                    "direct_transfer_thresholds"
                ][
                    f"{q:.2f}"
                ]
            )

            q_dir = (
                temp_root
                / q_label(q)
            )

            command = [
                sys.executable,
                "-u",
                "-m",
                "sc2.eval.evaluate_continuous_repair",
                "--config",
                str(eval_config),
                "--checkpoint",
                str(checkpoint),
                "--benchmark",
                str(benchmark),
                "--output-dir",
                str(q_dir),
                "--threshold",
                str(threshold),
            ]

            print(
                "RUNNING "
                + json.dumps(
                    {
                        "task_id": args.task_id,
                        "model_seed": model_seed,
                        "replicate": replicate,
                        "q": q,
                        "threshold": threshold,
                        "benchmark": str(
                            benchmark
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

            summary_path = (
                q_dir
                / "summary.json"
            )

            sweep_path = (
                q_dir
                / "threshold_sweep.csv"
            )

            risk_path = (
                q_dir
                / "risk_coverage.csv"
            )

            for required in (
                summary_path,
                sweep_path,
                risk_path,
                q_dir / "summary.csv",
            ):
                if not required.is_file():
                    raise RuntimeError(
                        f"Missing evaluator output: "
                        f"{required}"
                    )

            summary = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                summary["threshold_source"]
                != "frozen_external_threshold"
            ):
                raise RuntimeError(
                    "Per-panel automatic threshold "
                    "selection was unexpectedly used."
                )

            if not np.isclose(
                float(summary["threshold"]),
                threshold,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Evaluator summary threshold "
                    "differs from direct-transfer "
                    "threshold."
                )

            verify_sweep(
                sweep_path,
                threshold,
            )

            target_sd = float(
                summary["target_sd"]
            )

            masked_mse = float(
                summary["masked_mse"]
            )

            if not np.isfinite(
                target_sd
            ) or target_sd <= 0.0:
                raise RuntimeError(
                    "Invalid target_sd."
                )

            recovery_index = (
                1.0
                - masked_mse
                / (target_sd ** 2)
            )

            task_records.append(
                {
                    "q": q,
                    "replicate": replicate,
                    "thinning_seed": int(
                        panel_info["seed"]
                    ),
                    "model_seed": model_seed,
                    "benchmark_file": (
                        panel_info["file"]
                    ),
                    "benchmark_sha256": (
                        panel_info["sha256"]
                    ),
                    "checkpoint_sha256": (
                        checkpoint_sha
                    ),
                    "direct_transfer_threshold": (
                        threshold
                    ),
                    "masked_mse": masked_mse,
                    "masked_mae": float(
                        summary["masked_mae"]
                    ),
                    "target_sd": target_sd,
                    "recovery_index": float(
                        recovery_index
                    ),
                    "gene_spearman": float(
                        summary["gene_spearman"]
                    ),
                    "sample_spearman": float(
                        summary[
                            "sample_spearman"
                        ]
                    ),
                    "sd_ratio": float(
                        summary["sd_ratio"]
                    ),
                    "threshold_recall": float(
                        summary[
                            "threshold_recall"
                        ]
                    ),
                    "threshold_precision": float(
                        summary[
                            "threshold_precision"
                        ]
                    ),
                    "threshold_true_zero_fill": float(
                        summary[
                            "threshold_true_zero_fill"
                        ]
                    ),
                    "output_subdir": (
                        q_label(q)
                    ),
                }
            )

        if len(task_records) != 3:
            raise RuntimeError(
                "Expected three q evaluations."
            )

        task_manifest = {
            "schema_version": 1,
            "protocol_id": protocol[
                "protocol_id"
            ],
            "protocol_path": str(
                protocol_path
            ),
            "protocol_sha256": (
                sha256_file(
                    protocol_path
                )
            ),
            "task_id": args.task_id,
            "model_seed": model_seed,
            "replicate": replicate,
            "evaluations": (
                task_records
            ),
            "per_panel_threshold_selection_used": False,
            "direct_transfer_only_at_run_time": True,
            "final_common_threshold_not_selected_here": True,
            "internal_test_accessed": False,
            "baron_accessed": False,
            "zheng68k_accessed": False,
        }

        (
            temp_root
            / "task_manifest.json"
        ).write_text(
            json.dumps(
                task_manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        write_sha_manifest(
            temp_root
        )

        if final_root.exists():
            raise RuntimeError(
                "Final output appeared "
                "during task execution."
            )

        temp_root.rename(
            final_root
        )

        print(
            "PHASE1B_VALIDATION_TASK=PASS"
        )
        print(
            f"TASK_ID={args.task_id}"
        )
        print(
            f"MODEL_SEED={model_seed}"
        )
        print(
            f"REPLICATE={replicate}"
        )
        print(
            f"OUTPUT={final_root}"
        )

    except Exception:
        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )
        raise


if __name__ == "__main__":
    main()
