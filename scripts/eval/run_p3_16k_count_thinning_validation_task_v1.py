#!/usr/bin/env python3

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


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)
    return h.hexdigest()


def qlabel(q: float) -> str:
    return f"q{int(round(q * 100)):03d}"


def verify_sweep(path: Path) -> None:
    d = pd.read_csv(path)

    required = {
        "threshold",
        "recall",
        "precision",
        "true_zero_fill",
    }

    if not required <= set(d.columns):
        raise RuntimeError(
            f"bad sweep columns: {list(d.columns)}"
        )

    if len(d) != 201:
        raise RuntimeError(
            f"expected 201 thresholds, got {len(d)}"
        )

    expected = np.linspace(
        0.0,
        1.0,
        201,
    )

    actual = d["threshold"].to_numpy(
        dtype=np.float64
    )

    if not np.allclose(
        actual,
        expected,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "threshold grid changed"
        )


def write_manifest(root: Path) -> None:
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name != "SHA256SUMS.txt"
    )

    (
        root / "SHA256SUMS.txt"
    ).write_text(
        "".join(
            f"{sha(p)}  "
            f"{p.relative_to(root).as_posix()}\n"
            for p in files
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--protocol",
        required=True,
    )

    ap.add_argument(
        "--task-id",
        required=True,
        type=int,
    )

    ap.add_argument(
        "--check-only",
        action="store_true",
    )

    args = ap.parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    p = json.loads(
        protocol_path.read_text()
    )

    if p["protocol_id"] != \
            "sc2-p3-16k-count-thinning-validation-inference-v1":
        raise RuntimeError(
            "wrong P3 thinning protocol"
        )

    expected_protocol_sha = os.environ.get(
        "SC2_P3_THIN_PROTOCOL_SHA"
    )

    if (
        expected_protocol_sha
        and sha(protocol_path)
        != expected_protocol_sha
    ):
        raise RuntimeError(
            "protocol SHA mismatch"
        )

    tasks = {
        int(x["task_id"]): x
        for x in p["task_layout"]["tasks"]
    }

    if len(tasks) != 25:
        raise RuntimeError(
            "expected 25 tasks"
        )

    if args.task_id not in tasks:
        raise RuntimeError(
            "unknown task id"
        )

    task = tasks[args.task_id]

    seed = int(task["model_seed"])
    rep = int(task["replicate"])

    model = p["models"][str(seed)]

    config = Path(
        model["config_path"]
    )

    checkpoint = Path(
        model["checkpoint_path"]
    )

    if not config.is_file():
        raise RuntimeError(
            f"missing config: {config}"
        )

    if not checkpoint.is_file():
        raise RuntimeError(
            f"missing checkpoint: {checkpoint}"
        )

    if sha(config) != model["config_sha256"]:
        raise RuntimeError(
            "config SHA mismatch"
        )

    if sha(checkpoint) != \
            model["checkpoint_sha256"]:
        raise RuntimeError(
            "checkpoint SHA mismatch"
        )

    receipt_path = Path(
        p["panels"]["receipt_path"]
    )

    if sha(receipt_path) != \
            p["panels"]["receipt_sha256"]:
        raise RuntimeError(
            "panel receipt SHA mismatch"
        )

    receipt = json.loads(
        receipt_path.read_text()
    )

    if receipt["status"] != "PASS":
        raise RuntimeError(
            "panel receipt not PASS"
        )

    if receipt["panel_count"] != 15:
        raise RuntimeError(
            "panel receipt count mismatch"
        )

    panels = receipt["panel_files"]

    selected_panels = {}

    for q in p["task_layout"]["q_order"]:

        q = float(q)

        matches = [
            x for x in panels
            if (
                abs(float(x["q"]) - q)
                < 1e-12
                and int(x["replicate"])
                == rep
            )
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"panel lookup failed q={q}, rep={rep}"
            )

        item = matches[0]
        panel = Path(item["path"])

        if not panel.is_file():
            raise RuntimeError(
                f"missing panel {panel}"
            )

        if sha(panel) != item["sha256"]:
            raise RuntimeError(
                f"panel SHA mismatch {panel}"
            )

        selected_panels[q] = (
            item,
            panel,
        )

    if args.check_only:
        print(
            "P3_THINNING_TASK_PRECHECK=PASS "
            f"TASK={args.task_id} "
            f"SEED={seed} "
            f"REP={rep}"
        )
        return

    output_root = Path(
        p["output"]["root"]
    )

    final = (
        output_root
        / (
            f"task_{args.task_id:02d}"
            f"_model{seed}"
            f"_rep{rep:02d}"
        )
    )

    temp = (
        output_root
        / (
            f".tmp_task_{args.task_id:02d}"
            f"_model{seed}"
            f"_rep{rep:02d}"
            f"_{os.getpid()}"
        )
    )

    if final.exists() or temp.exists():
        raise RuntimeError(
            "output collision"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp.mkdir()

    placeholder = float(
        p["runtime_control"][
            "placeholder_threshold"
        ]
    )

    assert placeholder == 1.0

    records = []

    try:
        for q in p["task_layout"]["q_order"]:

            q = float(q)

            item, panel = selected_panels[q]

            dest = temp / qlabel(q)

            command = [
                sys.executable,
                "-u",
                "-m",
                "sc2.eval.evaluate_continuous_repair",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--benchmark",
                str(panel),
                "--output-dir",
                str(dest),
                "--threshold",
                str(placeholder),
            ]

            print(
                "RUNNING "
                + json.dumps(
                    {
                        "task_id":
                            args.task_id,
                        "model_seed":
                            seed,
                        "replicate":
                            rep,
                        "q":
                            q,
                        "runtime_placeholder_threshold":
                            placeholder,
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
                dest / "summary.json"
            )

            sweep_path = (
                dest / "threshold_sweep.csv"
            )

            risk_path = (
                dest / "risk_coverage.csv"
            )

            csv_path = (
                dest / "summary.csv"
            )

            for required in (
                summary_path,
                sweep_path,
                risk_path,
                csv_path,
            ):
                if not required.is_file():
                    raise RuntimeError(
                        f"missing evaluator output: "
                        f"{required}"
                    )

            s = json.loads(
                summary_path.read_text()
            )

            if s["checkpoint"] != \
                    str(checkpoint):
                raise RuntimeError(
                    "summary checkpoint mismatch"
                )

            if s["benchmark"] != \
                    str(panel):
                raise RuntimeError(
                    "summary panel mismatch"
                )

            if int(
                s["n_available_genes"]
            ) != 16384:
                raise RuntimeError(
                    "available gene count mismatch"
                )

            if float(
                s["available_gene_fraction"]
            ) != 1.0:
                raise RuntimeError(
                    "availability fraction mismatch"
                )

            if s["threshold_source"] != \
                    "frozen_external_threshold":
                raise RuntimeError(
                    "unexpected automatic threshold selection"
                )

            if not np.isclose(
                float(s["threshold"]),
                placeholder,
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "runtime threshold mismatch"
                )

            if float(
                s["observed_nonzero_changed_fraction"]
            ) != 0.0:
                raise RuntimeError(
                    "observed nonzero changed"
                )

            for key in (
                "observed_nonzero_mse",
                "observed_nonzero_mae",
                "observed_nonzero_max_abs_error",
            ):
                if float(s[key]) != 0.0:
                    raise RuntimeError(
                        f"{key} not zero"
                    )

            verify_sweep(
                sweep_path
            )

            target_sd = float(
                s["target_sd"]
            )

            masked_mse = float(
                s["masked_mse"]
            )

            if (
                not np.isfinite(target_sd)
                or target_sd <= 0.0
            ):
                raise RuntimeError(
                    "invalid target SD"
                )

            recovery = (
                1.0
                - masked_mse
                / (target_sd ** 2)
            )

            records.append({
                "q":
                    q,

                "model_seed":
                    seed,

                "replicate":
                    rep,

                "thinning_seed":
                    int(item["seed"]),

                "benchmark":
                    str(panel),

                "benchmark_sha256":
                    item["sha256"],

                "config":
                    str(config),

                "config_sha256":
                    model["config_sha256"],

                "checkpoint":
                    str(checkpoint),

                "checkpoint_sha256":
                    model["checkpoint_sha256"],

                "masked_mse":
                    masked_mse,

                "masked_mae":
                    float(s["masked_mae"]),

                "recovery_index":
                    float(recovery),

                "gene_spearman":
                    float(s["gene_spearman"]),

                "sample_spearman":
                    float(s["sample_spearman"]),

                "sd_ratio":
                    float(s["sd_ratio"]),

                "gate_auroc":
                    float(s["gate_auroc"]),

                "gate_auprc":
                    float(s["gate_auprc"]),

                "runtime_placeholder_threshold":
                    placeholder,

                "runtime_threshold_scientific_result":
                    False,

                "scientific_threshold_selected_here":
                    False,
            })

        if len(records) != 3:
            raise RuntimeError(
                "expected three q evaluations"
            )

        manifest = {
            "schema":
                "sc2-p3-16k-count-thinning-validation-task-v1",

            "status":
                "PASS",

            "task_id":
                args.task_id,

            "model_seed":
                seed,

            "replicate":
                rep,

            "evaluations":
                records,

            "protocol":
                str(protocol_path),

            "protocol_sha256":
                sha(protocol_path),

            "runtime_placeholder_threshold":
                placeholder,

            "runtime_placeholder_is_scientific_threshold":
                False,

            "full_201_point_sweeps_saved":
                True,

            "per_panel_threshold_selection_used":
                False,

            "scientific_threshold_selection_deferred":
                True,

            "internal_test_accessed":
                False,

            "baron_accessed":
                False,

            "zheng68k_accessed":
                False,
        }

        (
            temp / "task_manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n"
        )

        write_manifest(
            temp
        )

        os.replace(
            temp,
            final,
        )

    except Exception:
        shutil.rmtree(
            temp,
            ignore_errors=True,
        )
        raise

    print(
        "P3_16K_THINNING_VALIDATION_TASK=PASS"
    )

    print(
        f"TASK_ID={args.task_id}"
    )

    print(
        f"MODEL_SEED={seed}"
    )

    print(
        f"REPLICATE={rep}"
    )


if __name__ == "__main__":
    main()
