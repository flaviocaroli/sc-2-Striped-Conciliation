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

import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--task-id", required=True, type=int)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    protocol_path = Path(args.protocol).resolve()
    p = json.loads(protocol_path.read_text())

    expected = os.environ.get("SC2_ABLATION_VALIDATION_PROTOCOL_SHA")
    if expected and sha(protocol_path) != expected:
        raise RuntimeError("protocol SHA mismatch")

    output_root = Path(args.output_root).resolve()
    if output_root != Path(p["output_root"]).resolve():
        raise RuntimeError("output root mismatch")

    tasks = {
        int(x["task_id"]): x
        for x in p["tasks"]
    }

    if args.task_id not in tasks:
        raise RuntimeError("unknown task")

    task = tasks[args.task_id]

    # Verify everything before creating output.
    for run in task["model_runs"]:
        ckpt = Path(run["checkpoint"])
        if not ckpt.is_file():
            raise RuntimeError(f"missing checkpoint {ckpt}")
        if sha(ckpt) != run["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch {ckpt}")

    for panel in p["validation_panels"].values():
        path = Path(panel["path"])
        if not path.is_file():
            raise RuntimeError(f"missing panel {path}")
        if sha(path) != panel["sha256"]:
            raise RuntimeError(f"panel SHA mismatch {path}")

    if args.check_only:
        n = len(task["model_runs"]) * 3
        print(
            f"ABLATION_VALIDATION_CHECK=PASS "
            f"TASK={args.task_id} "
            f"MODEL_RUNS={len(task['model_runs'])} "
            f"EVALUATIONS={n}"
        )
        return

    final = output_root / f"task_{args.task_id:02d}"
    temp = output_root / f".tmp_task_{args.task_id:02d}_{os.getpid()}"

    if final.exists() or temp.exists():
        raise RuntimeError("output collision")

    output_root.mkdir(parents=True, exist_ok=True)
    temp.mkdir()

    records = []

    try:
        base_cfg = yaml.safe_load(
            Path(p["evaluator"]["config_path"]).read_text()
        )

        for run in task["model_runs"]:
            variant = run["variant"]
            seed = int(run["seed"])
            checkpoint = Path(run["checkpoint"])

            # Architecture-specific evaluation config.
            cfg = json.loads(json.dumps(base_cfg))

            if variant == "A1_no_routed_attention":
                cfg["model"]["n_attention_checkpoints"] = 0

            elif variant == "A2_unidirectional_mamba":
                cfg["model"]["bidirectional_mamba"] = False

            elif variant in {
                "A0_full",
                "A3_no_expected_zero",
                "A4_no_gate_loss",
                "A5_cell_structure",
            }:
                pass

            else:
                raise RuntimeError(f"unknown variant {variant}")

            model_dir = temp / variant / f"seed{seed}"
            model_dir.mkdir(parents=True)

            resolved_cfg = model_dir / "resolved_eval_config.yaml"
            resolved_cfg.write_text(
                yaml.safe_dump(cfg, sort_keys=False)
            )

            for mask in (15, 30, 50):
                panel = Path(
                    p["validation_panels"][str(mask)]["path"]
                )

                out = model_dir / f"mask{mask}"

                cmd = [
                    sys.executable,
                    "-m",
                    "sc2.eval.evaluate_continuous_repair",
                    "--config",
                    str(resolved_cfg),
                    "--checkpoint",
                    str(checkpoint),
                    "--benchmark",
                    str(panel),
                    "--output-dir",
                    str(out),
                    # Intentionally NO --threshold:
                    # this is validation and we need the full sweep.
                ]

                subprocess.run(cmd, check=True)

                for name in (
                    "summary.json",
                    "summary.csv",
                    "threshold_sweep.csv",
                    "risk_coverage.csv",
                ):
                    if not (out / name).is_file():
                        raise RuntimeError(
                            f"missing evaluator output {out/name}"
                        )

                records.append({
                    "variant": variant,
                    "seed": seed,
                    "mask_percent": mask,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": run["checkpoint_sha256"],
                    "summary": str(out / "summary.json"),
                    "threshold_sweep": str(out / "threshold_sweep.csv"),
                })

                print(
                    f"EVALUATED variant={variant} "
                    f"seed={seed} mask={mask}",
                    flush=True,
                )

        (temp / "task_manifest.json").write_text(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "evaluations": records,
                    "threshold_selected_here": False,
                    "test_data_used": False,
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )

        os.replace(temp, final)

    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print("PHASE4_ABLATION_VALIDATION_TASK=PASS")
    print(f"TASK_ID={args.task_id}")
    print(f"EVALUATIONS={len(records)}")


if __name__ == "__main__":
    main()
