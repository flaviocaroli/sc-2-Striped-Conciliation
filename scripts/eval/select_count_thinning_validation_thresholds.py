#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate frozen SC2 thinning-validation "
            "threshold sweeps and select one common "
            "threshold per q."
        )
    )
    p.add_argument("--protocol", required=True)
    p.add_argument("--output-dir", required=True)
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


def verify_manifest(
    root: Path,
) -> None:
    manifest = (
        root
        / "SHA256SUMS.txt"
    )

    if not manifest.is_file():
        raise RuntimeError(
            f"Missing task manifest: {manifest}"
        )

    for line in manifest.read_text().splitlines():
        digest, rel = line.split(
            None,
            1,
        )

        path = root / rel.strip()

        if not path.is_file():
            raise RuntimeError(
                f"Missing hashed file: {path}"
            )

        if sha256_file(path) != digest:
            raise RuntimeError(
                f"SHA mismatch: {path}"
            )


def aggregate_threshold_grid(
    runs: pd.DataFrame,
    *,
    max_true_zero_fill: float,
) -> pd.DataFrame:
    required = {
        "run_id",
        "threshold",
        "recall",
        "precision",
        "true_zero_fill",
    }

    missing = required.difference(
        runs.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    expected_runs = (
        runs["run_id"]
        .drop_duplicates()
        .shape[0]
    )

    rows = []

    for threshold, group in runs.groupby(
        "threshold",
        sort=True,
    ):
        if (
            group["run_id"]
            .nunique()
            != expected_runs
        ):
            raise ValueError(
                "Threshold grid is incomplete."
            )

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                "n_runs": int(
                    expected_runs
                ),
                "mean_recall": float(
                    group["recall"].mean()
                ),
                "mean_precision": float(
                    group[
                        "precision"
                    ].mean()
                ),
                "mean_true_zero_fill": float(
                    group[
                        "true_zero_fill"
                    ].mean()
                ),
                "max_true_zero_fill": float(
                    group[
                        "true_zero_fill"
                    ].max()
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result["eligible"] = (
        result["max_true_zero_fill"]
        <= float(max_true_zero_fill)
    )

    return result


def select_common_threshold(
    aggregate: pd.DataFrame,
) -> pd.Series:
    eligible = aggregate[
        aggregate["eligible"]
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No threshold satisfies the "
            "every-run originally-zero-fill gate."
        )

    ordered = eligible.sort_values(
        [
            "mean_recall",
            "mean_precision",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
        ],
        kind="mergesort",
    )

    return ordered.iloc[0]


def find_task_dir(
    root: Path,
    *,
    task_id: int,
    model_seed: int,
    replicate: int,
) -> Path:
    return (
        root
        / (
            f"task_{task_id:02d}"
            f"_model{model_seed}"
            f"_rep{replicate:02d}"
        )
    )


def q_label(q: float) -> str:
    return f"q{int(round(q * 100)):03d}"


def write_sha_manifest(
    root: Path,
) -> None:
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name != "SHA256SUMS.txt"
    )

    (
        root
        / "SHA256SUMS.txt"
    ).write_text(
        "\n".join(
            f"{sha256_file(p)}  "
            f"{p.relative_to(root).as_posix()}"
            for p in files
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    protocol = json.loads(
        protocol_path.read_text()
    )

    root = Path(
        protocol["output"]["root"]
    )

    selection_dir = Path(
        args.output_dir
    )

    if selection_dir.exists():
        raise RuntimeError(
            f"Selection output already exists: "
            f"{selection_dir}"
        )

    tasks = protocol[
        "task_layout"
    ][
        "tasks"
    ]

    if len(tasks) != 15:
        raise RuntimeError(
            "Protocol must contain 15 tasks."
        )

    q_values = [
        float(q)
        for q in protocol[
            "task_layout"
        ][
            "q_order"
        ]
    ]

    run_metrics = []
    sweep_rows = []

    for task in tasks:
        task_id = int(
            task["task_id"]
        )
        model_seed = int(
            task["model_seed"]
        )
        replicate = int(
            task["replicate"]
        )

        task_dir = find_task_dir(
            root,
            task_id=task_id,
            model_seed=model_seed,
            replicate=replicate,
        )

        if not task_dir.is_dir():
            raise RuntimeError(
                f"Missing task output: "
                f"{task_dir}"
            )

        verify_manifest(
            task_dir
        )

        manifest = json.loads(
            (
                task_dir
                / "task_manifest.json"
            ).read_text()
        )

        if (
            int(manifest["task_id"])
            != task_id
        ):
            raise RuntimeError(
                "Task-manifest ID mismatch."
            )

        if manifest[
            "per_panel_threshold_selection_used"
        ] is not False:
            raise RuntimeError(
                "Per-panel threshold selection "
                "was used unexpectedly."
            )

        for q in q_values:
            qdir = (
                task_dir
                / q_label(q)
            )

            summary = json.loads(
                (
                    qdir
                    / "summary.json"
                ).read_text()
            )

            sweep = pd.read_csv(
                qdir
                / "threshold_sweep.csv"
            )

            if len(sweep) != 201:
                raise RuntimeError(
                    "Expected 201 threshold rows."
                )

            run_id = (
                f"q={q:.2f}"
                f"|model={model_seed}"
                f"|rep={replicate}"
            )

            direct_threshold = float(
                protocol[
                    "direct_transfer_thresholds"
                ][
                    f"{q:.2f}"
                ]
            )

            if not np.isclose(
                float(summary["threshold"]),
                direct_threshold,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Direct-transfer threshold mismatch."
                )

            target_sd = float(
                summary["target_sd"]
            )

            masked_mse = float(
                summary["masked_mse"]
            )

            recovery_index = (
                1.0
                - masked_mse
                / (target_sd ** 2)
            )

            run_metrics.append(
                {
                    "run_id": run_id,
                    "q": q,
                    "model_seed": (
                        model_seed
                    ),
                    "replicate": (
                        replicate
                    ),
                    "masked_mse": (
                        masked_mse
                    ),
                    "masked_mae": float(
                        summary[
                            "masked_mae"
                        ]
                    ),
                    "recovery_index": float(
                        recovery_index
                    ),
                    "gene_spearman": float(
                        summary[
                            "gene_spearman"
                        ]
                    ),
                    "sample_spearman": float(
                        summary[
                            "sample_spearman"
                        ]
                    ),
                    "sd_ratio": float(
                        summary["sd_ratio"]
                    ),
                    "direct_threshold": (
                        direct_threshold
                    ),
                    "direct_recall": float(
                        summary[
                            "threshold_recall"
                        ]
                    ),
                    "direct_precision": float(
                        summary[
                            "threshold_precision"
                        ]
                    ),
                    "direct_true_zero_fill": float(
                        summary[
                            "threshold_true_zero_fill"
                        ]
                    ),
                }
            )

            local = sweep[
                [
                    "threshold",
                    "recall",
                    "precision",
                    "true_zero_fill",
                ]
            ].copy()

            local["run_id"] = run_id
            local["q"] = q
            local["model_seed"] = (
                model_seed
            )
            local["replicate"] = (
                replicate
            )

            sweep_rows.append(
                local
            )

    run_metrics_df = pd.DataFrame(
        run_metrics
    )

    if len(run_metrics_df) != 45:
        raise RuntimeError(
            f"Expected 45 evaluations, "
            f"found {len(run_metrics_df)}."
        )

    all_sweeps = pd.concat(
        sweep_rows,
        ignore_index=True,
    )

    selections = {}
    aggregate_frames = []
    direct_rows = []

    max_fill = float(
        protocol[
            "common_threshold_selection"
        ][
            "max_true_zero_fill"
        ]
    )

    for q in q_values:
        q_runs = all_sweeps[
            np.isclose(
                all_sweeps["q"],
                q,
                atol=1.0e-12,
                rtol=0.0,
            )
        ].copy()

        if (
            q_runs["run_id"].nunique()
            != 15
        ):
            raise RuntimeError(
                f"Expected 15 runs for q={q}."
            )

        aggregate = aggregate_threshold_grid(
            q_runs,
            max_true_zero_fill=max_fill,
        )

        if len(aggregate) != 201:
            raise RuntimeError(
                "Aggregate threshold grid "
                "must contain 201 rows."
            )

        selected = select_common_threshold(
            aggregate
        )

        aggregate["q"] = q

        aggregate_frames.append(
            aggregate
        )

        selections[
            f"{q:.2f}"
        ] = {
            "threshold": float(
                selected["threshold"]
            ),
            "mean_recall": float(
                selected["mean_recall"]
            ),
            "mean_precision": float(
                selected["mean_precision"]
            ),
            "mean_true_zero_fill": float(
                selected[
                    "mean_true_zero_fill"
                ]
            ),
            "max_true_zero_fill": float(
                selected[
                    "max_true_zero_fill"
                ]
            ),
            "n_runs": int(
                selected["n_runs"]
            ),
        }

        q_metric = run_metrics_df[
            np.isclose(
                run_metrics_df["q"],
                q,
                atol=1.0e-12,
                rtol=0.0,
            )
        ]

        direct_rows.append(
            {
                "q": q,
                "threshold": float(
                    q_metric[
                        "direct_threshold"
                    ].iloc[0]
                ),
                "n_runs": int(
                    len(q_metric)
                ),
                "mean_recall": float(
                    q_metric[
                        "direct_recall"
                    ].mean()
                ),
                "mean_precision": float(
                    q_metric[
                        "direct_precision"
                    ].mean()
                ),
                "mean_true_zero_fill": float(
                    q_metric[
                        "direct_true_zero_fill"
                    ].mean()
                ),
                "max_true_zero_fill": float(
                    q_metric[
                        "direct_true_zero_fill"
                    ].max()
                ),
            }
        )

    aggregate_df = pd.concat(
        aggregate_frames,
        ignore_index=True,
    )

    direct_df = pd.DataFrame(
        direct_rows
    )

    selection_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    run_metrics_df.to_csv(
        selection_dir
        / "run_level_metrics.csv",
        index=False,
    )

    aggregate_df.to_csv(
        selection_dir
        / "aggregate_threshold_sweep.csv",
        index=False,
    )

    direct_df.to_csv(
        selection_dir
        / "direct_transfer_summary.csv",
        index=False,
    )

    payload = {
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
        "validation_evaluations": 45,
        "runs_per_q": 15,
        "selection_unit": (
            "one common threshold per q"
        ),
        "eligibility": (
            "true_zero_fill <= 0.02 "
            "in every validation run"
        ),
        "objective": (
            "maximize mean recall, "
            "then mean precision, "
            "then stricter threshold"
        ),
        "selected_thresholds": (
            selections
        ),
        "internal_test_used": False,
        "baron_used": False,
        "zheng68k_used": False,
    }

    (
        selection_dir
        / "selected_thresholds.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_sha_manifest(
        selection_dir
    )

    print(
        json.dumps(
            selections,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "COMMON_THRESHOLD_SELECTION=PASS"
    )


if __name__ == "__main__":
    main()
