#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


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
            f"Missing manifest: "
            f"{manifest}"
        )

    for line in (
        manifest
        .read_text()
        .splitlines()
    ):

        digest, rel = line.split(
            None,
            1,
        )

        path = (
            root
            / rel.strip()
        )

        if (
            sha256_file(
                path
            )
            != digest
        ):
            raise RuntimeError(
                f"SHA mismatch: {path}"
            )


def write_manifest(
    root: Path,
) -> None:

    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file()
        and p.name
        != "SHA256SUMS.txt"
    )

    (
        root
        / "SHA256SUMS.txt"
    ).write_text(
        "".join(
            f"{sha256_file(p)}  "
            f"{p.name}\n"
            for p in files
        ),
        encoding="utf-8",
    )


def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--protocol",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    protocol = json.loads(
        protocol_path.read_text()
    )

    root = Path(
        protocol[
            "output_root"
        ]
    )

    out = Path(
        args.output_dir
    )

    if out.exists():
        raise RuntimeError(
            f"Output exists: {out}"
        )

    rows = []
    gate_rows = []

    tasks = protocol[
        "tasks"
    ]

    if len(tasks) != 5:
        raise RuntimeError(
            "Expected five tasks"
        )

    for task in tasks:

        task_id = int(
            task["task_id"]
        )

        seed = int(
            task["seed"]
        )

        task_dir = (
            root
            / (
                f"task_{task_id:02d}"
                f"_s{seed}"
            )
        )

        if not task_dir.is_dir():
            raise RuntimeError(
                f"Missing task: "
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

        if manifest[
            "status"
        ] != "PASS":
            raise RuntimeError(
                "Task not PASS"
            )

        if (
            manifest[
                "historical_ungated_reproduction"
            ]
            != "PASS"
        ):
            raise RuntimeError(
                "Historical reproduction "
                "did not pass"
            )

        for condition in (
            manifest[
                "conditions"
            ]
        ):

            metrics_path = (
                task_dir
                / condition[
                    "metrics"
                ]
            )

            x = json.loads(
                metrics_path.read_text()
            )

            dataset = x[
                "dataset"
            ]

            mask_rate = int(
                x[
                    "mask_rate"
                ]
            )

            for mode in (
                "ungated",
                "gated",
            ):

                record = {
                    "dataset":
                        dataset,

                    "mask_rate":
                        mask_rate,

                    "seed":
                        seed,

                    "mode":
                        mode,

                    "threshold":
                        float(
                            x[
                                "threshold"
                            ]
                        ),
                }

                record.update(
                    {
                        k: float(v)
                        if isinstance(
                            v,
                            (
                                int,
                                float,
                            ),
                        )
                        else v
                        for k, v in x[
                            mode
                        ].items()
                    }
                )

                rows.append(
                    record
                )

            gate_rows.append(
                {
                    "dataset":
                        dataset,

                    "mask_rate":
                        mask_rate,

                    "seed":
                        seed,

                    **{
                        f"gate_{k}":
                            float(v)
                        for k, v
                        in x[
                            "gate_metrics"
                        ].items()
                    },
                }
            )

    df = pd.DataFrame(
        rows
    )

    gate_df = pd.DataFrame(
        gate_rows
    )

    if len(df) != 90:
        raise RuntimeError(
            f"Expected 90 mode rows, "
            f"got {len(df)}"
        )

    if len(gate_df) != 45:
        raise RuntimeError(
            "Expected 45 gate rows"
        )

    metrics = {
        "masked_mse":
            "lower",

        "masked_mae":
            "lower",

        "recovery_index":
            "higher",

        "gene_spearman":
            "higher",

        "sample_spearman":
            "higher",

        "sd_ratio_abs_error":
            "lower",

        "policy_true_zero_fill":
            "lower",

        "selection_recall":
            "higher",

        "selection_precision":
            "higher",

        "numeric_true_zero_positive_fraction":
            "lower",

        "true_zero_mse":
            "lower",

        "true_zero_mae":
            "lower",

        "observed_zero_mse":
            "lower",

        "observed_zero_mae":
            "lower",

        "full_available_mse":
            "lower",

        "full_available_mae":
            "lower",
    }

    summary_rows = []
    comparison_rows = []

    conditions = (
        df[
            [
                "dataset",
                "mask_rate",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "dataset",
                "mask_rate",
            ]
        )
    )

    for _, condition in (
        conditions.iterrows()
    ):

        dataset = condition[
            "dataset"
        ]

        mask_rate = int(
            condition[
                "mask_rate"
            ]
        )

        local = df[
            (
                df["dataset"]
                == dataset
            )
            & (
                df["mask_rate"]
                == mask_rate
            )
        ]

        if len(local) != 10:
            raise RuntimeError(
                "Each condition must "
                "contain 5 seeds × 2 modes"
            )

        for mode in (
            "ungated",
            "gated",
        ):

            mode_df = local[
                local["mode"]
                == mode
            ]

            if len(mode_df) != 5:
                raise RuntimeError(
                    "Mode does not have "
                    "five seeds"
                )

            for metric in metrics:

                values = (
                    mode_df[
                        metric
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                summary_rows.append(
                    {
                        "dataset":
                            dataset,

                        "mask_rate":
                            mask_rate,

                        "mode":
                            mode,

                        "metric":
                            metric,

                        "n_seeds":
                            5,

                        "mean":
                            float(
                                np.mean(
                                    values
                                )
                            ),

                        "sd":
                            float(
                                np.std(
                                    values,
                                    ddof=1,
                                )
                            ),
                    }
                )

        pivot = {
            mode:
                local[
                    local["mode"]
                    == mode
                ]
                .sort_values(
                    "seed"
                )
            for mode in (
                "ungated",
                "gated",
            )
        }

        if not np.array_equal(
            pivot[
                "ungated"
            ][
                "seed"
            ].to_numpy(),
            pivot[
                "gated"
            ][
                "seed"
            ].to_numpy(),
        ):
            raise RuntimeError(
                "Seed pairing mismatch"
            )

        for metric, direction in (
            metrics.items()
        ):

            u = (
                pivot[
                    "ungated"
                ][
                    metric
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            g = (
                pivot[
                    "gated"
                ][
                    metric
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            diff = g - u

            mean_diff = float(
                np.mean(
                    diff
                )
            )

            sd_diff = float(
                np.std(
                    diff,
                    ddof=1,
                )
            )

            se = (
                sd_diff
                / math.sqrt(5)
            )

            critical = float(
                student_t.ppf(
                    0.975,
                    df=4,
                )
            )

            ci_low = (
                mean_diff
                - critical * se
            )

            ci_high = (
                mean_diff
                + critical * se
            )

            u_mean = float(
                np.mean(u)
            )

            g_mean = float(
                np.mean(g)
            )

            if np.isclose(
                u_mean,
                g_mean,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                better = "tie"

            elif direction == "lower":

                better = (
                    "gated"
                    if g_mean < u_mean
                    else "ungated"
                )

            else:

                better = (
                    "gated"
                    if g_mean > u_mean
                    else "ungated"
                )

            comparison_rows.append(
                {
                    "dataset":
                        dataset,

                    "mask_rate":
                        mask_rate,

                    "metric":
                        metric,

                    "direction":
                        direction,

                    "ungated_mean":
                        u_mean,

                    "gated_mean":
                        g_mean,

                    "gated_minus_ungated_mean":
                        mean_diff,

                    "paired_seed_difference_sd":
                        sd_diff,

                    "paired_seed_ci95_low":
                        float(ci_low),

                    "paired_seed_ci95_high":
                        float(ci_high),

                    "numerically_better_mode":
                        better,

                    "uncertainty_unit":
                        "model_seed",
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    winner_counts = (
        comparison_df
        .groupby(
            [
                "metric",
                "numerically_better_mode",
            ]
        )
        .size()
        .rename(
            "n_conditions"
        )
        .reset_index()
    )

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    df.to_csv(
        out
        / "run_level_mode_metrics.csv",
        index=False,
    )

    gate_df.to_csv(
        out
        / "run_level_gate_metrics.csv",
        index=False,
    )

    summary_df.to_csv(
        out
        / "condition_mode_summary.csv",
        index=False,
    )

    comparison_df.to_csv(
        out
        / "paired_metric_comparison.csv",
        index=False,
    )

    winner_counts.to_csv(
        out
        / "winner_counts.csv",
        index=False,
    )

    payload = {
        "schema":
            "sc2-p3-16k-normalized-gated-vs-ungated-aggregate-v1",

        "status":
            "PASS",

        "analysis_status":
            protocol[
                "analysis_status"
            ],

        "datasets": [
            "internal_test",
            "baron_pancreas",
            "zheng68k",
        ],

        "mask_rates": [
            15,
            30,
            50,
        ],

        "model_seeds":
            5,

        "paired_evaluations":
            45,

        "mode_rows":
            90,

        "ungated_definition":
            protocol[
                "mode_definitions"
            ][
                "ungated"
            ],

        "gated_definition":
            protocol[
                "mode_definitions"
            ][
                "gated"
            ],

        "historical_ungated_reproduction":
            "PASS",

        "historical_gated_selection_metric_reproduction":
            "PASS",

        "observed_nonzero_preservation":
            "EXACT",

        "baron_repaired_matrices_saved_for_lodo":
            True,

        "uncertainty_interpretation":
            (
                "Five SC2 model seeds represent "
                "optimization/training variability, "
                "not biological replication."
            ),

        "protocol":
            str(
                protocol_path
            ),

        "protocol_sha256":
            sha256_file(
                protocol_path
            ),
    }

    (
        out
        / "aggregate_receipt.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_manifest(
        out
    )

    print()
    print(
        "============================================================"
    )
    print(
        "GATED VS UNGATED — NORMALIZED MASKING"
    )
    print(
        "============================================================"
    )

    print(
        comparison_df[
            [
                "dataset",
                "mask_rate",
                "metric",
                "ungated_mean",
                "gated_mean",
                "numerically_better_mode",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "P3_NORMALIZED_GATED_VS_UNGATED_AGGREGATION=PASS"
    )


if __name__ == "__main__":
    main()
