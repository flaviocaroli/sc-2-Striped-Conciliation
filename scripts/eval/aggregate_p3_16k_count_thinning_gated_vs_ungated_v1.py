#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
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


def verify_manifest(
    root: Path,
) -> None:

    manifest = (
        root
        / "SHA256SUMS.txt"
    )

    if not manifest.is_file():
        raise RuntimeError(
            f"Missing manifest: {manifest}"
        )

    for line in (
        manifest.read_text()
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

        if sha(path) != digest:
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
            f"{sha(p)}  {p.name}\n"
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
        "task_layout"
    ][
        "tasks"
    ]

    if len(tasks) != 25:
        raise RuntimeError(
            "Expected 25 tasks"
        )

    for task in tasks:

        task_id = int(
            task["task_id"]
        )

        model_seed = int(
            task[
                "model_seed"
            ]
        )

        replicate = int(
            task[
                "replicate"
            ]
        )

        task_dir = (
            root
            / (
                f"task_{task_id:02d}_"
                f"model{model_seed}_"
                f"rep{replicate:02d}"
            )
        )

        if not task_dir.is_dir():
            raise RuntimeError(
                f"Missing task: {task_dir}"
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

        if len(
            manifest[
                "conditions"
            ]
        ) != 9:
            raise RuntimeError(
                "Task condition count != 9"
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

            dataset = str(
                x["dataset"]
            )

            q = float(
                x["q"]
            )

            for mode in (
                "ungated",
                "gated",
            ):

                record = {
                    "dataset":
                        dataset,

                    "q":
                        q,

                    "model_seed":
                        model_seed,

                    "replicate":
                        replicate,

                    "thinning_seed":
                        int(
                            x[
                                "thinning_seed"
                            ]
                        ),

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
                        k:
                            float(v)
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

                    "q":
                        q,

                    "model_seed":
                        model_seed,

                    "replicate":
                        replicate,

                    "thinning_seed":
                        int(
                            x[
                                "thinning_seed"
                            ]
                        ),

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

    if len(df) != 450:
        raise RuntimeError(
            f"Expected 450 mode rows, "
            f"got {len(df)}"
        )

    if len(gate_df) != 225:
        raise RuntimeError(
            f"Expected 225 gate rows, "
            f"got {len(gate_df)}"
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

    for dataset in (
        "internal_test",
        "baron_pancreas",
        "zheng68k",
    ):

        for q in (
            0.85,
            0.70,
            0.50,
        ):

            local = df[
                (
                    df[
                        "dataset"
                    ]
                    == dataset
                )
                & np.isclose(
                    df["q"],
                    q,
                    atol=1e-12,
                    rtol=0.0,
                )
            ].copy()

            if len(local) != 50:
                raise RuntimeError(
                    f"{dataset} q={q}: "
                    f"expected 50 mode rows"
                )

            for mode in (
                "ungated",
                "gated",
            ):

                m = local[
                    local["mode"]
                    == mode
                ]

                if len(m) != 25:
                    raise RuntimeError(
                        "Expected 25 runs/mode"
                    )

                for metric in metrics:

                    values = (
                        m[metric]
                        .to_numpy(
                            dtype=np.float64
                        )
                    )

                    seed_means = (
                        m.groupby(
                            "model_seed"
                        )[metric]
                        .mean()
                        .to_numpy(
                            dtype=np.float64
                        )
                    )

                    rep_means = (
                        m.groupby(
                            "replicate"
                        )[metric]
                        .mean()
                        .to_numpy(
                            dtype=np.float64
                        )
                    )

                    summary_rows.append(
                        {
                            "dataset":
                                dataset,

                            "q":
                                q,

                            "mode":
                                mode,

                            "metric":
                                metric,

                            "n_runs":
                                25,

                            "n_model_seeds":
                                5,

                            "n_technical_replicates":
                                5,

                            "mean":
                                float(
                                    np.mean(
                                        values
                                    )
                                ),

                            "run_sd_descriptive":
                                float(
                                    np.std(
                                        values,
                                        ddof=1,
                                    )
                                ),

                            "model_seed_mean_sd":
                                float(
                                    np.std(
                                        seed_means,
                                        ddof=1,
                                    )
                                ),

                            "technical_replicate_mean_sd":
                                float(
                                    np.std(
                                        rep_means,
                                        ddof=1,
                                    )
                                ),
                        }
                    )

            u = (
                local[
                    local["mode"]
                    == "ungated"
                ]
                .sort_values(
                    [
                        "model_seed",
                        "replicate",
                    ]
                )
            )

            g = (
                local[
                    local["mode"]
                    == "gated"
                ]
                .sort_values(
                    [
                        "model_seed",
                        "replicate",
                    ]
                )
            )

            if not np.array_equal(
                u[
                    [
                        "model_seed",
                        "replicate",
                    ]
                ].to_numpy(),
                g[
                    [
                        "model_seed",
                        "replicate",
                    ]
                ].to_numpy(),
            ):
                raise RuntimeError(
                    "Pairing mismatch"
                )

            for metric, direction in (
                metrics.items()
            ):

                uv = (
                    u[metric]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                gv = (
                    g[metric]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                delta = (
                    gv - uv
                )

                u_mean = float(
                    np.mean(
                        uv
                    )
                )

                g_mean = float(
                    np.mean(
                        gv
                    )
                )

                delta_frame = pd.DataFrame(
                    {
                        "model_seed":
                            u[
                                "model_seed"
                            ].to_numpy(),

                        "replicate":
                            u[
                                "replicate"
                            ].to_numpy(),

                        "delta":
                            delta,
                    }
                )

                seed_delta_means = (
                    delta_frame
                    .groupby(
                        "model_seed"
                    )[
                        "delta"
                    ]
                    .mean()
                    .to_numpy()
                )

                rep_delta_means = (
                    delta_frame
                    .groupby(
                        "replicate"
                    )[
                        "delta"
                    ]
                    .mean()
                    .to_numpy()
                )

                if np.isclose(
                    u_mean,
                    g_mean,
                    atol=1e-12,
                    rtol=1e-12,
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

                        "q":
                            q,

                        "metric":
                            metric,

                        "direction":
                            direction,

                        "ungated_mean":
                            u_mean,

                        "gated_mean":
                            g_mean,

                        "gated_minus_ungated_mean":
                            float(
                                np.mean(
                                    delta
                                )
                            ),

                        "paired_run_delta_sd_descriptive":
                            float(
                                np.std(
                                    delta,
                                    ddof=1,
                                )
                            ),

                        "model_seed_mean_delta_sd":
                            float(
                                np.std(
                                    seed_delta_means,
                                    ddof=1,
                                )
                            ),

                        "technical_replicate_mean_delta_sd":
                            float(
                                np.std(
                                    rep_delta_means,
                                    ddof=1,
                                )
                            ),

                        "numerically_better_mode":
                            better,

                        "biological_inference":
                            False,
                    }
                )

    summary_df = pd.DataFrame(
        summary_rows
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    if len(
        comparison_df
    ) != 144:
        raise RuntimeError(
            "Expected 144 comparison rows"
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
            "n_dataset_q_conditions"
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

    receipt = {
        "schema":
            "sc2-p3-16k-count-thinning-gated-vs-ungated-aggregate-v1",

        "status":
            "PASS",

        "datasets": [
            "internal_test",
            "baron_pancreas",
            "zheng68k",
        ],

        "q_values": [
            0.85,
            0.70,
            0.50,
        ],

        "model_seeds":
            5,

        "technical_replicates":
            5,

        "forward_evaluations":
            225,

        "paired_mode_rows":
            450,

        "gate_metric_rows":
            225,

        "thresholds":
            protocol[
                "frozen_thresholds"
            ],

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

        "uncertainty_interpretation":
            (
                "SC2 model seeds quantify optimization/training "
                "variability. Thinning replicates quantify "
                "technical corruption variability. Neither "
                "dimension is a biological replicate."
            ),

        "gate_metrics_mode_invariant":
            True,

        "threshold_retuning":
            False,

        "protocol":
            str(
                protocol_path
            ),

        "protocol_sha256":
            sha(
                protocol_path
            ),
    }

    (
        out
        / "aggregate_receipt.json"
    ).write_text(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    write_manifest(
        out
    )

    print()
    print(
        "============================================================"
    )

    print(
        "COUNT-THINNING GATED VS UNGATED"
    )

    print(
        "============================================================"
    )

    print(
        comparison_df[
            [
                "dataset",
                "q",
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
        "========== WINNER COUNTS =========="
    )

    print(
        winner_counts.to_string(
            index=False
        )
    )

    print()
    print(
        "P3_THINNING_GATED_VS_UNGATED_AGGREGATION=PASS"
    )


if __name__ == "__main__":
    main()
