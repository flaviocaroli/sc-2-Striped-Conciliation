#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

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

    output = Path(
        args.output
    )

    if output.exists():
        raise RuntimeError(
            f"output exists: {output}"
        )

    output.mkdir(
        parents=True
    )

    rows = []

    for task in protocol[
        "task_layout"
    ][
        "tasks"
    ]:

        task_id = int(
            task[
                "task_id"
            ]
        )

        model = task[
            "model"
        ]

        dataset = task[
            "dataset"
        ]

        q = float(
            task[
                "q"
            ]
        )

        summary_path = (
            root
            / (
                f"task_{task_id:02d}_"
                f"{model}_"
                f"{dataset}_"
                f"q{int(round(q * 100)):03d}"
            )
            / "task_summary.json"
        )

        if not summary_path.is_file():
            raise RuntimeError(
                f"missing: {summary_path}"
            )

        summary = json.loads(
            summary_path.read_text()
        )

        if (
            summary.get("status")
            != "PASS"
            or len(
                summary.get(
                    "conditions",
                    [],
                )
            )
            != 5
        ):
            raise RuntimeError(
                f"bad task: {task_id}"
            )

        for item in summary[
            "conditions"
        ]:

            row = {
                "model":
                    item[
                        "model"
                    ],

                "dataset":
                    item[
                        "dataset"
                    ],

                "q":
                    float(
                        item[
                            "q"
                        ]
                    ),

                "loss_fraction":
                    float(
                        item[
                            "loss_fraction"
                        ]
                    ),

                "replicate":
                    int(
                        item[
                            "replicate"
                        ]
                    ),

                "thinning_seed":
                    int(
                        item[
                            "thinning_seed"
                        ]
                    ),

                "model_p3_genes":
                    int(
                        item[
                            "model_p3_genes"
                        ]
                    ),

                "dataset_available_genes":
                    int(
                        item[
                            "dataset_available_genes"
                        ]
                    ),

                "evaluated_genes":
                    int(
                        item[
                            "evaluated_genes"
                        ]
                    ),
            }

            for key, value in item[
                "model_metrics"
            ].items():

                row[
                    "model_"
                    + key
                ] = value

            for key, value in item[
                "corrupted_baseline"
            ].items():

                row[
                    "baseline_"
                    + key
                ] = value

            row.update(
                item[
                    "gains"
                ]
            )

            rows.append(
                row
            )

    if len(rows) != 90:
        raise RuntimeError(
            "expected 90 replicate rows; "
            f"found {len(rows)}"
        )

    frame = pd.DataFrame(
        rows
    )

    sizes = frame.groupby(
        [
            "model",
            "dataset",
            "q",
        ]
    ).size()

    if (
        len(sizes) != 18
        or not np.all(
            sizes.to_numpy()
            == 5
        )
    ):
        raise RuntimeError(
            "condition/replicate grid mismatch"
        )

    group_keys = [
        "model",
        "dataset",
        "q",
        "loss_fraction",
    ]

    exclude = set(
        group_keys
        + [
            "replicate",
            "thinning_seed",
        ]
    )

    metrics = [
        column
        for column
        in frame.columns
        if (
            column not in exclude
            and pd.api.types.is_numeric_dtype(
                frame[
                    column
                ]
            )
        )
    ]

    grouped = frame.groupby(
        group_keys,
        sort=True,
    )

    mean = (
        grouped[
            metrics
        ]
        .mean()
        .add_suffix(
            "_mean"
        )
    )

    sd = (
        grouped[
            metrics
        ]
        .std(
            ddof=1
        )
        .add_suffix(
            "_sd"
        )
    )

    summary = (
        mean.join(sd)
        .reset_index()
    )

    per_rep_path = (
        output
        / "per_replicate_metrics.csv"
    )

    summary_path = (
        output
        / "summary_mean_sd.csv"
    )

    frame.to_csv(
        per_rep_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    receipt = {
        "schema":
            "sc2-p3-scprint-count-thinning-results-receipt-v1",

        "status":
            "FROZEN_COMPLETE",

        "protocol":
            str(
                protocol_path
            ),

        "protocol_sha256":
            sha256_file(
                protocol_path
            ),

        "technical_replicates":
            5,

        "technical_replicates_are_biological_replicates":
            False,

        "n_condition_rows":
            90,

        "n_aggregated_conditions":
            18,

        "models":
            [
                "scprint1",
                "scprint2",
            ],

        "datasets":
            [
                "internal_test",
                "baron_pancreas",
                "zheng68k",
            ],

        "q_values":
            [
                0.85,
                0.70,
                0.50,
            ],

        "per_replicate_csv":
            str(
                per_rep_path
            ),

        "per_replicate_csv_sha256":
            sha256_file(
                per_rep_path
            ),

        "summary_csv":
            str(
                summary_path
            ),

        "summary_csv_sha256":
            sha256_file(
                summary_path
            ),

        "evaluation_note":
            (
                "Coverage-specific model/dataset overlap. "
                "Full, thinned and native scprint_mu count "
                "matrices are independently CP10K+log1p "
                "normalized on the same fixed overlap."
            ),
    }

    (
        output
        / "results_receipt.json"
    ).write_text(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "SCPRINT_COUNT_THINNING_AGGREGATION=PASS"
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
