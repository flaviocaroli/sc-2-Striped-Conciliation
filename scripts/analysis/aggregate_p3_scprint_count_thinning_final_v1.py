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
            lambda: f.read(
                1024 * 1024
            ),
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
        "--amendment",
        required=True,
    )

    parser.add_argument(
        "--scprint1-root",
        required=True,
    )

    parser.add_argument(
        "--scprint2-root",
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

    amendment_path = Path(
        args.amendment
    ).resolve()

    root1 = Path(
        args.scprint1_root
    ).resolve()

    root2 = Path(
        args.scprint2_root
    ).resolve()

    output = Path(
        args.output
    ).resolve()

    if output.exists():

        raise RuntimeError(
            f"output already exists: "
            f"{output}"
        )

    protocol = json.loads(
        protocol_path.read_text()
    )

    amendment = json.loads(
        amendment_path.read_text()
    )

    assert (
        protocol[
            "protocol_id"
        ]
        ==
        "sc2-p3-scprint-count-thinning-v2"
    )

    assert (
        amendment[
            "schema"
        ]
        ==
        "sc2-p3-scprint2-count-thinning-completion-amendment-v1"
    )

    output.mkdir(
        parents=True,
    )

    rows = []
    manifest = []

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

        if model == "scprint1":

            root = root1

        elif model == "scprint2":

            root = root2

        else:

            raise RuntimeError(
                f"unexpected model: "
                f"{model}"
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
                f"missing summary: "
                f"{summary_path}"
            )

        summary = json.loads(
            summary_path.read_text()
        )

        assert summary[
            "status"
        ] == "PASS"

        assert int(
            summary[
                "task_id"
            ]
        ) == task_id

        assert summary[
            "model"
        ] == model

        assert summary[
            "dataset"
        ] == dataset

        assert abs(
            float(
                summary[
                    "q"
                ]
            )
            - q
        ) < 1e-12

        conditions = summary[
            "conditions"
        ]

        assert len(
            conditions
        ) == 5

        manifest.append(
            {
                "task_id":
                    task_id,

                "model":
                    model,

                "dataset":
                    dataset,

                "q":
                    q,

                "task_summary":
                    str(
                        summary_path
                    ),

                "task_summary_sha256":
                    sha256_file(
                        summary_path
                    ),

                "replicate_count":
                    5,
            }
        )

        for item in conditions:

            row = {
                "task_id":
                    task_id,

                "model":
                    model,

                "dataset":
                    dataset,

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

                "potential_evaluated_genes":
                    int(
                        item.get(
                            "potential_evaluated_genes",
                            item[
                                "evaluated_genes"
                            ],
                        )
                    ),

                "evaluated_genes":
                    int(
                        item[
                            "evaluated_genes"
                        ]
                    ),

                "input_cells":
                    int(
                        item[
                            "input_cells"
                        ]
                    ),

                "evaluated_cells":
                    int(
                        item[
                            "evaluated_cells"
                        ]
                    ),

                "dropped_cells":
                    int(
                        item[
                            "dropped_cells"
                        ]
                    ),

                "evaluated_cell_fraction":
                    float(
                        item[
                            "evaluated_cell_fraction"
                        ]
                    ),

                "native_returned_genes":
                    int(
                        item[
                            "native_returned_genes"
                        ]
                    ),

                "evaluated_gene_order_sha256":
                    item[
                        "evaluated_gene_order_sha256"
                    ],

                "evaluated_source_row_order_sha256":
                    item[
                        "evaluated_source_row_order_sha256"
                    ],

                "task_summary_sha256":
                    manifest[
                        -1
                    ][
                        "task_summary_sha256"
                    ],
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

            for key, value in item[
                "gains"
            ].items():

                row[key] = value

            rows.append(
                row
            )

    if len(rows) != 90:

        raise RuntimeError(
            f"expected 90 rows, "
            f"found {len(rows)}"
        )

    frame = pd.DataFrame(
        rows
    )

    if set(
        frame[
            "model"
        ]
    ) != {
        "scprint1",
        "scprint2",
    }:

        raise RuntimeError(
            "model set mismatch"
        )

    model_counts = (
        frame[
            "model"
        ]
        .value_counts()
        .to_dict()
    )

    if model_counts != {
        "scprint1": 45,
        "scprint2": 45,
    }:

        raise RuntimeError(
            f"model row mismatch: "
            f"{model_counts}"
        )

    expected_datasets = {
        "internal_test",
        "baron_pancreas",
        "zheng68k",
    }

    if set(
        frame[
            "dataset"
        ]
    ) != expected_datasets:

        raise RuntimeError(
            "dataset set mismatch"
        )

    if set(
        np.round(
            frame[
                "q"
            ].to_numpy(),
            2,
        )
    ) != {
        0.50,
        0.70,
        0.85,
    }:

        raise RuntimeError(
            "q set mismatch"
        )

    group_sizes = (
        frame.groupby(
            [
                "model",
                "dataset",
                "q",
            ]
        )
        .size()
    )

    if (
        len(group_sizes) != 18
        or not np.all(
            group_sizes.to_numpy()
            == 5
        )
    ):

        raise RuntimeError(
            "18 x 5 replicate grid mismatch"
        )

    for (
        model,
        dataset,
        q,
    ), group in frame.groupby(
        [
            "model",
            "dataset",
            "q",
        ]
    ):

        reps = sorted(
            group[
                "replicate"
            ].astype(int)
        )

        if reps != [
            1,
            2,
            3,
            4,
            5,
        ]:

            raise RuntimeError(
                f"replicate mismatch: "
                f"{model} {dataset} {q}"
            )

    if not np.all(
        frame[
            "input_cells"
        ].to_numpy()
        == 5000
    ):

        raise RuntimeError(
            "input cell count mismatch"
        )

    if not np.all(
        frame[
            "dropped_cells"
        ].to_numpy()
        ==
        5000
        - frame[
            "evaluated_cells"
        ].to_numpy()
    ):

        raise RuntimeError(
            "cell accounting mismatch"
        )

    if not np.all(
        (
            frame[
                "evaluated_cells"
            ].to_numpy()
            >= 100
        )
        &
        (
            frame[
                "evaluated_cells"
            ].to_numpy()
            <= 5000
        )
    ):

        raise RuntimeError(
            "invalid evaluated-cell coverage"
        )

    if not np.all(
        (
            frame[
                "evaluated_genes"
            ].to_numpy()
            >= 10000
        )
        &
        (
            frame[
                "evaluated_genes"
            ].to_numpy()
            <= 16384
        )
    ):

        raise RuntimeError(
            "invalid evaluated-gene coverage"
        )

    if not np.all(
        frame[
            "native_returned_genes"
        ].to_numpy()
        >=
        frame[
            "evaluated_genes"
        ].to_numpy()
    ):

        raise RuntimeError(
            "native gene accounting mismatch"
        )

    #
    # All numeric metrics except Spearman means must be finite.
    #
    numeric_columns = list(
        frame.select_dtypes(
            include=[
                np.number,
            ]
        ).columns
    )

    nullable_spearman = {
        "model_sample_spearman",
        "model_gene_spearman",
        "baseline_sample_spearman",
        "baseline_gene_spearman",
    }

    for column in numeric_columns:

        if column in nullable_spearman:
            continue

        values = frame[
            column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            values
        ).all():

            raise RuntimeError(
                f"nonfinite numeric values "
                f"in {column}"
            )

    group_keys = [
        "model",
        "dataset",
        "q",
        "loss_fraction",
    ]

    aggregate_exclude = set(
        group_keys
        + [
            "task_id",
            "replicate",
            "thinning_seed",
        ]
    )

    aggregate_columns = [
        column
        for column
        in numeric_columns
        if column
        not in aggregate_exclude
    ]

    grouped = frame.groupby(
        group_keys,
        sort=True,
    )

    means = (
        grouped[
            aggregate_columns
        ]
        .mean()
        .add_suffix(
            "_mean"
        )
    )

    sds = (
        grouped[
            aggregate_columns
        ]
        .std(
            ddof=1
        )
        .add_suffix(
            "_sd"
        )
    )

    summary = (
        means.join(
            sds
        )
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

    publication_path = (
        output
        / "publication_table.csv"
    )

    manifest_path = (
        output
        / "task_manifest.json"
    )

    frame.to_csv(
        per_rep_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    publication_columns = [
        "model",
        "dataset",
        "q",
        "loss_fraction",

        "evaluated_genes_mean",
        "evaluated_genes_sd",

        "evaluated_cells_mean",
        "evaluated_cells_sd",

        "dropped_cells_mean",

        "model_masked_mse_mean",
        "model_masked_mse_sd",

        "model_masked_mae_mean",
        "model_masked_mae_sd",

        "model_recovery_index_mean",
        "model_recovery_index_sd",

        "model_gene_spearman_mean",
        "model_gene_spearman_sd",

        "model_sample_spearman_mean",
        "model_sample_spearman_sd",

        "baseline_masked_mse_mean",
        "baseline_recovery_index_mean",

        "masked_mse_gain_vs_corrupted_mean",
        "masked_mse_gain_vs_corrupted_sd",

        "recovery_gain_vs_corrupted_mean",
        "recovery_gain_vs_corrupted_sd",

        "model_full_available_mse_mean",
        "model_full_available_mse_sd",

        "baseline_full_available_mse_mean",
    ]

    publication_columns = [
        column
        for column
        in publication_columns
        if column
        in summary.columns
    ]

    summary[
        publication_columns
    ].to_csv(
        publication_path,
        index=False,
    )

    manifest_path.write_text(
        json.dumps(
            {
                "schema":
                    "sc2-p3-scprint-final-task-manifest-v1",

                "status":
                    "COMPLETE",

                "task_count":
                    18,

                "replicate_result_count":
                    90,

                "tasks":
                    manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    receipt = {
        "schema":
            "sc2-p3-scprint-count-thinning-final-aggregate-v1",

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

        "scprint2_amendment":
            str(
                amendment_path
            ),

        "scprint2_amendment_sha256":
            sha256_file(
                amendment_path
            ),

        "scprint1_root":
            str(
                root1
            ),

        "scprint2_root":
            str(
                root2
            ),

        "task_count":
            18,

        "replicate_result_count":
            90,

        "technical_replicates_per_condition":
            5,

        "technical_replicates_are_biological_replicates":
            False,

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

        "per_replicate_metrics": {
            "path":
                str(
                    per_rep_path
                ),

            "sha256":
                sha256_file(
                    per_rep_path
                ),
        },

        "summary_mean_sd": {
            "path":
                str(
                    summary_path
                ),

            "sha256":
                sha256_file(
                    summary_path
                ),
        },

        "publication_table": {
            "path":
                str(
                    publication_path
                ),

            "sha256":
                sha256_file(
                    publication_path
                ),
        },

        "task_manifest": {
            "path":
                str(
                    manifest_path
                ),

            "sha256":
                sha256_file(
                    manifest_path
                ),
        },

        "cell_policy":
            (
                "Metrics are paired only on source rows "
                "retained by native preprocessing. "
                "No dropped-cell imputation."
            ),

        "gene_policy":
            (
                "Metrics use native returned P3 genes "
                "intersected with frozen dataset availability. "
                "No missing-prediction imputation."
            ),

        "evaluation_scale":
            (
                "Full counts, corrupted counts and native "
                "scprint_mu predictions independently "
                "CP10K+log1p normalized on the same "
                "per-result evaluated gene set."
            ),
    }

    receipt_path = (
        output
        / "aggregate_receipt.json"
    )

    receipt_path.write_text(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "FINAL_COMBINED_AGGREGATION=PASS"
    )

    print(
        "ROWS=90"
    )

    print(
        "CONDITIONS=18"
    )

    print(
        publication_path
    )

    print()

    print(
        summary[
            publication_columns
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":

    main()
