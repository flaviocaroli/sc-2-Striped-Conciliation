#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def verify_manifest(root):

    root = Path(root)

    manifest = (
        root
        / "SHA256SUMS.txt"
    )

    if not manifest.is_file():
        raise RuntimeError(
            f"missing manifest {manifest}"
        )

    for line in (
        manifest.read_text()
        .splitlines()
    ):

        digest, rel = line.split(
            None,
            1,
        )

        p = (
            root
            / rel.strip()
        )

        if sha(p) != digest:
            raise RuntimeError(
                f"SHA mismatch {p}"
            )


def write_manifest(root):

    root = Path(root)

    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file()
        and p.name != "SHA256SUMS.txt"
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


def summarize_donors(
    frame,
    metric,
):

    x = frame[
        [
            "heldout_donor",
            metric,
        ]
    ].copy()

    assert len(x) == 4

    values = x[
        metric
    ].to_numpy(
        dtype=np.float64
    )

    return {
        "mean":
            float(
                np.mean(values)
            ),

        "sd_across_donors":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),

        "min":
            float(
                np.min(values)
            ),
    }


def main():

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

    p = json.loads(
        protocol_path.read_text()
    )

    root = Path(
        p[
            "task_output_root"
        ]
    )

    out = Path(
        args.output_dir
    )

    if out.exists():
        raise RuntimeError(
            "aggregate output exists"
        )

    fold_frames = []
    class_frames = []
    confusion_frames = []
    representation_frames = []

    for task in p[
        "tasks"
    ]:

        task_id = int(
            task[
                "task_id"
            ]
        )

        d = (
            root
            / f"task_{task_id:02d}"
        )

        if not d.is_dir():
            raise RuntimeError(
                f"missing task {d}"
            )

        verify_manifest(d)

        receipt = json.loads(
            (
                d
                / "task_receipt.json"
            ).read_text()
        )

        if receipt[
            "status"
        ] != "PASS":
            raise RuntimeError(
                "task did not pass"
            )

        fold_frames.append(
            pd.read_csv(
                d
                / "fold_metrics.csv"
            )
        )

        class_frames.append(
            pd.read_csv(
                d
                / "per_class_metrics.csv"
            )
        )

        confusion_frames.append(
            pd.read_csv(
                d
                / "confusion_long.csv"
            )
        )

        representation_frames.append(
            pd.read_csv(
                d
                / "representation_summary.csv"
            )
        )

    folds = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    classes = pd.concat(
        class_frames,
        ignore_index=True,
    )

    confusion = pd.concat(
        confusion_frames,
        ignore_index=True,
    )

    representation = pd.concat(
        representation_frames,
        ignore_index=True,
    )

    if len(folds) != 136:
        raise RuntimeError(
            f"expected 136 fold rows, "
            f"got {len(folds)}"
        )

    if (
        folds[
            "representation_id"
        ].nunique()
        != 34
    ):
        raise RuntimeError(
            "expected 34 expression representations"
        )

    metrics = {
        "macro_f1":
            "higher",

        "balanced_accuracy":
            "higher",

        "log_loss":
            "lower",
    }

    condition_rows = []

    #
    # CLEAN
    #
    clean = folds[
        folds["kind"]
        == "clean"
    ]

    assert len(clean) == 4

    for metric in metrics:

        s = summarize_donors(
            clean,
            metric,
        )

        condition_rows.append({
            "kind":
                "clean",

            "mode":
                None,

            "mask_rate":
                None,

            "metric":
                metric,

            "mean":
                s["mean"],

            "sd_across_donors":
                s[
                    "sd_across_donors"
                ],

            "min":
                s["min"],

            "n_biological_units":
                4,

            "n_model_seeds":
                0,
        })


    #
    # CORRUPTED
    #
    corrupted = folds[
        folds["kind"]
        == "corrupted"
    ]

    assert len(corrupted) == 12

    for mask in (
        15,
        30,
        50,
    ):

        local = corrupted[
            corrupted[
                "mask_rate"
            ]
            == mask
        ]

        assert len(local) == 4

        for metric in metrics:

            s = summarize_donors(
                local,
                metric,
            )

            condition_rows.append({
                "kind":
                    "corrupted",

                "mode":
                    None,

                "mask_rate":
                    mask,

                "metric":
                    metric,

                "mean":
                    s["mean"],

                "sd_across_donors":
                    s[
                        "sd_across_donors"
                    ],

                "min":
                    s["min"],

                "n_biological_units":
                    4,

                "n_model_seeds":
                    0,
            })


    #
    # REPAIRED:
    # average optimization seeds inside each biological donor
    # before donor-level summary.
    #
    repaired = folds[
        folds["kind"]
        == "sc2_repaired"
    ]

    assert len(repaired) == 120

    donor_repaired_rows = []

    for mode in (
        "gated",
        "ungated",
    ):

        for mask in (
            15,
            30,
            50,
        ):

            local = repaired[
                (
                    repaired[
                        "mode"
                    ]
                    == mode
                )
                &
                (
                    repaired[
                        "mask_rate"
                    ]
                    == mask
                )
            ]

            assert len(local) == 20
            assert local[
                "model_seed"
            ].nunique() == 5
            assert local[
                "heldout_donor"
            ].nunique() == 4

            for donor in p[
                "heldout_donors"
            ]:

                d = local[
                    local[
                        "heldout_donor"
                    ]
                    == donor
                ]

                assert len(d) == 5

                row = {
                    "mode":
                        mode,

                    "mask_rate":
                        mask,

                    "heldout_donor":
                        donor,
                }

                for metric in metrics:
                    row[metric] = float(
                        d[
                            metric
                        ].mean()
                    )

                donor_repaired_rows.append(
                    row
                )

            donor_frame = pd.DataFrame(
                [
                    x
                    for x in donor_repaired_rows
                    if (
                        x["mode"]
                        == mode
                        and x["mask_rate"]
                        == mask
                    )
                ]
            )

            assert len(donor_frame) == 4

            for metric in metrics:

                values = donor_frame[
                    metric
                ].to_numpy(
                    dtype=np.float64
                )

                seed_means = (
                    local.groupby(
                        "model_seed"
                    )[metric]
                    .mean()
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                condition_rows.append({
                    "kind":
                        "sc2_repaired",

                    "mode":
                        mode,

                    "mask_rate":
                        mask,

                    "metric":
                        metric,

                    "mean":
                        float(
                            np.mean(
                                values
                            )
                        ),

                    "sd_across_donors":
                        float(
                            np.std(
                                values,
                                ddof=1,
                            )
                        ),

                    "min":
                        float(
                            np.min(
                                values
                            )
                        ),

                    "model_seed_mean_sd":
                        float(
                            np.std(
                                seed_means,
                                ddof=1,
                            )
                        ),

                    "n_biological_units":
                        4,

                    "n_model_seeds":
                        5,
                })


    donor_repaired = pd.DataFrame(
        donor_repaired_rows
    )

    condition_summary = pd.DataFrame(
        condition_rows
    )


    #
    # REPAIR GAIN VS SAME-MASK CORRUPTED BASELINE.
    # Positive improvement always means repaired is better.
    #
    gain_rows = []

    for mode in (
        "gated",
        "ungated",
    ):

        for mask in (
            15,
            30,
            50,
        ):

            rep = repaired[
                (
                    repaired["mode"]
                    == mode
                )
                &
                (
                    repaired[
                        "mask_rate"
                    ]
                    == mask
                )
            ]

            corr = corrupted[
                corrupted[
                    "mask_rate"
                ]
                == mask
            ][
                [
                    "heldout_donor",
                    "macro_f1",
                    "balanced_accuracy",
                    "log_loss",
                ]
            ]

            joined = rep.merge(
                corr,
                on="heldout_donor",
                suffixes=(
                    "_repaired",
                    "_corrupted",
                ),
                validate="many_to_one",
            )

            assert len(joined) == 20

            for metric, direction in (
                metrics.items()
            ):

                if direction == "higher":

                    joined[
                        "improvement"
                    ] = (
                        joined[
                            f"{metric}_repaired"
                        ]
                        - joined[
                            f"{metric}_corrupted"
                        ]
                    )

                else:

                    joined[
                        "improvement"
                    ] = (
                        joined[
                            f"{metric}_corrupted"
                        ]
                        - joined[
                            f"{metric}_repaired"
                        ]
                    )

                #
                # model seed is optimization variability,
                # so average seeds inside each donor.
                #
                donor_gain = (
                    joined.groupby(
                        "heldout_donor"
                    )[
                        "improvement"
                    ]
                    .mean()
                )

                values = donor_gain.to_numpy(
                    dtype=np.float64
                )

                gain_rows.append({
                    "mode":
                        mode,

                    "mask_rate":
                        mask,

                    "metric":
                        metric,

                    "positive_means_repair_improved":
                        True,

                    "mean_improvement_across_donors":
                        float(
                            np.mean(
                                values
                            )
                        ),

                    "sd_improvement_across_donors":
                        float(
                            np.std(
                                values,
                                ddof=1,
                            )
                        ),

                    "n_biological_units":
                        4,

                    "model_seeds_averaged_within_donor":
                        5,
                })


    repair_gain = pd.DataFrame(
        gain_rows
    )


    #
    # DIRECT GATED vs UNGATED LODO COMPARISON.
    # Same model seed, mask, donor => paired.
    #
    g = repaired[
        repaired["mode"]
        == "gated"
    ].copy()

    u = repaired[
        repaired["mode"]
        == "ungated"
    ].copy()

    pair_keys = [
        "mask_rate",
        "model_seed",
        "heldout_donor",
    ]

    paired = g.merge(
        u,
        on=pair_keys,
        suffixes=(
            "_gated",
            "_ungated",
        ),
        validate="one_to_one",
    )

    assert len(paired) == 60

    policy_rows = []

    for mask in (
        15,
        30,
        50,
    ):

        local = paired[
            paired[
                "mask_rate"
            ]
            == mask
        ].copy()

        assert len(local) == 20

        for metric, direction in (
            metrics.items()
        ):

            raw_delta = (
                local[
                    f"{metric}_gated"
                ]
                - local[
                    f"{metric}_ungated"
                ]
            )

            if direction == "higher":
                improvement = raw_delta

            else:
                improvement = -raw_delta

            local[
                "policy_improvement"
            ] = improvement

            donor_delta = (
                local.groupby(
                    "heldout_donor"
                )[
                    "policy_improvement"
                ]
                .mean()
            )

            values = donor_delta.to_numpy(
                dtype=np.float64
            )

            policy_rows.append({
                "mask_rate":
                    mask,

                "metric":
                    metric,

                "positive_means_gated_better":
                    True,

                "mean_gated_improvement_over_ungated":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "sd_across_donors":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ),

                "numerically_better_policy":
                    (
                        "gated"
                        if float(
                            np.mean(
                                values
                            )
                        ) > 0
                        else (
                            "ungated"
                            if float(
                                np.mean(
                                    values
                                )
                            ) < 0
                            else "tie"
                        )
                    ),

                "n_biological_units":
                    4,

                "model_seeds_averaged_within_donor":
                    5,
            })


    policy_comparison = pd.DataFrame(
        policy_rows
    )


    #
    # Reuse completed foundation representation LODO.
    #
    foundation_receipt_path = Path(
        p[
            "representation_experiment"
        ][
            "results_receipt"
        ][
            "path"
        ]
    )

    expected_foundation_sha = (
        p[
            "representation_experiment"
        ][
            "results_receipt"
        ][
            "sha256"
        ]
    )

    if (
        sha(
            foundation_receipt_path
        )
        != expected_foundation_sha
    ):
        raise RuntimeError(
            "foundation LODO receipt SHA mismatch"
        )

    foundation = json.loads(
        foundation_receipt_path.read_text()
    )

    assert foundation["status"] == "PASS"

    foundation_rows = []

    for model in (
        "Geneformer",
        "GeneMamba",
    ):

        x = foundation[
            "models"
        ][model]

        foundation_rows.append({
            "model":
                model,

            "macro_f1_mean":
                float(
                    x[
                        "macro_f1_mean"
                    ]
                ),

            "macro_f1_min":
                float(
                    x[
                        "macro_f1_min"
                    ]
                ),

            "balanced_accuracy_mean":
                float(
                    x[
                        "balanced_accuracy_mean"
                    ]
                ),

            "log_loss_mean":
                float(
                    x[
                        "log_loss_mean"
                    ]
                ),

            "heldout_donors":
                4,

            "training_only_preprocessing":
                True,
        })

    foundation_summary = pd.DataFrame(
        foundation_rows
    )


    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    folds.to_csv(
        out
        / "expression_fold_metrics.csv",
        index=False,
    )

    classes.to_csv(
        out
        / "expression_per_class_metrics.csv",
        index=False,
    )

    confusion.to_csv(
        out
        / "expression_confusion_long.csv",
        index=False,
    )

    representation.to_csv(
        out
        / "expression_representation_summary.csv",
        index=False,
    )

    donor_repaired.to_csv(
        out
        / "expression_repaired_donor_means.csv",
        index=False,
    )

    condition_summary.to_csv(
        out
        / "expression_condition_summary.csv",
        index=False,
    )

    repair_gain.to_csv(
        out
        / "expression_repair_gain_vs_corrupted.csv",
        index=False,
    )

    policy_comparison.to_csv(
        out
        / "expression_gated_vs_ungated_lodo.csv",
        index=False,
    )

    foundation_summary.to_csv(
        out
        / "foundation_representation_lodo_summary.csv",
        index=False,
    )


    #
    # Compact final scientific receipt.
    #
    clean_macro = float(
        condition_summary[
            (
                condition_summary[
                    "kind"
                ]
                == "clean"
            )
            &
            (
                condition_summary[
                    "metric"
                ]
                == "macro_f1"
            )
        ][
            "mean"
        ].iloc[0]
    )

    final_receipt = {
        "schema":
            "sc2-p3-baron-final-lodo-aggregate-v1",

        "status":
            "PASS",

        "dataset":
            "baron_pancreas",

        "biological_unit":
            "donor",

        "heldout_donors": [
            "human1",
            "human2",
            "human3",
            "human4",
        ],

        "expression_experiment": {
            "status":
                "PASS",

            "representations":
                34,

            "fold_evaluations":
                136,

            "clean_expression_macro_f1_mean":
                clean_macro,

            "conditions": [
                "clean",
                "corrupted_mask15",
                "corrupted_mask30",
                "corrupted_mask50",
                "SC2-Gated_mask15/30/50",
                "SC2-Ungated_mask15/30/50",
            ],

            "training_only_preprocessing":
                True,

            "primary_metric":
                "macro_f1",

            "repair_gain_reference":
                "same-mask corrupted expression",
        },

        "foundation_representation_experiment": {
            "status":
                "REUSED_FROZEN_COMPLETE",

            "Geneformer":
                foundation[
                    "models"
                ][
                    "Geneformer"
                ],

            "GeneMamba":
                foundation[
                    "models"
                ][
                    "GeneMamba"
                ],
        },

        "interpretation_contract": {
            "expression_and_embedding_tracks_are_distinct":
                True,

            "single_leaderboard_across_tracks":
                False,

            "model_seeds_are_biological_replicates":
                False,

            "donors_are_biological_units":
                True,
        },

        "protocol":
            str(
                protocol_path
            ),

        "protocol_sha256":
            sha(
                protocol_path
            ),

        "foundation_lodo_receipt":
            str(
                foundation_receipt_path
            ),

        "foundation_lodo_receipt_sha256":
            expected_foundation_sha,
    }

    (
        out
        / "final_lodo_receipt.json"
    ).write_text(
        json.dumps(
            final_receipt,
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
        "BARON FINAL LODO — EXPRESSION CONDITIONS"
    )
    print(
        "============================================================"
    )

    print(
        condition_summary[
            condition_summary[
                "metric"
            ]
            == "macro_f1"
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "REPAIR GAIN VS CORRUPTED — MACRO F1"
    )
    print(
        "============================================================"
    )

    print(
        repair_gain[
            repair_gain[
                "metric"
            ]
            == "macro_f1"
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "GATED VS UNGATED — LODO MACRO F1"
    )
    print(
        "============================================================"
    )

    print(
        policy_comparison[
            policy_comparison[
                "metric"
            ]
            == "macro_f1"
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "FOUNDATION REPRESENTATION LODO — ALREADY FROZEN"
    )
    print(
        "============================================================"
    )

    print(
        foundation_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "P3_BARON_FINAL_LODO_AGGREGATION=PASS"
    )


if __name__ == "__main__":
    main()
