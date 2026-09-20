#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sc2.config import load_yaml
from sc2.eval.selective_repair_metrics import (
    gate_discrimination,
    masked_value_metrics,
)
from sc2.models.striped.sc2_striped_full import (
    build_sc2_striped_full_from_config,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True

    return bool(
        np.isclose(
            a,
            b,
            rtol=2.0e-5,
            atol=2.0e-6,
        )
    )


def selection_metrics(
    selected: np.ndarray,
    positive: np.ndarray,
    true_zero: np.ndarray,
) -> dict[str, float]:

    tp = int(
        (selected & positive).sum()
    )

    fp = int(
        (selected & true_zero).sum()
    )

    n_positive = int(
        positive.sum()
    )

    n_true_zero = int(
        true_zero.sum()
    )

    return {
        "selection_recall":
            tp / max(1, n_positive),

        "selection_precision":
            tp / max(1, tp + fp),

        "policy_true_zero_fill":
            fp / max(1, n_true_zero),

        "selected_true_positive":
            tp,

        "selected_true_zero":
            fp,
    }


def reconstruction_metrics(
    *,
    prediction: np.ndarray,
    reconstruction: np.ndarray,
    selected: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    positive: np.ndarray,
    true_zero: np.ndarray,
    observed_zero: np.ndarray,
    observed_nonzero: np.ndarray,
    available: np.ndarray,
    zero_threshold: float,
) -> dict[str, float]:

    result = masked_value_metrics(
        prediction,
        y,
        positive,
    )

    target_sd = float(
        result["target_sd"]
    )

    result["recovery_index"] = float(
        1.0
        - float(result["masked_mse"])
        / (target_sd ** 2)
    )

    result["sd_ratio_abs_error"] = abs(
        float(result["sd_ratio"])
        - 1.0
    )

    result.update(
        selection_metrics(
            selected,
            positive,
            true_zero,
        )
    )

    if bool(true_zero.any()):
        zero_values = reconstruction[
            true_zero
        ]

        result[
            "numeric_true_zero_positive_fraction"
        ] = float(
            np.mean(
                zero_values
                > zero_threshold
            )
        )

        result[
            "true_zero_mse"
        ] = float(
            np.mean(
                zero_values ** 2
            )
        )

        result[
            "true_zero_mae"
        ] = float(
            np.mean(
                np.abs(
                    zero_values
                )
            )
        )

    else:
        result[
            "numeric_true_zero_positive_fraction"
        ] = float("nan")

        result[
            "true_zero_mse"
        ] = float("nan")

        result[
            "true_zero_mae"
        ] = float("nan")

    if bool(observed_zero.any()):
        zerr = (
            reconstruction[
                observed_zero
            ]
            - y[
                observed_zero
            ]
        )

        result[
            "observed_zero_mse"
        ] = float(
            np.mean(
                zerr ** 2
            )
        )

        result[
            "observed_zero_mae"
        ] = float(
            np.mean(
                np.abs(
                    zerr
                )
            )
        )

    available2 = np.broadcast_to(
        available[None, :],
        y.shape,
    )

    ferr = (
        reconstruction[
            available2
        ]
        - y[
            available2
        ]
    )

    result[
        "full_available_mse"
    ] = float(
        np.mean(
            ferr ** 2
        )
    )

    result[
        "full_available_mae"
    ] = float(
        np.mean(
            np.abs(
                ferr
            )
        )
    )

    if bool(observed_nonzero.any()):
        preservation = (
            reconstruction[
                observed_nonzero
            ]
            - x[
                observed_nonzero
            ]
        )

        result[
            "observed_nonzero_mse"
        ] = float(
            np.mean(
                preservation ** 2
            )
        )

        result[
            "observed_nonzero_mae"
        ] = float(
            np.mean(
                np.abs(
                    preservation
                )
            )
        )

        result[
            "observed_nonzero_max_abs_error"
        ] = float(
            np.max(
                np.abs(
                    preservation
                )
            )
        )

        result[
            "observed_nonzero_changed_fraction"
        ] = float(
            np.mean(
                preservation != 0.0
            )
        )

    else:
        raise RuntimeError(
            "No observed nonzero values"
        )

    return result


def write_sha_manifest(root: Path) -> None:

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
        "".join(
            f"{sha256_file(p)}  "
            f"{p.relative_to(root).as_posix()}\n"
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
        "--task-id",
        required=True,
        type=int,
    )

    args = ap.parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    protocol = json.loads(
        protocol_path.read_text()
    )

    if (
        protocol["protocol_id"]
        !=
        "sc2-p3-16k-normalized-gated-vs-ungated-v1"
    ):
        raise RuntimeError(
            "Wrong protocol"
        )

    expected_protocol_sha = (
        os.environ.get(
            "EXPECTED_PROTOCOL_SHA"
        )
    )

    if (
        expected_protocol_sha
        and sha256_file(
            protocol_path
        )
        != expected_protocol_sha
    ):
        raise RuntimeError(
            "Protocol SHA mismatch"
        )

    tasks = {
        int(t["task_id"]): t
        for t in protocol["tasks"]
    }

    if len(tasks) != 5:
        raise RuntimeError(
            "Expected five model-seed tasks"
        )

    if args.task_id not in tasks:
        raise RuntimeError(
            "Unknown task ID"
        )

    task = tasks[
        args.task_id
    ]

    seed = int(
        task["seed"]
    )

    config_path = Path(
        task["config"]
    )

    checkpoint_path = Path(
        task["checkpoint"]
    )

    if (
        sha256_file(
            config_path
        )
        != task[
            "config_sha256"
        ]
    ):
        raise RuntimeError(
            "Config SHA mismatch"
        )

    if (
        sha256_file(
            checkpoint_path
        )
        != task[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Checkpoint SHA mismatch"
        )

    cfg = load_yaml(
        config_path
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "GPU is required"
        )

    model = (
        build_sc2_striped_full_from_config(
            cfg["model"],
            n_genes=16384,
        )
        .to(device)
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    batch_size = int(
        cfg.get(
            "batch_size",
            32,
        )
    )

    output_root = Path(
        protocol[
            "output_root"
        ]
    )

    final_dir = (
        output_root
        / (
            f"task_{args.task_id:02d}"
            f"_s{seed}"
        )
    )

    temp_dir = (
        output_root
        / (
            f".tmp_task_{args.task_id:02d}"
            f"_s{seed}"
            f"_{os.getpid()}"
        )
    )

    if final_dir.exists():
        raise RuntimeError(
            f"Output exists: {final_dir}"
        )

    if temp_dir.exists():
        raise RuntimeError(
            f"Temporary output exists: {temp_dir}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir.mkdir()

    condition_records = []

    try:

        for condition in protocol[
            "conditions"
        ]:

            dataset = condition[
                "dataset"
            ]

            mask_rate = int(
                condition[
                    "mask_rate"
                ]
            )

            threshold = float(
                condition[
                    "threshold"
                ]
            )

            panel_path = Path(
                condition[
                    "panel"
                ]
            )

            panel_sha = (
                condition[
                    "panel_sha256"
                ]
            )

            historical_path = Path(
                task[
                    "historical_summaries"
                ][
                    condition[
                        "condition_id"
                    ]
                ][
                    "path"
                ]
            )

            historical_sha = (
                task[
                    "historical_summaries"
                ][
                    condition[
                        "condition_id"
                    ]
                ][
                    "sha256"
                ]
            )

            if (
                sha256_file(
                    panel_path
                )
                != panel_sha
            ):
                raise RuntimeError(
                    f"Panel SHA mismatch: "
                    f"{panel_path}"
                )

            if (
                sha256_file(
                    historical_path
                )
                != historical_sha
            ):
                raise RuntimeError(
                    "Historical summary "
                    "SHA mismatch"
                )

            historical = json.loads(
                historical_path.read_text()
            )

            if (
                historical[
                    "checkpoint"
                ]
                != str(
                    checkpoint_path
                )
            ):
                raise RuntimeError(
                    "Historical checkpoint "
                    "mismatch"
                )

            if (
                historical[
                    "benchmark"
                ]
                != str(
                    panel_path
                )
            ):
                raise RuntimeError(
                    "Historical benchmark "
                    "mismatch"
                )

            if not np.isclose(
                float(
                    historical[
                        "threshold"
                    ]
                ),
                threshold,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Historical threshold "
                    "mismatch"
                )

            with np.load(
                panel_path,
                allow_pickle=False,
            ) as data:

                x = np.asarray(
                    data["x"],
                    dtype=np.float32,
                )

                y = np.asarray(
                    data["y"],
                    dtype=np.float32,
                )

                positive = np.asarray(
                    data[
                        "synthetic_mask"
                    ],
                    dtype=bool,
                )

                if (
                    "available_gene_mask"
                    in data.files
                ):
                    available = np.asarray(
                        data[
                            "available_gene_mask"
                        ],
                        dtype=bool,
                    )
                else:
                    available = (
                        np.ones(
                            16384,
                            dtype=bool,
                        )
                    )

            if x.shape != (
                5000,
                16384,
            ):
                raise RuntimeError(
                    f"Unexpected shape: "
                    f"{x.shape}"
                )

            if (
                y.shape != x.shape
                or positive.shape
                != x.shape
            ):
                raise RuntimeError(
                    "Panel shape mismatch"
                )

            if available.shape != (
                16384,
            ):
                raise RuntimeError(
                    "Availability shape "
                    "mismatch"
                )

            expected_chunks = []
            probability_chunks = []

            with torch.inference_mode():

                for start in range(
                    0,
                    x.shape[0],
                    batch_size,
                ):

                    tensor = (
                        torch.from_numpy(
                            x[
                                start:
                                start
                                + batch_size
                            ]
                        )
                        .to(device)
                    )

                    outputs = model(
                        tensor,
                        modality="sc",
                        return_dict=True,
                    )

                    expected_chunks.append(
                        outputs[
                            "expected_repair"
                        ]
                        .float()
                        .cpu()
                        .numpy()
                    )

                    probability_chunks.append(
                        outputs[
                            "dropout_probability"
                        ]
                        .float()
                        .cpu()
                        .numpy()
                    )

            expected = np.concatenate(
                expected_chunks,
                axis=0,
            )

            probability = np.concatenate(
                probability_chunks,
                axis=0,
            )

            zero_threshold = float(
                model.zero_threshold
            )

            available2 = np.broadcast_to(
                available[
                    None,
                    :
                ],
                y.shape,
            )

            true_zero = (
                y
                <= zero_threshold
            ) & available2

            observed_zero = (
                np.abs(x)
                <= zero_threshold
            ) & available2

            observed_nonzero = (
                np.abs(x)
                > zero_threshold
            ) & available2

            if np.any(
                positive
                & ~available2
            ):
                raise RuntimeError(
                    "Synthetic mask contains "
                    "unavailable genes"
                )

            selected_gated = (
                probability
                >= threshold
            )

            selected_ungated = (
                observed_zero
            )

            #
            # IMPORTANT DEFINITIONS
            #
            # UNGATED:
            # expected_repair at every observed zero.
            #
            # GATED:
            # expected_repair only where
            # dropout_probability >= frozen threshold.
            #

            ungated_prediction = (
                expected
            )

            gated_prediction = np.where(
                selected_gated,
                expected,
                0.0,
            ).astype(
                np.float32,
                copy=False,
            )

            ungated_reconstruction = (
                x.copy()
            )

            ungated_reconstruction[
                observed_zero
            ] = expected[
                observed_zero
            ]

            gated_reconstruction = (
                x.copy()
            )

            gated_fill = (
                observed_zero
                & selected_gated
            )

            gated_reconstruction[
                gated_fill
            ] = expected[
                gated_fill
            ]

            ungated_metrics = (
                reconstruction_metrics(
                    prediction=
                        ungated_prediction,
                    reconstruction=
                        ungated_reconstruction,
                    selected=
                        selected_ungated,
                    x=x,
                    y=y,
                    positive=positive,
                    true_zero=true_zero,
                    observed_zero=
                        observed_zero,
                    observed_nonzero=
                        observed_nonzero,
                    available=available,
                    zero_threshold=
                        zero_threshold,
                )
            )

            gated_metrics = (
                reconstruction_metrics(
                    prediction=
                        gated_prediction,
                    reconstruction=
                        gated_reconstruction,
                    selected=
                        selected_gated,
                    x=x,
                    y=y,
                    positive=positive,
                    true_zero=true_zero,
                    observed_zero=
                        observed_zero,
                    observed_nonzero=
                        observed_nonzero,
                    available=available,
                    zero_threshold=
                        zero_threshold,
                )
            )

            gate_metrics = (
                gate_discrimination(
                    probability,
                    positive,
                    true_zero,
                )
            )

            #
            # Historical reproduction gate.
            #
            # Existing SC2 value metrics are expected_repair
            # without a hard threshold.
            #
            for key in (
                "masked_mse",
                "masked_mae",
                "sample_spearman",
                "gene_spearman",
                "prediction_sd",
                "target_sd",
                "sd_ratio",
            ):

                if not close(
                    float(
                        ungated_metrics[
                            key
                        ]
                    ),
                    float(
                        historical[
                            key
                        ]
                    ),
                ):
                    raise RuntimeError(
                        "Historical ungated "
                        f"reproduction failed: "
                        f"{dataset} "
                        f"mask{mask_rate} "
                        f"seed={seed} "
                        f"metric={key} "
                        f"new="
                        f"{ungated_metrics[key]} "
                        f"old="
                        f"{historical[key]}"
                    )

            #
            # Frozen threshold selection metrics must also
            # reproduce the original run.
            #
            for new_key, old_key in (
                (
                    "selection_recall",
                    "threshold_recall",
                ),
                (
                    "selection_precision",
                    "threshold_precision",
                ),
                (
                    "policy_true_zero_fill",
                    "threshold_true_zero_fill",
                ),
            ):

                if not close(
                    float(
                        gated_metrics[
                            new_key
                        ]
                    ),
                    float(
                        historical[
                            old_key
                        ]
                    ),
                ):
                    raise RuntimeError(
                        "Historical gated "
                        "selection reproduction "
                        f"failed: {dataset} "
                        f"mask{mask_rate} "
                        f"seed={seed} "
                        f"{new_key}"
                    )

            for new_key, old_key in (
                ("auroc", "gate_auroc"),
                ("auprc", "gate_auprc"),
                ("brier", "gate_brier"),
                ("ece", "gate_ece"),
            ):

                if not close(
                    float(
                        gate_metrics[
                            new_key
                        ]
                    ),
                    float(
                        historical[
                            old_key
                        ]
                    ),
                ):
                    raise RuntimeError(
                        "Gate metric "
                        "reproduction failed: "
                        f"{dataset} "
                        f"mask{mask_rate} "
                        f"seed={seed} "
                        f"{new_key}"
                    )

            for mode_metrics in (
                ungated_metrics,
                gated_metrics,
            ):

                if (
                    float(
                        mode_metrics[
                            "observed_nonzero_changed_fraction"
                        ]
                    )
                    != 0.0
                ):
                    raise RuntimeError(
                        "Observed nonzero "
                        "was changed"
                    )

                if (
                    float(
                        mode_metrics[
                            "observed_nonzero_max_abs_error"
                        ]
                    )
                    != 0.0
                ):
                    raise RuntimeError(
                        "Observed nonzero "
                        "preservation error"
                    )

            condition_dir = (
                temp_dir
                / dataset
                / f"mask{mask_rate}"
            )

            condition_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            payload = {
                "schema":
                    "sc2-p3-16k-normalized-gated-vs-ungated-condition-v1",

                "dataset":
                    dataset,

                "mask_rate":
                    mask_rate,

                "seed":
                    seed,

                "threshold":
                    threshold,

                "panel":
                    str(
                        panel_path
                    ),

                "panel_sha256":
                    panel_sha,

                "checkpoint":
                    str(
                        checkpoint_path
                    ),

                "checkpoint_sha256":
                    task[
                        "checkpoint_sha256"
                    ],

                "historical_summary":
                    str(
                        historical_path
                    ),

                "historical_summary_sha256":
                    historical_sha,

                "ungated_definition":
                    (
                        "expected_repair at every "
                        "observed zero; no hard "
                        "probability threshold"
                    ),

                "gated_definition":
                    (
                        "expected_repair only at "
                        "observed zeros where "
                        "dropout_probability >= "
                        "frozen threshold"
                    ),

                "historical_reproduction":
                    "PASS",

                "ungated":
                    ungated_metrics,

                "gated":
                    gated_metrics,

                "gate_metrics":
                    gate_metrics,
            }

            (
                condition_dir
                / "metrics.json"
            ).write_text(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )

            #
            # Preserve Baron repaired matrices now so the
            # later LODO does not require another SC2
            # inference pass.
            #
            if (
                dataset
                == "baron_pancreas"
            ):

                np.save(
                    condition_dir
                    / "ungated_reconstruction.npy",
                    np.asarray(
                        ungated_reconstruction,
                        dtype=np.float32,
                        order="C",
                    ),
                    allow_pickle=False,
                )

                np.save(
                    condition_dir
                    / "gated_reconstruction.npy",
                    np.asarray(
                        gated_reconstruction,
                        dtype=np.float32,
                        order="C",
                    ),
                    allow_pickle=False,
                )

            condition_records.append(
                {
                    "dataset":
                        dataset,

                    "mask_rate":
                        mask_rate,

                    "threshold":
                        threshold,

                    "metrics":
                        str(
                            (
                                condition_dir
                                / "metrics.json"
                            )
                            .relative_to(
                                temp_dir
                            )
                        ),

                    "baron_repaired_matrices_saved":
                        dataset
                        == "baron_pancreas",
                }
            )

            del (
                x,
                y,
                positive,
                expected,
                probability,
                ungated_prediction,
                gated_prediction,
                ungated_reconstruction,
                gated_reconstruction,
                true_zero,
                observed_zero,
                observed_nonzero,
            )

            torch.cuda.empty_cache()

        if len(
            condition_records
        ) != 9:
            raise RuntimeError(
                "Expected nine normalized "
                "conditions per seed"
            )

        manifest = {
            "schema":
                "sc2-p3-16k-normalized-gated-vs-ungated-task-v1",

            "status":
                "PASS",

            "analysis_status":
                protocol[
                    "analysis_status"
                ],

            "task_id":
                args.task_id,

            "seed":
                seed,

            "conditions":
                condition_records,

            "historical_ungated_reproduction":
                "PASS",

            "historical_threshold_metric_reproduction":
                "PASS",

            "observed_nonzero_preservation":
                "EXACT",

            "baron_repaired_matrices_saved":
                True,

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
            temp_dir
            / "task_manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        write_sha_manifest(
            temp_dir
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
        "P3_NORMALIZED_GATED_VS_UNGATED_TASK=PASS"
    )

    print(
        f"TASK_ID={args.task_id}"
    )

    print(
        f"SEED={seed}"
    )

    print(
        "CONDITIONS=9"
    )

    print(
        "BARON_REPAIRED_MATRICES_SAVED=true"
    )


if __name__ == "__main__":
    main()
