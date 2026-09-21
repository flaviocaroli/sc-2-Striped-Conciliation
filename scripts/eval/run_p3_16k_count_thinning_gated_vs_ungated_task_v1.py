#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from sc2.config import load_yaml

from sc2.eval.selective_repair_metrics import (
    gate_discrimination,
)

from sc2.models.striped.sc2_striped_full import (
    build_sc2_striped_full_from_config,
)

from scripts.eval.run_p3_16k_normalized_gated_vs_ungated_task_v1 import (
    reconstruction_metrics,
    sha256_file,
    write_sha_manifest,
)


PROTOCOL_ID = (
    "sc2-p3-16k-count-thinning-"
    "gated-vs-ungated-confirmatory-v1"
)


def qlabel(q: float) -> str:

    return (
        f"q{int(round(q * 100)):03d}"
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
        != PROTOCOL_ID
    ):
        raise RuntimeError(
            "Wrong protocol ID"
        )

    if (
        protocol["status"]
        !=
        "FROZEN_BEFORE_CONFIRMATORY_INFERENCE"
    ):
        raise RuntimeError(
            "Protocol not frozen"
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
        int(t["task_id"]):
            t
        for t in protocol[
            "task_layout"
        ][
            "tasks"
        ]
    }

    if len(tasks) != 25:
        raise RuntimeError(
            "Expected 25 tasks"
        )

    if args.task_id not in tasks:
        raise RuntimeError(
            "Unknown task"
        )

    task = tasks[
        args.task_id
    ]

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

    model_info = protocol[
        "models"
    ][
        str(model_seed)
    ]

    config_path = Path(
        model_info[
            "config_path"
        ]
    )

    checkpoint_path = Path(
        model_info[
            "checkpoint_path"
        ]
    )

    if (
        sha256_file(
            config_path
        )
        != model_info[
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
        != model_info[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Checkpoint SHA mismatch"
        )

    panel_receipt_path = Path(
        protocol[
            "panels"
        ][
            "receipt_path"
        ]
    )

    if (
        sha256_file(
            panel_receipt_path
        )
        != protocol[
            "panels"
        ][
            "receipt_sha256"
        ]
    ):
        raise RuntimeError(
            "Panel receipt SHA mismatch"
        )

    panel_receipt = json.loads(
        panel_receipt_path.read_text()
    )

    if (
        panel_receipt[
            "status"
        ]
        != "FROZEN_COMPLETE"
    ):
        raise RuntimeError(
            "Panel receipt not frozen"
        )

    #
    # Resolve all 9 panels before model inference.
    #
    selected = {}

    for dataset in protocol[
        "datasets"
    ]:

        records = panel_receipt[
            "datasets_detail"
        ][dataset][
            "panels"
        ]

        for q in protocol[
            "q_order"
        ]:

            q = float(q)

            matches = [
                x
                for x in records
                if (
                    np.isclose(
                        float(x["q"]),
                        q,
                        atol=1e-12,
                        rtol=0.0,
                    )
                    and int(
                        x["replicate"]
                    )
                    == replicate
                )
            ]

            if len(matches) != 1:
                raise RuntimeError(
                    "Panel lookup failed: "
                    f"{dataset} q={q} "
                    f"rep={replicate}"
                )

            item = matches[0]

            panel = Path(
                item["path"]
            )

            if not panel.is_file():
                raise RuntimeError(
                    f"Missing panel: {panel}"
                )

            if (
                sha256_file(panel)
                != item["sha256"]
            ):
                raise RuntimeError(
                    f"Panel SHA mismatch: "
                    f"{panel}"
                )

            selected[
                (
                    dataset,
                    q,
                )
            ] = (
                item,
                panel,
            )

    if len(selected) != 9:
        raise RuntimeError(
            "Expected 9 selected panels"
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
            "GPU required"
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
            f"task_{args.task_id:02d}_"
            f"model{model_seed}_"
            f"rep{replicate:02d}"
        )
    )

    temp_dir = (
        output_root
        / (
            f".tmp_task_{args.task_id:02d}_"
            f"model{model_seed}_"
            f"rep{replicate:02d}_"
            f"{os.getpid()}"
        )
    )

    if final_dir.exists():
        raise RuntimeError(
            f"Final output exists: "
            f"{final_dir}"
        )

    if temp_dir.exists():
        raise RuntimeError(
            f"Temp output exists: "
            f"{temp_dir}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir.mkdir()

    condition_records = []

    try:

        for dataset in protocol[
            "datasets"
        ]:

            for q_raw in protocol[
                "q_order"
            ]:

                q = float(
                    q_raw
                )

                item, panel_path = (
                    selected[
                        (
                            dataset,
                            q,
                        )
                    ]
                )

                threshold = float(
                    protocol[
                        "frozen_thresholds"
                    ][
                        f"{q:.2f}"
                    ]
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
                            "lost_positive_mask"
                        ],
                        dtype=bool,
                    )

                    synthetic = np.asarray(
                        data[
                            "synthetic_mask"
                        ],
                        dtype=bool,
                    )

                    originally_zero = (
                        np.asarray(
                            data[
                                "originally_zero_mask"
                            ],
                            dtype=bool,
                        )
                    )

                    available = np.asarray(
                        data[
                            "available_gene_mask"
                        ],
                        dtype=bool,
                    )

                if x.shape != (
                    5000,
                    16384,
                ):
                    raise RuntimeError(
                        f"Bad x shape: {x.shape}"
                    )

                if (
                    y.shape != x.shape
                    or positive.shape
                    != x.shape
                    or originally_zero.shape
                    != x.shape
                ):
                    raise RuntimeError(
                        "Panel array shape mismatch"
                    )

                if available.shape != (
                    16384,
                ):
                    raise RuntimeError(
                        "Availability shape mismatch"
                    )

                if not np.array_equal(
                    positive,
                    synthetic,
                ):
                    raise RuntimeError(
                        "Synthetic/lost-positive "
                        "mask mismatch"
                    )

                available2 = (
                    np.broadcast_to(
                        available[
                            None,
                            :
                        ],
                        x.shape,
                    )
                )

                if np.any(
                    positive
                    & ~available2
                ):
                    raise RuntimeError(
                        "Positive mask contains "
                        "unavailable genes"
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

                expected = (
                    np.concatenate(
                        expected_chunks,
                        axis=0,
                    )
                )

                probability = (
                    np.concatenate(
                        probability_chunks,
                        axis=0,
                    )
                )

                zero_threshold = float(
                    model.zero_threshold
                )

                computed_true_zero = (
                    (
                        y
                        <= zero_threshold
                    )
                    & available2
                )

                if not np.array_equal(
                    computed_true_zero,
                    originally_zero,
                ):
                    raise RuntimeError(
                        "Originally-zero mask does "
                        "not equal full-depth "
                        "normalized true-zero mask"
                    )

                observed_zero = (
                    (
                        np.abs(x)
                        <= zero_threshold
                    )
                    & available2
                )

                observed_nonzero = (
                    (
                        np.abs(x)
                        > zero_threshold
                    )
                    & available2
                )

                if np.any(
                    positive
                    & ~observed_zero
                ):
                    raise RuntimeError(
                        "Lost-positive entry is "
                        "not observed zero"
                    )

                if np.any(
                    positive
                    & originally_zero
                ):
                    raise RuntimeError(
                        "Gate classes overlap"
                    )

                selected_gated = (
                    observed_zero
                    & (
                        probability
                        >= threshold
                    )
                )

                selected_ungated = (
                    observed_zero
                )

                ungated_prediction = (
                    expected
                )

                gated_prediction = (
                    np.where(
                        selected_gated,
                        expected,
                        0.0,
                    )
                    .astype(
                        np.float32,
                        copy=False,
                    )
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

                gated_reconstruction[
                    selected_gated
                ] = expected[
                    selected_gated
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

                        positive=
                            positive,

                        true_zero=
                            originally_zero,

                        observed_zero=
                            observed_zero,

                        observed_nonzero=
                            observed_nonzero,

                        available=
                            available,

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

                        positive=
                            positive,

                        true_zero=
                            originally_zero,

                        observed_zero=
                            observed_zero,

                        observed_nonzero=
                            observed_nonzero,

                        available=
                            available,

                        zero_threshold=
                            zero_threshold,
                    )
                )

                gate_metrics = (
                    gate_discrimination(
                        probability,
                        positive,
                        originally_zero,
                    )
                )

                for metrics in (
                    ungated_metrics,
                    gated_metrics,
                ):

                    if (
                        float(
                            metrics[
                                "observed_nonzero_changed_fraction"
                            ]
                        )
                        != 0.0
                    ):
                        raise RuntimeError(
                            "Observed nonzero changed"
                        )

                    if (
                        float(
                            metrics[
                                "observed_nonzero_max_abs_error"
                            ]
                        )
                        != 0.0
                    ):
                        raise RuntimeError(
                            "Observed nonzero "
                            "preservation error"
                        )

                #
                # Operating-policy invariants.
                #
                if not np.isclose(
                    float(
                        ungated_metrics[
                            "selection_recall"
                        ]
                    ),
                    1.0,
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        "Ungated recall must be 1"
                    )

                if not np.isclose(
                    float(
                        ungated_metrics[
                            "policy_true_zero_fill"
                        ]
                    ),
                    1.0,
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        "Ungated true-zero "
                        "selection coverage must "
                        "be 1"
                    )

                condition_dir = (
                    temp_dir
                    / dataset
                    / qlabel(q)
                )

                condition_dir.mkdir(
                    parents=True,
                    exist_ok=False,
                )

                payload = {
                    "schema":
                        "sc2-p3-16k-count-thinning-gated-vs-ungated-condition-v1",

                    "dataset":
                        dataset,

                    "q":
                        q,

                    "replicate":
                        replicate,

                    "thinning_seed":
                        int(
                            item[
                                "thinning_seed"
                            ]
                        ),

                    "model_seed":
                        model_seed,

                    "threshold":
                        threshold,

                    "panel":
                        str(
                            panel_path
                        ),

                    "panel_sha256":
                        item["sha256"],

                    "checkpoint":
                        str(
                            checkpoint_path
                        ),

                    "checkpoint_sha256":
                        model_info[
                            "checkpoint_sha256"
                        ],

                    "ungated":
                        ungated_metrics,

                    "gated":
                        gated_metrics,

                    "gate_metrics":
                        gate_metrics,

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

                    "observed_nonzero_preservation":
                        "EXACT",
                }

                metrics_path = (
                    condition_dir
                    / "metrics.json"
                )

                metrics_path.write_text(
                    json.dumps(
                        payload,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                condition_records.append(
                    {
                        "dataset":
                            dataset,

                        "q":
                            q,

                        "replicate":
                            replicate,

                        "thinning_seed":
                            int(
                                item[
                                    "thinning_seed"
                                ]
                            ),

                        "threshold":
                            threshold,

                        "metrics":
                            str(
                                metrics_path.relative_to(
                                    temp_dir
                                )
                            ),
                    }
                )

                del (
                    x,
                    y,
                    positive,
                    synthetic,
                    originally_zero,
                    available,
                    available2,
                    expected,
                    probability,
                    ungated_prediction,
                    gated_prediction,
                    ungated_reconstruction,
                    gated_reconstruction,
                    observed_zero,
                    observed_nonzero,
                    computed_true_zero,
                )

                torch.cuda.empty_cache()

        if len(
            condition_records
        ) != 9:
            raise RuntimeError(
                "Expected 9 conditions"
            )

        manifest = {
            "schema":
                "sc2-p3-16k-count-thinning-gated-vs-ungated-task-v1",

            "status":
                "PASS",

            "task_id":
                args.task_id,

            "model_seed":
                model_seed,

            "replicate":
                replicate,

            "conditions":
                condition_records,

            "forward_evaluations":
                9,

            "paired_mode_results":
                18,

            "observed_nonzero_preservation":
                "EXACT",

            "threshold_retuning":
                False,

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
        "P3_THINNING_GATED_VS_UNGATED_TASK=PASS"
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
        "CONDITIONS=9"
    )


if __name__ == "__main__":
    main()
