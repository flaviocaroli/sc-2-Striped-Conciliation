#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.analysis.run_baron_foundation_lodo_v1 import (
    fold_eval,
)


PROTOCOL_ID = (
    "sc2-p3-baron-final-expression-lodo-v1"
)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def write_manifest(root):

    files = sorted(
        p
        for p in Path(root).iterdir()
        if p.is_file()
        and p.name != "SHA256SUMS.txt"
    )

    (
        Path(root)
        / "SHA256SUMS.txt"
    ).write_text(
        "".join(
            f"{sha(p)}  {p.name}\n"
            for p in files
        )
    )


def main():

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

    p = json.loads(
        protocol_path.read_text()
    )

    if (
        p["protocol_id"]
        != PROTOCOL_ID
    ):
        raise RuntimeError(
            "wrong protocol"
        )

    expected_sha = os.environ.get(
        "EXPECTED_PROTOCOL_SHA"
    )

    if (
        expected_sha
        and sha(protocol_path)
        != expected_sha
    ):
        raise RuntimeError(
            "protocol SHA mismatch"
        )

    tasks = {
        int(x["task_id"]):
            x
        for x in p["tasks"]
    }

    if len(tasks) != 11:
        raise RuntimeError(
            "expected 11 tasks"
        )

    if args.task_id not in tasks:
        raise RuntimeError(
            "unknown task"
        )

    task = tasks[
        args.task_id
    ]

    rep_by_id = {
        x["representation_id"]:
            x
        for x in p[
            "representations"
        ]
    }

    raw_bundle = Path(
        p[
            "raw_alignment_bundle"
        ][
            "path"
        ]
    )

    if (
        sha(raw_bundle)
        != p[
            "raw_alignment_bundle"
        ][
            "sha256"
        ]
    ):
        raise RuntimeError(
            "raw alignment bundle SHA mismatch"
        )

    with np.load(
        raw_bundle,
        allow_pickle=False,
    ) as d:

        raw_rows = np.asarray(
            d["source_row"],
            dtype=np.int64,
        )

        raw_cells = np.asarray(
            d["cell_id"]
        ).astype(str)

        labels = np.asarray(
            d["cell_type"]
        ).astype(str)

        donors = np.asarray(
            d["donor"]
        ).astype(str)

    if raw_rows.shape != (5000,):
        raise RuntimeError(
            "raw row shape mismatch"
        )

    if labels.shape != (5000,):
        raise RuntimeError(
            "label shape mismatch"
        )

    if set(donors) != {
        "human1",
        "human2",
        "human3",
        "human4",
    }:
        raise RuntimeError(
            "donor set mismatch"
        )

    donor_counts = {
        donor:
            int(
                np.sum(
                    donors == donor
                )
            )
        for donor in sorted(
            set(donors)
        )
    }

    if (
        donor_counts
        != p[
            "donor_counts"
        ]
    ):
        raise RuntimeError(
            "donor count mismatch"
        )

    global_classes = sorted(
        set(labels)
    )

    if len(global_classes) != 14:
        raise RuntimeError(
            "expected 14 Baron cell types"
        )

    heldouts = p[
        "heldout_donors"
    ]

    #
    # For baseline task prove all three frozen panel
    # targets encode exactly the same clean expression
    # and the exact same cell order.
    #
    if args.task_id == 0:

        clean_reference = None

        for mask in (
            15,
            30,
            50,
        ):

            source = p[
                "baron_panels"
            ][
                str(mask)
            ]

            path = Path(
                source["path"]
            )

            if sha(path) != source["sha256"]:
                raise RuntimeError(
                    "panel SHA mismatch"
                )

            with np.load(
                path,
                allow_pickle=False,
            ) as d:

                rows = np.asarray(
                    d["row"],
                    dtype=np.int64,
                )

                cells = np.asarray(
                    d["cell_id"]
                ).astype(str)

                types = np.asarray(
                    d["cell_type"]
                ).astype(str)

                target = np.asarray(
                    d["y"],
                    dtype=np.float32,
                )

            if not np.array_equal(
                rows,
                raw_rows,
            ):
                raise RuntimeError(
                    "panel/raw source_row mismatch"
                )

            if not np.array_equal(
                cells,
                raw_cells,
            ):
                raise RuntimeError(
                    "panel/raw cell_id mismatch"
                )

            if not np.array_equal(
                types,
                labels,
            ):
                raise RuntimeError(
                    "panel/raw cell_type mismatch"
                )

            if clean_reference is None:
                clean_reference = target

            elif not np.array_equal(
                clean_reference,
                target,
            ):
                raise RuntimeError(
                    "clean y differs across masks"
                )

        del clean_reference

    output_root = Path(
        p[
            "task_output_root"
        ]
    )

    final = (
        output_root
        / f"task_{args.task_id:02d}"
    )

    temp = (
        output_root
        / (
            f".tmp_task_{args.task_id:02d}_"
            f"{os.getpid()}"
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

    fold_rows = []
    class_rows = []
    confusion_rows = []
    summaries = []

    try:

        for rep_id in task[
            "representation_ids"
        ]:

            r = rep_by_id[
                rep_id
            ]

            source = Path(
                r["path"]
            )

            if sha(source) != r["sha256"]:
                raise RuntimeError(
                    f"representation SHA mismatch: "
                    f"{source}"
                )

            if r["format"] == "npz":

                with np.load(
                    source,
                    allow_pickle=False,
                ) as d:

                    X = np.asarray(
                        d[
                            r[
                                "array_key"
                            ]
                        ],
                        dtype=np.float32,
                    )

                    rows = np.asarray(
                        d["row"],
                        dtype=np.int64,
                    )

                    cells = np.asarray(
                        d["cell_id"]
                    ).astype(str)

                    types = np.asarray(
                        d["cell_type"]
                    ).astype(str)

                if not np.array_equal(
                    rows,
                    raw_rows,
                ):
                    raise RuntimeError(
                        "npz source-row mismatch"
                    )

                if not np.array_equal(
                    cells,
                    raw_cells,
                ):
                    raise RuntimeError(
                        "npz cell alignment mismatch"
                    )

                if not np.array_equal(
                    types,
                    labels,
                ):
                    raise RuntimeError(
                        "npz cell-type alignment mismatch"
                    )

            elif r["format"] == "npy":

                X = np.load(
                    source,
                    mmap_mode="r",
                    allow_pickle=False,
                )

            else:
                raise RuntimeError(
                    "unsupported representation format"
                )

            if X.shape != (
                5000,
                16384,
            ):
                raise RuntimeError(
                    f"{rep_id}: bad shape "
                    f"{X.shape}"
                )

            if not np.isfinite(
                np.asarray(X)
            ).all():
                raise RuntimeError(
                    f"{rep_id}: nonfinite values"
                )

            results = joblib.Parallel(
                n_jobs=4,
                prefer="threads",
            )(
                joblib.delayed(
                    fold_eval
                )(
                    X,
                    labels,
                    donors,
                    donor,
                    p[
                        "classifier"
                    ],
                    global_classes,
                )
                for donor in heldouts
            )

            rep_folds = []

            for (
                fold,
                classes,
                confusion,
            ) in results:

                metadata = {
                    "representation_id":
                        rep_id,

                    "kind":
                        r["kind"],

                    "mode":
                        r.get(
                            "mode"
                        ),

                    "mask_rate":
                        r.get(
                            "mask_rate"
                        ),

                    "model_seed":
                        r.get(
                            "model_seed"
                        ),
                }

                fold.update(
                    metadata
                )

                rep_folds.append(
                    fold
                )

                for row in classes:
                    row.update(
                        metadata
                    )

                for row in confusion:
                    row.update(
                        metadata
                    )

                fold_rows.append(
                    fold
                )

                class_rows.extend(
                    classes
                )

                confusion_rows.extend(
                    confusion
                )

            rdf = pd.DataFrame(
                rep_folds
            )

            if len(rdf) != 4:
                raise RuntimeError(
                    "representation does not "
                    "have four donor folds"
                )

            summaries.append({
                "representation_id":
                    rep_id,

                "kind":
                    r["kind"],

                "mode":
                    r.get(
                        "mode"
                    ),

                "mask_rate":
                    r.get(
                        "mask_rate"
                    ),

                "model_seed":
                    r.get(
                        "model_seed"
                    ),

                "macro_f1_mean":
                    float(
                        rdf[
                            "macro_f1"
                        ].mean()
                    ),

                "macro_f1_min":
                    float(
                        rdf[
                            "macro_f1"
                        ].min()
                    ),

                "balanced_accuracy_mean":
                    float(
                        rdf[
                            "balanced_accuracy"
                        ].mean()
                    ),

                "log_loss_mean":
                    float(
                        rdf[
                            "log_loss"
                        ].mean()
                    ),
            })

            print(
                json.dumps(
                    summaries[-1],
                    sort_keys=True,
                ),
                flush=True,
            )

            del X

        folds = pd.DataFrame(
            fold_rows
        )

        classes = pd.DataFrame(
            class_rows
        )

        confusion = pd.DataFrame(
            confusion_rows
        )

        summary_df = pd.DataFrame(
            summaries
        )

        expected_reps = len(
            task[
                "representation_ids"
            ]
        )

        if len(folds) != (
            expected_reps * 4
        ):
            raise RuntimeError(
                "fold count mismatch"
            )

        folds.to_csv(
            temp / "fold_metrics.csv",
            index=False,
        )

        classes.to_csv(
            temp / "per_class_metrics.csv",
            index=False,
        )

        confusion.to_csv(
            temp / "confusion_long.csv",
            index=False,
        )

        summary_df.to_csv(
            temp / "representation_summary.csv",
            index=False,
        )

        task_receipt = {
            "schema":
                "sc2-p3-baron-final-expression-lodo-task-v1",

            "status":
                "PASS",

            "task_id":
                args.task_id,

            "kind":
                task["kind"],

            "representation_count":
                expected_reps,

            "fold_count":
                int(
                    len(folds)
                ),

            "heldout_donors":
                heldouts,

            "biological_unit":
                "donor",

            "training_only_preprocessing":
                True,

            "standard_scaler":
                True,

            "pca_components":
                50,

            "classifier":
                "balanced_logistic_regression",

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
            temp
            / "task_receipt.json"
        ).write_text(
            json.dumps(
                task_receipt,
                indent=2,
                sort_keys=True,
            )
            + "\n"
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
        "P3_BARON_EXPRESSION_LODO_TASK=PASS"
    )

    print(
        f"TASK_ID={args.task_id}"
    )

    print(
        f"REPRESENTATIONS={len(task['representation_ids'])}"
    )


if __name__ == "__main__":
    main()
