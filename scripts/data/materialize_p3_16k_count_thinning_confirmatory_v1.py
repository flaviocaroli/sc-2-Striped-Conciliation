#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from sc2.data.count_thinning import (
    binomial_thin_counts,
    cp10k_log1p_counts,
    thinning_masks,
)


PROTOCOL_ID = (
    "sc2-p3-16k-count-thinning-"
    "confirmatory-materialization-v1"
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


def write_json(
    path: Path,
    payload,
) -> None:

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def panel_name(
    dataset: str,
    q: float,
    replicate: int,
    seed: int,
) -> str:

    qcode = int(
        round(q * 100)
    )

    return (
        f"{dataset}_"
        f"q{qcode:03d}_"
        f"rep{replicate:02d}_"
        f"seed{seed}.npz"
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
        protocol[
            "protocol_id"
        ]
        != PROTOCOL_ID
    ):
        raise RuntimeError(
            "Wrong protocol ID"
        )

    if (
        protocol["status"]
        !=
        "FROZEN_BEFORE_CONFIRMATORY_MATERIALIZATION"
    ):
        raise RuntimeError(
            "Protocol is not frozen"
        )

    tasks = {
        int(x["task_id"]):
            str(x["dataset"])
        for x in protocol[
            "tasks"
        ]
    }

    if args.task_id not in tasks:
        raise RuntimeError(
            "Unknown task ID"
        )

    dataset = tasks[
        args.task_id
    ]

    source = protocol[
        "datasets"
    ][dataset]

    output_dir = Path(
        source[
            "output_dir"
        ]
    ).resolve()

    if output_dir.exists():
        raise RuntimeError(
            f"Final output exists: "
            f"{output_dir}"
        )

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir = (
        output_dir.parent
        / (
            f".tmp_{output_dir.name}_"
            f"{os.getpid()}"
        )
    )

    if temp_dir.exists():
        raise RuntimeError(
            f"Temp exists: {temp_dir}"
        )

    counts_path = Path(
        source[
            "counts_path"
        ]
    )

    available_path = Path(
        source[
            "available_path"
        ]
    )

    metadata_path = Path(
        source[
            "metadata_path"
        ]
    )

    reference_path = Path(
        source[
            "reference_target_path"
        ]
    )

    expected_files = (
        (
            counts_path,
            source[
                "counts_file_sha256"
            ],
        ),
        (
            available_path,
            source[
                "available_file_sha256"
            ],
        ),
        (
            metadata_path,
            source[
                "metadata_sha256"
            ],
        ),
        (
            reference_path,
            source[
                "reference_target_sha256"
            ],
        ),
    )

    for path, expected_sha in (
        expected_files
    ):

        if not path.is_file():
            raise RuntimeError(
                f"Missing source: {path}"
            )

        observed = sha256_file(
            path
        )

        if observed != expected_sha:
            raise RuntimeError(
                f"SHA mismatch: {path}"
            )

    metadata = json.loads(
        metadata_path.read_text()
    )

    if metadata[
        "dataset"
    ] != dataset:
        raise RuntimeError(
            "Metadata dataset mismatch"
        )

    if metadata[
        "shape"
    ] != [5000, 16384]:
        raise RuntimeError(
            "Metadata shape mismatch"
        )

    counts = np.load(
        counts_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    available = np.asarray(
        np.load(
            available_path,
            allow_pickle=False,
        ),
        dtype=bool,
    )

    if counts.shape != (
        5000,
        16384,
    ):
        raise RuntimeError(
            f"Count shape={counts.shape}"
        )

    if counts.dtype != np.uint32:
        raise RuntimeError(
            f"Count dtype={counts.dtype}"
        )

    if available.shape != (
        16384,
    ):
        raise RuntimeError(
            "Availability shape mismatch"
        )

    if int(
        available.sum()
    ) != int(
        source[
            "n_available_genes"
        ]
    ):
        raise RuntimeError(
            "Available-gene count mismatch"
        )

    full_counts = np.asarray(
        counts,
        dtype=np.uint32,
    )

    if np.any(
        full_counts[
            :,
            ~available
        ] != 0
    ):
        raise RuntimeError(
            "Unavailable full-count genes "
            "are nonzero"
        )

    full_totals = full_counts.sum(
        axis=1,
        dtype=np.uint64,
    )

    if np.any(
        full_totals == 0
    ):
        raise RuntimeError(
            "Full-depth zero-library cell"
        )

    full_target = (
        cp10k_log1p_counts(
            full_counts,
            available,
            zero_library_rule="error",
        )
    )

    with np.load(
        reference_path,
        allow_pickle=False,
    ) as reference:

        frozen_y = np.asarray(
            reference["y"],
            dtype=np.float32,
        )

    if not np.array_equal(
        full_target,
        frozen_y,
    ):

        max_abs = float(
            np.max(
                np.abs(
                    full_target
                    - frozen_y
                )
            )
        )

        raise RuntimeError(
            "Full-target alignment failed; "
            f"max_abs={max_abs}"
        )

    full_positive = (
        (full_counts > 0)
        & available[None, :]
    )

    originally_zero_reference = (
        (full_counts == 0)
        & available[None, :]
    )

    n_full_positive = int(
        full_positive.sum()
    )

    n_original_zero = int(
        originally_zero_reference.sum()
    )

    full_molecules = int(
        full_counts.sum(
            dtype=np.uint64
        )
    )

    temp_dir.mkdir(
        exist_ok=False
    )

    records = []

    try:

        np.save(
            temp_dir
            / "available_gene_mask.npy",
            available,
            allow_pickle=False,
        )

        for specification in (
            source[
                "panel_specs"
            ]
        ):

            q = float(
                specification["q"]
            )

            replicate = int(
                specification[
                    "replicate"
                ]
            )

            seed = int(
                specification[
                    "seed"
                ]
            )

            thin = (
                binomial_thin_counts(
                    full_counts,
                    q=q,
                    seed=seed,
                    available_gene_mask=
                        available,
                )
            )

            if not np.issubdtype(
                thin.dtype,
                np.integer,
            ):
                raise RuntimeError(
                    "Thinned counts not integer"
                )

            if np.any(
                thin > full_counts
            ):
                raise RuntimeError(
                    "Thinned count exceeds full"
                )

            if np.any(
                thin[
                    :,
                    ~available
                ] != 0
            ):
                raise RuntimeError(
                    "Unavailable gene became "
                    "nonzero"
                )

            thin_totals = thin.sum(
                axis=1,
                dtype=np.uint64,
            )

            zero_library = (
                np.flatnonzero(
                    thin_totals == 0
                )
            )

            if zero_library.size:
                raise RuntimeError(
                    "ZERO_LIBRARY_STOP: "
                    f"dataset={dataset} "
                    f"q={q} "
                    f"rep={replicate} "
                    f"n={zero_library.size}"
                )

            x = (
                cp10k_log1p_counts(
                    thin,
                    available,
                    zero_library_rule=
                        "error",
                )
            )

            (
                lost_positive,
                originally_zero,
            ) = thinning_masks(
                full_counts,
                thin,
                available,
            )

            expected_lost = (
                (full_counts > 0)
                & (thin == 0)
                & available[None, :]
            )

            if not np.array_equal(
                lost_positive,
                expected_lost,
            ):
                raise RuntimeError(
                    "Lost-positive definition "
                    "mismatch"
                )

            if not np.array_equal(
                originally_zero,
                originally_zero_reference,
            ):
                raise RuntimeError(
                    "Originally-zero definition "
                    "mismatch"
                )

            if np.any(
                lost_positive
                & originally_zero
            ):
                raise RuntimeError(
                    "Positive/negative masks "
                    "overlap"
                )

            name = panel_name(
                dataset,
                q,
                replicate,
                seed,
            )

            panel_path = (
                temp_dir
                / name
            )

            np.savez(
                panel_path,

                x=x,

                y=full_target,

                synthetic_mask=
                    lost_positive,

                lost_positive_mask=
                    lost_positive,

                originally_zero_mask=
                    originally_zero,

                thinned_counts=
                    thin,

                available_gene_mask=
                    available,

                source_row_index=
                    np.arange(
                        5000,
                        dtype=np.int64,
                    ),

                dataset=
                    np.asarray(
                        dataset
                    ),

                split=
                    np.asarray(
                        source["split"]
                    ),

                modality=
                    np.asarray(
                        "sc"
                    ),

                thinning_q=
                    np.asarray(
                        q,
                        dtype=np.float32,
                    ),

                thinning_seed=
                    np.asarray(
                        seed,
                        dtype=np.int64,
                    ),

                thinning_replicate=
                    np.asarray(
                        replicate,
                        dtype=np.int64,
                    ),
            )

            thin_molecules = int(
                thin.sum(
                    dtype=np.uint64
                )
            )

            record = {
                "file":
                    name,

                "sha256":
                    sha256_file(
                        panel_path
                    ),

                "dataset":
                    dataset,

                "q":
                    q,

                "replicate":
                    replicate,

                "seed":
                    seed,

                "n_cells":
                    5000,

                "n_genes":
                    16384,

                "available_genes":
                    int(
                        available.sum()
                    ),

                "full_molecules":
                    full_molecules,

                "thinned_molecules":
                    thin_molecules,

                "molecule_retention_ratio":
                    (
                        thin_molecules
                        / full_molecules
                    ),

                "n_full_positive_entries":
                    n_full_positive,

                "n_lost_positive_entries":
                    int(
                        lost_positive.sum()
                    ),

                "lost_positive_fraction":
                    (
                        int(
                            lost_positive.sum()
                        )
                        / n_full_positive
                    ),

                "n_originally_zero_entries":
                    n_original_zero,

                "zero_library_cells":
                    0,

                "integer_counts_pass":
                    True,

                "count_bounds_pass":
                    True,

                "unavailable_genes_zero":
                    True,

                "lost_positive_definition_exact":
                    True,

                "originally_zero_definition_exact":
                    True,

                "normalization_recomputed_after_thinning":
                    True,

                "full_target_alignment_exact":
                    True,
            }

            records.append(
                record
            )

            print(
                json.dumps(
                    record,
                    sort_keys=True,
                ),
                flush=True,
            )

            del (
                thin,
                x,
                lost_positive,
                originally_zero,
            )

        expected_count = int(
            source[
                "expected_panel_count"
            ]
        )

        if len(records) != expected_count:
            raise RuntimeError(
                "Panel-count mismatch"
            )

        qc = {
            "schema":
                "sc2-p3-16k-count-thinning-confirmatory-dataset-qc-v1",

            "status":
                "PASS",

            "dataset":
                dataset,

            "n_cells":
                5000,

            "n_genes":
                16384,

            "available_genes":
                int(
                    available.sum()
                ),

            "panel_count":
                len(records),

            "q_values":
                [
                    0.85,
                    0.70,
                    0.50,
                ],

            "replicates_per_q":
                5,

            "all_zero_library_cells":
                0,

            "all_integer_counts_pass":
                True,

            "all_count_bounds_pass":
                True,

            "all_full_target_alignment_exact":
                True,

            "scientific_inference_performed":
                False,

            "threshold_retuning_performed":
                False,

            "panels":
                records,
        }

        write_json(
            temp_dir
            / "materialization_qc_summary.json",
            qc,
        )

        files = sorted(
            p
            for p in temp_dir.iterdir()
            if (
                p.is_file()
                and p.name
                != "SHA256SUMS.txt"
            )
        )

        (
            temp_dir
            / "SHA256SUMS.txt"
        ).write_text(
            "".join(
                f"{sha256_file(p)}  "
                f"{p.name}\n"
                for p in files
            ),
            encoding="utf-8",
        )

        for line in (
            temp_dir
            / "SHA256SUMS.txt"
        ).read_text().splitlines():

            digest, filename = (
                line.split(
                    None,
                    1,
                )
            )

            filename = filename.strip()

            if (
                sha256_file(
                    temp_dir
                    / filename
                )
                != digest
            ):
                raise RuntimeError(
                    "Manifest self-check failed"
                )

        os.replace(
            temp_dir,
            output_dir,
        )

    except Exception:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

        raise

    print(
        "P3_16K_CONFIRMATORY_THINNING_MATERIALIZATION=PASS"
    )

    print(
        f"DATASET={dataset}"
    )

    print(
        "PANELS=15"
    )

    print(
        f"OUTPUT={output_dir}"
    )


if __name__ == "__main__":
    main()
