#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from sc2.data.count_thinning import (
    binomial_thin_counts,
    cp10k_log1p_counts,
    thinning_masks,
)


PROTOCOL_ID = "sc2-count-thinning-confirmatory-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def q_label(q: float) -> str:
    return f"q{int(round(float(q) * 100)):03d}"


def canonical_identity_sha256(
    *,
    dataset: str,
    source: dict[str, Any],
    vocabulary_sha256: str,
) -> str:
    payload = {
        "dataset": dataset,
        "counts_file_sha256": source[
            "counts_file_sha256"
        ],
        "available_file_sha256": source[
            "available_file_sha256"
        ],
        "metadata_sha256": source[
            "metadata_sha256"
        ],
        "reference_target_sha256": source[
            "reference_target_sha256"
        ],
        "vocabulary_sha256": vocabulary_sha256,
        "n_cells": int(source["n_cells"]),
        "n_genes": int(source["n_genes"]),
        "n_available_genes": int(
            source["n_available_genes"]
        ),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def verify_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> None:
    if not path.is_file():
        raise RuntimeError(
            f"Missing {label}: {path}"
        )

    actual = sha256_file(path)

    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA mismatch: "
            f"expected={expected_sha256} "
            f"actual={actual} "
            f"path={path}"
        )


def materialize_dataset(
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    dataset: str,
    output_dir: Path,
) -> dict[str, Any]:

    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(
            "Unexpected protocol_id: "
            f"{protocol.get('protocol_id')}"
        )

    materialization = protocol[
        "materialization"
    ]

    sources = materialization[
        "datasets"
    ]

    if dataset not in sources:
        raise RuntimeError(
            f"Unknown dataset: {dataset}"
        )

    source = sources[dataset]

    expected_output = Path(
        source["output_dir"]
    ).resolve()

    if output_dir.resolve() != expected_output:
        raise RuntimeError(
            "Output path differs from frozen protocol: "
            f"expected={expected_output} "
            f"actual={output_dir.resolve()}"
        )

    if output_dir.exists():
        raise RuntimeError(
            f"Final output already exists: {output_dir}"
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
            f"Temporary output already exists: {temp_dir}"
        )

    counts_path = Path(
        source["counts_path"]
    )

    available_path = Path(
        source["available_path"]
    )

    metadata_path = Path(
        source["metadata_path"]
    )

    reference_target_path = Path(
        source["reference_target_path"]
    )

    verify_file(
        counts_path,
        source["counts_file_sha256"],
        "counts file",
    )

    verify_file(
        available_path,
        source["available_file_sha256"],
        "availability file",
    )

    verify_file(
        metadata_path,
        source["metadata_sha256"],
        "metadata file",
    )

    verify_file(
        reference_target_path,
        source["reference_target_sha256"],
        "reference target benchmark",
    )

    metadata = json.loads(
        metadata_path.read_text()
    )

    n_cells = int(
        source["n_cells"]
    )

    n_genes = int(
        source["n_genes"]
    )

    n_available_expected = int(
        source["n_available_genes"]
    )

    if metadata["dataset"] != dataset:
        raise RuntimeError(
            "Metadata dataset mismatch."
        )

    if list(metadata["shape"]) != [
        n_cells,
        n_genes,
    ]:
        raise RuntimeError(
            "Metadata shape mismatch."
        )

    if metadata["counts_dtype"] != "uint32":
        raise RuntimeError(
            "Metadata counts dtype mismatch."
        )

    metadata_checks = {
        "counts_file_sha256":
            source["counts_file_sha256"],

        "counts_uint32_sha256":
            source["counts_uint32_sha256"],

        "available_file_sha256":
            source["available_file_sha256"],

        "available_gene_mask_sha256":
            source["available_gene_mask_sha256"],

        "n_available_genes":
            n_available_expected,
    }

    for key, expected in metadata_checks.items():
        if metadata[key] != expected:
            raise RuntimeError(
                f"Metadata contract mismatch "
                f"for {key}: "
                f"{metadata[key]} != {expected}"
            )

    counts = np.load(
        counts_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    available = np.load(
        available_path,
        allow_pickle=False,
    )

    if counts.shape != (
        n_cells,
        n_genes,
    ):
        raise RuntimeError(
            f"Count shape mismatch: {counts.shape}"
        )

    if counts.dtype != np.uint32:
        raise RuntimeError(
            f"Count dtype mismatch: {counts.dtype}"
        )

    available = np.asarray(
        available,
        dtype=np.bool_,
    )

    if available.shape != (
        n_genes,
    ):
        raise RuntimeError(
            "Availability mask shape mismatch."
        )

    if int(available.sum()) != n_available_expected:
        raise RuntimeError(
            "Availability count mismatch."
        )

    full_counts = np.asarray(
        counts,
        dtype=np.uint32,
    )

    if np.any(
        full_counts[:, ~available] != 0
    ):
        raise RuntimeError(
            "Unavailable genes are nonzero "
            "in the frozen source counts."
        )

    full_row_totals = full_counts.sum(
        axis=1,
        dtype=np.uint64,
    )

    zero_full = np.flatnonzero(
        full_row_totals == 0
    )

    if zero_full.size:
        raise RuntimeError(
            "Full-depth source contains "
            "zero-library cells: "
            f"{zero_full[:20].tolist()} "
            f"n={zero_full.size}"
        )

    full_target = cp10k_log1p_counts(
        full_counts,
        available,
        zero_library_rule="error",
    )

    with np.load(
        reference_target_path,
        allow_pickle=False,
    ) as data:

        reference_y = np.asarray(
            data["y"],
            dtype=np.float32,
        )

    if not np.array_equal(
        full_target,
        reference_y,
    ):
        max_abs = float(
            np.max(
                np.abs(
                    full_target
                    - reference_y
                )
            )
        )

        raise RuntimeError(
            "Reconstructed full target does not "
            "exactly equal frozen reference y; "
            f"max_abs={max_abs}"
        )

    vocabulary_sha = str(
        protocol[
            "feature_space"
        ][
            "ordered_vocabulary_sha256"
        ]
    )

    source_identity_sha = (
        canonical_identity_sha256(
            dataset=dataset,
            source=source,
            vocabulary_sha256=vocabulary_sha,
        )
    )

    panel_specs = source[
        "panel_specs"
    ]

    expected_panels = int(
        source[
            "expected_panel_count"
        ]
    )

    if len(panel_specs) != expected_panels:
        raise RuntimeError(
            "Frozen panel-spec count mismatch."
        )

    seen = set()

    for item in panel_specs:
        key = (
            round(float(item["q"]), 12),
            int(item["replicate"]),
            int(item["seed"]),
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate panel specification: {key}"
            )

        seen.add(key)

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

    n_originally_zero = int(
        originally_zero_reference.sum()
    )

    protocol_sha = sha256_file(
        protocol_path
    )

    temp_dir.mkdir(
        parents=False,
        exist_ok=False,
    )

    try:
        np.save(
            temp_dir
            / "available_gene_mask.npy",
            available,
        )

        source_identity = {
            "dataset": dataset,
            "split": source["split"],
            "modality": "sc",

            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_available_genes":
                n_available_expected,

            "source_identity_sha256":
                source_identity_sha,

            "counts_path":
                str(counts_path),

            "counts_file_sha256":
                source["counts_file_sha256"],

            "counts_uint32_sha256":
                source["counts_uint32_sha256"],

            "available_path":
                str(available_path),

            "available_file_sha256":
                source["available_file_sha256"],

            "available_gene_mask_sha256":
                source[
                    "available_gene_mask_sha256"
                ],

            "metadata_path":
                str(metadata_path),

            "metadata_sha256":
                source["metadata_sha256"],

            "reference_target_path":
                str(reference_target_path),

            "reference_target_sha256":
                source[
                    "reference_target_sha256"
                ],

            "full_target_exactly_matches_reference_y":
                True,

            "full_depth_total_molecules":
                int(
                    full_counts.sum(
                        dtype=np.uint64
                    )
                ),

            "full_depth_zero_library_cells":
                0,

            "ordered_vocabulary_path":
                protocol[
                    "feature_space"
                ][
                    "ordered_vocabulary_path"
                ],

            "ordered_vocabulary_sha256":
                vocabulary_sha,

            "count_view_receipt_sha256":
                protocol[
                    "provenance"
                ][
                    "count_view_receipt_sha256"
                ],

            "phase1b_threshold_config_sha256":
                protocol[
                    "provenance"
                ][
                    "threshold_config_sha256"
                ],

            "protocol_sha256":
                protocol_sha,
        }

        write_json(
            temp_dir
            / "source_identity.json",
            source_identity,
        )

        panel_records = []

        for specification in panel_specs:

            q = float(
                specification["q"]
            )

            replicate = int(
                specification[
                    "replicate"
                ]
            )

            seed = int(
                specification["seed"]
            )

            thin = binomial_thin_counts(
                full_counts,
                q=q,
                seed=seed,
                available_gene_mask=available,
            )

            if not np.issubdtype(
                thin.dtype,
                np.integer,
            ):
                raise RuntimeError(
                    "Thinned counts are not integer."
                )

            if np.any(
                thin > full_counts
            ):
                raise RuntimeError(
                    "At least one thinned count "
                    "exceeds full count."
                )

            if np.any(
                thin[:, ~available] != 0
            ):
                raise RuntimeError(
                    "Unavailable gene became nonzero."
                )

            thin_row_totals = thin.sum(
                axis=1,
                dtype=np.uint64,
            )

            zero_library = np.flatnonzero(
                thin_row_totals == 0
            )

            if zero_library.size:
                raise RuntimeError(
                    "STOP: zero-library cells after "
                    f"thinning dataset={dataset} "
                    f"q={q} "
                    f"replicate={replicate} "
                    f"seed={seed}; "
                    f"cells={zero_library[:20].tolist()} "
                    f"n={zero_library.size}"
                )

            normalized_input = (
                cp10k_log1p_counts(
                    thin,
                    available,
                    zero_library_rule="error",
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

            expected_zero = (
                (full_counts == 0)
                & available[None, :]
            )

            if not np.array_equal(
                lost_positive,
                expected_lost,
            ):
                raise RuntimeError(
                    "Lost-positive mask is not exact."
                )

            if not np.array_equal(
                originally_zero,
                expected_zero,
            ):
                raise RuntimeError(
                    "Originally-zero mask is not exact."
                )

            if np.any(
                lost_positive
                & originally_zero
            ):
                raise RuntimeError(
                    "Lost-positive and originally-zero "
                    "masks overlap."
                )

            qcode = q_label(q)

            filename = (
                f"{dataset}_"
                f"{qcode}_"
                f"rep{replicate:02d}_"
                f"seed{seed}.npz"
            )

            panel_path = (
                temp_dir
                / filename
            )

            np.savez(
                panel_path,

                x=normalized_input,
                y=full_target,

                synthetic_mask=
                    lost_positive,

                lost_positive_mask=
                    lost_positive,

                originally_zero_mask=
                    originally_zero,

                thinned_counts=thin,

                available_gene_mask=
                    available,

                source_row_index=
                    np.arange(
                        n_cells,
                        dtype=np.int64,
                    ),

                dataset=np.asarray(
                    dataset
                ),

                split=np.asarray(
                    source["split"]
                ),

                modality=np.asarray(
                    "sc"
                ),

                thinning_q=np.asarray(
                    q,
                    dtype=np.float32,
                ),

                thinning_seed=np.asarray(
                    seed,
                    dtype=np.int64,
                ),

                thinning_replicate=np.asarray(
                    replicate,
                    dtype=np.int64,
                ),

                source_identity_sha256=
                    np.asarray(
                        source_identity_sha
                    ),

                source_counts_file_sha256=
                    np.asarray(
                        source[
                            "counts_file_sha256"
                        ]
                    ),

                source_available_file_sha256=
                    np.asarray(
                        source[
                            "available_file_sha256"
                        ]
                    ),

                source_metadata_sha256=
                    np.asarray(
                        source[
                            "metadata_sha256"
                        ]
                    ),

                protocol_sha256=
                    np.asarray(
                        protocol_sha
                    ),
            )

            n_lost = int(
                lost_positive.sum()
            )

            thin_molecules = int(
                thin.sum(
                    dtype=np.uint64
                )
            )

            full_molecules = int(
                full_counts.sum(
                    dtype=np.uint64
                )
            )

            panel_records.append(
                {
                    "dataset": dataset,
                    "q": q,
                    "replicate": replicate,
                    "seed": seed,

                    "file": filename,

                    "sha256":
                        sha256_file(
                            panel_path
                        ),

                    "n_cells": n_cells,
                    "n_genes": n_genes,

                    "n_available_genes":
                        n_available_expected,

                    "n_full_positive":
                        n_full_positive,

                    "n_lost_positive":
                        n_lost,

                    "lost_positive_fraction":
                        (
                            n_lost
                            / n_full_positive
                        ),

                    "n_originally_zero":
                        n_originally_zero,

                    "full_molecules":
                        full_molecules,

                    "thinned_molecules":
                        thin_molecules,

                    "realized_molecule_retention":
                        (
                            thin_molecules
                            / full_molecules
                        ),

                    "zero_library_cells":
                        0,

                    "counts_integer":
                        True,

                    "counts_never_increase":
                        True,

                    "unavailable_genes_zero":
                        True,

                    "lost_positive_definition_exact":
                        True,

                    "originally_zero_definition_exact":
                        True,

                    "normalization_recomputed_from_thinned_counts":
                        True,

                    "full_target_exactly_matches_reference_y":
                        True,

                    "source_identity_sha256":
                        source_identity_sha,
                }
            )

        if len(panel_records) != expected_panels:
            raise RuntimeError(
                "Materialized panel count mismatch."
            )

        qc_summary = {
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_sha,

            "dataset": dataset,

            "scope":
                "confirmatory_count_thinning",

            "n_cells": n_cells,
            "n_genes": n_genes,

            "n_available_genes":
                n_available_expected,

            "available_gene_fraction":
                (
                    n_available_expected
                    / n_genes
                ),

            "source_identity_sha256":
                source_identity_sha,

            "full_depth_total_molecules":
                int(
                    full_counts.sum(
                        dtype=np.uint64
                    )
                ),

            "full_depth_zero_library_cells":
                0,

            "full_target_exactly_matches_reference_y":
                True,

            "expected_panel_count":
                expected_panels,

            "materialized_panel_count":
                len(panel_records),

            "zero_library_rule":
                "STOP",

            "panels":
                panel_records,
        }

        qc_path = (
            temp_dir
            / "materialization_qc_summary.json"
        )

        write_json(
            qc_path,
            qc_summary,
        )

        manifest_files = sorted(
            [
                p
                for p in temp_dir.iterdir()
                if (
                    p.is_file()
                    and p.name
                    != "SHA256SUMS.txt"
                )
            ],
            key=lambda p: p.name,
        )

        manifest_path = (
            temp_dir
            / "SHA256SUMS.txt"
        )

        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  "
                f"{path.name}\n"
                for path in manifest_files
            )
        )

        # Verify our own manifest before atomic finalization.
        for line in manifest_path.read_text().splitlines():

            digest, filename = line.split(
                None,
                1,
            )

            filename = filename.strip()

            if sha256_file(
                temp_dir
                / filename
            ) != digest:
                raise RuntimeError(
                    "Internal SHA manifest "
                    "verification failed."
                )

        if output_dir.exists():
            raise RuntimeError(
                "Final output appeared before "
                "atomic finalization."
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

    result = {
        "dataset": dataset,
        "output_dir": str(
            output_dir
        ),
        "panel_count": expected_panels,
        "source_identity_sha256":
            source_identity_sha,
        "protocol_sha256":
            protocol_sha,
        "qc_summary_sha256":
            sha256_file(
                output_dir
                / "materialization_qc_summary.json"
            ),
        "manifest_sha256":
            sha256_file(
                output_dir
                / "SHA256SUMS.txt"
            ),
    }

    print(
        "CONFIRMATORY_COUNT_THINNING_MATERIALIZATION=PASS"
    )

    print(
        f"DATASET={dataset}"
    )

    print(
        f"PANELS={expected_panels}"
    )

    print(
        f"OUTPUT={output_dir}"
    )

    print(
        "SOURCE_IDENTITY_SHA256="
        f"{source_identity_sha}"
    )

    print(
        "QC_SUMMARY_SHA256="
        f"{result['qc_summary_sha256']}"
    )

    print(
        "MANIFEST_SHA256="
        f"{result['manifest_sha256']}"
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one frozen confirmatory "
            "raw-count thinning dataset."
        )
    )

    parser.add_argument(
        "--protocol",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    if not protocol_path.is_file():
        raise RuntimeError(
            f"Protocol missing: {protocol_path}"
        )

    protocol = json.loads(
        protocol_path.read_text()
    )

    materialize_dataset(
        protocol=protocol,
        protocol_path=protocol_path,
        dataset=str(args.dataset),
        output_dir=Path(
            args.output_dir
        ).resolve(),
    )


if __name__ == "__main__":
    main()
