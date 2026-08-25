#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sc2.data.count_thinning import (
    binomial_thin_counts,
    cp10k_log1p_counts,
    identity_sha256,
    thinning_masks,
)
from sc2.data.csr_shard import CSRMemmap
from sc2.data.shard_manifest import load_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def write_json(
    path: Path,
    payload: object,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_validation_identity(
    protocol: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    source = protocol["sources"]["validation_masking_panels"]

    reference_shard: np.ndarray | None = None
    reference_row: np.ndarray | None = None
    reference_target: np.ndarray | None = None

    for item in source:
        path = Path(item["path"])

        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(
                f"Source panel SHA mismatch: {path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as panel:
            split = str(panel["split"].item())

            if split != "validation":
                raise RuntimeError(
                    f"{path}: split={split}, expected validation"
                )

            shard = np.asarray(
                panel["shard_id"]
            ).astype(str)
            row = np.asarray(
                panel["row"],
                dtype=np.int64,
            )
            target = np.asarray(
                panel["y"],
                dtype=np.float32,
            )

            if target.shape != (
                int(protocol["validation"]["n_cells"]),
                int(protocol["feature_space"]["n_genes"]),
            ):
                raise RuntimeError(
                    f"{path}: target shape mismatch {target.shape}"
                )

            if reference_shard is None:
                reference_shard = shard.copy()
                reference_row = row.copy()
                reference_target = target.copy()
            else:
                if not np.array_equal(
                    shard,
                    reference_shard,
                ):
                    raise RuntimeError(
                        f"{path}: validation shard identities differ"
                    )

                if not np.array_equal(
                    row,
                    reference_row,
                ):
                    raise RuntimeError(
                        f"{path}: validation row identities differ"
                    )

                if not np.array_equal(
                    target,
                    reference_target,
                ):
                    raise RuntimeError(
                        f"{path}: full-depth targets differ"
                    )

    if (
        reference_shard is None
        or reference_row is None
        or reference_target is None
    ):
        raise RuntimeError(
            "No frozen validation source panels"
        )

    observed_identity = identity_sha256(
        reference_shard,
        reference_row,
    )

    expected_identity = str(
        protocol["validation"]["identity_sha256"]
    )

    if observed_identity != expected_identity:
        raise RuntimeError(
            "Validation identity SHA mismatch: "
            f"{observed_identity} != {expected_identity}"
        )

    return (
        reference_shard,
        reference_row,
        reference_target,
    )


def load_raw_counts(
    *,
    manifest_path: Path,
    expected_manifest_sha: str,
    shard_id: np.ndarray,
    row: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    if sha256_file(
        manifest_path
    ) != expected_manifest_sha:
        raise RuntimeError(
            "Raw manifest file SHA mismatch"
        )

    records, manifest_hash = load_manifest(
        manifest_path
    )

    if str(manifest_hash) != expected_manifest_sha:
        raise RuntimeError(
            "Manifest loader SHA does not equal frozen file SHA: "
            f"{manifest_hash} != {expected_manifest_sha}"
        )

    by_id = {
        str(record.shard_id): record
        for record in records
    }

    counts = np.empty(
        (
            len(row),
            n_genes,
        ),
        dtype=np.uint32,
    )

    opened: dict[str, CSRMemmap] = {}

    for position, (
        shard_name,
        source_row,
    ) in enumerate(
        zip(
            shard_id.tolist(),
            row.tolist(),
            strict=True,
        )
    ):
        shard_name = str(shard_name)

        if shard_name not in by_id:
            raise RuntimeError(
                f"Unknown shard id: {shard_name}"
            )

        record = by_id[shard_name]

        if str(record.split) != "validation":
            raise RuntimeError(
                f"{shard_name} is not a validation shard"
            )

        if str(record.modality) != "sc":
            raise RuntimeError(
                f"{shard_name} has unexpected modality"
            )

        if shard_name not in opened:
            opened[shard_name] = CSRMemmap.open(
                record.path,
                "counts",
            )

        matrix = opened[shard_name]

        if matrix.shape[1] != n_genes:
            raise RuntimeError(
                f"{shard_name}: gene count mismatch"
            )

        counts[position] = matrix.dense_row(
            int(source_row),
            dtype=np.uint32,
        )

    return counts


def panel_name(
    *,
    q: float,
    replicate: int,
    seed: int,
) -> str:
    q_label = f"{int(round(q * 100)):03d}"

    return (
        f"validation_q{q_label}_"
        f"rep{replicate:02d}_"
        f"seed{seed}.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize validation-only raw-count "
            "binomial thinning panels."
        )
    )
    parser.add_argument(
        "--protocol",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise SystemExit(
                f"Output directory is not empty: {output_dir}"
            )
    else:
        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    protocol = json.loads(
        protocol_path.read_text(
            encoding="utf-8"
        )
    )

    if protocol["protocol_id"] != "sc2-count-thinning-v1":
        raise RuntimeError(
            "Unexpected protocol_id"
        )

    if protocol["scope"]["materialize_now"] != [
        "internal_validation"
    ]:
        raise RuntimeError(
            "Protocol does not restrict this phase to validation"
        )

    n_cells = int(
        protocol["validation"]["n_cells"]
    )
    n_genes = int(
        protocol["feature_space"]["n_genes"]
    )

    shard_id, row, frozen_target = (
        load_validation_identity(
            protocol
        )
    )

    if len(row) != n_cells:
        raise RuntimeError(
            f"Validation identity has {len(row)} rows; expected {n_cells}"
        )

    manifest_path = Path(
        protocol["sources"]["corpus_manifest"]["path"]
    )
    manifest_sha = str(
        protocol["sources"]["corpus_manifest"]["sha256"]
    )

    full_counts = load_raw_counts(
        manifest_path=manifest_path,
        expected_manifest_sha=manifest_sha,
        shard_id=shard_id,
        row=row,
        n_genes=n_genes,
    )

    if full_counts.shape != (
        n_cells,
        n_genes,
    ):
        raise RuntimeError(
            f"Raw validation count shape mismatch: {full_counts.shape}"
        )

    available = np.ones(
        n_genes,
        dtype=np.bool_,
    )

    # Recompute normalization FROM RAW COUNTS.
    full_target = cp10k_log1p_counts(
        full_counts,
        available,
        zero_library_rule="error",
    )

    # This is the central raw-count alignment gate.
    if not np.array_equal(
        full_target,
        frozen_target,
    ):
        max_abs = float(
            np.max(
                np.abs(
                    full_target.astype(
                        np.float64
                    )
                    - frozen_target.astype(
                        np.float64
                    )
                )
            )
        )
        raise RuntimeError(
            "Raw-count normalization does not exactly reproduce "
            f"the frozen validation target; max_abs={max_abs}"
        )

    if np.any(
        full_counts.sum(
            axis=1,
            dtype=np.uint64,
        )
        == 0
    ):
        raise RuntimeError(
            "Full validation counts contain zero-library cells"
        )

    np.save(
        output_dir
        / "validation_full_counts_uint32.npy",
        full_counts,
    )
    np.save(
        output_dir
        / "available_gene_mask.npy",
        available,
    )
    np.save(
        output_dir
        / "shard_id.npy",
        shard_id,
    )
    np.save(
        output_dir
        / "row.npy",
        row,
    )

    source_identity = {
        "identity_sha256": identity_sha256(
            shard_id,
            row,
        ),
        "n_cells": n_cells,
        "n_genes": n_genes,
        "split": "validation",
        "modality": "sc",
        "manifest_path": str(
            manifest_path
        ),
        "manifest_sha256": manifest_sha,
        "full_target_exactly_matches_frozen_validation_y": True,
        "full_counts_total_molecules": int(
            full_counts.sum(
                dtype=np.uint64
            )
        ),
        "zero_library_cells_full": 0,
        "available_genes": int(
            available.sum()
        ),
    }

    write_json(
        output_dir
        / "source_identity.json",
        source_identity,
    )

    panel_records: list[
        dict[str, Any]
    ] = []

    full_positive = (
        (full_counts > 0)
        & available[None, :]
    )
    n_full_positive = int(
        full_positive.sum()
    )

    originally_zero_reference = (
        (full_counts == 0)
        & available[None, :]
    )
    n_originally_zero = int(
        originally_zero_reference.sum()
    )

    full_row_totals = full_counts.sum(
        axis=1,
        dtype=np.uint64,
    )

    for specification in protocol["thinning"]["panel_specs"]:
        q = float(
            specification["q"]
        )
        replicate = int(
            specification["replicate"]
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
                "Thinned counts are not integer"
            )

        if np.any(
            thin > full_counts
        ):
            raise RuntimeError(
                "At least one thinned count exceeds full count"
            )

        if np.any(
            thin[:, ~available] != 0
        ):
            raise RuntimeError(
                "Unavailable gene became nonzero"
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
                "Predeclared zero-library STOP rule triggered "
                f"for q={q}, replicate={replicate}, "
                f"seed={seed}, cells={zero_library[:20].tolist()}"
            )

        normalized_input = cp10k_log1p_counts(
            thin,
            available,
            zero_library_rule="error",
        )

        lost_positive, originally_zero = (
            thinning_masks(
                full_counts,
                thin,
                available,
            )
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
                "lost-positive mask is not exact"
            )

        if not np.array_equal(
            originally_zero,
            originally_zero_reference,
        ):
            raise RuntimeError(
                "originally-zero mask changed across panels"
            )

        if np.any(
            lost_positive
            & originally_zero
        ):
            raise RuntimeError(
                "Positive and negative gate classes overlap"
            )

        if not np.all(
            full_target[
                lost_positive
            ] > 0
        ):
            raise RuntimeError(
                "Lost-positive target includes non-positive value"
            )

        filename = panel_name(
            q=q,
            replicate=replicate,
            seed=seed,
        )
        path = (
            output_dir
            / filename
        )

        # Compatibility with the existing evaluator:
        #   x              = thinned normalized observation
        #   y              = full-depth normalized target
        #   synthetic_mask = depth-induced lost positives
        #
        # The extra arrays make the thinning provenance explicit.
        np.savez(
            path,
            x=normalized_input,
            y=full_target,
            synthetic_mask=lost_positive,
            lost_positive_mask=lost_positive,
            originally_zero_mask=originally_zero,
            thinned_counts=thin,
            available_gene_mask=available,
            shard_id=shard_id,
            row=row,
            split=np.asarray(
                "validation"
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
            manifest_sha256=np.asarray(
                manifest_sha
            ),
            identity_sha256=np.asarray(
                source_identity[
                    "identity_sha256"
                ]
            ),
        )

        thinned_molecules = int(
            thin_row_totals.sum(
                dtype=np.uint64
            )
        )
        full_molecules = int(
            full_row_totals.sum(
                dtype=np.uint64
            )
        )

        ratios = (
            thin_row_totals.astype(
                np.float64
            )
            / full_row_totals.astype(
                np.float64
            )
        )

        record = {
            "file": filename,
            "sha256": sha256_file(
                path
            ),
            "q": q,
            "nominal_count_loss": (
                1.0 - q
            ),
            "replicate": replicate,
            "seed": seed,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "available_genes": int(
                available.sum()
            ),
            "full_molecules": (
                full_molecules
            ),
            "thinned_molecules": (
                thinned_molecules
            ),
            "molecule_retention_ratio": (
                thinned_molecules
                / full_molecules
            ),
            "mean_cell_retention_ratio": float(
                ratios.mean()
            ),
            "min_cell_retention_ratio": float(
                ratios.min()
            ),
            "max_cell_retention_ratio": float(
                ratios.max()
            ),
            "n_full_positive_entries": (
                n_full_positive
            ),
            "n_lost_positive_entries": int(
                lost_positive.sum()
            ),
            "lost_positive_fraction_of_full_positives": (
                int(
                    lost_positive.sum()
                )
                / max(
                    1,
                    n_full_positive,
                )
            ),
            "n_originally_zero_entries": (
                n_originally_zero
            ),
            "zero_library_cells": int(
                zero_library.size
            ),
            "count_bounds_pass": True,
            "integer_counts_pass": True,
            "lost_positive_definition_pass": True,
            "originally_zero_definition_pass": True,
            "normalization_recomputed_after_thinning": True,
            "full_target_alignment_exact": True,
        }

        panel_records.append(
            record
        )

        print(
            json.dumps(
                record,
                sort_keys=True,
            ),
            flush=True,
        )

    expected_panel_count = int(
        protocol["thinning"]["expected_validation_panel_count"]
    )

    if len(panel_records) != expected_panel_count:
        raise RuntimeError(
            "Panel count mismatch: "
            f"{len(panel_records)} != {expected_panel_count}"
        )

    qc = {
        "protocol_id": protocol[
            "protocol_id"
        ],
        "scope": "internal_validation_only",
        "source_identity": (
            source_identity
        ),
        "panel_count": len(
            panel_records
        ),
        "panels": panel_records,
        "all_count_bounds_pass": all(
            item["count_bounds_pass"]
            for item in panel_records
        ),
        "all_integer_counts_pass": all(
            item["integer_counts_pass"]
            for item in panel_records
        ),
        "all_zero_library_cells": int(
            sum(
                item[
                    "zero_library_cells"
                ]
                for item
                in panel_records
            )
        ),
        "all_full_target_alignment_exact": all(
            item[
                "full_target_alignment_exact"
            ]
            for item
            in panel_records
        ),
        "model_outcomes_inspected": False,
        "test_or_external_materialized": False,
    }

    write_json(
        output_dir
        / "validation_qc_summary.json",
        qc,
    )

    files = sorted(
        path
        for path in output_dir.iterdir()
        if (
            path.is_file()
            and path.name
            != "SHA256SUMS.txt"
        )
    )

    manifest_lines = [
        (
            f"{sha256_file(path)}  "
            f"{path.name}"
        )
        for path in files
    ]

    (
        output_dir
        / "SHA256SUMS.txt"
    ).write_text(
        "\n".join(
            manifest_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "COUNT_THINNING_VALIDATION_MATERIALIZATION=PASS"
    )
    print(
        f"OUTPUT={output_dir}"
    )


if __name__ == "__main__":
    main()
