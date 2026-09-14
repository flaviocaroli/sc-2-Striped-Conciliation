#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


MASKING_ALGORITHM = (
    "Per panel initialize numpy.random.PCG64(seed); "
    "for each cell generate one uniform random priority for "
    "each positive gene; select the k smallest priorities, "
    "where k=max(1,rint(mask_rate*n_positive))."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as fh:
        for block in iter(
            lambda: fh.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def dense_log1p_rows(
    shard: Path,
    rows: np.ndarray,
    n_genes: int,
) -> np.ndarray:

    values = np.load(
        shard / "log1p.data.npy",
        mmap_mode="r",
    )
    indices = np.load(
        shard / "log1p.indices.npy",
        mmap_mode="r",
    )
    indptr = np.load(
        shard / "log1p.indptr.npy",
        mmap_mode="r",
    )

    shape = tuple(
        json.loads(
            (
                shard
                / "log1p.shape.json"
            ).read_text()
        )
    )

    if shape != (25000, n_genes):
        raise RuntimeError(
            f"Unexpected validation shard shape: {shape}"
        )

    out = np.zeros(
        (len(rows), n_genes),
        dtype=np.float32,
    )

    for out_row, source_row in enumerate(rows):
        source_row = int(source_row)

        if source_row < 0 or source_row >= shape[0]:
            raise RuntimeError(
                f"Validation row out of bounds: {source_row}"
            )

        start = int(
            indptr[source_row]
        )
        end = int(
            indptr[source_row + 1]
        )

        cols = np.asarray(
            indices[start:end],
            dtype=np.int64,
        )
        vals = np.asarray(
            values[start:end],
            dtype=np.float32,
        )

        out[out_row, cols] = vals

    return out


def make_mask(
    y: np.ndarray,
    rate: float,
    seed: int,
) -> np.ndarray:

    rng = np.random.Generator(
        np.random.PCG64(seed)
    )

    mask = np.zeros(
        y.shape,
        dtype=bool,
    )

    n_positive = (
        y > 0
    ).sum(axis=1)

    n_mask = np.maximum(
        1,
        np.rint(
            rate * n_positive
        ).astype(np.int64),
    )

    for row in range(y.shape[0]):
        positive = np.flatnonzero(
            y[row] > 0
        )

        k = int(
            n_mask[row]
        )

        if k < 1 or k > positive.size:
            raise RuntimeError(
                f"Invalid mask count at row {row}: "
                f"k={k}, positives={positive.size}"
            )

        priorities = rng.random(
            positive.size
        )

        if k == positive.size:
            chosen = positive
        else:
            chosen_local = np.argpartition(
                priorities,
                k - 1,
            )[:k]

            chosen = positive[
                chosen_local
            ]

        mask[row, chosen] = True

    return mask


def materialize(args):
    protocol_path = Path(
        args.protocol
    )
    identity_path = Path(
        args.identity
    )
    output_root = Path(
        args.output_root
    )

    if sha256(protocol_path) != args.protocol_sha:
        raise RuntimeError(
            "Protocol SHA mismatch"
        )

    if sha256(identity_path) != args.identity_sha:
        raise RuntimeError(
            "Identity SHA mismatch"
        )

    protocol = json.loads(
        protocol_path.read_text()
    )

    manifest_path = Path(
        protocol[
            "corpus"
        ][
            "manifest_path"
        ]
    )

    if sha256(manifest_path) != args.manifest_sha:
        raise RuntimeError(
            "Manifest SHA mismatch"
        )

    manifest = pd.read_parquet(
        manifest_path
    )

    identity = pd.read_csv(
        identity_path,
        sep="\t",
    )

    expected_columns = {
        "shard_id",
        "row",
        "cell_id",
        "soma_joinid",
    }

    if set(identity.columns) != expected_columns:
        raise RuntimeError(
            f"Unexpected identity columns: "
            f"{list(identity.columns)}"
        )

    if len(identity) != 2500:
        raise RuntimeError(
            f"Expected 2500 validation cells, "
            f"got {len(identity)}"
        )

    if identity["cell_id"].nunique() != 2500:
        raise RuntimeError(
            "Duplicate validation cell_id"
        )

    if identity["soma_joinid"].nunique() != 2500:
        raise RuntimeError(
            "Duplicate validation soma_joinid"
        )

    if args.limit_cells is not None:
        if (
            args.limit_cells < 1
            or args.limit_cells > 2500
        ):
            raise RuntimeError(
                "Invalid --limit-cells"
            )

        identity = identity.iloc[
            :args.limit_cells
        ].copy()

    shard_ids = identity[
        "shard_id"
    ].astype(str).to_numpy()

    if set(
        shard_ids.tolist()
    ) != {
        "sc_validation_00000"
    }:
        raise RuntimeError(
            "Identity contains non-validation shard"
        )

    validation_manifest = manifest.loc[
        manifest[
            "shard_id"
        ].astype(str)
        == "sc_validation_00000"
    ]

    if len(
        validation_manifest
    ) != 1:
        raise RuntimeError(
            "Validation shard is not unique"
        )

    shard = Path(
        validation_manifest.iloc[
            0
        ][
            "path"
        ]
    )

    if not shard.is_dir():
        raise RuntimeError(
            f"Missing validation shard: {shard}"
        )

    rows = identity[
        "row"
    ].to_numpy(
        dtype=np.int64
    )

    if len(
        np.unique(rows)
    ) != len(rows):
        raise RuntimeError(
            "Duplicate validation row"
        )

    y = dense_log1p_rows(
        shard=shard,
        rows=rows,
        n_genes=16384,
    )

    if y.shape != (
        len(identity),
        16384,
    ):
        raise RuntimeError(
            f"Unexpected target shape: {y.shape}"
        )

    if not np.all(
        np.isfinite(y)
    ):
        raise RuntimeError(
            "Non-finite target values"
        )

    if np.any(
        y < 0
    ):
        raise RuntimeError(
            "Negative normalized values"
        )

    rates = protocol[
        "normalized_masking"
    ][
        "mask_rates"
    ]

    seeds = protocol[
        "normalized_masking"
    ][
        "panel_seeds"
    ]

    tmp_root = output_root.with_name(
        output_root.name + ".tmp"
    )

    if output_root.exists():
        raise RuntimeError(
            f"Output already exists: {output_root}"
        )

    if tmp_root.exists():
        raise RuntimeError(
            f"Temporary output already exists: "
            f"{tmp_root}"
        )

    tmp_root.mkdir(
        parents=True
    )

    records = []

    try:
        for rate_value in rates:
            rate = float(
                rate_value
            )

            pct = int(
                round(
                    100.0 * rate
                )
            )

            if str(pct) not in seeds:
                raise RuntimeError(
                    f"Missing seed for {pct}%"
                )

            seed = int(
                seeds[
                    str(pct)
                ]
            )

            synthetic_mask = make_mask(
                y=y,
                rate=rate,
                seed=seed,
            )

            if np.any(
                synthetic_mask
                & ~(y > 0)
            ):
                raise RuntimeError(
                    f"{pct}% mask contains "
                    f"non-positive targets"
                )

            expected_counts = np.maximum(
                1,
                np.rint(
                    rate
                    * (y > 0).sum(axis=1)
                ).astype(np.int64),
            )

            actual_counts = (
                synthetic_mask.sum(
                    axis=1
                )
            )

            if not np.array_equal(
                expected_counts,
                actual_counts,
            ):
                raise RuntimeError(
                    f"{pct}% mask count mismatch"
                )

            x = y.copy()

            x[
                synthetic_mask
            ] = 0.0

            if not np.all(
                x[
                    synthetic_mask
                ] == 0
            ):
                raise RuntimeError(
                    f"{pct}% masked values "
                    f"are not zero"
                )

            if not np.array_equal(
                x[
                    ~synthetic_mask
                ],
                y[
                    ~synthetic_mask
                ],
            ):
                raise RuntimeError(
                    f"{pct}% unmasked values changed"
                )

            path = (
                tmp_root
                / f"validation_mask{pct}.npz"
            )

            np.savez_compressed(
                path,
                x=x.astype(
                    np.float32,
                    copy=False,
                ),
                y=y.astype(
                    np.float32,
                    copy=False,
                ),
                synthetic_mask=synthetic_mask,
                row=rows,
                shard_id=shard_ids.astype(
                    "U"
                ),
                split=np.asarray(
                    "validation"
                ),
                modality=np.asarray(
                    "sc"
                ),
                mask_rate=np.asarray(
                    rate,
                    dtype=np.float32,
                ),
                seed=np.asarray(
                    seed,
                    dtype=np.int64,
                ),
                manifest_sha256=np.asarray(
                    args.manifest_sha
                ),
                identity_sha256=np.asarray(
                    args.identity_sha
                ),
                protocol_sha256=np.asarray(
                    args.protocol_sha
                ),
            )

            with np.load(
                path,
                allow_pickle=False,
            ) as saved:
                if saved["x"].shape != (
                    len(identity),
                    16384,
                ):
                    raise RuntimeError(
                        "Saved x shape mismatch"
                    )

                if saved["y"].shape != (
                    len(identity),
                    16384,
                ):
                    raise RuntimeError(
                        "Saved y shape mismatch"
                    )

                if saved[
                    "synthetic_mask"
                ].shape != (
                    len(identity),
                    16384,
                ):
                    raise RuntimeError(
                        "Saved mask shape mismatch"
                    )

            record = {
                "file": path.name,
                "sha256": sha256(
                    path
                ),
                "mask_percent": pct,
                "mask_rate": rate,
                "seed": seed,
                "n_cells": len(
                    identity
                ),
                "n_genes": 16384,
                "n_positive_target_entries": int(
                    (y > 0).sum()
                ),
                "n_masked": int(
                    synthetic_mask.sum()
                ),
            }

            records.append(
                record
            )

            print(
                f"wrote={path.name} "
                f"cells={len(identity)} "
                f"genes=16384 "
                f"masked={record['n_masked']}"
            )

        manifest_out = {
            "status": "PASS",
            "phase": "P3-4",
            "cells": len(
                identity
            ),
            "genes": 16384,
            "masking_algorithm": MASKING_ALGORITHM,
            "rounding": (
                "numpy.rint nearest-even "
                "using nominal protocol rates"
            ),
            "numpy_version": np.__version__,
            "protocol_sha256": args.protocol_sha,
            "identity_sha256": args.identity_sha,
            "manifest_sha256": args.manifest_sha,
            "panels": records,
            "internal_test_used": False,
            "baron_used": False,
            "zheng68k_used": False,
        }

        (
            tmp_root
            / "validation_panels_manifest.json"
        ).write_text(
            json.dumps(
                manifest_out,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        tmp_root.rename(
            output_root
        )

    except Exception:
        shutil.rmtree(
            tmp_root,
            ignore_errors=True,
        )
        raise

    print(
        "P3_16K_VALIDATION_MATERIALIZATION=PASS"
    )
    print(
        f"OUTPUT_ROOT={output_root}"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--protocol",
        required=True,
    )
    p.add_argument(
        "--identity",
        required=True,
    )
    p.add_argument(
        "--output-root",
        required=True,
    )
    p.add_argument(
        "--protocol-sha",
        required=True,
    )
    p.add_argument(
        "--identity-sha",
        required=True,
    )
    p.add_argument(
        "--manifest-sha",
        required=True,
    )
    p.add_argument(
        "--limit-cells",
        type=int,
        default=None,
    )

    args = p.parse_args()

    materialize(
        args
    )


if __name__ == "__main__":
    main()
