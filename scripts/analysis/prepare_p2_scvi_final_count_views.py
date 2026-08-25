#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from sc2.data.csr_shard import CSRMemmap

from scripts.data.materialize_census_shards import (
    cp10k_log1p as census_cp10k_log1p,
)

from scripts.data.prepare_external_repair_benchmarks import (
    cp10k_log1p as external_cp10k_log1p,
    load_baron,
    load_vocabulary,
    load_zheng,
)


N_CELLS = 5000
N_GENES = 4096
SEED = 20260729
MASKS = (15, 30, 50)

VOCAB_SHA = (
    "220093512a48251f9b4f2d45e8bf9134725473ce5bab89c93e2440bea1f74d01"
)

PANEL_HASHES = {
    "internal_test": {
        15:
            "eb9d3d63f42bc30fb5233f0f39271f792b43535a6be4f46b690cc188ed99c3d7",
        30:
            "ca59d9407830484360c476865c1b339b0ed62b8c4e0a75e8ca5f07acbc1d7ecf",
        50:
            "e1a75d8b27eda99ccf0ba08857fad385708ca0dd5cbdc0166ded9f1607b7a4d2",
    },

    "zheng68k": {
        15:
            "35490be0d1a83b46517d2d5792a2c0a1bc9c5b3f800f29aeb90447d11d8a6ecc",
        30:
            "ba6b8e6d2698048faac52e6b384a795bd429ec3d58bb385b1fc769ad50da7871",
        50:
            "2313651071d4c2d69053a51e21ba477d3f2c3c146172dc187ac809af4590f040",
    },

    "baron_pancreas": {
        15:
            "20dd3ef3454c91fad9ef121518827df4b64ced1adb5d1b405e38ab0f6acc498b",
        30:
            "edc1422a764c1aa40a9028401df6629c771fc7f3809ff5e9a7443be6fabd1e3e",
        50:
            "7654fcc78719327f73fcbe46bf69beaf866e32b9331134f2aec1bc4964b2d792",
    },
}

SOURCE_HASHES = {
    "zheng68k":
        "3f35f37ff344bc9b32f97cd003ac986ebb9b5d7f31006c53dff4cb38da267931",

    "baron_pancreas":
        "6b257a03bcf61fa58c08ef9ca1de6957cb9072cfdd19bf78ce0c5743de031a63",
}


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sha256_uint32(
    x: np.ndarray,
) -> str:
    a = np.asarray(
        x,
        dtype="<u4",
        order="C",
    )

    return hashlib.sha256(
        a.tobytes(
            order="C"
        )
    ).hexdigest()


def sha256_bool(
    x: np.ndarray,
) -> str:
    a = np.asarray(
        x,
        dtype=np.uint8,
        order="C",
    )

    return hashlib.sha256(
        a.tobytes(
            order="C"
        )
    ).hexdigest()


def internal_counts(
    corpus: Path,
) -> np.ndarray:

    root = (
        corpus
        / "shards"
        / "sc_test_00000"
    )

    mm = CSRMemmap.open(
        root,
        "counts",
    )

    if tuple(
        mm.shape
    ) != (
        25000,
        N_GENES,
    ):
        raise RuntimeError(
            f"Unexpected test count "
            f"shard shape: {mm.shape}"
        )

    stop = int(
        mm.indptr[
            N_CELLS
        ]
    )

    matrix = sparse.csr_matrix(
        (
            np.array(
                mm.data[:stop],
                copy=True,
            ),

            np.array(
                mm.indices[:stop],
                dtype=np.int32,
                copy=True,
            ),

            np.array(
                mm.indptr[
                    : N_CELLS + 1
                ],
                dtype=np.int64,
                copy=True,
            ),
        ),

        shape=(
            N_CELLS,
            N_GENES,
        ),
    )

    dense = np.asarray(
        matrix.toarray(),
        dtype=np.uint32,
        order="C",
    )

    if dense.shape != (
        N_CELLS,
        N_GENES,
    ):
        raise RuntimeError(
            "Internal dense count "
            "shape mismatch"
        )

    return dense


def verify_selected_cells(
    benchmark_root: Path,
    source_rows: np.ndarray,
    cell_ids: np.ndarray,
    cell_types: np.ndarray | None,
) -> None:

    selected = pd.read_csv(
        benchmark_root
        / "selected_cells.csv"
    )

    if len(selected) != N_CELLS:
        raise RuntimeError(
            "selected_cells row "
            "count mismatch"
        )

    if not np.array_equal(
        selected[
            "row"
        ].to_numpy(
            dtype=np.int64
        ),
        np.asarray(
            source_rows,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "selected_cells "
            "source-row mismatch"
        )

    if not np.array_equal(
        selected[
            "cell_id"
        ].astype(
            str
        ).to_numpy(),
        np.asarray(
            cell_ids
        ).astype(
            str
        ),
    ):
        raise RuntimeError(
            "selected_cells "
            "cell-id mismatch"
        )

    if cell_types is not None:
        if (
            "cell_type"
            not in selected.columns
        ):
            raise RuntimeError(
                "Baron selected_cells "
                "missing cell_type"
            )

        if not np.array_equal(
            selected[
                "cell_type"
            ].astype(
                str
            ).to_numpy(),
            np.asarray(
                cell_types
            ).astype(
                str
            ),
        ):
            raise RuntimeError(
                "Baron selected_cells "
                "cell-type mismatch"
            )


def verify_panels(
    *,
    dataset: str,
    benchmark_root: Path,
    counts: np.ndarray,
    available: np.ndarray,
    normalized_target: np.ndarray,
) -> dict[str, Any]:

    if counts.shape != (
        N_CELLS,
        N_GENES,
    ):
        raise RuntimeError(
            f"{dataset}: "
            "count shape mismatch"
        )

    if available.shape != (
        N_GENES,
    ):
        raise RuntimeError(
            f"{dataset}: "
            "availability shape mismatch"
        )

    if normalized_target.shape != (
        N_CELLS,
        N_GENES,
    ):
        raise RuntimeError(
            f"{dataset}: target "
            "shape mismatch"
        )

    if np.any(
        normalized_target[
            :,
            ~available,
        ]
        != 0
    ):
        raise RuntimeError(
            f"{dataset}: unavailable "
            "target genes nonzero"
        )

    records = {}

    for mask_percent in MASKS:

        if dataset == "internal_test":
            panel_path = (
                benchmark_root
                / f"test_mask{mask_percent}.npz"
            )
        else:
            panel_path = (
                benchmark_root
                / f"mask{mask_percent}.npz"
            )

        panel_sha = sha256_file(
            panel_path
        )

        if (
            panel_sha
            != PANEL_HASHES[
                dataset
            ][
                mask_percent
            ]
        ):
            raise RuntimeError(
                f"{dataset} "
                f"mask{mask_percent}: "
                "panel SHA mismatch"
            )

        with np.load(
            panel_path,
            allow_pickle=False,
        ) as panel:

            x = np.asarray(
                panel["x"],
                dtype=np.float32,
            )

            y = np.asarray(
                panel["y"],
                dtype=np.float32,
            )

            mask = np.asarray(
                panel[
                    "synthetic_mask"
                ],
                dtype=bool,
            )

            if (
                x.shape
                != (
                    N_CELLS,
                    N_GENES,
                )
                or y.shape
                != x.shape
                or mask.shape
                != x.shape
            ):
                raise RuntimeError(
                    f"{dataset} "
                    f"mask{mask_percent}: "
                    "panel shape mismatch"
                )

            if dataset == "internal_test":

                shard = np.asarray(
                    panel[
                        "shard_id"
                    ]
                ).astype(
                    str
                )

                row = np.asarray(
                    panel["row"],
                    dtype=np.int64,
                )

                if not np.all(
                    shard
                    == "sc_test_00000"
                ):
                    raise RuntimeError(
                        "Internal test "
                        "shard mismatch"
                    )

                if not np.array_equal(
                    row,
                    np.arange(
                        N_CELLS,
                        dtype=np.int64,
                    ),
                ):
                    raise RuntimeError(
                        "Internal test "
                        "row mismatch"
                    )

                if (
                    str(
                        panel[
                            "split"
                        ].item()
                    )
                    != "test"
                ):
                    raise RuntimeError(
                        "Internal split "
                        "mismatch"
                    )

                if (
                    int(
                        panel[
                            "seed"
                        ].item()
                    )
                    != SEED
                ):
                    raise RuntimeError(
                        "Internal panel "
                        "seed mismatch"
                    )

            else:

                panel_available = (
                    np.asarray(
                        panel[
                            "available_gene_mask"
                        ],
                        dtype=bool,
                    )
                )

                if not np.array_equal(
                    panel_available,
                    available,
                ):
                    raise RuntimeError(
                        f"{dataset} "
                        f"mask{mask_percent}: "
                        "availability mismatch"
                    )

        if not np.array_equal(
            y,
            normalized_target,
        ):
            max_abs = float(
                np.max(
                    np.abs(
                        y.astype(
                            np.float64
                        )
                        - normalized_target.astype(
                            np.float64
                        )
                    )
                )
            )

            raise RuntimeError(
                f"{dataset} "
                f"mask{mask_percent}: "
                "clean-target mismatch "
                f"max_abs={max_abs}"
            )

        if np.any(
            mask
            & (
                ~available[
                    None,
                    :
                ]
            )
        ):
            raise RuntimeError(
                f"{dataset} "
                f"mask{mask_percent}: "
                "mask includes "
                "unavailable gene"
            )

        if not np.all(
            counts[
                mask
            ]
            > 0
        ):
            raise RuntimeError(
                f"{dataset} "
                f"mask{mask_percent}: "
                "mask includes "
                "nonpositive raw count"
            )

        if not np.all(
            y[
                mask
            ]
            > 0
        ):
            raise RuntimeError(
                f"{dataset} "
                f"mask{mask_percent}: "
                "masked target "
                "not positive"
            )

        if np.any(
            np.abs(
                x[
                    mask
                ]
            )
            > 1.0e-8
        ):
            raise RuntimeError(
                f"{dataset} "
                f"mask{mask_percent}: "
                "masked x nonzero"
            )

        if not np.array_equal(
            x[
                ~mask
            ],
            y[
                ~mask
            ],
        ):
            raise RuntimeError(
                f"{dataset} "
                f"mask{mask_percent}: "
                "x/y differ outside mask"
            )

        records[
            str(
                mask_percent
            )
        ] = {
            "path":
                str(
                    panel_path
                ),

            "sha256":
                panel_sha,

            "n_masked":
                int(
                    mask.sum()
                ),

            "target_alignment_max_abs_error":
                0.0,

            "x_equals_y_outside_mask":
                True,

            "x_zero_on_mask":
                True,

            "mask_positive_in_raw_counts":
                True,
        }

    return records


def write_dataset_bundle(
    root: Path,
    *,
    dataset: str,
    counts: np.ndarray,
    available: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:

    out = (
        root
        / dataset
    )

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    counts_path = (
        out
        / "counts_uint32.npy"
    )

    available_path = (
        out
        / "available_gene_mask.npy"
    )

    np.save(
        counts_path,
        np.asarray(
            counts,
            dtype=np.uint32,
            order="C",
        ),
        allow_pickle=False,
    )

    np.save(
        available_path,
        np.asarray(
            available,
            dtype=bool,
        ),
        allow_pickle=False,
    )

    metadata = dict(
        metadata
    )

    metadata.update({
        "dataset":
            dataset,

        "shape":
            [
                N_CELLS,
                N_GENES,
            ],

        "counts_dtype":
            "uint32",

        "counts_uint32_sha256":
            sha256_uint32(
                counts
            ),

        "available_gene_mask_sha256":
            sha256_bool(
                available
            ),

        "n_available_genes":
            int(
                available.sum()
            ),

        "n_unavailable_genes":
            int(
                (
                    ~available
                ).sum()
            ),

        "counts_file_sha256":
            sha256_file(
                counts_path
            ),

        "available_file_sha256":
            sha256_file(
                available_path
            ),
    })

    metadata_path = (
        out
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    files = (
        "counts_uint32.npy",
        "available_gene_mask.npy",
        "metadata.json",
    )

    with (
        out
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        for name in files:
            handle.write(
                f"{sha256_file(out / name)}"
                f"  {name}\n"
            )

    return {
        "metadata_sha256":
            sha256_file(
                metadata_path
            ),

        "bundle_manifest_sha256":
            sha256_file(
                out
                / "SHA256SUMS.txt"
            ),

        "counts_file_sha256":
            metadata[
                "counts_file_sha256"
            ],

        "counts_uint32_sha256":
            metadata[
                "counts_uint32_sha256"
            ],

        "available_file_sha256":
            metadata[
                "available_file_sha256"
            ],

        "available_gene_mask_sha256":
            metadata[
                "available_gene_mask_sha256"
            ],

        "n_available_genes":
            metadata[
                "n_available_genes"
            ],
    }


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--corpus",
        required=True,
    )

    p.add_argument(
        "--zheng-raw-root",
        required=True,
    )

    p.add_argument(
        "--zheng-archive",
        required=True,
    )

    p.add_argument(
        "--zheng-benchmark-root",
        required=True,
    )

    p.add_argument(
        "--baron-raw-root",
        required=True,
    )

    p.add_argument(
        "--baron-archive",
        required=True,
    )

    p.add_argument(
        "--baron-benchmark-root",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    return p.parse_args()


def main():
    args = parse_args()

    corpus = Path(
        args.corpus
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    zheng_raw_root = Path(
        args.zheng_raw_root
    ).resolve()

    zheng_archive = Path(
        args.zheng_archive
    ).resolve()

    zheng_benchmark_root = Path(
        args.zheng_benchmark_root
    ).resolve()

    baron_raw_root = Path(
        args.baron_raw_root
    ).resolve()

    baron_archive = Path(
        args.baron_archive
    ).resolve()

    baron_benchmark_root = Path(
        args.baron_benchmark_root
    ).resolve()

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: "
            f"{output_dir}"
        )

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        sha256_file(
            zheng_archive
        )
        != SOURCE_HASHES[
            "zheng68k"
        ]
    ):
        raise RuntimeError(
            "Zheng archive "
            "SHA mismatch"
        )

    if (
        sha256_file(
            baron_archive
        )
        != SOURCE_HASHES[
            "baron_pancreas"
        ]
    ):
        raise RuntimeError(
            "Baron archive "
            "SHA mismatch"
        )

    vocabulary = load_vocabulary(
        corpus
        / "gene_vocabulary.parquet",

        corpus
        / "train_gene_stats.parquet",
    )

    if (
        str(
            vocabulary[
                "vocabulary_sha256"
            ].iloc[0]
        )
        != VOCAB_SHA
    ):
        raise RuntimeError(
            "Vocabulary semantic "
            "SHA mismatch"
        )

    tmp = (
        output_dir.parent
        / (
            "."
            + output_dir.name
            + ".tmp"
        )
    )

    if tmp.exists():
        raise RuntimeError(
            f"Temporary path "
            f"already exists: {tmp}"
        )

    tmp.mkdir(
        exist_ok=False,
    )

    try:
        summary = {
            "schema_version":
                "sc2-p2-scvi-final-count-views-v1",

            "classification":
                "protocol_neutral_rederivation_from_frozen_artifacts",

            "n_cells":
                N_CELLS,

            "n_genes":
                N_GENES,

            "panel_seed":
                SEED,

            "vocabulary_sha256":
                VOCAB_SHA,

            "datasets":
                {},
        }

        # ----------------------------------------------------
        # Internal Census test
        # ----------------------------------------------------

        counts = internal_counts(
            corpus
        )

        available = np.ones(
            N_GENES,
            dtype=bool,
        )

        target = (
            census_cp10k_log1p(
                sparse.csr_matrix(
                    counts
                )
            )
            .toarray()
            .astype(
                np.float32,
                copy=False,
            )
        )

        panels = verify_panels(
            dataset="internal_test",

            benchmark_root=(
                corpus
                / "benchmarks"
            ),

            counts=counts,

            available=available,

            normalized_target=target,
        )

        summary[
            "datasets"
        ][
            "internal_test"
        ] = write_dataset_bundle(
            tmp,

            dataset="internal_test",

            counts=counts,

            available=available,

            metadata={
                "source":
                    (
                        "frozen CSR shard "
                        "sc_test_00000 "
                        "rows 0..4999"
                    ),

                "source_shard":
                    str(
                        corpus
                        / "shards"
                        / "sc_test_00000"
                    ),

                "normalization_verified":
                    (
                        "materialize_census_shards."
                        "cp10k_log1p"
                    ),

                "panels":
                    panels,
            },
        )

        del counts
        del available
        del target

        # ----------------------------------------------------
        # Zheng68K
        # ----------------------------------------------------

        (
            counts,
            available,
            source_rows,
            cell_ids,
            loader_metadata,
        ) = load_zheng(
            zheng_raw_root,
            vocabulary,
            n_cells=N_CELLS,
            seed=SEED,
        )

        counts = np.rint(
            counts
        ).astype(
            np.uint32,
            copy=False,
        )

        available = np.asarray(
            available,
            dtype=bool,
        )

        verify_selected_cells(
            zheng_benchmark_root,
            source_rows,
            cell_ids,
            None,
        )

        target = external_cp10k_log1p(
            counts
        )

        panels = verify_panels(
            dataset="zheng68k",

            benchmark_root=(
                zheng_benchmark_root
            ),

            counts=counts,

            available=available,

            normalized_target=target,
        )

        summary[
            "datasets"
        ][
            "zheng68k"
        ] = write_dataset_bundle(
            tmp,

            dataset="zheng68k",

            counts=counts,

            available=available,

            metadata={
                "source_archive":
                    str(
                        zheng_archive
                    ),

                "source_archive_sha256":
                    SOURCE_HASHES[
                        "zheng68k"
                    ],

                "raw_root":
                    str(
                        zheng_raw_root
                    ),

                "normalization_verified":
                    (
                        "prepare_external_"
                        "repair_benchmarks."
                        "cp10k_log1p"
                    ),

                "loader_metadata":
                    loader_metadata,

                "panels":
                    panels,
            },
        )

        del counts
        del available
        del target

        # ----------------------------------------------------
        # Baron pancreas
        # ----------------------------------------------------

        (
            counts,
            available,
            source_rows,
            cell_ids,
            cell_types,
            loader_metadata,
        ) = load_baron(
            baron_raw_root,
            vocabulary,
            n_cells=N_CELLS,
            seed=SEED,
        )

        counts = np.rint(
            counts
        ).astype(
            np.uint32,
            copy=False,
        )

        available = np.asarray(
            available,
            dtype=bool,
        )

        verify_selected_cells(
            baron_benchmark_root,
            source_rows,
            cell_ids,
            cell_types,
        )

        target = external_cp10k_log1p(
            counts
        )

        panels = verify_panels(
            dataset="baron_pancreas",

            benchmark_root=(
                baron_benchmark_root
            ),

            counts=counts,

            available=available,

            normalized_target=target,
        )

        summary[
            "datasets"
        ][
            "baron_pancreas"
        ] = write_dataset_bundle(
            tmp,

            dataset="baron_pancreas",

            counts=counts,

            available=available,

            metadata={
                "source_archive":
                    str(
                        baron_archive
                    ),

                "source_archive_sha256":
                    SOURCE_HASHES[
                        "baron_pancreas"
                    ],

                "raw_root":
                    str(
                        baron_raw_root
                    ),

                "normalization_verified":
                    (
                        "prepare_external_"
                        "repair_benchmarks."
                        "cp10k_log1p"
                    ),

                "loader_metadata":
                    loader_metadata,

                "panels":
                    panels,
            },
        )

        summary_path = (
            tmp
            / "summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            tmp
            / "SHA256SUMS.txt"
        ).write_text(
            (
                f"{sha256_file(summary_path)}"
                "  summary.json\n"
            ),
            encoding="utf-8",
        )

        tmp.rename(
            output_dir
        )

    except Exception:
        shutil.rmtree(
            tmp,
            ignore_errors=True,
        )
        raise

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    print(
        "P2_SCVI_FINAL_COUNT_VIEWS=PASS"
    )


if __name__ == "__main__":
    main()
