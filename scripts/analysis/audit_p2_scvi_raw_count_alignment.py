#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sc2.data.csr_shard import CSRMemmap
from scripts.data.materialize_census_shards import cp10k_log1p


EXPECTED_PANEL_HASHES = {
    15: "24de301e406200e6b23b457cac2ef8a3081e9cd627ae1f3823c08ac3aef6b430",
    30: "7709e275fc1f87ea64763e83380b47a1cfb4e8292bc594db85181a4483fb2f6e",
    50: "ca6b46fa5d262d99595a21c5c279a31241a2af395c54f7a7adcf4436ae2c3fb6",
}

EXPECTED_MANIFEST_SHA256 = (
    "2e5adfb5558113ad570651e57cabee78362ae70ac27c96db7e353eea51055b85"
)

EXPECTED_VOCAB_SEMANTIC_SHA256 = (
    "220093512a48251f9b4f2d45e8bf9134725473ce5bab89c93e2440bea1f74d01"
)

EXPECTED_VOCAB_FILE_SHA256 = (
    "2b955a4eb84e7c1d9fe2b74808bdc0bdd7bbe20932eb5440409131a2c1fbe2d2"
)

EXPECTED_SHARD_SHA256 = (
    "3ea43559f3df15c1dc28fe86855709fcaedc42c29cf85f7c01eb8b35b49f473b"
)

EXPECTED_SHARD_ID = "sc_validation_00000"
EXPECTED_N_PANEL_CELLS = 2500
EXPECTED_N_GENES = 4096
EXPECTED_SEED = 20260728


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def sha256_dense_uint32(x: np.ndarray) -> str:
    a = np.asarray(
        x,
        dtype="<u4",
        order="C",
    )
    return hashlib.sha256(
        a.tobytes(order="C")
    ).hexdigest()


def mask_seed(
    seed: int,
    split: str,
    shard_id: str,
    row: int,
) -> int:
    payload = (
        f"{seed}:{split}:{shard_id}:{row}"
    ).encode("utf-8")

    return int.from_bytes(
        hashlib.blake2b(
            payload,
            digest_size=8,
        ).digest(),
        "little",
    )


def first_rows_as_csr(
    matrix: CSRMemmap,
    n_rows: int,
) -> sparse.csr_matrix:
    if n_rows <= 0 or n_rows > matrix.shape[0]:
        raise ValueError(
            f"Invalid n_rows={n_rows}"
        )

    stop = int(
        matrix.indptr[n_rows]
    )

    data = np.asarray(
        matrix.data[:stop]
    )

    indices = np.asarray(
        matrix.indices[:stop],
        dtype=np.int32,
    )

    indptr = np.asarray(
        matrix.indptr[: n_rows + 1],
        dtype=np.int64,
    ).copy()

    if int(indptr[0]) != 0:
        raise RuntimeError(
            "Expected first selected row to start "
            "at CSR pointer zero"
        )

    return sparse.csr_matrix(
        (
            data,
            indices,
            indptr,
        ),
        shape=(
            n_rows,
            matrix.shape[1],
        ),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Protocol-neutral audit proving that "
            "frozen p2 validation panels align "
            "exactly to genuine stored raw counts."
        )
    )

    p.add_argument(
        "--corpus",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    corpus = Path(
        args.corpus
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if output_dir.exists():
        raise FileExistsError(
            output_dir
        )

    shard_dir = (
        corpus
        / "shards"
        / EXPECTED_SHARD_ID
    )

    benchmark_root = (
        corpus
        / "benchmarks"
    )

    vocab_path = (
        corpus
        / "gene_vocabulary.parquet"
    )

    manifest_path = (
        corpus
        / "shards.parquet"
    )

    #
    # Frozen vocabulary.
    #
    vocab_file_hash = sha256_file(
        vocab_path
    )

    if (
        vocab_file_hash
        != EXPECTED_VOCAB_FILE_SHA256
    ):
        raise RuntimeError(
            "Vocabulary file SHA256 mismatch"
        )

    vocab = pd.read_parquet(
        vocab_path
    )

    if len(vocab) != EXPECTED_N_GENES:
        raise RuntimeError(
            "Unexpected vocabulary length"
        )

    expected_indices = np.arange(
        EXPECTED_N_GENES,
        dtype=np.int64,
    )

    if not np.array_equal(
        vocab["gene_index"].to_numpy(
            dtype=np.int64
        ),
        expected_indices,
    ):
        raise RuntimeError(
            "Vocabulary gene_index is not "
            "exactly 0..4095"
        )

    semantic_hashes = set(
        vocab[
            "vocabulary_sha256"
        ].astype(str)
    )

    if semantic_hashes != {
        EXPECTED_VOCAB_SEMANTIC_SHA256
    }:
        raise RuntimeError(
            "Vocabulary semantic hash mismatch"
        )

    #
    # Canonical manifest record.
    #
    manifest = pd.read_parquet(
        manifest_path
    )

    hit = manifest[
        manifest["shard_id"].astype(str)
        == EXPECTED_SHARD_ID
    ]

    if len(hit) != 1:
        raise RuntimeError(
            "Expected exactly one validation "
            "shard manifest row"
        )

    record = hit.iloc[0]

    if str(record["split"]) != "validation":
        raise RuntimeError(
            "Unexpected validation shard split"
        )

    if str(record["modality"]) != "sc":
        raise RuntimeError(
            "Unexpected validation shard modality"
        )

    if int(record["n_rows"]) != 25000:
        raise RuntimeError(
            "Unexpected validation shard row count"
        )

    if int(record["n_genes"]) != EXPECTED_N_GENES:
        raise RuntimeError(
            "Unexpected validation shard gene count"
        )

    if str(record["sha256"]) != EXPECTED_SHARD_SHA256:
        raise RuntimeError(
            "Validation shard manifest hash mismatch"
        )

    if (
        str(record["gene_vocab_sha256"])
        != EXPECTED_VOCAB_SEMANTIC_SHA256
    ):
        raise RuntimeError(
            "Validation shard vocabulary hash mismatch"
        )

    #
    # Open immutable raw-count and stored
    # normalized matrices.
    #
    counts_mm = CSRMemmap.open(
        shard_dir,
        "counts",
    )

    log1p_mm = CSRMemmap.open(
        shard_dir,
        "log1p",
    )

    expected_shape = (
        25000,
        EXPECTED_N_GENES,
    )

    if counts_mm.shape != expected_shape:
        raise RuntimeError(
            "Unexpected count matrix shape"
        )

    if log1p_mm.shape != expected_shape:
        raise RuntimeError(
            "Unexpected log1p matrix shape"
        )

    #
    # The materializer writes CP10K log1p from
    # exactly the same CSR count support. Prove
    # support identity for the full immutable shard
    # without densifying 25k x 4096.
    #
    full_support_indices_equal = bool(
        np.array_equal(
            counts_mm.indices,
            log1p_mm.indices,
        )
    )

    full_support_indptr_equal = bool(
        np.array_equal(
            counts_mm.indptr,
            log1p_mm.indptr,
        )
    )

    if not full_support_indices_equal:
        raise RuntimeError(
            "Full-shard count/log1p CSR indices differ"
        )

    if not full_support_indptr_equal:
        raise RuntimeError(
            "Full-shard count/log1p CSR indptr differ"
        )

    #
    # Only the first 2500 validation rows are in
    # the frozen benchmark.
    #
    counts_csr = first_rows_as_csr(
        counts_mm,
        EXPECTED_N_PANEL_CELLS,
    )

    stored_log1p_csr = (
        first_rows_as_csr(
            log1p_mm,
            EXPECTED_N_PANEL_CELLS,
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    if counts_csr.dtype != np.uint32:
        raise RuntimeError(
            f"Expected uint32 raw counts, "
            f"got {counts_csr.dtype}"
        )

    if counts_csr.data.size == 0:
        raise RuntimeError(
            "Validation raw-count matrix is empty"
        )

    if np.any(counts_csr.data < 0):
        raise RuntimeError(
            "Negative raw counts found"
        )

    #
    # Exact source implementation re-derivation:
    # projected raw counts -> float32 CP10K -> log1p.
    #
    recomputed_log1p_csr = (
        cp10k_log1p(
            counts_csr
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    if not np.array_equal(
        recomputed_log1p_csr.indices,
        stored_log1p_csr.indices,
    ):
        raise RuntimeError(
            "Recomputed/stored log1p support differs"
        )

    if not np.array_equal(
        recomputed_log1p_csr.indptr,
        stored_log1p_csr.indptr,
    ):
        raise RuntimeError(
            "Recomputed/stored log1p indptr differs"
        )

    normalization_exact = bool(
        np.array_equal(
            recomputed_log1p_csr.data,
            stored_log1p_csr.data,
        )
    )

    normalization_max_abs_error = float(
        np.max(
            np.abs(
                recomputed_log1p_csr.data.astype(
                    np.float64
                )
                - stored_log1p_csr.data.astype(
                    np.float64
                )
            )
        )
    )

    #
    # If library/scipy execution produces a
    # sub-ULP difference, record it, but anything
    # above 1e-6 is an audit failure.
    #
    if normalization_max_abs_error > 1.0e-6:
        raise RuntimeError(
            "Raw counts do not reproduce stored "
            "log1p(CP10K) within 1e-6"
        )

    counts_dense = (
        counts_csr.toarray()
        .astype(
            np.uint32,
            copy=False,
        )
    )

    stored_log1p_dense = (
        stored_log1p_csr.toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    if counts_dense.shape != (
        EXPECTED_N_PANEL_CELLS,
        EXPECTED_N_GENES,
    ):
        raise RuntimeError(
            "Unexpected dense count shape"
        )

    if stored_log1p_dense.shape != counts_dense.shape:
        raise RuntimeError(
            "Unexpected dense log1p shape"
        )

    count_positive = (
        counts_dense > 0
    )

    stored_positive = (
        stored_log1p_dense > 0
    )

    support_identity = bool(
        np.array_equal(
            count_positive,
            stored_positive,
        )
    )

    if not support_identity:
        raise RuntimeError(
            "Raw-count positivity does not exactly "
            "match stored log1p positivity"
        )

    clean_counts_sha256 = (
        sha256_dense_uint32(
            counts_dense
        )
    )

    panel_results: dict[str, dict] = {}
    rows_out: list[dict] = []

    reference_shard = None
    reference_rows = None
    reference_y = None

    for mask_percent in (
        15,
        30,
        50,
    ):
        panel_path = (
            benchmark_root
            / f"validation_mask{mask_percent}.npz"
        )

        observed_panel_hash = (
            sha256_file(
                panel_path
            )
        )

        if (
            observed_panel_hash
            != EXPECTED_PANEL_HASHES[
                mask_percent
            ]
        ):
            raise RuntimeError(
                f"Panel hash mismatch for "
                f"mask{mask_percent}"
            )

        panel = np.load(
            panel_path,
            allow_pickle=False,
        )

        x = np.asarray(
            panel["x"],
            dtype=np.float32,
        )

        y = np.asarray(
            panel["y"],
            dtype=np.float32,
        )

        synthetic_mask = np.asarray(
            panel["synthetic_mask"],
            dtype=bool,
        )

        shard_ids = np.asarray(
            panel["shard_id"]
        )

        panel_rows = np.asarray(
            panel["row"],
            dtype=np.int64,
        )

        if x.shape != counts_dense.shape:
            raise RuntimeError(
                f"x shape mismatch for mask{mask_percent}"
            )

        if y.shape != counts_dense.shape:
            raise RuntimeError(
                f"y shape mismatch for mask{mask_percent}"
            )

        if synthetic_mask.shape != counts_dense.shape:
            raise RuntimeError(
                f"mask shape mismatch for mask{mask_percent}"
            )

        if str(panel["split"].item()) != "validation":
            raise RuntimeError(
                "Unexpected panel split"
            )

        if str(panel["modality"].item()) != "sc":
            raise RuntimeError(
                "Unexpected panel modality"
            )

        if int(panel["seed"].item()) != EXPECTED_SEED:
            raise RuntimeError(
                "Unexpected panel seed"
            )

        if (
            str(
                panel[
                    "manifest_sha256"
                ].item()
            )
            != EXPECTED_MANIFEST_SHA256
        ):
            raise RuntimeError(
                "Panel manifest hash mismatch"
            )

        expected_rows = np.arange(
            EXPECTED_N_PANEL_CELLS,
            dtype=np.int64,
        )

        if not np.array_equal(
            panel_rows,
            expected_rows,
        ):
            raise RuntimeError(
                "Panel rows are not exactly 0..2499"
            )

        if not np.all(
            shard_ids
            == EXPECTED_SHARD_ID
        ):
            raise RuntimeError(
                "Panel contains unexpected shard id"
            )

        if reference_shard is None:
            reference_shard = shard_ids.copy()
            reference_rows = panel_rows.copy()
            reference_y = y.copy()
        else:
            if not np.array_equal(
                shard_ids,
                reference_shard,
            ):
                raise RuntimeError(
                    "Shard identity differs across masks"
                )

            if not np.array_equal(
                panel_rows,
                reference_rows,
            ):
                raise RuntimeError(
                    "Row identity differs across masks"
                )

            if not np.array_equal(
                y,
                reference_y,
            ):
                raise RuntimeError(
                    "Clean target y differs across masks"
                )

        #
        # Strongest benchmark alignment:
        # y is byte-for-byte the shard log1p rows.
        #
        benchmark_y_exact = bool(
            np.array_equal(
                y,
                stored_log1p_dense,
            )
        )

        if not benchmark_y_exact:
            diff = np.max(
                np.abs(
                    y.astype(np.float64)
                    - stored_log1p_dense.astype(
                        np.float64
                    )
                )
            )

            raise RuntimeError(
                f"Benchmark y does not exactly match "
                f"stored log1p; max_abs={diff}"
            )

        #
        # x must equal y everywhere except the
        # exact synthetic mask, where x is zero.
        #
        if not np.array_equal(
            x[~synthetic_mask],
            y[~synthetic_mask],
        ):
            raise RuntimeError(
                "Benchmark x/y differ outside mask"
            )

        if not np.all(
            x[synthetic_mask] == 0.0
        ):
            raise RuntimeError(
                "Benchmark x is not zero at "
                "all synthetic-mask positions"
            )

        if not np.all(
            y[synthetic_mask] > 0.0
        ):
            raise RuntimeError(
                "Synthetic mask contains nonpositive "
                "target entries"
            )

        if not np.all(
            counts_dense[
                synthetic_mask
            ] > 0
        ):
            raise RuntimeError(
                "Synthetic mask contains positions "
                "without genuine positive raw counts"
            )

        #
        # Independently regenerate the frozen
        # benchmark mask from its published
        # builder algorithm.
        #
        mask_rate = float(
            panel["mask_rate"].item()
        )

        regenerated = np.zeros_like(
            synthetic_mask
        )

        for row in range(
            EXPECTED_N_PANEL_CELLS
        ):
            target = (
                stored_log1p_dense[
                    row
                ]
            )

            positive = np.flatnonzero(
                target > 0
            )

            n_mask = max(
                1,
                int(
                    round(
                        mask_rate
                        * positive.size
                    )
                ),
            )

            rng = np.random.default_rng(
                mask_seed(
                    EXPECTED_SEED,
                    "validation",
                    EXPECTED_SHARD_ID,
                    row,
                )
            )

            chosen = rng.choice(
                positive,
                size=min(
                    n_mask,
                    positive.size,
                ),
                replace=False,
            )

            regenerated[
                row,
                chosen,
            ] = True

        mask_regeneration_exact = bool(
            np.array_equal(
                regenerated,
                synthetic_mask,
            )
        )

        if not mask_regeneration_exact:
            raise RuntimeError(
                "Synthetic mask regeneration failed "
                f"for mask{mask_percent}"
            )

        #
        # Construct, in memory only, the exact
        # genuine corrupted raw-count view that
        # future scVI will receive.
        #
        corrupted_counts = (
            counts_dense.copy()
        )

        before_values = (
            corrupted_counts[
                synthetic_mask
            ].copy()
        )

        corrupted_counts[
            synthetic_mask
        ] = 0

        if not np.all(
            before_values > 0
        ):
            raise RuntimeError(
                "Masked raw-count values were not "
                "all genuinely positive"
            )

        if not np.all(
            corrupted_counts[
                synthetic_mask
            ] == 0
        ):
            raise RuntimeError(
                "Failed to zero corrupted counts"
            )

        if not np.array_equal(
            corrupted_counts[
                ~synthetic_mask
            ],
            counts_dense[
                ~synthetic_mask
            ],
        ):
            raise RuntimeError(
                "Corrupted raw counts changed "
                "outside synthetic mask"
            )

        original_library = (
            counts_dense.sum(
                axis=1,
                dtype=np.uint64,
            )
        )

        corrupted_library = (
            corrupted_counts.sum(
                axis=1,
                dtype=np.uint64,
            )
        )

        if np.any(
            corrupted_library
            > original_library
        ):
            raise RuntimeError(
                "Corrupted library size increased"
            )

        corrupted_counts_hash = (
            sha256_dense_uint32(
                corrupted_counts
            )
        )

        n_masked = int(
            synthetic_mask.sum()
        )

        removed_counts = int(
            before_values.astype(
                np.uint64
            ).sum(
                dtype=np.uint64
            )
        )

        result = {
            "mask_percent":
                mask_percent,

            "panel_sha256":
                observed_panel_hash,

            "mask_rate":
                mask_rate,

            "n_cells":
                EXPECTED_N_PANEL_CELLS,

            "n_genes":
                EXPECTED_N_GENES,

            "n_masked_positive_positions":
                n_masked,

            "benchmark_y_exact_stored_log1p":
                benchmark_y_exact,

            "mask_regeneration_exact":
                mask_regeneration_exact,

            "all_mask_positions_positive_raw_counts":
                True,

            "x_equals_y_outside_mask":
                True,

            "x_zero_at_mask":
                True,

            "corrupted_counts_unchanged_outside_mask":
                True,

            "clean_counts_uint32_sha256":
                clean_counts_sha256,

            "corrupted_counts_uint32_sha256":
                corrupted_counts_hash,

            "removed_raw_count_total":
                removed_counts,

            "original_library_size_min":
                int(
                    original_library.min()
                ),

            "original_library_size_max":
                int(
                    original_library.max()
                ),

            "corrupted_library_size_min":
                int(
                    corrupted_library.min()
                ),

            "corrupted_library_size_max":
                int(
                    corrupted_library.max()
                ),

            "mean_fraction_library_removed":
                float(
                    np.mean(
                        1.0
                        - (
                            corrupted_library.astype(
                                np.float64
                            )
                            / original_library.astype(
                                np.float64
                            )
                        )
                    )
                ),
        }

        panel_results[
            str(mask_percent)
        ] = result

        rows_out.append(
            result
        )

    summary = {
        "audit_name":
            "p2_scvi_raw_count_alignment_v1",

        "classification":
            "protocol_neutral_rederivation_from_frozen_artifacts",

        "status":
            "alignment_proven_before_scvi_validation",

        "scvi_training_started":
            False,

        "inverse_normalization_used":
            False,

        "raw_count_source":
            str(
                shard_dir
                / "counts.*"
            ),

        "stored_log1p_source":
            str(
                shard_dir
                / "log1p.*"
            ),

        "validation_shard_id":
            EXPECTED_SHARD_ID,

        "validation_shard_sha256":
            EXPECTED_SHARD_SHA256,

        "manifest_sha256":
            EXPECTED_MANIFEST_SHA256,

        "vocabulary_semantic_sha256":
            EXPECTED_VOCAB_SEMANTIC_SHA256,

        "vocabulary_file_sha256":
            EXPECTED_VOCAB_FILE_SHA256,

        "n_validation_cells":
            EXPECTED_N_PANEL_CELLS,

        "n_genes":
            EXPECTED_N_GENES,

        "raw_counts_dtype":
            str(
                counts_csr.dtype
            ),

        "raw_counts_are_nonnegative_integers":
            True,

        "full_shard_count_log1p_indices_equal":
            full_support_indices_equal,

        "full_shard_count_log1p_indptr_equal":
            full_support_indptr_equal,

        "raw_count_log1p_positive_support_exact":
            support_identity,

        "normalization_rederived_with":
            (
                "scripts.data.materialize_census_shards."
                "cp10k_log1p"
            ),

        "normalization_rederived_exact":
            normalization_exact,

        "normalization_max_abs_error":
            normalization_max_abs_error,

        "clean_validation_counts_uint32_sha256":
            clean_counts_sha256,

        "panels":
            panel_results,

        "locks": [
            "No expm1/inverse normalization was used to create counts.",
            "Counts came directly from the immutable validation shard.",
            "No scVI model was fit or evaluated during this audit.",
            "No test or external panel was accessed.",
            "Future scVI corrupted counts must zero exactly the frozen synthetic_mask positions.",
        ],
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary_path = (
        output_dir
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

    pd.DataFrame(
        rows_out
    ).to_csv(
        output_dir
        / "panel_alignment.csv",
        index=False,
    )

    files = (
        "summary.json",
        "panel_alignment.csv",
    )

    with (
        output_dir
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        for name in files:
            path = output_dir / name
            handle.write(
                f"{sha256_file(path)}  {name}\n"
            )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "P2_SCVI_RAW_COUNT_ALIGNMENT=PASS"
    )


if __name__ == "__main__":
    main()
