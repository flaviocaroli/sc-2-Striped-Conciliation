#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts.analysis import (
    prepare_p2_scvi_final_count_views as base
)

from scripts.data.materialize_census_shards import (
    cp10k_log1p as census_cp10k_log1p,
)

from scripts.data.prepare_external_repair_benchmarks_p3_16k import (
    cp10k_log1p as external_cp10k_log1p,
    load_vocabulary,
    load_zheng,
)


N_CELLS = 5000
N_GENES = 16384
SEED = 20260729

VOCAB_FILE_SHA = (
    "0f1ecc3f50484bcd817dd26c5275b9a9"
    "ccd68072d774db5c297d4f405e5db0fa"
)

VOCAB_SEMANTIC_SHA = (
    "4697c8de54f42f6e6352a94ea22c820e"
    "6309221aa808d6053377e23371564a9e"
)

ZHENG_ARCHIVE_SHA = (
    "3f35f37ff344bc9b32f97cd003ac986e"
    "bb9b5d7f31006c53dff4cb38da267931"
)

BARON_BUNDLE_SHA = (
    "7316ccc0efe849ff06a09148f09adc0b2"
    "ae7cf0e13ce10a2ec29d095bb472c0d"
)

PANEL_HASHES = {
    "internal_test": {
        15:
            "84a5bf413c0bcdf192fb18f43326871ff"
            "3fb4c9f700c31a9ce0e321f9c8eb33c",

        30:
            "fc804ea6496660c950bd3ada3930e8879"
            "df8a96c59883adba06312cb63e60533",

        50:
            "5392e7fc9b24cb5a1a017ab5e537dd90"
            "17d6ee36b6ec5a0ae9011b5d38c47628",
    },

    "baron_pancreas": {
        15:
            "8c90c598f5732e081cf87a69ccdbd576"
            "3ad2fb7bb207509b614d4a2da8655199",

        30:
            "26c3959a2a557ad8e19aa56a6f021799"
            "8c8fa892f234e43429e42254c9a4e7f0",

        50:
            "96a33461cece231780d62bc0967c293eb"
            "50286df68aefcea4112394fe80031da",
    },

    "zheng68k": {
        15:
            "338c70ea2a755d00cafdfa3e0dc9dd01"
            "2fbc5c207835f55c8c43948a4473c076",

        30:
            "619ec7bcafd39daa7a1c51bba0aa2274"
            "4c736d0fde6e3694a05a0286e5d30c8a",

        50:
            "71657d901729bf05367cebb36135e3fd4"
            "e69ee456f1abf36db84684cacade154",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def write_root_manifest(root: Path) -> None:

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
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if output_dir.exists():
        raise RuntimeError(
            f"Output exists: {output_dir}"
        )

    temp_dir = (
        output_dir.parent
        / f".{output_dir.name}.tmp.{os.getpid()}"
    )

    if temp_dir.exists():
        raise RuntimeError(
            f"Temporary path exists: {temp_dir}"
        )

    temp_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # Patch only the dimension/hash constants of the already
    # tested count-view verification utilities.
    #
    base.N_CELLS = N_CELLS
    base.N_GENES = N_GENES
    base.SEED = SEED
    base.PANEL_HASHES = PANEL_HASHES

    corpus = Path(
        "/home/3159436/sc2/data/"
        "census_curated_pretrain_pilot250k_g16384"
    )

    vocabulary_path = (
        corpus
        / "gene_vocabulary.parquet"
    )

    gene_stats_path = (
        corpus
        / "train_gene_stats.parquet"
    )

    internal_benchmark = (
        corpus
        / "benchmarks"
        / "test_masking_v1"
    )

    zheng_raw_root = Path(
        "/home/3159436/sc2/data/"
        "external_raw/zheng68k"
    )

    zheng_archive = (
        zheng_raw_root
        / "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
    )

    zheng_benchmark = Path(
        "/home/3159436/sc2/data/"
        "external_benchmarks/"
        "zheng68k_g16384_v1"
    )

    baron_benchmark = Path(
        "/home/3159436/sc2/data/"
        "external_benchmarks/"
        "baron_pancreas_g16384_v1"
    )

    baron_bundle = Path(
        "/home/3159436/sc2/outputs/"
        "paper_extension_2026_p3/"
        "foundation_baron_representation_v1/"
        "baron5000_raw_counts_g16384.npz"
    )

    #
    # Immutable source gates.
    #
    if (
        sha256_file(vocabulary_path)
        != VOCAB_FILE_SHA
    ):
        raise RuntimeError(
            "P3 vocabulary SHA mismatch"
        )

    if (
        sha256_file(zheng_archive)
        != ZHENG_ARCHIVE_SHA
    ):
        raise RuntimeError(
            "Zheng archive SHA mismatch"
        )

    if (
        sha256_file(baron_bundle)
        != BARON_BUNDLE_SHA
    ):
        raise RuntimeError(
            "Baron frozen raw16k bundle SHA mismatch"
        )

    vocabulary = load_vocabulary(
        vocabulary_path,
        gene_stats_path,
    )

    if len(vocabulary) != N_GENES:
        raise RuntimeError(
            "Vocabulary length mismatch"
        )

    semantic_values = set(
        vocabulary[
            "vocabulary_sha256"
        ].astype(str)
    )

    if semantic_values != {
        VOCAB_SEMANTIC_SHA
    }:
        raise RuntimeError(
            "Vocabulary semantic SHA mismatch: "
            f"{semantic_values}"
        )

    temp_dir.mkdir(
        parents=False,
        exist_ok=False,
    )

    summary = {
        "schema":
            "sc2-p3-16k-count-thinning-confirmatory-count-views-v1",

        "status":
            "PASS",

        "classification":
            "protocol_neutral_rederivation_from_frozen_p3_artifacts",

        "n_cells":
            N_CELLS,

        "n_genes":
            N_GENES,

        "panel_seed":
            SEED,

        "vocabulary_file_sha256":
            VOCAB_FILE_SHA,

        "vocabulary_semantic_sha256":
            VOCAB_SEMANTIC_SHA,

        "datasets":
            {},
    }

    try:

        # ====================================================
        # INTERNAL TEST
        # ====================================================

        counts = base.internal_counts(
            corpus
        )

        counts = np.asarray(
            counts,
            dtype=np.uint32,
            order="C",
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

        panels = base.verify_panels(
            dataset="internal_test",

            benchmark_root=
                internal_benchmark,

            counts=counts,

            available=available,

            normalized_target=target,
        )

        summary[
            "datasets"
        ][
            "internal_test"
        ] = base.write_dataset_bundle(
            temp_dir,

            dataset="internal_test",

            counts=counts,

            available=available,

            metadata={
                "source":
                    (
                        "frozen P3 CSR shard "
                        "sc_test_00000 rows 0..4999"
                    ),

                "source_shard":
                    str(
                        corpus
                        / "shards"
                        / "sc_test_00000"
                    ),

                "reference_panel":
                    str(
                        internal_benchmark
                        / "test_mask15.npz"
                    ),

                "reference_panel_sha256":
                    PANEL_HASHES[
                        "internal_test"
                    ][15],

                "full_target_alignment_exact":
                    True,

                "panels":
                    panels,
            },
        )

        print(
            "P3_THINNING_SOURCE_INTERNAL=PASS",
            flush=True,
        )

        del (
            counts,
            available,
            target,
        )

        # ====================================================
        # BARON
        # ====================================================

        with np.load(
            baron_bundle,
            allow_pickle=False,
        ) as bundle:

            counts = np.asarray(
                bundle["counts"],
                dtype=np.uint32,
            )

            available = np.asarray(
                bundle[
                    "available_gene_mask"
                ],
                dtype=bool,
            )

            source_rows = np.asarray(
                bundle[
                    "source_row"
                ],
                dtype=np.int64,
            )

            cell_ids = np.asarray(
                bundle[
                    "cell_id"
                ]
            ).astype(str)

            cell_types = np.asarray(
                bundle[
                    "cell_type"
                ]
            ).astype(str)

        if counts.shape != (
            N_CELLS,
            N_GENES,
        ):
            raise RuntimeError(
                "Baron raw16k shape mismatch"
            )

        if int(
            available.sum()
        ) != 13762:
            raise RuntimeError(
                "Baron availability mismatch"
            )

        base.verify_selected_cells(
            baron_benchmark,
            source_rows,
            cell_ids,
            cell_types,
        )

        target = external_cp10k_log1p(
            counts
        )

        panels = base.verify_panels(
            dataset="baron_pancreas",

            benchmark_root=
                baron_benchmark,

            counts=counts,

            available=available,

            normalized_target=target,
        )

        summary[
            "datasets"
        ][
            "baron_pancreas"
        ] = base.write_dataset_bundle(
            temp_dir,

            dataset="baron_pancreas",

            counts=counts,

            available=available,

            metadata={
                "source":
                    "frozen P3 Baron raw16k bundle",

                "source_bundle":
                    str(
                        baron_bundle
                    ),

                "source_bundle_sha256":
                    BARON_BUNDLE_SHA,

                "reference_panel":
                    str(
                        baron_benchmark
                        / "mask15.npz"
                    ),

                "reference_panel_sha256":
                    PANEL_HASHES[
                        "baron_pancreas"
                    ][15],

                "full_target_alignment_exact":
                    True,

                "panels":
                    panels,
            },
        )

        print(
            "P3_THINNING_SOURCE_BARON=PASS",
            flush=True,
        )

        del (
            counts,
            available,
            target,
            source_rows,
            cell_ids,
            cell_types,
        )

        # ====================================================
        # ZHENG68K
        # ====================================================

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

        if int(
            available.sum()
        ) != 15683:
            raise RuntimeError(
                "Zheng availability mismatch"
            )

        base.verify_selected_cells(
            zheng_benchmark,
            source_rows,
            cell_ids,
            None,
        )

        target = external_cp10k_log1p(
            counts
        )

        panels = base.verify_panels(
            dataset="zheng68k",

            benchmark_root=
                zheng_benchmark,

            counts=counts,

            available=available,

            normalized_target=target,
        )

        summary[
            "datasets"
        ][
            "zheng68k"
        ] = base.write_dataset_bundle(
            temp_dir,

            dataset="zheng68k",

            counts=counts,

            available=available,

            metadata={
                "source":
                    (
                        "P3 exact-Ensembl rederivation "
                        "from frozen Zheng68K raw archive"
                    ),

                "source_archive":
                    str(
                        zheng_archive
                    ),

                "source_archive_sha256":
                    ZHENG_ARCHIVE_SHA,

                "loader_metadata":
                    loader_metadata,

                "reference_panel":
                    str(
                        zheng_benchmark
                        / "mask15.npz"
                    ),

                "reference_panel_sha256":
                    PANEL_HASHES[
                        "zheng68k"
                    ][15],

                "full_target_alignment_exact":
                    True,

                "panels":
                    panels,
            },
        )

        print(
            "P3_THINNING_SOURCE_ZHENG=PASS",
            flush=True,
        )

        del (
            counts,
            available,
            target,
            source_rows,
            cell_ids,
        )

        #
        # Root-level scientific receipt.
        #
        receipt = (
            temp_dir
            / "count_views_receipt.json"
        )

        receipt.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        write_root_manifest(
            temp_dir
        )

        #
        # Self-verify root manifest before atomic finalization.
        #
        for line in (
            temp_dir
            / "SHA256SUMS.txt"
        ).read_text().splitlines():

            digest, rel = line.split(
                None,
                1,
            )

            path = (
                temp_dir
                / rel.strip()
            )

            if (
                sha256_file(path)
                != digest
            ):
                raise RuntimeError(
                    f"Manifest verification "
                    f"failed: {path}"
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
        "P3_16K_CONFIRMATORY_COUNT_VIEWS=PASS"
    )

    print(
        f"OUTPUT={output_dir}"
    )

    print(
        "DATASETS=3"
    )

    print(
        "CELLS_PER_DATASET=5000"
    )

    print(
        "GENES=16384"
    )

    print(
        "FULL_TARGET_ALIGNMENT_EXACT=true"
    )

    print(
        "ROOT_MANIFEST_SHA256="
        + sha256_file(
            output_dir
            / "SHA256SUMS.txt"
        )
    )


if __name__ == "__main__":
    main()
