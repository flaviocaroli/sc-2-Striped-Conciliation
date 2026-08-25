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
from sc2.data.shard_manifest import load_manifest


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--vocabulary",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=1.0e-8,
    )

    return parser.parse_args()


def open_csr(record) -> sparse.csr_matrix:
    matrix = CSRMemmap.open(
        record.path,
        "log1p",
    )

    return sparse.csr_matrix(
        (
            np.asarray(matrix.data),
            np.asarray(matrix.indices),
            np.asarray(matrix.indptr),
        ),
        shape=matrix.shape,
    )


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    vocabulary_path = Path(args.vocabulary)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    records, manifest_hash = load_manifest(
        manifest_path
    )

    selected = [
        record
        for record in records
        if (
            record.split == "train"
            and record.modality == "sc"
        )
    ]

    if len(selected) != 8:
        raise ValueError(
            f"Expected 8 train shards, got {len(selected)}"
        )

    n_genes_set = {
        int(record.n_genes)
        for record in selected
    }

    if n_genes_set != {4096}:
        raise ValueError(
            f"Unexpected gene dimensions: {n_genes_set}"
        )

    n_genes = 4096

    n_train_cells = sum(
        int(record.n_rows)
        for record in selected
    )

    if n_train_cells != 200000:
        raise ValueError(
            f"Expected 200000 train cells, got {n_train_cells}"
        )

    zero_threshold = float(
        args.zero_threshold
    )

    positive_count = np.zeros(
        n_genes,
        dtype=np.int64,
    )

    positive_sum = np.zeros(
        n_genes,
        dtype=np.float64,
    )

    print("===== PASS 1 COUNTS / SUMS =====")

    for record in selected:
        csr = open_csr(record)

        data = np.asarray(
            csr.data,
            dtype=np.float32,
        )

        indices = np.asarray(
            csr.indices,
            dtype=np.int64,
        )

        keep = data > zero_threshold

        positive_count += np.bincount(
            indices[keep],
            minlength=n_genes,
        ).astype(np.int64)

        positive_sum += np.bincount(
            indices[keep],
            weights=data[keep].astype(np.float64),
            minlength=n_genes,
        )

        print(
            f"pass1_shard={record.shard_id} "
            f"positive_values={int(keep.sum())}"
        )

    if np.any(positive_count <= 0):
        bad = np.flatnonzero(
            positive_count <= 0
        )

        raise ValueError(
            "Genes with no positive train observations: "
            f"{bad.tolist()}"
        )

    positive_prevalence = (
        positive_count.astype(np.float64)
        / float(n_train_cells)
    ).astype(np.float32)

    positive_mean = (
        positive_sum
        / positive_count.astype(np.float64)
    ).astype(np.float32)

    total_positive = int(
        positive_count.sum()
    )

    offsets = np.zeros(
        n_genes + 1,
        dtype=np.int64,
    )

    offsets[1:] = np.cumsum(
        positive_count,
        dtype=np.int64,
    )

    scratch_path = (
        output_dir
        / "positive_values.float32.memmap"
    )

    values = np.memmap(
        scratch_path,
        dtype=np.float32,
        mode="w+",
        shape=(total_positive,),
    )

    filled = np.zeros(
        n_genes,
        dtype=np.int64,
    )

    print("===== PASS 2 VALUES =====")

    for record in selected:
        csr = open_csr(record)
        csc = csr.tocsc()

        for gene in range(n_genes):
            start = int(
                csc.indptr[gene]
            )

            end = int(
                csc.indptr[gene + 1]
            )

            if end <= start:
                continue

            gene_values = np.asarray(
                csc.data[start:end],
                dtype=np.float32,
            )

            gene_values = gene_values[
                gene_values > zero_threshold
            ]

            count = int(
                gene_values.size
            )

            if count == 0:
                continue

            destination_start = (
                int(offsets[gene])
                + int(filled[gene])
            )

            destination_end = (
                destination_start
                + count
            )

            values[
                destination_start:
                destination_end
            ] = gene_values

            filled[gene] += count

        print(
            f"pass2_shard={record.shard_id}"
        )

    values.flush()

    if not np.array_equal(
        filled,
        positive_count,
    ):
        bad = np.flatnonzero(
            filled != positive_count
        )

        raise ValueError(
            "Collection mismatch for genes: "
            f"{bad.tolist()}"
        )

    print("===== EXACT MEDIANS =====")

    positive_median = np.empty(
        n_genes,
        dtype=np.float32,
    )

    for gene in range(n_genes):
        start = int(
            offsets[gene]
        )

        end = int(
            offsets[gene + 1]
        )

        positive_median[gene] = np.float32(
            np.median(
                np.asarray(
                    values[start:end],
                    dtype=np.float32,
                )
            )
        )

        if (
            gene % 512 == 0
            or gene == n_genes - 1
        ):
            print(
                f"median_gene={gene + 1}/{n_genes}"
            )

    del values
    scratch_path.unlink()

    vocabulary = pd.read_parquet(
        vocabulary_path
    )

    if len(vocabulary) != n_genes:
        raise ValueError(
            f"Vocabulary length {len(vocabulary)} != {n_genes}"
        )

    frame = vocabulary.copy()

    if "gene_index" in frame.columns:
        observed_index = frame[
            "gene_index"
        ].to_numpy()

        expected_index = np.arange(
            n_genes,
            dtype=np.int64,
        )

        if not np.array_equal(
            observed_index,
            expected_index,
        ):
            raise ValueError(
                "Existing gene_index column is not canonical"
            )
    else:
        frame.insert(
            0,
            "gene_index",
            np.arange(
                n_genes,
                dtype=np.int64,
            ),
        )

    frame[
        "positive_count_train"
    ] = positive_count

    frame[
        "positive_prevalence_train"
    ] = positive_prevalence

    frame[
        "positive_mean_log1p_train"
    ] = positive_mean

    frame[
        "positive_median_log1p_train"
    ] = positive_median

    output_npz = (
        output_dir
        / "positive_train_statistics.npz"
    )

    np.savez_compressed(
        output_npz,
        positive_count=positive_count,
        positive_prevalence=positive_prevalence,
        positive_mean=positive_mean,
        positive_median=positive_median,
        n_train_cells=np.asarray(
            n_train_cells,
            dtype=np.int64,
        ),
        zero_threshold=np.asarray(
            zero_threshold,
            dtype=np.float64,
        ),
        manifest_sha256=np.asarray(
            manifest_hash,
        ),
    )

    output_csv = (
        output_dir
        / "positive_train_statistics.csv"
    )

    frame.to_csv(
        output_csv,
        index=False,
    )

    summary = {
        "manifest":
            str(manifest_path.resolve()),
        "manifest_sha256":
            manifest_hash,
        "vocabulary":
            str(vocabulary_path.resolve()),
        "vocabulary_sha256":
            sha256(vocabulary_path),
        "n_train_shards":
            len(selected),
        "n_train_cells":
            n_train_cells,
        "n_genes":
            n_genes,
        "zero_threshold":
            zero_threshold,
        "total_positive_values":
            total_positive,
        "positive_prevalence_min":
            float(positive_prevalence.min()),
        "positive_prevalence_max":
            float(positive_prevalence.max()),
        "positive_mean_min":
            float(positive_mean.min()),
        "positive_mean_max":
            float(positive_mean.max()),
        "positive_median_min":
            float(positive_median.min()),
        "positive_median_max":
            float(positive_median.max()),
        "definition": {
            "positive":
                "train log1p value > 1e-8",
            "prevalence":
                "positive count divided by 200000 train cells",
            "mean":
                "arithmetic mean among positive train values only",
            "median":
                "exact median among positive train values only",
        },
    }

    summary_path = (
        output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    files = [
        output_npz,
        output_csv,
        summary_path,
    ]

    hash_path = (
        output_dir
        / "SHA256SUMS.txt"
    )

    with hash_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in files:
            handle.write(
                f"{sha256(path)}  {path.name}\n"
            )

    print(summary_path.read_text())
    print("POSITIVE_TRAIN_STATS=PASS")


if __name__ == "__main__":
    main()
