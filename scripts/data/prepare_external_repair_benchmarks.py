#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


MASK_RATES = (0.15, 0.30, 0.50)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        "little",
    )


def cp10k_log1p(counts: np.ndarray) -> np.ndarray:
    matrix = np.asarray(counts, dtype=np.float32).copy()

    if np.any(matrix < 0):
        raise ValueError("Counts contain negative values")

    totals = matrix.sum(axis=1, dtype=np.float64)

    scales = np.divide(
        10000.0,
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals > 0,
    )

    matrix *= scales.astype(np.float32)[:, None]
    np.log1p(matrix, out=matrix)

    return matrix


def build_mask(
    target: np.ndarray,
    available_gene_mask: np.ndarray,
    *,
    mask_rate: float,
    seed: int,
    dataset: str,
    source_rows: np.ndarray,
) -> np.ndarray:
    result = np.zeros(target.shape, dtype=np.bool_)

    for row_index in range(target.shape[0]):
        positive = np.flatnonzero(
            (target[row_index] > 0) & available_gene_mask
        )

        if positive.size == 0:
            raise ValueError(
                f"Selected cell has no positive mapped genes: "
                f"dataset={dataset} row={source_rows[row_index]}"
            )

        n_mask = max(
            1,
            int(round(float(mask_rate) * positive.size)),
        )

        rng = np.random.default_rng(
            deterministic_seed(
                seed,
                "external",
                dataset,
                int(source_rows[row_index]),
            )
        )

        chosen = rng.choice(
            positive,
            size=min(n_mask, positive.size),
            replace=False,
        )

        result[row_index, chosen] = True

    return result


def load_vocabulary(
    vocabulary_path: Path,
    gene_stats_path: Path,
) -> pd.DataFrame:
    vocabulary = (
        pd.read_parquet(vocabulary_path)
        .sort_values("gene_index")
        .reset_index(drop=True)
    )

    required = {"gene_index", "ensembl_id", "vocabulary_sha256"}
    missing = required - set(vocabulary.columns)
    if missing:
        raise ValueError(f"Vocabulary missing columns: {sorted(missing)}")

    if vocabulary["gene_index"].tolist() != list(range(len(vocabulary))):
        raise ValueError("Vocabulary gene_index is not contiguous")

    if len(vocabulary) != 4096:
        raise ValueError(f"Expected 4096 genes, got {len(vocabulary)}")

    stats = pd.read_parquet(
        gene_stats_path,
        columns=["ensembl_id", "feature_name"],
    )

    if stats["ensembl_id"].duplicated().any():
        raise ValueError("train_gene_stats has duplicate ensembl_id values")

    merged = vocabulary.merge(
        stats,
        on="ensembl_id",
        how="left",
        validate="one_to_one",
    )

    if merged["feature_name"].isna().any():
        raise ValueError("Some vocabulary genes have no feature_name")

    if merged["feature_name"].duplicated().any():
        raise ValueError(
            "Frozen vocabulary contains ambiguous feature names"
        )

    return merged


def select_rows(
    n_source: int,
    n_cells: int,
    *,
    seed: int,
    dataset: str,
) -> np.ndarray:
    if n_cells > n_source:
        raise ValueError(
            f"Requested {n_cells} cells but source contains {n_source}"
        )

    rng = np.random.default_rng(
        deterministic_seed(seed, "cell_selection", dataset)
    )

    return np.sort(
        rng.choice(
            n_source,
            size=n_cells,
            replace=False,
        ).astype(np.int64)
    )


def load_zheng(
    root: Path,
    vocabulary: pd.DataFrame,
    *,
    n_cells: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    hg19 = root / "extracted/filtered_matrices_mex/hg19"

    genes_path = hg19 / "genes.tsv"
    barcodes_path = hg19 / "barcodes.tsv"
    matrix_path = hg19 / "matrix.mtx"

    genes = pd.read_csv(
        genes_path,
        sep="\t",
        header=None,
        names=["ensembl_id", "source_feature_name"],
        dtype=str,
    )

    if genes["ensembl_id"].duplicated().any():
        raise ValueError("Zheng genes.tsv has duplicate Ensembl IDs")

    barcodes = pd.read_csv(
        barcodes_path,
        header=None,
        names=["barcode"],
        dtype=str,
    )["barcode"].to_numpy()

    source_lookup = {
        value: index
        for index, value in enumerate(
            genes["ensembl_id"].astype(str).tolist()
        )
    }

    source_rows_for_vocab = np.asarray(
        [
            source_lookup.get(ensembl_id, -1)
            for ensembl_id in vocabulary["ensembl_id"].astype(str)
        ],
        dtype=np.int64,
    )

    available = source_rows_for_vocab >= 0
    available_positions = np.flatnonzero(available)

    matrix = mmread(matrix_path).tocsr()

    if matrix.shape != (len(genes), len(barcodes)):
        raise ValueError(
            f"Unexpected Zheng matrix shape {matrix.shape}"
        )

    if np.any(matrix.data < 0):
        raise ValueError("Zheng matrix contains negative counts")

    if np.any(np.abs(matrix.data - np.rint(matrix.data)) > 1.0e-6):
        raise ValueError("Zheng matrix is not integer-like")

    selected_rows = select_rows(
        len(barcodes),
        n_cells,
        seed=seed,
        dataset="zheng68k",
    )

    source_gene_rows = source_rows_for_vocab[available]

    selected_sparse = (
        matrix[source_gene_rows, :][:, selected_rows]
        .transpose()
        .tocsr()
    )

    counts = np.zeros(
        (n_cells, len(vocabulary)),
        dtype=np.float32,
    )

    counts[:, available_positions] = selected_sparse.toarray().astype(
        np.float32,
        copy=False,
    )

    cell_ids = barcodes[selected_rows].astype(str)

    metadata = {
        "dataset": "zheng68k",
        "source_cells": int(len(barcodes)),
        "selected_cells": int(n_cells),
        "source_genes": int(len(genes)),
        "available_genes": int(available.sum()),
        "unavailable_genes": int((~available).sum()),
        "coverage_fraction": float(available.mean()),
        "mapping": "exact_ensembl_id",
    }

    return counts, available, selected_rows, cell_ids, metadata


def baron_header(path: Path) -> list[str]:
    with gzip.open(path, "rt", newline="") as handle:
        return next(csv.reader(handle))


def load_baron(
    root: Path,
    vocabulary: pd.DataFrame,
    *,
    n_cells: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    extracted = root / "extracted"

    paths = [
        extracted / "GSM2230757_human1_umifm_counts.csv.gz",
        extracted / "GSM2230758_human2_umifm_counts.csv.gz",
        extracted / "GSM2230759_human3_umifm_counts.csv.gz",
        extracted / "GSM2230760_human4_umifm_counts.csv.gz",
    ]

    headers = [baron_header(path) for path in paths]

    if any(header != headers[0] for header in headers[1:]):
        raise ValueError("Baron human files have inconsistent headers")

    source_symbols = headers[0][3:]

    if len(source_symbols) != len(set(source_symbols)):
        raise ValueError("Baron source contains duplicate gene symbols")

    source_symbol_set = set(source_symbols)

    feature_names = vocabulary["feature_name"].astype(str).tolist()

    available = np.asarray(
        [
            symbol in source_symbol_set
            for symbol in feature_names
        ],
        dtype=bool,
    )

    available_positions = np.flatnonzero(available)
    available_symbols = [
        feature_names[index]
        for index in available_positions
    ]

    symbol_to_column = {
        symbol: 3 + index
        for index, symbol in enumerate(source_symbols)
    }

    gene_usecols = [
        symbol_to_column[symbol]
        for symbol in available_symbols
    ]

    frames: list[pd.DataFrame] = []
    global_offset = 0

    for path in paths:
        usecols = [0, 1, 2, *gene_usecols]

        frame = pd.read_csv(
            path,
            compression="gzip",
            usecols=usecols,
        )

        if "barcode" not in frame.columns:
            raise ValueError(f"Missing barcode column in {path}")

        if "assigned_cluster" not in frame.columns:
            raise ValueError(f"Missing assigned_cluster in {path}")

        identifier_columns = [
            column
            for column in frame.columns
            if column not in {"barcode", "assigned_cluster"}
            and column not in available_symbols
        ]

        if len(identifier_columns) != 1:
            raise ValueError(
                f"Could not identify Baron cell-id column: "
                f"{identifier_columns}"
            )

        identifier_column = identifier_columns[0]

        frame["source_row"] = (
            global_offset + np.arange(len(frame), dtype=np.int64)
        )

        sample = path.name.split("_umifm_counts")[0]

        frame["external_cell_id"] = (
            sample
            + ":"
            + frame[identifier_column].astype(str)
        )

        frame["source_sample"] = sample

        global_offset += len(frame)
        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    selected_indices = select_rows(
        len(combined),
        n_cells,
        seed=seed,
        dataset="baron_pancreas",
    )

    selected = combined.iloc[selected_indices].reset_index(drop=True)

    counts = np.zeros(
        (n_cells, len(vocabulary)),
        dtype=np.float32,
    )

    counts[:, available_positions] = selected[
        available_symbols
    ].to_numpy(dtype=np.float32)

    if np.any(counts < 0):
        raise ValueError("Baron matrix contains negative counts")

    if np.any(np.abs(counts - np.rint(counts)) > 1.0e-6):
        raise ValueError("Baron matrix is not integer-like")

    source_rows = selected["source_row"].to_numpy(dtype=np.int64)
    cell_ids = selected["external_cell_id"].astype(str).to_numpy()
    cell_types = selected["assigned_cluster"].astype(str).to_numpy()

    metadata = {
        "dataset": "baron_pancreas",
        "source_cells": int(len(combined)),
        "selected_cells": int(n_cells),
        "source_genes": int(len(source_symbols)),
        "available_genes": int(available.sum()),
        "unavailable_genes": int((~available).sum()),
        "coverage_fraction": float(available.mean()),
        "mapping": "exact_unique_gene_symbol",
        "human_samples": [
            path.name
            for path in paths
        ],
    }

    return (
        counts,
        available,
        source_rows,
        cell_ids,
        cell_types,
        metadata,
    )


def write_panels(
    *,
    output_dir: Path,
    dataset: str,
    target: np.ndarray,
    available_gene_mask: np.ndarray,
    source_rows: np.ndarray,
    cell_ids: np.ndarray,
    cell_types: np.ndarray | None,
    seed: int,
    vocabulary_sha256: str,
    source_sha256: str,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)

    selected_frame = pd.DataFrame(
        {
            "row": source_rows,
            "cell_id": cell_ids,
        }
    )

    if cell_types is not None:
        selected_frame["cell_type"] = cell_types

    selected_frame.to_csv(
        output_dir / "selected_cells.csv",
        index=False,
    )

    panel_paths: list[Path] = []

    for rate in MASK_RATES:
        mask = build_mask(
            target,
            available_gene_mask,
            mask_rate=rate,
            seed=seed,
            dataset=dataset,
            source_rows=source_rows,
        )

        observed = target.copy()
        observed[mask] = 0.0

        label = int(round(100 * rate))
        destination = output_dir / f"mask{label}.npz"

        payload = {
            "x": observed.astype(np.float32, copy=False),
            "y": target.astype(np.float32, copy=False),
            "synthetic_mask": mask,
            "available_gene_mask": available_gene_mask,
            "cell_id": cell_ids,
            "row": source_rows,
            "dataset": np.asarray(dataset),
            "split": np.asarray("external"),
            "modality": np.asarray("sc"),
            "mask_rate": np.asarray(rate, dtype=np.float32),
            "seed": np.asarray(seed, dtype=np.int64),
            "vocabulary_sha256": np.asarray(vocabulary_sha256),
            "source_sha256": np.asarray(source_sha256),
        }

        if cell_types is not None:
            payload["cell_type"] = cell_types

        with destination.open("wb") as handle:
            np.savez_compressed(handle, **payload)

        panel_paths.append(destination)

        masked_per_cell = mask.sum(axis=1)

        print(
            f"panel={destination} "
            f"cells={target.shape[0]} "
            f"genes={target.shape[1]} "
            f"available={available_gene_mask.sum()} "
            f"masked={mask.sum()} "
            f"masked_per_cell_min={masked_per_cell.min()} "
            f"masked_per_cell_max={masked_per_cell.max()}",
            flush=True,
        )

    metadata = {
        **metadata,
        "normalization": "log1p(CP10K_after_frozen_vocabulary_projection)",
        "masking": (
            "For each selected cell, mask round(mask_rate * "
            "number_of_positive_available_genes), minimum 1, "
            "without replacement using deterministic per-cell BLAKE2b seed."
        ),
        "mask_rates": list(MASK_RATES),
        "seed": int(seed),
        "n_genes": int(target.shape[1]),
        "vocabulary_sha256": vocabulary_sha256,
        "source_sha256": source_sha256,
        "available_gene_indices": np.flatnonzero(
            available_gene_mask
        ).astype(int).tolist(),
        "unavailable_gene_indices": np.flatnonzero(
            ~available_gene_mask
        ).astype(int).tolist(),
    }

    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    hash_lines = []

    for path in [
        *panel_paths,
        output_dir / "selected_cells.csv",
        output_dir / "benchmark_metadata.json",
    ]:
        hash_lines.append(
            f"{file_sha256(path)}  {path.name}"
        )

    (output_dir / "benchmark_panels.sha256").write_text(
        "\n".join(hash_lines) + "\n",
        encoding="utf-8",
    )

    print(
        f"external_benchmark=PASS "
        f"dataset={dataset} "
        f"output={output_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build frozen external SC2 selective-repair benchmarks "
            "from canonical Zheng68K or Baron raw counts."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=["zheng68k", "baron_pancreas"],
        required=True,
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--raw-archive", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--gene-stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-cells", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if output_dir.exists():
        raise SystemExit(
            f"Output already exists: {output_dir}. "
            "External benchmark outputs are immutable; "
            "use a new versioned directory."
        )

    vocabulary_path = Path(args.vocabulary)
    gene_stats_path = Path(args.gene_stats)
    raw_root = Path(args.raw_root)
    raw_archive = Path(args.raw_archive)

    vocabulary = load_vocabulary(
        vocabulary_path,
        gene_stats_path,
    )

    vocabulary_sha256 = str(
        vocabulary["vocabulary_sha256"].iloc[0]
    )

    source_sha256 = file_sha256(raw_archive)

    if args.dataset == "zheng68k":
        (
            counts,
            available,
            source_rows,
            cell_ids,
            metadata,
        ) = load_zheng(
            raw_root,
            vocabulary,
            n_cells=args.n_cells,
            seed=args.seed,
        )

        cell_types = None

    else:
        (
            counts,
            available,
            source_rows,
            cell_ids,
            cell_types,
            metadata,
        ) = load_baron(
            raw_root,
            vocabulary,
            n_cells=args.n_cells,
            seed=args.seed,
        )

    if counts.shape != (args.n_cells, len(vocabulary)):
        raise AssertionError(
            f"Unexpected projected matrix shape {counts.shape}"
        )

    if not np.all(np.isfinite(counts)):
        raise ValueError("Projected counts contain non-finite values")

    target = cp10k_log1p(counts)

    if np.any(target[:, ~available] != 0):
        raise AssertionError(
            "Unavailable external genes are not zero after projection"
        )

    if np.any(target.sum(axis=1) <= 0):
        raise ValueError(
            "At least one selected external cell has no mapped expression"
        )

    write_panels(
        output_dir=output_dir,
        dataset=args.dataset,
        target=target,
        available_gene_mask=available,
        source_rows=source_rows,
        cell_ids=cell_ids,
        cell_types=cell_types,
        seed=args.seed,
        vocabulary_sha256=vocabulary_sha256,
        source_sha256=source_sha256,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
