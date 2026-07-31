#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sc2.data.census_pipeline import VALID_SPLITS, file_sha256, load_yaml, require_columns
from sc2.data.csr_shard import shard_directory_sha256, write_csr


def _axis_joinids(frame: pd.DataFrame) -> np.ndarray:
    if "soma_joinid" in frame.columns:
        return frame["soma_joinid"].to_numpy(dtype=np.int64, copy=False)
    return frame.index.to_numpy(dtype=np.int64, copy=False)


def cp10k_log1p(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.astype(np.float32, copy=True)
    totals = np.asarray(matrix.sum(axis=1)).reshape(-1)
    scales = np.divide(10000.0, totals, out=np.zeros_like(totals, dtype=np.float32), where=totals > 0)
    normalized = sparse.diags(scales) @ matrix
    normalized = normalized.tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def _fetch_counts(
    census: object,
    *,
    organism: str,
    rows: pd.DataFrame,
    gene_ids: list[str],
    gene_coords: np.ndarray,
    fetch_chunk_size: int,
) -> sparse.csr_matrix:
    import cellxgene_census

    parts: list[sparse.csr_matrix] = []
    for start in range(0, len(rows), fetch_chunk_size):
        block = rows.iloc[start : start + fetch_chunk_size]
        requested_rows = block["soma_joinid"].to_numpy(dtype=np.int64)
        adata = cellxgene_census.get_anndata(
            census,
            organism=organism,
            measurement_name="RNA",
            X_name="raw",
            obs_coords=requested_rows,
            var_coords=gene_coords,
            obs_column_names=["soma_joinid"],
            var_column_names=["feature_id", "feature_name", "soma_joinid"],
        )
        returned_rows = _axis_joinids(adata.obs)
        row_positions = pd.Index(returned_rows).get_indexer(requested_rows)
        if np.any(row_positions < 0):
            raise ValueError("Census shard query did not return all requested cells")
        returned_genes = adata.var["feature_id"].astype(str).tolist()
        gene_positions = pd.Index(returned_genes).get_indexer(gene_ids)
        if np.any(gene_positions < 0):
            missing = [gene_ids[index] for index, value in enumerate(gene_positions) if value < 0]
            raise ValueError(f"Census shard query is missing vocabulary genes: {missing[:20]}")
        matrix = sparse.csr_matrix(adata.X)[row_positions][:, gene_positions]
        if np.any(matrix.data < 0) or np.any(np.abs(matrix.data - np.rint(matrix.data)) > 1.0e-4):
            raise ValueError("Census raw matrix contains negative or non-integer values")
        matrix.data = np.rint(matrix.data).astype(np.uint32)
        parts.append(matrix.tocsr())
    return sparse.vstack(parts, format="csr")


def _existing_manifest_row(shard_dir: Path) -> dict[str, object]:
    meta = json.loads((shard_dir / "meta.json").read_text(encoding="utf-8"))
    return {
        "shard_id": str(meta["shard_id"]),
        "path": str(shard_dir.resolve()),
        "split": str(meta["split"]),
        "modality": str(meta["modality"]),
        "n_rows": int(meta["n_rows"]),
        "n_genes": int(meta["n_genes"]),
        "sha256": shard_directory_sha256(shard_dir),
        "gene_vocab_sha256": str(meta["gene_vocab_sha256"]),
        "census_release": str(meta["census_release"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize selected Census cells directly into immutable fixed-vocabulary SC2 CSR shards"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--selected-cells", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cells-per-shard", type=int, default=None)
    parser.add_argument("--fetch-chunk-size", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    census_release = str(config["census_release"])
    organism = str(config.get("species", "Homo sapiens"))
    cells_per_shard = int(args.cells_per_shard or config.get("cells_per_shard", 25000))

    selected_path = Path(args.selected_cells)
    selected = pd.read_parquet(selected_path)
    require_columns(
        selected,
        ["cell_id", "soma_joinid", "dataset_id", "donor_id", "tissue", "cell_type", "assay", "split"],
        name="selected cells",
    )
    if selected["cell_id"].duplicated().any() or selected["soma_joinid"].duplicated().any():
        raise ValueError("Selected cells contain duplicate IDs")
    vocabulary_path = Path(args.vocabulary)
    vocabulary = pd.read_parquet(vocabulary_path).sort_values("gene_index")
    require_columns(vocabulary, ["gene_index", "ensembl_id", "vocabulary_sha256"], name="vocabulary")
    if vocabulary["gene_index"].tolist() != list(range(len(vocabulary))):
        raise ValueError("Vocabulary gene_index must be contiguous and start at zero")
    if vocabulary["vocabulary_sha256"].nunique() != 1:
        raise ValueError("Vocabulary contains multiple hashes")
    gene_ids = vocabulary["ensembl_id"].astype(str).tolist()
    vocab_hash = str(vocabulary["vocabulary_sha256"].iloc[0])

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output_root.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {output_root}. Use --resume only for a verified interrupted build, "
            "or choose a new versioned directory."
        )

    try:
        import cellxgene_census
    except ImportError as error:
        raise SystemExit("cellxgene-census is required in the active environment") from error

    manifest_rows: list[dict[str, object]] = []
    with cellxgene_census.open_soma(census_version=census_release) as census:
        var = census["census_data"]["homo_sapiens"].ms["RNA"].var.read(
            column_names=["soma_joinid", "feature_id", "feature_name"]
        ).concat().to_pandas()
        require_columns(var, ["soma_joinid", "feature_id"], name="Census var")
        if var["feature_id"].duplicated().any():
            raise ValueError("Census feature_id is not unique")
        mapping = var.assign(feature_id=var["feature_id"].astype(str)).set_index("feature_id")["soma_joinid"]
        gene_coords_series = mapping.reindex(gene_ids)
        if gene_coords_series.isna().any():
            missing = [gene_ids[index] for index, value in enumerate(gene_coords_series.isna()) if value]
            raise ValueError(f"Census release is missing vocabulary genes: {missing[:20]}")
        gene_coords = gene_coords_series.to_numpy(dtype=np.int64)

        for split in VALID_SPLITS:
            split_rows = selected[selected["split"].astype(str) == split].sort_values(
                ["candidate_rank", "soma_joinid"], kind="mergesort"
            ).reset_index(drop=True)
            if split_rows.empty:
                raise ValueError(f"No selected cells for split={split}")
            for shard_number, start in enumerate(range(0, len(split_rows), cells_per_shard)):
                rows = split_rows.iloc[start : start + cells_per_shard].copy()
                shard_id = f"sc_{split}_{shard_number:05d}"
                shard_dir = output_root / shard_id
                if shard_dir.exists():
                    if not args.resume:
                        raise FileExistsError(shard_dir)
                    row = _existing_manifest_row(shard_dir)
                    if (
                        row["shard_id"] != shard_id
                        or row["split"] != split
                        or row["n_rows"] != len(rows)
                        or row["n_genes"] != len(gene_ids)
                        or row["gene_vocab_sha256"] != vocab_hash
                        or row["census_release"] != census_release
                    ):
                        raise ValueError(f"Existing shard is incompatible with requested build: {shard_dir}")
                    manifest_rows.append(row)
                    print(f"resume_skip={shard_id} rows={len(rows)}")
                    continue

                temporary = output_root / f".{shard_id}.tmp"
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir(parents=True, exist_ok=False)
                try:
                    counts = _fetch_counts(
                        census,
                        organism=organism,
                        rows=rows,
                        gene_ids=gene_ids,
                        gene_coords=gene_coords,
                        fetch_chunk_size=args.fetch_chunk_size,
                    )
                    if counts.shape != (len(rows), len(gene_ids)):
                        raise AssertionError(f"Unexpected counts shape {counts.shape}")
                    log1p = cp10k_log1p(counts)
                    write_csr(temporary, "counts", counts)
                    write_csr(temporary, "log1p", log1p)
                    obs_columns = [
                        "cell_id",
                        "soma_joinid",
                        "dataset_id",
                        "donor_id",
                        "tissue",
                        "cell_type",
                        "assay",
                        "split_group",
                        "split",
                        "mito_fraction",
                        "nnz",
                        "raw_sum",
                    ]
                    available = [column for column in obs_columns if column in rows.columns]
                    rows[available].to_parquet(temporary / "obs.parquet", index=False)
                    meta = {
                        "schema_version": "sc2-csr-shard-v1",
                        "shard_id": shard_id,
                        "census_release": census_release,
                        "split": split,
                        "modality": "sc",
                        "n_rows": int(len(rows)),
                        "n_genes": int(len(gene_ids)),
                        "gene_vocab_sha256": vocab_hash,
                        "normalization": "log1p(CP10K)",
                        "counts_dtype": "uint32",
                        "selected_cells_sha256": file_sha256(selected_path),
                        "vocabulary_file_sha256": file_sha256(vocabulary_path),
                    }
                    (temporary / "meta.json").write_text(
                        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    os.replace(temporary, shard_dir)
                except Exception:
                    if temporary.exists():
                        shutil.rmtree(temporary)
                    raise
                digest = shard_directory_sha256(shard_dir)
                manifest_rows.append(
                    {
                        "shard_id": shard_id,
                        "path": str(shard_dir.resolve()),
                        "split": split,
                        "modality": "sc",
                        "n_rows": int(len(rows)),
                        "n_genes": int(len(gene_ids)),
                        "sha256": digest,
                        "gene_vocab_sha256": vocab_hash,
                        "census_release": census_release,
                    }
                )
                print(f"wrote={shard_id} rows={len(rows)} sha256={digest}", flush=True)

    manifest = pd.DataFrame(manifest_rows).sort_values(["split", "shard_id"]).reset_index(drop=True)
    canonical_manifest = output_root.parent / "shards.parquet"
    modality_manifest = output_root.parent / "sc_shards.parquet"
    for destination in (canonical_manifest, modality_manifest):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        manifest.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    summary = {
        "census_release": census_release,
        "selected_cells": int(manifest["n_rows"].sum()),
        "shards": int(len(manifest)),
        "genes": int(len(gene_ids)),
        "vocabulary_sha256": vocab_hash,
        "manifest_sha256": file_sha256(canonical_manifest),
    }
    (output_root.parent / "materialization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"materialization=ok manifest={canonical_manifest} shards={len(manifest)}")


if __name__ == "__main__":
    main()
