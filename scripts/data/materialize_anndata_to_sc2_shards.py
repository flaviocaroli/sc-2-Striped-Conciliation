#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from sc2.data.csr_shard import shard_directory_sha256, write_csr


def cp10k_log1p(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.astype(np.float32, copy=True)
    totals = np.asarray(matrix.sum(axis=1)).reshape(-1)
    scales = np.divide(10000.0, totals, out=np.zeros_like(totals, dtype=np.float32), where=totals > 0)
    normalized = sparse.diags(scales) @ matrix
    normalized.data = np.log1p(normalized.data)
    return normalized.tocsr()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a split AnnData file into immutable SC2 CSR shards")
    parser.add_argument("--source-h5ad", required=True)
    parser.add_argument("--vocabulary", required=True, help="Parquet with gene_index and ensembl_id")
    parser.add_argument("--split-manifest", required=True, help="Parquet with cell_id and split")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--census-release", required=True)
    parser.add_argument("--modality", choices=["sc", "bulk", "pseudobulk"], default="sc")
    parser.add_argument("--cells-per-shard", type=int, default=25000)
    parser.add_argument("--counts-layer", default=None, help="AnnData layer containing raw counts; default X")
    args = parser.parse_args()

    vocabulary = pd.read_parquet(args.vocabulary).sort_values("gene_index")
    genes = vocabulary["ensembl_id"].astype(str).tolist()
    vocab_hash = str(vocabulary["vocabulary_sha256"].iloc[0])
    splits = pd.read_parquet(args.split_manifest)
    if splits["cell_id"].duplicated().any():
        raise ValueError("split manifest contains duplicate cell_id")
    split_map = splits.set_index("cell_id")["split"].astype(str)
    adata = ad.read_h5ad(args.source_h5ad, backed="r")
    var_ids = pd.Index(adata.var_names.astype(str))
    positions = var_ids.get_indexer(genes)
    missing = [genes[index] for index, position in enumerate(positions) if position < 0]
    if missing:
        raise ValueError(f"Source AnnData is missing {len(missing)} vocabulary genes; first={missing[:10]}")
    cell_ids = pd.Index(adata.obs_names.astype(str))
    selected_splits = split_map.reindex(cell_ids)
    if selected_splits.isna().any():
        raise ValueError(f"{int(selected_splits.isna().sum())} AnnData cells are missing split assignments")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for split in ("train", "validation", "test"):
        row_indices = np.flatnonzero(selected_splits.to_numpy() == split)
        for shard_number, start in enumerate(range(0, row_indices.size, args.cells_per_shard)):
            rows = row_indices[start : start + args.cells_per_shard]
            if rows.size == 0:
                continue
            source = adata.layers[args.counts_layer] if args.counts_layer else adata.X
            counts = source[rows, positions]
            counts = sparse.csr_matrix(counts)
            if np.any(counts.data < 0) or np.any(np.abs(counts.data - np.rint(counts.data)) > 1.0e-4):
                raise ValueError("Raw-count matrix contains negative or non-integer values")
            counts.data = np.rint(counts.data).astype(np.uint32)
            log1p = cp10k_log1p(counts)
            shard_id = f"{args.modality}_{split}_{shard_number:05d}"
            shard_dir = output_root / shard_id
            shard_dir.mkdir(parents=True, exist_ok=False)
            write_csr(shard_dir, "counts", counts)
            write_csr(shard_dir, "log1p", log1p)
            obs = adata.obs.iloc[rows].copy()
            obs.insert(0, "cell_id", cell_ids[rows].astype(str))
            obs["split"] = split
            required = ["cell_id", "dataset_id", "donor_id", "tissue", "cell_type", "split"]
            for column in required:
                if column not in obs:
                    raise ValueError(f"AnnData obs missing required metadata column: {column}")
            obs[required].to_parquet(shard_dir / "obs.parquet", index=False)
            meta = {
                "schema_version": "sc2-csr-shard-v1",
                "shard_id": shard_id,
                "census_release": args.census_release,
                "split": split,
                "modality": args.modality,
                "n_rows": int(rows.size),
                "n_genes": len(genes),
                "gene_vocab_sha256": vocab_hash,
                "normalization": "log1p(CP10K)",
                "counts_dtype": "uint32",
            }
            (shard_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
            digest = shard_directory_sha256(shard_dir)
            manifest_rows.append({
                "shard_id": shard_id,
                "path": str(shard_dir.resolve()),
                "split": split,
                "modality": args.modality,
                "n_rows": int(rows.size),
                "n_genes": len(genes),
                "sha256": digest,
                "gene_vocab_sha256": vocab_hash,
                "census_release": args.census_release,
            })
            print(f"wrote={shard_id} rows={rows.size} sha256={digest}")
    manifest = output_root.parent / f"{args.modality}_shards.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest, index=False)
    print(f"manifest={manifest} shards={len(manifest_rows)}")


if __name__ == "__main__":
    main()
