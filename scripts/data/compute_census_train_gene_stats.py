#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from sc2.data.census_pipeline import file_sha256, load_yaml, require_columns


def _axis_joinids(frame: pd.DataFrame) -> np.ndarray:
    if "soma_joinid" in frame.columns:
        return frame["soma_joinid"].to_numpy(dtype=np.int64, copy=False)
    return frame.index.to_numpy(dtype=np.int64, copy=False)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _save_arrays(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _cp10k_log1p(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    counts = counts.astype(np.float64, copy=False)
    totals = np.asarray(counts.sum(axis=1)).reshape(-1)
    scales = np.divide(10000.0, totals, out=np.zeros_like(totals, dtype=np.float64), where=totals > 0)
    normalized = sparse.diags(scales) @ counts
    normalized = normalized.tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream raw Census counts for selected training cells and compute train-only log1p(CP10K) gene statistics"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--selected-cells", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--save-every-chunks", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    census_release = str(config["census_release"])
    organism = str(config.get("species", "Homo sapiens"))
    selected_path = Path(args.selected_cells)
    selected_hash = file_sha256(selected_path)
    selected = pd.read_parquet(selected_path)
    require_columns(selected, ["soma_joinid", "split"], name="selected cells")
    train = selected[selected["split"].astype(str) == "train"].sort_values("soma_joinid").reset_index(drop=True)
    if train.empty:
        raise ValueError("No training cells in selected plan")
    if train["soma_joinid"].duplicated().any():
        raise ValueError("Duplicate training soma_joinid")

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_json = state_dir / "state.json"
    state_arrays = state_dir / "accumulators.npz"

    try:
        import cellxgene_census
    except ImportError as error:
        raise SystemExit("cellxgene-census is required in the active environment") from error

    with cellxgene_census.open_soma(census_version=census_release) as census:
        experiment = census["census_data"]["homo_sapiens"]
        var = experiment.ms["RNA"].var.read(
            column_names=["soma_joinid", "feature_id", "feature_name"]
        ).concat().to_pandas()
        require_columns(var, ["soma_joinid", "feature_id", "feature_name"], name="Census var")
        var = var.sort_values("soma_joinid").reset_index(drop=True)
        if var["feature_id"].duplicated().any():
            raise ValueError("Census feature_id is not unique")
        var_coords = var["soma_joinid"].to_numpy(dtype=np.int64)
        feature_ids = var["feature_id"].astype(str).to_numpy()
        feature_names = var["feature_name"].astype(str).to_numpy()
        n_genes = len(var)
        import hashlib
        digest = hashlib.sha256()
        for joinid, feature_id in zip(var_coords.tolist(), feature_ids.tolist(), strict=True):
            digest.update(f"{int(joinid)}\t{feature_id}\n".encode("utf-8"))
        var_hash = digest.hexdigest()

        start_offset = 0
        detected = np.zeros(n_genes, dtype=np.int64)
        sum_log = np.zeros(n_genes, dtype=np.float64)
        sumsq_log = np.zeros(n_genes, dtype=np.float64)
        processed_cells = 0

        if args.resume and state_json.exists() and state_arrays.exists():
            state = json.loads(state_json.read_text(encoding="utf-8"))
            expected = {
                "census_release": census_release,
                "selected_cells_sha256": selected_hash,
                "var_sha256": var_hash,
                "n_genes": n_genes,
                "train_cells": len(train),
                "chunk_size": args.chunk_size,
            }
            for key, value in expected.items():
                if state.get(key) != value:
                    raise ValueError(f"Resume state mismatch for {key}: {state.get(key)!r} != {value!r}")
            arrays = np.load(state_arrays)
            detected = arrays["detected"].astype(np.int64, copy=False)
            sum_log = arrays["sum_log"].astype(np.float64, copy=False)
            sumsq_log = arrays["sumsq_log"].astype(np.float64, copy=False)
            start_offset = int(state["next_offset"])
            processed_cells = int(state["processed_cells"])
            print(f"resumed_gene_stats next_offset={start_offset} processed_cells={processed_cells}")

        chunk_number = start_offset // args.chunk_size
        for start in range(start_offset, len(train), args.chunk_size):
            block = train.iloc[start : start + args.chunk_size]
            requested = block["soma_joinid"].to_numpy(dtype=np.int64)
            adata = cellxgene_census.get_anndata(
                census,
                organism=organism,
                measurement_name="RNA",
                X_name="raw",
                obs_coords=requested,
                obs_column_names=["soma_joinid"],
                var_column_names=["feature_id", "feature_name", "soma_joinid"],
            )
            returned_rows = _axis_joinids(adata.obs)
            if set(returned_rows.tolist()) != set(requested.tolist()):
                raise ValueError("Census returned a different set of training cells than requested")
            returned_features = adata.var["feature_id"].astype(str).to_numpy()
            if len(returned_features) != n_genes:
                raise ValueError(f"Census returned {len(returned_features)} genes; expected {n_genes}")
            positions = pd.Index(returned_features).get_indexer(feature_ids)
            if np.any(positions < 0):
                raise ValueError("Census chunk is missing expected genes")
            counts = sparse.csr_matrix(adata.X)[:, positions]
            if np.any(counts.data < 0) or np.any(np.abs(counts.data - np.rint(counts.data)) > 1.0e-4):
                raise ValueError("Census raw layer contains negative or non-integer values")
            log1p = _cp10k_log1p(counts)
            detected += np.asarray((counts > 0).sum(axis=0)).reshape(-1).astype(np.int64)
            sum_log += np.asarray(log1p.sum(axis=0)).reshape(-1)
            sumsq_log += np.asarray(log1p.power(2).sum(axis=0)).reshape(-1)
            processed_cells += len(block)
            next_offset = start + len(block)
            chunk_number += 1

            if chunk_number % args.save_every_chunks == 0 or next_offset == len(train):
                _save_arrays(
                    state_arrays,
                    detected=detected,
                    sum_log=sum_log,
                    sumsq_log=sumsq_log,
                )
                _atomic_json(
                    {
                        "census_release": census_release,
                        "selected_cells_sha256": selected_hash,
                        "var_sha256": var_hash,
                        "n_genes": n_genes,
                        "train_cells": len(train),
                        "chunk_size": args.chunk_size,
                        "next_offset": next_offset,
                        "processed_cells": processed_cells,
                    },
                    state_json,
                )
            print(f"gene_stats_cells={next_offset}/{len(train)}", flush=True)

    if processed_cells != len(train):
        raise AssertionError(f"Processed {processed_cells} cells, expected {len(train)}")
    mean = sum_log / float(processed_cells)
    variance = np.maximum(sumsq_log / float(processed_cells) - mean**2, 0.0)
    output = pd.DataFrame(
        {
            "soma_joinid": var_coords,
            "ensembl_id": feature_ids,
            "feature_name": feature_names,
            "detected_cells": detected,
            "mean_log1p": mean,
            "variance_log1p": variance,
        }
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    output.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    summary = {
        "census_release": census_release,
        "train_cells": int(processed_cells),
        "genes": int(n_genes),
        "selected_cells_sha256": selected_hash,
        "var_sha256": var_hash,
        "output_sha256": file_sha256(destination),
    }
    destination.with_name("train_gene_stats_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"train_gene_stats=ok output={destination} genes={n_genes} cells={processed_cells}")


if __name__ == "__main__":
    main()
