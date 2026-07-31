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

from sc2.data.census_pipeline import (
    VALID_SPLITS,
    cap_violations,
    file_sha256,
    largest_remainder_quotas,
    load_yaml,
    require_columns,
    split_counts,
)


def _axis_joinids(frame: pd.DataFrame) -> np.ndarray:
    if "soma_joinid" in frame.columns:
        return frame["soma_joinid"].to_numpy(dtype=np.int64, copy=False)
    try:
        return frame.index.to_numpy(dtype=np.int64, copy=False)
    except Exception as error:
        raise ValueError("AnnData axis does not expose soma_joinid") from error


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute mitochondrial fractions for the candidate plan and select exact split quotas"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--planned-cells", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qc-output", default=None)
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()

    config = load_yaml(args.config)
    census_release = str(config["census_release"])
    organism = str(config.get("species", "Homo sapiens"))
    target_cells = int(config["target_cells"])
    max_mito_fraction = float(config.get("quality_control", {}).get("max_mito_fraction", 0.20))
    fractions = {key: float(value) for key, value in config["splits"]["fractions"].items()}
    final_quotas = largest_remainder_quotas(target_cells, fractions)
    caps = {key: int(value) for key, value in config.get("sampling_caps", {}).items()}

    planned_path = Path(args.planned_cells)
    planned = pd.read_parquet(planned_path)
    require_columns(
        planned,
        [
            "soma_joinid",
            "cell_id",
            "dataset_id",
            "donor_id",
            "tissue",
            "cell_type",
            "assay",
            "split_group",
            "split",
            "candidate_rank",
            "raw_sum",
            "nnz",
        ],
        name="planned cells",
    )
    if planned["soma_joinid"].duplicated().any():
        raise ValueError("planned cells contain duplicate soma_joinid")
    if set(planned["split"].astype(str)) - set(VALID_SPLITS):
        raise ValueError("planned cells contain invalid split labels")

    try:
        import cellxgene_census
    except ImportError as error:
        raise SystemExit("cellxgene-census is required in the active environment") from error

    mito_by_joinid: dict[int, float] = {}
    with cellxgene_census.open_soma(census_version=census_release) as census:
        experiment = census["census_data"]["homo_sapiens"]
        var = experiment.ms["RNA"].var.read(
            column_names=["soma_joinid", "feature_id", "feature_name"]
        ).concat().to_pandas()
        require_columns(var, ["soma_joinid", "feature_id", "feature_name"], name="Census var")
        mito = var[var["feature_name"].astype(str).str.upper().str.startswith("MT-")].copy()
        if mito.empty:
            raise ValueError("No mitochondrial genes with feature_name prefix 'MT-' were found")
        mito_coords = mito["soma_joinid"].to_numpy(dtype=np.int64)
        print(f"mitochondrial_genes={len(mito_coords)}")

        ordered = planned.sort_values("soma_joinid").reset_index(drop=True)
        for start in range(0, len(ordered), args.chunk_size):
            block = ordered.iloc[start : start + args.chunk_size]
            requested = block["soma_joinid"].to_numpy(dtype=np.int64)
            adata = cellxgene_census.get_anndata(
                census,
                organism=organism,
                measurement_name="RNA",
                X_name="raw",
                obs_coords=requested,
                var_coords=mito_coords,
                obs_column_names=["soma_joinid"],
                var_column_names=["feature_id", "feature_name", "soma_joinid"],
            )
            matrix = sparse.csr_matrix(adata.X)
            returned = _axis_joinids(adata.obs)
            if len(returned) != len(requested):
                raise ValueError(f"Census returned {len(returned)} rows for {len(requested)} requested cells")
            mito_counts = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float64)
            raw_sum_map = block.set_index("soma_joinid")["raw_sum"].astype(float)
            totals = raw_sum_map.reindex(returned).to_numpy(dtype=np.float64)
            if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
                raise ValueError("Invalid raw_sum encountered during mitochondrial QC")
            fractions_chunk = mito_counts / totals
            for joinid, fraction in zip(returned.tolist(), fractions_chunk.tolist(), strict=True):
                mito_by_joinid[int(joinid)] = float(fraction)
            print(f"mito_qc_cells={min(start + len(block), len(ordered))}/{len(ordered)}", flush=True)

    qc = planned.copy()
    qc["mito_fraction"] = qc["soma_joinid"].map(mito_by_joinid)
    if qc["mito_fraction"].isna().any():
        raise ValueError(f"Missing mitochondrial fractions for {int(qc['mito_fraction'].isna().sum())} cells")
    qc["passes_mito_qc"] = qc["mito_fraction"] <= max_mito_fraction
    qc_output = Path(args.qc_output) if args.qc_output else Path(args.output).with_name("candidate_cell_qc.parquet")
    _atomic_parquet(qc, qc_output)

    passing = qc[qc["passes_mito_qc"]].copy()
    selected_parts: list[pd.DataFrame] = []
    shortages: dict[str, int] = {}
    for split in VALID_SPLITS:
        candidates = passing[passing["split"].astype(str) == split].sort_values(
            ["candidate_rank", "cell_priority", "soma_joinid"], kind="mergesort"
        )
        needed = final_quotas[split]
        if len(candidates) < needed:
            shortages[split] = needed - len(candidates)
        selected_parts.append(candidates.head(needed))
    if shortages:
        raise SystemExit(
            f"Mitochondrial QC left insufficient cells for exact split quotas: {shortages}. "
            "Increase planning.candidate_multiplier and rerun plan-only."
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(["split", "candidate_rank"], kind="mergesort").reset_index(drop=True)
    if len(selected) != target_cells:
        raise AssertionError(f"Selected {len(selected)} cells, expected {target_cells}")
    if selected["soma_joinid"].duplicated().any() or selected["cell_id"].duplicated().any():
        raise ValueError("Selected plan contains duplicate cells")
    crossing = selected.groupby("split_group")["split"].nunique()
    if not crossing.empty and int(crossing.max()) != 1:
        raise ValueError("A donor/study split_group crosses train/validation/test")
    violations = cap_violations(selected, caps)
    if violations:
        raise ValueError("Sampling cap violations after QC: " + " | ".join(violations))

    output = Path(args.output)
    _atomic_parquet(selected, output)
    split_manifest = selected[["cell_id", "soma_joinid", "dataset_id", "donor_id", "split_group", "split"]].copy()
    _atomic_parquet(split_manifest, output.with_name("cell_splits.parquet"))

    summary = {
        "census_release": census_release,
        "candidate_cells": int(len(qc)),
        "mito_pass_cells": int(qc["passes_mito_qc"].sum()),
        "mito_fail_cells": int((~qc["passes_mito_qc"]).sum()),
        "max_mito_fraction": max_mito_fraction,
        "selected_cells": int(len(selected)),
        "split_counts": split_counts(selected),
        "split_quotas": final_quotas,
        "datasets": int(selected["dataset_id"].nunique()),
        "donor_groups": int(selected["split_group"].nunique()),
        "tissues": int(selected["tissue"].nunique()),
        "planned_cells_sha256": file_sha256(planned_path),
        "candidate_qc_sha256": file_sha256(qc_output),
        "selected_cells_sha256": file_sha256(output),
    }
    summary_path = output.with_name("selected_cells_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"selected_cells=ok path={output} rows={len(selected)} split_counts={summary['split_counts']}")


if __name__ == "__main__":
    main()
