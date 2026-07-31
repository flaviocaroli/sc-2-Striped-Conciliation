#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from sc2.data.census_pipeline import (
    VALID_SPLITS,
    cap_violations,
    dataset_matches_registry,
    file_sha256,
    largest_remainder_quotas,
    load_yaml,
    nonempty_string_mask,
    registry_exclusions,
    require_columns,
    split_counts,
    validate_registry_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a planned or selected SC2 Census corpus")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--stage", choices=["candidate", "selected"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    census_release = str(config["census_release"])
    registry_path = Path(args.registry or config["leakage"]["benchmark_registry"])
    if not registry_path.is_absolute():
        registry_path = (Path.cwd() / registry_path).resolve()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry_errors = validate_registry_payload(registry, expected_release=census_release)
    exclusions = registry_exclusions(registry)

    cells_path = Path(args.cells)
    cells = pd.read_parquet(cells_path)
    required = [
        "soma_joinid",
        "cell_id",
        "dataset_id",
        "collection_id",
        "dataset_title",
        "collection_name",
        "donor_id",
        "tissue",
        "cell_type",
        "assay",
        "split_group",
        "split",
        "nnz",
        "raw_sum",
        "census_release",
    ]
    require_columns(cells, required, name="corpus cells")
    errors: list[str] = []
    warnings: list[str] = []

    if registry_errors:
        errors.extend(f"registry: {value}" for value in registry_errors)
    if cells.empty:
        errors.append("cell table is empty")
    if cells["soma_joinid"].duplicated().any():
        errors.append(f"duplicate soma_joinid: {int(cells['soma_joinid'].duplicated().sum())}")
    if cells["cell_id"].duplicated().any():
        errors.append(f"duplicate cell_id: {int(cells['cell_id'].duplicated().sum())}")
    releases = set(cells["census_release"].astype(str))
    if releases != {census_release}:
        errors.append(f"cell table release values {sorted(releases)} differ from config {census_release}")
    invalid_splits = set(cells["split"].astype(str)) - set(VALID_SPLITS)
    if invalid_splits:
        errors.append(f"invalid split labels: {sorted(invalid_splits)}")
    crossing = cells.groupby("split_group")["split"].nunique()
    if not crossing.empty and int(crossing.max()) > 1:
        errors.append(f"split_group leakage: {int((crossing > 1).sum())} groups cross splits")
    for column in config.get("required_metadata", []):
        if column not in cells.columns:
            errors.append(f"required metadata missing: {column}")
        else:
            missing = int((~nonempty_string_mask(cells[column])).sum())
            if missing:
                errors.append(f"required metadata {column} has {missing} missing/unknown values")
    min_genes = int(config.get("quality_control", {}).get("min_detected_genes", 0))
    below = int((cells["nnz"].astype(float) < min_genes).sum())
    if below:
        errors.append(f"{below} cells have nnz below min_detected_genes={min_genes}")
    invalid_depth = int((cells["raw_sum"].astype(float) <= 0).sum())
    if invalid_depth:
        errors.append(f"{invalid_depth} cells have non-positive raw_sum")
    if args.stage == "selected":
        if "mito_fraction" not in cells.columns:
            errors.append("selected-stage audit requires mito_fraction")
        else:
            max_mito = float(config.get("quality_control", {}).get("max_mito_fraction", 1.0))
            failures = int((cells["mito_fraction"].astype(float) > max_mito).sum())
            if failures:
                errors.append(f"{failures} selected cells exceed max_mito_fraction={max_mito}")
        target = int(config["target_cells"])
        if len(cells) != target:
            errors.append(f"selected cell count={len(cells)} differs from target={target}")
        expected = largest_remainder_quotas(
            target, {key: float(value) for key, value in config["splits"]["fractions"].items()}
        )
        actual = split_counts(cells)
        if actual != expected:
            errors.append(f"selected split counts={actual} differ from quotas={expected}")
    else:
        if "mito_fraction" not in cells.columns:
            warnings.append("candidate plan has no mito_fraction yet; run finalize_census_plan_qc.py")

    errors.extend(cap_violations(cells, config.get("sampling_caps", {})))

    dataset_table = cells[
        ["dataset_id", "collection_id", "dataset_title", "collection_name"]
    ].drop_duplicates()
    overlap_rows = []
    for row in dataset_table.to_dict(orient="records"):
        matches = dataset_matches_registry(row, exclusions)
        if matches:
            overlap_rows.append({**row, "matches": matches})
    if overlap_rows:
        errors.append(f"benchmark overlap detected in {len(overlap_rows)} dataset/collection rows")

    report = {
        "stage": args.stage,
        "cells_path": str(cells_path.resolve()),
        "cells_sha256": file_sha256(cells_path),
        "registry_path": str(registry_path),
        "registry_sha256": file_sha256(registry_path),
        "census_release": census_release,
        "rows": int(len(cells)),
        "split_counts": split_counts(cells),
        "datasets": int(cells["dataset_id"].nunique()),
        "donor_groups": int(cells["split_group"].nunique()),
        "tissues": int(cells["tissue"].nunique()),
        "cell_types": int(cells["cell_type"].nunique()),
        "overlaps": overlap_rows,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        print(f"corpus_audit=FAIL report={output}")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(2)
    print(f"corpus_audit=ok report={output}")
    for warning in warnings:
        print(f"warning={warning}")


if __name__ == "__main__":
    main()
