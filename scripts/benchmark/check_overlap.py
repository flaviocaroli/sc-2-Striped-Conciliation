#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from sc2.data.census_pipeline import (
    dataset_matches_registry,
    registry_exclusions,
    validate_registry_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail on benchmark/corpus dataset, collection, accession or alias overlap")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--planned-cells", required=True, help="Parquet with dataset and collection metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--census-release", default=None)
    args = parser.parse_args()

    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Registry YAML must contain a mapping")
    errors = validate_registry_payload(registry, expected_release=args.census_release)
    if errors:
        raise SystemExit("Benchmark registry is not frozen:\n" + "\n".join(f"- {value}" for value in errors))
    frame = pd.read_parquet(args.planned_cells)
    searchable_columns = [
        name
        for name in ("dataset_id", "collection_id", "dataset_title", "collection_name", "collection_doi", "citation")
        if name in frame.columns
    ]
    if not searchable_columns:
        raise ValueError("Planned cells contain no searchable dataset identifiers")
    exclusions = registry_exclusions(registry)
    overlaps = []
    for row in frame[searchable_columns].drop_duplicates().to_dict(orient="records"):
        matches = dataset_matches_registry(row, exclusions)
        if matches:
            overlaps.append({**row, "matches": matches})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(overlaps, indent=2, sort_keys=True), encoding="utf-8")
    if overlaps:
        raise SystemExit(f"benchmark_overlap=FAIL n={len(overlaps)} details={output}")
    print("benchmark_overlap=ok")


if __name__ == "__main__":
    main()
