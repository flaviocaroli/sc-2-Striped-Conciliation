#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def normalized(value: object) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail on benchmark/corpus identifier or alias overlap")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--planned-cells", required=True, help="Parquet with dataset_id and optional title/accession")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.planned_cells)
    searchable_columns = [name for name in ("dataset_id", "collection_id", "accession", "dataset_title", "collection_name") if name in frame]
    if not searchable_columns:
        raise ValueError("Planned cells contain no searchable dataset identifiers")
    banned = set()
    for benchmark in registry.get("benchmarks", []):
        for value in benchmark.get("accessions", []) + benchmark.get("aliases", []):
            if str(value).strip():
                banned.add(normalized(value))
    overlaps = []
    for column in searchable_columns:
        for value in frame[column].dropna().astype(str).unique():
            value_norm = normalized(value)
            matched = [item for item in banned if item and (item == value_norm or item in value_norm or value_norm in item)]
            if matched:
                overlaps.append({"column": column, "value": value, "matched_registry_tokens": sorted(matched)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(overlaps, indent=2), encoding="utf-8")
    if overlaps:
        raise SystemExit(f"benchmark_overlap=FAIL n={len(overlaps)} details={output}")
    print("benchmark_overlap=ok")


if __name__ == "__main__":
    main()
