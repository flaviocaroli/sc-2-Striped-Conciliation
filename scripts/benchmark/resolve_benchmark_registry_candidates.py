#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sc2.data.census_pipeline import normalized

SEARCH_COLUMNS = (
    "dataset_id",
    "collection_id",
    "dataset_title",
    "collection_name",
    "collection_doi",
    "citation",
)


def _read_census_datasets(census_release: str) -> pd.DataFrame:
    try:
        import cellxgene_census
    except ImportError as error:
        raise SystemExit(
            "cellxgene-census is not installed. Install requirements-continuous.txt in the sc2-data environment."
        ) from error
    with cellxgene_census.open_soma(census_version=census_release) as census:
        frame = census["census_info"]["datasets"].read().concat().to_pandas()
    return frame


def _tokens(benchmark: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("name",):
        text = str(benchmark.get(key, "")).strip()
        if text:
            values.append(text)
    for key in ("accessions", "aliases"):
        for value in benchmark.get(key, []):
            text = str(value).strip()
            if text:
                values.append(text)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = normalized(value)
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find candidate Census dataset/collection IDs for each unresolved benchmark; review is mandatory"
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--census-release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets-cache", default=None, help="Optional cached Census datasets parquet")
    parser.add_argument("--max-candidates", type=int, default=50)
    args = parser.parse_args()

    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Registry YAML must contain a mapping")

    if args.datasets_cache and Path(args.datasets_cache).exists():
        datasets = pd.read_parquet(args.datasets_cache)
    else:
        datasets = _read_census_datasets(args.census_release)
        if args.datasets_cache:
            cache = Path(args.datasets_cache)
            cache.parent.mkdir(parents=True, exist_ok=True)
            datasets.to_parquet(cache, index=False)

    available = [column for column in SEARCH_COLUMNS if column in datasets.columns]
    if not available:
        raise ValueError("Census datasets table has none of the expected searchable columns")
    search_text = datasets[available].fillna("").astype(str).agg(" ".join, axis=1).map(normalized)

    rows: list[dict[str, Any]] = []
    for benchmark in registry.get("benchmarks", []):
        name = str(benchmark.get("name", "")).strip()
        tokens = _tokens(benchmark)
        if not tokens:
            rows.append({"benchmark": name, "candidate_rank": 0, "matched_tokens": "", "review_decision": ""})
            continue
        scores = pd.Series(0, index=datasets.index, dtype="int64")
        matches_per_row: dict[int, list[str]] = {}
        for token in tokens:
            mask = search_text.str.contains(token, regex=False)
            scores.loc[mask] += max(1, len(token))
            for index in datasets.index[mask]:
                matches_per_row.setdefault(int(index), []).append(token)
        candidate_indices = scores[scores > 0].sort_values(ascending=False).head(args.max_candidates).index
        if len(candidate_indices) == 0:
            rows.append({
                "benchmark": name,
                "candidate_rank": 0,
                "match_score": 0,
                "matched_tokens": "",
                "review_decision": "NO_MATCH_REVIEW_REQUIRED",
            })
            continue
        for rank, index in enumerate(candidate_indices, start=1):
            record = datasets.loc[index].to_dict()
            row = {
                "benchmark": name,
                "candidate_rank": rank,
                "match_score": int(scores.loc[index]),
                "matched_tokens": ",".join(sorted(set(matches_per_row.get(int(index), [])))),
                "review_decision": "",
                "review_note": "",
            }
            for column in SEARCH_COLUMNS:
                if column in datasets.columns:
                    row[column] = record.get(column)
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"benchmark_candidates={output} rows={len(rows)}")
    print("Review every benchmark manually; this script does not freeze or edit the registry.")


if __name__ == "__main__":
    main()
