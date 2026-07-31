#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SEARCH_COLUMNS = [
    "dataset_id",
    "collection_id",
    "dataset_title",
    "collection_name",
    "collection_doi",
    "collection_doi_label",
    "citation",
]

# These are candidate-generation terms only. Every candidate still requires
# manual confirmation against the publication and the pinned Census table.
QUERY_MAP: dict[str, list[str]] = {
    "Bubble_paired": [
        "GSE118767",
        "GSE86337",
        "Designing a single cell RNA sequencing benchmark dataset",
        "Transcriptome profiling of 5 human adenocarcinoma cell lines",
        "CellBench",
    ],
    "BAL_paired": [
        "GSE316545",
        "bronchoalveolar lavage",
    ],
    "Baron_pancreas": [
        "Baron",
        "GSE84133",
        "A Single-Cell Transcriptomic Map of the Human and Mouse Pancreas",
        "10.1016/j.cels.2016.08.011",
    ],
    "Zheng68K": [
        "Zheng68K",
        "PBMC68K",
        "68K PBMC",
        "68k PBMCs",
        "Fresh 68k PBMCs",
        "10x Genomics PBMC",
    ],
    "Segerstolpe_pancreas": [
        "Segerstolpe",
        "E-MTAB-5061",
        "Single-Cell Transcriptome Profiling of Human Pancreatic Islets",
        "10.1016/j.cmet.2016.08.020",
    ],
    "hPancreas": [
        "hPancreas",
        "human pancreas benchmark",
        "pancreas integrated benchmark",
    ],
    "MS_E_HCAD_35": [
        "E-HCAD-35",
        "multiple sclerosis",
    ],
    "Norman": [
        "Norman",
        "GSE133344",
        "Exploring genetic interaction manifolds constructed from rich phenotypes",
    ],
    "Adamson": [
        "Adamson",
        "GSE90546",
        "A multiplexed single-cell CRISPR screening platform",
        "unfolded protein response",
    ],
    "Replogle": [
        "Replogle",
        "Mapping information-rich genotype-phenotype landscapes",
        "genome-scale Perturb-seq",
        "direct guide RNA capture",
    ],
}


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def score_term(term: str) -> int:
    normalized = normalize(term)
    if re.fullmatch(r"gse\d+", normalized):
        return 200
    if normalized.startswith("10") and len(normalized) >= 12:
        return 180
    if len(normalized) >= 45:
        return 100
    if len(normalized) >= 20:
        return 60
    if len(normalized) >= 8:
        return 25
    return 10


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate broad review candidates for SC2 benchmark exclusions."
    )
    parser.add_argument("--datasets-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    datasets = pd.read_parquet(args.datasets_cache).reset_index(drop=True)
    available = [column for column in SEARCH_COLUMNS if column in datasets.columns]
    if not available:
        raise ValueError("No searchable Census metadata columns were found")

    searchable = (
        datasets[available]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(normalize)
    )

    output_rows: list[dict[str, object]] = []

    for benchmark, terms in QUERY_MAP.items():
        scores = pd.Series(0, index=datasets.index, dtype="int64")
        matched: dict[int, list[str]] = {}

        for term in terms:
            token = normalize(term)
            if not token:
                continue
            mask = searchable.str.contains(token, regex=False)
            scores.loc[mask] += score_term(term)
            for row_index in datasets.index[mask]:
                matched.setdefault(int(row_index), []).append(term)

        candidate_indices = (
            scores[scores > 0]
            .sort_values(ascending=False, kind="stable")
            .head(args.top_k)
            .index
        )

        if len(candidate_indices) == 0:
            output_rows.append(
                {
                    "benchmark": benchmark,
                    "candidate_rank": 0,
                    "match_score": 0,
                    "matched_terms": "",
                    "review_decision": "NO_MATCH_REVIEW_REQUIRED",
                    "review_note": "",
                }
            )
            continue

        for rank, index in enumerate(candidate_indices, start=1):
            record = datasets.loc[index].to_dict()
            row: dict[str, object] = {
                "benchmark": benchmark,
                "candidate_rank": rank,
                "match_score": int(scores.loc[index]),
                "matched_terms": " | ".join(
                    sorted(set(matched.get(int(index), [])))
                ),
                "review_decision": "",
                "review_note": "",
            }
            for column in available:
                row[column] = record.get(column)
            output_rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(output_rows)
    result.to_csv(output, index=False)

    print(f"broad_candidates={output}")
    print(f"rows={len(result)}")
    print("Manual publication-level review is still required before freezing the registry.")


if __name__ == "__main__":
    main()
