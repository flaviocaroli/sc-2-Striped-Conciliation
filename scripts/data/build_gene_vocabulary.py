#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a train-only Ensembl gene vocabulary")
    parser.add_argument("--gene-stats", required=True, help="Parquet computed only from training cells")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--required-genes", default=None, help="Optional newline-separated predeclared Ensembl IDs")
    parser.add_argument("--min-detected-cells", type=int, default=100)
    args = parser.parse_args()
    frame = pd.read_parquet(args.gene_stats)
    required_columns = {"ensembl_id", "detected_cells", "variance_log1p"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Gene statistics missing columns: {sorted(missing)}")
    candidates = frame[frame["detected_cells"] >= args.min_detected_cells].copy()
    candidates = candidates.sort_values(["variance_log1p", "detected_cells", "ensembl_id"], ascending=[False, False, True])
    required: list[str] = []
    if args.required_genes:
        required = [
            line.strip()
            for line in Path(args.required_genes).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    missing_required = sorted(set(required) - set(candidates["ensembl_id"].astype(str)))
    if missing_required:
        raise ValueError(f"Required genes absent from eligible train-only statistics: {missing_required[:20]}")
    ordered = []
    seen = set()
    for gene in required + candidates["ensembl_id"].astype(str).tolist():
        if gene not in seen:
            seen.add(gene)
            ordered.append(gene)
        if len(ordered) == args.target_count:
            break
    if len(ordered) != args.target_count:
        raise ValueError(f"Only selected {len(ordered)} genes, expected {args.target_count}")
    output = pd.DataFrame({"gene_index": range(len(ordered)), "ensembl_id": ordered})
    digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
    output["vocabulary_sha256"] = digest
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    destination.with_suffix(".sha256.txt").write_text(digest + "\n", encoding="utf-8")
    print(f"genes={len(output)} sha256={digest}")


if __name__ == "__main__":
    main()
