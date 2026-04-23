from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shared gene intersection between Census and ARCHS4.")
    parser.add_argument("--census-gene-table", required=True, help="Path to Census gene table TSV")
    parser.add_argument("--archs4-gene-table", required=True, help="Path to ARCHS4 gene table TSV")
    parser.add_argument("--output", required=True, help="Output TSV path")
    args = parser.parse_args()

    census = pd.read_csv(args.census_gene_table, sep="\t")
    archs4 = pd.read_csv(args.archs4_gene_table, sep="\t")

    census["feature_name"] = census["feature_name"].astype(str)
    archs4["symbol"] = archs4["symbol"].astype(str)

    merged = census.merge(
        archs4,
        left_on="feature_name",
        right_on="symbol",
        how="inner",
    )

    merged = merged[
        [
            "feature_id",
            "feature_name",
            "gene_index",
            "archs4_gene_index",
            "symbol",
            "ensembl_gene",
            "biotype",
        ]
    ].drop_duplicates()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, sep="\t", index=False)

    print(f"saved shared intersection: {output_path}")
    print(f"n_shared_genes={len(merged)}")
    print(merged.head())


if __name__ == "__main__":
    main()