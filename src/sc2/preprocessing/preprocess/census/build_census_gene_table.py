from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract gene table and ordered feature-id vocab from a Census h5ad.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input h5ad. Defaults to $SC2_DATA_ROOT/raw/census/census_pilot_5k_lung.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to $SC2_DATA_ROOT/artifacts/gene_tables",
    )
    args = parser.parse_args()

    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    input_path = Path(args.input) if args.input else data_root / "raw" / "census" / "census_pilot_5k_lung.h5ad"
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "artifacts" / "gene_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path, backed="r")
    var = adata.var.copy()

    gene_table = pd.DataFrame(index=var.index.astype(str))
    gene_table["var_index"] = gene_table.index
    gene_table["feature_id"] = var["feature_id"].astype(str) if "feature_id" in var.columns else gene_table.index
    gene_table["feature_name"] = var["feature_name"].astype(str) if "feature_name" in var.columns else gene_table.index
    gene_table["vocab_index"] = range(len(gene_table))

    table_path = output_dir / "census_pilot_gene_table.tsv"
    vocab_path = output_dir / "census_pilot_feature_id_vocab.json"

    gene_table.to_csv(table_path, sep="\t", index=False)

    ordered_feature_ids = gene_table["feature_id"].tolist()
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(ordered_feature_ids, f, indent=2)

    print(f"saved gene table: {table_path}")
    print(f"saved feature-id vocab: {vocab_path}")
    print(f"n_genes={len(gene_table)}")
    print(f"n_unique_feature_id={gene_table['feature_id'].nunique()}")
    print(f"n_unique_feature_name={gene_table['feature_name'].nunique()}")

    if getattr(adata, "file", None) is not None:
        adata.file.close()


if __name__ == "__main__":
    main()