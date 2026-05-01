from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a generic gene table from any h5ad.")
    parser.add_argument("--input", required=True, help="Input h5ad path")
    parser.add_argument("--output", default=None, help="Output TSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_path = (
            input_path.parent.parent.parent
            / "artifacts"
            / "gene_tables"
            / f"{input_path.stem}_gene_table.tsv"
        )
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path, backed="r")
    var = adata.var.copy()

    df = pd.DataFrame(index=var.index.astype(str))
    df["var_index"] = df.index
    df["feature_id"] = var["feature_id"].astype(str) if "feature_id" in var.columns else df.index
    df["feature_name"] = var["feature_name"].astype(str) if "feature_name" in var.columns else df.index
    df["gene_index"] = range(len(df))

    df.to_csv(output_path, sep="\t", index=False)

    print(f"saved gene table: {output_path}")
    print(f"n_genes={len(df)}")
    print(df.head())

    if getattr(adata, "file", None) is not None:
        adata.file.close()


if __name__ == "__main__":
    main()