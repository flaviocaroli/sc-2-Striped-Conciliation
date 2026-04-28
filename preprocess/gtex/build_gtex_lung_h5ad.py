from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def strip_ensembl_version(gene_id: str) -> str:
    return str(gene_id).split(".")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GTEx lung h5ad from sample metadata and expression matrix.")
    parser.add_argument("--expression-tsv", required=True, help="GTEx expression matrix TSV/TSV.GZ, genes x samples")
    parser.add_argument("--sample-attributes-tsv", required=True, help="GTEx sample attributes TSV")
    parser.add_argument("--output-h5ad", required=True, help="Output h5ad path")
    parser.add_argument("--tissue-match-column", default="SMTSD")
    parser.add_argument("--tissue-match-value", default="Lung")
    args = parser.parse_args()

    sample_df = pd.read_csv(args.sample_attributes_tsv, sep="\t", low_memory=False).copy()

    if args.tissue_match_column not in sample_df.columns:
        raise ValueError(f"Column {args.tissue_match_column} not found in sample attributes")

    lung_mask = sample_df[args.tissue_match_column].astype(str).str.contains(args.tissue_match_value, case=False, na=False)
    lung_df = sample_df[lung_mask].copy()

    if "SAMPID" not in lung_df.columns:
        raise ValueError("Expected SAMPID column in sample attributes")

    lung_sample_ids = lung_df["SAMPID"].astype(str).tolist()
    if len(lung_sample_ids) == 0:
        raise ValueError("No GTEx lung samples found")

    header = pd.read_csv(args.expression_tsv, sep="\t", nrows=0)
    cols = list(header.columns)

    gene_id_col = cols[0]
    gene_name_col = None
    if len(cols) > 1 and cols[1] not in lung_sample_ids:
        gene_name_col = cols[1]

    wanted_cols = [gene_id_col]
    if gene_name_col is not None:
        wanted_cols.append(gene_name_col)

    sample_cols_present = [c for c in lung_sample_ids if c in cols]
    wanted_cols.extend(sample_cols_present)

    expr = pd.read_csv(args.expression_tsv, sep="\t", usecols=wanted_cols, low_memory=False).copy()

    var = pd.DataFrame({
        "gene_id_raw": expr[gene_id_col].astype(str),
        "ensembl_gene": expr[gene_id_col].astype(str).map(strip_ensembl_version),
    })
    if gene_name_col is not None:
        var["gene_name"] = expr[gene_name_col].astype(str)
    else:
        var["gene_name"] = var["ensembl_gene"]

    X = expr[sample_cols_present].T.to_numpy(dtype=np.float32)

    obs = lung_df.set_index("SAMPID").loc[sample_cols_present].copy()
    obs["sample_id"] = obs.index.astype(str)

    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=obs.reset_index(drop=True),
        var=var.reset_index(drop=True),
    )

    out_path = Path(args.output_h5ad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)

    print(f"saved GTEx lung h5ad: {out_path}")
    print(f"n_samples={adata.n_obs}, n_genes={adata.n_vars}")
    if args.tissue_match_column in obs.columns:
        print(obs[args.tissue_match_column].value_counts().head())


if __name__ == "__main__":
    main()