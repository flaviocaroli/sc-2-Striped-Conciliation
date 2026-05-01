from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _mean_rowwise(x):
    m = x.mean(axis=0)
    if sp.issparse(m):
        m = m.A1
    else:
        m = np.asarray(m).reshape(-1)
    return m.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pseudobulk h5ad from Census h5ad + split manifest.")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-h5ad", required=True)
    parser.add_argument("--cells-per-pseudobulk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    adata = ad.read_h5ad(args.input_h5ad)
    manifest = pd.read_parquet(args.split_manifest).copy()

    manifest["cell_id"] = manifest["cell_id"].astype(int)
    manifest["dataset_id"] = manifest["dataset_id"].astype(str)
    manifest["split"] = manifest["split"].astype(str)

    rows = []
    pb_vectors = []

    for split_name in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split_name].copy()

        for dataset_id, group in split_df.groupby("dataset_id"):
            cell_ids = group["cell_id"].astype(int).to_numpy()
            rng.shuffle(cell_ids)

            for start in range(0, len(cell_ids), args.cells_per_pseudobulk):
                chunk = cell_ids[start:start + args.cells_per_pseudobulk]
                if len(chunk) == 0:
                    continue

                x = adata.X[chunk]
                mean_vec = _mean_rowwise(x)

                pb_id = f"{split_name}__{dataset_id}__pb_{start // args.cells_per_pseudobulk:05d}"

                rows.append(
                    {
                        "pseudobulk_id": pb_id,
                        "dataset_id": dataset_id,
                        "source_split": split_name,
                        "n_cells": int(len(chunk)),
                    }
                )
                pb_vectors.append(mean_vec)

    X = np.vstack(pb_vectors).astype(np.float32)
    pb_adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(rows),
        var=adata.var.copy(),
    )

    out_path = Path(args.output_h5ad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pb_adata.write_h5ad(out_path)

    print(f"saved pseudobulk h5ad: {out_path}")
    print(f"n_pseudobulk={pb_adata.n_obs}, n_genes={pb_adata.n_vars}")
    print(pb_adata.obs["source_split"].value_counts())
    print(pb_adata.obs["dataset_id"].value_counts().head())


if __name__ == "__main__":
    main()