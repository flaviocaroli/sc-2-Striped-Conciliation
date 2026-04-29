from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sc2.data.bulk_corruption import corrupt_bulk_vector


def _to_dense_1d(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x).reshape(-1)
    return x.astype(np.float32)


class GTExSharedDataset(Dataset):
    def __init__(
        self,
        h5ad_path: str | Path,
        shared_gene_table_path: str | Path,
        n_genes: int = 4096,
        log1p_input: bool = True,
        mask_prob: float = 0.15,
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        adata = ad.read_h5ad(h5ad_path)
        shared = pd.read_csv(shared_gene_table_path, sep="\t").copy()
        shared = shared.sort_values("shared_gene_index").reset_index(drop=True)

        if "ensembl_gene" not in adata.var.columns:
            raise ValueError("GTEx h5ad var must contain 'ensembl_gene'")

        var_df = adata.var.copy().reset_index(drop=True)
        var_df["gtex_var_index"] = np.arange(len(var_df))
        var_df["ensembl_gene"] = var_df["ensembl_gene"].astype(str)

        matched = (
            shared[["shared_gene_index", "ensembl_gene"]]
            .merge(
                var_df[["gtex_var_index", "ensembl_gene"]],
                on="ensembl_gene",
                how="inner",
            )
            .sort_values("shared_gene_index")
            .reset_index(drop=True)
        )

        matched = matched.iloc[: int(n_genes)].copy()

        self.obs = adata.obs.copy().reset_index(drop=True)
        self.X = adata.X[:, matched["gtex_var_index"].astype(int).tolist()]
        self.n_features = len(matched)
        self.log1p_input = bool(log1p_input)
        self.mask_prob = float(mask_prob)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.obs.iloc[idx]
        x_clean = _to_dense_1d(self.X[idx])

        if self.log1p_input:
            x_clean = np.log1p(x_clean)

        x_corrupt = corrupt_bulk_vector(
            x_clean,
            mask_prob=self.mask_prob,
            noise_std=self.noise_std,
            seed=self.seed + idx,
        )

        sample_id = row["sample_id"] if "sample_id" in row else str(idx)
        tissue = row["SMTSD"] if "SMTSD" in row else "GTEx_lung"

        return {
            "x": torch.from_numpy(x_corrupt.astype(np.float32)),
            "y": torch.from_numpy(x_clean.astype(np.float32)),
            "sample_id": str(sample_id),
            "tissue": str(tissue),
        }