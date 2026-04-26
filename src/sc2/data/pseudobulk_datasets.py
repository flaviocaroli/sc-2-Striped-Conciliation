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


class PseudobulkSharedDataset(Dataset):
    def __init__(
        self,
        split: str,
        h5ad_path: str | Path,
        shared_gene_table_path: str | Path,
        n_genes: int = 4096,
        log1p_input: bool = True,
        mask_prob: float = 0.15,
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        adata = ad.read_h5ad(h5ad_path)
        shared = pd.read_csv(shared_gene_table_path, sep="\t").copy()
        shared = shared.sort_values("shared_gene_index").reset_index(drop=True)
        shared = shared.iloc[: int(n_genes)].copy()

        gene_indices = shared["gene_index"].astype(int).tolist()

        obs = adata.obs.copy()
        obs["source_split"] = obs["source_split"].astype(str)
        mask = obs["source_split"] == split

        self.obs = obs[mask].reset_index(drop=True)
        row_indices = np.where(mask.to_numpy())[0]

        X = adata.X[row_indices]
        X = X[:, gene_indices]

        self.X = X
        self.n_features = len(gene_indices)
        self.split = split
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

        return {
            "x": torch.from_numpy(x_corrupt.astype(np.float32)),
            "y": torch.from_numpy(x_clean.astype(np.float32)),
            "pseudobulk_id": str(row["pseudobulk_id"]),
            "dataset_id": str(row["dataset_id"]),
            "split": str(row["source_split"]),
            "n_cells": int(row["n_cells"]),
        }