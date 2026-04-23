from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sc2.data.bulk_corruption import corrupt_bulk_vector
from sc2.data.census_reader import load_census_h5ad, load_census_split_manifest


def _to_dense_1d(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x).reshape(-1)
    return x.astype(np.float32)


class CensusSharedDataset(Dataset):
    """
    Census dataset projected onto the shared Census/ARCHS4 gene space.

    Returns:
      x = corrupted sc vector
      y = clean sc vector
    """

    def __init__(
        self,
        split: str,
        h5ad_path: str | Path,
        split_manifest_path: str | Path,
        shared_gene_table_path: str | Path,
        n_genes: int = 2048,
        log1p_input: bool = True,
        mask_prob: float = 0.15,
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        manifest = load_census_split_manifest(split_manifest_path).copy()
        manifest["cell_id"] = manifest["cell_id"].astype(str)
        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)

        shared = pd.read_csv(shared_gene_table_path, sep="\t").copy()
        shared = shared.sort_values("shared_gene_index").reset_index(drop=True)
        shared = shared.iloc[: int(n_genes)].copy()

        self.census_gene_indices = shared["gene_index"].astype(int).tolist()

        adata = load_census_h5ad(path=h5ad_path, backed=None)
        row_indices = self.manifest["cell_id"].astype(int).tolist()

        self.X = adata.X[row_indices, self.census_gene_indices]
        self.n_features = len(self.census_gene_indices)
        self.split = split
        self.log1p_input = bool(log1p_input)
        self.mask_prob = float(mask_prob)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.manifest.iloc[idx]
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
            "cell_id": str(row["cell_id"]),
            "split": str(row["split"]),
            "dataset_id": str(row.get("dataset_id", "")),
            "assay": str(row.get("assay", "")),
            "cell_type": str(row.get("cell_type", "")),
        }