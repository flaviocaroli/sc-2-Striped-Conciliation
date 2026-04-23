from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ARCHS4SubsetDataset(Dataset):
    """
    Pilot ARCHS4 dataset.

    Uses:
    - a sample manifest with train/val/test split labels
    - a shared gene table defining which ARCHS4 genes to read
    """

    def __init__(
        self,
        split: str,
        h5_path: str | Path,
        sample_manifest_path: str | Path,
        shared_gene_table_path: str | Path,
        n_genes: int = 4096,
        log1p_input: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.h5_path = Path(h5_path)
        self.sample_manifest = pd.read_parquet(sample_manifest_path).copy()
        self.shared_gene_table = pd.read_csv(shared_gene_table_path, sep="\t").copy()
        self.log1p_input = bool(log1p_input)

        self.sample_manifest = self.sample_manifest[self.sample_manifest["split"] == split].reset_index(drop=True)
        self.sample_indices = self.sample_manifest["sample_idx"].astype(int).tolist()

        self.shared_gene_table = self.shared_gene_table.sort_values("shared_gene_index").reset_index(drop=True)
        self.shared_gene_table = self.shared_gene_table.iloc[: int(n_genes)].copy()
        self.gene_indices = self.shared_gene_table["archs4_gene_index"].astype(int).tolist()

        self.n_features = len(self.gene_indices)
        self.split = split

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_idx = self.sample_indices[idx]

        with h5py.File(self.h5_path, "r") as f:
            x = f["data"]["expression"][self.gene_indices, sample_idx]

        x = np.asarray(x, dtype=np.float32)

        if self.log1p_input:
            x = np.log1p(x)

        return {
            "x": torch.from_numpy(x),
            "sample_idx": int(sample_idx),
            "split": self.split,
        }