from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sc2.data.bulk_corruption import corrupt_bulk_vector


class ARCHS4DenoiseDataset(Dataset):
    """
    Bulk denoising dataset:
    - input  : corrupted bulk vector
    - target : clean bulk vector
    """

    def __init__(
        self,
        split: str,
        h5_path: str | Path,
        sample_manifest_path: str | Path,
        shared_gene_table_path: str | Path,
        n_genes: int = 4096,
        log1p_input: bool = True,
        mask_prob: float = 0.15,
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.h5_path = Path(h5_path)
        self.sample_manifest = pd.read_parquet(sample_manifest_path).copy()
        self.shared_gene_table = pd.read_csv(shared_gene_table_path, sep="\t").copy()

        self.sample_manifest = self.sample_manifest[self.sample_manifest["split"] == split].reset_index(drop=True)
        self.sample_indices = self.sample_manifest["sample_idx"].astype(int).tolist()

        self.shared_gene_table = self.shared_gene_table.sort_values("shared_gene_index").reset_index(drop=True)
        self.shared_gene_table = self.shared_gene_table.iloc[: int(n_genes)].copy()

        original_gene_indices = self.shared_gene_table["archs4_gene_index"].astype(int).to_numpy()
        sorted_order = np.argsort(original_gene_indices)
        self.sorted_gene_indices = original_gene_indices[sorted_order]
        self.restore_order = np.argsort(sorted_order)

        self.n_features = len(original_gene_indices)
        self.split = split
        self.log1p_input = bool(log1p_input)
        self.mask_prob = float(mask_prob)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_idx = self.sample_indices[idx]

        with h5py.File(self.h5_path, "r") as f:
            x_sorted = f["data"]["expression"][self.sorted_gene_indices, sample_idx]

        x_sorted = np.asarray(x_sorted, dtype=np.float32)
        x_clean = x_sorted[self.restore_order]

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
            "sample_idx": int(sample_idx),
            "split": self.split,
        }