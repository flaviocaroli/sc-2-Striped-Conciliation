from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from sc2.data.census_reader import load_census_h5ad, load_census_split_manifest


def _to_dense_1d(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x).reshape(-1)
    return x.astype(np.float32)


class CensusPilotDataset(Dataset):
    """
    Small in-memory dataset for the current pilot export.

    This dataset is designed for the first engineering training run.
    It uses the split manifest and returns:
    - x: expression vector
    - metadata fields for later debugging
    """

    def __init__(
        self,
        split: str,
        n_genes: int = 2048,
        log1p_input: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        manifest = load_census_split_manifest().copy()
        manifest["cell_id"] = manifest["cell_id"].astype(str)

        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        self.split = split
        self.n_genes = int(n_genes)
        self.log1p_input = bool(log1p_input)

        adata = load_census_h5ad(backed=None)

        # In this pilot build, cell_id is the row number of the exported adata.
        row_indices = self.manifest["cell_id"].astype(int).tolist()
        X = adata.X[row_indices, : self.n_genes]
        self.X = X
        self.n_features = self.n_genes

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.manifest.iloc[idx]
        x = _to_dense_1d(self.X[idx])

        if self.log1p_input:
            x = np.log1p(x)

        return {
            "x": torch.from_numpy(x),
            "cell_id": str(row["cell_id"]),
            "split": str(row["split"]),
            "dataset_id": str(row.get("dataset_id", "")),
            "assay": str(row.get("assay", "")),
            "tissue": str(row.get("tissue", "")),
            "tissue_general": str(row.get("tissue_general", "")),
            "cell_type": str(row.get("cell_type", "")),
        }