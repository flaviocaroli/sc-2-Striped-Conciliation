from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from sc2.benchmarks.masks import make_fixed_numpy_masks
from sc2.benchmarks.metrics import (
    false_positive_zero_fill_rate,
    gene_wise_correlation,
    samplewise_reconstruction_metrics,
)
from sc2.models.factory import build_model


class MatrixDataset(Dataset):
    def __init__(self, x: np.ndarray, mask: np.ndarray):
        self.x = x.astype(np.float32)
        self.mask = mask.astype(bool)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        target = torch.from_numpy(self.x[idx])
        mask = torch.from_numpy(self.mask[idx])
        corrupted = target.clone()
        corrupted[mask] = 0.0
        return {
            "x": corrupted,
            "target": target,
            "mask": mask,
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--paths", required=True)
    return p.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, modality: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    masks = []

    for batch in loader:
        x = batch["x"].to(device)
        pred = model(x, modality=modality)
        preds.append(pred.detach().cpu().numpy())
        targets.append(batch["target"].numpy())
        masks.append(batch["mask"].numpy())

    return np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = load_yaml(args.paths)

    def get_root(paths_cfg: dict, key: str, env_key: str) -> Path:
        if "paths" in paths_cfg and key in paths_cfg["paths"]:
            return Path(paths_cfg["paths"][key])

        if key in paths_cfg:
            return Path(paths_cfg[key])

        import os

        env_value = os.environ.get(env_key)
        if env_value:
            return Path(env_value)

        raise KeyError(
            f"Could not find {key}. Expected either paths.{key}, top-level {key}, "
            f"or environment variable {env_key}."
        )


    data_root = get_root(paths, "data_root", "SC2_DATA_ROOT")
    output_root = get_root(paths, "output_root", "SC2_OUTPUT_ROOT")

    eval_name = cfg["eval_name"]
    out_dir = output_root / "evals" / eval_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    print(f"device={device}")

    h5ad_path = resolve(data_root, cfg["data"]["h5ad_path"])
    checkpoint_path = resolve(output_root, cfg["eval"]["checkpoint_path"])

    print(f"h5ad_path={h5ad_path}")
    print(f"checkpoint_path={checkpoint_path}")

    adata = ad.read_h5ad(h5ad_path)
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)

    n_genes = int(cfg["data"].get("n_genes", x.shape[1]))
    x = x[:, :n_genes]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = checkpoint.get("config", {})
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    n_genes = int(ckpt_cfg.get("data", {}).get("n_genes", n_genes))

    model_kind, model = build_model(model_cfg, n_genes=n_genes)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"model_kind={model_kind}")
    print(f"n_cells={x.shape[0]}")
    print(f"n_genes={x.shape[1]}")

    all_overall = []

    for mask_prob in cfg["benchmark"]["mask_probs"]:
        mask = make_fixed_numpy_masks(
            matrix=x,
            mask_prob=float(mask_prob),
            seed=int(cfg["seed"]),
            nonzero_only=bool(cfg["benchmark"].get("nonzero_only", True)),
        )

        ds = MatrixDataset(x, mask)
        loader = DataLoader(
            ds,
            batch_size=int(cfg["data"].get("batch_size", 64)),
            shuffle=False,
            num_workers=int(cfg["data"].get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
        )

        pred, target, used_mask = predict(
            model=model,
            loader=loader,
            device=device,
            modality=str(cfg["benchmark"].get("modality", "sc")),
        )

        sample_df = samplewise_reconstruction_metrics(pred, target, used_mask)
        gene_df = gene_wise_correlation(pred, target, list(adata.var_names[:n_genes]))

        sample_df["mask_prob"] = mask_prob
        gene_df["mask_prob"] = mask_prob

        fp_rate = false_positive_zero_fill_rate(
            pred=pred,
            target=target,
            threshold=float(cfg["benchmark"].get("zero_fill_threshold", 0.1)),
        )

        overall = {
            "mask_prob": mask_prob,
            "n_cells": int(x.shape[0]),
            "n_genes": int(x.shape[1]),
            "sample_mse_mean": float(sample_df["mse"].mean()),
            "sample_rmse_mean": float(sample_df["rmse"].mean()),
            "sample_mae_mean": float(sample_df["mae"].mean()),
            "sample_pearson_mean": float(sample_df["pearson"].mean()),
            "sample_spearman_mean": float(sample_df["spearman"].mean()),
            "gene_pearson_mean": float(gene_df["pearson"].mean()),
            "gene_spearman_mean": float(gene_df["spearman"].mean()),
            "false_positive_zero_fill_rate": fp_rate,
        }
        all_overall.append(overall)

        sample_df.to_csv(out_dir / f"sample_metrics_mask_{mask_prob}.csv", index=False)
        gene_df.to_csv(out_dir / f"gene_metrics_mask_{mask_prob}.csv", index=False)
        np.savez_compressed(out_dir / f"mask_{mask_prob}.npz", mask=used_mask)

    overall_df = pd.DataFrame(all_overall)
    overall_df.to_csv(out_dir / "overall_masked_reconstruction.csv", index=False)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"overall": all_overall}, f, indent=2)

    print("saved evaluation outputs to:")
    print(out_dir)
    print(overall_df)


if __name__ == "__main__":
    main()