from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sc2.config import load_yaml, merge_train_and_paths
from sc2.data.gtex_shared_dataset import GTExSharedDataset
from sc2.eval.group_metrics import summarize_by_group
from sc2.eval.metrics import samplewise_mae, samplewise_mse
from sc2.models.bulk_autoencoder import BulkAutoencoder
from sc2.models.sc2lite_bridge_denoiser import SC2LiteBridgeDenoiser
from sc2.models.sc2lite_denoiser import SC2LiteDenoiser
from sc2.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a bulk-capable model on GTEx lung.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg_device: str) -> torch.device:
    if cfg_device == "cpu":
        return torch.device("cpu")
    if cfg_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_data_path(base_data_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_data_root / path


def resolve_output_path(base_output_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_output_root / path


def build_model(model_cfg: dict, input_dim: int) -> torch.nn.Module:
    kind = model_cfg["kind"]

    if kind == "bulk_autoencoder":
        return BulkAutoencoder(
            input_dim=input_dim,
            hidden_dims=model_cfg["hidden_dims"],
            dropout=float(model_cfg["dropout"]),
        )

    if kind == "sc2lite_denoiser":
        return SC2LiteDenoiser(
            input_dim=input_dim,
            adapter_dim=int(model_cfg["adapter_dim"]),
            latent_dim=int(model_cfg["latent_dim"]),
            dropout=float(model_cfg["dropout"]),
        )

    if kind == "sc2lite_bridge":
        return SC2LiteBridgeDenoiser(
            input_dim=input_dim,
            adapter_dim=int(model_cfg["adapter_dim"]),
            latent_dim=int(model_cfg["latent_dim"]),
            dropout=float(model_cfg["dropout"]),
        )

    raise ValueError(f"Unsupported model kind: {kind}")


@torch.no_grad()
def collect_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        if modality == "bulk_autoencoder":
            pred = model(x)
        else:
            pred = model(x, modality="bulk")

        mse = samplewise_mse(pred, y).cpu().numpy()
        mae = samplewise_mae(pred, y).cpu().numpy()

        for i in range(x.shape[0]):
            rows.append(
                {
                    "split": "external_test",
                    "sample_id": batch["sample_id"][i],
                    "tissue": batch["tissue"][i],
                    "mse": float(mse[i]),
                    "mae": float(mae[i]),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    eval_cfg = load_yaml(args.config)
    path_cfg = load_yaml(args.paths)
    cfg = merge_train_and_paths(eval_cfg, path_cfg)

    seed = int(cfg["seed"])
    seed_everything(seed)

    data_root = Path(cfg["paths"]["data_root"])
    output_root = Path(cfg["paths"]["output_root"])
    eval_dir = output_root / "evals" / cfg["eval_name"]
    ensure_dir(eval_dir)

    with (eval_dir / "resolved_eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = get_device(cfg["device"])
    print(f"device={device}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_section = cfg["eval"]

    gtex_h5ad_path = resolve_data_path(data_root, data_cfg["gtex_h5ad_path"])
    shared_gene_table_path = resolve_data_path(data_root, data_cfg["shared_gene_table_path"])
    checkpoint_path = resolve_output_path(output_root, eval_section["checkpoint_path"])

    print(f"gtex_h5ad_path={gtex_h5ad_path}")
    print(f"shared_gene_table_path={shared_gene_table_path}")
    print(f"checkpoint_path={checkpoint_path}")

    ds = GTExSharedDataset(
        h5ad_path=gtex_h5ad_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["mask_prob"]),
        noise_std=float(data_cfg["noise_std"]),
        seed=seed,
    )

    loader = DataLoader(
        ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(model_cfg, input_dim=ds.n_features).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    modality_key = model_cfg["kind"]
    df = collect_metrics(model, loader, device, modality=modality_key)
    overall = summarize_by_group(df, ["split"])
    by_tissue = summarize_by_group(df, ["split", "tissue"])

    df.to_csv(eval_dir / "sample_metrics.csv", index=False)
    overall.to_csv(eval_dir / "overall_by_split.csv", index=False)
    by_tissue.to_csv(eval_dir / "by_tissue.csv", index=False)

    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_by_split": overall.to_dict(orient="records"),
                "paths": {
                    "sample_metrics": str(eval_dir / "sample_metrics.csv"),
                    "overall_by_split": str(eval_dir / "overall_by_split.csv"),
                    "by_tissue": str(eval_dir / "by_tissue.csv"),
                },
            },
            f,
            indent=2,
        )

    print("saved evaluation outputs to:")
    print(eval_dir)
    print("overall metrics:")
    print(overall)


if __name__ == "__main__":
    main()