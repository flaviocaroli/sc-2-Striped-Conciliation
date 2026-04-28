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
from sc2.data.archs4_denoise_datasets import ARCHS4DenoiseDataset
from sc2.data.census_shared_datasets import CensusSharedDataset
from sc2.eval.group_metrics import summarize_by_group
from sc2.eval.metrics import samplewise_mae, samplewise_mse
from sc2.models.bulk_autoencoder import BulkAutoencoder
from sc2.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a mixed bulk/sc autoencoder.")
    parser.add_argument("--config", required=True, help="Path to eval config yaml")
    parser.add_argument("--paths", required=True, help="Path to path config yaml")
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


@torch.no_grad()
def collect_bulk_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred = model(x)
        mse = samplewise_mse(pred, y).cpu().numpy()
        mae = samplewise_mae(pred, y).cpu().numpy()

        batch_size = x.shape[0]
        for i in range(batch_size):
            rows.append(
                {
                    "modality": "bulk",
                    "split": split_name,
                    "sample_idx": int(batch["sample_idx"][i]),
                    "mse": float(mse[i]),
                    "mae": float(mae[i]),
                }
            )

    return pd.DataFrame(rows)


@torch.no_grad()
def collect_sc_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred = model(x)
        mse = samplewise_mse(pred, y).cpu().numpy()
        mae = samplewise_mae(pred, y).cpu().numpy()

        batch_size = x.shape[0]
        for i in range(batch_size):
            rows.append(
                {
                    "modality": "sc",
                    "split": split_name,
                    "cell_id": batch["cell_id"][i],
                    "dataset_id": batch["dataset_id"][i],
                    "assay": batch["assay"][i],
                    "cell_type": batch["cell_type"][i],
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

    bulk_h5_path = resolve_data_path(data_root, data_cfg["bulk_h5_path"])
    bulk_manifest_path = resolve_data_path(data_root, data_cfg["bulk_manifest_path"])
    sc_h5ad_path = resolve_data_path(data_root, data_cfg["sc_h5ad_path"])
    sc_split_manifest_path = resolve_data_path(data_root, data_cfg["sc_split_manifest_path"])
    shared_gene_table_path = resolve_data_path(data_root, data_cfg["shared_gene_table_path"])
    checkpoint_path = resolve_output_path(output_root, eval_section["checkpoint_path"])

    print(f"bulk_h5_path={bulk_h5_path}")
    print(f"bulk_manifest_path={bulk_manifest_path}")
    print(f"sc_h5ad_path={sc_h5ad_path}")
    print(f"sc_split_manifest_path={sc_split_manifest_path}")
    print(f"shared_gene_table_path={shared_gene_table_path}")
    print(f"checkpoint_path={checkpoint_path}")

    bulk_train = ARCHS4DenoiseDataset(
        split="train",
        h5_path=bulk_h5_path,
        sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["bulk_mask_prob"]),
        noise_std=float(data_cfg["bulk_noise_std"]),
        seed=seed,
    )
    bulk_val = ARCHS4DenoiseDataset(
        split="val",
        h5_path=bulk_h5_path,
        sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["bulk_mask_prob"]),
        noise_std=float(data_cfg["bulk_noise_std"]),
        seed=seed,
    )
    bulk_test = ARCHS4DenoiseDataset(
        split="test",
        h5_path=bulk_h5_path,
        sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["bulk_mask_prob"]),
        noise_std=float(data_cfg["bulk_noise_std"]),
        seed=seed,
    )

    sc_train = CensusSharedDataset(
        split="train",
        h5ad_path=sc_h5ad_path,
        split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["sc_mask_prob"]),
        noise_std=float(data_cfg["sc_noise_std"]),
        seed=seed,
    )
    sc_val = CensusSharedDataset(
        split="val",
        h5ad_path=sc_h5ad_path,
        split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["sc_mask_prob"]),
        noise_std=float(data_cfg["sc_noise_std"]),
        seed=seed,
    )
    sc_test = CensusSharedDataset(
        split="test",
        h5ad_path=sc_h5ad_path,
        split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["sc_mask_prob"]),
        noise_std=float(data_cfg["sc_noise_std"]),
        seed=seed,
    )

    bulk_train_loader = DataLoader(bulk_train, batch_size=int(data_cfg["bulk_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    bulk_val_loader = DataLoader(bulk_val, batch_size=int(data_cfg["bulk_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    bulk_test_loader = DataLoader(bulk_test, batch_size=int(data_cfg["bulk_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())

    sc_train_loader = DataLoader(sc_train, batch_size=int(data_cfg["sc_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    sc_val_loader = DataLoader(sc_val, batch_size=int(data_cfg["sc_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    sc_test_loader = DataLoader(sc_test, batch_size=int(data_cfg["sc_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())

    model = BulkAutoencoder(
        input_dim=int(data_cfg["n_genes"]),
        hidden_dims=model_cfg["hidden_dims"],
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    bulk_df = pd.concat(
        [
            collect_bulk_metrics(model, bulk_train_loader, "train", device),
            collect_bulk_metrics(model, bulk_val_loader, "val", device),
            collect_bulk_metrics(model, bulk_test_loader, "test", device),
        ],
        ignore_index=True,
    )

    sc_df = pd.concat(
        [
            collect_sc_metrics(model, sc_train_loader, "train", device),
            collect_sc_metrics(model, sc_val_loader, "val", device),
            collect_sc_metrics(model, sc_test_loader, "test", device),
        ],
        ignore_index=True,
    )

    all_df = pd.concat([bulk_df, sc_df], ignore_index=True, sort=False)

    overall_modality_split = summarize_by_group(all_df, ["modality", "split"])
    bulk_by_split = summarize_by_group(bulk_df, ["split"])
    sc_by_split = summarize_by_group(sc_df, ["split"])
    sc_by_dataset = summarize_by_group(sc_df, ["split", "dataset_id"])
    sc_by_assay = summarize_by_group(sc_df, ["split", "assay"])
    sc_by_cell_type = summarize_by_group(sc_df, ["split", "cell_type"])

    bulk_df.to_csv(eval_dir / "bulk_sample_metrics.csv", index=False)
    sc_df.to_csv(eval_dir / "sc_sample_metrics.csv", index=False)
    all_df.to_csv(eval_dir / "all_sample_metrics.csv", index=False)

    overall_modality_split.to_csv(eval_dir / "overall_by_modality_split.csv", index=False)
    bulk_by_split.to_csv(eval_dir / "bulk_by_split.csv", index=False)
    sc_by_split.to_csv(eval_dir / "sc_by_split.csv", index=False)
    sc_by_dataset.to_csv(eval_dir / "sc_by_dataset_id.csv", index=False)
    sc_by_assay.to_csv(eval_dir / "sc_by_assay.csv", index=False)
    sc_by_cell_type.to_csv(eval_dir / "sc_by_cell_type.csv", index=False)

    summary = {
        "paths": {
            "overall_by_modality_split": str(eval_dir / "overall_by_modality_split.csv"),
            "bulk_by_split": str(eval_dir / "bulk_by_split.csv"),
            "sc_by_split": str(eval_dir / "sc_by_split.csv"),
            "sc_by_dataset_id": str(eval_dir / "sc_by_dataset_id.csv"),
            "sc_by_assay": str(eval_dir / "sc_by_assay.csv"),
            "sc_by_cell_type": str(eval_dir / "sc_by_cell_type.csv"),
        },
        "overall_by_modality_split": overall_modality_split.to_dict(orient="records"),
    }

    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("saved evaluation outputs to:")
    print(eval_dir)
    print("overall metrics:")
    print(overall_modality_split)


if __name__ == "__main__":
    main()