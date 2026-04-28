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
from sc2.data.census_shared_datasets import CensusSharedDataset
from sc2.eval.group_metrics import summarize_by_group
from sc2.eval.metrics import samplewise_mae, samplewise_mse
from sc2.models.sc2lite_denoiser import SC2LiteDenoiser
from sc2.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SC2Lite sc-only denoiser.")
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

        pred = model(x, modality="sc")
        mse = samplewise_mse(pred, y).cpu().numpy()
        mae = samplewise_mae(pred, y).cpu().numpy()

        batch_size = x.shape[0]
        for i in range(batch_size):
            rows.append(
                {
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

    sc_h5ad_path = resolve_data_path(data_root, data_cfg["sc_h5ad_path"])
    sc_split_manifest_path = resolve_data_path(data_root, data_cfg["sc_split_manifest_path"])
    shared_gene_table_path = resolve_data_path(data_root, data_cfg["shared_gene_table_path"])
    checkpoint_path = resolve_output_path(output_root, eval_section["checkpoint_path"])

    print(f"sc_h5ad_path={sc_h5ad_path}")
    print(f"sc_split_manifest_path={sc_split_manifest_path}")
    print(f"shared_gene_table_path={shared_gene_table_path}")
    print(f"checkpoint_path={checkpoint_path}")

    train_ds = CensusSharedDataset(
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
    val_ds = CensusSharedDataset(
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
    test_ds = CensusSharedDataset(
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

    train_loader = DataLoader(train_ds, batch_size=int(data_cfg["sc_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=int(data_cfg["sc_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=int(data_cfg["sc_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())

    model = SC2LiteDenoiser(
        input_dim=int(data_cfg["n_genes"]),
        adapter_dim=int(model_cfg["adapter_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    df = pd.concat(
        [
            collect_sc_metrics(model, train_loader, "train", device),
            collect_sc_metrics(model, val_loader, "val", device),
            collect_sc_metrics(model, test_loader, "test", device),
        ],
        ignore_index=True,
    )

    overall_df = summarize_by_group(df, ["split"])
    by_dataset_df = summarize_by_group(df, ["split", "dataset_id"])
    by_assay_df = summarize_by_group(df, ["split", "assay"])
    by_cell_type_df = summarize_by_group(df, ["split", "cell_type"])

    df.to_csv(eval_dir / "sample_metrics.csv", index=False)
    overall_df.to_csv(eval_dir / "overall_by_split.csv", index=False)
    by_dataset_df.to_csv(eval_dir / "by_dataset_id.csv", index=False)
    by_assay_df.to_csv(eval_dir / "by_assay.csv", index=False)
    by_cell_type_df.to_csv(eval_dir / "by_cell_type.csv", index=False)

    summary = {
        "paths": {
            "sample_metrics": str(eval_dir / "sample_metrics.csv"),
            "overall_by_split": str(eval_dir / "overall_by_split.csv"),
            "by_dataset_id": str(eval_dir / "by_dataset_id.csv"),
            "by_assay": str(eval_dir / "by_assay.csv"),
            "by_cell_type": str(eval_dir / "by_cell_type.csv"),
        },
        "overall_by_split": overall_df.to_dict(orient="records"),
    }

    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("saved evaluation outputs to:")
    print(eval_dir)
    print("overall metrics:")
    print(overall_df)


if __name__ == "__main__":
    main()