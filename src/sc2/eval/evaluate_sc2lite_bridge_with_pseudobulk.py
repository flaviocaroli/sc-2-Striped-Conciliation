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
from sc2.data.pseudobulk_datasets import PseudobulkSharedDataset
from sc2.eval.group_metrics import summarize_by_group
from sc2.eval.metrics import samplewise_mae, samplewise_mse
from sc2.models.sc2lite_bridge_denoiser import SC2LiteBridgeDenoiser
from sc2.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SC2Lite bridge denoiser.")
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
def collect_metrics(model, loader, modality, split_name, device):
    model.eval()
    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x, modality=modality)
        mse = samplewise_mse(pred, y).cpu().numpy()
        mae = samplewise_mae(pred, y).cpu().numpy()
        for i in range(x.shape[0]):
            row = {"modality": modality, "split": split_name, "mse": float(mse[i]), "mae": float(mae[i])}
            if modality == "bulk":
                row["sample_idx"] = int(batch["sample_idx"][i])
            elif modality == "sc":
                row["cell_id"] = batch["cell_id"][i]
                row["dataset_id"] = batch["dataset_id"][i]
            else:
                row["pseudobulk_id"] = batch["pseudobulk_id"][i]
                row["dataset_id"] = batch["dataset_id"][i]
            rows.append(row)
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

    device = get_device(cfg["device"])
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    checkpoint_path = resolve_output_path(output_root, cfg["eval"]["checkpoint_path"])

    bulk_h5_path = resolve_data_path(data_root, data_cfg["bulk_h5_path"])
    bulk_manifest_path = resolve_data_path(data_root, data_cfg["bulk_manifest_path"])
    sc_h5ad_path = resolve_data_path(data_root, data_cfg["sc_h5ad_path"])
    sc_split_manifest_path = resolve_data_path(data_root, data_cfg["sc_split_manifest_path"])
    pb_h5ad_path = resolve_data_path(data_root, data_cfg["pseudobulk_h5ad_path"])
    shared_gene_table_path = resolve_data_path(data_root, data_cfg["shared_gene_table_path"])

    bulk_test = ARCHS4DenoiseDataset("test", bulk_h5_path, bulk_manifest_path, shared_gene_table_path, int(data_cfg["n_genes"]), bool(data_cfg["log1p_input"]), float(data_cfg["bulk_mask_prob"]), float(data_cfg["bulk_noise_std"]), seed)
    sc_test = CensusSharedDataset("test", sc_h5ad_path, sc_split_manifest_path, shared_gene_table_path, int(data_cfg["n_genes"]), bool(data_cfg["log1p_input"]), float(data_cfg["sc_mask_prob"]), float(data_cfg["sc_noise_std"]), seed)
    pb_test = PseudobulkSharedDataset("test", pb_h5ad_path, shared_gene_table_path, int(data_cfg["n_genes"]), bool(data_cfg["log1p_input"]), float(data_cfg["pb_mask_prob"]), float(data_cfg["pb_noise_std"]), seed)

    bulk_loader = DataLoader(bulk_test, batch_size=int(data_cfg["bulk_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]))
    sc_loader = DataLoader(sc_test, batch_size=int(data_cfg["sc_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]))
    pb_loader = DataLoader(pb_test, batch_size=int(data_cfg["pb_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]))

    model = SC2LiteBridgeDenoiser(
        input_dim=int(data_cfg["n_genes"]),
        adapter_dim=int(model_cfg["adapter_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    bulk_df = collect_metrics(model, bulk_loader, "bulk", "test", device)
    sc_df = collect_metrics(model, sc_loader, "sc", "test", device)
    pb_df = collect_metrics(model, pb_loader, "pseudobulk", "test", device)

    all_df = pd.concat([bulk_df, sc_df, pb_df], ignore_index=True)
    overall = summarize_by_group(all_df, ["modality", "split"])

    all_df.to_csv(eval_dir / "all_sample_metrics.csv", index=False)
    overall.to_csv(eval_dir / "overall_by_modality_split.csv", index=False)

    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"overall": overall.to_dict(orient="records")}, f, indent=2)

    print("saved evaluation outputs to:")
    print(eval_dir)
    print("overall metrics:")
    print(overall)


if __name__ == "__main__":
    main()