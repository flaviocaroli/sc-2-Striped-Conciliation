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
from sc2.models.sc2_mamba_bridge import SC2MambaBridge
from sc2.models.sc2lite_bridge_denoiser import SC2LiteBridgeDenoiser
from sc2.utils.paths import ensure_dir
from sc2.models.sc2_native_mamba_bridge import SC2NativeMambaBridge

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bridge-style denoiser on bulk + sc + pseudobulk.")
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
def collect_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    modality: str,
    split_name: str,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred = model(x, modality=modality)
        mse = samplewise_mse(pred, y).cpu().numpy()
        mae = samplewise_mae(pred, y).cpu().numpy()

        for i in range(x.shape[0]):
            row: dict[str, object] = {
                "modality": modality,
                "split": split_name,
                "mse": float(mse[i]),
                "mae": float(mae[i]),
            }
            if modality == "bulk":
                row["sample_idx"] = int(batch["sample_idx"][i])
            elif modality == "sc":
                row["cell_id"] = batch["cell_id"][i]
                row["dataset_id"] = batch["dataset_id"][i]
            elif modality == "pseudobulk":
                row["pseudobulk_id"] = batch["pseudobulk_id"][i]
                row["dataset_id"] = batch["dataset_id"][i]
            else:
                raise ValueError(f"Unsupported modality: {modality}")
            rows.append(row)

    return pd.DataFrame(rows)


def build_model(model_cfg: dict, n_genes: int, device: torch.device) -> torch.nn.Module:
    kind = model_cfg.get("kind", "sc2lite_bridge")

    if kind == "sc2lite_bridge":
        return SC2LiteBridgeDenoiser(
            input_dim=n_genes,
            adapter_dim=int(model_cfg["adapter_dim"]),
            latent_dim=int(model_cfg["latent_dim"]),
            dropout=float(model_cfg["dropout"]),
        ).to(device)

    if kind == "sc2_mamba_bridge":
        return SC2MambaBridge(
            n_genes=n_genes,
            d_model=int(model_cfg["d_model"]),
            n_layers=int(model_cfg["n_layers"]),
            d_state=int(model_cfg["d_state"]),
            d_conv=int(model_cfg["d_conv"]),
            expand=int(model_cfg["expand"]),
            dropout=float(model_cfg["dropout"]),
        ).to(device)
    if kind == "native_mamba_bridge":
        return SC2NativeMambaBridge(
            n_genes=n_genes,
            d_model=int(model_cfg["d_model"]),
            n_layers=int(model_cfg["n_layers"]),
            d_state=int(model_cfg["d_state"]),
            d_conv=int(model_cfg["d_conv"]),
            expand=int(model_cfg["expand"]),
            dropout=float(model_cfg["dropout"]),
            mixer_type=str(model_cfg.get("mixer_type", "mamba1")),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            merge_mode=str(model_cfg.get("merge_mode", "sum")),
            smart_flip=bool(model_cfg.get("smart_flip", False)),
            rank_input=bool(model_cfg.get("rank_input", False)),
            preserve_prefix_tokens=int(model_cfg.get("preserve_prefix_tokens", 0)),
            norm_type=str(model_cfg.get("norm_type", "rmsnorm")),
        ).to(device)
    raise ValueError(f"Unsupported model kind: {kind}")


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
    checkpoint_path = resolve_output_path(output_root, cfg["eval"]["checkpoint_path"])

    bulk_h5_path = resolve_data_path(data_root, data_cfg["bulk_h5_path"])
    bulk_manifest_path = resolve_data_path(data_root, data_cfg["bulk_manifest_path"])
    sc_h5ad_path = resolve_data_path(data_root, data_cfg["sc_h5ad_path"])
    sc_split_manifest_path = resolve_data_path(data_root, data_cfg["sc_split_manifest_path"])
    pb_h5ad_path = resolve_data_path(data_root, data_cfg["pseudobulk_h5ad_path"])
    shared_gene_table_path = resolve_data_path(data_root, data_cfg["shared_gene_table_path"])

    print(f"bulk_h5_path={bulk_h5_path}")
    print(f"bulk_manifest_path={bulk_manifest_path}")
    print(f"sc_h5ad_path={sc_h5ad_path}")
    print(f"sc_split_manifest_path={sc_split_manifest_path}")
    print(f"pseudobulk_h5ad_path={pb_h5ad_path}")
    print(f"shared_gene_table_path={shared_gene_table_path}")
    print(f"checkpoint_path={checkpoint_path}")

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
    pb_test = PseudobulkSharedDataset(
        split="test",
        h5ad_path=pb_h5ad_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["pb_mask_prob"]),
        noise_std=float(data_cfg["pb_noise_std"]),
        seed=seed,
    )

    bulk_loader = DataLoader(
        bulk_test,
        batch_size=int(data_cfg["bulk_batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    sc_loader = DataLoader(
        sc_test,
        batch_size=int(data_cfg["sc_batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    pb_loader = DataLoader(
        pb_test,
        batch_size=int(data_cfg["pb_batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(model_cfg, n_genes=int(data_cfg["n_genes"]), device=device)
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