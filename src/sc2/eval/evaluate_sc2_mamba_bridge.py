from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from sc2.config import load_yaml, merge_train_and_paths
from sc2.data.archs4_denoise_datasets import ARCHS4DenoiseDataset
from sc2.data.census_shared_datasets import CensusSharedDataset
from sc2.data.pseudobulk_datasets import PseudobulkSharedDataset
from sc2.eval.group_metrics import summarize_by_group
from sc2.eval.metrics import samplewise_mae, samplewise_mse
from sc2.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SC2 bridge models on bulk + sc + pseudobulk."
    )
    parser.add_argument("--config", required=True, help="Eval YAML config.")
    parser.add_argument("--paths", required=True, help="Path-root YAML config.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg_device: str | None) -> torch.device:
    device_name = str(cfg_device or "auto").lower()
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name in {"cuda", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def require_existing_file(name: str, path: Path | None) -> Path:
    if path is None:
        raise ValueError(f"Missing required path for {name}.")
    if not path.exists():
        raise FileNotFoundError(f"Resolved path for {name} does not exist: {path}")
    return path


def build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def ensure_non_empty_loader(name: str, loader: DataLoader) -> None:
    if len(loader) == 0:
        raise ValueError(
            f"{name} is empty. Check split manifests, dataset paths, and config values."
        )


def move_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=device.type == "cuda")


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def build_model(model_cfg: dict[str, Any], n_genes: int) -> tuple[str, nn.Module]:
    kind = str(model_cfg.get("kind", "sc2_mamba_bridge")).strip().lower()
    common_kwargs = {
        "n_genes": int(n_genes),
        "d_model": int(model_cfg.get("d_model", 128)),
        "n_layers": int(model_cfg.get("n_layers", 4)),
        "d_state": int(model_cfg.get("d_state", 64)),
        "d_conv": int(model_cfg.get("d_conv", 4)),
        "expand": int(model_cfg.get("expand", 2)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }

    if kind in {"sc2_mamba_bridge", "mamba_bridge", "bridge_mamba"}:
        try:
            from sc2.models.sc2_mamba_bridge import SC2MambaBridge
        except ImportError as exc:
            raise ImportError(
                "model.kind='sc2_mamba_bridge' requires mamba-ssm. "
                "Install it with `pip install mamba-ssm`, or evaluate a "
                "native model with model.kind='native_mamba_bridge'."
            ) from exc
        model = SC2MambaBridge(**common_kwargs)
        return kind, model

    if kind in {
        "native_mamba_bridge",
        "sc2_native_mamba_bridge",
        "native_like_mamba_bridge",
    }:
        from sc2.models.sc2_native_mamba_bridge import SC2NativeMambaBridge

        model = SC2NativeMambaBridge(
            **common_kwargs,
            mixer_type=str(model_cfg.get("mixer_type", "mamba1")),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            merge_mode=str(model_cfg.get("merge_mode", "sum")),
            smart_flip=bool(model_cfg.get("smart_flip", False)),
            rank_input=bool(model_cfg.get("rank_input", False)),
            preserve_prefix_tokens=int(model_cfg.get("preserve_prefix_tokens", 0)),
            norm_type=str(model_cfg.get("norm_type", "rmsnorm")),
        )
        return kind, model

    if kind in {
        "sc2_hybrid_bridge",
        "hybrid_bridge",
        "striped_hybrid_bridge",
        "sc2_striped_bridge",
    }:
        try:
            from sc2.models.sc2_hybrid_bridge import SC2HybridBridge
        except ImportError as exc:
            raise ImportError(
                "model.kind='sc2_hybrid_bridge' requires mamba-ssm. "
                "Install it with `pip install mamba-ssm`."
            ) from exc
        model = SC2HybridBridge(
            **common_kwargs,
            n_heads=int(model_cfg.get("n_heads", 4)),
            attn_every=int(model_cfg.get("attn_every", 3)),
        )
        return kind, model

    supported = [
        "sc2_mamba_bridge",
        "native_mamba_bridge",
        "sc2_hybrid_bridge",
    ]
    raise ValueError(
        f"Unsupported model.kind={kind!r}. Supported values: {supported}"
    )


@torch.inference_mode()
def collect_metrics(
    model: nn.Module,
    loader: DataLoader,
    modality: str,
    split_name: str,
    device: torch.device,
) -> pd.DataFrame:
    ensure_non_empty_loader(f"eval loader ({modality})", loader)
    model.eval()
    rows: list[dict[str, object]] = []

    for batch in loader:
        x = move_tensor(batch["x"], device)
        y = move_tensor(batch["y"], device)
        pred = model(x, modality=modality)

        mse = samplewise_mse(pred, y).detach().cpu().numpy()
        mae = samplewise_mae(pred, y).detach().cpu().numpy()

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


def main() -> None:
    args = parse_args()
    eval_cfg = load_yaml(args.config)
    path_cfg = load_yaml(args.paths)
    cfg = merge_train_and_paths(eval_cfg, path_cfg)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    data_root = Path(cfg["paths"]["data_root"])
    output_root = Path(cfg["paths"]["output_root"])
    eval_dir = output_root / "evals" / str(cfg["eval_name"])
    ensure_dir(eval_dir)

    with (eval_dir / "resolved_eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = get_device(cfg.get("device", "auto"))
    print(f"device={device}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    checkpoint_path = require_existing_file(
        "checkpoint_path",
        resolve_output_path(output_root, cfg["eval"].get("checkpoint_path")),
    )
    bulk_h5_path = require_existing_file(
        "bulk_h5_path",
        resolve_data_path(data_root, data_cfg.get("bulk_h5_path")),
    )
    bulk_manifest_path = require_existing_file(
        "bulk_manifest_path",
        resolve_data_path(data_root, data_cfg.get("bulk_manifest_path")),
    )
    sc_h5ad_path = require_existing_file(
        "sc_h5ad_path",
        resolve_data_path(data_root, data_cfg.get("sc_h5ad_path")),
    )
    sc_split_manifest_path = require_existing_file(
        "sc_split_manifest_path",
        resolve_data_path(data_root, data_cfg.get("sc_split_manifest_path")),
    )
    pb_h5ad_path = require_existing_file(
        "pseudobulk_h5ad_path",
        resolve_data_path(data_root, data_cfg.get("pseudobulk_h5ad_path")),
    )
    shared_gene_table_path = require_existing_file(
        "shared_gene_table_path",
        resolve_data_path(data_root, data_cfg.get("shared_gene_table_path")),
    )

    print(f"checkpoint_path={checkpoint_path}")
    print(f"bulk_h5_path={bulk_h5_path}")
    print(f"bulk_manifest_path={bulk_manifest_path}")
    print(f"sc_h5ad_path={sc_h5ad_path}")
    print(f"sc_split_manifest_path={sc_split_manifest_path}")
    print(f"pseudobulk_h5ad_path={pb_h5ad_path}")
    print(f"shared_gene_table_path={shared_gene_table_path}")

    n_genes_requested = int(data_cfg["n_genes"])
    log1p_input = bool(data_cfg.get("log1p_input", True))

    bulk_test = ARCHS4DenoiseDataset(
        split="test",
        h5_path=bulk_h5_path,
        sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("bulk_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("bulk_noise_std", 0.0)),
        seed=seed,
    )
    sc_test = CensusSharedDataset(
        split="test",
        h5ad_path=sc_h5ad_path,
        split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("sc_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("sc_noise_std", 0.0)),
        seed=seed,
    )
    pb_test = PseudobulkSharedDataset(
        split="test",
        h5ad_path=pb_h5ad_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("pb_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("pb_noise_std", 0.0)),
        seed=seed,
    )

    input_dim = int(bulk_test.n_features)
    for name, n_features in [
        ("sc_test", sc_test.n_features),
        ("pb_test", pb_test.n_features),
    ]:
        if int(n_features) != input_dim:
            raise ValueError(
                f"Feature dimension mismatch: bulk_test has {input_dim}, "
                f"but {name} has {n_features}."
            )
    print(f"input_dim={input_dim}")

    num_workers = int(data_cfg.get("num_workers", 0))
    bulk_loader = build_loader(
        bulk_test,
        batch_size=int(data_cfg.get("bulk_batch_size", 16)),
        num_workers=num_workers,
    )
    sc_loader = build_loader(
        sc_test,
        batch_size=int(data_cfg.get("sc_batch_size", 32)),
        num_workers=num_workers,
    )
    pb_loader = build_loader(
        pb_test,
        batch_size=int(data_cfg.get("pb_batch_size", 16)),
        num_workers=num_workers,
    )

    ensure_non_empty_loader("bulk_loader", bulk_loader)
    ensure_non_empty_loader("sc_loader", sc_loader)
    ensure_non_empty_loader("pb_loader", pb_loader)

    model_kind, model = build_model(model_cfg, n_genes=input_dim)
    model = model.to(device)
    params = count_parameters(model)
    print(f"model_kind={model_kind}")
    print(f"parameters_total={params['total']}")
    print(f"parameters_trainable={params['trainable']}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    bulk_df = collect_metrics(model, bulk_loader, "bulk", "test", device)
    sc_df = collect_metrics(model, sc_loader, "sc", "test", device)
    pb_df = collect_metrics(model, pb_loader, "pseudobulk", "test", device)

    all_df = pd.concat([bulk_df, sc_df, pb_df], ignore_index=True)
    overall = summarize_by_group(all_df, ["modality", "split"])

    all_df.to_csv(eval_dir / "all_sample_metrics.csv", index=False)
    overall.to_csv(eval_dir / "overall_by_modality_split.csv", index=False)

    summary = {
        "eval_name": str(cfg["eval_name"]),
        "checkpoint_path": str(checkpoint_path),
        "model_kind": model_kind,
        "parameter_counts": params,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "overall": overall.to_dict(orient="records"),
    }
    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("saved evaluation outputs to:")
    print(eval_dir)
    print("overall metrics:")
    print(overall)


if __name__ == "__main__":
    main()