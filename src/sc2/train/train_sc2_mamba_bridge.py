from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from sc2.config import load_yaml, merge_train_and_paths
from sc2.data.archs4_denoise_datasets import ARCHS4DenoiseDataset
from sc2.data.census_shared_datasets import CensusSharedDataset
from sc2.data.mixed_loaders import infinite_loader
from sc2.data.pseudobulk_datasets import PseudobulkSharedDataset
from sc2.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SC2 bridge models on bulk + sc + pseudobulk."
    )
    parser.add_argument("--config", required=True, help="Training YAML config.")
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


def require_existing_file(name: str, path: Path | None) -> Path:
    if path is None:
        raise ValueError(f"Missing required path for {name}.")
    if not path.exists():
        raise FileNotFoundError(f"Resolved path for {name} does not exist: {path}")
    return path


def build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=generator,
    )


def ensure_non_empty_loader(name: str, loader: DataLoader) -> None:
    if len(loader) == 0:
        raise ValueError(
            f"{name} is empty. Check split manifests, dataset paths, and config values."
        )


def move_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=device.type == "cuda")


def resolve_amp(train_cfg: dict[str, Any], device: torch.device) -> bool:
    amp_cfg = train_cfg.get("amp", "auto")
    if isinstance(amp_cfg, str):
        value = amp_cfg.strip().lower()
        if value == "auto":
            return device.type == "cuda"
        return value in {"1", "true", "yes", "on"}
    return bool(amp_cfg) and device.type == "cuda"


def resolve_amp_dtype(train_cfg: dict[str, Any], device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    dtype_cfg = str(train_cfg.get("amp_dtype", "bfloat16")).strip().lower()
    if dtype_cfg in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_cfg in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError("Unsupported amp_dtype. Use 'bfloat16' or 'float16'.")


def autocast_context(device: torch.device, enabled: bool, amp_dtype: torch.dtype | None):
    if enabled and device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def mean_alignment_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    mu_a = F.normalize(z_a.mean(dim=0, keepdim=True), dim=1)
    mu_b = F.normalize(z_b.mean(dim=0, keepdim=True), dim=1)
    return F.mse_loss(mu_a, mu_b)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


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
                "Install it with `pip install mamba-ssm`, or switch to "
                "model.kind='native_mamba_bridge' for the native-like implementation."
            ) from exc
        return kind, SC2MambaBridge(**common_kwargs)

    if kind in {
        "native_mamba_bridge",
        "sc2_native_mamba_bridge",
        "native_like_mamba_bridge",
    }:
        from sc2.models.sc2_native_mamba_bridge import SC2NativeMambaBridge

        return kind, SC2NativeMambaBridge(
            **common_kwargs,
            mixer_type=str(model_cfg.get("mixer_type", "mamba1")),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            merge_mode=str(model_cfg.get("merge_mode", "sum")),
            smart_flip=bool(model_cfg.get("smart_flip", False)),
            rank_input=bool(model_cfg.get("rank_input", False)),
            preserve_prefix_tokens=int(model_cfg.get("preserve_prefix_tokens", 0)),
            norm_type=str(model_cfg.get("norm_type", "rmsnorm")),
        )

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
        return kind, SC2HybridBridge(
            **common_kwargs,
            n_heads=int(model_cfg.get("n_heads", 4)),
            attn_every=int(model_cfg.get("attn_every", 3)),
        )

    raise ValueError(
        "Unsupported model.kind. Supported values: "
        "['sc2_mamba_bridge', 'native_mamba_bridge', 'sc2_hybrid_bridge']"
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = str(train_cfg.get("scheduler", "none")).strip().lower()
    if scheduler_name in {"", "none", "null"}:
        return None
    if scheduler_name == "cosine":
        epochs = int(train_cfg["epochs"])
        min_lr = float(train_cfg.get("min_lr", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=min_lr,
        )
    if scheduler_name == "plateau":
        factor = float(train_cfg.get("plateau_factor", 0.5))
        patience = int(train_cfg.get("plateau_patience", 2))
        min_lr = float(train_cfg.get("min_lr", 0.0))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    raise ValueError("Unsupported scheduler. Use one of: none, cosine, plateau.")


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: torch.amp.GradScaler,
    cfg: dict[str, Any],
    model_kind: str,
    best_val_total: float,
) -> None:
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model_kind": model_kind,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "best_val_total": float(best_val_total),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def model_forward_with_latent(
    model: nn.Module,
    x: torch.Tensor,
    modality: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_with_latent"):
        return model.forward_with_latent(x, modality=modality)
    y = model(x, modality=modality)
    z = model.encode(x, modality=modality)
    return y, z


@torch.inference_mode()
def eval_loader(
    model: nn.Module,
    loader: DataLoader,
    modality: str,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> float:
    ensure_non_empty_loader(f"eval loader ({modality})", loader)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = move_tensor(batch["x"], device)
        y = move_tensor(batch["y"], device)
        with autocast_context(device, amp_enabled, amp_dtype):
            pred = model(x, modality=modality)
            loss = criterion(pred, y)
        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate_all(
    model: nn.Module,
    bulk_loader: DataLoader,
    sc_loader: DataLoader,
    pb_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    weights: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    bulk_loss = eval_loader(model, bulk_loader, "bulk", device, amp_enabled, amp_dtype)
    sc_loss = eval_loader(model, sc_loader, "sc", device, amp_enabled, amp_dtype)
    pb_loss = eval_loader(model, pb_loader, "pseudobulk", device, amp_enabled, amp_dtype)
    total = (
        weights["bulk"] * bulk_loss
        + weights["sc"] * sc_loss
        + weights["pb"] * pb_loss
    )
    return {
        f"{prefix}_bulk_loss": bulk_loss,
        f"{prefix}_sc_loss": sc_loss,
        f"{prefix}_pb_loss": pb_loss,
        f"{prefix}_total": total,
    }


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml(args.config)
    path_cfg = load_yaml(args.paths)
    cfg = merge_train_and_paths(train_cfg, path_cfg)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_root = Path(cfg["paths"]["output_root"])
    data_root = Path(cfg["paths"]["data_root"])
    run_name = str(cfg["run_name"])
    run_dir = output_root / run_name
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = get_device(cfg.get("device", "auto"))
    print(f"device={device}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_section = cfg["train"]

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

    print(f"bulk_h5_path={bulk_h5_path}")
    print(f"bulk_manifest_path={bulk_manifest_path}")
    print(f"sc_h5ad_path={sc_h5ad_path}")
    print(f"sc_split_manifest_path={sc_split_manifest_path}")
    print(f"pseudobulk_h5ad_path={pb_h5ad_path}")
    print(f"shared_gene_table_path={shared_gene_table_path}")

    n_genes_requested = int(data_cfg["n_genes"])
    log1p_input = bool(data_cfg.get("log1p_input", True))

    bulk_train = ARCHS4DenoiseDataset(
        split="train",
        h5_path=bulk_h5_path,
        sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("bulk_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("bulk_noise_std", 0.0)),
        seed=seed,
    )
    bulk_val = ARCHS4DenoiseDataset(
        split="val",
        h5_path=bulk_h5_path,
        sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("bulk_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("bulk_noise_std", 0.0)),
        seed=seed,
    )
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

    sc_train = CensusSharedDataset(
        split="train",
        h5ad_path=sc_h5ad_path,
        split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("sc_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("sc_noise_std", 0.0)),
        seed=seed,
    )
    sc_val = CensusSharedDataset(
        split="val",
        h5ad_path=sc_h5ad_path,
        split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("sc_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("sc_noise_std", 0.0)),
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

    pb_train = PseudobulkSharedDataset(
        split="train",
        h5ad_path=pb_h5ad_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("pb_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("pb_noise_std", 0.0)),
        seed=seed,
    )
    pb_val = PseudobulkSharedDataset(
        split="val",
        h5ad_path=pb_h5ad_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("pb_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("pb_noise_std", 0.0)),
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

    print(
        f"bulk_train={len(bulk_train)} bulk_val={len(bulk_val)} bulk_test={len(bulk_test)}"
    )
    print(f"sc_train={len(sc_train)} sc_val={len(sc_val)} sc_test={len(sc_test)}")
    print(f"pb_train={len(pb_train)} pb_val={len(pb_val)} pb_test={len(pb_test)}")

    input_dim = int(bulk_train.n_features)
    for name, n_features in [
        ("bulk_val", bulk_val.n_features),
        ("bulk_test", bulk_test.n_features),
        ("sc_train", sc_train.n_features),
        ("sc_val", sc_val.n_features),
        ("sc_test", sc_test.n_features),
        ("pb_train", pb_train.n_features),
        ("pb_val", pb_val.n_features),
        ("pb_test", pb_test.n_features),
    ]:
        if int(n_features) != input_dim:
            raise ValueError(
                f"Feature dimension mismatch: bulk_train has {input_dim}, "
                f"but {name} has {n_features}."
            )
    print(f"input_dim={input_dim}")

    num_workers = int(data_cfg.get("num_workers", 0))
    bulk_train_loader = build_loader(
        bulk_train,
        batch_size=int(data_cfg.get("bulk_batch_size", 8)),
        shuffle=True,
        num_workers=num_workers,
        seed=seed + 11,
    )
    bulk_val_loader = build_loader(
        bulk_val,
        batch_size=int(data_cfg.get("bulk_batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 12,
    )
    bulk_test_loader = build_loader(
        bulk_test,
        batch_size=int(data_cfg.get("bulk_batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 13,
    )

    sc_train_loader = build_loader(
        sc_train,
        batch_size=int(data_cfg.get("sc_batch_size", 16)),
        shuffle=True,
        num_workers=num_workers,
        seed=seed + 21,
    )
    sc_val_loader = build_loader(
        sc_val,
        batch_size=int(data_cfg.get("sc_batch_size", 16)),
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 22,
    )
    sc_test_loader = build_loader(
        sc_test,
        batch_size=int(data_cfg.get("sc_batch_size", 16)),
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 23,
    )

    pb_train_loader = build_loader(
        pb_train,
        batch_size=int(data_cfg.get("pb_batch_size", 8)),
        shuffle=True,
        num_workers=num_workers,
        seed=seed + 31,
    )
    pb_val_loader = build_loader(
        pb_val,
        batch_size=int(data_cfg.get("pb_batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 32,
    )
    pb_test_loader = build_loader(
        pb_test,
        batch_size=int(data_cfg.get("pb_batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 33,
    )

    ensure_non_empty_loader("bulk_train_loader", bulk_train_loader)
    ensure_non_empty_loader("sc_train_loader", sc_train_loader)
    ensure_non_empty_loader("pb_train_loader", pb_train_loader)
    ensure_non_empty_loader("bulk_val_loader", bulk_val_loader)
    ensure_non_empty_loader("sc_val_loader", sc_val_loader)
    ensure_non_empty_loader("pb_val_loader", pb_val_loader)
    ensure_non_empty_loader("bulk_test_loader", bulk_test_loader)
    ensure_non_empty_loader("sc_test_loader", sc_test_loader)
    ensure_non_empty_loader("pb_test_loader", pb_test_loader)

    model_kind, model = build_model(model_cfg, n_genes=input_dim)
    model = model.to(device)
    param_counts = count_parameters(model)
    print(f"model_kind={model_kind}")
    print(f"parameters_total={param_counts['total']}")
    print(f"parameters_trainable={param_counts['trainable']}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_section.get("lr", 5e-4)),
        weight_decay=float(train_section.get("weight_decay", 1e-4)),
    )
    scheduler = build_scheduler(optimizer, train_section)

    amp_enabled = resolve_amp(train_section, device)
    amp_dtype = resolve_amp_dtype(train_section, device) if amp_enabled else None
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and device.type == "cuda" and amp_dtype == torch.float16,
    )

    epochs = int(train_section.get("epochs", 10))
    grad_clip_norm = float(train_section.get("grad_clip_norm", 1.0))
    grad_accum_steps = max(1, int(train_section.get("grad_accum_steps", 1)))
    steps_per_epoch_cfg = train_section.get("steps_per_epoch")
    pb_every = max(1, int(train_section.get("pb_every", 1)))
    eval_every = max(1, int(train_section.get("eval_every", 1)))

    n_steps = max(len(bulk_train_loader), len(sc_train_loader), len(pb_train_loader))
    if steps_per_epoch_cfg is not None:
        n_steps = max(1, int(steps_per_epoch_cfg))

    weights = {
        "bulk": float(train_section.get("bulk_loss_weight", 1.0)),
        "sc": float(train_section.get("sc_loss_weight", 5.0)),
        "pb": float(train_section.get("pb_loss_weight", 1.0)),
        "align": float(train_section.get("align_loss_weight", 0.5)),
    }

    print(f"amp_enabled={amp_enabled}")
    print(f"amp_dtype={amp_dtype}")
    print(f"grad_accum_steps={grad_accum_steps}")
    print(f"steps_per_epoch={n_steps}")
    print(f"pb_every={pb_every}")
    print(f"eval_every={eval_every}")

    best_val_total = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int | str]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        bulk_iter = infinite_loader(bulk_train_loader)
        sc_iter = infinite_loader(sc_train_loader)
        pb_iter = infinite_loader(pb_train_loader)

        optimizer.zero_grad(set_to_none=True)

        totals = {
            "train_bulk_loss": 0.0,
            "train_sc_loss": 0.0,
            "train_pb_loss": 0.0,
            "train_align_loss": 0.0,
            "train_total": 0.0,
        }

        for step in range(1, n_steps + 1):
            bulk_batch = next(bulk_iter)
            sc_batch = next(sc_iter)

            xb = move_tensor(bulk_batch["x"], device)
            yb = move_tensor(bulk_batch["y"], device)
            xs = move_tensor(sc_batch["x"], device)
            ys = move_tensor(sc_batch["y"], device)

            with autocast_context(device, amp_enabled, amp_dtype):
                pred_b, z_b = model_forward_with_latent(model, xb, modality="bulk")
                pred_s, _ = model_forward_with_latent(model, xs, modality="sc")

                loss_b = criterion(pred_b, yb)
                loss_s = criterion(pred_s, ys)

                loss_p = torch.zeros((), device=device)
                loss_align = torch.zeros((), device=device)
                pb_scale = 1.0

                if step % pb_every == 0:
                    pb_batch = next(pb_iter)
                    xp = move_tensor(pb_batch["x"], device)
                    yp = move_tensor(pb_batch["y"], device)

                    pred_p, z_p = model_forward_with_latent(model, xp, modality="pseudobulk")
                    loss_p = criterion(pred_p, yp)
                    loss_align = mean_alignment_loss(z_b, z_p)
                    pb_scale = float(pb_every)

                raw_total_loss = (
                    weights["bulk"] * loss_b
                    + weights["sc"] * loss_s
                    + weights["pb"] * pb_scale * loss_p
                    + weights["align"] * pb_scale * loss_align
                )
                scaled_loss = raw_total_loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if step % grad_accum_steps == 0 or step == n_steps:
                if grad_clip_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            totals["train_bulk_loss"] += float(loss_b.item())
            totals["train_sc_loss"] += float(loss_s.item())
            totals["train_pb_loss"] += float(loss_p.item()) * pb_scale
            totals["train_align_loss"] += float(loss_align.item()) * pb_scale
            totals["train_total"] += float(raw_total_loss.item())

        for key in totals:
            totals[key] /= n_steps

        did_eval = (epoch % eval_every == 0) or (epoch == epochs)
        if did_eval:
            val_metrics = evaluate_all(
                model,
                bulk_val_loader,
                sc_val_loader,
                pb_val_loader,
                device,
                amp_enabled,
                amp_dtype,
                weights,
                prefix="val",
            )
        else:
            val_metrics = {
                "val_bulk_loss": float("nan"),
                "val_sc_loss": float("nan"),
                "val_pb_loss": float("nan"),
                "val_total": float("nan"),
            }

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if did_eval:
                    scheduler.step(val_metrics["val_total"])
            else:
                scheduler.step()

        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "model_kind": model_kind,
            "lr": current_lr(optimizer),
            **totals,
            **val_metrics,
        }
        history.append(row)

        if did_eval:
            print(
                f"epoch={epoch} "
                f"lr={row['lr']:.8f} "
                f"train_bulk_loss={row['train_bulk_loss']:.6f} "
                f"train_sc_loss={row['train_sc_loss']:.6f} "
                f"train_pb_loss={row['train_pb_loss']:.6f} "
                f"train_align_loss={row['train_align_loss']:.6f} "
                f"train_total={row['train_total']:.6f} "
                f"val_bulk_loss={row['val_bulk_loss']:.6f} "
                f"val_sc_loss={row['val_sc_loss']:.6f} "
                f"val_pb_loss={row['val_pb_loss']:.6f} "
                f"val_total={row['val_total']:.6f}"
            )
        else:
            print(
                f"epoch={epoch} "
                f"lr={row['lr']:.8f} "
                f"train_bulk_loss={row['train_bulk_loss']:.6f} "
                f"train_sc_loss={row['train_sc_loss']:.6f} "
                f"train_pb_loss={row['train_pb_loss']:.6f} "
                f"train_align_loss={row['train_align_loss']:.6f} "
                f"train_total={row['train_total']:.6f} "
                f"val_skipped=1"
            )

        save_checkpoint(
            ckpt_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            model_kind=model_kind,
            best_val_total=best_val_total,
        )

        if did_eval and float(val_metrics["val_total"]) < best_val_total:
            best_val_total = float(val_metrics["val_total"])
            best_epoch = epoch
            save_checkpoint(
                ckpt_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                cfg=cfg,
                model_kind=model_kind,
                best_val_total=best_val_total,
            )

        with (run_dir / "metrics_partial.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_val_total": best_val_total,
                    "history": history,
                },
                f,
                indent=2,
            )

    best_ckpt_path = ckpt_dir / "best.pt"
    if best_ckpt_path.exists():
        best_checkpoint = load_checkpoint(best_ckpt_path, model=model, device=device)
        best_epoch = int(best_checkpoint.get("epoch", best_epoch))
    else:
        best_checkpoint = load_checkpoint(ckpt_dir / "last.pt", model=model, device=device)
        best_epoch = int(best_checkpoint.get("epoch", best_epoch))

    test_metrics = evaluate_all(
        model,
        bulk_test_loader,
        sc_test_loader,
        pb_test_loader,
        device,
        amp_enabled,
        amp_dtype,
        weights,
        prefix="test",
    )

    summary = {
        "run_name": run_name,
        "device": str(device),
        "model_kind": model_kind,
        "parameter_counts": param_counts,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
        "grad_accum_steps": grad_accum_steps,
        "steps_per_epoch": n_steps,
        "pb_every": pb_every,
        "eval_every": eval_every,
        "best_epoch": best_epoch,
        "best_val_total": best_val_total,
        **test_metrics,
        "history": history,
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"best_epoch={best_epoch}")
    print(f"best_val_total={best_val_total:.6f}")
    print(f"test_bulk_loss={test_metrics['test_bulk_loss']:.6f}")
    print(f"test_sc_loss={test_metrics['test_sc_loss']:.6f}")
    print(f"test_pb_loss={test_metrics['test_pb_loss']:.6f}")
    print(f"test_total={test_metrics['test_total']:.6f}")
    print(f"saved outputs to {run_dir}")


if __name__ == "__main__":
    main()