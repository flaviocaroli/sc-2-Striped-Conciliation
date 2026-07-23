from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_

from sc2.config import load_yaml, merge_train_and_paths
from sc2.eval.benchmarks.evaluate_masked_reconstruction import (
    genewise_metrics,
    samplewise_metrics,
)
from sc2.losses.striped_full_losses import (
    compute_corruption_reconstruction_loss,
    compute_sc_objective,
)
from sc2.models.striped.sc2_striped_full import build_sc2_striped_full_from_config
from sc2.train import train_sc2_mamba_bridge as bridge
from sc2.train import train_sc2_striped_medium as medium_train


_STOP_REQUESTED = False
_STOP_SIGNAL: int | None = None


def _handle_stop_signal(signum: int, frame: Any) -> None:
    del frame
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = int(signum)
    print(
        f"received_signal={signum}; saving a resumable checkpoint at the next safe point",
        flush=True,
    )


def install_signal_handlers() -> None:
    for current_signal in (signal.SIGTERM, signal.SIGINT):
        signal.signal(current_signal, _handle_stop_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_stop_signal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a configurable 20+4 SC2 striped model with exact step-level resume."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path or 'auto' for the run's checkpoints/last.pt.",
    )
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def resolve_output_path(output_root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    path = Path(value)
    return path if path.is_absolute() else output_root / path


def resolve_resume_path(
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> Path | None:
    if args.fresh:
        return None
    if args.resume is not None:
        if str(args.resume).strip().lower() == "auto":
            candidate = checkpoint_dir / "last.pt"
            return candidate if candidate.exists() else None
        return Path(args.resume)
    if args.auto_resume:
        candidate = checkpoint_dir / "last.pt"
        return candidate if candidate.exists() else None
    return None


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    """Restore Python, NumPy, CPU torch, and CUDA RNG states safely."""
    if not state:
        return

    import random

    import numpy as np
    import torch

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = state.get("torch")
    if torch_state is not None:
        if not isinstance(torch_state, torch.Tensor):
            torch_state = torch.as_tensor(
                torch_state,
                dtype=torch.uint8,
            )

        torch_state = (
            torch_state.detach()
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
        )

        torch.set_rng_state(torch_state)

    cuda_state = state.get("cuda")
    if cuda_state is None:
        cuda_state = state.get("cuda_all")

    if (
        cuda_state is not None
        and torch.cuda.is_available()
    ):
        if isinstance(cuda_state, torch.Tensor):
            cuda_states = [cuda_state]
        else:
            cuda_states = list(cuda_state)

        normalized_cuda_states = []

        for item in cuda_states:
            if not isinstance(item, torch.Tensor):
                item = torch.as_tensor(
                    item,
                    dtype=torch.uint8,
                )

            normalized_cuda_states.append(
                item.detach()
                .to(device="cpu", dtype=torch.uint8)
                .contiguous()
            )

        available_devices = torch.cuda.device_count()

        if len(normalized_cuda_states) == available_devices:
            torch.cuda.set_rng_state_all(
                normalized_cuda_states
            )
        elif normalized_cuda_states:
            current_device = torch.cuda.current_device()

            source_index = min(
                current_device,
                len(normalized_cuda_states) - 1,
            )

            torch.cuda.set_rng_state(
                normalized_cuda_states[source_index],
                device=current_device,
            )


def group_name(name: str, n_mamba_blocks: int, upper_mamba_blocks: int) -> str:
    if name.startswith(("reconstruction_head.", "dropout_head.", "latent_norm.")):
        return "heads"
    if name.startswith("attention_blocks."):
        return "upper"
    if name.startswith("mamba_blocks."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            index = int(parts[1])
            if index >= max(0, n_mamba_blocks - upper_mamba_blocks):
                return "upper"
    return "base"


def build_optimizer(
    model: nn.Module,
    *,
    n_mamba_blocks: int,
    upper_mamba_blocks: int,
    weight_decay: float,
    betas: tuple[float, float],
) -> torch.optim.AdamW:
    grouped: dict[str, list[nn.Parameter]] = {"heads": [], "upper": [], "base": []}
    for name, parameter in model.named_parameters():
        grouped[group_name(name, n_mamba_blocks, upper_mamba_blocks)].append(parameter)
    return torch.optim.AdamW(
        [
            {"params": grouped["heads"], "name": "heads", "lr": 0.0},
            {"params": grouped["upper"], "name": "upper", "lr": 0.0},
            {"params": grouped["base"], "name": "base", "lr": 0.0},
        ],
        weight_decay=float(weight_decay),
        betas=betas,
    )


def configure_train_groups(
    model: nn.Module,
    stage: Mapping[str, Any],
    *,
    n_mamba_blocks: int,
    upper_mamba_blocks: int,
) -> dict[str, int]:
    requested = {str(value) for value in stage.get("train_groups", ["heads", "upper", "base"])}
    if "all" in requested:
        requested = {"heads", "upper", "base"}
    unknown = requested - {"heads", "upper", "base"}
    if unknown:
        raise ValueError(f"Unknown train_groups: {sorted(unknown)}")

    counts = {"heads": 0, "upper": 0, "base": 0}
    for name, parameter in model.named_parameters():
        current_group = group_name(name, n_mamba_blocks, upper_mamba_blocks)
        parameter.requires_grad = current_group in requested
        if parameter.requires_grad:
            counts[current_group] += parameter.numel()
    return counts


def cosine_scale(progress: float, end_ratio: float, warmup_ratio: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    warmup_ratio = min(max(float(warmup_ratio), 0.0), 0.99)
    if warmup_ratio > 0.0 and progress < warmup_ratio:
        return max(progress / warmup_ratio, 1.0e-3)
    adjusted = (progress - warmup_ratio) / max(1.0 - warmup_ratio, 1.0e-8)
    return float(end_ratio) + (1.0 - float(end_ratio)) * 0.5 * (
        1.0 + math.cos(math.pi * adjusted)
    )


def set_stage_learning_rates(
    optimizer: torch.optim.Optimizer,
    stage: Mapping[str, Any],
    progress: float,
) -> dict[str, float]:
    scale = cosine_scale(
        progress,
        end_ratio=float(stage.get("lr_end_ratio", 0.10)),
        warmup_ratio=float(stage.get("warmup_ratio", 0.05)),
    )
    values = {
        "heads": float(stage.get("lr_heads", stage.get("lr", 1.0e-4))) * scale,
        "upper": float(stage.get("lr_upper", stage.get("lr", 1.0e-4))) * scale,
        "base": float(stage.get("lr_base", stage.get("lr", 1.0e-4))) * scale,
    }
    for parameter_group in optimizer.param_groups:
        name = str(parameter_group["name"])
        active = any(parameter.requires_grad for parameter in parameter_group["params"])
        parameter_group["lr"] = values[name] if active else 0.0
    return {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}


def forward_dict(model: nn.Module, x: torch.Tensor, modality: str) -> dict[str, torch.Tensor]:
    output = model(x, modality=modality, return_dict=True)
    if not isinstance(output, dict):
        raise TypeError("SC2 striped full training requires return_dict=True")
    return output


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


@torch.inference_mode()
def evaluate_sc_recovery(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    *,
    prediction_key: str,
    zero_fill_threshold: float,
) -> dict[str, float]:
    model.eval()
    prediction_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []

    for batch in loader:
        observed_x = bridge.move_tensor(batch["x"], device)
        clean_target = bridge.move_tensor(batch["y"], device)
        with bridge.autocast_context(device, amp_enabled, amp_dtype):
            outputs = forward_dict(model, observed_x, modality="sc")
            prediction = outputs[prediction_key]
        mask = (observed_x <= 1.0e-8) & (clean_target > 1.0e-8)
        prediction_chunks.append(prediction.float().cpu().numpy())
        target_chunks.append(clean_target.float().cpu().numpy())
        mask_chunks.append(mask.cpu().numpy())

    prediction_np = np.concatenate(prediction_chunks, axis=0)
    target_np = np.concatenate(target_chunks, axis=0)
    mask_np = np.concatenate(mask_chunks, axis=0).astype(bool)

    if not bool(mask_np.any()):
        raise RuntimeError("Validation loader produced no synthetically masked positives")

    difference = prediction_np[mask_np] - target_np[mask_np]
    sample_df = samplewise_metrics(prediction_np, target_np, mask_np)
    gene_df = genewise_metrics(
        prediction_np,
        target_np,
        mask_np,
        gene_names=[str(index) for index in range(prediction_np.shape[1])],
    )

    masked_prediction = prediction_np[mask_np]
    masked_target = target_np[mask_np]
    prediction_std = float(np.std(masked_prediction))
    target_std = float(np.std(masked_target))
    std_ratio = prediction_std / target_std if target_std > 0.0 else float("nan")
    true_zero = target_np <= 1.0e-8
    zero_fill = (
        float((prediction_np[true_zero] > float(zero_fill_threshold)).mean())
        if bool(true_zero.any())
        else float("nan")
    )

    return {
        "masked_mse": float(np.mean(difference**2)),
        "masked_rmse": float(np.sqrt(np.mean(difference**2))),
        "masked_mae": float(np.mean(np.abs(difference))),
        "sample_pearson": finite_mean(sample_df["pearson"].to_numpy()),
        "sample_spearman": finite_mean(sample_df["spearman"].to_numpy()),
        "gene_pearson": finite_mean(gene_df["pearson"].to_numpy()),
        "gene_spearman": finite_mean(gene_df["spearman"].to_numpy()),
        "prediction_mean": float(masked_prediction.mean()),
        "prediction_std": prediction_std,
        "target_mean": float(masked_target.mean()),
        "target_std": target_std,
        "std_ratio": float(std_ratio),
        "raw_zero_fill_rate": zero_fill,
        "n_masked": float(mask_np.sum()),
    }


def checkpoint_eligibility(metrics: Mapping[str, float], selection_cfg: Mapping[str, Any]) -> bool:
    std_ratio = float(metrics["std_ratio"])
    zero_fill = float(metrics["raw_zero_fill_rate"])
    min_std = float(selection_cfg.get("min_std_ratio", 0.15))
    max_zero_fill = float(selection_cfg.get("max_zero_fill_rate", 0.25))
    return (
        math.isfinite(std_ratio)
        and std_ratio >= min_std
        and math.isfinite(zero_fill)
        and zero_fill <= max_zero_fill
    )


def checkpoint_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
    return (
        float(metrics["masked_mse"]),
        -float(metrics["sample_spearman"]),
        -float(metrics["gene_spearman"]),
        abs(float(metrics["std_ratio"]) - 1.0),
        float(metrics["raw_zero_fill_rate"]),
    )


def make_cursor(stage_index: int, epoch_index: int, next_step: int) -> dict[str, int]:
    return {
        "stage_index": int(stage_index),
        "epoch_index": int(epoch_index),
        "next_step": int(next_step),
    }


def next_epoch_cursor(
    stages: list[dict[str, Any]],
    stage_index: int,
    epoch_index: int,
) -> dict[str, int]:
    if epoch_index + 1 < int(stages[stage_index]["epochs"]):
        return make_cursor(stage_index, epoch_index + 1, 1)
    if stage_index + 1 < len(stages):
        return make_cursor(stage_index + 1, 0, 1)
    return make_cursor(len(stages), 0, 1)


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: dict[str, Any],
    cursor: Mapping[str, int],
    global_step: int,
    completed_epochs: int,
    history: list[dict[str, Any]],
    best_any_key: tuple[float, ...] | None,
    best_eligible_key: tuple[float, ...] | None,
    best_any_epoch: int | None,
    best_eligible_epoch: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_kind": "sc2_striped_full",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "cursor": dict(cursor),
        "global_step": int(global_step),
        "completed_epochs": int(completed_epochs),
        "history": history,
        "best_any_key": list(best_any_key) if best_any_key is not None else None,
        "best_eligible_key": list(best_eligible_key) if best_eligible_key is not None else None,
        "best_any_epoch": best_any_epoch,
        "best_eligible_epoch": best_eligible_epoch,
        "rng_state": capture_rng_state(),
        "saved_at_unix": time.time(),
    }
    if scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def save_training_checkpoint(path: Path, **kwargs: Any) -> None:
    atomic_torch_save(checkpoint_payload(**kwargs), path)


def merge_loss_config(
    base_loss_cfg: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base_loss_cfg))
    merged.update(dict(stage.get("loss_overrides", {})))
    return merged


def main() -> None:
    install_signal_handlers()
    args = parse_args()
    train_cfg = load_yaml(args.config)
    paths_cfg = load_yaml(args.paths)
    cfg = merge_train_and_paths(train_cfg, paths_cfg)

    seed = int(cfg.get("seed", 42))
    bridge.seed_everything(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_root = Path(cfg["paths"]["output_root"])
    run_name = str(cfg["run_name"])
    run_dir = output_root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(cfg, run_dir / "resolved_config.json")

    device = bridge.get_device(cfg.get("device", "auto"))
    loaders, input_dim = medium_train.make_loaders(cfg, seed)
    model = build_sc2_striped_full_from_config(cfg["model"], n_genes=input_dim).to(device)
    model_kind = "sc2_striped_full"

    train_section = cfg["train"]
    n_mamba_blocks = int(cfg["model"]["n_mamba_blocks"])
    upper_mamba_blocks = int(train_section.get("upper_mamba_blocks", 6))
    optimizer = build_optimizer(
        model,
        n_mamba_blocks=n_mamba_blocks,
        upper_mamba_blocks=upper_mamba_blocks,
        weight_decay=float(train_section.get("weight_decay", 1.0e-4)),
        betas=(
            float(train_section.get("beta1", 0.9)),
            float(train_section.get("beta2", 0.95)),
        ),
    )

    amp_enabled = bridge.resolve_amp(train_section, device)
    amp_dtype = bridge.resolve_amp_dtype(train_section, device) if amp_enabled else None
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and device.type == "cuda" and amp_dtype == torch.float16,
    )

    stages = [dict(stage) for stage in train_section["stages"]]
    resume_path = resolve_resume_path(args, checkpoint_dir)
    cursor = make_cursor(0, 0, 1)
    global_step = 0
    completed_epochs = 0
    history: list[dict[str, Any]] = []
    best_any_key: tuple[float, ...] | None = None
    best_eligible_key: tuple[float, ...] | None = None
    best_any_epoch: int | None = None
    best_eligible_epoch: int | None = None

    if resume_path is not None:
        checkpoint_data = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        if scaler.is_enabled() and "scaler_state_dict" in checkpoint_data:
            scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
        cursor = dict(checkpoint_data.get("cursor", cursor))
        global_step = int(checkpoint_data.get("global_step", 0))
        completed_epochs = int(checkpoint_data.get("completed_epochs", 0))
        history = list(checkpoint_data.get("history", []))
        stored_any = checkpoint_data.get("best_any_key")
        stored_eligible = checkpoint_data.get("best_eligible_key")
        best_any_key = tuple(float(value) for value in stored_any) if stored_any else None
        best_eligible_key = (
            tuple(float(value) for value in stored_eligible) if stored_eligible else None
        )
        best_any_epoch = checkpoint_data.get("best_any_epoch")
        best_eligible_epoch = checkpoint_data.get("best_eligible_epoch")
        restore_rng_state(checkpoint_data.get("rng_state"))
        print(
            f"resumed_from={resume_path} cursor={cursor} global_step={global_step} "
            f"completed_epochs={completed_epochs}",
            flush=True,
        )
    else:
        init_path = resolve_output_path(output_root, train_section.get("init_checkpoint_path"))
        if init_path is None:
            print("initialization=from_scratch", flush=True)
        else:
            if not init_path.exists():
                raise FileNotFoundError(f"Initial checkpoint not found: {init_path}")
            initial = torch.load(init_path, map_location=device)
            strict_init = bool(train_section.get("strict_init", True))
            load_result = model.load_state_dict(initial["model_state_dict"], strict=strict_init)
            print(
                f"initialized_model_weights_from={init_path} strict={strict_init} "
                f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}",
                flush=True,
            )
        print("optimizer_reset=1 scheduler_reset=1", flush=True)

    parameter_counts = bridge.count_parameters(model)
    print(f"device={device}")
    print(f"model_kind={model_kind}")
    print(f"parameters_total={parameter_counts['total']}")
    print(f"parameters_trainable={parameter_counts['trainable']}")
    print(f"attention_positions={getattr(model, 'attention_positions', None)}")

    if int(cursor["stage_index"]) >= len(stages):
        print("training_already_complete=1", flush=True)
    else:
        grad_accum_steps = max(1, int(train_section.get("grad_accum_steps", 4)))
        grad_clip_norm = float(train_section.get("grad_clip_norm", 1.0))
        save_every_steps = max(1, int(train_section.get("save_every_steps", 20)))
        pb_every = max(1, int(train_section.get("pb_every", 4)))
        base_loss_cfg = dict(train_section.get("loss", {}))
        selection_cfg = dict(train_section.get("selection", {}))
        prediction_key = str(selection_cfg.get("prediction_key", "reconstruction"))
        zero_fill_threshold = float(selection_cfg.get("zero_fill_threshold", 0.10))

        for stage_index in range(int(cursor["stage_index"]), len(stages)):
            stage = stages[stage_index]
            stage_epochs = int(stage["epochs"])
            epoch_start = int(cursor["epoch_index"]) if stage_index == int(cursor["stage_index"]) else 0
            trainable_counts = configure_train_groups(
                model,
                stage,
                n_mamba_blocks=n_mamba_blocks,
                upper_mamba_blocks=upper_mamba_blocks,
            )

            for epoch_index in range(epoch_start, stage_epochs):
                steps_per_epoch = int(stage["steps_per_epoch"])
                start_step = (
                    int(cursor["next_step"])
                    if stage_index == int(cursor["stage_index"])
                    and epoch_index == int(cursor["epoch_index"])
                    else 1
                )
                if start_step > steps_per_epoch:
                    continue

                stage_name = str(stage["name"])
                stage_total_steps = stage_epochs * steps_per_epoch
                stage_step_offset = epoch_index * steps_per_epoch
                active_loss_cfg = merge_loss_config(base_loss_cfg, stage)
                modality_weights = dict(stage.get("modality_weights", {}))
                sc_weight = float(modality_weights.get("sc", 1.0))
                bulk_weight = float(modality_weights.get("bulk", 0.25))
                pb_weight = float(modality_weights.get("pseudobulk", 0.25))
                unpaired_align_weight = float(modality_weights.get("unpaired_align", 0.0))

                print(
                    f"stage={stage_name} stage_index={stage_index + 1}/{len(stages)} "
                    f"epoch={epoch_index + 1}/{stage_epochs} start_step={start_step}/{steps_per_epoch} "
                    f"trainable_by_group={trainable_counts} loss={active_loss_cfg.get('name')} "
                    f"modality_weights={modality_weights}",
                    flush=True,
                )

                model.train()
                sc_iterator = bridge.infinite_loader(loaders["sc_train"])
                bulk_iterator = bridge.infinite_loader(loaders["bulk_train"])
                pb_iterator = bridge.infinite_loader(loaders["pb_train"])
                optimizer.zero_grad(set_to_none=True)

                totals: dict[str, float] = {
                    "train_total": 0.0,
                    "train_sc": 0.0,
                    "train_bulk": 0.0,
                    "train_pseudobulk": 0.0,
                    "train_unpaired_align": 0.0,
                }
                processed_steps = 0

                for step in range(start_step, steps_per_epoch + 1):
                    progress = (stage_step_offset + step - 1) / float(max(stage_total_steps - 1, 1))
                    lrs = set_stage_learning_rates(optimizer, stage, progress)

                    sc_batch = next(sc_iterator)
                    bulk_batch = next(bulk_iterator)
                    observed_sc = bridge.move_tensor(sc_batch["x"], device)
                    target_sc = bridge.move_tensor(sc_batch["y"], device)
                    observed_bulk = bridge.move_tensor(bulk_batch["x"], device)
                    target_bulk = bridge.move_tensor(bulk_batch["y"], device)

                    with bridge.autocast_context(device, amp_enabled, amp_dtype):
                        sc_outputs = forward_dict(model, observed_sc, modality="sc")
                        bulk_outputs = forward_dict(model, observed_bulk, modality="bulk")
                        sc_objective = compute_sc_objective(
                            sc_outputs,
                            observed_sc,
                            target_sc,
                            active_loss_cfg,
                        )
                        bulk_loss = compute_corruption_reconstruction_loss(
                            bulk_outputs,
                            observed_bulk,
                            target_bulk,
                            active_loss_cfg.get("bulk", {}),
                        )

                        pb_loss = torch.zeros((), device=device)
                        pb_scale = 1.0
                        pb_outputs: dict[str, torch.Tensor] | None = None
                        if step % pb_every == 0:
                            pb_batch = next(pb_iterator)
                            observed_pb = bridge.move_tensor(pb_batch["x"], device)
                            target_pb = bridge.move_tensor(pb_batch["y"], device)
                            pb_outputs = forward_dict(model, observed_pb, modality="pseudobulk")
                            pb_loss = compute_corruption_reconstruction_loss(
                                pb_outputs,
                                observed_pb,
                                target_pb,
                                active_loss_cfg.get("pseudobulk", {}),
                            )
                            pb_scale = float(pb_every)

                        align_loss = torch.zeros((), device=device)
                        if unpaired_align_weight > 0.0 and pb_outputs is not None:
                            bulk_latent = bulk_outputs["latent"].mean(dim=1)
                            pb_latent = pb_outputs["latent"].mean(dim=1)
                            common = min(bulk_latent.shape[0], pb_latent.shape[0])
                            align_loss = 1.0 - F.cosine_similarity(
                                bulk_latent[:common],
                                pb_latent[:common],
                                dim=1,
                            ).mean()

                        raw_total = (
                            sc_weight * sc_objective.loss
                            + bulk_weight * bulk_loss
                            + pb_weight * pb_scale * pb_loss
                            + unpaired_align_weight * align_loss
                        )
                        scaled_total = raw_total / float(grad_accum_steps)

                    if scaler.is_enabled():
                        scaler.scale(scaled_total).backward()
                    else:
                        scaled_total.backward()

                    optimizer_step = step % grad_accum_steps == 0 or step == steps_per_epoch
                    if optimizer_step:
                        trainable_parameters = [
                            parameter for parameter in model.parameters() if parameter.requires_grad
                        ]
                        if grad_clip_norm > 0.0:
                            if scaler.is_enabled():
                                scaler.unscale_(optimizer)
                            clip_grad_norm_(trainable_parameters, grad_clip_norm)
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    processed_steps += 1
                    totals["train_total"] += float(raw_total.item())
                    totals["train_sc"] += float(sc_objective.loss.item())
                    totals["train_bulk"] += float(bulk_loss.item())
                    totals["train_pseudobulk"] += float(pb_scale * pb_loss.item())
                    totals["train_unpaired_align"] += float(align_loss.item())
                    for component_name, component_value in sc_objective.components.items():
                        key = f"train_sc_{component_name}"
                        totals[key] = totals.get(key, 0.0) + float(component_value.item())

                    next_cursor = make_cursor(stage_index, epoch_index, step + 1)
                    should_save = (
                        step < steps_per_epoch
                        and step % save_every_steps == 0
                        and optimizer_step
                    )
                    stop_mid_epoch = (
                        _STOP_REQUESTED
                        and step < steps_per_epoch
                        and optimizer_step
                    )
                    if should_save or stop_mid_epoch:
                        save_training_checkpoint(
                            checkpoint_dir / "last.pt",
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            cfg=cfg,
                            cursor=next_cursor,
                            global_step=global_step,
                            completed_epochs=completed_epochs,
                            history=history,
                            best_any_key=best_any_key,
                            best_eligible_key=best_eligible_key,
                            best_any_epoch=best_any_epoch,
                            best_eligible_epoch=best_eligible_epoch,
                        )
                        print(
                            f"checkpoint_saved=last.pt stage={stage_name} "
                            f"epoch={epoch_index + 1} step={step}/{steps_per_epoch} "
                            f"next_cursor={next_cursor}",
                            flush=True,
                        )
                        if stop_mid_epoch:
                            print(
                                f"stop_requested=1 signal={_STOP_SIGNAL} "
                                "resume_with='same sbatch command --auto-resume'",
                                flush=True,
                            )
                            return

                if processed_steps == 0:
                    continue
                for key in totals:
                    totals[key] /= float(processed_steps)

                validation = evaluate_sc_recovery(
                    model,
                    loaders["sc_val"],
                    device,
                    amp_enabled,
                    amp_dtype,
                    prediction_key=prediction_key,
                    zero_fill_threshold=zero_fill_threshold,
                )
                eligible = checkpoint_eligibility(validation, selection_cfg)
                current_key = checkpoint_key(validation)
                completed_epochs += 1
                epoch_cursor = next_epoch_cursor(stages, stage_index, epoch_index)
                row: dict[str, Any] = {
                    "completed_epoch": completed_epochs,
                    "stage": stage_name,
                    "stage_epoch": epoch_index + 1,
                    "global_step": global_step,
                    "lrs": lrs,
                    "trainable_parameters": int(sum(trainable_counts.values())),
                    "checkpoint_eligible": eligible,
                    **totals,
                    **{f"val_{key}": value for key, value in validation.items()},
                }

                is_best_any = best_any_key is None or current_key < best_any_key
                is_best_eligible = eligible and (
                    best_eligible_key is None or current_key < best_eligible_key
                )
                row["is_best_any"] = is_best_any
                row["is_best_eligible"] = is_best_eligible
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

                common_checkpoint_args = dict(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    cfg=cfg,
                    cursor=epoch_cursor,
                    global_step=global_step,
                    completed_epochs=completed_epochs,
                    history=history,
                    best_any_key=best_any_key,
                    best_eligible_key=best_eligible_key,
                    best_any_epoch=best_any_epoch,
                    best_eligible_epoch=best_eligible_epoch,
                )

                if is_best_any:
                    best_any_key = current_key
                    best_any_epoch = completed_epochs
                    common_checkpoint_args.update(
                        best_any_key=best_any_key,
                        best_any_epoch=best_any_epoch,
                    )
                    save_training_checkpoint(checkpoint_dir / "best_any.pt", **common_checkpoint_args)
                if is_best_eligible:
                    best_eligible_key = current_key
                    best_eligible_epoch = completed_epochs
                    common_checkpoint_args.update(
                        best_eligible_key=best_eligible_key,
                        best_eligible_epoch=best_eligible_epoch,
                    )
                    save_training_checkpoint(
                        checkpoint_dir / "best_eligible.pt", **common_checkpoint_args
                    )

                save_training_checkpoint(
                    checkpoint_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    cfg=cfg,
                    cursor=epoch_cursor,
                    global_step=global_step,
                    completed_epochs=completed_epochs,
                    history=history,
                    best_any_key=best_any_key,
                    best_eligible_key=best_eligible_key,
                    best_any_epoch=best_any_epoch,
                    best_eligible_epoch=best_eligible_epoch,
                )
                atomic_json_dump(
                    {
                        "history": history,
                        "best_any_epoch": best_any_epoch,
                        "best_eligible_epoch": best_eligible_epoch,
                        "recovery_gate_passed": best_eligible_epoch is not None,
                    },
                    run_dir / "metrics_partial.json",
                )
                cursor = epoch_cursor
                if _STOP_REQUESTED:
                    print(
                        f"stop_requested_after_epoch=1 signal={_STOP_SIGNAL} "
                        "resume_with='same sbatch command --auto-resume'",
                        flush=True,
                    )
                    return

    best_eligible_path = checkpoint_dir / "best_eligible.pt"
    best_any_path = checkpoint_dir / "best_any.pt"
    selected_source = best_eligible_path if best_eligible_path.exists() else best_any_path
    if not selected_source.exists():
        raise RuntimeError("Training produced neither best_eligible.pt nor best_any.pt")
    shutil.copy2(selected_source, checkpoint_dir / "best.pt")
    selected_checkpoint = torch.load(selected_source, map_location=device)
    model.load_state_dict(selected_checkpoint["model_state_dict"], strict=True)

    selection_cfg = dict(train_section.get("selection", {}))
    test_metrics = evaluate_sc_recovery(
        model,
        loaders["sc_test"],
        device,
        amp_enabled,
        amp_dtype,
        prediction_key=str(selection_cfg.get("prediction_key", "reconstruction")),
        zero_fill_threshold=float(selection_cfg.get("zero_fill_threshold", 0.10)),
    )
    summary = {
        "run_name": run_name,
        "model_kind": model_kind,
        "parameter_counts": parameter_counts,
        "attention_positions": getattr(model, "attention_positions", None),
        "selected_checkpoint": selected_source.name,
        "best_any_epoch": best_any_epoch,
        "best_eligible_epoch": best_eligible_epoch,
        "recovery_gate_passed": best_eligible_epoch is not None,
        "test_recovery": test_metrics,
        "history": history,
    }
    atomic_json_dump(summary, run_dir / "metrics.json")
    print(f"selected_checkpoint={selected_source}")
    print(f"recovery_gate_passed={best_eligible_epoch is not None}")
    print(json.dumps({f"test_{key}": value for key, value in test_metrics.items()}, sort_keys=True))
    print(f"saved_outputs_to={run_dir}")


if __name__ == "__main__":
    main()
