from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

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
from sc2.losses.dropout_losses import dropout_bce_loss
from sc2.losses.recovery_losses import single_cell_recovery_loss
from sc2.train import train_sc2_mamba_bridge as bridge
from sc2.train import train_sc2_striped_medium as v1


_STOP_REQUESTED = False
_STOP_SIGNAL: int | None = None


def _handle_stop_signal(signum: int, frame: Any) -> None:
    del frame
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = signum
    print(
        f"received_signal={signum}; checkpoint will be saved at the next safe point",
        flush=True,
    )


def install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handle_stop_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_stop_signal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recovery-focused SC2-medium-v2 fine-tuning from SC2-medium-v1."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="V2 checkpoint path, or 'auto' for the v2 run's checkpoints/last.pt.",
    )
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def resolve_output_path(output_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else output_root / path


def resolve_resume_path(args: argparse.Namespace, checkpoint_dir: Path) -> Path | None:
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


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parameter_group_name(name: str, n_mamba_blocks: int, upper_mamba_blocks: int) -> str:
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
) -> torch.optim.AdamW:
    grouped: dict[str, list[nn.Parameter]] = {"heads": [], "upper": [], "base": []}
    for name, parameter in model.named_parameters():
        grouped[parameter_group_name(name, n_mamba_blocks, upper_mamba_blocks)].append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": grouped["heads"], "name": "heads", "lr": 0.0},
            {"params": grouped["upper"], "name": "upper", "lr": 0.0},
            {"params": grouped["base"], "name": "base", "lr": 0.0},
        ],
        weight_decay=float(weight_decay),
    )
    return optimizer


def configure_stage(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stage: dict[str, Any],
    *,
    n_mamba_blocks: int,
    upper_mamba_blocks: int,
) -> dict[str, int]:
    train_groups = {str(value) for value in stage.get("train_groups", ["heads"])}
    valid_groups = {"heads", "upper", "base"}
    unknown = train_groups - valid_groups
    if unknown:
        raise ValueError(f"Unknown train_groups: {sorted(unknown)}")

    trainable_counts = {"heads": 0, "upper": 0, "base": 0}
    for name, parameter in model.named_parameters():
        group = parameter_group_name(name, n_mamba_blocks, upper_mamba_blocks)
        parameter.requires_grad = group in train_groups
        if parameter.requires_grad:
            trainable_counts[group] += parameter.numel()

    stage_lrs = {
        "heads": float(stage.get("lr_heads", 0.0)),
        "upper": float(stage.get("lr_upper", 0.0)),
        "base": float(stage.get("lr_base", 0.0)),
    }
    for group in optimizer.param_groups:
        group_name = str(group["name"])
        group["lr"] = stage_lrs[group_name] if group_name in train_groups else 0.0

    return trainable_counts


def current_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}


def forward_dict(model: nn.Module, x: torch.Tensor, modality: str) -> dict[str, torch.Tensor]:
    output = model(x, modality=modality, return_dict=True)
    if not isinstance(output, dict):
        raise TypeError("SC2-medium-v2 requires return_dict=True to return a dictionary")
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
    zero_fill_threshold: float,
) -> dict[str, float]:
    model.eval()
    prediction_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []

    masked_squared_error = 0.0
    masked_absolute_error = 0.0
    n_masked = 0
    dropout_loss_weighted = 0.0
    dropout_supervised = 0
    dropout_positive = 0

    for batch in loader:
        observed_x = bridge.move_tensor(batch["x"], device)
        clean_target = bridge.move_tensor(batch["y"], device)

        with bridge.autocast_context(device, amp_enabled, amp_dtype):
            output = forward_dict(model, observed_x, modality="sc")
            prediction = output["reconstruction"]
            dropout = dropout_bce_loss(
                output.get("dropout_logits"),
                observed_x,
                clean_target,
            )

        mask = (observed_x <= 1.0e-8) & (clean_target > 1.0e-8)
        if bool(mask.any()):
            difference = prediction[mask] - clean_target[mask]
            masked_squared_error += float(difference.square().sum().item())
            masked_absolute_error += float(difference.abs().sum().item())
            n_masked += int(mask.sum().item())

        dropout_loss_weighted += float(dropout.loss.item()) * max(dropout.n_supervised, 1)
        dropout_supervised += int(dropout.n_supervised)
        dropout_positive += int(dropout.n_positive)

        prediction_chunks.append(prediction.float().cpu().numpy())
        target_chunks.append(clean_target.float().cpu().numpy())
        mask_chunks.append(mask.cpu().numpy())

    prediction_np = np.concatenate(prediction_chunks, axis=0)
    target_np = np.concatenate(target_chunks, axis=0)
    mask_np = np.concatenate(mask_chunks, axis=0).astype(bool)

    sample_df = samplewise_metrics(prediction_np, target_np, mask_np)
    gene_df = genewise_metrics(
        prediction_np,
        target_np,
        mask_np,
        gene_names=[str(index) for index in range(prediction_np.shape[1])],
    )

    masked_prediction = prediction_np[mask_np]
    masked_target = target_np[mask_np]
    prediction_std = float(np.std(masked_prediction)) if masked_prediction.size else float("nan")
    target_std = float(np.std(masked_target)) if masked_target.size else float("nan")
    std_ratio = prediction_std / target_std if target_std > 0.0 else float("nan")

    true_zero = target_np <= 1.0e-8
    raw_zero_fill_rate = (
        float((prediction_np[true_zero] > float(zero_fill_threshold)).mean())
        if bool(true_zero.any())
        else float("nan")
    )

    return {
        "masked_mse": masked_squared_error / float(max(n_masked, 1)),
        "masked_rmse": math.sqrt(masked_squared_error / float(max(n_masked, 1))),
        "masked_mae": masked_absolute_error / float(max(n_masked, 1)),
        "sample_pearson": finite_mean(sample_df["pearson"].to_numpy()),
        "sample_spearman": finite_mean(sample_df["spearman"].to_numpy()),
        "gene_pearson": finite_mean(gene_df["pearson"].to_numpy()),
        "gene_spearman": finite_mean(gene_df["spearman"].to_numpy()),
        "prediction_mean": float(masked_prediction.mean()) if masked_prediction.size else float("nan"),
        "prediction_std": prediction_std,
        "target_mean": float(masked_target.mean()) if masked_target.size else float("nan"),
        "target_std": target_std,
        "std_ratio": float(std_ratio),
        "raw_zero_fill_rate": raw_zero_fill_rate,
        "dropout_loss": dropout_loss_weighted / float(max(dropout_supervised, 1)),
        "dropout_positive_rate": float(dropout_positive) / float(max(dropout_supervised, 1)),
        "n_masked": float(n_masked),
    }


def selection_key(metrics: dict[str, float], min_std_ratio: float) -> tuple[float, ...]:
    std_ratio = float(metrics["std_ratio"])
    masked_mse = float(metrics["masked_mse"])
    sample_spearman = float(metrics["sample_spearman"])
    zero_fill = float(metrics["raw_zero_fill_rate"])
    eligible = math.isfinite(std_ratio) and std_ratio >= float(min_std_ratio)
    if eligible:
        return (0.0, masked_mse, -sample_spearman, zero_fill)
    return (1.0, -std_ratio if math.isfinite(std_ratio) else float("inf"), masked_mse, -sample_spearman)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: dict[str, Any],
    completed_global_epoch: int,
    best_epoch: int,
    best_key: tuple[float, ...] | None,
    history: list[dict[str, Any]],
    stage_name: str,
    stage_epoch: int,
    step: int,
    steps_per_epoch: int,
    partial_epoch: bool,
) -> None:
    payload: dict[str, Any] = {
        "epoch": int(completed_global_epoch),
        "completed_global_epoch": int(completed_global_epoch),
        "model_kind": "sc2_striped_medium",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "best_epoch": int(best_epoch),
        "best_selection_key": list(best_key) if best_key is not None else None,
        "history": history,
        "stage_name": stage_name,
        "stage_epoch": int(stage_epoch),
        "step": int(step),
        "n_steps": int(steps_per_epoch),
        "partial_epoch": bool(partial_epoch),
        "saved_at_unix": time.time(),
    }
    if scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()
    atomic_torch_save(payload, path)


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
    bridge.ensure_dir(run_dir)
    bridge.ensure_dir(checkpoint_dir)

    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    device = bridge.get_device(cfg.get("device", "auto"))
    loaders, input_dim = v1.make_loaders(cfg, seed)
    model_kind, model = v1.build_model(cfg.get("model", {}), n_genes=input_dim)
    model = model.to(device)

    train_section = cfg["train"]
    n_mamba_blocks = int(cfg["model"]["n_mamba_blocks"])
    upper_mamba_blocks = int(train_section.get("upper_mamba_blocks", 4))
    optimizer = build_optimizer(
        model,
        n_mamba_blocks=n_mamba_blocks,
        upper_mamba_blocks=upper_mamba_blocks,
        weight_decay=float(train_section.get("weight_decay", 1.0e-4)),
    )

    amp_enabled = bridge.resolve_amp(train_section, device)
    amp_dtype = bridge.resolve_amp_dtype(train_section, device) if amp_enabled else None
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and device.type == "cuda" and amp_dtype == torch.float16,
    )

    resume_path = resolve_resume_path(args, checkpoint_dir)
    completed_global_epoch = 0
    best_epoch = 0
    best_key: tuple[float, ...] | None = None
    history: list[dict[str, Any]] = []

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler.is_enabled() and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        completed_global_epoch = int(checkpoint.get("completed_global_epoch", checkpoint.get("epoch", 0)))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        stored_key = checkpoint.get("best_selection_key")
        best_key = tuple(float(value) for value in stored_key) if stored_key is not None else None
        history = list(checkpoint.get("history", []))
        print(
            f"resumed_from={resume_path} completed_global_epoch={completed_global_epoch} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
    else:
        init_path = resolve_output_path(output_root, train_section["init_checkpoint_path"])
        if not init_path.exists():
            raise FileNotFoundError(f"Initial SC2-medium-v1 checkpoint not found: {init_path}")
        checkpoint = torch.load(init_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"initialized_model_weights_from={init_path}", flush=True)
        print("optimizer_reset=1 scheduler_reset=1", flush=True)

    counts = bridge.count_parameters(model)
    print(f"model_kind={model_kind}")
    print(f"parameters_total={counts['total']}")

    stages = list(train_section["stages"])
    total_epochs = sum(int(stage["epochs"]) for stage in stages)
    grad_accum_steps = max(1, int(train_section.get("grad_accum_steps", 2)))
    grad_clip_norm = float(train_section.get("grad_clip_norm", 1.0))
    save_every_steps = max(0, int(train_section.get("save_every_steps", 25)))
    pb_every = max(1, int(train_section.get("pb_every", 4)))
    min_std_ratio = float(train_section.get("min_std_ratio_for_best", 0.20))
    zero_fill_threshold = float(train_section.get("zero_fill_threshold", 0.10))

    global_epoch_slot = 0
    for stage_index, stage in enumerate(stages, start=1):
        stage_name = str(stage["name"])
        stage_epochs = int(stage["epochs"])
        steps_per_epoch = int(stage["steps_per_epoch"])

        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch_slot += 1
            if global_epoch_slot <= completed_global_epoch:
                continue

            trainable_counts = configure_stage(
                model,
                optimizer,
                stage,
                n_mamba_blocks=n_mamba_blocks,
                upper_mamba_blocks=upper_mamba_blocks,
            )
            trainable_total = sum(trainable_counts.values())
            print(
                f"stage={stage_name} stage_index={stage_index} stage_epoch={stage_epoch}/{stage_epochs} "
                f"global_epoch={global_epoch_slot}/{total_epochs} steps_per_epoch={steps_per_epoch} "
                f"trainable={trainable_total} trainable_by_group={trainable_counts} lrs={current_lrs(optimizer)}",
                flush=True,
            )

            model.train()
            sc_iterator = bridge.infinite_loader(loaders["sc_train"])
            bulk_iterator = bridge.infinite_loader(loaders["bulk_train"])
            pb_iterator = bridge.infinite_loader(loaders["pb_train"])
            optimizer.zero_grad(set_to_none=True)

            totals = {
                "train_total": 0.0,
                "train_sc_recovery": 0.0,
                "train_masked_positive": 0.0,
                "train_observed_nonzero": 0.0,
                "train_zero_regularization": 0.0,
                "train_dropout": 0.0,
                "train_bulk": 0.0,
                "train_pseudobulk": 0.0,
            }

            for step in range(1, steps_per_epoch + 1):
                sc_batch = next(sc_iterator)
                bulk_batch = next(bulk_iterator)
                observed_sc = bridge.move_tensor(sc_batch["x"], device)
                target_sc = bridge.move_tensor(sc_batch["y"], device)
                observed_bulk = bridge.move_tensor(bulk_batch["x"], device)
                target_bulk = bridge.move_tensor(bulk_batch["y"], device)

                with bridge.autocast_context(device, amp_enabled, amp_dtype):
                    sc_output = forward_dict(model, observed_sc, modality="sc")
                    bulk_output = forward_dict(model, observed_bulk, modality="bulk")

                    recovery = single_cell_recovery_loss(
                        sc_output["reconstruction"],
                        observed_sc,
                        target_sc,
                        masked_positive_weight=float(train_section.get("masked_positive_weight", 4.0)),
                        observed_nonzero_weight=float(train_section.get("observed_nonzero_weight", 1.0)),
                        zero_regularization_weight=float(train_section.get("zero_regularization_weight", 0.05)),
                        zero_margin=float(train_section.get("zero_margin", 0.05)),
                        smooth_l1_beta=float(train_section.get("smooth_l1_beta", 0.2)),
                    )
                    dropout = dropout_bce_loss(
                        sc_output.get("dropout_logits"),
                        observed_sc,
                        target_sc,
                    )
                    bulk_loss = F.smooth_l1_loss(
                        bulk_output["reconstruction"],
                        target_bulk,
                        beta=float(train_section.get("smooth_l1_beta", 0.2)),
                    )

                    pseudobulk_loss = torch.zeros((), device=device)
                    pseudobulk_scale = 1.0
                    if step % pb_every == 0:
                        pb_batch = next(pb_iterator)
                        observed_pb = bridge.move_tensor(pb_batch["x"], device)
                        target_pb = bridge.move_tensor(pb_batch["y"], device)
                        pb_output = forward_dict(model, observed_pb, modality="pseudobulk")
                        pseudobulk_loss = F.smooth_l1_loss(
                            pb_output["reconstruction"],
                            target_pb,
                            beta=float(train_section.get("smooth_l1_beta", 0.2)),
                        )
                        pseudobulk_scale = float(pb_every)

                    raw_total = (
                        recovery.loss
                        + float(train_section.get("dropout_loss_weight", 0.10)) * dropout.loss
                        + float(train_section.get("bulk_loss_weight", 0.25)) * bulk_loss
                        + float(train_section.get("pb_loss_weight", 0.25))
                        * pseudobulk_scale
                        * pseudobulk_loss
                    )
                    scaled_total = raw_total / float(grad_accum_steps)

                if scaler.is_enabled():
                    scaler.scale(scaled_total).backward()
                else:
                    scaled_total.backward()

                optimizer_step = step % grad_accum_steps == 0 or step == steps_per_epoch
                if optimizer_step:
                    if grad_clip_norm > 0.0:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        clip_grad_norm_(
                            [parameter for parameter in model.parameters() if parameter.requires_grad],
                            max_norm=grad_clip_norm,
                        )
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                totals["train_total"] += float(raw_total.item())
                totals["train_sc_recovery"] += float(recovery.loss.item())
                totals["train_masked_positive"] += float(recovery.masked_positive_loss.item())
                totals["train_observed_nonzero"] += float(recovery.observed_nonzero_loss.item())
                totals["train_zero_regularization"] += float(recovery.zero_regularization_loss.item())
                totals["train_dropout"] += float(dropout.loss.item())
                totals["train_bulk"] += float(bulk_loss.item())
                totals["train_pseudobulk"] += float(pseudobulk_loss.item()) * pseudobulk_scale

                periodic_save = save_every_steps > 0 and step % save_every_steps == 0 and optimizer_step
                if periodic_save or _STOP_REQUESTED:
                    save_checkpoint(
                        checkpoint_dir / "last.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        cfg=cfg,
                        completed_global_epoch=global_epoch_slot - 1,
                        best_epoch=best_epoch,
                        best_key=best_key,
                        history=history,
                        stage_name=stage_name,
                        stage_epoch=stage_epoch,
                        step=step,
                        steps_per_epoch=steps_per_epoch,
                        partial_epoch=True,
                    )
                    if periodic_save:
                        print(
                            f"checkpoint_saved=last.pt partial_epoch=1 completed_global_epoch={global_epoch_slot - 1} "
                            f"stage={stage_name} step={step}/{steps_per_epoch}",
                            flush=True,
                        )
                    if _STOP_REQUESTED:
                        print(
                            f"stop_requested=1 signal={_STOP_SIGNAL} resume_with='--auto-resume'",
                            flush=True,
                        )
                        return

            for key in totals:
                totals[key] /= float(steps_per_epoch)

            validation = evaluate_sc_recovery(
                model,
                loaders["sc_val"],
                device,
                amp_enabled,
                amp_dtype,
                zero_fill_threshold=zero_fill_threshold,
            )
            key = selection_key(validation, min_std_ratio=min_std_ratio)
            is_best = best_key is None or key < best_key

            row: dict[str, Any] = {
                "epoch": global_epoch_slot,
                "stage": stage_name,
                "stage_epoch": stage_epoch,
                "lrs": current_lrs(optimizer),
                "trainable_parameters": trainable_total,
                **totals,
                **{f"val_{name}": value for name, value in validation.items()},
                "val_checkpoint_eligible": bool(validation["std_ratio"] >= min_std_ratio),
                "is_best": bool(is_best),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

            completed_global_epoch = global_epoch_slot
            if is_best:
                best_key = key
                best_epoch = global_epoch_slot
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    cfg=cfg,
                    completed_global_epoch=completed_global_epoch,
                    best_epoch=best_epoch,
                    best_key=best_key,
                    history=history,
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                    step=steps_per_epoch,
                    steps_per_epoch=steps_per_epoch,
                    partial_epoch=False,
                )

            save_checkpoint(
                checkpoint_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                completed_global_epoch=completed_global_epoch,
                best_epoch=best_epoch,
                best_key=best_key,
                history=history,
                stage_name=stage_name,
                stage_epoch=stage_epoch,
                step=steps_per_epoch,
                steps_per_epoch=steps_per_epoch,
                partial_epoch=False,
            )
            with (run_dir / "metrics_partial.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_selection_key": list(best_key) if best_key is not None else None,
                        "history": history,
                    },
                    handle,
                    indent=2,
                )

    best_path = checkpoint_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Training completed without producing best.pt")
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate_sc_recovery(
        model,
        loaders["sc_test"],
        device,
        amp_enabled,
        amp_dtype,
        zero_fill_threshold=zero_fill_threshold,
    )

    summary = {
        "run_name": run_name,
        "model_kind": model_kind,
        "parameter_counts": counts,
        "initialized_from": train_section["init_checkpoint_path"],
        "best_epoch": best_epoch,
        "best_selection_key": list(best_key) if best_key is not None else None,
        "test_recovery": test_metrics,
        "history": history,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"best_epoch={best_epoch}")
    print(json.dumps({f"test_{key}": value for key, value in test_metrics.items()}, sort_keys=True))
    print(f"saved_outputs_to={run_dir}")


if __name__ == "__main__":
    main()
