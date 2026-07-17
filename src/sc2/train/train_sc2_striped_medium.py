from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_

from sc2.config import load_yaml, merge_train_and_paths
from sc2.losses.dropout_losses import dropout_bce_loss, zero_false_positive_penalty
from sc2.models.striped.sc2_striped_medium import build_sc2_striped_medium_from_config
from sc2.train import train_sc2_mamba_bridge as bridge

_STOP_REQUESTED = False
_STOP_SIGNAL = None


def _handle_stop_signal(signum: int, frame: Any) -> None:
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = signum
    print(
        f"received_signal={signum}; will save checkpoint and exit at the next safe point",
        flush=True,
    )


def install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handle_stop_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_stop_signal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SC2-medium with staged reconstruction + dropout losses.")
    parser.add_argument("--config", required=True, help="Training YAML config.")
    parser.add_argument("--paths", required=True, help="Path-root YAML config.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to resume from, or 'auto' to resume from <run_dir>/checkpoints/last.pt when present.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from <run_dir>/checkpoints/last.pt if it exists. Ignored when --fresh is set.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing last.pt and start a new run from scratch.",
    )
    return parser.parse_args()


def build_model(model_cfg: Dict[str, Any], n_genes: int) -> tuple[str, nn.Module]:
    kind = str(model_cfg.get("kind", "sc2_striped_medium")).strip().lower()
    if kind not in {"sc2_striped_medium", "striped_medium", "sc2_medium"}:
        raise ValueError(f"train_sc2_striped_medium.py expects model.kind=sc2_striped_medium, got {kind!r}")
    return "sc2_striped_medium", build_sc2_striped_medium_from_config(model_cfg, n_genes=n_genes)


def stage_value(stage: Dict[str, Any], train_cfg: Dict[str, Any], name: str, default: Any) -> Any:
    if name in stage:
        return stage[name]
    return train_cfg.get(name, default)


def stage_weights(stage: Dict[str, Any], train_cfg: Dict[str, Any]) -> dict[str, float]:
    return {
        "bulk": float(stage_value(stage, train_cfg, "bulk_loss_weight", 1.0)),
        "sc": float(stage_value(stage, train_cfg, "sc_loss_weight", 5.0)),
        "pb": float(stage_value(stage, train_cfg, "pb_loss_weight", 1.0)),
        "align": float(stage_value(stage, train_cfg, "align_loss_weight", 0.5)),
        "dropout": float(stage_value(stage, train_cfg, "dropout_loss_weight", 0.0)),
        "zero_fp": float(stage_value(stage, train_cfg, "zero_fp_loss_weight", 0.0)),
    }


def forward_dict(model: nn.Module, x: torch.Tensor, modality: str) -> dict[str, torch.Tensor]:
    out = model(x, modality=modality, return_dict=True)
    if not isinstance(out, dict):
        raise TypeError("SC2-medium forward(return_dict=True) must return a dict")
    return out


def latent_pool(out: dict[str, torch.Tensor]) -> torch.Tensor:
    latent = out["latent"]
    if latent.ndim == 3:
        return latent.mean(dim=1)
    return latent


def make_loaders(cfg: Dict[str, Any], seed: int):
    data_cfg = cfg["data"]
    data_root = Path(cfg["paths"]["data_root"])
    bulk_h5_path = bridge.require_existing_file(
        "bulk_h5_path", bridge.resolve_data_path(data_root, data_cfg.get("bulk_h5_path"))
    )
    bulk_manifest_path = bridge.require_existing_file(
        "bulk_manifest_path", bridge.resolve_data_path(data_root, data_cfg.get("bulk_manifest_path"))
    )
    sc_h5ad_path = bridge.require_existing_file(
        "sc_h5ad_path", bridge.resolve_data_path(data_root, data_cfg.get("sc_h5ad_path"))
    )
    sc_split_manifest_path = bridge.require_existing_file(
        "sc_split_manifest_path", bridge.resolve_data_path(data_root, data_cfg.get("sc_split_manifest_path"))
    )
    pb_h5ad_path = bridge.require_existing_file(
        "pseudobulk_h5ad_path", bridge.resolve_data_path(data_root, data_cfg.get("pseudobulk_h5ad_path"))
    )
    shared_gene_table_path = bridge.require_existing_file(
        "shared_gene_table_path", bridge.resolve_data_path(data_root, data_cfg.get("shared_gene_table_path"))
    )

    n_genes_requested = int(data_cfg["n_genes"])
    log1p_input = bool(data_cfg.get("log1p_input", True))
    bulk_kwargs = dict(
        h5_path=bulk_h5_path,
        sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("bulk_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("bulk_noise_std", 0.0)),
        seed=seed,
    )
    sc_kwargs = dict(
        h5ad_path=sc_h5ad_path,
        split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("sc_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("sc_noise_std", 0.0)),
        seed=seed,
    )
    pb_kwargs = dict(
        h5ad_path=pb_h5ad_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=n_genes_requested,
        log1p_input=log1p_input,
        mask_prob=float(data_cfg.get("pb_mask_prob", 0.15)),
        noise_std=float(data_cfg.get("pb_noise_std", 0.0)),
        seed=seed,
    )

    bulk_train = bridge.ARCHS4DenoiseDataset(split="train", **bulk_kwargs)
    bulk_val = bridge.ARCHS4DenoiseDataset(split="val", **bulk_kwargs)
    bulk_test = bridge.ARCHS4DenoiseDataset(split="test", **bulk_kwargs)
    sc_train = bridge.CensusSharedDataset(split="train", **sc_kwargs)
    sc_val = bridge.CensusSharedDataset(split="val", **sc_kwargs)
    sc_test = bridge.CensusSharedDataset(split="test", **sc_kwargs)
    pb_train = bridge.PseudobulkSharedDataset(split="train", **pb_kwargs)
    pb_val = bridge.PseudobulkSharedDataset(split="val", **pb_kwargs)
    pb_test = bridge.PseudobulkSharedDataset(split="test", **pb_kwargs)

    input_dim = int(bulk_train.n_features)
    for name, dataset in [
        ("bulk_val", bulk_val),
        ("bulk_test", bulk_test),
        ("sc_train", sc_train),
        ("sc_val", sc_val),
        ("sc_test", sc_test),
        ("pb_train", pb_train),
        ("pb_val", pb_val),
        ("pb_test", pb_test),
    ]:
        if int(dataset.n_features) != input_dim:
            raise ValueError(f"Feature dimension mismatch: bulk_train={input_dim}, {name}={dataset.n_features}")

    num_workers = int(data_cfg.get("num_workers", 0))
    loaders = {
        "bulk_train": bridge.build_loader(bulk_train, int(data_cfg.get("bulk_batch_size", 8)), True, num_workers, seed + 11),
        "bulk_val": bridge.build_loader(bulk_val, int(data_cfg.get("bulk_batch_size", 8)), False, num_workers, seed + 12),
        "bulk_test": bridge.build_loader(bulk_test, int(data_cfg.get("bulk_batch_size", 8)), False, num_workers, seed + 13),
        "sc_train": bridge.build_loader(sc_train, int(data_cfg.get("sc_batch_size", 16)), True, num_workers, seed + 21),
        "sc_val": bridge.build_loader(sc_val, int(data_cfg.get("sc_batch_size", 16)), False, num_workers, seed + 22),
        "sc_test": bridge.build_loader(sc_test, int(data_cfg.get("sc_batch_size", 16)), False, num_workers, seed + 23),
        "pb_train": bridge.build_loader(pb_train, int(data_cfg.get("pb_batch_size", 8)), True, num_workers, seed + 31),
        "pb_val": bridge.build_loader(pb_val, int(data_cfg.get("pb_batch_size", 8)), False, num_workers, seed + 32),
        "pb_test": bridge.build_loader(pb_test, int(data_cfg.get("pb_batch_size", 8)), False, num_workers, seed + 33),
    }
    for name, loader in loaders.items():
        bridge.ensure_non_empty_loader(name, loader)
    return loaders, input_dim


@torch.inference_mode()
def evaluate_with_dropout(
    model: nn.Module,
    loaders: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    weights: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    base = bridge.evaluate_all(
        model,
        loaders[f"bulk_{prefix}"],
        loaders[f"sc_{prefix}"],
        loaders[f"pb_{prefix}"],
        device,
        amp_enabled,
        amp_dtype,
        {"bulk": weights["bulk"], "sc": weights["sc"], "pb": weights["pb"]},
        prefix=prefix,
    )

    model.eval()
    total_drop_loss = 0.0
    total_supervised = 0
    total_positive = 0
    total_zero_fp = 0.0
    total_batches = 0
    for batch in loaders[f"sc_{prefix}"]:
        xs = bridge.move_tensor(batch["x"], device)
        ys = bridge.move_tensor(batch["y"], device)
        with bridge.autocast_context(device, amp_enabled, amp_dtype):
            out = forward_dict(model, xs, modality="sc")
            drop = dropout_bce_loss(out.get("dropout_logits"), xs, ys)
            zero_fp = zero_false_positive_penalty(out["reconstruction"], xs, ys)
        total_drop_loss += float(drop.loss.item()) * max(drop.n_supervised, 1)
        total_supervised += int(drop.n_supervised)
        total_positive += int(drop.n_positive)
        total_zero_fp += float(zero_fp.item())
        total_batches += 1

    drop_loss = total_drop_loss / float(max(total_supervised, 1))
    zero_fp_loss = total_zero_fp / float(max(total_batches, 1))
    positive_rate = float(total_positive) / float(max(total_supervised, 1))
    base_total = float(base[f"{prefix}_total"])
    base[f"{prefix}_base_total"] = base_total
    base[f"{prefix}_dropout_loss"] = drop_loss
    base[f"{prefix}_dropout_positive_rate"] = positive_rate
    base[f"{prefix}_zero_fp_loss"] = zero_fp_loss
    base[f"{prefix}_total"] = base_total + weights["dropout"] * drop_loss + weights["zero_fp"] * zero_fp_loss
    return base


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def save_training_checkpoint(
    path: Path,
    *,
    completed_global_epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: torch.amp.GradScaler,
    cfg: dict[str, Any],
    model_kind: str,
    best_val_total: float,
    best_epoch: int,
    history: list[dict[str, Any]],
    stage_name: str | None = None,
    stage_idx: int | None = None,
    stage_epoch_idx: int | None = None,
    step: int | None = None,
    n_steps: int | None = None,
    partial_epoch: bool = False,
    interrupted: bool = False,
) -> None:
    payload: dict[str, Any] = {
        "epoch": int(completed_global_epoch),
        "completed_global_epoch": int(completed_global_epoch),
        "model_kind": model_kind,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "best_val_total": float(best_val_total),
        "best_epoch": int(best_epoch),
        "history": history,
        "stage_name": stage_name,
        "stage_idx": stage_idx,
        "stage_epoch_idx": stage_epoch_idx,
        "step": step,
        "n_steps": n_steps,
        "partial_epoch": bool(partial_epoch),
        "interrupted": bool(interrupted),
        "saved_at_unix": time.time(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()
    atomic_torch_save(payload, path)


def load_partial_metrics(run_dir: Path) -> tuple[int, float, list[dict[str, Any]]]:
    path = run_dir / "metrics_partial.json"
    if not path.exists():
        return 0, float("inf"), []
    try:
        data = json.loads(path.read_text())
        return int(data.get("best_epoch", 0)), float(data.get("best_val_total", float("inf"))), list(data.get("history", []))
    except Exception as exc:
        print(f"warning=could_not_read_metrics_partial path={path} error={exc}", flush=True)
        return 0, float("inf"), []


def resolve_resume_path(args: argparse.Namespace, ckpt_dir: Path) -> Path | None:
    if args.fresh:
        return None
    if args.resume is not None:
        if str(args.resume).strip().lower() == "auto":
            candidate = ckpt_dir / "last.pt"
            return candidate if candidate.exists() else None
        return Path(args.resume)
    if args.auto_resume:
        candidate = ckpt_dir / "last.pt"
        return candidate if candidate.exists() else None
    return None


def patch_train_cfg_for_scheduler(cfg: dict[str, Any]) -> None:
    train_cfg = cfg.get("train", {})
    stages = train_cfg.get("stages") or []
    if stages:
        total_epochs = sum(int(stage.get("epochs", 1)) for stage in stages)
        train_cfg["epochs"] = int(train_cfg.get("epochs", total_epochs) or total_epochs)
        if int(train_cfg["epochs"]) != total_epochs:
            train_cfg["epochs"] = total_epochs


def main() -> None:
    install_signal_handlers()
    args = parse_args()
    train_cfg = load_yaml(args.config)
    path_cfg = load_yaml(args.paths)
    cfg = merge_train_and_paths(train_cfg, path_cfg)
    patch_train_cfg_for_scheduler(cfg)

    seed = int(cfg.get("seed", 42))
    bridge.seed_everything(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_root = Path(cfg["paths"]["output_root"])
    run_name = str(cfg["run_name"])
    run_dir = output_root / run_name
    ckpt_dir = run_dir / "checkpoints"
    bridge.ensure_dir(run_dir)
    bridge.ensure_dir(ckpt_dir)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = bridge.get_device(cfg.get("device", "auto"))
    train_section = cfg.get("train", {})
    loaders, input_dim = make_loaders(cfg, seed)
    model_kind, model = build_model(cfg.get("model", {}), n_genes=input_dim)
    model = model.to(device)
    param_counts = bridge.count_parameters(model)
    print(f"model_kind={model_kind}")
    print(f"parameters_total={param_counts['total']}")
    print(f"parameters_trainable={param_counts['trainable']}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_section.get("lr", 3e-4)),
        weight_decay=float(train_section.get("weight_decay", 1e-4)),
    )
    scheduler = bridge.build_scheduler(optimizer, train_section)
    amp_enabled = bridge.resolve_amp(train_section, device)
    amp_dtype = bridge.resolve_amp_dtype(train_section, device) if amp_enabled else None
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and device.type == "cuda" and amp_dtype == torch.float16,
    )

    best_epoch, best_val_total, history = load_partial_metrics(run_dir)
    completed_global_epoch = 0
    resume_path = resolve_resume_path(args, ckpt_dir)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        checkpoint = bridge.load_checkpoint(
            resume_path,
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        completed_global_epoch = int(checkpoint.get("completed_global_epoch", checkpoint.get("epoch", 0)))
        best_val_total = float(checkpoint.get("best_val_total", best_val_total))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        if checkpoint.get("history"):
            history = list(checkpoint.get("history", history))
        print(
            "resumed_from="
            f"{resume_path} completed_global_epoch={completed_global_epoch} "
            f"partial_epoch={bool(checkpoint.get('partial_epoch', False))} "
            f"best_epoch={best_epoch} best_val_total={best_val_total}",
            flush=True,
        )
    else:
        print("resume=none fresh_start=1", flush=True)

    stages = train_section.get("stages")
    if not stages:
        stages = [{"name": "main", "epochs": int(train_section.get("epochs", 8))}]

    grad_clip_norm = float(train_section.get("grad_clip_norm", 1.0))
    grad_accum_steps = max(1, int(train_section.get("grad_accum_steps", 1)))
    eval_every = max(1, int(train_section.get("eval_every", 1)))
    save_every_steps = int(train_section.get("save_every_steps", 25))
    if save_every_steps < 0:
        save_every_steps = 0

    total_epochs = sum(int(stage.get("epochs", 1)) for stage in stages)
    if completed_global_epoch >= total_epochs:
        print(f"training_already_complete completed_global_epoch={completed_global_epoch} total_epochs={total_epochs}")

    global_epoch_slot = 0
    current_stage_name = None

    try:
        for stage_idx, stage in enumerate(stages, start=1):
            stage_name = str(stage.get("name", f"stage{stage_idx}"))
            n_stage_epochs = int(stage.get("epochs", 1))
            n_steps = int(stage_value(stage, train_section, "steps_per_epoch", train_section.get("steps_per_epoch", 400)))
            pb_every = max(1, int(stage_value(stage, train_section, "pb_every", train_section.get("pb_every", 4))))
            weights = stage_weights(stage, train_section)

            for stage_epoch_idx in range(1, n_stage_epochs + 1):
                global_epoch_slot += 1
                if global_epoch_slot <= completed_global_epoch:
                    continue

                if current_stage_name != stage_name:
                    current_stage_name = stage_name
                    print(
                        f"stage={stage_name} epochs={n_stage_epochs} steps_per_epoch={n_steps} "
                        f"pb_every={pb_every} weights={weights}",
                        flush=True,
                    )

                global_epoch = global_epoch_slot
                model.train()
                bulk_iter = bridge.infinite_loader(loaders["bulk_train"])
                sc_iter = bridge.infinite_loader(loaders["sc_train"])
                pb_iter = bridge.infinite_loader(loaders["pb_train"])
                optimizer.zero_grad(set_to_none=True)
                totals = {
                    "train_bulk_loss": 0.0,
                    "train_sc_loss": 0.0,
                    "train_pb_loss": 0.0,
                    "train_align_loss": 0.0,
                    "train_dropout_loss": 0.0,
                    "train_zero_fp_loss": 0.0,
                    "train_total": 0.0,
                }

                for step in range(1, n_steps + 1):
                    bulk_batch = next(bulk_iter)
                    sc_batch = next(sc_iter)
                    xb = bridge.move_tensor(bulk_batch["x"], device)
                    yb = bridge.move_tensor(bulk_batch["y"], device)
                    xs = bridge.move_tensor(sc_batch["x"], device)
                    ys = bridge.move_tensor(sc_batch["y"], device)

                    with bridge.autocast_context(device, amp_enabled, amp_dtype):
                        out_b = forward_dict(model, xb, modality="bulk")
                        out_s = forward_dict(model, xs, modality="sc")
                        pred_b = out_b["reconstruction"]
                        pred_s = out_s["reconstruction"]
                        z_b = latent_pool(out_b)
                        loss_b = criterion(pred_b, yb)
                        loss_s = criterion(pred_s, ys)
                        drop = dropout_bce_loss(out_s.get("dropout_logits"), xs, ys)
                        loss_drop = drop.loss
                        loss_zero_fp = zero_false_positive_penalty(pred_s, xs, ys)
                        loss_p = torch.zeros((), device=device)
                        loss_align = torch.zeros((), device=device)
                        pb_scale = 1.0
                        if step % pb_every == 0:
                            pb_batch = next(pb_iter)
                            xp = bridge.move_tensor(pb_batch["x"], device)
                            yp = bridge.move_tensor(pb_batch["y"], device)
                            out_p = forward_dict(model, xp, modality="pseudobulk")
                            pred_p = out_p["reconstruction"]
                            z_p = latent_pool(out_p)
                            loss_p = criterion(pred_p, yp)
                            loss_align = bridge.mean_alignment_loss(z_b, z_p)
                            pb_scale = float(pb_every)

                        raw_total_loss = (
                            weights["bulk"] * loss_b
                            + weights["sc"] * loss_s
                            + weights["pb"] * pb_scale * loss_p
                            + weights["align"] * pb_scale * loss_align
                            + weights["dropout"] * loss_drop
                            + weights["zero_fp"] * loss_zero_fp
                        )
                        scaled_loss = raw_total_loss / grad_accum_steps

                    if scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    did_optim_step = False
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
                        did_optim_step = True

                    totals["train_bulk_loss"] += float(loss_b.item())
                    totals["train_sc_loss"] += float(loss_s.item())
                    totals["train_pb_loss"] += float(loss_p.item()) * pb_scale
                    totals["train_align_loss"] += float(loss_align.item()) * pb_scale
                    totals["train_dropout_loss"] += float(loss_drop.item())
                    totals["train_zero_fp_loss"] += float(loss_zero_fp.item())
                    totals["train_total"] += float(raw_total_loss.item())

                    should_periodic_save = bool(save_every_steps and step % save_every_steps == 0 and did_optim_step)
                    if should_periodic_save or _STOP_REQUESTED:
                        if not did_optim_step:
                            optimizer.zero_grad(set_to_none=True)
                        save_training_checkpoint(
                            ckpt_dir / "last.pt",
                            completed_global_epoch=global_epoch - 1,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            cfg=cfg,
                            model_kind=model_kind,
                            best_val_total=best_val_total,
                            best_epoch=best_epoch,
                            history=history,
                            stage_name=stage_name,
                            stage_idx=stage_idx,
                            stage_epoch_idx=stage_epoch_idx,
                            step=step,
                            n_steps=n_steps,
                            partial_epoch=True,
                            interrupted=bool(_STOP_REQUESTED),
                        )
                        if should_periodic_save:
                            print(
                                f"checkpoint_saved=last.pt partial_epoch=1 completed_global_epoch={global_epoch - 1} "
                                f"stage={stage_name} epoch={global_epoch} step={step}/{n_steps}",
                                flush=True,
                            )
                        if _STOP_REQUESTED:
                            print(
                                f"stop_requested=1 signal={_STOP_SIGNAL} saved_checkpoint={ckpt_dir / 'last.pt'} "
                                f"resume_with='--auto-resume'",
                                flush=True,
                            )
                            return

                for key in totals:
                    totals[key] /= float(n_steps)

                did_eval = (global_epoch % eval_every == 0) or (global_epoch == total_epochs)
                if did_eval:
                    val_metrics = evaluate_with_dropout(model, loaders, device, amp_enabled, amp_dtype, weights, prefix="val")
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

                row = {
                    "epoch": global_epoch,
                    "stage": stage_name,
                    "stage_epoch": stage_epoch_idx,
                    "model_kind": model_kind,
                    "lr": bridge.current_lr(optimizer),
                    **totals,
                    **val_metrics,
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

                completed_global_epoch = global_epoch
                save_training_checkpoint(
                    ckpt_dir / "last.pt",
                    completed_global_epoch=completed_global_epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    cfg=cfg,
                    model_kind=model_kind,
                    best_val_total=best_val_total,
                    best_epoch=best_epoch,
                    history=history,
                    stage_name=stage_name,
                    stage_idx=stage_idx,
                    stage_epoch_idx=stage_epoch_idx,
                    step=n_steps,
                    n_steps=n_steps,
                    partial_epoch=False,
                    interrupted=False,
                )
                if did_eval and float(val_metrics["val_total"]) < best_val_total:
                    best_val_total = float(val_metrics["val_total"])
                    best_epoch = global_epoch
                    save_training_checkpoint(
                        ckpt_dir / "best.pt",
                        completed_global_epoch=completed_global_epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        cfg=cfg,
                        model_kind=model_kind,
                        best_val_total=best_val_total,
                        best_epoch=best_epoch,
                        history=history,
                        stage_name=stage_name,
                        stage_idx=stage_idx,
                        stage_epoch_idx=stage_epoch_idx,
                        step=n_steps,
                        n_steps=n_steps,
                        partial_epoch=False,
                        interrupted=False,
                    )
                with (run_dir / "metrics_partial.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {"best_epoch": best_epoch, "best_val_total": best_val_total, "history": history},
                        f,
                        indent=2,
                    )
    finally:
        if _STOP_REQUESTED:
            print("exit_after_signal=1", flush=True)

    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        bridge.load_checkpoint(best_path, model=model, device=device)
    final_weights = stage_weights(stages[-1], train_section)
    test_metrics = evaluate_with_dropout(model, loaders, device, amp_enabled, amp_dtype, final_weights, prefix="test")
    summary = {
        "run_name": run_name,
        "device": str(device),
        "model_kind": model_kind,
        "parameter_counts": param_counts,
        "best_epoch": best_epoch,
        "best_val_total": best_val_total,
        **test_metrics,
        "history": history,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"best_epoch={best_epoch}")
    print(f"best_val_total={best_val_total:.6f}")
    print(f"test_total={test_metrics['test_total']:.6f}")
    print(f"saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
