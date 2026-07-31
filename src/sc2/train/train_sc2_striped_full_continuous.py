from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from sc2.config import load_yaml, merge_train_and_paths
from sc2.data.deterministic_mixture import iter_batches
from sc2.data.shard_manifest import filter_records, load_manifest
from sc2.data.sharded_expression_dataset import CounterBasedExpressionStream
from sc2.losses.continuous_repair_losses import compute_continuous_objective
from sc2.models.striped.sc2_striped_full import build_sc2_striped_full_from_config
from sc2.train.checkpointing import atomic_torch_save, build_checkpoint, config_sha256, load_checkpoint
from sc2.train.gradient_diagnostics import objective_gradient_diagnostics
from sc2.train.pareto import ParetoFront
from sc2.train.schedules import cosine_with_warmup, scheduled_weights

_STOP_REQUESTED = False


def _handle_signal(signum: int, frame: Any) -> None:
    del frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"received_signal={signum}; will checkpoint after current optimizer step", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous SC2 training over immutable fixed-vocabulary shards")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'auto'")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output = dict(batch)
    for key in ("x", "y", "counts", "synthetic_mask", "library_size", "sample_index", "row", "mask_rate"):
        if isinstance(output.get(key), torch.Tensor):
            output[key] = output[key].to(device, non_blocking=True)
    return output


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    for current in (signal.SIGTERM, signal.SIGINT):
        signal.signal(current, _handle_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_signal)

    args = parse_args()
    cfg = merge_train_and_paths(load_yaml(args.config), load_yaml(args.paths))
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)
    device = _device(str(cfg.get("device", "auto")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    run_dir = Path(cfg["paths"]["output_root"]) / str(cfg["run_name"])
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(cfg, run_dir / "resolved_config.json")

    manifest_path = Path(cfg["data"]["manifest"])
    records, manifest_hash = load_manifest(manifest_path)
    train_records = filter_records(records, split="train")
    n_genes = train_records[0].n_genes
    train_cfg = cfg["train"]
    if int(train_cfg.get("num_workers", 0)) != 0:
        raise ValueError("Exact-resume v1 requires train.num_workers=0")

    stream = CounterBasedExpressionStream(
        train_records,
        seed=seed,
        start_index=0,
        modality_weights=train_cfg["modality_weights"],
        mask_rates=train_cfg["mask_rates"],
        mask_probabilities=train_cfg["mask_probabilities"],
    )
    if args.dry_run:
        first = stream.sample_at(0)
        print(
            f"dry_run=ok n_genes={n_genes} modality={first['modality']} "
            f"shard={first['shard_id']} row={first['row']} mask_rate={first['mask_rate']}",
            flush=True,
        )
        return

    model = build_sc2_striped_full_from_config(cfg["model"], n_genes=n_genes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.95))),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    total_steps = int(train_cfg["total_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_with_warmup(
            step,
            total_steps,
            warmup_fraction=float(train_cfg.get("warmup_fraction", 0.05)),
            end_ratio=float(train_cfg.get("lr_end_ratio", 0.10)),
        ),
    )
    amp_dtype = torch.bfloat16 if str(train_cfg.get("amp_dtype", "bfloat16")) == "bfloat16" else torch.float16
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    gene_mean = None
    gene_mean_path = cfg["data"].get("gene_mean_log1p")
    if gene_mean_path:
        gene_mean = torch.from_numpy(np.load(gene_mean_path).astype(np.float32)).to(device)
        if gene_mean.numel() != n_genes:
            raise ValueError("gene_mean_log1p length does not match n_genes")

    global_step = 0
    next_sample_index = 0
    history: list[dict[str, Any]] = []
    pareto = ParetoFront(
        minimize=("masked_mse", "true_zero_fill"),
        maximize=("sample_spearman", "gate_auprc", "sd_ratio"),
    )
    resume_path: Path | None = None
    if not args.fresh:
        if args.resume == "auto":
            candidate = checkpoint_dir / "last.pt"
            resume_path = candidate if candidate.exists() else None
        elif args.resume:
            resume_path = Path(args.resume)
    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            expected_manifest_sha256=manifest_hash,
            expected_config_sha256=config_sha256(cfg),
        )
        global_step = int(checkpoint["global_step"])
        next_sample_index = int(checkpoint["next_sample_index"])
        history = list(checkpoint.get("history", []))
        pareto = ParetoFront.from_state_dict(checkpoint.get("pareto_state", pareto.state_dict()))
        print(f"resumed_from={resume_path} global_step={global_step} next_sample_index={next_sample_index}")

    batch_iterator = iter_batches(
        stream,
        batch_size=int(train_cfg["batch_size"]),
        start_sample_index=next_sample_index,
    )
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    save_every = max(1, int(train_cfg.get("save_every_steps", 100)))
    log_every = max(1, int(train_cfg.get("log_every_steps", 10)))
    diagnostic_every = max(0, int(train_cfg.get("gradient_diagnostic_every_steps", 0)))
    base_weights = dict(train_cfg["loss"]["weights"])
    ramps = dict(train_cfg["loss"].get("ramps", {}))

    while global_step < total_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        for microstep in range(grad_accum):
            batch = _move(next(batch_iterator), device)
            next_sample_index = int(batch["next_sample_index"])
            current_loss_cfg = dict(train_cfg["loss"])
            current_loss_cfg["weights"] = scheduled_weights(base_weights, ramps, global_step)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(batch["x"], modality=batch["modality"], return_dict=True)
                objective = compute_continuous_objective(outputs, batch, current_loss_cfg, gene_mean=gene_mean)
                scaled_loss = objective.loss / float(grad_accum)
            if diagnostic_every > 0 and global_step % diagnostic_every == 0 and microstep == 0:
                diagnostics = objective_gradient_diagnostics(objective.components, model)
                _json_dump(diagnostics, run_dir / "gradient_diagnostics" / f"step_{global_step:08d}.json")
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            aggregate["loss"] = aggregate.get("loss", 0.0) + float(objective.loss.detach().item()) / grad_accum
            for name, value in objective.components.items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value.detach().item()) / grad_accum

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = float(clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip_norm", 1.0))).item())
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        global_step += 1
        aggregate.update(
            step=global_step,
            next_sample_index=next_sample_index,
            grad_norm=grad_norm,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )
        history.append(aggregate)
        if global_step % log_every == 0:
            print(json.dumps(aggregate, sort_keys=True), flush=True)
        if global_step % save_every == 0 or _STOP_REQUESTED or global_step == total_steps:
            payload = build_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                next_sample_index=next_sample_index,
                manifest_sha256=manifest_hash,
                config=cfg,
                pareto_state=pareto.state_dict(),
                history=history[-1000:],
                git_commit=_git_commit(),
            )
            atomic_torch_save(payload, checkpoint_dir / "last.pt")
            _json_dump({"global_step": global_step, "next_sample_index": next_sample_index}, run_dir / "status.json")
        if _STOP_REQUESTED:
            print("stopped_after_checkpoint=1", flush=True)
            break


if __name__ == "__main__":
    main()
