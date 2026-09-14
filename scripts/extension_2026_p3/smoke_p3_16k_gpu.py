#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import yaml

from sc2.losses.striped_full_losses import compute_sc_objective
from sc2.models.striped.sc2_striped_full import SC2StripedFull


EXPECTED = {
    "small": {
        "params": 5_564_042,
        "mamba": 20,
        "attention": 4,
        "width": 128,
    },
    "large": {
        "params": 12_799_692,
        "mamba": 25,
        "attention": 5,
        "width": 192,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("small", "large"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for P3 H200 smoke")

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    expected = EXPECTED[args.variant]

    model_cfg = dict(cfg["model"])

    assert int(model_cfg["n_mamba_blocks"]) == expected["mamba"]
    assert int(model_cfg["n_attention_checkpoints"]) == expected["attention"]
    assert int(model_cfg["d_model"]) == expected["width"]
    assert int(model_cfg["d_state"]) == 4
    assert int(model_cfg["d_conv"]) == 4
    assert int(model_cfg["top_k"]) == 256
    assert int(model_cfg["n_heads"]) == 4
    assert bool(model_cfg["bidirectional_mamba"]) is True
    assert str(model_cfg["mamba_backend"]) == "official"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")

    print("VARIANT =", args.variant)
    print("CONFIG =", cfg_path)
    print("CONFIG_SHA256 =", sha256(cfg_path))
    print("DEVICE_NAME =", torch.cuda.get_device_name(0))
    print("TORCH =", torch.__version__)
    print("TORCH_CUDA =", torch.version.cuda)
    print("BATCH_SIZE =", args.batch_size)
    print("GENES = 16384")

    model = SC2StripedFull(
        n_genes=16384,
        **model_cfg,
    ).to(device)

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    assert total_params == expected["params"], (
        total_params,
        expected["params"],
    )
    assert trainable_params == expected["params"]

    print("TOTAL_PARAMS =", total_params)
    print("TRAINABLE_PARAMS =", trainable_params)
    print("ATTENTION_POSITIONS =", model.attention_positions)

    target = torch.zeros(
        args.batch_size,
        16384,
        dtype=torch.float32,
        device=device,
    )

    positive_selector = torch.rand_like(target) < 0.18
    positive_values = 0.05 + 4.0 * torch.rand_like(target)
    target = torch.where(
        positive_selector,
        positive_values,
        target,
    )

    observed = target.clone()

    positive = target > 0
    synthetic_mask = (
        (torch.rand_like(target) < 0.15) &
        positive
    )

    observed[synthetic_mask] = 0.0

    n_positive = int(positive.sum().item())
    n_masked = int(synthetic_mask.sum().item())

    assert n_positive > 0
    assert n_masked > 0

    print("SYNTHETIC_POSITIVE =", n_positive)
    print("SYNTHETIC_MASKED =", n_masked)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        betas=(
            float(cfg["train"]["beta1"]),
            float(cfg["train"]["beta2"]),
        ),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        output = model(
            observed,
            modality="sc",
            return_dict=True,
        )

        if not isinstance(output, dict):
            raise TypeError(
                f"Expected dict output, got {type(output)!r}"
            )

        objective = compute_sc_objective(
            output,
            observed,
            target,
            dict(cfg["train"]["loss"]),
        )

        loss = objective.loss

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"non-finite loss: {float(loss.detach().cpu())}"
        )

    forward_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [
            p for p in model.parameters()
            if p.requires_grad
        ],
        float(cfg["train"]["grad_clip_norm"]),
    )

    if not math.isfinite(float(grad_norm)):
        raise RuntimeError(
            f"non-finite grad norm: {float(grad_norm)}"
        )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    backward_step_seconds = time.perf_counter() - t1
    total_seconds = time.perf_counter() - t0

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    print("OUTPUT_KEYS =", sorted(output.keys()))
    print("LOSS =", float(loss.detach().cpu()))
    print("GRAD_NORM =", float(grad_norm))
    print("FORWARD_SECONDS =", forward_seconds)
    print("BACKWARD_OPTIMIZER_SECONDS =", backward_step_seconds)
    print("TOTAL_SECONDS =", total_seconds)
    print(
        "PEAK_ALLOCATED_GIB =",
        peak_allocated / (1024 ** 3),
    )
    print(
        "PEAK_RESERVED_GIB =",
        peak_reserved / (1024 ** 3),
    )

    summary = {
        "variant": args.variant,
        "config": str(cfg_path),
        "config_sha256": sha256(cfg_path),
        "batch_size": args.batch_size,
        "genes": 16384,
        "parameters": total_params,
        "attention_positions": list(model.attention_positions),
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm),
        "forward_seconds": forward_seconds,
        "backward_optimizer_seconds": backward_step_seconds,
        "total_seconds": total_seconds,
        "peak_allocated_gib": peak_allocated / (1024 ** 3),
        "peak_reserved_gib": peak_reserved / (1024 ** 3),
    }

    print("SMOKE_JSON =", json.dumps(summary, sort_keys=True))
    print("P3_GPU_SMOKE=PASS")


if __name__ == "__main__":
    main()
