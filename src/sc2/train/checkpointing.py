from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch



def config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch_state = torch.as_tensor(state["torch"], dtype=torch.uint8).cpu().contiguous()
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        normalized = [torch.as_tensor(item, dtype=torch.uint8).cpu().contiguous() for item in cuda_state]
        if len(normalized) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(normalized)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    global_step: int,
    next_sample_index: int,
    manifest_sha256: str,
    config: Mapping[str, Any],
    pareto_state: Mapping[str, Any],
    history: list[dict[str, Any]],
    git_commit: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": "sc2-continuous-checkpoint-v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(global_step),
        "next_sample_index": int(next_sample_index),
        "manifest_sha256": str(manifest_sha256),
        "config": dict(config),
        "config_sha256": config_sha256(config),
        "pareto_state": dict(pareto_state),
        "history": history,
        "rng_state": capture_rng_state(),
        "git_commit": git_commit,
        "saved_at_unix": time.time(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    expected_manifest_sha256: str,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "sc2-continuous-checkpoint-v1":
        raise ValueError("Unsupported checkpoint format")
    if str(checkpoint["manifest_sha256"]) != str(expected_manifest_sha256):
        raise ValueError("Manifest changed; refuse exact resume and create a new run")
    if expected_config_sha256 is not None:
        checkpoint_config_hash = checkpoint.get("config_sha256")
        if checkpoint_config_hash is None:
            raise ValueError("Checkpoint has no config hash; refuse exact resume and create a new run")
        if str(checkpoint_config_hash) != str(expected_config_sha256):
            raise ValueError("Resolved configuration changed; refuse exact resume and create a new run")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer_to_device(optimizer, device)
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint
