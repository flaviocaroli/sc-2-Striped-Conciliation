#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return str(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an SC2 continuous checkpoint without requiring CUDA")
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    summary = {
        "checkpoint": str(path.resolve()),
        "format": checkpoint.get("format"),
        "global_step": checkpoint.get("global_step"),
        "next_sample_index": checkpoint.get("next_sample_index"),
        "manifest_sha256": checkpoint.get("manifest_sha256"),
        "config_sha256": checkpoint.get("config_sha256"),
        "git_commit": checkpoint.get("git_commit"),
        "history_entries": len(checkpoint.get("history", [])),
        "model_tensors": len(checkpoint.get("model_state_dict", {})),
        "optimizer_present": "optimizer_state_dict" in checkpoint,
        "scheduler_present": "scheduler_state_dict" in checkpoint,
        "scaler_present": "scaler_state_dict" in checkpoint,
        "rng_keys": sorted(checkpoint.get("rng_state", {}).keys()),
        "pareto_state": _safe(checkpoint.get("pareto_state", {})),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
