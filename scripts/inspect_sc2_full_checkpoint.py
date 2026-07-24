from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sc2.models.striped.sc2_striped_full import build_sc2_striped_full_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and strictly reload an SC2 full checkpoint.")
    parser.add_argument("checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.checkpoint)
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config")
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError("Checkpoint has no resolved model config")
    model = build_sc2_striped_full_from_config(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    summary = {
        "checkpoint": str(path),
        "model_kind": checkpoint.get("model_kind"),
        "parameters_total": total,
        "parameters_trainable": trainable,
        "attention_positions": getattr(model, "attention_positions", None),
        "cursor": checkpoint.get("cursor"),
        "global_step": checkpoint.get("global_step"),
        "completed_epochs": checkpoint.get("completed_epochs"),
        "best_any_epoch": checkpoint.get("best_any_epoch"),
        "best_eligible_epoch": checkpoint.get("best_eligible_epoch"),
        "model_config": config["model"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
