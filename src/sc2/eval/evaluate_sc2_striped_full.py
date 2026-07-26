from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sc2.config import load_yaml, merge_train_and_paths
from sc2.models.striped.sc2_striped_full import build_sc2_striped_full_from_config
from sc2.train import train_sc2_mamba_bridge as bridge
from sc2.train import train_sc2_striped_medium as medium_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal bulk/sc/pseudobulk evaluation for SC2 20+4.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate_loader(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    modality: str,
    prediction_key: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    sample_index = 0
    model.eval()
    for batch in loader:
        observed = bridge.move_tensor(batch["x"], device)
        target = bridge.move_tensor(batch["y"], device)
        with bridge.autocast_context(device, amp_enabled, amp_dtype):
            output = model(observed, modality=modality, return_dict=True)
            if not isinstance(output, dict):
                raise TypeError("Expected dictionary model output")
            prediction = output[prediction_key]

        difference = prediction.float() - target.float()
        mse = difference.square().mean(dim=1).cpu().numpy()
        mae = difference.abs().mean(dim=1).cpu().numpy()
        corrupted_mask = (observed - target).abs() > 1.0e-8
        for local_index in range(observed.shape[0]):
            current_mask = corrupted_mask[local_index]
            if bool(current_mask.any()):
                masked_difference = difference[local_index][current_mask]
                masked_mse = float(masked_difference.square().mean().item())
                masked_mae = float(masked_difference.abs().mean().item())
                n_corrupted = int(current_mask.sum().item())
            else:
                masked_mse = float("nan")
                masked_mae = float("nan")
                n_corrupted = 0
            rows.append(
                {
                    "modality": modality,
                    "split": "test",
                    "sample_index": sample_index,
                    "mse": float(mse[local_index]),
                    "mae": float(mae[local_index]),
                    "masked_mse": masked_mse,
                    "masked_mae": masked_mae,
                    "n_corrupted": n_corrupted,
                }
            )
            sample_index += 1
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    eval_cfg = load_yaml(args.config)
    paths_cfg = load_yaml(args.paths)
    cfg = merge_train_and_paths(eval_cfg, paths_cfg)

    seed = int(cfg.get("seed", 42))
    bridge.seed_everything(seed)
    device = bridge.get_device(cfg.get("device", "auto"))
    output_root = Path(cfg["paths"]["output_root"])
    checkpoint_path = resolve_path(output_root, cfg["eval"]["checkpoint_path"])
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, dict) or "model" not in checkpoint_cfg:
        raise ValueError("Checkpoint must contain its resolved training config")

    loaders, input_dim = medium_train.make_loaders(cfg, seed)
    model = build_sc2_striped_full_from_config(checkpoint_cfg["model"], n_genes=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(False)

    amp_enabled = bridge.resolve_amp(cfg.get("eval", {}), device)
    amp_dtype = bridge.resolve_amp_dtype(cfg.get("eval", {}), device) if amp_enabled else None
    prediction_key = str(cfg["eval"].get("prediction_key", "reconstruction"))

    print(f"device={device}")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"model_kind=sc2_striped_full")
    counts = bridge.count_parameters(model)
    print(f"parameters_total={counts['total']}")
    print(f"attention_positions={getattr(model, 'attention_positions', None)}")
    print(f"prediction_key={prediction_key}")

    frames = [
        evaluate_loader(
            model,
            loaders["bulk_test"],
            device,
            modality="bulk",
            prediction_key=prediction_key,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        ),
        evaluate_loader(
            model,
            loaders["sc_test"],
            device,
            modality="sc",
            prediction_key=prediction_key,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        ),
        evaluate_loader(
            model,
            loaders["pb_test"],
            device,
            modality="pseudobulk",
            prediction_key=prediction_key,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        ),
    ]
    all_metrics = pd.concat(frames, ignore_index=True)
    overall = (
        all_metrics.groupby(["modality", "split"], as_index=False)
        .agg(
            n_samples=("sample_index", "count"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            masked_mse_mean=("masked_mse", "mean"),
            masked_mae_mean=("masked_mae", "mean"),
            n_corrupted_mean=("n_corrupted", "mean"),
        )
    )

    out_dir = output_root / "evals" / str(cfg["eval_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(out_dir / "all_sample_metrics.csv", index=False)
    overall.to_csv(out_dir / "overall_by_modality_split.csv", index=False)
    atomic_json_dump(
        {
            "eval_name": cfg["eval_name"],
            "checkpoint_path": str(checkpoint_path),
            "prediction_key": prediction_key,
            "parameter_counts": counts,
            "attention_positions": getattr(model, "attention_positions", None),
            "overall": overall.to_dict(orient="records"),
        },
        out_dir / "summary.json",
    )
    print(f"saved evaluation outputs to:\n{out_dir}")
    print("overall metrics:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
