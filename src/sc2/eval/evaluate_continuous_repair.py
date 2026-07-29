from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sc2.config import load_yaml
from sc2.eval.selective_repair_metrics import (
    choose_threshold,
    gate_discrimination,
    masked_value_metrics,
    risk_coverage_curve,
    threshold_sweep,
)
from sc2.models.striped.sc2_striped_full import build_sc2_striped_full_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SC2 selective repair on a frozen NPZ benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", required=True, help="NPZ with x, y and synthetic_mask")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=None, help="Frozen threshold; omit only on validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data = np.load(args.benchmark)
    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    positive = np.asarray(data["synthetic_mask"], dtype=bool)
    if x.shape != y.shape or positive.shape != y.shape:
        raise ValueError("x, y and synthetic_mask must have identical shapes")
    true_zero = y <= float(cfg.get("zero_threshold", 1.0e-8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sc2_striped_full_from_config(cfg["model"], n_genes=x.shape[1]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    batch_size = int(cfg.get("batch_size", 32))
    expected_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            tensor = torch.from_numpy(x[start : start + batch_size]).to(device)
            output = model(tensor, modality="sc", return_dict=True)
            expected_chunks.append(output["expected_repair"].float().cpu().numpy())
            probability_chunks.append(output["dropout_probability"].float().cpu().numpy())
    expected = np.concatenate(expected_chunks, axis=0)
    probability = np.concatenate(probability_chunks, axis=0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    value_metrics = masked_value_metrics(expected, y, positive)
    gate_metrics = gate_discrimination(probability, positive, true_zero)
    sweep = threshold_sweep(probability, positive, true_zero)
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    risk_coverage_curve(expected, y, positive, probability).to_csv(output_dir / "risk_coverage.csv", index=False)
    if args.threshold is None:
        selected = choose_threshold(
            sweep,
            max_true_zero_fill=float(cfg.get("max_true_zero_fill", 0.02)),
            min_precision=float(cfg.get("min_precision", 0.0)),
        )
        threshold = selected.threshold
        threshold_source = "selected_on_this_validation_set"
    else:
        threshold = float(args.threshold)
        threshold_source = "frozen_external_threshold"
    row = sweep.iloc[int(np.argmin(np.abs(sweep["threshold"].to_numpy() - threshold)))]
    summary: dict[str, Any] = {
        **value_metrics,
        **{f"gate_{key}": value for key, value in gate_metrics.items()},
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_recall": float(row["recall"]),
        "threshold_precision": float(row["precision"]),
        "threshold_true_zero_fill": float(row["true_zero_fill"]),
        "checkpoint": str(args.checkpoint),
        "benchmark": str(args.benchmark),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
