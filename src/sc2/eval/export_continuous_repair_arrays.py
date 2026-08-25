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
    parser.add_argument("--array-output", default=None, help="Optional NPZ containing expected repair and gate probability")
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

    if "available_gene_mask" in data.files:
        available_gene_mask = np.asarray(data["available_gene_mask"], dtype=bool)
        if available_gene_mask.shape != (y.shape[1],):
            raise ValueError(
                "available_gene_mask must have shape (n_genes,), "
                f"got {available_gene_mask.shape}"
            )
    else:
        available_gene_mask = np.ones(y.shape[1], dtype=bool)

    available = available_gene_mask[None, :]
    if np.any(positive & ~available):
        raise ValueError("synthetic_mask contains unavailable external genes")

    true_zero = (
        y <= float(cfg.get("zero_threshold", 1.0e-8))
    ) & available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sc2_striped_full_from_config(cfg["model"], n_genes=x.shape[1]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    batch_size = int(cfg.get("batch_size", 32))
    expected_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    reconstruction_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            tensor = torch.from_numpy(x[start : start + batch_size]).to(device)
            output = model(tensor, modality="sc", return_dict=True)
            expected_chunks.append(output["expected_repair"].float().cpu().numpy())
            probability_chunks.append(output["dropout_probability"].float().cpu().numpy())
            reconstruction_chunks.append(output["reconstruction"].float().cpu().numpy())
    expected = np.concatenate(expected_chunks, axis=0)
    probability = np.concatenate(probability_chunks, axis=0)
    reconstruction = np.concatenate(reconstruction_chunks, axis=0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    value_metrics = masked_value_metrics(expected, y, positive)

    observed_nonzero = (
        np.abs(x) > float(model.zero_threshold)
    ) & available
    preservation_error = reconstruction[observed_nonzero] - x[observed_nonzero]

    if preservation_error.size:
        preservation_metrics = {
            "observed_nonzero_mse": float(np.mean(preservation_error ** 2)),
            "observed_nonzero_mae": float(np.mean(np.abs(preservation_error))),
            "observed_nonzero_max_abs_error": float(np.max(np.abs(preservation_error))),
            "observed_nonzero_changed_fraction": float(np.mean(preservation_error != 0.0)),
            "n_observed_nonzero": int(preservation_error.size),
        }
    else:
        preservation_metrics = {
            "observed_nonzero_mse": float("nan"),
            "observed_nonzero_mae": float("nan"),
            "observed_nonzero_max_abs_error": float("nan"),
            "observed_nonzero_changed_fraction": float("nan"),
            "n_observed_nonzero": 0,
        }

    gate_metrics = gate_discrimination(probability, positive, true_zero)
    if args.array_output is not None:
        array_path = Path(args.array_output)
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            array_path,
            expected_repair=np.asarray(expected, dtype=np.float32),
            dropout_probability=np.asarray(probability, dtype=np.float32),
        )
        print(f"ARRAY_EXPORT={array_path}")

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
        **preservation_metrics,
        **{f"gate_{key}": value for key, value in gate_metrics.items()},
        "n_available_genes": int(available_gene_mask.sum()),
        "available_gene_fraction": float(available_gene_mask.mean()),
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
