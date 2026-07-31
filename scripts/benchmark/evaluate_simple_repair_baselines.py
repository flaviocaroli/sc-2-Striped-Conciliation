#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sc2.eval.selective_repair_metrics import masked_value_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed simple value-recovery baselines on a frozen SC2 NPZ benchmark")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--gene-mean", required=True, help="Train-only mean log1p vector")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shrinkage-alpha", type=float, default=0.75, help="Weight on train gene mean")
    args = parser.parse_args()

    data = np.load(args.benchmark)
    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    mask = np.asarray(data["synthetic_mask"], dtype=bool)
    if x.shape != y.shape or mask.shape != y.shape:
        raise ValueError("x, y and synthetic_mask must have identical shapes")
    gene_mean = np.load(args.gene_mean).astype(np.float32)
    if gene_mean.shape != (x.shape[1],):
        raise ValueError(f"Gene mean shape {gene_mean.shape} does not match genes={x.shape[1]}")
    if not 0.0 <= args.shrinkage_alpha <= 1.0:
        raise ValueError("shrinkage-alpha must be in [0, 1]")

    observed_counts = (~mask).sum(axis=1, keepdims=True)
    observed_sum = np.where(mask, 0.0, x).sum(axis=1, keepdims=True)
    cell_mean = np.divide(
        observed_sum,
        observed_counts,
        out=np.zeros_like(observed_sum, dtype=np.float32),
        where=observed_counts > 0,
    )

    predictions = {
        "zero": np.zeros_like(y, dtype=np.float32),
        "train_gene_mean": np.broadcast_to(gene_mean[None, :], y.shape).copy(),
        "observed_cell_mean": np.broadcast_to(cell_mean, y.shape).copy(),
    }
    alpha = float(args.shrinkage_alpha)
    predictions["gene_cell_shrinkage"] = (
        alpha * predictions["train_gene_mean"] + (1.0 - alpha) * predictions["observed_cell_mean"]
    )

    rows = []
    for name, prediction in predictions.items():
        rows.append({"baseline": name, **masked_value_metrics(prediction, y, mask)})
    frame = pd.DataFrame(rows).sort_values("masked_mse").reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "simple_baselines.csv", index=False)
    (output_dir / "simple_baselines.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
