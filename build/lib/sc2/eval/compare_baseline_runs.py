from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sc-only, bulk-only, and mixed baseline runs.")
    parser.add_argument("--sc-baseline-metrics", required=True, help="Path to sc-only metrics.json")
    parser.add_argument("--bulk-baseline-metrics", required=True, help="Path to bulk-only denoising metrics.json")
    parser.add_argument("--mixed-overall-csv", required=True, help="Path to mixed eval overall_by_modality_split.csv")
    parser.add_argument("--output-csv", required=True, help="Path to output comparison CSV")
    parser.add_argument("--output-json", required=True, help="Path to output comparison JSON")
    args = parser.parse_args()

    with open(args.sc_baseline_metrics, "r", encoding="utf-8") as f:
        sc_baseline = json.load(f)

    with open(args.bulk_baseline_metrics, "r", encoding="utf-8") as f:
        bulk_baseline = json.load(f)

    mixed_df = pd.read_csv(args.mixed_overall_csv)

    def mixed_value(modality: str, split: str, col: str) -> float:
        row = mixed_df[(mixed_df["modality"] == modality) & (mixed_df["split"] == split)]
        if len(row) != 1:
            raise ValueError(f"Expected exactly one row for modality={modality}, split={split}")
        return float(row.iloc[0][col])

    rows = [
        {
            "modality": "sc",
            "model": "sc_only",
            "split": "val_best",
            "metric": "mse",
            "value": float(sc_baseline["best_val_loss"]),
        },
        {
            "modality": "sc",
            "model": "sc_only",
            "split": "test",
            "metric": "mse",
            "value": float(sc_baseline["test_loss"]),
        },
        {
            "modality": "bulk",
            "model": "bulk_only_denoise",
            "split": "val_best",
            "metric": "mse",
            "value": float(bulk_baseline["best_val_loss"]),
        },
        {
            "modality": "bulk",
            "model": "bulk_only_denoise",
            "split": "test",
            "metric": "mse",
            "value": float(bulk_baseline["test_loss"]),
        },
        {
            "modality": "sc",
            "model": "mixed",
            "split": "train",
            "metric": "mse",
            "value": mixed_value("sc", "train", "mse_mean"),
        },
        {
            "modality": "sc",
            "model": "mixed",
            "split": "val",
            "metric": "mse",
            "value": mixed_value("sc", "val", "mse_mean"),
        },
        {
            "modality": "sc",
            "model": "mixed",
            "split": "test",
            "metric": "mse",
            "value": mixed_value("sc", "test", "mse_mean"),
        },
        {
            "modality": "bulk",
            "model": "mixed",
            "split": "train",
            "metric": "mse",
            "value": mixed_value("bulk", "train", "mse_mean"),
        },
        {
            "modality": "bulk",
            "model": "mixed",
            "split": "val",
            "metric": "mse",
            "value": mixed_value("bulk", "val", "mse_mean"),
        },
        {
            "modality": "bulk",
            "model": "mixed",
            "split": "test",
            "metric": "mse",
            "value": mixed_value("bulk", "test", "mse_mean"),
        },
    ]

    df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print("saved comparison outputs:")
    print(output_csv)
    print(output_json)
    print(df)


if __name__ == "__main__":
    main()