from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare bulk-only large vs SC2Lite mixed/bridge models.")
    parser.add_argument("--bulk-overall-csv", required=True)
    parser.add_argument("--mixed-overall-csv", required=True)
    parser.add_argument("--bridge-overall-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    bulk_df = pd.read_csv(args.bulk_overall_csv)
    mixed_df = pd.read_csv(args.mixed_overall_csv)
    bridge_df = pd.read_csv(args.bridge_overall_csv)

    rows = []

    for _, row in bulk_df.iterrows():
        rows.append(
            {
                "model": "bulk_only_large",
                "modality": "bulk",
                "split": row["split"],
                "mse_mean": row["mse_mean"],
                "mae_mean": row["mae_mean"],
            }
        )

    for _, row in mixed_df.iterrows():
        rows.append(
            {
                "model": "sc2lite_mixed_large",
                "modality": row["modality"],
                "split": row["split"],
                "mse_mean": row["mse_mean"],
                "mae_mean": row["mae_mean"],
            }
        )

    for _, row in bridge_df.iterrows():
        rows.append(
            {
                "model": "sc2lite_bridge_large",
                "modality": row["modality"],
                "split": row["split"],
                "mse_mean": row["mse_mean"],
                "mae_mean": row["mae_mean"],
            }
        )

    out_df = pd.DataFrame(rows)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_csv, index=False)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(out_df.to_dict(orient="records"), f, indent=2)

    print("saved comparison outputs:")
    print(output_csv)
    print(output_json)
    print(out_df)


if __name__ == "__main__":
    main()