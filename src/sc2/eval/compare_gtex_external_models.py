from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare bulk-capable models on GTEx external lung.")
    parser.add_argument("--bulk-only-csv", required=True)
    parser.add_argument("--mixed-csv", required=True)
    parser.add_argument("--bridge-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    bulk = pd.read_csv(args.bulk_only_csv).copy()
    mixed = pd.read_csv(args.mixed_csv).copy()
    bridge = pd.read_csv(args.bridge_csv).copy()

    bulk["model"] = "bulk_only_large"
    mixed["model"] = "sc2lite_mixed_large"
    bridge["model"] = "sc2lite_bridge_large"

    out = pd.concat([bulk, mixed, bridge], ignore_index=True)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_csv, index=False)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(out.to_dict(orient="records"), f, indent=2)

    print("saved comparison outputs:")
    print(output_csv)
    print(output_json)
    print(out)


if __name__ == "__main__":
    main()