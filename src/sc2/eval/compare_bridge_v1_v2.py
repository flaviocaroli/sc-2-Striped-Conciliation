from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SC2Lite bridge v1 vs v2.")
    parser.add_argument("--bridge-v1-csv", required=True)
    parser.add_argument("--bridge-v2-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    v1 = pd.read_csv(args.bridge_v1_csv).copy()
    v2 = pd.read_csv(args.bridge_v2_csv).copy()

    v1["model"] = "bridge_v1"
    v2["model"] = "bridge_v2"

    out = pd.concat([v1, v2], ignore_index=True)

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