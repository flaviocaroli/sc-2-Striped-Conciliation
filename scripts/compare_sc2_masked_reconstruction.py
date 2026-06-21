#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine masked reconstruction CSVs for native controls and SC2 variants.")
    parser.add_argument("--output-root", default="/home/3159436/sc2/outputs/evals")
    parser.add_argument("--out", default="/home/3159436/sc2/outputs/evals/masked_reconstruction_comparison.csv")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[
            "benchmark_current_lung_sc_masked_m2",
            "benchmark_current_lung_sc_masked_orig",
            "benchmark_current_lung_sc_masked_sc2_mini",
            "benchmark_current_lung_sc_masked_sc2_medium",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    frames = []
    for run in args.runs:
        path = root / run / "overall_masked_reconstruction.csv"
        if not path.exists():
            print(f"missing: {path}")
            continue
        df = pd.read_csv(path)
        df.insert(0, "run", run)
        frames.append(df)
    if not frames:
        raise SystemExit("No CSV files found.")
    out = pd.concat(frames, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
