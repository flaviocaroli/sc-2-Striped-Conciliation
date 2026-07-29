#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def stable_fraction(seed: int, value: str) -> float:
    digest = hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-resistant donor/study splits")
    parser.add_argument("--cells", required=True, help="Input parquet with cell metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--validation", type=float, default=0.10)
    parser.add_argument("--group-priority", nargs="+", default=["donor_id", "dataset_id"])
    args = parser.parse_args()
    frame = pd.read_parquet(args.cells)
    group_column = next((name for name in args.group_priority if name in frame and frame[name].notna().any()), None)
    if group_column is None:
        raise ValueError(f"None of the split group columns exist: {args.group_priority}")
    if "cell_id" not in frame:
        raise ValueError("Input metadata requires cell_id")
    if args.train <= 0.0 or args.validation <= 0.0 or args.train + args.validation >= 1.0:
        raise ValueError("Invalid split fractions")

    groups = frame[group_column].fillna(frame["dataset_id"]).astype(str)
    unique = pd.DataFrame({"split_group": groups.unique()})
    unique["fraction"] = unique["split_group"].map(lambda value: stable_fraction(args.seed, value))
    unique["split"] = "test"
    unique.loc[unique["fraction"] < args.train, "split"] = "train"
    unique.loc[
        (unique["fraction"] >= args.train) & (unique["fraction"] < args.train + args.validation),
        "split",
    ] = "validation"
    mapping = dict(zip(unique["split_group"], unique["split"], strict=True))
    output = frame.copy()
    output["split_group_column"] = group_column
    output["split_group"] = groups
    output["split"] = groups.map(mapping)
    if output.groupby("split_group")["split"].nunique().max() != 1:
        raise RuntimeError("A split group crosses splits")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    print(output["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
