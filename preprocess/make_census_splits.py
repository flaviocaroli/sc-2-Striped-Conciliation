from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic dataset-level splits for Census pilot.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Input manifest parquet. Defaults to $SC2_DATA_ROOT/processed/manifests/census_pilot_manifest.parquet",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet with splits. Defaults to $SC2_DATA_ROOT/splits/census_pilot_manifest_with_splits.parquet",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    manifest_path = Path(args.manifest) if args.manifest else data_root / "processed" / "manifests" / "census_pilot_manifest.parquet"
    output_path = Path(args.output) if args.output else data_root / "splits" / "census_pilot_manifest_with_splits.parquet"
    summary_path = output_path.with_suffix(".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(manifest_path)

    if "dataset_id" not in df.columns:
        raise ValueError("Manifest must contain a dataset_id column for dataset-level splitting.")

    dataset_ids = sorted(df["dataset_id"].astype(str).unique().tolist())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(dataset_ids)

    n = len(dataset_ids)
    n_train = max(1, int(round(n * 0.8)))
    n_val = max(1, int(round(n * 0.1)))
    n_test = max(1, n - n_train - n_val)

    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val

    train_ids = set(dataset_ids[:n_train])
    val_ids = set(dataset_ids[n_train:n_train + n_val])
    test_ids = set(dataset_ids[n_train + n_val:])

    def assign_split(dataset_id: str) -> str:
        if dataset_id in train_ids:
            return "train"
        if dataset_id in val_ids:
            return "val"
        return "test"

    df["dataset_id"] = df["dataset_id"].astype(str)
    df["split"] = df["dataset_id"].apply(assign_split)

    df.to_parquet(output_path, index=False)

    summary = {
        "seed": args.seed,
        "n_rows": int(len(df)),
        "n_dataset_ids": int(len(dataset_ids)),
        "split_counts_rows": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "split_counts_dataset_ids": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved split manifest: {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()