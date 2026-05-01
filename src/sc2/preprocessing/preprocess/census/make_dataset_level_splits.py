from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_counts(n_datasets: int) -> tuple[int, int, int]:
    if n_datasets < 3:
        raise ValueError("Need at least 3 dataset_ids for dataset-level train/val/test splitting.")

    n_train = max(1, int(np.floor(0.8 * n_datasets)))
    n_val = max(1, int(np.floor(0.1 * n_datasets)))
    n_test = n_datasets - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_train = n_datasets - n_val - n_test

    if n_train < 1:
        raise ValueError("Not enough dataset_ids to create a valid train split.")

    return n_train, n_val, n_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic dataset-level splits from a manifest.")
    parser.add_argument("--manifest", required=True, help="Input manifest parquet")
    parser.add_argument("--output", default=None, help="Optional output parquet path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if args.output is None:
        output_path = manifest_path.parent.parent / "splits" / f"{manifest_path.stem}_with_splits.parquet"
    else:
        output_path = Path(args.output)

    summary_path = output_path.with_suffix(".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(manifest_path).copy()
    if "dataset_id" not in df.columns:
        raise ValueError("Manifest must contain dataset_id for dataset-level splitting.")

    df["dataset_id"] = df["dataset_id"].astype(str)
    dataset_ids = sorted(df["dataset_id"].unique().tolist())

    n_train, n_val, n_test = compute_counts(len(dataset_ids))

    rng = np.random.default_rng(args.seed)
    rng.shuffle(dataset_ids)

    train_ids = set(dataset_ids[:n_train])
    val_ids = set(dataset_ids[n_train:n_train + n_val])
    test_ids = set(dataset_ids[n_train + n_val:])

    def assign_split(dataset_id: str) -> str:
        if dataset_id in train_ids:
            return "train"
        if dataset_id in val_ids:
            return "val"
        return "test"

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