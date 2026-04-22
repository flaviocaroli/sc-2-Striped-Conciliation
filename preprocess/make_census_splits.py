from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic cell-level splits for the Census pilot.")
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

    df = pd.read_parquet(manifest_path).copy()

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))

    n = len(df)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    n_test = n - n_train - n_val

    split = np.empty(n, dtype=object)
    split[perm[:n_train]] = "train"
    split[perm[n_train:n_train + n_val]] = "val"
    split[perm[n_train + n_val:]] = "test"

    df["split"] = split
    df.to_parquet(output_path, index=False)

    summary = {
        "seed": args.seed,
        "n_rows": int(len(df)),
        "split_counts_rows": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "note": "Temporary cell-level split for pilot engineering only. Not for final scientific evaluation."
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved split manifest: {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()