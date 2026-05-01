from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lung-filtered ARCHS4 pilot manifest.")
    parser.add_argument("--input", required=True, help="Filtered lung metadata parquet")
    parser.add_argument("--output", required=True, help="Output pilot manifest parquet")
    parser.add_argument("--pilot-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input).copy()
    rng = np.random.default_rng(args.seed)

    if len(df) > args.pilot_size:
        df = df.sample(n=args.pilot_size, random_state=args.seed).reset_index(drop=True)

    n = len(df)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    n_test = n - n_train - n_val

    perm = rng.permutation(n)
    split = np.empty(n, dtype=object)
    split[perm[:n_train]] = "train"
    split[perm[n_train:n_train + n_val]] = "val"
    split[perm[n_train + n_val:]] = "test"

    manifest = df[["sample_idx"]].copy()
    manifest["split"] = split

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)

    print(f"saved lung pilot manifest: {output_path}")
    print(manifest["split"].value_counts())


if __name__ == "__main__":
    main()