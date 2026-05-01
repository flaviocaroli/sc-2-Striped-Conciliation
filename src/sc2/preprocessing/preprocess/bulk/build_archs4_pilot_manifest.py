from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a random ARCHS4 pilot manifest.")
    parser.add_argument("--n-samples-total", type=int, default=10000, help="Total number of samples in ARCHS4")
    parser.add_argument("--pilot-size", type=int, default=2000, help="How many samples to include in the pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(args.n_samples_total, size=args.pilot_size, replace=False)
    sample_indices = np.sort(sample_indices)

    n = len(sample_indices)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    n_test = n - n_train - n_val

    perm = rng.permutation(n)
    split = np.empty(n, dtype=object)
    split[perm[:n_train]] = "train"
    split[perm[n_train:n_train + n_val]] = "val"
    split[perm[n_train + n_val:]] = "test"

    df = pd.DataFrame(
        {
            "sample_idx": sample_indices,
            "split": split,
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"saved ARCHS4 pilot manifest: {output_path}")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()