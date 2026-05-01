from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/3159436/sc2/data/raw/census/census_pilot.h5ad",
        help="Path to h5ad file",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    adata = ad.read_h5ad(path)
    print(f"path: {path}")
    print(f"n_obs: {adata.n_obs}")
    print(f"n_vars: {adata.n_vars}")
    print("\nobs columns:")
    print(list(adata.obs.columns))
    print("\nvar columns:")
    print(list(adata.var.columns))
    print("\nfirst obs rows:")
    print(adata.obs.head())
    print("\nfirst var rows:")
    print(adata.var.head())


if __name__ == "__main__":
    main()