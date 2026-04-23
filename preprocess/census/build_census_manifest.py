from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import pandas as pd


DEFAULT_COLUMNS = [
    "dataset_id",
    "assay",
    "tissue",
    "tissue_general",
    "cell_type",
    "soma_joinid",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple manifest from a Census pilot h5ad.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the input h5ad. Defaults to $SC2_DATA_ROOT/raw/census/census_pilot_5k_lung.h5ad",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output parquet. Defaults to $SC2_DATA_ROOT/processed/manifests/census_pilot_manifest.parquet",
    )
    args = parser.parse_args()

    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    input_path = Path(args.input) if args.input else data_root / "raw" / "census" / "census_pilot_5k_lung.h5ad"
    output_path = Path(args.output) if args.output else data_root / "processed" / "manifests" / "census_pilot_manifest.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path, backed="r")
    obs = adata.obs.copy()

    keep_cols = [c for c in DEFAULT_COLUMNS if c in obs.columns]
    manifest = obs[keep_cols].copy()

    manifest["cell_id"] = obs.index.astype(str)
    manifest["source_file"] = str(input_path)
    manifest["source_name"] = "census_pilot"

    cols = ["cell_id", "source_name", "source_file"] + keep_cols
    manifest = manifest[cols]

    manifest.to_parquet(output_path, index=False)

    print(f"saved manifest: {output_path}")
    print(f"rows={len(manifest)} cols={len(manifest.columns)}")
    print(manifest.head())

    if getattr(adata, "file", None) is not None:
        adata.file.close()


if __name__ == "__main__":
    main()