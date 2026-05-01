from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Build a generic manifest from any h5ad file.")
    parser.add_argument("--input", required=True, help="Input .h5ad path")
    parser.add_argument("--output", default=None, help="Optional output parquet path")
    parser.add_argument("--source-name", default=None, help="Optional source name")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output is None:
        output_path = (
            input_path.parent.parent.parent
            / "processed"
            / "manifests"
            / f"{input_path.stem}_manifest.parquet"
        )
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path, backed="r")
    obs = adata.obs.copy()

    keep_cols = [c for c in DEFAULT_COLUMNS if c in obs.columns]
    manifest = obs[keep_cols].copy()

    manifest["cell_id"] = obs.index.astype(str)
    manifest["source_name"] = args.source_name if args.source_name else input_path.stem
    manifest["source_file"] = str(input_path)

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