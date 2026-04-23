from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import pandas as pd


def top_counts(df: pd.DataFrame, column: str, n: int = 10) -> dict[str, int]:
    if column not in df.columns:
        return {}
    vc = df[column].astype(str).value_counts(dropna=False).head(n)
    return {str(k): int(v) for k, v in vc.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Census pilot h5ad file.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the input h5ad. Defaults to $SC2_DATA_ROOT/raw/census/census_pilot_5k_lung.h5ad",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to write the JSON summary. Defaults to $SC2_DATA_ROOT/artifacts/census_pilot_summary.json",
    )
    args = parser.parse_args()

    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    input_path = Path(args.input) if args.input else data_root / "raw" / "census" / "census_pilot_5k_lung.h5ad"
    output_json = Path(args.output_json) if args.output_json else data_root / "artifacts" / "census_pilot_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path, backed="r")

    obs_df = adata.obs.copy()
    var_df = adata.var.copy()

    summary = {
        "input_path": str(input_path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": list(map(str, obs_df.columns)),
        "var_columns": list(map(str, var_df.columns)),
        "top_dataset_id": top_counts(obs_df, "dataset_id"),
        "top_assay": top_counts(obs_df, "assay"),
        "top_tissue": top_counts(obs_df, "tissue"),
        "top_tissue_general": top_counts(obs_df, "tissue_general"),
        "top_cell_type": top_counts(obs_df, "cell_type"),
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    if getattr(adata, "file", None) is not None:
        adata.file.close()


if __name__ == "__main__":
    main()