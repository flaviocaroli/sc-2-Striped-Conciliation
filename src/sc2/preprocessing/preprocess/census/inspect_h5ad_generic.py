from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd


COMMON_OBS_COLUMNS = [
    "dataset_id",
    "assay",
    "tissue",
    "tissue_general",
    "cell_type",
]


def top_counts(df: pd.DataFrame, column: str, n: int = 10) -> dict[str, int]:
    if column not in df.columns:
        return {}
    vc = df[column].astype(str).value_counts(dropna=False).head(n)
    return {str(k): int(v) for k, v in vc.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect any h5ad file and save a JSON summary.")
    parser.add_argument("--input", required=True, help="Input .h5ad path")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output_json is None:
        output_json = input_path.with_suffix(".summary.json")
    else:
        output_json = Path(args.output_json)

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
    }

    for col in COMMON_OBS_COLUMNS:
        summary[f"top_{col}"] = top_counts(obs_df, col)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    if getattr(adata, "file", None) is not None:
        adata.file.close()


if __name__ == "__main__":
    main()