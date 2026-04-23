from __future__ import annotations

import argparse
from email import parser
from html import parser
import os
from pathlib import Path

import cellxgene_census as cxg
import pandas as pd


OBS_COLUMNS = [
    "dataset_id",
    "assay",
    "tissue",
    "tissue_general",
    "cell_type",
    "soma_joinid",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a multi-dataset Census pilot.")
    parser.add_argument("--organism", default="Homo sapiens")
    parser.add_argument("--census-version", default="2025-11-08")
    parser.add_argument("--tissue-general", default="lung")
    parser.add_argument("--max-datasets", type=int, default=5)
    parser.add_argument("--cells-per-dataset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    out_dir = data_root / "raw" / "census"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name is None:
        output_name = (
            f"census_multi_dataset_{args.tissue_general}_"
            f"{args.max_datasets}ds_{args.cells_per_dataset}cpd.h5ad"
        )
    else:
        output_name = args.output_name

    out_path = out_dir / output_name

    with cxg.open_soma(census_version=args.census_version) as census:
        obs_df = cxg.get_obs(
            census,
            organism=args.organism,
            value_filter=(
                f"is_primary_data == True and tissue_general == '{args.tissue_general}'"
            ),
            column_names=OBS_COLUMNS,
        )

        obs_df["dataset_id"] = obs_df["dataset_id"].astype(str)

        dataset_counts = obs_df["dataset_id"].value_counts()
        chosen_dataset_ids = dataset_counts.head(args.max_datasets).index.tolist()

        filtered = obs_df[obs_df["dataset_id"].isin(chosen_dataset_ids)].copy()

        sampled = (
            filtered.groupby("dataset_id", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(len(g), args.cells_per_dataset),
                    random_state=args.seed,
                )
            )
            .reset_index(drop=True)
        )

        soma_ids = sampled["soma_joinid"].astype(int).tolist()

        adata = cxg.get_anndata(
            census,
            organism=args.organism,
            obs_coords=soma_ids,
            obs_column_names=OBS_COLUMNS,
            var_column_names=["feature_id", "feature_name", "soma_joinid"],
        )

    adata.write_h5ad(out_path)

    print(f"saved to {out_path}")
    print(f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print("dataset counts:")
    print(pd.Series(adata.obs["dataset_id"]).astype(str).value_counts())


if __name__ == "__main__":
    main()