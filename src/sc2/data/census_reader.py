from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import pandas as pd


def get_data_root() -> Path:
    return Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))


def get_census_pilot_h5ad_path() -> Path:
    return get_data_root() / "raw" / "census" / "census_pilot_5k_lung.h5ad"


def get_census_manifest_path() -> Path:
    return get_data_root() / "processed" / "manifests" / "census_pilot_manifest.parquet"


def get_census_split_manifest_path() -> Path:
    return get_data_root() / "splits" / "census_pilot_manifest_with_splits.parquet"


def get_census_gene_table_path() -> Path:
    return get_data_root() / "artifacts" / "gene_tables" / "census_pilot_gene_table.tsv"


def load_census_h5ad(backed: str | None = None):
    path = get_census_pilot_h5ad_path()
    if backed is None:
        return ad.read_h5ad(path)
    return ad.read_h5ad(path, backed=backed)


def load_census_manifest() -> pd.DataFrame:
    return pd.read_parquet(get_census_manifest_path())


def load_census_split_manifest() -> pd.DataFrame:
    return pd.read_parquet(get_census_split_manifest_path())


def load_census_gene_table() -> pd.DataFrame:
    return pd.read_csv(get_census_gene_table_path(), sep="\t")