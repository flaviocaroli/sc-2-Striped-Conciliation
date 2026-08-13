from .census_reader import (
    load_census_gene_table,
    load_census_h5ad,
    load_census_manifest,
    load_census_split_manifest,
)

__all__ = [
    "load_census_gene_table",
    "load_census_h5ad",
    "load_census_manifest",
    "load_census_split_manifest",
    "CensusPilotDataset",
]


def __getattr__(name):
    if name == "CensusPilotDataset":
        from .census_datasets import CensusPilotDataset

        return CensusPilotDataset

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
