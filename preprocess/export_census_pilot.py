from __future__ import annotations

import os
from pathlib import Path

import cellxgene_census as cxg


def main() -> None:
    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    out_dir = data_root / "raw" / "census"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "census_pilot.h5ad"

    census = cxg.open_soma(census_version="2025-11-08")

    adata = cxg.get_anndata(
        census,
        organism="Homo sapiens",
        obs_value_filter=(
            "is_primary_data == True and "
            "tissue_general in ['lung', 'blood', 'brain']"
        ),
        obs_column_names=[
            "dataset_id",
            "assay",
            "tissue",
            "tissue_general",
            "cell_type",
            "soma_joinid",
        ],
        var_column_names=["feature_id", "feature_name", "soma_joinid"],
    )

    adata.write_h5ad(out_path)
    census.close()
    print(f"saved to {out_path}")
    print(f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")


if __name__ == "__main__":
    main()