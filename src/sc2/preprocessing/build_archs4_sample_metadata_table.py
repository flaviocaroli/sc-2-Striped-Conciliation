from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import pandas as pd


def _decode_array(arr) -> list[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


COLUMNS = [
    "geo_accession",
    "series_id",
    "sample",
    "source_name_ch1",
    "characteristics_ch1",
    "library_strategy",
    "organism_ch1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ARCHS4 sample metadata to parquet.")
    parser.add_argument("--input", default=None, help="Path to ARCHS4 h5 file")
    parser.add_argument("--output", default=None, help="Output parquet path")
    args = parser.parse_args()

    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    input_path = Path(args.input) if args.input else data_root / "raw" / "archs4" / "human_gene_v2.5.h5"
    output_path = Path(args.output) if args.output else data_root / "processed" / "manifests" / "archs4_sample_metadata.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as f:
        meta = f["meta"]["samples"]
        n_samples = len(meta["geo_accession"])
        df = pd.DataFrame({"sample_idx": range(n_samples)})

        for col in COLUMNS:
            if col in meta:
                df[col] = _decode_array(meta[col][:])
            else:
                df[col] = ""

    df.to_parquet(output_path, index=False)

    print(f"saved metadata table: {output_path}")
    print(f"n_samples={len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()