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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ARCHS4 gene table.")
    parser.add_argument("--input", default=None, help="Path to ARCHS4 h5")
    parser.add_argument("--output", default=None, help="Output TSV path")
    args = parser.parse_args()

    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    input_path = Path(args.input) if args.input else data_root / "raw" / "archs4" / "human_gene_v2.5.h5"
    output_path = Path(args.output) if args.output else data_root / "artifacts" / "gene_tables" / "archs4_gene_table.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as f:
        symbols = _decode_array(f["meta"]["genes"]["symbol"][:])
        ensembl = _decode_array(f["meta"]["genes"]["ensembl_gene"][:])
        biotype = _decode_array(f["meta"]["genes"]["biotype"][:])

    df = pd.DataFrame(
        {
            "archs4_gene_index": range(len(symbols)),
            "symbol": symbols,
            "ensembl_gene": ensembl,
            "biotype": biotype,
        }
    )

    df.to_csv(output_path, sep="\t", index=False)

    print(f"saved gene table: {output_path}")
    print(f"n_genes={len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()