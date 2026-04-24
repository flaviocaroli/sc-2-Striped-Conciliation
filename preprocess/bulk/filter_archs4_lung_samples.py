from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_TEXT_COLS = [
    "sample",
    "source_name_ch1",
    "characteristics_ch1",
]

DEFAULT_KEYWORDS = [
    "lung",
    "pulmonary",
    "alveolar",
    "bronch",
    "bronchi",
    "airway",
    "respiratory",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter ARCHS4 metadata for lung-related bulk samples.")
    parser.add_argument("--input", required=True, help="Input metadata parquet")
    parser.add_argument("--output", required=True, help="Output filtered parquet")
    parser.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)
    args = parser.parse_args()

    df = pd.read_parquet(args.input).copy()

    text = pd.Series([""] * len(df), index=df.index, dtype=object)
    for col in DEFAULT_TEXT_COLS:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str)

    text = text.str.lower()
    pattern = "|".join(re.escape(k.lower()) for k in args.keywords)

    keep = text.str.contains(pattern, regex=True, na=False)
    filtered = df[keep].reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path, index=False)

    print(f"saved lung-filtered metadata: {output_path}")
    print(f"n_filtered={len(filtered)}")
    print(filtered.head())


if __name__ == "__main__":
    main()