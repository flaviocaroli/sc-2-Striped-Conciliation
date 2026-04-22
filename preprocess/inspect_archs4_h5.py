from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def print_group(name: str, obj) -> None:
    obj_type = type(obj).__name__
    shape = getattr(obj, "shape", None)
    print(f"{name} | type={obj_type} | shape={shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/3159436/sc2/data/raw/archs4/human_gene_v2.5.h5",
        help="Path to ARCHS4 H5 file",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"opening: {path}")
    with h5py.File(path, "r") as f:
        print("top-level keys:")
        for k in f.keys():
            print(" -", k)

        print("\nfull object listing:")
        f.visititems(print_group)


if __name__ == "__main__":
    main()