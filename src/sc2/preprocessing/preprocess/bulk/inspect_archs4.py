from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py


def walk_group(group: h5py.Group, lines: list[str], prefix: str = "") -> None:
    for key in group.keys():
        obj = group[key]
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        if isinstance(obj, h5py.Group):
            lines.append(f"[GROUP]   {path}")
            walk_group(obj, lines, path)
        elif isinstance(obj, h5py.Dataset):
            lines.append(f"[DATASET] {path} shape={obj.shape} dtype={obj.dtype}")
        else:
            lines.append(f"[OTHER]   {path} type={type(obj)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the internal structure of an ARCHS4 HDF5 file.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to ARCHS4 HDF5. Defaults to $SC2_DATA_ROOT/raw/archs4/human_gene_v2.5.h5",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to text report. Defaults to $SC2_DATA_ROOT/artifacts/archs4_h5_structure.txt",
    )
    args = parser.parse_args()

    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    input_path = Path(args.input) if args.input else data_root / "raw" / "archs4" / "human_gene_v2.5.h5"
    output_path = Path(args.output) if args.output else data_root / "artifacts" / "archs4_h5_structure.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    with h5py.File(input_path, "r") as f:
        lines.append(f"FILE: {input_path}")
        walk_group(f, lines)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved structure report: {output_path}")
    print("first 40 lines:")
    for line in lines[:40]:
        print(line)


if __name__ == "__main__":
    main()