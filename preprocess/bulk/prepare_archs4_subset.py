from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def decode_array(arr: Any) -> list[str]:
    out: list[str] = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out


def collect_datasets(h5: h5py.File) -> list[tuple[str, tuple[int, ...], str]]:
    rows: list[tuple[str, tuple[int, ...], str]] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            rows.append((name, obj.shape, str(obj.dtype)))

    h5.visititems(visitor)
    return rows


def suggest_candidates(datasets: list[tuple[str, tuple[int, ...], str]]) -> dict[str, list[str]]:
    names = [d[0] for d in datasets]

    gene_candidates = [
        n for n in names
        if any(tok in n.lower() for tok in ["gene", "symbol", "genes"])
    ]
    sample_candidates = [
        n for n in names
        if any(tok in n.lower() for tok in ["sample", "gsm", "geo_accession", "accession"])
    ]
    matrix_candidates = [
        n for n in names
        if any(tok in n.lower() for tok in ["matrix", "expression", "data", "counts"])
    ]

    return {
        "gene_candidates": gene_candidates[:20],
        "sample_candidates": sample_candidates[:20],
        "matrix_candidates": matrix_candidates[:20],
    }


def load_vector_dataset(h5: h5py.File, dataset_path: str) -> list[str]:
    arr = h5[dataset_path][()]
    return decode_array(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a tiny ARCHS4 pilot subset.")
    parser.add_argument(
        "--input",
        default="/home/3159436/sc2/data/raw/archs4/human_gene_v2.5.h5",
        help="Path to ARCHS4 H5 file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to $SC2_DATA_ROOT/interim/archs4_pilot",
    )
    parser.add_argument("--gene-dataset", default=None, help="HDF5 dataset path containing gene names")
    parser.add_argument("--sample-dataset", default=None, help="HDF5 dataset path containing sample names")
    parser.add_argument("--matrix-dataset", default="/data/expression", help="HDF5 dataset path containing matrix")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--n-genes", type=int, default=2048)
    args = parser.parse_args()

    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "interim" / "archs4_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"ARCHS4 file not found: {input_path}")

    with h5py.File(input_path, "r") as h5:
        datasets = collect_datasets(h5)
        suggestions = suggest_candidates(datasets)

        suggestion_path = output_dir / "archs4_dataset_suggestions.json"
        with suggestion_path.open("w", encoding="utf-8") as f:
            json.dump(suggestions, f, indent=2)

        if not (args.gene_dataset and args.sample_dataset):
            print(f"Saved suggestions to: {suggestion_path}")
            print(json.dumps(suggestions, indent=2))
            print(
                "\nNeed explicit gene and sample dataset paths.\n"
                "Matrix path is probably already correct: /data/expression\n"
                "Rerun with --gene-dataset ... --sample-dataset ..."
            )
            return

        genes = load_vector_dataset(h5, args.gene_dataset)
        samples = load_vector_dataset(h5, args.sample_dataset)

        ds = h5[args.matrix_dataset]
        if len(ds.shape) != 2:
            raise ValueError(f"Expected 2D matrix at {args.matrix_dataset}, got shape={ds.shape}")

        rows, cols = ds.shape
        n_genes = min(args.n_genes, len(genes))
        n_samples = min(args.n_samples, len(samples))

        # Detect orientation automatically
        if rows == len(genes) and cols == len(samples):
            orientation = "genes_by_samples"
            sub = ds[:n_genes, :n_samples]
            # convert to samples x genes for downstream consistency
            matrix = np.asarray(sub).T
            genes = genes[:n_genes]
            samples = samples[:n_samples]
        elif rows == len(samples) and cols == len(genes):
            orientation = "samples_by_genes"
            sub = ds[:n_samples, :n_genes]
            matrix = np.asarray(sub)
            genes = genes[:n_genes]
            samples = samples[:n_samples]
        else:
            raise ValueError(
                "Could not match matrix dimensions to provided gene/sample vectors. "
                f"matrix_shape={ds.shape}, n_genes={len(genes)}, n_samples={len(samples)}"
            )

    np.save(output_dir / "x_model.npy", matrix.astype(np.float32))

    pd.DataFrame(
        {
            "sample_id": samples,
            "source": "archs4",
            "modality": "bulk",
        }
    ).to_parquet(output_dir / "obs.parquet", index=False)

    pd.DataFrame(
        {
            "gene_name": genes,
            "gene_index": np.arange(len(genes)),
        }
    ).to_parquet(output_dir / "var.parquet", index=False)

    manifest = {
        "source": "archs4",
        "modality": "bulk",
        "input_h5": str(input_path),
        "gene_dataset": args.gene_dataset,
        "sample_dataset": args.sample_dataset,
        "matrix_dataset": args.matrix_dataset,
        "detected_orientation": orientation,
        "n_samples": len(samples),
        "n_genes": len(genes),
        "matrix_saved_shape": list(matrix.shape),
        "output_dir": str(output_dir),
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"saved ARCHS4 pilot subset to: {output_dir}")
    print(f"detected_orientation: {orientation}")
    print(f"saved matrix shape (samples x genes): {matrix.shape}")


if __name__ == "__main__":
    main()