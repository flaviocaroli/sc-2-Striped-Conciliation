#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse

from sc2.data.csr_shard import CSRMemmap
from sc2.data.shard_manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the train-only mean log1p expression vector from immutable shards")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modality", default="sc")
    args = parser.parse_args()

    records, _ = load_manifest(args.manifest)
    selected = [
        record for record in records if record.split == "train" and record.modality == args.modality
    ]
    if not selected:
        raise ValueError(f"No train shards for modality={args.modality!r}")
    n_genes = selected[0].n_genes
    total = np.zeros(n_genes, dtype=np.float64)
    n_rows = 0
    for record in selected:
        matrix = CSRMemmap.open(record.path, "log1p")
        csr = sparse.csr_matrix(
            (
                np.asarray(matrix.data),
                np.asarray(matrix.indices),
                np.asarray(matrix.indptr),
            ),
            shape=matrix.shape,
        )
        total += np.asarray(csr.sum(axis=0)).reshape(-1)
        n_rows += matrix.shape[0]
        print(f"gene_mean_shard={record.shard_id} cumulative_rows={n_rows}")
    mean = (total / float(n_rows)).astype(np.float32)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, mean)
    temporary.replace(destination)
    print(f"gene_mean=ok output={destination} rows={n_rows} genes={n_genes}")


if __name__ == "__main__":
    main()
