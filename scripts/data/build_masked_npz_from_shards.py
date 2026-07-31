#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from sc2.data.csr_shard import CSRMemmap
from sc2.data.shard_manifest import load_manifest


def _seed(seed: int, split: str, shard_id: str, row: int) -> int:
    payload = f"{seed}:{split}:{shard_id}:{row}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen synthetic-positive masked benchmark from SC2 shards")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-cells", type=int, required=True)
    parser.add_argument("--mask-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--modality", default="sc")
    args = parser.parse_args()
    if not 0.0 < args.mask_rate < 1.0:
        raise ValueError("mask-rate must be between 0 and 1")

    records, manifest_hash = load_manifest(args.manifest)
    records = sorted(
        [record for record in records if record.split == args.split and record.modality == args.modality],
        key=lambda record: record.shard_id,
    )
    if not records:
        raise ValueError(f"No shards for split={args.split}, modality={args.modality}")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    shard_ids: list[str] = []
    row_ids: list[int] = []
    for record in records:
        matrix = CSRMemmap.open(record.path, "log1p")
        for row in range(matrix.shape[0]):
            target = matrix.dense_row(row, dtype=np.float32)
            positive = np.flatnonzero(target > 0)
            if positive.size == 0:
                continue
            n_mask = max(1, int(round(args.mask_rate * positive.size)))
            rng = np.random.default_rng(_seed(args.seed, args.split, record.shard_id, row))
            chosen = rng.choice(positive, size=min(n_mask, positive.size), replace=False)
            mask = np.zeros(target.shape[0], dtype=np.bool_)
            mask[chosen] = True
            observed = target.copy()
            observed[mask] = 0.0
            xs.append(observed)
            ys.append(target)
            masks.append(mask)
            shard_ids.append(record.shard_id)
            row_ids.append(row)
            if len(xs) >= args.n_cells:
                break
        if len(xs) >= args.n_cells:
            break
    if len(xs) != args.n_cells:
        raise ValueError(f"Requested {args.n_cells} benchmark cells, found {len(xs)}")

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            x=np.stack(xs),
            y=np.stack(ys),
            synthetic_mask=np.stack(masks),
            shard_id=np.asarray(shard_ids),
            row=np.asarray(row_ids, dtype=np.int64),
            split=np.asarray(args.split),
            modality=np.asarray(args.modality),
            mask_rate=np.asarray(args.mask_rate, dtype=np.float32),
            seed=np.asarray(args.seed, dtype=np.int64),
            manifest_sha256=np.asarray(manifest_hash),
        )
    temporary.replace(destination)
    print(f"masked_benchmark=ok output={destination} cells={len(xs)} genes={xs[0].size}")


if __name__ == "__main__":
    main()
