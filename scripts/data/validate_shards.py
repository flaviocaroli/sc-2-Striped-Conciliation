#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sc2.data.csr_shard import CSRMemmap, shard_directory_sha256
from sc2.data.shard_manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SC2 CSR shards and immutable manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    records, manifest_hash = load_manifest(args.manifest)
    rows = []
    for record in records:
        counts = CSRMemmap.open(record.path, "counts")
        log1p = CSRMemmap.open(record.path, "log1p")
        if counts.shape != log1p.shape or counts.shape != (record.n_rows, record.n_genes):
            raise ValueError(f"Matrix dimension mismatch for {record.shard_id}")
        obs = pd.read_parquet(record.path / "obs.parquet")
        if len(obs) != record.n_rows:
            raise ValueError(f"obs row mismatch for {record.shard_id}")
        with (record.path / "meta.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("schema_version") != "sc2-csr-shard-v1":
            raise ValueError(f"Unsupported schema for {record.shard_id}")
        actual_hash = shard_directory_sha256(record.path) if args.verify_hashes else record.sha256
        if args.verify_hashes and actual_hash != record.sha256:
            raise ValueError(f"Hash mismatch for {record.shard_id}: {actual_hash} != {record.sha256}")
        rows.append({
            "shard_id": record.shard_id,
            "split": record.split,
            "modality": record.modality,
            "n_rows": record.n_rows,
            "n_genes": record.n_genes,
            "hash_ok": actual_hash == record.sha256,
        })
    summary = pd.DataFrame(rows)
    print(f"manifest_sha256={manifest_hash}")
    print(summary.groupby(["split", "modality"])["n_rows"].sum().to_string())
    print("shard_validation=ok")


if __name__ == "__main__":
    main()
