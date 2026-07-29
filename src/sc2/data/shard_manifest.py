from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "shard_id",
    "path",
    "split",
    "modality",
    "n_rows",
    "n_genes",
    "sha256",
    "gene_vocab_sha256",
    "census_release",
}
VALID_SPLITS = {"train", "validation", "test"}
VALID_MODALITIES = {"sc", "bulk", "pseudobulk"}


@dataclass(frozen=True)
class ShardRecord:
    shard_id: str
    path: Path
    split: str
    modality: str
    n_rows: int
    n_genes: int
    sha256: str
    gene_vocab_sha256: str
    census_release: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> tuple[list[ShardRecord], str]:
    manifest_path = Path(path).resolve()
    if manifest_path.suffix.lower() == ".csv":
        frame = pd.read_csv(manifest_path)
    else:
        frame = pd.read_parquet(manifest_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Manifest is empty")
    if frame["shard_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate shard_id values")
    base = manifest_path.parent
    records: list[ShardRecord] = []
    vocab_hashes = set()
    gene_counts = set()
    for row in frame.to_dict(orient="records"):
        split = str(row["split"])
        modality = str(row["modality"])
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split={split!r}")
        if modality not in VALID_MODALITIES:
            raise ValueError(f"Invalid modality={modality!r}")
        shard_path = Path(str(row["path"]))
        if not shard_path.is_absolute():
            shard_path = (base / shard_path).resolve()
        if not shard_path.is_dir():
            raise FileNotFoundError(shard_path)
        n_rows = int(row["n_rows"])
        n_genes = int(row["n_genes"])
        if n_rows <= 0 or n_genes <= 0:
            raise ValueError(f"Non-positive dimensions for {row['shard_id']}")
        vocab_hash = str(row["gene_vocab_sha256"])
        vocab_hashes.add(vocab_hash)
        gene_counts.add(n_genes)
        records.append(
            ShardRecord(
                shard_id=str(row["shard_id"]),
                path=shard_path,
                split=split,
                modality=modality,
                n_rows=n_rows,
                n_genes=n_genes,
                sha256=str(row["sha256"]),
                gene_vocab_sha256=vocab_hash,
                census_release=str(row["census_release"]),
            )
        )
    if len(vocab_hashes) != 1 or len(gene_counts) != 1:
        raise ValueError("All shards in one run must share one vocabulary and gene count")
    return records, file_sha256(manifest_path)


def filter_records(
    records: Iterable[ShardRecord], *, split: str, modalities: set[str] | None = None
) -> list[ShardRecord]:
    selected = [
        record
        for record in records
        if record.split == split and (modalities is None or record.modality in modalities)
    ]
    if not selected:
        raise ValueError(f"No shards for split={split!r}, modalities={modalities}")
    return selected
