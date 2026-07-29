from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from sc2.data.csr_shard import CSRMemmap
from sc2.data.shard_manifest import ShardRecord


def _u64(seed: int, index: int, stream: str) -> int:
    payload = f"{int(seed)}:{int(index)}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _uniform(seed: int, index: int, stream: str) -> float:
    return _u64(seed, index, stream) / float(2**64)


def _choice_index(weights: Sequence[float], value: float) -> int:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("Choice weights must have positive sum")
    target = value * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += float(weight)
        if target < cumulative:
            return index
    return len(weights) - 1


@dataclass
class _OpenedShard:
    record: ShardRecord
    counts: CSRMemmap
    log1p: CSRMemmap

    @classmethod
    def open(cls, record: ShardRecord) -> "_OpenedShard":
        with (record.path / "meta.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if int(meta["n_rows"]) != record.n_rows or int(meta["n_genes"]) != record.n_genes:
            raise ValueError(f"Manifest/meta dimension mismatch for {record.shard_id}")
        if str(meta["gene_vocab_sha256"]) != record.gene_vocab_sha256:
            raise ValueError(f"Vocabulary hash mismatch for {record.shard_id}")
        return cls(
            record=record,
            counts=CSRMemmap.open(record.path, "counts"),
            log1p=CSRMemmap.open(record.path, "log1p"),
        )


class CounterBasedExpressionStream(IterableDataset[dict[str, Any]]):
    """Exact-resume stream indexed only by a global sample counter.

    The same `(seed, sample_index, manifest)` produces the same modality, shard,
    row and synthetic mask. Exact-resume mode intentionally requires one worker.
    """

    def __init__(
        self,
        records: Sequence[ShardRecord],
        *,
        seed: int,
        start_index: int = 0,
        modality_weights: Mapping[str, float] | None = None,
        mask_rates: Sequence[float] = (0.0, 0.10, 0.15, 0.30, 0.50),
        mask_probabilities: Sequence[float] = (0.10, 0.25, 0.30, 0.25, 0.10),
        stop_index: int | None = None,
    ) -> None:
        super().__init__()
        if not records:
            raise ValueError("records must not be empty")
        if len(mask_rates) != len(mask_probabilities):
            raise ValueError("mask_rates and mask_probabilities length mismatch")
        self.records = list(records)
        self.seed = int(seed)
        self.start_index = int(start_index)
        self.stop_index = None if stop_index is None else int(stop_index)
        self.mask_rates = tuple(float(value) for value in mask_rates)
        self.mask_probabilities = tuple(float(value) for value in mask_probabilities)
        modalities = sorted({record.modality for record in self.records})
        requested = dict(modality_weights or {name: 1.0 for name in modalities})
        self.modalities = [name for name in modalities if float(requested.get(name, 0.0)) > 0.0]
        self.modality_weights = [float(requested[name]) for name in self.modalities]
        if not self.modalities:
            raise ValueError("No modality has positive sampling weight")
        self.by_modality = {
            name: [record for record in self.records if record.modality == name]
            for name in self.modalities
        }
        if any(not values for values in self.by_modality.values()):
            raise ValueError("Requested modality has no shard")
        self._cache: dict[str, _OpenedShard] = {}

    def _opened(self, record: ShardRecord) -> _OpenedShard:
        if record.shard_id not in self._cache:
            self._cache[record.shard_id] = _OpenedShard.open(record)
        return self._cache[record.shard_id]

    def sample_at(self, sample_index: int) -> dict[str, Any]:
        modality_position = _choice_index(
            self.modality_weights,
            _uniform(self.seed, sample_index, "modality"),
        )
        modality = self.modalities[modality_position]
        candidates = self.by_modality[modality]
        row_weights = [record.n_rows for record in candidates]
        shard_position = _choice_index(
            row_weights,
            _uniform(self.seed, sample_index, f"shard:{modality}"),
        )
        record = candidates[shard_position]
        row = _u64(self.seed, sample_index, "row") % record.n_rows
        opened = self._opened(record)
        clean = opened.log1p.dense_row(int(row), dtype=np.float32)
        counts = opened.counts.dense_row(int(row), dtype=np.float32)
        mask_position = _choice_index(
            self.mask_probabilities,
            _uniform(self.seed, sample_index, "mask_rate"),
        )
        mask_rate = self.mask_rates[mask_position]
        positive_indices = np.flatnonzero(clean > 0.0)
        n_mask = int(round(mask_rate * positive_indices.size))
        synthetic_mask = np.zeros(clean.shape[0], dtype=np.bool_)
        if n_mask > 0 and positive_indices.size > 0:
            local_seed = _u64(self.seed, sample_index, "mask_members")
            rng = np.random.default_rng(local_seed)
            selected = rng.choice(positive_indices, size=min(n_mask, positive_indices.size), replace=False)
            synthetic_mask[selected] = True
        observed = clean.copy()
        observed[synthetic_mask] = 0.0
        return {
            "x": torch.from_numpy(observed),
            "y": torch.from_numpy(clean),
            "counts": torch.from_numpy(counts),
            "synthetic_mask": torch.from_numpy(synthetic_mask),
            "library_size": torch.tensor(float(counts.sum()), dtype=torch.float32),
            "modality": modality,
            "sample_index": int(sample_index),
            "shard_id": record.shard_id,
            "row": int(row),
            "mask_rate": float(mask_rate),
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            raise RuntimeError(
                "Exact-resume stream currently requires num_workers=0; "
                "add and test a worker-state contract before enabling multiprocessing"
            )
        index = self.start_index
        while self.stop_index is None or index < self.stop_index:
            yield self.sample_at(index)
            index += 1


def collate_expression_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    modalities = {str(item["modality"]) for item in items}
    if len(modalities) != 1:
        raise ValueError("A batch must contain one modality; use modality-homogeneous streams")
    return {
        "x": torch.stack([item["x"] for item in items]),
        "y": torch.stack([item["y"] for item in items]),
        "counts": torch.stack([item["counts"] for item in items]),
        "synthetic_mask": torch.stack([item["synthetic_mask"] for item in items]),
        "library_size": torch.stack([item["library_size"] for item in items]),
        "modality": next(iter(modalities)),
        "sample_index": torch.tensor([item["sample_index"] for item in items], dtype=torch.int64),
        "shard_id": [item["shard_id"] for item in items],
        "row": torch.tensor([item["row"] for item in items], dtype=torch.int64),
        "mask_rate": torch.tensor([item["mask_rate"] for item in items], dtype=torch.float32),
    }
