from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_paper: str
    task_groups: list[str]
    organism: str
    tissue: str
    modality: list[str]
    has_bulk: bool
    has_sc: bool
    has_paired_bulk_sc: bool
    local_paths: dict[str, str]
    split_policy: str
    primary_metrics: list[str]
    raw: dict[str, Any]


def load_dataset_registry(path: str | Path) -> dict[str, DatasetSpec]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    datasets = obj.get("datasets", {})
    out: dict[str, DatasetSpec] = {}

    for name, item in datasets.items():
        out[name] = DatasetSpec(
            name=name,
            source_paper=str(item.get("source_paper", "")),
            task_groups=list(item.get("task_groups", [])),
            organism=str(item.get("organism", "")),
            tissue=str(item.get("tissue", "")),
            modality=list(item.get("modality", [])),
            has_bulk=bool(item.get("has_bulk", False)),
            has_sc=bool(item.get("has_sc", False)),
            has_paired_bulk_sc=bool(item.get("has_paired_bulk_sc", False)),
            local_paths=dict(item.get("local_paths", {})),
            split_policy=str(item.get("split_policy", "")),
            primary_metrics=list(item.get("primary_metrics", [])),
            raw=dict(item),
        )

    return out


def get_dataset_spec(path: str | Path, dataset_name: str) -> DatasetSpec:
    registry = load_dataset_registry(path)
    if dataset_name not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")
    return registry[dataset_name]


def resolve_dataset_paths(
    spec: DatasetSpec,
    data_root: str | Path,
) -> dict[str, Path]:
    root = Path(data_root)
    resolved: dict[str, Path] = {}

    for key, value in spec.local_paths.items():
        p = Path(value)
        resolved[key] = p if p.is_absolute() else root / p

    return resolved