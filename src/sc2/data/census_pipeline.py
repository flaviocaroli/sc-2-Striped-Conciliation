from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

UNRESOLVED_STATUS_TOKENS = ("verify", "resolve", "pending", "draft", "todo", "unresolved")
VALID_SPLITS = ("train", "validation", "test")


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML: {path}")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized(value: object) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _hash_u64(text: str, *, seed: int, namespace: str) -> int:
    payload = f"{namespace}:{int(seed)}:{text}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Fast deterministic 64-bit mixing for integer SOMA join IDs."""
    x = np.asarray(values, dtype=np.uint64) ^ np.uint64(seed)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = x.copy()
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def largest_remainder_quotas(total: int, fractions: Mapping[str, float]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    missing = set(VALID_SPLITS) - set(fractions)
    if missing:
        raise ValueError(f"Split fractions missing keys: {sorted(missing)}")
    values = {name: float(fractions[name]) for name in VALID_SPLITS}
    if any(value < 0 for value in values.values()):
        raise ValueError("Split fractions must be non-negative")
    denominator = sum(values.values())
    if denominator <= 0:
        raise ValueError("Split fractions must sum to a positive number")
    exact = {name: total * values[name] / denominator for name in VALID_SPLITS}
    quotas = {name: int(math.floor(exact[name])) for name in VALID_SPLITS}
    remainder = total - sum(quotas.values())
    order = sorted(VALID_SPLITS, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def stable_group_split(group_id: str, *, seed: int, fractions: Mapping[str, float]) -> str:
    total = sum(float(fractions[name]) for name in VALID_SPLITS)
    if total <= 0:
        raise ValueError("Split fractions must have a positive sum")
    value = _hash_u64(str(group_id), seed=seed, namespace="split") / float(2**64)
    cumulative = 0.0
    for name in VALID_SPLITS:
        cumulative += float(fractions[name]) / total
        if value < cumulative:
            return name
    return VALID_SPLITS[-1]


def dataset_priority(dataset_id: str, *, seed: int) -> int:
    return _hash_u64(str(dataset_id), seed=seed, namespace="dataset-order")


@dataclass(frozen=True)
class RegistryExclusions:
    dataset_ids: frozenset[str]
    collection_ids: frozenset[str]
    tokens: frozenset[str]


def validate_registry_payload(registry: Mapping[str, Any], *, expected_release: str | None = None) -> list[str]:
    errors: list[str] = []
    benchmarks = registry.get("benchmarks", [])
    if not isinstance(benchmarks, list) or not benchmarks:
        errors.append("registry.benchmarks must be a non-empty list")
        return errors
    names: set[str] = set()
    for index, benchmark in enumerate(benchmarks):
        if not isinstance(benchmark, Mapping):
            errors.append(f"benchmarks[{index}] is not a mapping")
            continue
        name = str(benchmark.get("name", "")).strip()
        if not name:
            errors.append(f"benchmarks[{index}] has no name")
            name = f"index_{index}"
        if name in names:
            errors.append(f"duplicate benchmark name: {name}")
        names.add(name)
        status = str(benchmark.get("status", "")).strip().lower()
        if not status:
            errors.append(f"{name}: missing status")
        if any(token in status for token in UNRESOLVED_STATUS_TOKENS):
            errors.append(f"{name}: unresolved status={status!r}")
        if status and status != "frozen":
            errors.append(f"{name}: status must be 'frozen', found {status!r}")
        resolution = benchmark.get("resolution", {})
        if not isinstance(resolution, Mapping):
            errors.append(f"{name}: resolution must be a mapping")
            resolution = {}
        disposition = str(resolution.get("disposition", "")).strip()
        allowed = {"exact_ids_frozen", "absent_from_census_release", "external_only"}
        if disposition not in allowed:
            errors.append(
                f"{name}: resolution.disposition must be one of {sorted(allowed)}, found {disposition!r}"
            )
        release = str(resolution.get("census_release", "")).strip()
        if expected_release and release and release != expected_release:
            errors.append(f"{name}: resolution census_release={release} differs from {expected_release}")
        if expected_release and not release:
            errors.append(f"{name}: resolution.census_release is required")
        evidence = str(resolution.get("evidence", "")).strip()
        if not evidence:
            errors.append(f"{name}: resolution.evidence is required")
        dataset_ids = [str(value).strip() for value in benchmark.get("dataset_ids", []) if str(value).strip()]
        collection_ids = [str(value).strip() for value in benchmark.get("collection_ids", []) if str(value).strip()]
        if disposition == "exact_ids_frozen" and not (dataset_ids or collection_ids):
            errors.append(f"{name}: exact_ids_frozen requires dataset_ids or collection_ids")
    return errors


def registry_exclusions(registry: Mapping[str, Any]) -> RegistryExclusions:
    dataset_ids: set[str] = set()
    collection_ids: set[str] = set()
    tokens: set[str] = set()
    for benchmark in registry.get("benchmarks", []):
        for value in benchmark.get("dataset_ids", []):
            text = str(value).strip()
            if text:
                dataset_ids.add(text)
        for value in benchmark.get("collection_ids", []):
            text = str(value).strip()
            if text:
                collection_ids.add(text)
        # Accessions, aliases, and benchmark names are provenance metadata,
        # not automatic substring exclusions. Short values such as "Xin" or
        # "BAL" can otherwise match unrelated titles. Only explicit
        # exclusion_tokens are eligible for normalized substring matching.
        for value in benchmark.get("exclusion_tokens", []):
            token = normalized(value)
            if token:
                tokens.add(token)
    return RegistryExclusions(
        dataset_ids=frozenset(dataset_ids),
        collection_ids=frozenset(collection_ids),
        tokens=frozenset(tokens),
    )


def dataset_matches_registry(row: Mapping[str, Any], exclusions: RegistryExclusions) -> list[str]:
    matches: list[str] = []
    dataset_id = str(row.get("dataset_id", "")).strip()
    collection_id = str(row.get("collection_id", "")).strip()
    if dataset_id and dataset_id in exclusions.dataset_ids:
        matches.append(f"dataset_id:{dataset_id}")
    if collection_id and collection_id in exclusions.collection_ids:
        matches.append(f"collection_id:{collection_id}")
    searchable = " ".join(
        str(row.get(column, ""))
        for column in ("dataset_title", "collection_name", "collection_doi", "citation")
    )
    searchable_norm = normalized(searchable)
    for token in exclusions.tokens:
        if token and token in searchable_norm:
            matches.append(f"token:{token}")
    return sorted(set(matches))


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def nonempty_string_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip().str.lower()
    return ~text.isin({"", "unknown", "na", "n/a", "none", "nan"})


def split_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["split"].astype(str).value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in VALID_SPLITS}


def cap_violations(frame: pd.DataFrame, caps: Mapping[str, int]) -> list[str]:
    violations: list[str] = []
    checks: list[tuple[str, Sequence[str]]] = [
        ("per_dataset", ["dataset_id"]),
        ("per_donor", ["split_group"]),
        ("per_tissue", ["tissue"]),
        ("per_tissue_cell_type", ["tissue", "cell_type"]),
    ]
    for cap_name, group_columns in checks:
        if cap_name not in caps:
            continue
        limit = int(caps[cap_name])
        counts = frame.groupby(list(group_columns), dropna=False).size()
        if not counts.empty and int(counts.max()) > limit:
            examples = counts[counts > limit].sort_values(ascending=False).head(10)
            violations.append(
                f"{cap_name} limit={limit} exceeded: "
                + "; ".join(f"{index}={int(value)}" for index, value in examples.items())
            )
    return violations
