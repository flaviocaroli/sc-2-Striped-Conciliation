#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from sc2.data.census_pipeline import (
    VALID_SPLITS,
    canonical_json_sha256,
    dataset_matches_registry,
    dataset_priority,
    file_sha256,
    largest_remainder_quotas,
    load_yaml,
    nonempty_string_mask,
    registry_exclusions,
    require_columns,
    splitmix64,
    stable_group_split,
    validate_registry_payload,
)

OBS_COLUMNS = [
    "soma_joinid",
    "dataset_id",
    "donor_id",
    "tissue",
    "tissue_general",
    "cell_type",
    "assay",
    "is_primary_data",
    "tissue_type",
    "nnz",
    "raw_sum",
]
DATASET_COLUMNS = [
    "soma_joinid",
    "collection_id",
    "collection_name",
    "collection_doi",
    "citation",
    "dataset_id",
    "dataset_title",
    "dataset_h5ad_path",
    "dataset_total_cell_count",
]


def _iter_tables(reader: Any) -> Iterable[pd.DataFrame]:
    if hasattr(reader, "tables"):
        for table in reader.tables():
            yield table.to_pandas()
        return
    if hasattr(reader, "concat"):
        yield reader.concat().to_pandas()
        return
    raise TypeError(f"Unsupported SOMA read iterator: {type(reader)!r}")


def _read_datasets(census: Any) -> pd.DataFrame:
    table = census["census_info"]["datasets"]
    available = set(table.keys()) if hasattr(table, "keys") else None
    columns = DATASET_COLUMNS if available is None else [column for column in DATASET_COLUMNS if column in available]
    try:
        frame = table.read(column_names=columns).concat().to_pandas()
    except Exception:
        frame = table.read().concat().to_pandas()
    require_columns(frame, ["dataset_id", "collection_id", "dataset_title", "collection_name"], name="Census datasets")
    return frame


def _dataset_top_candidates(
    obs: Any,
    *,
    dataset_id: str,
    min_detected_genes: int,
    per_dataset_cap: int,
    seed: int,
    required_metadata: list[str],
) -> pd.DataFrame:
    value_filter = (
        f"dataset_id == '{dataset_id}' and is_primary_data == True "
        f"and nnz >= {int(min_detected_genes)}"
    )
    try:
        reader = obs.read(value_filter=value_filter, column_names=OBS_COLUMNS)
    except Exception as error:
        raise RuntimeError(f"Failed Census obs query for dataset_id={dataset_id}: {error}") from error

    best = pd.DataFrame()
    for batch in _iter_tables(reader):
        if batch.empty:
            continue
        require_columns(batch, ["soma_joinid", "dataset_id", "nnz", "raw_sum"], name="Census obs batch")
        valid = pd.Series(True, index=batch.index)
        for column in required_metadata:
            if column not in batch.columns:
                raise ValueError(f"Required Census metadata column is absent: {column}")
            valid &= nonempty_string_mask(batch[column])
        valid &= batch["raw_sum"].astype(float) > 0.0
        batch = batch.loc[valid].copy()
        if batch.empty:
            continue
        ids = batch["soma_joinid"].to_numpy(dtype=np.uint64, copy=False)
        batch["cell_priority"] = splitmix64(ids, seed=seed)
        if len(batch) > per_dataset_cap:
            batch = batch.nsmallest(per_dataset_cap, "cell_priority")
        if best.empty:
            best = batch
        else:
            best = pd.concat([best, batch], ignore_index=True)
            if len(best) > per_dataset_cap:
                best = best.nsmallest(per_dataset_cap, "cell_priority")
    if best.empty:
        return best
    return best.sort_values(["cell_priority", "soma_joinid"], kind="mergesort").reset_index(drop=True)


def _candidate_quotas(target_cells: int, fractions: dict[str, float], multiplier: float) -> tuple[dict[str, int], dict[str, int]]:
    final = largest_remainder_quotas(target_cells, fractions)
    candidate = {name: int(math.ceil(final[name] * multiplier)) for name in VALID_SPLITS}
    return final, candidate


def _write_atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic, leakage-controlled Census cell plan without reading the expression matrix"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite-plan", action="store_true")
    parser.add_argument("--datasets-cache", default=None)
    args = parser.parse_args()

    if not args.plan_only:
        raise SystemExit(
            "This command intentionally supports plan-only selection. Use finalize_census_plan_qc.py, "
            "compute_census_train_gene_stats.py and materialize_census_shards.py for later phases."
        )

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    name = str(config.get("name", "census_curated_pretrain_pilot250k"))
    census_release = str(config["census_release"])
    target_cells = int(config["target_cells"])
    quality = dict(config.get("quality_control", {}))
    min_detected_genes = int(quality.get("min_detected_genes", 200))
    min_cells_per_dataset = int(quality.get("min_cells_per_dataset", 0))
    caps = {key: int(value) for key, value in config.get("sampling_caps", {}).items()}
    required_metadata = [str(value) for value in config.get("required_metadata", [])]
    fractions = {key: float(value) for key, value in config["splits"]["fractions"].items()}
    split_seed = int(config["splits"].get("seed", 20260728))
    planning = dict(config.get("planning", {}))
    candidate_multiplier = float(planning.get("candidate_multiplier", 1.25))
    if candidate_multiplier < 1.0:
        raise ValueError("planning.candidate_multiplier must be at least 1.0")
    final_quotas, candidate_quotas = _candidate_quotas(target_cells, fractions, candidate_multiplier)

    root = Path(args.data_root).resolve() / name
    plan_dir = root / "plan"
    if plan_dir.exists() and any(plan_dir.iterdir()):
        if not args.overwrite_plan:
            raise SystemExit(
                f"Plan directory is not empty: {plan_dir}. Use a new corpus name or --overwrite-plan after review."
            )
        shutil.rmtree(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)

    registry_path = Path(config["leakage"]["benchmark_registry"])
    if not registry_path.is_absolute():
        registry_path = (Path.cwd() / registry_path).resolve()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Benchmark registry YAML must contain a mapping")
    registry_errors = validate_registry_payload(registry, expected_release=census_release)
    if registry_errors:
        print("benchmark_registry=INCOMPLETE")
        for error in registry_errors:
            print(f"- {error}")
        raise SystemExit(
            "Freeze the benchmark registry before planning. Run scripts/benchmark/"
            "resolve_benchmark_registry_candidates.py and validate_benchmark_registry.py."
        )
    exclusions = registry_exclusions(registry)

    try:
        import cellxgene_census
    except ImportError as error:
        raise SystemExit(
            "cellxgene-census is not installed. Use the sc2-data environment and install requirements-continuous.txt."
        ) from error

    selected_rows: list[dict[str, Any]] = []
    counts_by_split = Counter({name: 0 for name in VALID_SPLITS})
    counts_dataset: Counter[str] = Counter()
    counts_donor: Counter[str] = Counter()
    counts_tissue: Counter[str] = Counter()
    counts_tissue_cell: Counter[tuple[str, str]] = Counter()
    excluded_rows: list[dict[str, Any]] = []
    scanned_datasets: list[dict[str, Any]] = []

    with cellxgene_census.open_soma(census_version=census_release) as census:
        if args.datasets_cache and Path(args.datasets_cache).exists():
            datasets = pd.read_parquet(args.datasets_cache)
        else:
            datasets = _read_datasets(census)
            cache = Path(args.datasets_cache) if args.datasets_cache else plan_dir / "census_datasets.parquet"
            cache.parent.mkdir(parents=True, exist_ok=True)
            datasets.to_parquet(cache, index=False)
        datasets = datasets.drop_duplicates("dataset_id").copy()
        dataset_records = datasets.to_dict(orient="records")
        eligible_records: list[dict[str, Any]] = []
        for record in dataset_records:
            matches = dataset_matches_registry(record, exclusions)
            if matches:
                excluded = dict(record)
                excluded["exclusion_reasons"] = ";".join(matches)
                excluded_rows.append(excluded)
            else:
                eligible_records.append(record)
        eligible_records.sort(key=lambda row: (dataset_priority(str(row["dataset_id"]), seed=split_seed), str(row["dataset_id"])))

        obs = census["census_data"]["homo_sapiens"].obs
        per_dataset_cap = int(caps.get("per_dataset", target_cells))
        per_donor_cap = int(caps.get("per_donor", target_cells))
        per_tissue_cap = int(caps.get("per_tissue", target_cells))
        per_tissue_cell_cap = int(caps.get("per_tissue_cell_type", target_cells))

        for dataset_number, dataset_record in enumerate(eligible_records, start=1):
            if all(counts_by_split[name] >= candidate_quotas[name] for name in VALID_SPLITS):
                break
            dataset_id = str(dataset_record["dataset_id"])
            total_cells = pd.to_numeric(
                pd.Series([dataset_record.get("dataset_total_cell_count")]), errors="coerce"
            ).iloc[0]
            if min_cells_per_dataset > 0 and pd.notna(total_cells) and int(total_cells) < min_cells_per_dataset:
                scanned_datasets.append({
                    "dataset_number": dataset_number,
                    "dataset_id": dataset_id,
                    "eligible_cells_considered": 0,
                    "accepted_candidates": 0,
                    "cumulative_candidates": int(len(selected_rows)),
                    "skip_reason": f"dataset_total_cell_count<{min_cells_per_dataset}",
                })
                continue
            candidates = _dataset_top_candidates(
                obs,
                dataset_id=dataset_id,
                min_detected_genes=min_detected_genes,
                per_dataset_cap=per_dataset_cap,
                seed=split_seed,
                required_metadata=required_metadata,
            )
            if min_cells_per_dataset > 0 and len(candidates) < min_cells_per_dataset:
                scanned_datasets.append({
                    "dataset_number": dataset_number,
                    "dataset_id": dataset_id,
                    "eligible_cells_considered": int(len(candidates)),
                    "accepted_candidates": 0,
                    "cumulative_candidates": int(len(selected_rows)),
                    "skip_reason": f"qc_eligible_cells<{min_cells_per_dataset}",
                })
                continue
            accepted_before = len(selected_rows)
            if not candidates.empty:
                for row in candidates.to_dict(orient="records"):
                    donor = str(row.get("donor_id", "")).strip()
                    tissue = str(row.get("tissue", "")).strip()
                    cell_type = str(row.get("cell_type", "")).strip()
                    split_group = f"{dataset_id}::{donor}"
                    split = stable_group_split(split_group, seed=split_seed, fractions=fractions)
                    if counts_by_split[split] >= candidate_quotas[split]:
                        continue
                    if counts_dataset[dataset_id] >= per_dataset_cap:
                        continue
                    if counts_donor[split_group] >= per_donor_cap:
                        continue
                    if counts_tissue[tissue] >= per_tissue_cap:
                        continue
                    tissue_cell = (tissue, cell_type)
                    if counts_tissue_cell[tissue_cell] >= per_tissue_cell_cap:
                        continue
                    record = dict(row)
                    for column in ("collection_id", "collection_name", "collection_doi", "citation", "dataset_title"):
                        record[column] = dataset_record.get(column)
                    record["split_group"] = split_group
                    record["split"] = split
                    record["candidate_rank"] = counts_by_split[split]
                    record["census_release"] = census_release
                    selected_rows.append(record)
                    counts_by_split[split] += 1
                    counts_dataset[dataset_id] += 1
                    counts_donor[split_group] += 1
                    counts_tissue[tissue] += 1
                    counts_tissue_cell[tissue_cell] += 1
            scanned_datasets.append({
                "dataset_number": dataset_number,
                "dataset_id": dataset_id,
                "eligible_cells_considered": int(len(candidates)),
                "accepted_candidates": int(len(selected_rows) - accepted_before),
                "cumulative_candidates": int(len(selected_rows)),
            })
            print(
                f"dataset={dataset_number}/{len(eligible_records)} id={dataset_id} "
                f"considered={len(candidates)} accepted={len(selected_rows) - accepted_before} "
                f"split_counts={dict(counts_by_split)}",
                flush=True,
            )

    shortages = {name: candidate_quotas[name] - counts_by_split[name] for name in VALID_SPLITS if counts_by_split[name] < candidate_quotas[name]}
    if shortages:
        diagnostic = pd.DataFrame(selected_rows)
        if not diagnostic.empty:
            _write_atomic_parquet(diagnostic, plan_dir / "incomplete_planned_cells.parquet")
        raise SystemExit(
            f"Unable to meet candidate quotas: {shortages}. Review metadata requirements/caps or increase available datasets."
        )

    planned = pd.DataFrame(selected_rows)
    planned = planned.sort_values(["split", "candidate_rank"], kind="mergesort").reset_index(drop=True)
    if planned["soma_joinid"].duplicated().any():
        raise ValueError("Planned Census soma_joinid values are not unique")
    planned["cell_id"] = planned["soma_joinid"].map(lambda value: f"census:{census_release}:{int(value)}")

    planned_path = plan_dir / "planned_cells.parquet"
    _write_atomic_parquet(planned, planned_path)
    excluded_columns = list(dict.fromkeys(DATASET_COLUMNS + ["exclusion_reasons"]))
    pd.DataFrame(excluded_rows, columns=excluded_columns).to_parquet(
        plan_dir / "excluded_benchmark_datasets.parquet", index=False
    )
    pd.DataFrame(scanned_datasets).to_csv(plan_dir / "dataset_scan.csv", index=False)
    shutil.copy2(config_path, plan_dir / "corpus_config.yaml")
    shutil.copy2(registry_path, plan_dir / "benchmark_registry.yaml")

    summary = {
        "name": name,
        "census_release": census_release,
        "target_cells_after_mito_qc": target_cells,
        "candidate_multiplier": candidate_multiplier,
        "final_split_quotas": final_quotas,
        "candidate_split_quotas": candidate_quotas,
        "candidate_split_counts": {name: int(counts_by_split[name]) for name in VALID_SPLITS},
        "candidate_cells": int(len(planned)),
        "datasets_used": int(planned["dataset_id"].nunique()),
        "donor_groups": int(planned["split_group"].nunique()),
        "tissues": int(planned["tissue"].nunique()),
        "cell_types": int(planned["cell_type"].nunique()),
        "excluded_benchmark_datasets": int(len(excluded_rows)),
        "min_detected_genes_enforced": min_detected_genes,
        "min_cells_per_dataset_enforced": min_cells_per_dataset,
        "max_mito_fraction_status": "pending_finalize_census_plan_qc.py",
        "config_sha256": file_sha256(config_path),
        "registry_sha256": file_sha256(registry_path),
        "planned_cells_sha256": file_sha256(planned_path),
        "resolved_config_sha256": canonical_json_sha256(config),
    }
    (plan_dir / "plan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"plan_only=ok planned_cells={planned_path} candidates={len(planned)}")
    print("hard_stop=run overlap check, plan audit and mitochondrial QC before expression materialization")


if __name__ == "__main__":
    main()
