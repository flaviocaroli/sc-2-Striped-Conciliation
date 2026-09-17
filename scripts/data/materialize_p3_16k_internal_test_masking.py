#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dense_rows(shard: Path, rows: np.ndarray) -> np.ndarray:
    data = np.load(shard / "log1p.data.npy", mmap_mode="r")
    indices = np.load(shard / "log1p.indices.npy", mmap_mode="r")
    indptr = np.load(shard / "log1p.indptr.npy", mmap_mode="r")

    shape = tuple(
        json.loads((shard / "log1p.shape.json").read_text())
    )

    if shape != (25000, 16384):
        raise RuntimeError(f"Unexpected test shard shape: {shape}")

    out = np.zeros((len(rows), 16384), dtype=np.float32)

    for i, source_row in enumerate(rows):
        source_row = int(source_row)

        if source_row < 0 or source_row >= shape[0]:
            raise RuntimeError(
                f"test row out of bounds: {source_row}"
            )

        lo = int(indptr[source_row])
        hi = int(indptr[source_row + 1])

        cols = np.asarray(indices[lo:hi], dtype=np.int64)
        vals = np.asarray(data[lo:hi], dtype=np.float32)

        out[i, cols] = vals

    return out


def make_mask(
    y: np.ndarray,
    rate: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))

    mask = np.zeros(y.shape, dtype=np.bool_)

    for row in range(y.shape[0]):
        positive = np.flatnonzero(y[row] > 0.0)

        if positive.size <= 0:
            raise RuntimeError(f"No positive genes in row {row}")

        n_mask = max(
            1,
            int(np.rint(float(rate) * positive.size)),
        )

        if n_mask > positive.size:
            raise RuntimeError(
                f"Invalid mask count row={row}: "
                f"{n_mask}>{positive.size}"
            )

        chosen = rng.choice(
            positive,
            size=n_mask,
            replace=False,
        )

        mask[row, chosen] = True

    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()

    protocol = json.loads(protocol_path.read_text())

    if protocol["status"] != (
        "FROZEN_BEFORE_INTERNAL_TEST_PANEL_MATERIALIZATION"
    ):
        raise RuntimeError("Unexpected protocol state")

    p3_manifest = Path(
        protocol["p3_corpus"]["runtime_manifest"]
    )

    manifest = pd.read_parquet(p3_manifest)

    rec = manifest.loc[
        (manifest["shard_id"].astype(str) == "sc_test_00000")
        & (manifest["split"].astype(str) == "test")
        & (manifest["modality"].astype(str) == "sc")
    ]

    if len(rec) != 1:
        raise RuntimeError(
            f"Expected one test shard, found {len(rec)}"
        )

    rec = rec.iloc[0]

    if int(rec["n_rows"]) != 25000:
        raise RuntimeError("Unexpected test row count")

    if int(rec["n_genes"]) != 16384:
        raise RuntimeError("Unexpected test gene count")

    if str(rec["sha256"]) != (
        "e90f07b9b92e409a1fdd5111030d6d6ebd4ecf07775e4a5992ab949ead0740db"
    ):
        raise RuntimeError("Unexpected test shard SHA field")

    if str(rec["gene_vocab_sha256"]) != (
        "4697c8de54f42f6e6352a94ea22c820e6309221aa808d6053377e23371564a9e"
    ):
        raise RuntimeError("Unexpected gene vocabulary SHA")

    shard = Path(str(rec["path"]))

    # Exact cell rows are inherited from the frozen P2 test panels.
    p2_panel_paths = [
        Path(
            protocol["test_source"]["p2_test_panels"][f"mask{pct}"]["path"]
        )
        for pct in (15, 30, 50)
    ]

    rows_ref = None
    shards_ref = None

    for pct, panel_path in zip((15, 30, 50), p2_panel_paths):
        expected_sha = (
            protocol["test_source"]["p2_test_panels"]
            [f"mask{pct}"]["sha256"]
        )

        if sha256(panel_path) != expected_sha:
            raise RuntimeError(
                f"P2 source panel SHA mismatch: {panel_path}"
            )

        with np.load(panel_path, allow_pickle=False) as z:
            rows = np.asarray(z["row"], dtype=np.int64)
            shards = np.asarray(z["shard_id"]).astype(str)
            seed = int(np.asarray(z["seed"]).item())

        if seed != 20260729:
            raise RuntimeError(
                f"Unexpected P2 test seed: {seed}"
            )

        if rows_ref is None:
            rows_ref = rows.copy()
            shards_ref = shards.copy()
        else:
            if not np.array_equal(rows, rows_ref):
                raise RuntimeError(
                    "P2 test row identities differ by mask rate"
                )
            if not np.array_equal(shards, shards_ref):
                raise RuntimeError(
                    "P2 test shard identities differ by mask rate"
                )

    assert rows_ref is not None
    assert shards_ref is not None

    if rows_ref.shape != (5000,):
        raise RuntimeError(
            f"Expected 5000 test rows, got {rows_ref.shape}"
        )

    if np.unique(rows_ref).size != 5000:
        raise RuntimeError("Duplicate test rows")

    if np.unique(shards_ref).tolist() != ["sc_test_00000"]:
        raise RuntimeError("Unexpected source test shard")

    y = dense_rows(shard, rows_ref)

    if y.shape != (5000, 16384):
        raise RuntimeError(f"Unexpected dense target shape {y.shape}")

    output_root.mkdir(parents=True, exist_ok=False)

    panel_rows = []

    for pct, rate in (
        (15, 0.15),
        (30, 0.30),
        (50, 0.50),
    ):
        seed = 20260729

        synthetic_mask = make_mask(
            y=y,
            rate=rate,
            seed=seed,
        )

        expected_counts = np.maximum(
            1,
            np.rint(
                rate * (y > 0.0).sum(axis=1)
            ).astype(np.int64),
        )

        actual_counts = synthetic_mask.sum(axis=1)

        if not np.array_equal(
            expected_counts,
            actual_counts,
        ):
            raise RuntimeError(
                f"Mask-count rule mismatch at {pct}%"
            )

        if not np.all(y[synthetic_mask] > 0.0):
            raise RuntimeError(
                f"Masked targets not positive at {pct}%"
            )

        x = y.copy()
        x[synthetic_mask] = 0.0

        if not np.all(x[synthetic_mask] == 0.0):
            raise RuntimeError(
                f"Masked inputs not zero at {pct}%"
            )

        if not np.array_equal(
            x[~synthetic_mask],
            y[~synthetic_mask],
        ):
            raise RuntimeError(
                f"Unmasked values changed at {pct}%"
            )

        panel = output_root / f"test_mask{pct}.npz"

        np.savez_compressed(
            panel,
            x=x,
            y=y,
            synthetic_mask=synthetic_mask,
            row=rows_ref.astype(np.int64),
            shard_id=np.asarray(
                ["sc_test_00000"] * len(rows_ref)
            ),
            split=np.asarray("test"),
            modality=np.asarray("sc"),
            mask_rate=np.asarray(rate, dtype=np.float32),
            seed=np.asarray(seed, dtype=np.int64),
            manifest_sha256=np.asarray(
                protocol["p3_corpus"]
                ["original_frozen_manifest_sha256"]
            ),
            runtime_manifest_sha256=np.asarray(
                protocol["p3_corpus"]
                ["runtime_manifest_sha256"]
            ),
            protocol_sha256=np.asarray(
                sha256(protocol_path)
            ),
        )

        panel_sha = sha256(panel)

        masked = int(synthetic_mask.sum())

        panel_rows.append({
            "mask_rate": rate,
            "mask_percent": pct,
            "seed": seed,
            "cells": 5000,
            "genes": 16384,
            "masked_entries": masked,
            "file": str(panel),
            "sha256": panel_sha,
        })

        print(
            f"wrote={panel.name} "
            f"cells=5000 genes=16384 "
            f"masked={masked} "
            f"sha256={panel_sha}"
        )

    manifest_out = {
        "schema":
            "sc2-p3-16k-internal-test-panels-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "source_split": "test",
        "source_shard_id": "sc_test_00000",
        "cells": 5000,
        "genes": 16384,
        "panel_seed": 20260729,
        "panels": panel_rows,
        "model_inference_performed": False,
        "threshold_retuning_performed": False,
        "baron_used": False,
        "zheng68k_used": False,
        "status": "PASS",
    }

    manifest_path = (
        output_root / "test_panels_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest_out,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "TEST_PANELS_MANIFEST_SHA256="
        + sha256(manifest_path)
    )
    print("P3_16K_INTERNAL_TEST_MATERIALIZATION=PASS")
    print(f"OUTPUT_ROOT={output_root}")


if __name__ == "__main__":
    main()
