#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from sc2.data.csr_shard import shard_directory_sha256, write_csr
from sc2.data.shard_manifest import load_manifest
from sc2.data.sharded_expression_dataset import CounterBasedExpressionStream
from sc2.eval.selective_repair_metrics import choose_threshold, threshold_sweep
from sc2.losses.continuous_repair_losses import compute_continuous_objective
from sc2.train.pareto import ParetoFront
from sc2.train.schedules import scheduled_weights


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        shard = root / "shard_000"
        matrix = sparse.csr_matrix(np.array([[0, 1, 3, 0], [2, 0, 1, 4], [0, 0, 5, 1]], dtype=np.float32))
        write_csr(shard, "counts", matrix)
        write_csr(shard, "log1p", sparse.csr_matrix(np.log1p(matrix.toarray())))
        pd.DataFrame({
            "cell_id": ["a", "b", "c"],
            "dataset_id": ["d"] * 3,
            "donor_id": ["x", "y", "z"],
            "tissue": ["lung"] * 3,
            "cell_type": ["t"] * 3,
            "split": ["train"] * 3,
        }).to_csv(shard / "obs.csv", index=False)
        (shard / "meta.json").write_text(
            '{"schema_version":"sc2-csr-shard-v1","shard_id":"s0","census_release":"test","split":"train","modality":"sc","n_rows":3,"n_genes":4,"gene_vocab_sha256":"v","normalization":"log1p","counts_dtype":"float32"}',
            encoding="utf-8",
        )
        digest = shard_directory_sha256(shard)
        manifest = root / "shards.csv"
        pd.DataFrame([{
            "shard_id": "s0", "path": str(shard), "split": "train", "modality": "sc",
            "n_rows": 3, "n_genes": 4, "sha256": digest, "gene_vocab_sha256": "v", "census_release": "test",
        }]).to_csv(manifest, index=False)
        records, _ = load_manifest(manifest)
        stream = CounterBasedExpressionStream(records, seed=7, mask_rates=[0.5], mask_probabilities=[1.0])
        first = stream.sample_at(12)
        second = stream.sample_at(12)
        assert torch.equal(first["x"], second["x"])
        batch = {
            "x": first["x"].unsqueeze(0),
            "y": first["y"].unsqueeze(0),
            "counts": first["counts"].unsqueeze(0),
            "synthetic_mask": first["synthetic_mask"].unsqueeze(0),
            "library_size": first["library_size"].unsqueeze(0),
        }
        shape = batch["x"].shape
        outputs = {
            "positive_value": torch.ones(shape, requires_grad=True),
            "expected_repair": torch.full(shape, 0.5, requires_grad=True),
            "dropout_logits": torch.zeros(shape, requires_grad=True),
        }
        objective = compute_continuous_objective(
            outputs,
            batch,
            {"weights": {"expected_positive": 1.0, "repair_gate": 0.1}},
        )
        assert torch.isfinite(objective.loss)
        objective.loss.backward()
        weights = scheduled_weights({"rank": 0.0}, {"rank": {"start_step": 0, "end_step": 10, "start": 0.0, "end": 1.0}}, 5)
        assert abs(weights["rank"] - 0.5) < 1.0e-8
        sweep = threshold_sweep(np.array([0.9, 0.8, 0.2]), np.array([True, False, False]), np.array([False, True, True]))
        choice = choose_threshold(sweep, max_true_zero_fill=0.5)
        assert 0.0 <= choice.threshold <= 1.0
        front = ParetoFront(minimize=("loss",), maximize=("rho",))
        assert front.add({"loss": 1.0, "rho": 0.1})
        print("continuous_bundle_smoke_test=ok")


if __name__ == "__main__":
    main()
