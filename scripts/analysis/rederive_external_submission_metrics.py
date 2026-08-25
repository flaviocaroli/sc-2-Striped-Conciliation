from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/3159436/sc2")
EVAL = ROOT / "outputs/evals"
OUT = ROOT / "outputs/analysis/p1_submission_derived_external_metrics_v1"

DATASETS = {
    "zheng68k": ROOT / "data/external_benchmarks/zheng68k_v1",
    "baron_pancreas": ROOT / "data/external_benchmarks/baron_pancreas_v1",
}

SEEDS = [20260728, 20260729, 20260730]
MASKS = [15, 30, 50]
ZERO_THRESHOLD = 1.0e-8

OUT.mkdir(parents=True, exist_ok=False)

panel_rows = []
seed_rows = []

for dataset, benchdir in DATASETS.items():
    for mask in MASKS:
        panel = np.load(benchdir / f"mask{mask}.npz")

        x = np.asarray(panel["x"], dtype=np.float32)
        y = np.asarray(panel["y"], dtype=np.float32)
        synthetic = np.asarray(panel["synthetic_mask"], dtype=bool)
        available = np.asarray(panel["available_gene_mask"], dtype=bool)

        if x.shape != y.shape or synthetic.shape != y.shape:
            raise ValueError("panel shape mismatch")

        if available.shape != (y.shape[1],):
            raise ValueError("available_gene_mask shape mismatch")

        if np.any(synthetic & ~available[None, :]):
            raise ValueError("synthetic mask includes unavailable genes")

        target = y[synthetic].astype(np.float64)

        target_var = float(np.var(target, ddof=0))
        target_sd = float(np.std(target, ddof=0))
        n_masked = int(synthetic.sum())

        observed_nonzero = (
            (np.abs(x) > ZERO_THRESHOLD)
            & available[None, :]
        )
        observed_nonzero_count = int(observed_nonzero.sum())

        panel_rows.append({
            "dataset": dataset,
            "mask": mask,
            "n_cells": int(y.shape[0]),
            "n_genes": int(y.shape[1]),
            "n_available_genes": int(available.sum()),
            "n_masked": n_masked,
            "masked_target_mean": float(target.mean()),
            "masked_target_variance": target_var,
            "masked_target_sd": target_sd,
            "observed_nonzero_count": observed_nonzero_count,
        })

        for seed in SEEDS:
            summary_path = (
                EVAL
                / f"p1_external_{dataset}_s{seed}_step004000_mask{mask}"
                / "summary.json"
            )

            summary = json.loads(summary_path.read_text())

            required = [
                "masked_mse",
                "prediction_sd",
                "target_sd",
                "sd_ratio",
                "observed_nonzero_max_abs_error",
            ]

            missing = [
                key for key in required
                if key not in summary
            ]

            if missing:
                raise KeyError(
                    f"{summary_path} missing keys: {missing}"
                )

            if not np.isclose(
                float(summary["target_sd"]),
                target_sd,
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError(
                    f"target_sd mismatch for "
                    f"{dataset} seed={seed} mask={mask}"
                )

            mse = float(summary["masked_mse"])
            recovery = 1.0 - mse / target_var

            seed_rows.append({
                "dataset": dataset,
                "seed": seed,
                "mask": mask,
                "n_masked": n_masked,
                "observed_nonzero_count": observed_nonzero_count,
                "masked_target_variance": target_var,
                "masked_target_sd_rederived": target_sd,
                "masked_mse": mse,
                "recovery_index_R": recovery,
                "prediction_sd": float(summary["prediction_sd"]),
                "target_sd_summary": float(summary["target_sd"]),
                "sd_ratio": float(summary["sd_ratio"]),
                "observed_nonzero_max_abs_error": float(
                    summary["observed_nonzero_max_abs_error"]
                ),
            })

panel_df = (
    pd.DataFrame(panel_rows)
    .sort_values(["dataset", "mask"])
)

seed_df = (
    pd.DataFrame(seed_rows)
    .sort_values(["dataset", "mask", "seed"])
)

agg = (
    seed_df.groupby(["dataset", "mask"], as_index=False)
    .agg(
        recovery_index_R_mean=("recovery_index_R", "mean"),
        recovery_index_R_sd=("recovery_index_R", "std"),
        prediction_sd_mean=("prediction_sd", "mean"),
        prediction_sd_sd=("prediction_sd", "std"),
        sd_ratio_mean=("sd_ratio", "mean"),
        sd_ratio_sd=("sd_ratio", "std"),
        masked_mse_mean=("masked_mse", "mean"),
        masked_mse_sd=("masked_mse", "std"),
        observed_nonzero_max_abs_error_max=(
            "observed_nonzero_max_abs_error",
            "max",
        ),
    )
)

agg = agg.merge(
    panel_df[
        [
            "dataset",
            "mask",
            "n_masked",
            "observed_nonzero_count",
            "masked_target_variance",
            "masked_target_sd",
        ]
    ],
    on=["dataset", "mask"],
    how="left",
    validate="one_to_one",
)

panel_df.to_csv(
    OUT / "external_panel_integrity.csv",
    index=False,
)

seed_df.to_csv(
    OUT / "external_derived_by_seed.csv",
    index=False,
)

agg.to_csv(
    OUT / "external_derived_mean_sd.csv",
    index=False,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


files = [
    OUT / "external_panel_integrity.csv",
    OUT / "external_derived_by_seed.csv",
    OUT / "external_derived_mean_sd.csv",
]

with (OUT / "SHA256SUMS.txt").open("w") as handle:
    for path in files:
        handle.write(
            f"{sha256(path)}  {path.name}\n"
        )

print("===== PANEL INTEGRITY =====")
print(panel_df.to_string(index=False))

print()
print("===== DERIVED MEAN SD =====")
print(agg.to_string(index=False))

print()
print("DERIVED_EXTERNAL_METRICS=PASS")
