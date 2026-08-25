#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sc2.eval.p2_selective import (
    choose_exact_threshold,
    exact_threshold_frontier,
    presentation_frontier,
)

from sc2.eval.selective_repair_metrics import (
    gate_discrimination,
    masked_value_metrics,
    risk_coverage_curve,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only p2 evaluator for frozen "
            "positive-train mean/median baselines."
        )
    )

    parser.add_argument(
        "--benchmark",
        required=True,
    )

    parser.add_argument(
        "--benchmark-sha256",
        required=True,
    )

    parser.add_argument(
        "--train-stats",
        required=True,
    )

    parser.add_argument(
        "--train-stats-sha256",
        required=True,
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=(
            "positive_train_mean",
            "positive_train_median",
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--max-true-zero-fill",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=1.0e-8,
    )

    return parser.parse_args()


def prefixed(
    values: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in values.items()
    }


def main() -> None:
    args = parse_args()

    benchmark_path = Path(
        args.benchmark
    ).resolve()

    stats_path = Path(
        args.train_stats
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if output_dir.exists():
        raise FileExistsError(
            output_dir
        )

    if (
        sha256(benchmark_path)
        != args.benchmark_sha256
    ):
        raise RuntimeError(
            "Benchmark SHA256 mismatch"
        )

    if (
        sha256(stats_path)
        != args.train_stats_sha256
    ):
        raise RuntimeError(
            "Train-statistics SHA256 mismatch"
        )

    panel = np.load(
        benchmark_path,
        allow_pickle=False,
    )

    required_panel = {
        "x",
        "y",
        "synthetic_mask",
    }

    if not required_panel.issubset(
        panel.files
    ):
        raise ValueError(
            "Benchmark missing required arrays"
        )

    x = np.asarray(
        panel["x"],
        dtype=np.float32,
    )

    y = np.asarray(
        panel["y"],
        dtype=np.float32,
    )

    positive = np.asarray(
        panel["synthetic_mask"],
        dtype=bool,
    )

    if (
        x.shape != y.shape
        or x.shape != positive.shape
    ):
        raise ValueError(
            "Panel array shape mismatch"
        )

    if x.shape[1] != 4096:
        raise ValueError(
            f"Expected 4096 genes, got {x.shape[1]}"
        )

    zero_threshold = float(
        args.zero_threshold
    )

    true_zero = (
        y <= zero_threshold
    )

    eligible_zero = (
        np.abs(x) <= zero_threshold
    )

    if np.any(
        positive & true_zero
    ):
        raise ValueError(
            "Synthetic positive mask overlaps true zero"
        )

    if np.any(
        positive & ~eligible_zero
    ):
        raise ValueError(
            "Synthetic masked positives are not zero in x"
        )

    stats = np.load(
        stats_path,
        allow_pickle=False,
    )

    prevalence = np.asarray(
        stats["positive_prevalence"],
        dtype=np.float32,
    )

    if args.method == "positive_train_mean":
        gene_value = np.asarray(
            stats["positive_mean"],
            dtype=np.float32,
        )

    elif args.method == "positive_train_median":
        gene_value = np.asarray(
            stats["positive_median"],
            dtype=np.float32,
        )

    else:
        raise AssertionError(
            args.method
        )

    if (
        prevalence.shape != (4096,)
        or gene_value.shape != (4096,)
    ):
        raise ValueError(
            "Train-statistic vector shape mismatch"
        )

    if not np.all(
        np.isfinite(prevalence)
    ):
        raise ValueError(
            "Non-finite prevalence"
        )

    if not np.all(
        np.isfinite(gene_value)
    ):
        raise ValueError(
            "Non-finite baseline value"
        )

    score = np.broadcast_to(
        prevalence[None, :],
        x.shape,
    )

    raw_prediction = np.broadcast_to(
        gene_value[None, :],
        x.shape,
    )

    frontier = exact_threshold_frontier(
        score,
        positive,
        true_zero,
    )

    selected = choose_exact_threshold(
        frontier,
        max_true_zero_fill=(
            args.max_true_zero_fill
        ),
    )

    threshold = float(
        selected["threshold"]
    )

    selected_repair = (
        eligible_zero
        & (score >= threshold)
    )

    reconstruction = x.copy()

    reconstruction[
        selected_repair
    ] = raw_prediction[
        selected_repair
    ]

    observed_nonzero = (
        np.abs(x) > zero_threshold
    )

    preservation_error = (
        reconstruction[
            observed_nonzero
        ]
        - x[
            observed_nonzero
        ]
    )

    if preservation_error.size:
        preservation = {
            "observed_nonzero_mse":
                float(
                    np.mean(
                        preservation_error ** 2
                    )
                ),
            "observed_nonzero_mae":
                float(
                    np.mean(
                        np.abs(
                            preservation_error
                        )
                    )
                ),
            "observed_nonzero_max_abs_error":
                float(
                    np.max(
                        np.abs(
                            preservation_error
                        )
                    )
                ),
            "observed_nonzero_changed_fraction":
                float(
                    np.mean(
                        preservation_error
                        != 0.0
                    )
                ),
            "n_observed_nonzero":
                int(
                    preservation_error.size
                ),
        }
    else:
        preservation = {
            "observed_nonzero_mse":
                float("nan"),
            "observed_nonzero_mae":
                float("nan"),
            "observed_nonzero_max_abs_error":
                float("nan"),
            "observed_nonzero_changed_fraction":
                float("nan"),
            "n_observed_nonzero":
                0,
        }

    if (
        preservation[
            "observed_nonzero_max_abs_error"
        ] != 0.0
    ):
        raise RuntimeError(
            "Observed nonzero preservation failed"
        )

    raw_value_metrics = masked_value_metrics(
        raw_prediction,
        y,
        positive,
    )

    selective_value_metrics = (
        masked_value_metrics(
            reconstruction,
            y,
            positive,
        )
    )

    gate_metrics = gate_discrimination(
        score,
        positive,
        true_zero,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    frontier.to_csv(
        output_dir
        / "exact_threshold_frontier.csv",
        index=False,
    )

    presentation_frontier(
        frontier,
        threshold,
    ).to_csv(
        output_dir
        / "threshold_curve.csv",
        index=False,
    )

    risk_coverage_curve(
        raw_prediction,
        y,
        positive,
        score,
    ).to_csv(
        output_dir
        / "risk_coverage.csv",
        index=False,
    )

    summary: dict[str, Any] = {
        "method": args.method,
        "classification":
            "confirmatory_validation_only",
        "threshold_policy":
            "exact_distinct_score_boundaries",
        "threshold":
            threshold,
        "threshold_recall":
            float(selected["recall"]),
        "threshold_precision":
            float(selected["precision"]),
        "threshold_true_zero_fill":
            float(
                selected[
                    "true_zero_fill"
                ]
            ),
        "threshold_tp":
            int(selected["tp"]),
        "threshold_fp":
            int(selected["fp"]),
        "threshold_selected":
            int(selected["selected"]),
        "max_true_zero_fill":
            float(
                args.max_true_zero_fill
            ),
        "zero_threshold":
            zero_threshold,
        "n_masked_positive":
            int(positive.sum()),
        "n_true_zero":
            int(true_zero.sum()),
        "n_eligible_zero":
            int(eligible_zero.sum()),
        "benchmark":
            str(benchmark_path),
        "benchmark_sha256":
            sha256(benchmark_path),
        "train_statistics":
            str(stats_path),
        "train_statistics_sha256":
            sha256(stats_path),
        **raw_value_metrics,
        **prefixed(
            selective_value_metrics,
            "selective_",
        ),
        **preservation,
        **prefixed(
            gate_metrics,
            "gate_",
        ),
    }

    summary_path = (
        output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    hash_files = [
        output_dir
        / "summary.json",
        output_dir
        / "summary.csv",
        output_dir
        / "exact_threshold_frontier.csv",
        output_dir
        / "threshold_curve.csv",
        output_dir
        / "risk_coverage.csv",
    ]

    with (
        output_dir
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in hash_files:
            handle.write(
                f"{sha256(path)}  "
                f"{path.name}\n"
            )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "P2_TRAIN_STAT_BASELINE_EVAL=PASS"
    )


if __name__ == "__main__":
    main()
